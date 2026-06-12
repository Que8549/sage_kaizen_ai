"""
tests/test_pg.py

Unit tests for rag_v1/db/pg.py — thread-local PostgreSQL connection caching.

Key behaviors under test:
1. get_conn returns a cached connection on subsequent calls (same thread).
2. conn_ctx yields the cached connection and evicts it on exception.
3. A closed connection is replaced by a fresh one.
4. Connections are NOT shared between threads.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, call, patch

import pytest


DSN = "postgresql://user:pass@localhost/testdb"


@pytest.fixture(autouse=True)
def clear_thread_local():
    """Reset thread-local connection cache before every test."""
    from rag_v1.db import pg as pg_module
    if hasattr(pg_module._local, "cache"):
        pg_module._local.cache.clear()
    yield
    if hasattr(pg_module._local, "cache"):
        pg_module._local.cache.clear()


def _make_open_conn():
    conn = MagicMock()
    conn.closed = False
    return conn


# ---------------------------------------------------------------------------
# get_conn
# ---------------------------------------------------------------------------

class TestGetConn:
    def test_returns_connection(self):
        mock_conn = _make_open_conn()
        with patch("psycopg.connect", return_value=mock_conn):
            from rag_v1.db.pg import get_conn
            conn = get_conn(DSN)
            assert conn is mock_conn

    def test_caches_connection_on_second_call(self):
        mock_conn = _make_open_conn()
        with patch("psycopg.connect", return_value=mock_conn) as mock_connect:
            from rag_v1.db.pg import get_conn
            c1 = get_conn(DSN)
            c2 = get_conn(DSN)
            assert c1 is c2
            # psycopg.connect called only once
            assert mock_connect.call_count == 1

    def test_reconnects_when_connection_is_closed(self):
        conn1 = _make_open_conn()
        conn2 = _make_open_conn()
        with patch("psycopg.connect", side_effect=[conn1, conn2]) as mock_connect:
            from rag_v1.db.pg import get_conn
            c1 = get_conn(DSN)
            # Simulate server closing the connection
            c1.closed = True
            c2 = get_conn(DSN)
            assert c2 is conn2
            assert mock_connect.call_count == 2

    def test_different_dsn_different_connection(self):
        conn_a = _make_open_conn()
        conn_b = _make_open_conn()
        dsn_a = "postgresql://a/db"
        dsn_b = "postgresql://b/db"
        with patch("psycopg.connect", side_effect=[conn_a, conn_b]):
            from rag_v1.db.pg import get_conn
            ca = get_conn(dsn_a)
            cb = get_conn(dsn_b)
            assert ca is conn_a
            assert cb is conn_b

    def test_different_threads_get_different_connections(self):
        connections: list = []
        lock = threading.Lock()

        def worker():
            conn = _make_open_conn()
            with patch("psycopg.connect", return_value=conn):
                from rag_v1.db.pg import get_conn
                c = get_conn("postgresql://localhost/db")
                with lock:
                    connections.append(c)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # Each thread should have received its own mock connection object
        assert len(connections) == 2


# ---------------------------------------------------------------------------
# conn_ctx
# ---------------------------------------------------------------------------

class TestConnCtx:
    def test_yields_connection(self):
        mock_conn = _make_open_conn()
        with patch("psycopg.connect", return_value=mock_conn):
            from rag_v1.db.pg import conn_ctx
            with conn_ctx(DSN) as conn:
                assert conn is mock_conn

    def test_evicts_connection_on_exception(self):
        conn1 = _make_open_conn()
        conn2 = _make_open_conn()
        with patch("psycopg.connect", side_effect=[conn1, conn2]) as mock_connect:
            from rag_v1.db.pg import conn_ctx, get_conn

            # First context — force an exception to evict
            with pytest.raises(RuntimeError):
                with conn_ctx(DSN) as conn:
                    raise RuntimeError("DB error")

            # conn1 should have been closed
            conn1.close.assert_called_once()

            # Next get_conn must reconnect (conn1 was evicted)
            c = get_conn(DSN)
            assert c is conn2
            assert mock_connect.call_count == 2

    def test_does_not_evict_on_success(self):
        conn1 = _make_open_conn()
        conn2 = _make_open_conn()
        with patch("psycopg.connect", side_effect=[conn1, conn2]) as mock_connect:
            from rag_v1.db.pg import conn_ctx, get_conn

            with conn_ctx(DSN):
                pass  # success, no exception

            # Connection is still cached; get_conn should return same object
            c = get_conn(DSN)
            assert c is conn1
            assert mock_connect.call_count == 1

    def test_reraises_original_exception(self):
        mock_conn = _make_open_conn()
        with patch("psycopg.connect", return_value=mock_conn):
            from rag_v1.db.pg import conn_ctx
            with pytest.raises(ValueError, match="boom"):
                with conn_ctx(DSN):
                    raise ValueError("boom")
