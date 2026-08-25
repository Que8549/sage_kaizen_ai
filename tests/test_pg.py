"""
tests/test_pg.py

Unit tests for rag_v1/db/pg.py.

Rewritten 2026-08-05: this module moved from a hand-rolled threading.local()
connection cache onto psycopg_pool.ConnectionPool, converging with memory/db.py
(the project previously had two different connection strategies).

The pool is mocked — nothing here opens a socket.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

import rag_v1.db.pg as pg

DSN = "postgresql://user@localhost/testdb"
DSN2 = "postgresql://user@localhost/other"


@pytest.fixture(autouse=True)
def clean_pools():
    """Each test starts with no pools registered."""
    with patch.dict(pg._pools, {}, clear=True):
        yield


@pytest.fixture
def fake_pool_cls():
    """Patch ConnectionPool with a factory recording constructor kwargs."""
    created: list[MagicMock] = []

    def _factory(**kwargs):
        pool = MagicMock()
        pool.init_kwargs = kwargs
        conn = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        pool.conn = conn
        created.append(pool)
        return pool

    with patch.object(pg, "ConnectionPool", side_effect=_factory) as cls:
        cls.check_connection = MagicMock(name="check_connection")
        yield cls, created


# ---------------------------------------------------------------------------
# Pool creation
# ---------------------------------------------------------------------------

class TestGetPool:
    def test_creates_a_pool_on_first_use(self, fake_pool_cls):
        cls, created = fake_pool_cls
        pg._get_pool(DSN)
        assert len(created) == 1
        assert cls.call_args.kwargs["conninfo"] == DSN

    def test_reuses_the_pool_for_the_same_dsn(self, fake_pool_cls):
        _, created = fake_pool_cls
        assert pg._get_pool(DSN) is pg._get_pool(DSN)
        assert len(created) == 1

    def test_separate_pools_per_dsn(self, fake_pool_cls):
        _, created = fake_pool_cls
        assert pg._get_pool(DSN) is not pg._get_pool(DSN2)
        assert len(created) == 2

    def test_opens_eagerly(self, fake_pool_cls):
        cls, _ = fake_pool_cls
        pg._get_pool(DSN)
        assert cls.call_args.kwargs["open"] is True

    def test_uses_dict_rows_and_autocommit(self, fake_pool_cls):
        """RAG reads are single-statement; autocommit avoids idle-in-transaction."""
        cls, _ = fake_pool_cls
        pg._get_pool(DSN)
        kw = cls.call_args.kwargs["kwargs"]
        assert kw["autocommit"] is True
        assert kw["row_factory"] is pg.dict_row

    def test_validates_connections_on_checkout(self, fake_pool_cls):
        """
        The reason for moving off threading.local(): a connection killed
        server-side stayed cached and every first query after a Postgres
        restart failed.
        """
        cls, _ = fake_pool_cls
        pg._get_pool(DSN)
        assert cls.call_args.kwargs["check"] is cls.check_connection

    def test_pool_is_bounded(self, fake_pool_cls):
        cls, _ = fake_pool_cls
        pg._get_pool(DSN)
        assert cls.call_args.kwargs["min_size"] == pg._POOL_MIN_SIZE
        assert cls.call_args.kwargs["max_size"] == pg._POOL_MAX_SIZE
        assert pg._POOL_MIN_SIZE < pg._POOL_MAX_SIZE

    def test_max_size_covers_the_context_injector_fanout(self):
        """The executor is sized at 2x a 5-way fan-out; the pool must not be the bottleneck."""
        from rag_v1.runtime.context_injector import _POOL
        assert pg._POOL_MAX_SIZE >= _POOL._max_workers

    def test_checkout_timeout_is_shorter_than_the_shortest_worker_budget(self):
        """
        Pool exhaustion should fail one context source fast, not consume that
        worker's whole ceiling. music/news have the tightest budget at 10 s.
        """
        from rag_v1.runtime.context_injector import _WORKER_TIMEOUTS
        assert pg._POOL_TIMEOUT_S < min(_WORKER_TIMEOUTS.values())

    def test_created_once_under_concurrency(self, fake_pool_cls):
        _, created = fake_pool_cls
        barrier = threading.Barrier(8)

        def _race():
            barrier.wait()
            pg._get_pool(DSN)

        threads = [threading.Thread(target=_race) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(created) == 1, f"pool built {len(created)} times — lock is broken"


# ---------------------------------------------------------------------------
# conn_ctx / get_conn
# ---------------------------------------------------------------------------

class TestConnCtx:
    def test_yields_a_connection(self, fake_pool_cls):
        _, created = fake_pool_cls
        with pg.conn_ctx(DSN) as conn:
            assert conn is created[0].conn

    def test_returns_the_connection_on_exit(self, fake_pool_cls):
        _, created = fake_pool_cls
        with pg.conn_ctx(DSN):
            pass
        created[0].connection.return_value.__exit__.assert_called_once()

    def test_returns_the_connection_on_exception(self, fake_pool_cls):
        _, created = fake_pool_cls
        with pytest.raises(RuntimeError):
            with pg.conn_ctx(DSN):
                raise RuntimeError("query blew up")
        created[0].connection.return_value.__exit__.assert_called_once()

    def test_exception_propagates(self, fake_pool_cls):
        with pytest.raises(ValueError, match="boom"):
            with pg.conn_ctx(DSN):
                raise ValueError("boom")


class TestGetConn:
    def test_is_a_context_manager(self, fake_pool_cls):
        _, created = fake_pool_cls
        with pg.get_conn(DSN) as conn:
            assert conn is created[0].conn

    def test_returns_the_connection_on_exit(self, fake_pool_cls):
        _, created = fake_pool_cls
        with pg.get_conn(DSN):
            pass
        created[0].connection.return_value.__exit__.assert_called_once()

    def test_shares_the_pool_with_conn_ctx(self, fake_pool_cls):
        _, created = fake_pool_cls
        with pg.get_conn(DSN):
            pass
        with pg.conn_ctx(DSN):
            pass
        assert len(created) == 1


# ---------------------------------------------------------------------------
# close_all_pools
# ---------------------------------------------------------------------------

class TestCloseAllPools:
    def test_closes_every_pool(self, fake_pool_cls):
        _, created = fake_pool_cls
        pg._get_pool(DSN)
        pg._get_pool(DSN2)
        pg.close_all_pools()
        for pool in created:
            pool.close.assert_called_once()

    def test_clears_the_registry(self, fake_pool_cls):
        pg._get_pool(DSN)
        pg.close_all_pools()
        assert pg._pools == {}

    def test_is_safe_with_no_pools(self):
        pg.close_all_pools()

    def test_survives_a_close_failure(self, fake_pool_cls):
        _, created = fake_pool_cls
        pg._get_pool(DSN)
        created[0].close.side_effect = OSError("already gone")
        pg.close_all_pools()      # must not raise
        assert pg._pools == {}

    def test_next_use_reopens(self, fake_pool_cls):
        _, created = fake_pool_cls
        pg._get_pool(DSN)
        pg.close_all_pools()
        pg._get_pool(DSN)
        assert len(created) == 2


# ---------------------------------------------------------------------------
# Convergence with memory/db.py
# ---------------------------------------------------------------------------

class TestSingleConnectionStrategy:
    def test_both_modules_use_psycopg_pool(self):
        """
        Before 2026-08-05 the project had two connection strategies:
        memory/db.py used ConnectionPool while rag_v1/db/pg.py hand-rolled a
        threading.local() cache. They now agree.
        """
        from memory import db as mdb
        assert mdb.ConnectionPool is pg.ConnectionPool

    def test_rag_pg_no_longer_uses_thread_local(self):
        assert not hasattr(pg, "_local")
        assert not hasattr(pg, "_thread_conn")
