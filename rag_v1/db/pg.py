"""
rag_v1/db/pg.py

PostgreSQL connection helpers for the RAG pipeline.

Connection caching
------------------
psycopg.connect() performs a full TCP handshake + SSL + auth round-trip on
every call (typically 5–50 ms on localhost).  With 2–4 parallel DB queries per
chat turn (doc-RAG, wiki-RAG, possibly feedback) that overhead compounds.

`get_conn` caches one connection *per thread* via threading.local().  This is
safe because:
  - The RAG thread pool (ThreadPoolExecutor, max_workers=4) keeps its threads
    alive for the process lifetime, so cached connections persist across turns.
  - Each worker thread uses its own connection independently — no mutex needed.
  - autocommit=True means every SELECT is its own implicit transaction; no
    explicit BEGIN/COMMIT is needed for read-only RAG queries.
  - On any failure the connection is evicted so the next call reconnects cleanly.

get_conn() is a drop-in replacement: callers that do `with get_conn(dsn) as conn`
still work because psycopg3 Connection.__exit__ is a no-op in autocommit mode.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import psycopg
from psycopg.rows import dict_row, DictRow


# Per-thread connection cache:  {dsn: connection}
_local = threading.local()


def _thread_conn(dsn: str) -> psycopg.Connection[DictRow]:
    """Return or create a cached autocommit connection for this thread + DSN."""
    cache: dict = getattr(_local, "cache", None)
    if cache is None:
        _local.cache = cache = {}

    conn = cache.get(dsn)
    if conn is None or conn.closed:
        conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
        cache[dsn] = conn

    return cast(psycopg.Connection[DictRow], conn)


def get_conn(dsn: str) -> psycopg.Connection[DictRow]:
    """
    Return a thread-local cached connection for the given DSN.

    The first call per thread/DSN opens the connection; subsequent calls
    return the cached connection instantly (no TCP overhead).

    Callers may use the returned connection directly or as a context manager:
        with get_conn(dsn) as conn:   # __exit__ is a no-op in autocommit mode
            rows = conn.execute(sql, params).fetchall()
    """
    return _thread_conn(dsn)


@contextmanager
def conn_ctx(dsn: str) -> Iterator[psycopg.Connection[DictRow]]:
    """
    Context manager that yields a thread-local cached connection.

    On exception, closes and evicts the connection so the next caller
    gets a fresh one (handles server-side disconnects gracefully).
    """
    conn = _thread_conn(dsn)
    try:
        yield conn
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        getattr(_local, "cache", {}).pop(dsn, None)
        raise
