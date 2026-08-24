"""
rag_v1/db/pg.py

PostgreSQL connection helpers for the RAG pipeline.

Connection pooling
------------------
psycopg.connect() performs a full TCP handshake + SSL + auth round-trip on
every call (typically 5–50 ms on localhost).  With 5 parallel DB queries per
chat turn (doc-RAG, wiki-RAG, music, news, memory) that overhead compounds.

This module wraps ``psycopg_pool.ConnectionPool``, the same primitive
``memory/db.py`` already used — the project had two different connection
strategies before 2026-08-05, and this is the surviving one.

Why the pool rather than the previous threading.local() cache
-------------------------------------------------------------
The old implementation cached one connection per (thread, DSN) forever and only
evicted it when a query *raised*.  That has two problems the pool solves:

  1. A connection killed server-side — Postgres restart, idle timeout,
     ``pg_terminate_backend`` — stayed in the cache looking healthy, so the
     first query after any such event always failed and only the *second*
     succeeded.  ``ConnectionPool(check=...)`` validates on checkout instead.
  2. The cache grew with the thread pool and never shrank.  Since
     context_injector's executor was resized to 10 workers, that was up to 10
     idle connections per DSN held open for the process lifetime.  The pool
     bounds this with min_size/max_size.

The public API is unchanged: ``get_conn(dsn)`` and ``conn_ctx(dsn)`` still work
as before for every existing call site.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row, DictRow
from psycopg_pool import ConnectionPool

from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.rag_v1.db.pg")

# One pool per DSN.  Keyed rather than a single global because the retrievers
# are constructed with an explicit dsn argument and tests use a fake one.
# ConnectionPool is generic over its connection type; the bare form defaults
# to Connection[TupleRow], which our dict_row kwargs contradict.
_pools: dict[str, ConnectionPool[Any]] = {}
_pools_lock = threading.Lock()

# Sized for the context injector's fan-out (5 concurrent workers per turn,
# executor capped at 10) plus headroom for the memory service's background
# episode writer.  min_size=1 keeps one connection warm so the first query of a
# session does not pay the handshake.
_POOL_MIN_SIZE = 1
_POOL_MAX_SIZE = 12

# Seconds to wait for a free connection before raising.  Deliberately shorter
# than context_injector's shortest per-worker ceiling (10 s for music/news) so
# that pool exhaustion surfaces as a fast, logged failure for one context
# source rather than consuming that worker's whole budget.
_POOL_TIMEOUT_S = 8.0


def _reset_session(conn: psycopg.Connection[Any]) -> None:
    """
    Clear per-session state before a connection goes back to the pool.

    Without this, ANN tuning leaks between unrelated callers.  Every retriever
    shares one pool per DSN, and several of them issue session-level ``SET``
    before querying:

        wiki_retriever  SET ivfflat.probes / SET statement_timeout
        retriever       SET hnsw.ef_search

    A ``SET`` outside a transaction lasts for the life of the *connection*, not
    the query, so whichever retriever ran last silently dictated the recall and
    timeout of whoever checked that connection out next.  The media, music and
    lyrics retrievers set nothing at all, which made them the worst affected:
    their effective ``hnsw.ef_search`` depended on connection reuse order, and
    after a wiki query they inherited a 25 s ``statement_timeout`` they were
    never designed around.

    ``RESET ALL`` restores every GUC to its configured default.  It does not
    touch ``row_factory`` or ``autocommit`` — those are client-side psycopg
    settings applied at connect time, not server GUCs.

    psycopg calls this on every return to the pool and guarantees the
    connection is idle (no open transaction) when it does.  If it raises, the
    pool discards the connection rather than handing on a dirty one, which is
    the correct failure mode.
    """
    conn.execute("RESET ALL")


def _get_pool(dsn: str) -> ConnectionPool[Any]:
    """Return (creating on first use) the shared pool for this DSN."""
    pool = _pools.get(dsn)
    if pool is not None:
        return pool
    with _pools_lock:
        # Double-checked: another thread may have opened it while we waited.
        pool = _pools.get(dsn)
        if pool is None:
            pool = ConnectionPool(
                conninfo=dsn,
                min_size=_POOL_MIN_SIZE,
                max_size=_POOL_MAX_SIZE,
                timeout=_POOL_TIMEOUT_S,
                kwargs={"row_factory": dict_row, "autocommit": True},
                # Validate on checkout so a server-side disconnect costs a
                # transparent reconnect instead of failing the caller's query.
                check=ConnectionPool.check_connection,
                # Scrub session GUCs on return so ANN tuning set by one
                # retriever cannot silently govern the next one's query.
                reset=_reset_session,
                open=True,
            )
            _pools[dsn] = pool
            _LOG.info(
                "rag_v1.db.pg | pool opened (min=%d max=%d timeout=%.0fs)",
                _POOL_MIN_SIZE, _POOL_MAX_SIZE, _POOL_TIMEOUT_S,
            )
    return pool


@contextmanager
def conn_ctx(dsn: str) -> Iterator[psycopg.Connection[DictRow]]:
    """
    Yield a pooled autocommit connection with dict rows.

    The connection is returned to the pool on exit, including on exception.
    Callers must not hold the connection beyond the ``with`` block.
    """
    with _get_pool(dsn).connection() as conn:
        yield conn  # type: ignore[misc]


@contextmanager
def get_conn(dsn: str) -> Iterator[psycopg.Connection[DictRow]]:
    """
    Alias of :func:`conn_ctx`, kept for the existing call sites.

    NOTE: this is a context manager, not a bare connection.  It was already
    used as ``with get_conn(dsn) as conn:`` everywhere, which worked under the
    old thread-local implementation because psycopg's ``Connection.__exit__``
    is a no-op in autocommit mode.  Under the pool that ``with`` is load-bearing
    — it is what returns the connection — so the signature is now honest about
    it rather than relying on a coincidence.
    """
    with conn_ctx(dsn) as conn:
        yield conn


def close_all_pools() -> None:
    """
    Close every open pool.  For test teardown and clean process shutdown.

    Safe to call when no pool was ever opened.
    """
    with _pools_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        try:
            pool.close()
        except Exception:
            _LOG.debug("rag_v1.db.pg | pool close failed (ignored)", exc_info=True)
