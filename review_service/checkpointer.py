"""
review_service/checkpointer.py — PostgreSQL checkpoint persistence.

Uses the existing pg_settings.py DSN (same database as feedback dataset).
Creates LangGraph checkpoint tables on first call (idempotent via asetup()).

Why not a module-level singleton: AsyncPostgresSaver owns a connection pool
tied to a specific asyncio event loop. The ReviewRunner creates a new event
loop per run (to isolate from Streamlit). The checkpointer must be created
inside that loop — hence a context manager, not a singleton.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from pg_settings import PgSettings
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.review_service.checkpointer")
_setup_done: bool = False


@asynccontextmanager
async def make_checkpointer():
    """
    Async context manager that yields a ready AsyncPostgresSaver.

    Creates the three LangGraph tables (checkpoints, checkpoint_blobs,
    checkpoint_migrations) on first use via asetup(). Safe to call every
    run — asetup() is idempotent.

    Usage
    -----
    async with make_checkpointer() as cp:
        graph = build_review_graph(cp)
        await graph.ainvoke(state, config)
    """
    global _setup_done
    cfg = PgSettings()
    _LOG.debug("Opening PostgreSQL checkpointer at %s", cfg.pg_host)
    async with AsyncPostgresSaver.from_conn_string(cfg.pg_dsn) as saver:
        if not _setup_done:
            await saver.asetup()
            _setup_done = True
            _LOG.info("LangGraph checkpoint tables created/verified")
        yield saver
