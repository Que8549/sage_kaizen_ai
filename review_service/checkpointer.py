"""
review_service/checkpointer.py — PostgreSQL checkpoint persistence.

Uses the existing pg_settings.py DSN (same database as feedback dataset).
LangGraph checkpoint tables live in the dedicated `langgraph` schema
(not `public`) so the `sage` user only needs privileges on that schema.

Pre-requisite: run scripts/setup_langgraph_schema.sql once as the postgres
superuser to create the schema and grant sage ownership.

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

# Dedicated schema for LangGraph tables — keeps them out of public and avoids
# the PostgreSQL 15+ restriction where non-owners cannot CREATE in public.
_LANGGRAPH_SCHEMA = "langgraph"


@asynccontextmanager
async def make_checkpointer():
    """
    Async context manager that yields a ready AsyncPostgresSaver.

    The connection string sets search_path=langgraph so that LangGraph's
    un-qualified CREATE TABLE / SELECT statements resolve to the langgraph
    schema, not public.

    setup() creates three tables on first use (idempotent):
      langgraph.checkpoints
      langgraph.checkpoint_blobs
      langgraph.checkpoint_migrations

    Pre-requisite: run scripts/setup_langgraph_schema.sql once as postgres
    superuser before the first review run.

    Usage
    -----
    async with make_checkpointer() as cp:
        graph = build_review_graph(cp)
        await graph.ainvoke(state, config)
    """
    global _setup_done
    cfg = PgSettings()

    # Append search_path to DSN so all un-qualified table refs hit langgraph
    # before public.  psycopg supports options= as a query parameter.
    dsn_with_schema = (
        cfg.pg_dsn + f"?options=-csearch_path%3D{_LANGGRAPH_SCHEMA}"
    )

    _LOG.debug("Opening PostgreSQL checkpointer | schema=%s host=%s", _LANGGRAPH_SCHEMA, cfg.pg_host)
    async with AsyncPostgresSaver.from_conn_string(dsn_with_schema) as saver:
        if not _setup_done:
            await saver.setup()
            _setup_done = True
            _LOG.info(
                "LangGraph checkpoint tables created/verified in schema=%s",
                _LANGGRAPH_SCHEMA,
            )
        yield saver
