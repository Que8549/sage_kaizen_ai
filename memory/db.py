"""
memory/db.py
Shared database primitives for the memory package.

All memory modules import from here — never from each other's private symbols.
This is the single place to swap the pool implementation.
"""
from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from pg_settings import PgSettings
from sk_logging import get_logger

_LOG  = get_logger("sage_kaizen.memory.db")
_lock = threading.Lock()
_pool: Optional[ConnectionPool] = None


def get_pool() -> ConnectionPool:
    """Return the shared psycopg3 connection pool, initialising it on first call."""
    global _pool
    if _pool is not None:
        return _pool
    with _lock:
        # Double-checked locking — another thread may have initialised while we waited.
        if _pool is None:
            dsn = PgSettings().pg_dsn
            _pool = ConnectionPool(
                conninfo=dsn,
                min_size=1,
                max_size=5,
                kwargs={"row_factory": dict_row, "autocommit": False},
                open=True,
            )
            _LOG.info("memory.db | connection pool opened (min=1, max=5)")
    return _pool


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """Context manager that yields a connection from the shared pool."""
    with get_pool().connection() as conn:
        yield conn


def new_uuid() -> str:
    return str(uuid.uuid4())


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def dumps(obj: Any) -> str:
    return json.dumps(obj)


def vec_str(embedding: list[float]) -> str:
    """
    Serialise a float vector to pgvector's '[f1,f2,...]' literal.

    Uses format specifier '.17g' instead of str() to avoid locale-dependent
    decimal separators (e.g. commas on LC_NUMERIC=de_DE systems).
    """
    return "[" + ",".join(format(v, ".17g") for v in embedding) + "]"
