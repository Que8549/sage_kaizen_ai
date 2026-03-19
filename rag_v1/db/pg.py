from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import psycopg
from psycopg.rows import dict_row, DictRow


def get_conn(dsn: str) -> psycopg.Connection[DictRow]:
    conn = psycopg.connect(dsn, row_factory=dict_row)  # type: ignore[arg-type]
    return cast(psycopg.Connection[DictRow], conn)


@contextmanager
def conn_ctx(dsn: str) -> Iterator[psycopg.Connection[DictRow]]:
    """Context manager that opens a DictRow connection and always closes it."""
    conn = get_conn(dsn)
    try:
        yield conn
    finally:
        conn.close()
