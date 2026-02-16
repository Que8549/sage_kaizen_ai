import psycopg
from psycopg.rows import dict_row, DictRow
from typing import cast

def get_conn(dsn: str) -> psycopg.Connection[DictRow]:
    conn = psycopg.connect(dsn, row_factory=dict_row)  # type: ignore[arg-type]
    return cast(psycopg.Connection[DictRow], conn)
