from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import psycopg
from psycopg import sql as _sql
from psycopg.rows import DictRow, dict_row

from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.feedback")

_SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"


# ──────────────────────────────────────────────────────────────────────────── #
# Connection                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

def get_conn(dsn: str) -> psycopg.Connection[DictRow]:
    conn = psycopg.connect(dsn, row_factory=dict_row)  # type: ignore[arg-type]
    return cast(psycopg.Connection[DictRow], conn)


# ──────────────────────────────────────────────────────────────────────────── #
# Schema init guard                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

def ensure_schema(dsn: str) -> None:
    """Create feedback schema + table if they do not exist.

    Idempotent — safe to call on every Streamlit startup.
    """
    with get_conn(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL.read_bytes())  # bytes satisfies QueryNoTemplate
        conn.commit()
    _LOG.debug("feedback.ensure_schema: OK")


# ──────────────────────────────────────────────────────────────────────────── #
# Insert                                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

def insert_rating(
    conn: psycopg.Connection[DictRow],
    *,
    id: str,
    brain: str,
    model_id: str,
    endpoint: str,
    route_score: float,
    route_reasons: list[Any],
    templates: list[Any],
    prompt_messages: list[Any],
    assistant_text: str,
    thumb: int,
    notes: str = "",
) -> None:
    """Insert one feedback rating row.

    Uses ON CONFLICT DO NOTHING — safe against Streamlit double-rerun button clicks.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.ratings
                (id, brain, model_id, endpoint, route_score, route_reasons,
                 templates, prompt_messages, assistant_text, thumb, notes)
            VALUES
                (%s::uuid, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
            id,
            brain,
            model_id,
            endpoint,
            route_score,
            json.dumps(route_reasons),
            json.dumps(templates),
            json.dumps(prompt_messages),
            assistant_text,
            thumb,
            notes,
        ))
    conn.commit()
    _LOG.info(
        "feedback.insert_rating | id=%s | brain=%s | thumb=%+d",
        id, brain, thumb,
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Stats                                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

def fetch_stats(conn: psycopg.Connection[DictRow]) -> dict[str, int]:
    """Return rating counters for the sidebar widget."""
    with conn.cursor() as cur:
        row = cur.execute(
            """
            SELECT
                COUNT(*)                                               AS total,
                COUNT(*) FILTER (WHERE thumb =  1)                    AS thumbs_up,
                COUNT(*) FILTER (WHERE thumb = -1)                    AS thumbs_down,
                COUNT(*) FILTER (WHERE brain = 'FAST' AND thumb =  1) AS fast_up,
                COUNT(*) FILTER (WHERE brain = 'FAST' AND thumb = -1) AS fast_down,
                COUNT(*) FILTER (WHERE brain = 'ARCHITECT' AND thumb =  1) AS arch_up,
                COUNT(*) FILTER (WHERE brain = 'ARCHITECT' AND thumb = -1) AS arch_down
            FROM public.ratings
            """
        ).fetchone()
    if row is None:
        return {}
    return {k: int(v or 0) for k, v in row.items()}


# ──────────────────────────────────────────────────────────────────────────── #
# Export query                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

def fetch_kto_rows(
    conn: psycopg.Connection[DictRow],
    *,
    brain: str | None = None,
    thumb: int | None = None,
    min_chars: int = 50,
) -> list[dict[str, Any]]:
    """Return all rows that pass filters, ordered by ts_utc ascending.

    Args:
        brain:     Optional filter: "FAST" or "ARCHITECT".
        thumb:     Optional filter: +1 or -1.
        min_chars: Skip responses shorter than this (low-signal rows).
    """
    filter_parts: list[_sql.Composable] = [
        _sql.SQL("char_length(assistant_text) >= %s"),
    ]
    params: list[Any] = [min_chars]

    if brain is not None:
        filter_parts.append(_sql.SQL("brain = %s"))
        params.append(brain)
    if thumb is not None:
        filter_parts.append(_sql.SQL("thumb = %s"))
        params.append(thumb)

    query = _sql.SQL(
        """
        SELECT id::text, ts_utc, brain, model_id, prompt_messages,
               assistant_text, thumb, notes
        FROM public.ratings
        WHERE {where}
        ORDER BY ts_utc ASC
        """
    ).format(where=_sql.SQL(" AND ").join(filter_parts))

    with conn.cursor() as cur:
        rows = cur.execute(query, params).fetchall()
    return [dict(r) for r in rows]
