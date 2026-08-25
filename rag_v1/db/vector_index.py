"""
rag_v1/db/vector_index.py

One source of truth for pgvector index tuning, plus the single execution path
that applies it.

Why this module exists
----------------------
Three separate places used to carry their own copy of these constants:

    scripts/migrate_wiki_chunks_partitioned.py   VECTOR_INDEX_AMS, IVFFLAT_*
    rag_v1/wiki/wiki_retriever.py                _WIKI_VECTOR_INDEX_AM, ...
    sage_kaizen_ai_ingest wiki_ingest.py         _VECTOR_INDEX_AMS

They had already drifted: the tuples were written in different orders, and the
migration script's comment claimed "WikiRetriever sets 10" long after
WikiRetriever had been retuned to 5.  Nothing detects that kind of drift,
because each copy is individually correct-looking.

This module lives in ``rag_v1/db/`` deliberately.  Per CLAUDE.md §13 that
package resolves to THIS repo for both projects, so the ingest app imports the
same object rather than a second copy — changing a value here changes it
everywhere, which is the entire point.

Why ``vector_search`` exists
----------------------------
Applying the tuning is as easy to get wrong as choosing it.  Five retrievers
run pgvector queries and, before this, only two configured anything:

    wiki_retriever   SET ivfflat.probes + SET statement_timeout
    retriever        SET hnsw.ef_search
    media/music/lyrics_retriever   nothing at all

The three that set nothing inherited whatever the previous user of that pooled
connection happened to leave behind, so their recall depended on connection
reuse order, and they had no timeout backstop — a slow query kept a Postgres
backend scanning long after the caller had given up.  Routing every vector
query through one function makes the tuning a property of the query rather
than of whichever retriever ran last.

``rag_v1/db/pg.py`` now also resets session GUCs when a connection returns to
the pool, so the two halves of that bug are closed independently: the reset
stops leakage, and this function stops the omission.
"""
from __future__ import annotations

from typing import Any, LiteralString, Sequence

from psycopg import sql
from psycopg.rows import DictRow, dict_row

from rag_v1.db.pg import conn_ctx

# --------------------------------------------------------------------------- #
# Index shape                                                                  #
# --------------------------------------------------------------------------- #

# Access methods that count as a usable vector index, most-preferred first.
# ivfflat leads because that is what wiki_chunks actually carries since the
# 2026-08-24 migration; hnsw stays because every other embedding table uses it
# and because an abandoned build may have left one behind.
VECTOR_INDEX_AMS: tuple[str, ...] = ("ivfflat", "hnsw")

# jina-clip-v2 output width.  The wiki halfvec index is built on
# (embedding::halfvec(1024)), and pgvector only uses that index when the query
# casts to exactly the same type and width — so this number is load-bearing in
# the SQL, not merely descriptive.
WIKI_EMBED_DIMS: int = 1024

# --------------------------------------------------------------------------- #
# Recall / latency tuning                                                      #
# --------------------------------------------------------------------------- #

# Lists used when building the wiki ivfflat indexes.  Changing this invalidates
# every existing index — it is a build-time property, not a query knob.
WIKI_IVFFLAT_LISTS: int = 4000

# NOT sqrt(lists).  pgvector's probes ~= sqrt(lists) guidance assumes ONE index.
# wiki_chunks is 32 HASH partitions and a nearest-neighbour query carries no
# page_id predicate, so it cannot prune — every query probes all 32 and the
# cost multiplies by 32.  sqrt(4000) = 63 would mean ~2016 lists scanned.
#
# Measured 2026-08-24 on COLD pages (10 random vectors across the 1.3 TB index;
# a warm repeat is ~1 s at any setting, so cold is the case that decides it):
#
#     probes=10   median 15.36 s   p90 74.17 s   1/10 OVER the 25 s timeout
#     probes=5    median  4.10 s   p90  8.76 s   0/10 over      <- chosen
#     probes=3    median  1.49 s   p90  2.96 s   0/10 over
#
# Recall@10 was identical at 5/10/20, so probing more buys latency and nothing.
# probes=3 is faster still but its recall sits at an unmeasured edge, so 5 buys
# margin for ~2.6 s of median latency.  A query that exceeds the timeout
# returns nothing and is indistinguishable from "no matches" — the exact
# failure wiki_retriever's index guard exists to prevent.
WIKI_IVFFLAT_PROBES: int = 5

# pgvector's ivfflat default is probes = 1, which scans one list out of 4000
# and returns almost nothing while looking exactly like an empty result set.
# Never rely on it.
IVFFLAT_DEFAULT_PROBES: int = 1

# pgvector's hnsw default is ef_search = 40.  100 is the value the doc-RAG path
# has always used; it is a recall/latency tradeoff, not a correctness one.
DEFAULT_HNSW_EF_SEARCH: int = 100

# Backstop, not the primary guard.  An indexed query returns in milliseconds;
# anything approaching this bound is a plan regression, and the caller's own
# deadline has long since passed.  Postgres must abandon it because the caller
# cannot: context_injector drops the future and moves on while the backend
# keeps scanning.
DEFAULT_QUERY_TIMEOUT_MS: int = 25_000


def apply_vector_tuning(
    conn: Any,
    *,
    index_am: str = "hnsw",
    probes: int = WIKI_IVFFLAT_PROBES,
    ef_search: int = DEFAULT_HNSW_EF_SEARCH,
    timeout_ms: int = DEFAULT_QUERY_TIMEOUT_MS,
) -> None:
    """
    Apply the recall knob and timeout to a connection the caller already holds.

    ``vector_search`` covers the common single-shot case.  This exists for the
    paths that cannot use it — music_retriever resolves a source row and then
    runs a similarity query against it, and both statements have to share one
    connection.  Splitting those into two pooled checkouts would be a
    correctness change (the second could see different data), so they keep the
    connection and call this instead.

    Same knobs, same defaults, one definition.
    """
    if index_am == "ivfflat":
        conn.execute(
            sql.SQL("SET ivfflat.probes = {n}").format(n=sql.Literal(probes))
        )
    else:
        conn.execute(
            sql.SQL("SET hnsw.ef_search = {n}").format(n=sql.Literal(ef_search))
        )
    conn.execute(
        sql.SQL("SET statement_timeout = {ms}").format(ms=sql.Literal(timeout_ms))
    )


def vector_search(
    dsn: str,
    query: LiteralString,
    params: Sequence[Any],
    *,
    index_am: str = "hnsw",
    probes: int = WIKI_IVFFLAT_PROBES,
    ef_search: int = DEFAULT_HNSW_EF_SEARCH,
    timeout_ms: int = DEFAULT_QUERY_TIMEOUT_MS,
) -> list[DictRow]:
    """
    Run one pgvector query with the right recall knob and a timeout.

    ``index_am`` selects which knob applies: ivfflat and hnsw have different
    ones and setting the wrong one is silent — an ivfflat scan left at the
    default probes = 1 simply returns too few rows.

    Rows come back as dicts; mapping them to a result type is the caller's job,
    since every retriever has its own.  The connection is returned to the pool
    before this returns, and ``pg.py``'s reset hook scrubs the GUCs set here.

    ``query`` is typed ``LiteralString`` because psycopg accepts only those as a
    bare query string — the type system enforcing its own injection guard.
    Callers pass module-level SQL constants, which satisfy it.
    """
    with conn_ctx(dsn) as conn:
        apply_vector_tuning(
            conn, index_am=index_am, probes=probes,
            ef_search=ef_search, timeout_ms=timeout_ms,
        )
        with conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(query, params).fetchall()
