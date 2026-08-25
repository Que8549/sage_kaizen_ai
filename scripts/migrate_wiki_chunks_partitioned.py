"""
scripts/migrate_wiki_chunks_partitioned.py

Convert wiki_chunks into a HASH-partitioned table and build one halfvec ANN
index per partition — resumably, on a machine that reboots under load.

    python scripts/migrate_wiki_chunks_partitioned.py --status
    python scripts/migrate_wiki_chunks_partitioned.py --create
    python scripts/migrate_wiki_chunks_partitioned.py --copy
    python scripts/migrate_wiki_chunks_partitioned.py --index
    python scripts/migrate_wiki_chunks_partitioned.py --swap --i-understand-this-is-destructive

Every phase is idempotent and restartable. Run any of them repeatedly; each
picks up where the last crash left it.

WHY THIS EXISTS
---------------
wiki_chunks is 508M rows / 3503 GB with no ANN index, so wiki-RAG is a full
sequential scan and is effectively dead (main CLAUDE.md §17). The obvious fix —
one CREATE INDEX over the whole table — cannot work here: the estimated build
is days to weeks, and this host records ~9 unclean shutdowns in 7 days with the
root cause still open (ingest CLAUDE.md §18). PostgreSQL has no resumable index
build, so every crash throws away all progress and leaves an INVALID index.

Partitioning makes the unit of work smaller than the mean time between crashes.
Each partition's index is an independent, restartable build, and a crash costs
one partition instead of the entire run.

WHY 32 PARTITIONS
-----------------
Originally chosen so one partition's HNSW graph (~36 GB at halfvec) fit in a
48 GB maintenance_work_mem. That reasoning was sound but incomplete, and HNSW
has since been abandoned — see the ivfflat block further down for the measured
reason. The partition count still stands on its own merits: it bounds each unit
of work below this host's mean time between crashes, which is the whole design
constraint.

halfvec remains, for two independent reasons: it halves the vector bytes read
per build (the actual bottleneck), and full-precision vector(1024) would put the
finished index past the capacity of the NVMe it lives on.

HASH, not the existing first_letter column: letter distribution is heavily
skewed (S/C/M vs X/Z), and vector search probes every partition anyway, so even
sizing is worth more than semantic grouping.

SCHEMA CONSEQUENCE YOU MUST ACCEPT
----------------------------------
PostgreSQL requires every UNIQUE/PRIMARY KEY on a partitioned table to include
all partition key columns. Partitioning by HASH(page_id) means:

    uq_wiki_chunks_page_hash (page_id, chunk_hash)  -> already includes it, unchanged
    wiki_chunks_pkey (chunk_id)                     -> ILLEGAL, becomes (chunk_id, page_id)

The ingest project's INSERT uses ON CONFLICT (page_id, chunk_hash) DO NOTHING
and its delete is WHERE page_id = %s, so both keep working — the delete even
gains partition pruning. But chunk_id alone stops being globally unique-enforced
(it stays unique in practice: it is a bigint identity column). This is a
cross-project schema change; sage_kaizen_ai_ingest writes this table.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import urllib.parse  # noqa: E402

import psycopg  # noqa: E402
from psycopg import sql  # noqa: E402
from psycopg.rows import DictRow, dict_row  # noqa: E402

from pg_settings import PgSettings  # noqa: E402
from rag_v1.db.vector_index import (  # noqa: E402
    VECTOR_INDEX_AMS,
    WIKI_EMBED_DIMS,
    WIKI_IVFFLAT_LISTS,
    WIKI_IVFFLAT_PROBES,
)


def _one(cur: psycopg.Cursor[DictRow]) -> DictRow:
    """
    fetchone() that fails loudly instead of returning Optional.

    Every call site here runs an aggregate or a to_regclass lookup, which always
    yields exactly one row. None would mean the server returned no result set at
    all — a bug worth surfacing here rather than as a NoneType subscript error
    three frames away.
    """
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("expected exactly one row, got none")
    return row

SOURCE = "wiki_chunks"
TARGET = "wiki_chunks_part"
STATE_TABLE = "wiki_chunks_migration"
CORRUPT_TABLE = "wiki_chunks_corrupt"

DEFAULT_PARTITIONS = 32
DEFAULT_BATCH_ROWS = 200_000
DEFAULT_MWM = "48GB"
DEFAULT_INDEX_TABLESPACE = "sage_nvme"
EMBED_DIMS = WIKI_EMBED_DIMS

# ---------------------------------------------------------------------------
# ANN index: ivfflat, not HNSW  (decision 2026-08-17)
# ---------------------------------------------------------------------------
# HNSW was tried first and abandoned on measurement, not theory. Partition 1 of
# 32 ran for 8.24 hours and had written 8.6 GB of an expected ~35 GB — about
# 25% — which projects to ~33 h per partition and ~44 DAYS for the table.
#
# The bottleneck was never the graph or the NVMe (which was writing 94.5 MB/s
# with 4 parallel workers). It was reading each partition's source off the HDD:
# ~24 GB of heap plus ~65 GB of TOASTed vectors, randomly accessed, delivering
# only 23.4 MB/s. The 36 GB-per-partition sizing that made the graph fit in
# maintenance_work_mem was sound; it just did not address read cost.
#
# ivfflat builds in essentially one pass (k-means on a sample, then assign each
# row to a list) instead of incremental graph construction, so it pays that read
# cost once rather than repeatedly. Lower recall than HNSW, but tunable via
# probes — and wiki-RAG returning good-enough results in hours beats perfect
# results in six weeks, given the whole point is that it currently returns
# nothing at all (CLAUDE.md §17).
#
# lists: pgvector recommends sqrt(rows) above 1M rows. Per PARTITION that is
# sqrt(512M / 32) = sqrt(16.0M) ~= 4000 — not sqrt of the whole table, because
# each partition carries its own independent index.
# probes at query time is NOT sqrt(lists): a vector query cannot prune
# partitions, so all 32 are probed and the cost multiplies by 32. The full
# cold-page measurement table lives in rag_v1/db/vector_index.py, which is the
# one place these values are defined — this script and WikiRetriever both read
# them from there, because three local copies had already drifted apart once.
IVFFLAT_LISTS = WIKI_IVFFLAT_LISTS
IVFFLAT_PROBES = WIKI_IVFFLAT_PROBES

# VECTOR_INDEX_AMS is imported: it also lets _drop_invalid_indexes and
# _partition_has_valid_index recognise (and clean up) leftovers from the
# abandoned HNSW attempt.


# --------------------------------------------------------------------------- #
# State                                                                        #
# --------------------------------------------------------------------------- #

_DDL_STATE = f"""
CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
    phase          text PRIMARY KEY,
    last_chunk_id  bigint,
    rows_done      bigint      NOT NULL DEFAULT 0,
    partitions     integer,
    started_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    finished_at    timestamptz
);
"""


@dataclass
class Phase:
    name: str
    last_chunk_id: int | None
    rows_done: int
    partitions: int | None
    finished: bool


class _OwnerSettings(PgSettings):
    """
    PG_OWNER / PG_OWNER_PASSWORD on top of the shared PG_* bindings.

    Subclassed rather than read with os.getenv() because these live in the
    project's .env, which only pydantic-settings loads — os.getenv() sees them
    only if something exported them first. (wiki_ingest.py's own _owner_dsn()
    uses os.getenv and therefore depends on that export; this does not.)
    """

    pg_owner: str = "postgres"
    pg_owner_password: str = ""


def _owner_dsn() -> str | None:
    """
    Build a DSN for the table OWNER.

    Measured 2026-08-06: `wiki_chunks` is owned by `postgres`, the app role
    `sage` is not a superuser, and `has_schema_privilege('sage','public',
    'CREATE')` is FALSE. So every phase here — creating the partitioned table,
    renaming, and CREATE TABLESPACE (superuser-only) — has to run as the owner.
    Returns None when the password is unset, so the caller can say why rather
    than failing later with a bare permission error.
    """
    cfg = _OwnerSettings()
    if not cfg.pg_owner_password:
        return None
    parsed = urllib.parse.urlparse(cfg.pg_dsn)
    # safe="" matters: quote() defaults to safe="/", so a password containing a
    # slash is left unencoded and silently corrupts the netloc — urlparse then
    # reads the rest as a path and the password comes back as None, producing a
    # DSN that fails to authenticate for no visible reason.
    return urllib.parse.urlunparse(
        parsed._replace(
            netloc=f"{urllib.parse.quote(cfg.pg_owner, safe='')}:"
                   f"{urllib.parse.quote(cfg.pg_owner_password, safe='')}"
                   f"@{parsed.hostname}:{parsed.port or 5432}"
        )
    )


def _connect(dsn: str) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=True)
    # The app role's search_path is 'langgraph, public'. Unqualified DDL here
    # would land in the langgraph schema, which CLAUDE.md §5.6 reserves for
    # LangGraph checkpoints — that is exactly what an earlier --status run did
    # before this line existed. Pin it.
    conn.execute("SET search_path = public")
    return conn


def _read_phase(conn: psycopg.Connection, phase: str) -> Phase | None:
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(
            f"SELECT * FROM {STATE_TABLE} WHERE phase = %s", (phase,)
        ).fetchone()
    if row is None:
        return None
    return Phase(
        name=row["phase"],
        last_chunk_id=row["last_chunk_id"],
        rows_done=row["rows_done"],
        partitions=row["partitions"],
        finished=row["finished_at"] is not None,
    )


def _mark(conn: psycopg.Connection, phase: str, *, last_chunk_id: int | None = None,
          rows_done: int | None = None, partitions: int | None = None,
          finished: bool = False) -> None:
    conn.execute(
        f"""
        INSERT INTO {STATE_TABLE} (phase, last_chunk_id, rows_done, partitions,
                                   finished_at)
        VALUES (%s, %s, COALESCE(%s, 0), %s, CASE WHEN %s THEN now() END)
        ON CONFLICT (phase) DO UPDATE SET
            last_chunk_id = COALESCE(EXCLUDED.last_chunk_id, {STATE_TABLE}.last_chunk_id),
            rows_done     = COALESCE(%s, {STATE_TABLE}.rows_done),
            partitions    = COALESCE(EXCLUDED.partitions, {STATE_TABLE}.partitions),
            updated_at    = now(),
            finished_at   = CASE WHEN %s THEN now() ELSE {STATE_TABLE}.finished_at END
        """,
        (phase, last_chunk_id, rows_done, partitions, finished, rows_done, finished),
    )


# --------------------------------------------------------------------------- #
# Phase 1 — create the partitioned table                                       #
# --------------------------------------------------------------------------- #

def _partition_name(i: int) -> str:
    return f"{TARGET}_p{i:03d}"


def phase_create(conn: psycopg.Connection, partitions: int) -> int:
    """
    Build the empty partitioned table and its partitions.

    Deliberately WITHOUT the ANN index and without the btree indexes: every
    index present during the copy is index maintenance paid per row, on a
    machine we expect to be interrupted. They are added afterwards.
    """
    existing = _read_phase(conn, "create")
    if existing and existing.finished:
        print(f"  create: already done ({existing.partitions} partitions) — skipping")
        return existing.partitions or partitions

    print(f"  create: {TARGET} PARTITION BY HASH (page_id), {partitions} partitions")
    # Composed via psycopg.sql rather than f-strings. psycopg accepts a bare
    # str query only when it is a LiteralString; anything interpolated becomes
    # plain str and is refused, which is the type system enforcing the
    # injection guard rather than being pedantic. Identifier() also quotes the
    # names correctly instead of trusting that they need no quoting.
    conn.execute(
        sql.SQL("""
            CREATE TABLE IF NOT EXISTS {target} (
                chunk_id     bigint GENERATED ALWAYS AS IDENTITY,
                page_id      uuid   NOT NULL,
                bundle_id    uuid,
                title        text,
                first_letter character(1),
                section_path text[],
                chunk_index  integer,
                text         text,
                chunk_hash   text,
                embedding    vector({dims})
            ) PARTITION BY HASH (page_id)
        """).format(target=sql.Identifier(TARGET), dims=sql.Literal(EMBED_DIMS))
    )
    for i in range(partitions):
        conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {part} PARTITION OF {target} "
                "FOR VALUES WITH (MODULUS {mod}, REMAINDER {rem})"
            ).format(
                part=sql.Identifier(_partition_name(i)),
                target=sql.Identifier(TARGET),
                mod=sql.Literal(partitions),
                rem=sql.Literal(i),
            )
        )
    _mark(conn, "create", partitions=partitions, finished=True)
    print(f"  create: done")
    return partitions


# --------------------------------------------------------------------------- #
# Phase 2 — copy, in committed batches                                         #
# --------------------------------------------------------------------------- #

# page_id/title are what make a quarantined row ACTIONABLE. chunk_id alone is
# not: the new table generates fresh identity values, so after --swap the
# recorded chunk_id refers to nothing that still exists. Re-ingesting a lost
# chunk needs to know which page it came from.
#
# These columns are readable even for corrupt rows: the damage is in the TOAST
# relation (the 4104-byte embedding), while page_id and title live in the heap.
_DDL_CORRUPT = f"""
CREATE TABLE IF NOT EXISTS {CORRUPT_TABLE} (
    chunk_id    bigint PRIMARY KEY,
    page_id     uuid,
    title       text,
    detected_at timestamptz NOT NULL DEFAULT now(),
    error       text
);
ALTER TABLE {CORRUPT_TABLE} ADD COLUMN IF NOT EXISTS page_id uuid;
ALTER TABLE {CORRUPT_TABLE} ADD COLUMN IF NOT EXISTS title   text;
"""

_INSERT_RANGE = f"""
INSERT INTO {TARGET}
    (page_id, bundle_id, title, first_letter, section_path,
     chunk_index, text, chunk_hash, embedding)
SELECT page_id, bundle_id, title, first_letter, section_path,
       chunk_index, text, chunk_hash, embedding
FROM {SOURCE}
WHERE chunk_id > %s AND chunk_id <= %s
"""


def _set_partition_autovacuum(conn: psycopg.Connection, partitions: int,
                              enabled: bool) -> None:
    """
    Turn autovacuum off on the target partitions for the duration of the copy.

    Measured 2026-08-10: with autovacuum on, three workers ran VACUUM ANALYZE
    against the partitions being written while the copy's INSERT sat in
    DataFileRead for 114 s, and throughput collapsed from ~1,806 to **340
    rows/s**. The workers and the copy were contending for the same HDD.

    This is self-inflicted — config/postgres/sage_kaizen_tuning.conf raises
    autovacuum_vacuum_cost_limit to 2000 and drops naptime to 30 s, which is
    right for steady state and wrong during a bulk load.

    During the copy the target is INSERT-only: no dead tuples to reclaim, so the
    work is near-pure waste. Anti-wraparound vacuum still runs regardless of this
    setting, so freezing is not at risk. It MUST be re-enabled afterwards —
    phase_constraints does that, along with the ANALYZE the planner needs.
    """
    for i in range(partitions):
        conn.execute(
            sql.SQL("ALTER TABLE {} SET (autovacuum_enabled = {})").format(
                sql.Identifier(_partition_name(i)),
                sql.SQL("true" if enabled else "false"),
            )
        )


def _corrupt_count(conn: psycopg.Connection) -> int:
    with conn.cursor(row_factory=dict_row) as cur:
        return _one(cur.execute(f"SELECT count(*) AS n FROM {CORRUPT_TABLE}"))["n"]


def _quarantine(conn: psycopg.Connection, chunk_id: int, error: str) -> None:
    """Record an unreadable row, with the heap columns needed to re-ingest it."""
    conn.execute(
        f"""
        INSERT INTO {CORRUPT_TABLE} (chunk_id, page_id, title, error)
        SELECT %s, page_id, title, %s FROM {SOURCE} WHERE chunk_id = %s
        ON CONFLICT (chunk_id) DO NOTHING
        """,
        (chunk_id, error[:500], chunk_id),
    )


def backfill_corrupt_metadata(conn: psycopg.Connection) -> int:
    """
    Fill page_id/title for quarantined rows recorded before those columns existed.

    Safe to re-run: it only touches rows still missing page_id, and reads only
    heap columns, which are intact even for rows whose TOAST is damaged. Called
    from --status so the record self-heals rather than needing a separate step.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            UPDATE {CORRUPT_TABLE} q
            SET page_id = s.page_id, title = s.title
            FROM {SOURCE} s
            WHERE s.chunk_id = q.chunk_id AND q.page_id IS NULL
            """
        )
        return cur.rowcount


def _copy_range(conn: psycopg.Connection, lo: int, hi: int,
                skipped: list[int]) -> int:
    """
    Copy source rows in (lo, hi], isolating and skipping unreadable ones.

    The source table has at least one damaged TOAST page — confirmed
    2026-08-10, chunk_ids 173,810,706 and 173,810,708, detected only because
    `data_checksums = on`. Postgres raises DataCorrupted for the whole
    statement, so a single bad 8 KB page would otherwise block a 100,000-row
    batch and stall the migration permanently.

    On failure the range is bisected down to individual rows; the ones that
    genuinely cannot be read are recorded in wiki_chunks_corrupt and skipped.
    A single INSERT is atomic, so a failed attempt copies nothing and
    subdividing cannot double-insert.

    Rows quarantined here are lost from the new table but NOT from the source,
    which is left untouched — no zero_damaged_pages, no destructive repair.
    They can be re-ingested from the ZIM dump later; wiki_ingest is resume-safe
    by content_hash.
    """
    try:
        # Nested inside phase_copy's transaction this becomes a SAVEPOINT, so a
        # DataCorrupted failure rolls back only this sub-range instead of
        # poisoning the whole batch transaction and aborting the run.
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(_INSERT_RANGE, (lo, hi))
            return cur.rowcount
    except psycopg.errors.DataCorrupted as exc:
        if hi - lo <= 1:
            _quarantine(conn, hi, str(exc))
            skipped.append(hi)
            print(f"      quarantined corrupt chunk_id {hi:,}", flush=True)
            return 0
        mid = (lo + hi) // 2
        return (_copy_range(conn, lo, mid, skipped)
                + _copy_range(conn, mid, hi, skipped))


def phase_copy(conn: psycopg.Connection, batch_rows: int) -> None:
    """
    Copy source rows in chunk_id order, committing every batch.

    A single INSERT INTO ... SELECT over 3.5 TB would be one transaction: a
    crash at 90% rolls back 100%, and on this host that is the expected
    outcome, not the unlucky one. Batching by the source primary key means a
    crash costs at most one batch, and the resume point is a single bigint.
    """
    state = _read_phase(conn, "copy")
    if state and state.finished:
        print("  copy: already done — skipping")
        return

    cursor_id = state.last_chunk_id if state and state.last_chunk_id is not None else 0
    rows_done = state.rows_done if state else 0

    with conn.cursor(row_factory=dict_row) as cur:
        max_id = _one(cur.execute(
            f"SELECT COALESCE(max(chunk_id), 0) AS m FROM {SOURCE}"
        ))["m"]

    if cursor_id:
        print(f"  copy: resuming after chunk_id {cursor_id:,} ({rows_done:,} rows done)")
    print(f"  copy: target max chunk_id {max_id:,}, batch {batch_rows:,}")

    part_count = (state.partitions if state and state.partitions
                  else (_read_phase(conn, "create") or Phase("", None, 0, None, False)).partitions
                  or DEFAULT_PARTITIONS)
    _set_partition_autovacuum(conn, part_count, enabled=False)
    print(f"  copy: autovacuum disabled on {part_count} target partitions "
          f"(re-enabled by --constraints)")

    started = time.monotonic()
    rows_at_start = rows_done          # so the rate reflects THIS run, not resumed totals
    skipped_total = _corrupt_count(conn)
    if skipped_total:
        print(f"  copy: {skipped_total} previously-quarantined corrupt row(s)")

    while cursor_id < max_id:
        upper = cursor_id + batch_rows
        skipped: list[int] = []

        # The INSERT and the resume marker MUST commit together. They used to be
        # two autocommit statements, and killing the process in the gap left the
        # rows committed with the marker unmoved — so the next run re-copied that
        # batch. Measured 2026-08-12 after ~7 interruptions: 0.157% of rows
        # duplicated (~808k of 512M), every pair exactly one batch apart in
        # insertion order, which is that gap's precise signature. The unique
        # index in --constraints is what caught it; --dedupe repairs it.
        with conn.transaction():
            copied = _copy_range(conn, cursor_id, upper, skipped)
            _mark(conn, "copy", last_chunk_id=upper, rows_done=rows_done + copied)

        skipped_total += len(skipped)
        cursor_id = upper
        rows_done += copied

        pct = 100.0 * cursor_id / max_id if max_id else 100.0
        rate = (rows_done - rows_at_start) / max(time.monotonic() - started, 1e-6)
        note = f"  SKIPPED {len(skipped)} corrupt" if skipped else ""
        print(f"    {pct:5.1f}%  chunk_id<={cursor_id:>13,}  rows={rows_done:>13,}  "
              f"{rate:,.0f} rows/s{note}", flush=True)

    _mark(conn, "copy", last_chunk_id=cursor_id, rows_done=rows_done, finished=True)
    print(f"  copy: done — {rows_done:,} rows")


# --------------------------------------------------------------------------- #
# Phase 3 — indexes                                                            #
# --------------------------------------------------------------------------- #

def _partition_has_valid_index(conn: psycopg.Connection, part: str) -> bool:
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(
            """
            SELECT count(*) AS n
            FROM pg_index i
            JOIN pg_class idx ON idx.oid = i.indexrelid
            JOIN pg_class tbl ON tbl.oid = i.indrelid
            JOIN pg_am    am  ON am.oid  = idx.relam
            WHERE tbl.relname = %s AND am.amname = ANY(%s) AND i.indisvalid
            """,
            (part, list(VECTOR_INDEX_AMS)),
        ).fetchone()
    return bool(row and row["n"])


def _drop_invalid_indexes(conn: psycopg.Connection, part: str) -> int:
    """
    Remove INVALID leftovers from a build the machine interrupted.

    This is the specific debris a crash mid-CREATE INDEX leaves behind. An
    invalid index is never used by the planner but still costs write
    maintenance and disk, and its name blocks the retry.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            """
            SELECT idx.relname AS name
            FROM pg_index i
            JOIN pg_class idx ON idx.oid = i.indexrelid
            JOIN pg_class tbl ON tbl.oid = i.indrelid
            JOIN pg_am    am  ON am.oid  = idx.relam
            WHERE tbl.relname = %s AND am.amname = ANY(%s) AND NOT i.indisvalid
            """,
            (part, list(VECTOR_INDEX_AMS)),
        ).fetchall()
    for r in rows:
        print(f"    dropping invalid index left by an interrupted build: {r['name']}")
        # Identifier() rather than hand-rolled double quotes: the name comes
        # back from the catalog, so it is not necessarily quote-free.
        conn.execute(
            sql.SQL("DROP INDEX IF EXISTS {idx}").format(idx=sql.Identifier(r["name"]))
        )
    return len(rows)


def _index_stmt(part: str, tablespace: str | None) -> sql.Composed:
    """
    The per-partition halfvec ivfflat index statement.

    Extracted so the generated DDL can be asserted without a database. The
    halfvec cast in particular must match what WikiRetriever queries with, or
    the index is built correctly and then never used — pgvector only uses an
    expression index when the query casts identically.
    """
    stmt = sql.SQL(
        "CREATE INDEX {idx} ON {part} "
        "USING ivfflat ((embedding::halfvec({dims})) halfvec_cosine_ops) "
        "WITH (lists = {lists})"
    ).format(
        idx=sql.Identifier(f"{part}_hv_ivf"),
        part=sql.Identifier(part),
        dims=sql.Literal(EMBED_DIMS),
        lists=sql.Literal(IVFFLAT_LISTS),
    )
    if tablespace:
        stmt += sql.SQL(" TABLESPACE {ts}").format(ts=sql.Identifier(tablespace))
    return stmt


def phase_index(conn: psycopg.Connection, partitions: int, mwm: str,
                tablespace: str | None, parallel_workers: int) -> None:
    """
    Build one halfvec ivfflat index per partition, skipping partitions already done.

    Built per-partition and NON-concurrently on purpose. CREATE INDEX
    CONCURRENTLY is single-threaded and scans twice; it exists so writers are
    not blocked, and nothing is reading or writing this table until the swap.
    A plain CREATE INDEX can use max_parallel_maintenance_workers.
    """
    print(f"  index: {partitions} ivfflat partitions, maintenance_work_mem={mwm}, "
          f"tablespace={tablespace or 'default'}")

    built = skipped = 0

    for i in range(partitions):
        part = _partition_name(i)
        if _partition_has_valid_index(conn, part):
            skipped += 1
            continue

        _drop_invalid_indexes(conn, part)

        # Session-scoped: deliberately not a global postgresql.conf value, so
        # three autovacuum workers can never each claim 48 GB.
        conn.execute(
            sql.SQL("SET maintenance_work_mem = {mwm}").format(mwm=sql.Literal(mwm))
        )
        conn.execute(
            sql.SQL("SET max_parallel_maintenance_workers = {n}").format(
                n=sql.Literal(parallel_workers)
            )
        )

        started = time.monotonic()
        print(f"    [{i + 1}/{partitions}] building {part} ...", flush=True)
        conn.execute(_index_stmt(part, tablespace))
        built += 1
        print(f"    [{i + 1}/{partitions}] {part} done in "
              f"{time.monotonic() - started:,.0f}s", flush=True)

    _mark(conn, "index", partitions=partitions, finished=True)
    print(f"  index: {built} built, {skipped} already present")


def phase_dedupe(conn: psycopg.Connection, partitions: int) -> int:
    """
    Remove rows the copy inserted twice, keeping the earliest of each pair.

    Needed because of the atomicity bug described in phase_copy: a batch could
    commit without its resume marker, and the next run would re-copy it. The
    source table has UNIQUE (page_id, chunk_hash), so two target rows sharing
    that key can only be the same source row copied twice — verified 2026-08-12
    by comparing payloads across 300 sampled pairs: all identical, never more
    than 2 copies. Deleting the later of each pair is therefore lossless.

    Runs per partition so one unit of work is bounded and interruptible, and it
    is naturally idempotent: a second run finds nothing to delete. Do this
    BEFORE --constraints, which cannot build the unique index while duplicates
    exist.
    """
    print(f"  dedupe: scanning {partitions} partitions for duplicate "
          f"(page_id, chunk_hash)")
    total = 0
    for i in range(partitions):
        part = _partition_name(i)
        started = time.monotonic()
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    DELETE FROM {p}
                    WHERE chunk_id IN (
                        SELECT chunk_id FROM (
                            SELECT chunk_id,
                                   row_number() OVER (PARTITION BY page_id, chunk_hash
                                                      ORDER BY chunk_id) AS rn
                            FROM {p}
                        ) t WHERE t.rn > 1)
                """).format(p=sql.Identifier(part))
            )
            removed = cur.rowcount
        total += removed
        print(f"    [{i + 1}/{partitions}] {part}: removed {removed:,} "
              f"({time.monotonic() - started:,.0f}s)", flush=True)
    _mark(conn, "dedupe", rows_done=total, finished=True)
    print(f"  dedupe: removed {total:,} duplicate row(s)")
    return total


def _has_duplicates(conn: psycopg.Connection, partitions: int) -> bool:
    """Cheap existence probe — stops at the first duplicate rather than counting."""
    for i in range(partitions):
        with conn.cursor(row_factory=dict_row) as cur:
            row = cur.execute(
                sql.SQL("SELECT 1 AS x FROM {p} GROUP BY page_id, chunk_hash "
                        "HAVING count(*) > 1 LIMIT 1").format(
                    p=sql.Identifier(_partition_name(i)))
            ).fetchone()
        if row:
            return True
    return False


def _constraint_exists(conn: psycopg.Connection, name: str) -> bool:
    with conn.cursor(row_factory=dict_row) as cur:
        return bool(_one(cur.execute(
            "SELECT count(*) AS n FROM pg_constraint WHERE conname = %s", (name,)
        ))["n"])


def phase_constraints(conn: psycopg.Connection) -> None:
    """
    Restore every constraint the source table had, plus the supporting indexes.

    The partitioned table is created bare so the copy pays no per-row constraint
    or index maintenance. That is deliberate, but it means this phase is the
    ONLY thing standing between the migration and silent schema drift:
    rag_v1/db/wiki_schema.sql declares 7 NOT NULLs and 2 ON DELETE CASCADE
    foreign keys that the bare CREATE TABLE does not have. Losing them would not
    fail any test — it would just quietly stop cascading deletes and stop
    rejecting bad rows.

    NOT NULL is applied as ONE combined ALTER TABLE so PostgreSQL makes a single
    heap pass instead of seven. Checking NOT NULL reads only the null bitmap, so
    it does not detoast the 2.6 TB of embeddings.

    Foreign keys are added NOT VALID (instant, enforced for new rows) and then
    VALIDATEd separately. The data already satisfied these constraints in the
    source, so validation is a formality — but an unvalidated constraint is a
    lie in the catalog, and VALIDATE takes ShareUpdateExclusiveLock rather than
    blocking writes.
    """
    print("  constraints: unique (page_id, chunk_hash) + supporting indexes")
    conn.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS uq_{TARGET}_page_hash "
        f"ON {TARGET} (page_id, chunk_hash)"
    )
    # PK must include the partition key — see the module docstring.
    conn.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS {TARGET}_pkey "
        f"ON {TARGET} (chunk_id, page_id)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TARGET}_bundle_id ON {TARGET} (bundle_id)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TARGET}_page_id ON {TARGET} (page_id)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TARGET}_first_letter ON {TARGET} (first_letter)"
    )

    # --- NOT NULL: one statement, one heap pass -----------------------------
    print("  constraints: restoring NOT NULL on 7 columns (single table scan)")
    cols = ("bundle_id", "title", "first_letter", "chunk_index",
            "text", "chunk_hash", "embedding")
    conn.execute(
        sql.SQL("ALTER TABLE {t} {clauses}").format(
            t=sql.Identifier(TARGET),
            clauses=sql.SQL(", ").join(
                sql.SQL("ALTER COLUMN {c} SET NOT NULL").format(c=sql.Identifier(c))
                for c in cols
            ),
        )
    )

    # --- Foreign keys: match wiki_schema.sql, including ON DELETE CASCADE ---
    for name, col, ref, refcol in (
        (f"fk_{TARGET}_page",   "page_id",   "wiki_pages",   "page_id"),
        (f"fk_{TARGET}_bundle", "bundle_id", "wiki_bundles", "bundle_id"),
    ):
        if _constraint_exists(conn, name):
            print(f"  constraints: {name} already present")
            continue
        print(f"  constraints: adding {name} (NOT VALID, then validating)")
        conn.execute(
            sql.SQL(
                "ALTER TABLE {t} ADD CONSTRAINT {n} FOREIGN KEY ({c}) "
                "REFERENCES {r} ({rc}) ON DELETE CASCADE NOT VALID"
            ).format(
                t=sql.Identifier(TARGET), n=sql.Identifier(name),
                c=sql.Identifier(col), r=sql.Identifier(ref),
                rc=sql.Identifier(refcol),
            )
        )
        conn.execute(
            sql.SQL("ALTER TABLE {t} VALIDATE CONSTRAINT {n}").format(
                t=sql.Identifier(TARGET), n=sql.Identifier(name)
            )
        )

    # Undo the copy-phase autovacuum suppression and give the planner statistics.
    # Without the ANALYZE the new table ships with none at all — the same state
    # that made wiki_chunks read n_live_tup = 0 (CLAUDE.md §17) — and the main
    # app would plan every query against it blind.
    created = _read_phase(conn, "create")
    part_count = (created.partitions if created and created.partitions
                  else DEFAULT_PARTITIONS)
    print(f"  constraints: re-enabling autovacuum on {part_count} partitions")
    _set_partition_autovacuum(conn, part_count, enabled=True)

    print("  constraints: ANALYZE (sampled, not a full scan)")
    started = time.monotonic()
    conn.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(TARGET)))
    print(f"  constraints: ANALYZE done in {time.monotonic() - started:,.0f}s")

    _mark(conn, "constraints", finished=True)
    print("  constraints: done")


# --------------------------------------------------------------------------- #
# Phase 4 — verify + swap                                                      #
# --------------------------------------------------------------------------- #

def phase_verify(conn: psycopg.Connection) -> bool:
    """Compare row counts before allowing the swap. Slow, and worth it."""
    print("  verify: counting both tables (this scans; expect it to be slow)")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SET statement_timeout = 0")
        src = _one(cur.execute(f"SELECT count(*) AS n FROM {SOURCE}"))["n"]
        dst = _one(cur.execute(f"SELECT count(*) AS n FROM {TARGET}"))["n"]
    quarantined = _corrupt_count(conn)
    expected = src - quarantined
    print(f"  verify: source={src:,}  target={dst:,}  "
          f"quarantined={quarantined:,}  expected={expected:,}")
    if dst != expected:
        print(f"  verify: MISMATCH ({dst - expected:+,}) — refusing to swap. "
              f"Re-run --copy to catch up.")
        return False
    if quarantined:
        # Not a silent pass: the new table is knowingly short by these rows,
        # and that fact should be visible at the moment of the swap.
        print(f"  verify: counts reconcile, but {quarantined:,} row(s) were "
              f"unreadable in the source and are ABSENT from the new table.")
        print(f"          SELECT chunk_id FROM {CORRUPT_TABLE};")
    else:
        print("  verify: row counts match")
    return True


def _replicate_grants(conn: psycopg.Connection, src: str, dst: str,
                      partitions: int) -> None:
    """
    Copy `src`'s owner and privileges onto `dst` and all of its partitions.

    A rename moves the NAME, not the access control attached to it. The new
    table was created by this script under the owner DSN, so it carried that
    role's default ACL — which is NULL, meaning owner-only. The instant the
    swap completed, both applications lost the table:

        permission denied for table wiki_chunks

    That is not a small outage. Main READING every ingest-owned table at all
    times is a hard contract (CLAUDE.md §19.1), and this broke it for both
    projects simultaneously, immediately after a four-day index build. It was
    repaired by hand in the database on 2026-08-24 and the fix lived nowhere
    else until now, so re-running this — or applying the same migration to
    wiki_images, which is the obvious next candidate — would have reproduced
    it exactly.

    Partitions need it too: privileges are not inherited from the parent at
    query time, so a SELECT that touches a partition checks that partition's
    own ACL.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        row = _one(cur.execute(
            """
            SELECT pg_get_userbyid(c.relowner) AS owner,
                   c.relacl::text[]            AS acl
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = %s AND n.nspname = 'public'
            """,
            (src,),
        ))

    owner = row["owner"]
    targets = [dst] + [_partition_name(i) for i in range(partitions)]

    for name in targets:
        conn.execute(
            sql.SQL("ALTER TABLE {t} OWNER TO {o}").format(
                t=sql.Identifier(name), o=sql.Identifier(owner))
        )

    # relacl is NULL when the table has never been GRANTed on — the owner
    # simply has everything. Nothing to replicate in that case, and the
    # ALTER ... OWNER above has already done the meaningful part.
    for entry in row["acl"] or []:
        # Entries look like 'grantee=privs/grantor'; an empty grantee means
        # PUBLIC. Re-granting the full set is deliberate: this runs once, at
        # swap time, and under-granting is what caused the outage.
        grantee = entry.split("=", 1)[0]
        if not grantee:
            grantee = "PUBLIC"
        for name in targets:
            conn.execute(
                sql.SQL("GRANT ALL ON TABLE {t} TO {g}").format(
                    t=sql.Identifier(name),
                    g=sql.SQL("PUBLIC") if grantee == "PUBLIC"
                      else sql.Identifier(grantee),
                )
            )
    print(f"  swap: owner={owner}, grants replicated to "
          f"{len(targets)} relation(s)")


def phase_swap(conn: psycopg.Connection, partitions: int) -> None:
    """
    Rename the tables. The only destructive, hard-to-reverse step.

    The old table is RENAMED, never dropped: 3.5 TB of re-ingest is not
    something to gamble on a rename going as expected. Drop it by hand once
    wiki-RAG has been confirmed working against the new table.

    Ownership and grants are copied BEFORE the rename, while the source table
    still exists under its original name — see _replicate_grants for why that
    step is not optional.
    """
    print("  swap: replicating owner and grants from the source table")
    _replicate_grants(conn, SOURCE, TARGET, partitions)

    print("  swap: renaming (old table is kept as wiki_chunks_old)")
    with conn.transaction():
        conn.execute(f"ALTER TABLE {SOURCE} RENAME TO wiki_chunks_old")
        conn.execute(f"ALTER TABLE {TARGET} RENAME TO {SOURCE}")
    _mark(conn, "swap", finished=True)
    print("  swap: done")
    print()
    print("  wiki_chunks_old still holds the original 3.5 TB. Verify wiki-RAG")
    print("  works, then reclaim it:  DROP TABLE wiki_chunks_old;")


# --------------------------------------------------------------------------- #
# Status                                                                       #
# --------------------------------------------------------------------------- #

def phase_status(conn: psycopg.Connection, partitions: int) -> None:
    filled = backfill_corrupt_metadata(conn)
    if filled:
        print(f"  backfilled page_id/title for {filled} quarantined row(s)")

    quarantined = _corrupt_count(conn)
    if quarantined:
        print(f"  quarantined (unreadable in source): {quarantined}")
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT chunk_id, page_id, left(title, 44) AS title "
                f"FROM {CORRUPT_TABLE} ORDER BY chunk_id LIMIT 10"
            )
            for r in cur.fetchall():
                print(f"    {r['chunk_id']:>13,}  {r['page_id']}  {r['title']}")

    print("  phase state:")
    for name in ("create", "copy", "constraints", "index", "swap"):
        st = _read_phase(conn, name)
        if st is None:
            print(f"    {name:<12} not started")
        else:
            flag = "done" if st.finished else "IN PROGRESS"
            extra = f"  rows={st.rows_done:,}" if st.rows_done else ""
            cursor = f"  last_chunk_id={st.last_chunk_id:,}" if st.last_chunk_id else ""
            print(f"    {name:<12} {flag}{extra}{cursor}")

    with conn.cursor(row_factory=dict_row) as cur:
        exists = _one(cur.execute(
            "SELECT to_regclass(%s) IS NOT NULL AS e", (TARGET,)
        ))["e"]
    if not exists:
        return

    done = sum(1 for i in range(partitions)
               if _partition_has_valid_index(conn, _partition_name(i)))
    print(f"  partition indexes: {done}/{partitions} valid")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Resumable hash-partition migration for wiki_chunks.",
        epilog="Phases are idempotent; re-run after a crash to continue.",
    )
    p.add_argument("--status", action="store_true", help="report progress and exit")
    p.add_argument("--create", action="store_true", help="phase 1: partitioned table")
    p.add_argument("--copy", action="store_true", help="phase 2: batched copy")
    p.add_argument("--dedupe", action="store_true",
                   help="phase 2b: remove rows copied twice (run before --constraints)")
    p.add_argument("--constraints", action="store_true", help="phase 3a: btree indexes")
    p.add_argument("--index", action="store_true", help="phase 3b: per-partition ANN")
    p.add_argument("--verify", action="store_true", help="phase 4: compare row counts")
    p.add_argument("--swap", action="store_true", help="phase 5: rename (destructive)")
    p.add_argument("--i-understand-this-is-destructive", action="store_true",
                   dest="confirmed", help="required with --swap")
    p.add_argument("--partitions", type=int, default=DEFAULT_PARTITIONS)
    p.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    p.add_argument("--maintenance-work-mem", default=DEFAULT_MWM)
    p.add_argument("--index-tablespace", default=DEFAULT_INDEX_TABLESPACE,
                   help="tablespace for the ANN indexes; '' for the default")
    p.add_argument("--parallel-workers", type=int, default=4)
    args = p.parse_args(argv)

    if not any((args.status, args.create, args.copy, args.dedupe, args.constraints,
                args.index, args.verify, args.swap)):
        p.error("pick at least one phase (try --status)")

    if args.swap and not args.confirmed:
        print("error: --swap renames the live wiki_chunks table. Re-run with "
              "--i-understand-this-is-destructive once --verify has passed.",
              file=sys.stderr)
        return 2

    dsn = _owner_dsn()
    if dsn is None:
        print(
            "error: PG_OWNER_PASSWORD is not set.\n"
            "  wiki_chunks is owned by 'postgres' and the app role 'sage' has no\n"
            "  CREATE on schema public, so every phase of this migration needs the\n"
            "  owner connection. Set PG_OWNER / PG_OWNER_PASSWORD (same variables\n"
            "  wiki_ingest.py --manage-indexes uses).",
            file=sys.stderr,
        )
        return 2

    conn = _connect(dsn)
    conn.execute(_DDL_STATE)
    conn.execute(_DDL_CORRUPT)

    partitions = args.partitions
    created = _read_phase(conn, "create")
    if created and created.partitions:
        if created.partitions != partitions:
            print(f"  note: table was created with {created.partitions} partitions; "
                  f"using that instead of --partitions {partitions}")
        partitions = created.partitions

    try:
        if args.status:
            phase_status(conn, partitions)
            return 0
        if args.create:
            partitions = phase_create(conn, partitions)
        if args.copy:
            phase_copy(conn, args.batch_rows)
        if args.dedupe:
            phase_dedupe(conn, partitions)
        if args.constraints:
            # Fail early and actionably rather than partway through a long
            # unique-index build with a bare UniqueViolation.
            if _has_duplicates(conn, partitions):
                print("error: duplicate (page_id, chunk_hash) rows exist — the "
                      "unique index cannot be built.\n"
                      "  Run --dedupe first (safe and idempotent).",
                      file=sys.stderr)
                return 1
            phase_constraints(conn)
        if args.index:
            phase_index(conn, partitions, args.maintenance_work_mem,
                        args.index_tablespace or None, args.parallel_workers)
        if args.verify and not phase_verify(conn):
            return 1
        if args.swap:
            if not phase_verify(conn):
                return 1
            phase_swap(conn, partitions)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
