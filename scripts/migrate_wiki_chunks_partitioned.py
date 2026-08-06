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

THE TWO NUMBERS THAT SET THE PARTITION COUNT
--------------------------------------------
An HNSW build is fast while the graph fits in maintenance_work_mem and
collapses to random disk I/O when it does not ("Indexes build significantly
faster when the graph fits into maintenance_work_mem" — pgvector README). So
partitions are sized so ONE partition's graph fits in RAM:

    halfvec(1024) element  ~= 2056 B vector + ~200 B neighbour lists  ~= 2.3 KB
    508M rows / 32 partitions                                          ~= 15.9M rows
    15.9M x 2.3 KB                                                     ~= 36 GB

which fits in a 48 GB maintenance_work_mem on a 190 GB host, with one build
running at a time. Full-precision vector(1024) would be ~4.3 KB/element (~68 GB
per partition) and would not fit — which is the other reason for halfvec.

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
from psycopg.rows import dict_row  # noqa: E402

from pg_settings import PgSettings  # noqa: E402

SOURCE = "wiki_chunks"
TARGET = "wiki_chunks_part"
STATE_TABLE = "wiki_chunks_migration"

DEFAULT_PARTITIONS = 32
DEFAULT_BATCH_ROWS = 200_000
DEFAULT_MWM = "48GB"
DEFAULT_INDEX_TABLESPACE = "sage_nvme"
EMBED_DIMS = 1024

# HNSW parameters copied from the index ingest already knows how to drop and
# rebuild, so the two projects stay describing the same thing.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100


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
    return urllib.parse.urlunparse(
        parsed._replace(
            netloc=f"{urllib.parse.quote(cfg.pg_owner)}:"
                   f"{urllib.parse.quote(cfg.pg_owner_password)}"
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
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TARGET} (
            chunk_id     bigint GENERATED ALWAYS AS IDENTITY,
            page_id      uuid   NOT NULL,
            bundle_id    uuid,
            title        text,
            first_letter character(1),
            section_path text[],
            chunk_index  integer,
            text         text,
            chunk_hash   text,
            embedding    vector({EMBED_DIMS})
        ) PARTITION BY HASH (page_id)
    """)
    for i in range(partitions):
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_partition_name(i)} "
            f"PARTITION OF {TARGET} "
            f"FOR VALUES WITH (MODULUS {partitions}, REMAINDER {i})"
        )
    _mark(conn, "create", partitions=partitions, finished=True)
    print(f"  create: done")
    return partitions


# --------------------------------------------------------------------------- #
# Phase 2 — copy, in committed batches                                         #
# --------------------------------------------------------------------------- #

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
        max_id = cur.execute(
            f"SELECT COALESCE(max(chunk_id), 0) AS m FROM {SOURCE}"
        ).fetchone()["m"]

    if cursor_id:
        print(f"  copy: resuming after chunk_id {cursor_id:,} ({rows_done:,} rows done)")
    print(f"  copy: target max chunk_id {max_id:,}, batch {batch_rows:,}")

    started = time.monotonic()
    while cursor_id < max_id:
        upper = cursor_id + batch_rows
        with conn.cursor() as cur:
            # Column list is explicit and omits chunk_id: the target's identity
            # column generates its own. Nothing downstream depends on source
            # chunk_id values surviving (the retriever selects it, never joins
            # on it across the migration).
            cur.execute(
                f"""
                INSERT INTO {TARGET}
                    (page_id, bundle_id, title, first_letter, section_path,
                     chunk_index, text, chunk_hash, embedding)
                SELECT page_id, bundle_id, title, first_letter, section_path,
                       chunk_index, text, chunk_hash, embedding
                FROM {SOURCE}
                WHERE chunk_id > %s AND chunk_id <= %s
                """,
                (cursor_id, upper),
            )
            copied = cur.rowcount

        cursor_id = upper
        rows_done += copied
        _mark(conn, "copy", last_chunk_id=cursor_id, rows_done=rows_done)

        pct = 100.0 * cursor_id / max_id if max_id else 100.0
        rate = rows_done / max(time.monotonic() - started, 1e-6)
        print(f"    {pct:5.1f}%  chunk_id<={cursor_id:>13,}  rows={rows_done:>13,}  "
              f"{rate:,.0f} rows/s", flush=True)

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
            WHERE tbl.relname = %s AND am.amname = 'hnsw' AND i.indisvalid
            """,
            (part,),
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
            WHERE tbl.relname = %s AND am.amname = 'hnsw' AND NOT i.indisvalid
            """,
            (part,),
        ).fetchall()
    for r in rows:
        print(f"    dropping invalid index left by an interrupted build: {r['name']}")
        conn.execute(f'DROP INDEX IF EXISTS "{r["name"]}"')
    return len(rows)


def phase_index(conn: psycopg.Connection, partitions: int, mwm: str,
                tablespace: str | None, parallel_workers: int) -> None:
    """
    Build one halfvec HNSW index per partition, skipping partitions already done.

    Built per-partition and NON-concurrently on purpose. CREATE INDEX
    CONCURRENTLY is single-threaded and scans twice; it exists so writers are
    not blocked, and nothing is reading or writing this table until the swap.
    A plain CREATE INDEX can use max_parallel_maintenance_workers.
    """
    print(f"  index: {partitions} partitions, maintenance_work_mem={mwm}, "
          f"tablespace={tablespace or 'default'}")

    ts_clause = f" TABLESPACE {tablespace}" if tablespace else ""
    built = skipped = 0

    for i in range(partitions):
        part = _partition_name(i)
        if _partition_has_valid_index(conn, part):
            skipped += 1
            continue

        _drop_invalid_indexes(conn, part)

        # Session-scoped: deliberately not a global postgresql.conf value, so
        # three autovacuum workers can never each claim 48 GB.
        conn.execute(f"SET maintenance_work_mem = '{mwm}'")
        conn.execute(f"SET max_parallel_maintenance_workers = {parallel_workers}")

        started = time.monotonic()
        print(f"    [{i + 1}/{partitions}] building {part} ...", flush=True)
        conn.execute(
            f"""
            CREATE INDEX {part}_hv_hnsw ON {part}
            USING hnsw ((embedding::halfvec({EMBED_DIMS})) halfvec_cosine_ops)
            WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})
            {ts_clause}
            """
        )
        built += 1
        print(f"    [{i + 1}/{partitions}] {part} done in "
              f"{time.monotonic() - started:,.0f}s", flush=True)

    _mark(conn, "index", partitions=partitions, finished=True)
    print(f"  index: {built} built, {skipped} already present")


def phase_constraints(conn: psycopg.Connection) -> None:
    """
    Add the btree constraints after the copy.

    uq_wiki_chunks_page_hash is what ingest's ON CONFLICT (page_id, chunk_hash)
    depends on, so it must exist before the swap or every ingest insert fails.
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
        src = cur.execute(f"SELECT count(*) AS n FROM {SOURCE}").fetchone()["n"]
        dst = cur.execute(f"SELECT count(*) AS n FROM {TARGET}").fetchone()["n"]
    print(f"  verify: source={src:,}  target={dst:,}  delta={dst - src:+,}")
    if src != dst:
        print("  verify: MISMATCH — refusing to swap. Re-run --copy to catch up.")
        return False
    print("  verify: row counts match")
    return True


def phase_swap(conn: psycopg.Connection) -> None:
    """
    Rename the tables. The only destructive, hard-to-reverse step.

    The old table is RENAMED, never dropped: 3.5 TB of re-ingest is not
    something to gamble on a rename going as expected. Drop it by hand once
    wiki-RAG has been confirmed working against the new table.
    """
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
        exists = cur.execute(
            "SELECT to_regclass(%s) IS NOT NULL AS e", (TARGET,)
        ).fetchone()["e"]
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

    if not any((args.status, args.create, args.copy, args.constraints,
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
        if args.constraints:
            phase_constraints(conn)
        if args.index:
            phase_index(conn, partitions, args.maintenance_work_mem,
                        args.index_tablespace or None, args.parallel_workers)
        if args.verify and not phase_verify(conn):
            return 1
        if args.swap:
            if not phase_verify(conn):
                return 1
            phase_swap(conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
