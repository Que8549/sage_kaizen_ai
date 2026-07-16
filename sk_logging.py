from __future__ import annotations

# NOTE: This file is intentionally duplicated in sage_kaizen_ai_ingest/.
# Each copy's project_root() resolves to its own directory so that log files
# land in the correct project's logs/ folder. Do not consolidate into a shared
# module — the per-project copy is the correct design.
#
# 2026-07-16: added PostgresLogHandler — a buffered, non-blocking handler that
# mirrors structured log records into the `log` schema (see
# log/db/log_schema.sql) alongside the existing rotating file. It is
# best-effort and never a hard dependency: if psycopg, the DSN, or the
# schema/tables are unavailable, logging silently degrades to file-only. DB
# writes are batched (up to ~2s delay) — the .log file remains the
# authoritative source for the last few lines before an abrupt process death;
# the DB copy is a queryable, correlatable aggregate, not a replacement
# disaster-forensics source for that specific case.

from pathlib import Path
import atexit
import logging
import logging.handlers
import os
import queue
import threading
import traceback
import uuid
from datetime import datetime, timezone


def project_root() -> Path:
    env = os.environ.get("SAGE_KAIZEN_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent


# ── Process-level run_id correlation ──────────────────────────────────────── #
# Every LogRecord in this process gets a run_id stamped via a global record
# factory — not a per-handler/per-logger Filter, which would have to be
# attached everywhere individually and would miss the root-logger capture
# some scripts use. Child processes inherit the SAME run_id when the parent
# passes SAGE_KAIZEN_RUN_ID in the subprocess env, giving real cross-process
# correlation for one logical run.
#
# This is deliberately separate from job-level run_id (news_runs table,
# per-job not per-process) and the voice project's turn-level ZMQ session_id —
# not unifying those, just adding a process-level axis for log rows.

RUN_ID: str = os.environ.get("SAGE_KAIZEN_RUN_ID") or str(uuid.uuid4())

_prev_record_factory = logging.getLogRecordFactory()


def _record_factory(*args, **kwargs):
    record = _prev_record_factory(*args, **kwargs)
    record.run_id = RUN_ID
    return record


logging.setLogRecordFactory(_record_factory)


# ── PostgresLogHandler ────────────────────────────────────────────────────── #

_SOURCE_PROJECT = "sage_kaizen_ai"

# Explicit allowlist (file_name -> log.<table>) — never derive the SQL
# identifier by string-stripping ".log": an unmapped file_name logs file-only,
# it never silently falls back to some default table.
_TABLE_MAP = {
    "sage_kaizen.log": "sage_kaizen",
    "sage_kaizen_ingest.log": "sage_kaizen_ingest",
    "sage_kaizen_voice.log": "sage_kaizen_voice",
    "news_agent.log": "news_agent",
    "media_ingest.log": "media_ingest",
    "wiki_ingest.log": "wiki_ingest",
}

_DESCRIPTION_CAP = 65536    # 64 KB — guards against one pathological line bloating a row
_EXCEPTION_CAP = 131072     # 128 KB
_QUEUE_MAXSIZE = 20000
_FLUSH_INTERVAL_S = 2.0
_FLUSH_BATCH_SIZE = 200

# Dedicated internal-diagnostics logger for "DB down"/"DB recovered" notices.
# Must NEVER get a PostgresLogHandler attached (would recurse into the outage
# it's reporting) and must never propagate to a handler-less root logger.
_internal_logger = logging.getLogger("sk_logging._internal")
_internal_logger.propagate = False
if not _internal_logger.handlers:
    _internal_logger.setLevel(logging.INFO)
    _internal_dir = project_root() / "logs"
    _internal_dir.mkdir(parents=True, exist_ok=True)
    _internal_handler = logging.handlers.RotatingFileHandler(
        filename=str(_internal_dir / "sk_logging_internal.log"),
        maxBytes=1 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    _internal_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _internal_logger.addHandler(_internal_handler)


class _BoundedQueueHandler(logging.handlers.QueueHandler):
    """
    QueueHandler over a bounded queue.

    Drops the newest record (and counts the drop) instead of blocking the
    calling thread or growing unbounded when the consumer thread falls behind
    a stalled DB. The stock QueueHandler's default unbounded queue would
    otherwise be an unbounded memory leak in the producer process during a
    sustained outage.
    """

    def __init__(self, q: "queue.Queue[logging.LogRecord | None]") -> None:
        super().__init__(q)
        self.dropped = 0

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1


class PostgresLogHandler:
    """
    Owns a bounded queue + background consumer thread that batches
    LogRecords into log.<table> via psycopg3.

    Not a logging.Handler itself — .queue_handler is the actual Handler
    attached to loggers (non-blocking enqueue only); this object owns the
    consumer thread, the DB connection, and the batched INSERT.

    Never raises: any failure (missing psycopg, bad DSN, DB down, missing
    schema) degrades to file-only logging via the sibling RotatingFileHandler,
    with at most one diagnostic notice per state transition.
    """

    def __init__(self, table: str, source_project: str) -> None:
        self.table = table
        self.source_project = source_project
        self._queue: "queue.Queue[logging.LogRecord | None]" = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self.queue_handler = _BoundedQueueHandler(self._queue)
        self._stop = threading.Event()
        self._conn = None
        self._db_down = False
        self._thread = threading.Thread(
            target=self._run, name=f"pg-log-{table}", daemon=True,
        )
        self._thread.start()
        atexit.register(self.close)

    # -- consumer thread ------------------------------------------------- #

    def _connect(self):
        try:
            import psycopg
        except Exception:
            return None
        try:
            from pg_settings import PgSettings
            dsn = PgSettings().pg_dsn
        except Exception:
            return None
        try:
            return psycopg.connect(dsn, autocommit=True, connect_timeout=5)
        except Exception:
            return None

    def _run(self) -> None:
        batch: list[logging.LogRecord] = []
        while True:
            try:
                record = self._queue.get(timeout=_FLUSH_INTERVAL_S)
            except queue.Empty:
                if batch:
                    self._flush(batch)
                    batch = []
                if self._stop.is_set():
                    break
                continue

            if record is None:  # shutdown sentinel
                if batch:
                    self._flush(batch)
                break

            batch.append(record)
            if len(batch) >= _FLUSH_BATCH_SIZE:
                self._flush(batch)
                batch = []

    def _flush(self, batch: list[logging.LogRecord]) -> None:
        if not batch:
            return
        if self._conn is None or self._conn.closed:
            self._conn = self._connect()
        if self._conn is None:
            self._note_down()
            return

        rows = []
        for record in batch:
            try:
                rows.append(self._row(record))
            except Exception:
                continue
        if not rows:
            return

        try:
            import psycopg.sql as sql
            with self._conn.cursor() as cur:
                cur.executemany(
                    sql.SQL(
                        "INSERT INTO log.{} "
                        "(log_date, log_type, log_name, description, exception, run_id, source_project) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
                    ).format(sql.Identifier(self.table)),
                    rows,
                )
            self._note_recovered()
        except Exception:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._note_down()

    def _row(self, record: logging.LogRecord) -> tuple:
        exc_text = None
        if record.exc_info:
            exc_text = "".join(traceback.format_exception(*record.exc_info))[:_EXCEPTION_CAP]
        return (
            datetime.fromtimestamp(record.created, tz=timezone.utc),
            record.levelname,
            record.name,
            record.getMessage()[:_DESCRIPTION_CAP],
            exc_text,
            getattr(record, "run_id", None),
            self.source_project,
        )

    def _note_down(self) -> None:
        if not self._db_down:
            self._db_down = True
            _internal_logger.warning(
                "PostgresLogHandler(%s): Postgres unreachable — buffering "
                "(bounded, dropped=%d so far) and continuing file-only until recovery.",
                self.table, self.queue_handler.dropped,
            )

    def _note_recovered(self) -> None:
        if self._db_down:
            self._db_down = False
            _internal_logger.info(
                "PostgresLogHandler(%s): Postgres connection recovered.", self.table,
            )

    def close(self) -> None:
        """Drain the queue, flush the final partial batch, and stop the thread.

        Registered via atexit — without this, a job's last ~2s of logs (or
        less than a full batch) would be silently lost on normal process exit,
        which is exactly the tail most worth having for a job that just
        finished or crashed.
        """
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass


_pg_handlers: dict[str, PostgresLogHandler] = {}
_pg_handlers_lock = threading.Lock()


def get_postgres_handler(file_name: str) -> logging.Handler | None:
    """
    Return the (possibly newly created) queue-backed Postgres handler for the
    log.<table> mapped from file_name, or None if file_name is not in
    _TABLE_MAP. One PostgresLogHandler (one thread, one connection) is shared
    by every logger targeting the same table.

    Public so scripts that build their own handlers instead of calling
    get_logger() can still attach it.
    """
    table = _TABLE_MAP.get(file_name)
    if table is None:
        return None
    with _pg_handlers_lock:
        h = _pg_handlers.get(table)
        if h is None:
            h = PostgresLogHandler(table, _SOURCE_PROJECT)
            _pg_handlers[table] = h
        return h.queue_handler


def get_logger(name: str, *, file_name: str = "sage_kaizen.log") -> logging.Logger:
    """
    Idempotent rotating-file + buffered-Postgres logger.
    Safe to call repeatedly across Streamlit reruns.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_dir = project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / file_name

    handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger.addHandler(handler)

    pg_handler = get_postgres_handler(file_name)
    if pg_handler is not None:
        logger.addHandler(pg_handler)

    return logger
