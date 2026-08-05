"""
tests/test_sk_logging.py

Unit tests for sk_logging.py — the structured-logging layer that mirrors
records into the `log` Postgres schema (CLAUDE.md §12).

The contract that matters most here is "never raises": a missing psycopg,
unset DSN, unreachable DB or absent schema must all degrade to file-only
logging rather than taking down the caller.

Also pins the 2026-08-04 fix to _BoundedQueueHandler.prepare(), which restored
the `exception` column (the stock QueueHandler strips exc_info before the
consumer ever sees it).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import sk_logging as skl


# ---------------------------------------------------------------------------
# project_root / RUN_ID
# ---------------------------------------------------------------------------

class TestProjectRoot:
    def test_defaults_to_module_directory(self, monkeypatch):
        monkeypatch.delenv("SAGE_KAIZEN_ROOT", raising=False)
        assert skl.project_root().name == "sage_kaizen_ai"

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAGE_KAIZEN_ROOT", str(tmp_path))
        assert skl.project_root() == tmp_path.resolve()


class TestRunId:
    def test_is_a_non_empty_string(self):
        assert isinstance(skl.RUN_ID, str) and skl.RUN_ID

    def test_stamped_onto_every_record_process_wide(self):
        """Set via logging.setLogRecordFactory, so third-party loggers get it too."""
        rec = logging.getLogRecordFactory()(
            "any.third.party", logging.INFO, __file__, 1, "msg", None, None
        )
        assert getattr(rec, "run_id", None) == skl.RUN_ID

    def test_is_stable_within_a_process(self):
        a = logging.getLogRecordFactory()("a", logging.INFO, __file__, 1, "m", None, None)
        b = logging.getLogRecordFactory()("b", logging.INFO, __file__, 1, "m", None, None)
        assert a.run_id == b.run_id  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _BoundedQueueHandler
# ---------------------------------------------------------------------------

def make_record(msg="hello", args=None, exc_info=None, level=logging.INFO):
    return logging.LogRecord("t", level, __file__, 1, msg, args, exc_info)


class TestBoundedQueueHandler:
    def test_enqueues_records(self):
        q: queue.Queue = queue.Queue(maxsize=10)
        h = skl._BoundedQueueHandler(q)
        h.emit(make_record())
        assert q.qsize() == 1

    def test_drops_newest_when_full_instead_of_blocking(self):
        q: queue.Queue = queue.Queue(maxsize=2)
        h = skl._BoundedQueueHandler(q)
        for _ in range(5):
            h.emit(make_record())
        assert q.qsize() == 2
        assert h.dropped == 3

    def test_drop_never_raises(self):
        q: queue.Queue = queue.Queue(maxsize=1)
        h = skl._BoundedQueueHandler(q)
        for _ in range(100):
            h.emit(make_record())   # must not raise

    def test_prepare_preserves_exc_info(self):
        """
        The 2026-08-04 fix. Stock QueueHandler.prepare() nulls exc_info, which
        made PostgresLogHandler._row()'s `exception` column permanently NULL.
        """
        q: queue.Queue = queue.Queue(maxsize=10)
        h = skl._BoundedQueueHandler(q)
        try:
            raise ValueError("boom")
        except ValueError:
            h.emit(make_record("failed", exc_info=True and __import__("sys").exc_info()))
        rec = q.get_nowait()
        assert rec.exc_info is not None

    def test_prepare_preserves_args_for_lazy_formatting(self):
        q: queue.Queue = queue.Queue(maxsize=10)
        h = skl._BoundedQueueHandler(q)
        h.emit(make_record("value=%s", args=("x",)))
        rec = q.get_nowait()
        assert rec.getMessage() == "value=x"

    def test_prepare_preserves_run_id(self):
        q: queue.Queue = queue.Queue(maxsize=10)
        h = skl._BoundedQueueHandler(q)
        rec_in = logging.getLogRecordFactory()("t", logging.INFO, __file__, 1, "m", None, None)
        h.emit(rec_in)
        assert q.get_nowait().run_id == skl.RUN_ID

    def test_prepare_copies_so_the_file_handler_is_unaffected(self):
        q: queue.Queue = queue.Queue(maxsize=10)
        h = skl._BoundedQueueHandler(q)
        original = make_record()
        h.emit(original)
        assert q.get_nowait() is not original


# ---------------------------------------------------------------------------
# PostgresLogHandler._row
# ---------------------------------------------------------------------------

@pytest.fixture
def handler():
    """A PostgresLogHandler with its consumer thread and DB connection stubbed."""
    with (
        patch.object(skl.threading, "Thread") as T,
        patch.object(skl.atexit, "register"),
    ):
        T.return_value = MagicMock()
        h = skl.PostgresLogHandler("sage_kaizen", "sage_kaizen_ai")
    return h


class TestPostgresLogHandlerRow:
    def test_column_order(self, handler):
        row = handler._row(make_record("msg"))
        assert len(row) == 7
        assert row[1] == "INFO"          # log_type
        assert row[2] == "t"             # log_name
        assert row[3] == "msg"           # description
        assert row[6] == "sage_kaizen_ai"  # source_project

    def test_timestamp_is_timezone_aware_utc(self, handler):
        ts = handler._row(make_record())[0]
        assert ts.tzinfo is not None
        assert ts.utcoffset().total_seconds() == 0  # type: ignore[union-attr]

    def test_exception_is_null_without_exc_info(self, handler):
        assert handler._row(make_record())[4] is None

    def test_exception_is_populated_with_exc_info(self, handler):
        import sys
        try:
            raise KeyError("missing")
        except KeyError:
            rec = make_record("failed", exc_info=sys.exc_info())
        exc_text = handler._row(rec)[4]
        assert exc_text is not None
        assert "KeyError" in exc_text

    def test_description_excludes_the_traceback(self, handler):
        """The traceback belongs in `exception`, not folded into `description`."""
        import sys
        try:
            raise KeyError("missing")
        except KeyError:
            rec = make_record("failed doing thing", exc_info=sys.exc_info())
        row = handler._row(rec)
        assert row[3] == "failed doing thing"
        assert "Traceback" not in row[3]

    def test_description_is_capped(self, handler):
        rec = make_record("x" * (skl._DESCRIPTION_CAP + 5_000))
        assert len(handler._row(rec)[3]) == skl._DESCRIPTION_CAP

    def test_message_args_are_formatted(self, handler):
        assert handler._row(make_record("a=%s b=%d", args=("x", 7)))[3] == "a=x b=7"

    def test_levels_map_to_log_type(self, handler):
        for level, name in [
            (logging.DEBUG, "DEBUG"), (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"), (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]:
            assert handler._row(make_record(level=level))[1] == name


# ---------------------------------------------------------------------------
# PostgresLogHandler — degradation contract
# ---------------------------------------------------------------------------

class TestPostgresLogHandlerDegradation:
    def test_connect_returns_none_when_psycopg_missing(self, handler):
        with patch.dict("sys.modules", {"psycopg": None}):
            assert handler._connect() is None

    def test_connect_returns_none_when_dsn_unavailable(self, handler):
        with patch.dict("sys.modules", {"pg_settings": MagicMock(
            PgSettings=MagicMock(side_effect=RuntimeError("no env"))
        )}):
            assert handler._connect() is None

    def test_connect_returns_none_when_db_unreachable(self, handler):
        fake = MagicMock()
        fake.connect.side_effect = OSError("connection refused")
        with patch.dict("sys.modules", {"psycopg": fake}):
            assert handler._connect() is None

    def test_flush_with_empty_batch_is_a_noop(self, handler):
        handler._flush([])   # must not raise

    def test_flush_notes_down_when_connect_fails(self, handler):
        with patch.object(handler, "_connect", return_value=None):
            handler._flush([make_record()])
        assert handler._db_down is True

    def test_down_notice_is_emitted_once_per_transition(self, handler):
        with (
            patch.object(handler, "_connect", return_value=None),
            patch.object(skl._internal_logger, "warning") as warn,
        ):
            for _ in range(5):
                handler._flush([make_record()])
        assert warn.call_count == 1

    def test_recovery_notice_after_down(self, handler):
        conn = MagicMock()
        conn.closed = False
        handler._db_down = True
        handler._conn = conn
        with patch.object(skl._internal_logger, "info") as info:
            handler._flush([make_record()])
        assert handler._db_down is False
        assert info.call_count == 1

    def test_insert_failure_drops_the_connection_for_retry(self, handler):
        conn = MagicMock()
        conn.closed = False
        conn.cursor.side_effect = RuntimeError("relation log.sage_kaizen does not exist")
        handler._conn = conn
        handler._flush([make_record()])
        assert handler._conn is None
        assert handler._db_down is True

    def test_unrowable_record_is_skipped_not_fatal(self, handler):
        conn = MagicMock()
        conn.closed = False
        handler._conn = conn
        bad = make_record("%d", args=("not-an-int",))   # getMessage() raises
        handler._flush([bad])   # must not raise

    def test_close_is_idempotent(self, handler):
        handler._thread = MagicMock()
        handler.close()
        handler.close()   # must not raise

    def test_close_sets_stop_and_joins(self, handler):
        handler._thread = MagicMock()
        handler.close()
        assert handler._stop.is_set()
        handler._thread.join.assert_called_once()

    def test_close_survives_a_full_queue(self, handler):
        handler._thread = MagicMock()
        handler._queue = queue.Queue(maxsize=1)
        handler._queue.put_nowait(make_record())
        handler.close()   # sentinel can't fit; must not raise


# ---------------------------------------------------------------------------
# get_postgres_handler / get_logger
# ---------------------------------------------------------------------------

class TestGetPostgresHandler:
    def test_unmapped_file_name_returns_none(self):
        assert skl.get_postgres_handler("something_else.log") is None

    @pytest.mark.parametrize("file_name", list(skl._TABLE_MAP))
    def test_mapped_file_names_return_a_handler(self, file_name):
        assert skl.get_postgres_handler(file_name) is not None

    def test_one_handler_shared_per_table(self):
        a = skl.get_postgres_handler("sage_kaizen.log")
        b = skl.get_postgres_handler("sage_kaizen.log")
        assert a is b

    def test_table_map_has_the_six_documented_sources(self):
        assert set(skl._TABLE_MAP.values()) == {
            "sage_kaizen", "sage_kaizen_ingest", "sage_kaizen_voice",
            "news_agent", "media_ingest", "wiki_ingest",
        }

    def test_table_names_are_never_derived_by_stripping(self):
        """Explicit allowlist — an unmapped name must not fall back to a default."""
        assert skl.get_postgres_handler("arbitrary_name.log") is None


class TestGetLogger:
    def test_returns_a_logger_with_handlers(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAGE_KAIZEN_ROOT", str(tmp_path))
        lg = skl.get_logger("test.unmapped.a", file_name="unmapped_a.log")
        assert lg.handlers

    def test_is_idempotent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAGE_KAIZEN_ROOT", str(tmp_path))
        a = skl.get_logger("test.idem", file_name="idem.log")
        n = len(a.handlers)
        b = skl.get_logger("test.idem", file_name="idem.log")
        assert a is b and len(b.handlers) == n

    def test_does_not_propagate_to_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAGE_KAIZEN_ROOT", str(tmp_path))
        assert skl.get_logger("test.noprop", file_name="noprop.log").propagate is False

    def test_unmapped_gets_only_a_file_handler(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAGE_KAIZEN_ROOT", str(tmp_path))
        lg = skl.get_logger("test.unmapped.b", file_name="unmapped_b.log")
        assert len(lg.handlers) == 1
        assert isinstance(lg.handlers[0], logging.handlers.RotatingFileHandler)

    def test_unmapped_uses_the_standard_file_size(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAGE_KAIZEN_ROOT", str(tmp_path))
        lg = skl.get_logger("test.unmapped.c", file_name="unmapped_c.log")
        assert lg.handlers[0].maxBytes == 5 * 1024 * 1024  # type: ignore[attr-defined]

    def test_mapped_gets_file_plus_db_handler(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAGE_KAIZEN_ROOT", str(tmp_path))
        lg = skl.get_logger("test.mapped.a", file_name="sage_kaizen.log")
        assert len(lg.handlers) == 2

    def test_mapped_file_is_the_small_crash_safety_size(self, monkeypatch, tmp_path):
        """Postgres is the archive; this file only bridges a crash window."""
        monkeypatch.setenv("SAGE_KAIZEN_ROOT", str(tmp_path))
        lg = skl.get_logger("test.mapped.b", file_name="sage_kaizen.log")
        fh = [h for h in lg.handlers if isinstance(h, logging.handlers.RotatingFileHandler)][0]
        assert fh.maxBytes == skl.FALLBACK_MAX_BYTES
        assert fh.backupCount == skl.FALLBACK_BACKUP_CNT

    def test_crash_safety_file_is_smaller_than_the_standard_one(self):
        assert skl.FALLBACK_MAX_BYTES < 5 * 1024 * 1024

    def test_writes_actually_reach_the_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAGE_KAIZEN_ROOT", str(tmp_path))
        lg = skl.get_logger("test.write", file_name="write_probe.log")
        lg.info("a distinctive marker line")
        for h in lg.handlers:
            h.flush()
        text = (tmp_path / "logs" / "write_probe.log").read_text(encoding="utf-8")
        assert "a distinctive marker line" in text

    def test_internal_logger_never_gets_a_db_handler(self):
        """It reports on the DB path's health; it must not recurse into it."""
        assert all(
            not isinstance(h, skl._BoundedQueueHandler)
            for h in skl._internal_logger.handlers
        )

    def test_internal_logger_does_not_propagate(self):
        assert skl._internal_logger.propagate is False
