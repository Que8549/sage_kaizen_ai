"""
tests/test_migration_phases.py

Phase-logic tests for scripts/migrate_wiki_chunks_partitioned.py.

tests/test_migration_sql.py asserts the DDL this script *generates*.  This
file asserts what it *does*: resumability, idempotency, and the two behaviours
that exist purely because the host reboots under load — bisecting around
corrupt TOAST pages, and skipping partitions whose index already built.

Why this file exists at all: `scripts/*` was omitted from coverage until
2026-08-24, so the 993-line script that issues destructive DDL against a
1.3 TB table reported as untested-and-invisible rather than untested.  Omitting
a path does not flag it — it removes it from the denominator entirely.

No database.  psycopg connections are doubles; the assertions are about
control flow (what gets skipped, what gets retried, what refuses to proceed),
which is exactly the part that only runs after a multi-day copy and therefore
cannot be discovered by running it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "_migration_phases_under_test",
        _ROOT / "scripts" / "migrate_wiki_chunks_partitioned.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mig = _load()


class _Cur:
    """Cursor double: scripted fetch results, recorded statements."""

    def __init__(self, one=None, many=None, rowcount=0):
        self._one, self._many, self.rowcount = one, many or [], rowcount
        self.executed: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, q, params=None):
        self.executed.append((q if isinstance(q, str) else q.as_string(None), params))
        return self

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many


class _Conn:
    def __init__(self, cur: _Cur | None = None):
        self.cur = cur or _Cur()
        self.executed: list[str] = []
        self.transactions = 0

    def cursor(self, **kw):
        return self.cur

    def execute(self, q, params=None):
        self.executed.append(q if isinstance(q, str) else q.as_string(None))
        return self.cur

    def transaction(self):
        self.transactions += 1
        return MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Index phase — skip/retry logic
# ---------------------------------------------------------------------------

class TestPartitionHasValidIndex:
    def test_true_when_a_valid_vector_index_exists(self):
        assert mig._partition_has_valid_index(_Conn(_Cur(one={"n": 1})), "p") is True

    def test_false_when_none_exist(self):
        assert mig._partition_has_valid_index(_Conn(_Cur(one={"n": 0})), "p") is False

    def test_query_requires_indisvalid(self):
        """
        An INVALID index left by an interrupted build still appears in
        pg_index.  Counting it as "done" would skip the partition forever and
        leave a 40 GB index the planner never uses.
        """
        conn = _Conn(_Cur(one={"n": 0}))
        mig._partition_has_valid_index(conn, "p")
        assert "indisvalid" in conn.cur.executed[0][0]

    def test_matches_both_access_methods(self):
        conn = _Conn(_Cur(one={"n": 0}))
        mig._partition_has_valid_index(conn, "p")
        assert conn.cur.executed[0][1][1] == list(mig.VECTOR_INDEX_AMS)


class TestDropInvalidIndexes:
    def test_drops_each_invalid_index_and_reports_the_count(self, capsys):
        conn = _Conn(_Cur(many=[{"name": "x_hv_ivf"}, {"name": "y_hv_ivf"}]))
        assert mig._drop_invalid_indexes(conn, "p") == 2
        drops = [s for s in conn.executed if "DROP INDEX" in s]
        assert len(drops) == 2
        assert '"x_hv_ivf"' in drops[0]

    def test_no_statements_when_there_is_nothing_invalid(self):
        conn = _Conn(_Cur(many=[]))
        assert mig._drop_invalid_indexes(conn, "p") == 0
        assert conn.executed == []

    def test_names_from_the_catalog_are_quoted_as_identifiers(self):
        # The name is read back from pg_class, so it is not guaranteed to be
        # a bare lowercase token.
        conn = _Conn(_Cur(many=[{"name": 'weird "name'}]))
        mig._drop_invalid_indexes(conn, "p")
        assert '"weird ""name"' in conn.executed[0]


class TestPhaseIndex:
    def _run(self, monkeypatch, has_index):
        calls = {"built": [], "dropped": []}
        monkeypatch.setattr(mig, "_partition_has_valid_index",
                            lambda c, p: has_index(p))
        monkeypatch.setattr(mig, "_drop_invalid_indexes",
                            lambda c, p: calls["dropped"].append(p) or 0)
        monkeypatch.setattr(mig, "_mark", lambda *a, **k: None)
        conn = _Conn()
        mig.phase_index(conn, 4, "48GB", "sage_nvme", 4)
        calls["built"] = [s for s in conn.executed if "CREATE INDEX" in s]
        return calls, conn

    def test_skips_partitions_that_are_already_built(self, monkeypatch, capsys):
        """
        Resumability is the whole design: the real build took 4.27 days across
        32 partitions on a host that reboots under load.  Rebuilding a
        finished partition would restart that clock.
        """
        calls, _ = self._run(monkeypatch, lambda p: True)
        assert calls["built"] == []
        assert "0 built, 4 already present" in capsys.readouterr().out

    def test_builds_only_the_missing_partitions(self, monkeypatch, capsys):
        done = {mig._partition_name(0), mig._partition_name(2)}
        calls, _ = self._run(monkeypatch, lambda p: p in done)
        assert len(calls["built"]) == 2
        assert "2 built, 2 already present" in capsys.readouterr().out

    def test_clears_invalid_debris_before_rebuilding(self, monkeypatch):
        # A crash mid-CREATE INDEX leaves an invalid index whose NAME blocks
        # the retry; dropping it must happen first or the rebuild errors.
        calls, _ = self._run(monkeypatch, lambda p: False)
        assert len(calls["dropped"]) == 4

    def test_memory_settings_are_session_scoped_per_partition(self, monkeypatch):
        """
        maintenance_work_mem is deliberately NOT a postgresql.conf value: three
        autovacuum workers each claiming 48 GB on a 190 GB host is a swap
        storm.
        """
        _, conn = self._run(monkeypatch, lambda p: False)
        assert any("SET maintenance_work_mem = '48GB'" in s for s in conn.executed)
        assert any("SET max_parallel_maintenance_workers = 4" in s
                   for s in conn.executed)


# ---------------------------------------------------------------------------
# Copy phase — corruption bisection
# ---------------------------------------------------------------------------

class TestCopyRangeBisection:
    def test_clean_range_copies_in_one_statement(self):
        conn = _Conn(_Cur(rowcount=500))
        skipped: list[int] = []
        assert mig._copy_range(conn, 0, 1000, skipped) == 500
        assert skipped == []

    def test_corrupt_row_is_quarantined_not_fatal(self, monkeypatch, capsys):
        """
        One damaged 8 KB TOAST page would otherwise abort a 100,000-row batch
        and stall the migration permanently — the source has four such rows.
        """
        import psycopg
        monkeypatch.setattr(mig, "_quarantine", lambda *a: None)
        conn = _Conn()
        conn.cur.execute = MagicMock(
            side_effect=psycopg.errors.DataCorrupted("bad page"))
        skipped: list[int] = []
        assert mig._copy_range(conn, 0, 1, skipped) == 0
        assert skipped == [1]
        assert "quarantined corrupt chunk_id" in capsys.readouterr().out

    def test_bisects_to_isolate_the_bad_row(self, monkeypatch):
        """
        Only the genuinely unreadable row may be lost.  A whole-batch skip on
        first error would discard up to 100,000 good rows alongside it.
        """
        import psycopg
        bad = 7
        monkeypatch.setattr(mig, "_quarantine", lambda *a: None)

        class _BisectConn(_Conn):
            def transaction(self):
                return MagicMock(__enter__=MagicMock(),
                                 __exit__=MagicMock(return_value=False))

        conn = _BisectConn()
        state = {"lo": 0, "hi": 0}

        def _execute(q, params=None):
            state["lo"], state["hi"] = params
            if state["lo"] < bad <= state["hi"]:
                raise psycopg.errors.DataCorrupted("bad page")
            conn.cur.rowcount = state["hi"] - state["lo"]
            return conn.cur

        conn.cur.execute = _execute
        skipped: list[int] = []
        copied = mig._copy_range(conn, 0, 8, skipped)

        assert skipped == [bad]          # exactly one row lost
        assert copied == 7               # the other seven survived

    def test_single_row_insert_is_atomic_so_bisection_cannot_double_insert(self):
        # Each attempt runs inside its own SAVEPOINT; a failed attempt copies
        # nothing, which is what makes subdividing safe to retry.
        conn = _Conn(_Cur(rowcount=1))
        mig._copy_range(conn, 0, 1, [])
        assert conn.transactions == 1


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------

class TestDedupe:
    def test_totals_removals_across_partitions(self, monkeypatch, capsys):
        monkeypatch.setattr(mig, "_mark", lambda *a, **k: None)
        conn = _Conn(_Cur(rowcount=3))
        assert mig.phase_dedupe(conn, 4) == 12
        assert "removed 12 duplicate row(s)" in capsys.readouterr().out

    def test_is_idempotent_when_there_is_nothing_to_remove(self, monkeypatch):
        monkeypatch.setattr(mig, "_mark", lambda *a, **k: None)
        assert mig.phase_dedupe(_Conn(_Cur(rowcount=0)), 4) == 0

    def test_keeps_the_earliest_of_each_duplicate_pair(self, monkeypatch):
        """
        The pair is the same source row copied twice (the source has UNIQUE
        (page_id, chunk_hash)), so either is fine — but the choice must be
        deterministic, or a re-run could delete the other one instead.
        """
        monkeypatch.setattr(mig, "_mark", lambda *a, **k: None)
        conn = _Conn(_Cur(rowcount=0))
        mig.phase_dedupe(conn, 1)
        sql_text = conn.cur.executed[0][0]
        assert "row_number() OVER" in sql_text
        assert "ORDER BY chunk_id" in sql_text
        assert "rn > 1" in sql_text


class TestHasDuplicates:
    def test_stops_at_the_first_partition_with_a_duplicate(self):
        conn = _Conn(_Cur(one={"x": 1}))
        assert mig._has_duplicates(conn, 32) is True
        # Short-circuits rather than scanning all 32.
        assert len(conn.cur.executed) == 1

    def test_false_when_every_partition_is_clean(self):
        conn = _Conn(_Cur(one=None))
        assert mig._has_duplicates(conn, 8) is False
        assert len(conn.cur.executed) == 8


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

class TestPhaseVerify:
    def _conn(self, src, dst):
        cur = _Cur()
        results = iter([{"n": src}, {"n": dst}])
        cur.execute = lambda q, params=None: (
            setattr(cur, "_one", next(results, {"n": 0})) or cur
        ) if "count(*)" in str(q) else cur
        return _Conn(cur)

    def test_passes_when_counts_reconcile_exactly(self, monkeypatch, capsys):
        monkeypatch.setattr(mig, "_corrupt_count", lambda c: 0)
        assert mig.phase_verify(self._conn(100, 100)) is True
        assert "row counts match" in capsys.readouterr().out

    def test_quarantined_rows_are_subtracted_from_the_expectation(self, monkeypatch, capsys):
        # target = source - quarantined is a PASS: those rows are knowingly
        # absent because they are unreadable in the source.
        monkeypatch.setattr(mig, "_corrupt_count", lambda c: 4)
        assert mig.phase_verify(self._conn(100, 96)) is True
        out = capsys.readouterr().out
        assert "counts reconcile" in out
        assert "ABSENT from the new table" in out

    def test_refuses_when_the_target_is_short(self, monkeypatch, capsys):
        monkeypatch.setattr(mig, "_corrupt_count", lambda c: 0)
        assert mig.phase_verify(self._conn(100, 97)) is False
        assert "MISMATCH" in capsys.readouterr().out

    def test_refuses_when_the_target_has_too_many(self, monkeypatch):
        # Extra rows mean duplicates from the copy-atomicity bug; swapping
        # would make them permanent.
        monkeypatch.setattr(mig, "_corrupt_count", lambda c: 0)
        assert mig.phase_verify(self._conn(100, 103)) is False

    def test_disables_the_statement_timeout_before_counting(self, monkeypatch):
        """
        Counting 512M rows takes far longer than any session default; a
        timeout here would read as a verification failure and block the swap.
        """
        monkeypatch.setattr(mig, "_corrupt_count", lambda c: 0)
        conn = self._conn(1, 1)
        seen = []
        inner = conn.cur.execute
        conn.cur.execute = lambda q, params=None: (seen.append(str(q)), inner(q, params))[1]
        mig.phase_verify(conn)
        assert any("statement_timeout = 0" in s for s in seen)


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------

class TestReadPhase:
    def test_none_when_the_phase_has_never_run(self):
        assert mig._read_phase(_Conn(_Cur(one=None)), "copy") is None

    def test_unfinished_phase_reports_its_resume_point(self):
        row = {"phase": "copy", "last_chunk_id": 12345, "rows_done": 900,
               "partitions": 32, "finished_at": None}
        st = mig._read_phase(_Conn(_Cur(one=row)), "copy")
        assert st is not None
        assert st.finished is False
        assert st.last_chunk_id == 12345
        assert st.rows_done == 900

    def test_finished_is_derived_from_finished_at(self):
        row = {"phase": "index", "last_chunk_id": None, "rows_done": 0,
               "partitions": 32, "finished_at": "2026-08-24T00:00:00"}
        st = mig._read_phase(_Conn(_Cur(one=row)), "index")
        assert st is not None and st.finished is True


class TestPartitionName:
    def test_is_zero_padded_to_three_digits(self):
        # The names are matched against pg_class by prefix elsewhere; changing
        # the width silently orphans every existing partition.
        assert mig._partition_name(0).endswith("p000")
        assert mig._partition_name(7).endswith("p007")
        assert mig._partition_name(31).endswith("p031")

    def test_is_prefixed_with_the_target_table(self):
        assert mig._partition_name(3).startswith(mig.TARGET)


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

class TestCli:
    """
    main() is the only thing standing between a typo and a renamed 1.3 TB
    table, so its guard rails are worth asserting directly: the destructive
    phase needs a second flag, --swap re-verifies even if --verify was not
    asked for, and --constraints refuses to start a multi-hour unique-index
    build that duplicates would doom.
    """

    @pytest.fixture
    def wired(self, monkeypatch):
        called: list[str] = []
        conn = _Conn()
        monkeypatch.setattr(mig, "_owner_dsn", lambda: "postgresql://o/w")
        monkeypatch.setattr(mig, "_connect", lambda dsn: conn)
        monkeypatch.setattr(mig, "_read_phase", lambda c, p: None)
        for name in ("phase_create", "phase_copy", "phase_dedupe",
                     "phase_constraints", "phase_index", "phase_swap",
                     "phase_status"):
            monkeypatch.setattr(
                mig, name,
                (lambda n: lambda *a, **k: called.append(n) or 32)(name))
        monkeypatch.setattr(mig, "phase_verify",
                            lambda c: called.append("phase_verify") or True)
        monkeypatch.setattr(mig, "_has_duplicates", lambda c, p: False)
        return called, conn

    def test_no_phase_selected_is_a_usage_error(self, wired):
        with pytest.raises(SystemExit):
            mig.main([])

    def test_status_runs_alone_and_exits_zero(self, wired):
        called, _ = wired
        assert mig.main(["--status"]) == 0
        assert called == ["phase_status"]

    def test_swap_without_the_confirmation_flag_is_refused(self, wired, capsys):
        called, _ = wired
        assert mig.main(["--swap"]) == 2
        assert "phase_swap" not in called
        assert "destructive" in capsys.readouterr().err

    def test_swap_reverifies_even_when_verify_was_not_requested(self, wired):
        """
        --verify is a separate flag, so a swap could otherwise run against
        counts nobody checked.  Verification is unconditional here.
        """
        called, _ = wired
        assert mig.main(["--swap", "--i-understand-this-is-destructive"]) == 0
        assert called == ["phase_verify", "phase_swap"]

    def test_swap_aborts_when_verification_fails(self, wired, monkeypatch):
        called, _ = wired
        monkeypatch.setattr(mig, "phase_verify", lambda c: False)
        assert mig.main(["--swap", "--i-understand-this-is-destructive"]) == 1
        assert "phase_swap" not in called

    def test_constraints_refuses_while_duplicates_exist(self, wired, monkeypatch, capsys):
        called, _ = wired
        monkeypatch.setattr(mig, "_has_duplicates", lambda c, p: True)
        assert mig.main(["--constraints"]) == 1
        assert "phase_constraints" not in called
        assert "--dedupe" in capsys.readouterr().err

    def test_missing_owner_credentials_exits_before_connecting(self, monkeypatch, capsys):
        monkeypatch.setattr(mig, "_owner_dsn", lambda: None)
        monkeypatch.setattr(mig, "_connect",
                            lambda dsn: pytest.fail("must not connect"))
        assert mig.main(["--status"]) == 2
        assert "PG_OWNER_PASSWORD" in capsys.readouterr().err

    def test_phases_run_in_dependency_order(self, wired):
        called, _ = wired
        mig.main(["--create", "--copy", "--dedupe", "--constraints", "--index",
                  "--verify"])
        assert called == ["phase_create", "phase_copy", "phase_dedupe",
                          "phase_constraints", "phase_index", "phase_verify"]

    def test_verify_failure_stops_the_run(self, wired, monkeypatch):
        monkeypatch.setattr(mig, "phase_verify", lambda c: False)
        assert mig.main(["--verify"]) == 1

    def test_recorded_partition_count_overrides_the_flag(self, wired, monkeypatch, capsys):
        """
        Passing a different --partitions than the table was built with would
        address partitions that do not exist.  The recorded value wins, loudly.
        """
        called, _ = wired
        monkeypatch.setattr(
            mig, "_read_phase",
            lambda c, p: mig.Phase(name="create", last_chunk_id=None,
                                   rows_done=0, partitions=32, finished=True))
        seen: list[int] = []
        monkeypatch.setattr(mig, "phase_index",
                            lambda c, parts, *a: seen.append(parts))
        mig.main(["--index", "--partitions", "8"])
        assert seen == [32]
        assert "using that instead" in capsys.readouterr().out

    def test_empty_tablespace_becomes_none_not_an_empty_identifier(self, wired, monkeypatch):
        # '' means "server default"; passing it through as an identifier would
        # render TABLESPACE "" and fail the build after the scan.
        seen: list[object] = []
        monkeypatch.setattr(mig, "phase_index",
                            lambda c, parts, mwm, ts, pw: seen.append(ts))
        mig.main(["--index", "--index-tablespace", ""])
        assert seen == [None]

    def test_connection_is_closed_even_when_a_phase_raises(self, wired, monkeypatch):
        _, conn = wired
        conn.close = MagicMock()
        monkeypatch.setattr(mig, "phase_copy",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            mig.main(["--copy"])
        conn.close.assert_called_once()
