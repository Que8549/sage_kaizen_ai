"""
tests/test_news_scheduler.py

Unit tests for news/scheduler/news_scheduler.py (added 2026-08-03 — previously
0% covered despite owning all nine production jobs).

Moved here from sage_kaizen_ai_ingest on 2026-08-24.  It had been testing the
ingest copy of news_scheduler.py, which was dead code: both projects import
this module from the MAIN repo (CLAUDE.md §13).  So the tested copy and the
running copy were different files, and they had diverged — the ingest copy
carried a singleton lock, context-managed jobs, and a COALESCE fix that the
live one lacked.  Those fixes are now in the live module and this suite tests
it, which is why the coverage for it went from 0% to real.

The APScheduler BackgroundScheduler is replaced with a recording fake, so no
threads are started and no job ever fires on a timer.  The job functions are
exercised directly with their heavy collaborators monkeypatched.

The property under test throughout is the one that actually matters in
production: **every _job_* function swallows its exceptions**.  APScheduler
would otherwise drop the job's next run on an uncaught error, silently killing
one of the nine pipelines for the lifetime of the process.
"""
from __future__ import annotations

import importlib

import pytest

from news.scheduler import news_scheduler as ns


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeJob:
    """Minimal stand-in for an APScheduler Job (only what logging touches)."""

    def __init__(self, job_id, name):
        self.id = job_id
        self.name = name
        self.next_run_time = None


class _FakeScheduler:
    def __init__(self, *a, **k):
        self.init_kwargs = k
        self.jobs: list[dict] = []
        self.started = False
        self.shutdown_called = False

    def add_job(self, fn, **kw):
        self.jobs.append({"fn": fn, **kw})

    def get_jobs(self):
        """_register_jobs() logs the registered set via get_jobs()."""
        return [_FakeJob(j["id"], j.get("name", "")) for j in self.jobs]

    def start(self):
        self.started = True

    def shutdown(self, wait=True):
        self.shutdown_called = True


@pytest.fixture
def fake_scheduler(monkeypatch):
    """Patch BackgroundScheduler and reset the singleton around each test."""
    monkeypatch.setattr(ns, "BackgroundScheduler", _FakeScheduler)
    ns.NewsScheduler._instance = None
    yield
    ns.NewsScheduler._instance = None


# ---------------------------------------------------------------------------
# Singleton lifecycle
# ---------------------------------------------------------------------------

class TestSchedulerLifecycle:
    def test_start_registers_jobs_and_starts_once(self, fake_scheduler):
        inst = ns.NewsScheduler.start()
        assert inst._scheduler.started is True
        assert inst.is_running is True
        assert len(inst._scheduler.jobs) == 9, "all nine pipeline jobs must register"

    def test_start_is_idempotent(self, fake_scheduler):
        first = ns.NewsScheduler.start()
        second = ns.NewsScheduler.start()
        assert first is second
        assert len(second._scheduler.jobs) == 9, "jobs must not be registered twice"

    def test_stop_shuts_down_and_clears_running(self, fake_scheduler):
        inst = ns.NewsScheduler.start()
        ns.NewsScheduler.stop()
        assert inst._scheduler.shutdown_called is True
        assert inst.is_running is False

    def test_stop_without_start_is_a_noop(self, fake_scheduler):
        ns.NewsScheduler.stop()   # must not raise

    def test_stop_twice_is_a_noop(self, fake_scheduler):
        ns.NewsScheduler.start()
        ns.NewsScheduler.stop()
        ns.NewsScheduler.stop()   # must not raise

    def test_scheduler_is_configured_utc_with_job_defaults(self, fake_scheduler):
        inst = ns.NewsScheduler.start()
        kw = inst._scheduler.init_kwargs
        assert kw["timezone"] == "UTC"
        assert kw["job_defaults"]["coalesce"] is True
        assert kw["job_defaults"]["max_instances"] == 1


class TestJobRegistration:
    def test_every_job_has_a_unique_id(self, fake_scheduler):
        inst = ns.NewsScheduler.start()
        ids = [j["id"] for j in inst._scheduler.jobs]
        assert len(ids) == len(set(ids))

    def test_expected_job_ids_are_present(self, fake_scheduler):
        inst = ns.NewsScheduler.start()
        ids = {j["id"] for j in inst._scheduler.jobs}
        assert "collect_all_topics" in ids
        assert "enrich_articles" in ids

    def test_all_registered_callables_are_module_level_functions(self, fake_scheduler):
        """APScheduler needs picklable callables — no bound methods or lambdas."""
        inst = ns.NewsScheduler.start()
        for job in inst._scheduler.jobs:
            fn = job["fn"]
            assert getattr(ns, fn.__name__, None) is fn, f"{fn} is not module-level"

    def test_every_job_uses_a_supported_trigger(self, fake_scheduler):
        inst = ns.NewsScheduler.start()
        for job in inst._scheduler.jobs:
            assert job["trigger"] in {"interval", "cron"}


# ---------------------------------------------------------------------------
# Job functions — the swallow-everything contract
# ---------------------------------------------------------------------------

_SIMPLE_JOBS = [
    ("_job_collect", "news.collectors.topic_collector", "TopicCollector", False),
    ("_job_cluster", "news.clustering.article_clusterer", "ArticleClusterer", False),
    ("_job_summarize_articles",
     "news.summaries.article_summarizer", "ArticleSummarizer", False),
    ("_job_summarize_clusters",
     "news.summaries.cluster_summarizer", "ClusterSummarizer", False),
    ("_job_enrich", "news.enrichment.article_enricher", "ArticleEnricher", True),
    ("_job_images", "news.images.news_image_pipeline", "NewsImagePipeline", True),
]


def _install(monkeypatch, module_path, cls_name, *, context_manager, raises=None,
             recorder=None):
    mod = importlib.import_module(module_path)

    class _Fake:
        def run_once(self, *a, **k):
            if raises is not None:
                raise raises
            if recorder is not None:
                recorder.append(cls_name)
            return {"ok": 1}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(mod, cls_name, _Fake)


class TestJobFunctions:
    @pytest.mark.parametrize("job_name,module_path,cls_name,is_cm", _SIMPLE_JOBS)
    def test_job_invokes_run_once(self, monkeypatch, job_name, module_path,
                                  cls_name, is_cm):
        seen: list[str] = []
        _install(monkeypatch, module_path, cls_name,
                 context_manager=is_cm, recorder=seen)
        getattr(ns, job_name)()
        assert seen == [cls_name]

    @pytest.mark.parametrize("job_name,module_path,cls_name,is_cm", _SIMPLE_JOBS)
    def test_job_swallows_exceptions(self, monkeypatch, job_name, module_path,
                                     cls_name, is_cm):
        """An uncaught error here would make APScheduler drop the job for good."""
        _install(monkeypatch, module_path, cls_name,
                 context_manager=is_cm, raises=RuntimeError("pipeline exploded"))
        getattr(ns, job_name)()   # must not raise

    @pytest.mark.parametrize(
        "job_name,method",
        [("_job_daily_brief", "run_daily"), ("_job_rolling_brief", "run_rolling_7day")],
    )
    def test_brief_jobs_call_the_right_method(self, monkeypatch, job_name, method):
        import news.summaries.brief_finalizer as bf

        called = []

        class _Fake:
            def run_daily(self, *a, **k):
                called.append("run_daily")
                return {}

            def run_rolling_7day(self, *a, **k):
                called.append("run_rolling_7day")
                return {}

        monkeypatch.setattr(bf, "BriefFinalizer", _Fake)
        getattr(ns, job_name)()
        assert called == [method]

    @pytest.mark.parametrize("job_name", ["_job_daily_brief", "_job_rolling_brief"])
    def test_brief_jobs_swallow_exceptions(self, monkeypatch, job_name):
        import news.summaries.brief_finalizer as bf

        class _Boom:
            def run_daily(self, *a, **k):
                raise RuntimeError("brain down")

            def run_rolling_7day(self, *a, **k):
                raise RuntimeError("brain down")

        monkeypatch.setattr(bf, "BriefFinalizer", _Boom)
        getattr(ns, job_name)()   # must not raise


class TestReconcileJob:
    def _patch_conn(self, monkeypatch, calls, rowcount=3, raises=None):
        from contextlib import contextmanager

        class _Result:
            def __init__(self, n):
                self.rowcount = n

        class _Conn:
            def execute(self, sql, params=None):
                if raises is not None:
                    raise raises
                calls.append((sql, params))
                return _Result(rowcount)

        @contextmanager
        def _ctx(dsn):
            yield _Conn()

        import rag_v1.db.pg as pg
        monkeypatch.setattr(pg, "conn_ctx", _ctx)

    def test_issues_three_reset_statements(self, monkeypatch):
        calls: list[tuple] = []
        self._patch_conn(monkeypatch, calls)
        ns._job_reconcile()

        assert len(calls) == 3
        joined = " ".join(sql for sql, _ in calls)
        assert "failed_fetch" in joined
        assert "'fetching'" in joined
        assert "summary_status" in joined

    def test_retry_cap_is_passed_from_settings(self, monkeypatch):
        calls: list[tuple] = []
        self._patch_conn(monkeypatch, calls)
        ns._job_reconcile()

        from news.news_settings import get_news_settings
        assert calls[0][1] == [get_news_settings().fetch_max_retries]

    def test_swallows_db_errors(self, monkeypatch):
        self._patch_conn(monkeypatch, [], raises=RuntimeError("db gone"))
        ns._job_reconcile()   # must not raise
