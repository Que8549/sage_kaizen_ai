"""
news/scheduler/news_scheduler.py

APScheduler BackgroundScheduler singleton for the News Runtime.

Started once via ui_streamlit_server.py using @st.cache_resource.
All news pipeline jobs run in a ThreadPoolExecutor (max_workers=4) so they
never block Streamlit's main thread.

Job registry:
  collect_all_topics       every cfg.collection_interval_minutes
  enrich_articles          every cfg.enrichment_interval_minutes
  process_images           every cfg.image_interval_minutes
  cluster_articles         every cfg.clustering_interval_hours
  summarize_articles       every cfg.summarization_interval_minutes (off-peak)
  summarize_clusters       every cfg.summarization_interval_minutes (off-peak)
  finalize_daily_brief     daily at cfg.daily_brief_hour:cfg.daily_brief_minute
  finalize_rolling_brief   daily at cfg.rolling_brief_hour:cfg.rolling_brief_minute
  reconcile_failed         daily at cfg.reconciliation_hour:00

Safety:
  - coalesce=True  prevents piling up missed runs during sleep/restart
  - misfire_grace_time=120  allows jobs to run up to 2 min after scheduled time
  - All jobs log exceptions rather than crashing the scheduler thread

Usage (in ui_streamlit_server.py):
    @st.cache_resource
    def _start_news_scheduler():
        from news.scheduler.news_scheduler import NewsScheduler
        return NewsScheduler.start()
    _start_news_scheduler()
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor as APSThreadPoolExecutor

from news.news_settings import get_news_settings
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.scheduler", file_name="news_agent.log")


class NewsScheduler:
    """
    Singleton wrapper around APScheduler BackgroundScheduler.

    All heavy imports (collector, enricher, etc.) are deferred to job
    execution time so that importing this module at startup is fast.
    """

    _instance: ClassVar[Optional[NewsScheduler]] = None

    def __init__(self) -> None:
        cfg = get_news_settings()
        executors = {
            "default": APSThreadPoolExecutor(max_workers=cfg.scheduler_max_workers),
        }
        job_defaults = {
            "coalesce": True,
            "misfire_grace_time": 120,
            "max_instances": 1,
        }
        self._scheduler = BackgroundScheduler(
            executors=executors,
            job_defaults=job_defaults,
            timezone="UTC",
        )
        self._cfg = cfg
        self._running = False

    # ------------------------------------------------------------------
    # Singleton factory
    # ------------------------------------------------------------------

    @classmethod
    def start(cls) -> "NewsScheduler":
        """Start the scheduler singleton. Safe to call multiple times."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_jobs()
            cls._instance._scheduler.start()
            cls._instance._running = True
            _LOG.info("news_scheduler | started")
        return cls._instance

    @classmethod
    def stop(cls) -> None:
        """Gracefully shut down the scheduler (called on app exit if needed)."""
        if cls._instance and cls._instance._running:
            cls._instance._scheduler.shutdown(wait=False)
            cls._instance._running = False
            _LOG.info("news_scheduler | stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def _register_jobs(self) -> None:
        cfg = self._cfg
        s = self._scheduler

        s.add_job(
            _job_collect,
            trigger="interval",
            minutes=cfg.collection_interval_minutes,
            id="collect_all_topics",
            name="News: collect all topics",
        )
        s.add_job(
            _job_enrich,
            trigger="interval",
            minutes=cfg.enrichment_interval_minutes,
            id="enrich_articles",
            name="News: enrich articles",
        )
        s.add_job(
            _job_images,
            trigger="interval",
            minutes=cfg.image_interval_minutes,
            id="process_images",
            name="News: process images",
        )
        s.add_job(
            _job_cluster,
            trigger="interval",
            hours=cfg.clustering_interval_hours,
            id="cluster_articles",
            name="News: cluster articles",
        )
        s.add_job(
            _job_summarize_articles,
            trigger="interval",
            minutes=cfg.summarization_interval_minutes,
            id="summarize_articles",
            name="News: summarize articles (off-peak)",
        )
        s.add_job(
            _job_summarize_clusters,
            trigger="interval",
            minutes=cfg.summarization_interval_minutes,
            id="summarize_clusters",
            name="News: summarize clusters (off-peak)",
        )
        s.add_job(
            _job_daily_brief,
            trigger="cron",
            hour=cfg.daily_brief_hour,
            minute=cfg.daily_brief_minute,
            id="finalize_daily_brief",
            name="News: finalize daily brief",
        )
        s.add_job(
            _job_rolling_brief,
            trigger="cron",
            hour=cfg.rolling_brief_hour,
            minute=cfg.rolling_brief_minute,
            id="finalize_rolling_brief",
            name="News: finalize rolling 7-day brief",
        )
        s.add_job(
            _job_reconcile,
            trigger="cron",
            hour=cfg.reconciliation_hour,
            minute=0,
            id="reconcile_failed",
            name="News: reconcile failed articles",
        )

        _LOG.info("news_scheduler | registered %d jobs", len(s.get_jobs()))


# ---------------------------------------------------------------------------
# Job functions (module-level; APScheduler requires picklable callables)
# ---------------------------------------------------------------------------

def _job_collect() -> None:
    try:
        from news.collectors.topic_collector import TopicCollector
        result = TopicCollector().run_once()
        _LOG.info("job:collect | %s", result)
    except Exception as exc:
        _LOG.error("job:collect | failed: %s", exc, exc_info=True)


def _job_enrich() -> None:
    try:
        from news.enrichment.article_enricher import ArticleEnricher
        result = ArticleEnricher().run_once()
        _LOG.info("job:enrich | %s", result)
    except Exception as exc:
        _LOG.error("job:enrich | failed: %s", exc, exc_info=True)


def _job_images() -> None:
    try:
        from news.images.news_image_pipeline import NewsImagePipeline
        result = NewsImagePipeline().run_once()
        _LOG.info("job:images | %s", result)
    except Exception as exc:
        _LOG.error("job:images | failed: %s", exc, exc_info=True)


def _job_cluster() -> None:
    try:
        from news.clustering.article_clusterer import ArticleClusterer
        result = ArticleClusterer().run_once()
        _LOG.info("job:cluster | %s", result)
    except Exception as exc:
        _LOG.error("job:cluster | failed: %s", exc, exc_info=True)


def _job_summarize_articles() -> None:
    try:
        from news.summaries.article_summarizer import ArticleSummarizer
        result = ArticleSummarizer().run_once()
        _LOG.info("job:summarize_articles | %s", result)
    except Exception as exc:
        _LOG.error("job:summarize_articles | failed: %s", exc, exc_info=True)


def _job_summarize_clusters() -> None:
    try:
        from news.summaries.cluster_summarizer import ClusterSummarizer
        result = ClusterSummarizer().run_once()
        _LOG.info("job:summarize_clusters | %s", result)
    except Exception as exc:
        _LOG.error("job:summarize_clusters | failed: %s", exc, exc_info=True)


def _job_daily_brief() -> None:
    try:
        from news.summaries.brief_finalizer import BriefFinalizer
        result = BriefFinalizer().run_daily()
        _LOG.info("job:daily_brief | %s", result)
    except Exception as exc:
        _LOG.error("job:daily_brief | failed: %s", exc, exc_info=True)


def _job_rolling_brief() -> None:
    try:
        from news.summaries.brief_finalizer import BriefFinalizer
        result = BriefFinalizer().run_rolling_7day()
        _LOG.info("job:rolling_brief | %s", result)
    except Exception as exc:
        _LOG.error("job:rolling_brief | failed: %s", exc, exc_info=True)


def _job_reconcile() -> None:
    """
    Reset stuck or retryable articles so they re-enter the pipeline.

    - failed_fetch with retry_count < max_retries → pending
    - Articles stuck in 'fetching' for > 30 min → pending (crashed worker)
    """
    try:
        from news.news_settings import get_news_settings
        from rag_v1.db.pg import conn_ctx
        cfg = get_news_settings()

        with conn_ctx(cfg.pg_dsn) as conn:
            # Reset genuinely retryable failed fetches.
            r1 = conn.execute("""
                UPDATE daily_news
                SET fetch_status = 'pending', updated_at = now()
                WHERE fetch_status = 'failed_fetch'
                  AND (metadata->>'fetch_retry_count')::int < %s
            """, [cfg.fetch_max_retries])

            # Reset articles stuck mid-fetch (worker crash).
            r2 = conn.execute("""
                UPDATE daily_news
                SET fetch_status = 'pending', updated_at = now()
                WHERE fetch_status = 'fetching'
                  AND updated_at < now() - INTERVAL '30 minutes'
            """)

            # Reset stuck summarization.
            r3 = conn.execute("""
                UPDATE daily_news
                SET summary_status = 'pending', updated_at = now()
                WHERE summary_status = 'summarizing'
                  AND updated_at < now() - INTERVAL '30 minutes'
            """)

        _LOG.info("job:reconcile | fetch_reset=%s | stuck_fetch=%s | sum_reset=%s",
                  r1.rowcount, r2.rowcount, r3.rowcount)
    except Exception as exc:
        _LOG.error("job:reconcile | failed: %s", exc, exc_info=True)
