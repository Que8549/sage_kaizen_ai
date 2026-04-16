"""
news/pipeline_runner.py

On-demand full pipeline trigger for the "Get News" button in the Streamlit UI.

Runs all 6 pipeline stages sequentially in a daemon thread and exposes
a module-level state dict so the UI polling fragment can display live progress.

Stages (in order):
  1  Collect articles          — SearXNG → daily_news (TopicCollector)
  2  Enrich articles           — full-text fetch + BGE-M3 embeddings (ArticleEnricher)
  3  Cluster articles          — DBSCAN story clustering (ArticleClusterer)
  4  Summarize articles        — FAST brain per-article summaries (ArticleSummarizer)
  5  Summarize clusters        — FAST/ARCHITECT cluster synthesis (ClusterSummarizer)
  6  Finalize briefs           — ARCHITECT daily + 7-day briefs (BriefFinalizer)

Stages 4 and 5 are run with force=True, bypassing the off-peak guard.

Thread safety:
  _state is a plain dict.  The pipeline thread writes; the Streamlit fragment reads.
  Python's GIL guarantees that dict key reads/writes are atomic for CPython, making
  this safe without an explicit lock for the simple scalar values used here.
  Timestamps and metrics are immutable-once-written so no torn reads are possible.

Usage:
    from news.pipeline_runner import run_pipeline_async, get_state

    if st.button("Get News"):
        run_pipeline_async()

    state = get_state()   # read in @st.fragment polling loop
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.pipeline_runner", file_name="news_agent.log")

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGE_LABELS = [
    "Collecting articles",
    "Enriching articles",
    "Clustering articles",
    "Summarizing articles",
    "Summarizing clusters",
    "Finalizing briefs",
]
TOTAL_STAGES = len(STAGE_LABELS)

# ---------------------------------------------------------------------------
# Shared state — written by pipeline thread, read by UI fragment
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "running":      False,
    "stage_num":    0,          # 1-TOTAL_STAGES while running; 0 when idle
    "stage_label":  "",
    "stage_metrics": {},        # metrics dict returned by the last completed stage
    "all_metrics":  [],         # list of (label, metrics) for all stages this run
    "error":        None,       # Exception str if any stage raised
    "started_at":   None,       # datetime (UTC)
    "finished_at":  None,       # datetime (UTC)
    "fast_ok":      True,       # brain health check result at run start
    "arch_ok":      True,
    "fast_warn":    "",         # human-readable warning if brain unavailable
    "arch_warn":    "",
}

_run_lock = threading.Lock()   # prevents two simultaneous pipeline runs


def get_state() -> dict[str, Any]:
    """Return a shallow copy of the pipeline state (safe to read from any thread)."""
    return dict(_state)


def is_running() -> bool:
    """True while the pipeline thread is executing."""
    return bool(_state["running"])


# ---------------------------------------------------------------------------
# Brain health checks
# ---------------------------------------------------------------------------

def _check_brains() -> tuple[bool, str, bool, str]:
    """
    Check FAST (8011) and ARCHITECT (8012) availability.
    Returns (fast_ok, fast_warn, arch_ok, arch_warn).
    """
    fast_ok = arch_ok = False
    fast_warn = arch_warn = ""
    try:
        from openai_client import health_check, HttpTimeouts
        from news.news_settings import get_news_settings
        cfg = get_news_settings()
        _t = HttpTimeouts(connect_s=2.0, read_s=3.0)

        fast_ok, _fast_lat = health_check(cfg.fast_brain_url, timeouts=_t)
        arch_ok, _arch_lat = health_check(cfg.architect_brain_url, timeouts=_t)

        if not fast_ok:
            fast_warn = (
                f"FAST brain ({cfg.fast_brain_url}) is not responding. "
                "Article summarization (stage 4) will fail or be skipped."
            )
        if not arch_ok:
            arch_warn = (
                f"ARCHITECT brain ({cfg.architect_brain_url}) is not responding. "
                "Cluster summarization (stage 5) and brief finalization (stage 6) "
                "will fail or be skipped."
            )
    except Exception as exc:
        _LOG.warning("pipeline_runner | brain health check failed: %s", exc)

    return fast_ok, fast_warn, arch_ok, arch_warn


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _stage_collect() -> dict:
    from news.collectors.topic_collector import TopicCollector
    r = TopicCollector().run_once()
    # CollectionResult is a dataclass — convert to plain dict for the UI.
    return {
        "topics_run":     r.topics_run,
        "topics_ok":      r.topics_ok,
        "topics_failed":  r.topics_failed,
        "new_articles":   r.total_new,
        "updated":        r.total_updated,
        "duration_s":     round(r.duration_s, 2),
    }


def _stage_enrich() -> dict:
    from news.enrichment.article_enricher import ArticleEnricher
    return ArticleEnricher().run_once() or {}


def _stage_cluster() -> dict:
    from news.clustering.article_clusterer import ArticleClusterer
    return ArticleClusterer().run_once() or {}


def _stage_summarize_articles() -> dict:
    from news.summaries.article_summarizer import ArticleSummarizer
    return ArticleSummarizer().run_once(force=True) or {}


def _stage_summarize_clusters() -> dict:
    from news.summaries.cluster_summarizer import ClusterSummarizer
    return ClusterSummarizer().run_once(force=True) or {}


def _stage_finalize_briefs() -> dict:
    from news.summaries.brief_finalizer import BriefFinalizer
    f = BriefFinalizer()
    daily   = f.run_daily(force=True)          or {}
    rolling = f.run_rolling_7day(force=True)   or {}
    return {
        "daily_finalized":   daily.get("finalized", 0),
        "daily_skipped":     daily.get("skipped",   0),
        "daily_failed":      daily.get("failed",    0),
        "rolling_finalized": rolling.get("finalized", 0),
        "rolling_skipped":   rolling.get("skipped",   0),
        "rolling_failed":    rolling.get("failed",    0),
    }


_STAGE_FNS = [
    _stage_collect,
    _stage_enrich,
    _stage_cluster,
    _stage_summarize_articles,
    _stage_summarize_clusters,
    _stage_finalize_briefs,
]


# ---------------------------------------------------------------------------
# Sync runner (called inside daemon thread)
# ---------------------------------------------------------------------------

def _run_pipeline_sync() -> None:
    _state["running"]     = True
    _state["started_at"]  = datetime.now(timezone.utc)
    _state["finished_at"] = None
    _state["error"]       = None
    _state["all_metrics"] = []
    _state["stage_num"]   = 0
    _state["stage_label"] = ""
    _state["stage_metrics"] = {}

    _LOG.info("pipeline_runner | start | stages=%d", TOTAL_STAGES)

    try:
        for i, (label, fn) in enumerate(zip(STAGE_LABELS, _STAGE_FNS), start=1):
            _state["stage_num"]   = i
            _state["stage_label"] = label
            _state["stage_metrics"] = {}
            _LOG.info("pipeline_runner | stage %d/%d | %s", i, TOTAL_STAGES, label)

            try:
                metrics = fn()
                _state["stage_metrics"] = metrics
                _state["all_metrics"].append((label, metrics))
                _LOG.info("pipeline_runner | stage %d done | %s", i, metrics)
            except Exception as exc:
                _state["stage_metrics"] = {"error": str(exc)}
                _state["all_metrics"].append((label, {"error": str(exc)}))
                _LOG.error(
                    "pipeline_runner | stage %d (%s) failed: %s",
                    i, label, exc, exc_info=True,
                )
                # Continue to next stage rather than aborting the whole pipeline.
                # A collection failure should not prevent brief finalization if
                # there are already sufficient articles in the DB.

    except Exception as exc:
        _state["error"] = str(exc)
        _LOG.error("pipeline_runner | unhandled error: %s", exc, exc_info=True)

    finally:
        _state["running"]     = False
        _state["finished_at"] = datetime.now(timezone.utc)
        _state["stage_num"]   = 0
        _state["stage_label"] = ""
        elapsed = (
            _state["finished_at"] - _state["started_at"]
        ).total_seconds()
        _LOG.info("pipeline_runner | finished | elapsed=%.1fs", elapsed)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline_async() -> bool:
    """
    Start the full news pipeline in a daemon thread.

    Returns True if the pipeline was started, False if it was already running.
    The caller should check get_state()["running"] to avoid double-triggers.

    Brain health is checked synchronously before the thread starts so that
    warnings appear in the UI immediately.
    """
    if not _run_lock.acquire(blocking=False):
        _LOG.info("pipeline_runner | already running — ignoring duplicate trigger")
        return False

    # Health-check the brains before starting (fast, 2-3 s).
    fast_ok, fast_warn, arch_ok, arch_warn = _check_brains()
    _state["fast_ok"]   = fast_ok
    _state["arch_ok"]   = arch_ok
    _state["fast_warn"] = fast_warn
    _state["arch_warn"] = arch_warn

    def _thread_body() -> None:
        try:
            _run_pipeline_sync()
        finally:
            _run_lock.release()

    t = threading.Thread(target=_thread_body, name="news-pipeline", daemon=True)
    t.start()
    _LOG.info("pipeline_runner | thread started")
    return True
