"""
news/summaries/brief_finalizer.py

Daily and rolling-7-day brief generation using the ARCHITECT brain.

Produces final user-facing news_briefs rows with is_final=True.

Concurrency lock:
  Before starting, queries news_runs for any 'running' brief_finalization
  row with the same profile_id + brief_date + brief_kind started within
  cfg.finalizer_lock_window_minutes minutes.  If found, skips gracefully.
  This prevents overlapping finalizers without advisory locks.

Brief kinds produced here:
  - daily          (today's top stories across profile topics)
  - rolling_7_day  (past 7 days; only generated if daily is final)

Off-peak guard:
  Both run_daily() and run_rolling_7day() respect the off-peak idle check
  inherited from BasePipelineJob.  Pass force=True to bypass (used by the
  manual "Get News" pipeline trigger).

Shared infrastructure:
  Inherits _is_off_peak(), _get_model(), _call_brain(), _parse_json(),
  _begin_run(), _finish_run() from BasePipelineJob.
"""
from __future__ import annotations

import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from openai_client import HttpTimeouts, LlamaServerError
from news.summaries.base_summarizer import BasePipelineJob
from rag_v1.db.pg import conn_ctx
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.brief_finalizer", file_name="news_agent.log")

# Timeout for brief finalization calls.
# read_s=300 allows ARCHITECT to write multi-paragraph prose briefs (2048 tokens)
# with multi-source synthesis across up to 15 cluster summaries.
_BRAIN_TIMEOUTS = HttpTimeouts(connect_s=5.0, read_s=300.0)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are Sage Kaizen's news briefing engine. "
    "Write clear, well-structured news briefs in English. "
    "Be factual, comprehensive, and neutral. "
    "Avoid editorializing."
)

_DAILY_PROMPT_TMPL = """\
Write a daily news brief for {brief_date} based on the following story summaries.

{clusters_block}

Respond with valid JSON only in this structure:
{{
  "headline_summary": "One sentence capturing the most important story of the day.",
  "summary_short": "2-3 sentence overview of today's news.",
  "summary_long": "Comprehensive brief covering all top stories, written in flowing prose, 3-5 paragraphs."
}}"""

_ROLLING_PROMPT_TMPL = """\
Write a 7-day rolling news summary covering {start_date} to {end_date}.

{clusters_block}

Respond with valid JSON only in this structure:
{{
  "headline_summary": "One sentence capturing the most significant story of the past week.",
  "summary_short": "2-3 sentence week-in-review.",
  "summary_long": "Comprehensive 7-day summary covering all major themes, written in flowing prose, 4-6 paragraphs."
}}"""

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_CHECK_LOCK_SQL = """
SELECT run_id FROM news_runs
WHERE run_type  = 'brief_finalization'
  AND profile_id = %s::uuid
  AND status    = 'running'
  AND started_at > now() - (%s || ' minutes')::interval
LIMIT 1
"""

_LOAD_PROFILES_SQL = """
SELECT profile_id::text, profile_name
FROM news_profiles
WHERE is_enabled = true
ORDER BY profile_name
"""

_LOAD_TOP_CLUSTERS_SQL = """
SELECT
    c.cluster_id::text,
    c.cluster_title,
    c.article_count,
    cs.summary_short,
    cs.summary_medium,
    cs.top_facts_json
FROM news_story_clusters c
JOIN news_cluster_summaries cs
  ON cs.cluster_id = c.cluster_id AND cs.is_active = true
WHERE c.story_start_at >= %s
  AND c.story_start_at <  %s
  AND EXISTS (
      SELECT 1 FROM news_profile_topics pt
      WHERE pt.profile_id = %s::uuid
        AND pt.topic_id   = c.topic_id
  )
ORDER BY c.importance_score DESC
LIMIT 15
"""

_CHECK_EXISTING_FINAL_SQL = """
SELECT brief_id FROM news_briefs
WHERE profile_id = %s::uuid
  AND brief_date = %s
  AND brief_kind = %s
  AND is_final   = true
LIMIT 1
"""

_INSERT_BRIEF_SQL = """
INSERT INTO news_briefs (
    brief_id, profile_id, run_id,
    brief_date, window_start_at, window_end_at,
    brief_kind, headline_summary, summary_short, summary_long,
    top_story_cluster_ids, model_name, is_final, freshness_at
) VALUES (
    %s::uuid, %s::uuid, %s::uuid,
    %s, %s, %s,
    %s, %s, %s, %s,
    %s::uuid[], %s, true, now()
)
"""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_clusters_block(clusters: list[dict]) -> str:
    lines = []
    for i, c in enumerate(clusters, 1):
        title  = c.get("cluster_title") or "(untitled)"
        short  = c.get("summary_short") or ""
        medium = c.get("summary_medium") or ""
        text   = (medium or short)[:600]
        lines.append(f"Story {i}: {title}\n{text}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# BriefFinalizer
# ---------------------------------------------------------------------------

class BriefFinalizer(BasePipelineJob):
    """
    Generates daily and 7-day briefs using the ARCHITECT brain.

    Typical call (from NewsScheduler at 06:00 and 06:30):
        finalizer = BriefFinalizer()
        finalizer.run_daily()
        finalizer.run_rolling_7day()

    Off-peak guard is applied by default; pass force=True to bypass.
    """

    # __init__ is fully provided by BasePipelineJob; no instance state to add.

    # ------------------------------------------------------------------
    # Public entry point (BasePipelineJob contract)
    # ------------------------------------------------------------------

    def run_once(self, force: bool = False) -> dict:
        """Run both daily and rolling-7-day finalization in sequence.

        Args:
            force: When True, bypass the off-peak guard and run unconditionally.
                   Used by the manual "Get News" pipeline trigger.
        """
        daily   = self.run_daily(force=force)   or {}
        rolling = self.run_rolling_7day(force=force) or {}
        return {
            "daily_finalized":   daily.get("finalized", 0),
            "daily_skipped":     daily.get("skipped",   0),
            "daily_failed":      daily.get("failed",    0),
            "rolling_finalized": rolling.get("finalized", 0),
            "rolling_skipped":   rolling.get("skipped",   0),
            "rolling_failed":    rolling.get("failed",    0),
        }

    # ------------------------------------------------------------------
    # Separate entry points (also used by pipeline_runner directly)
    # ------------------------------------------------------------------

    def run_daily(self, brief_date: Optional[date] = None,
                  force: bool = False) -> dict:
        """Generate today's daily briefs for all enabled profiles.

        Args:
            brief_date: Date to generate for; defaults to today.
            force:      When True, bypass the off-peak guard.
        """
        if not force and not self._is_off_peak():
            _LOG.debug("news.brief_finalizer | chat active — skipping daily")
            return {"skipped": "chat_active"}
        brief_date = brief_date or date.today()
        return self._run_for_all_profiles("daily", brief_date)

    def run_rolling_7day(self, as_of_date: Optional[date] = None,
                         force: bool = False) -> dict:
        """Generate rolling 7-day briefs for all enabled profiles.

        Args:
            as_of_date: End date for the 7-day window; defaults to today.
            force:      When True, bypass the off-peak guard.
        """
        if not force and not self._is_off_peak():
            _LOG.debug("news.brief_finalizer | chat active — skipping rolling_7day")
            return {"skipped": "chat_active"}
        brief_date = as_of_date or date.today()
        return self._run_for_all_profiles("rolling_7_day", brief_date)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_for_all_profiles(self, kind: str, brief_date: date) -> dict:
        profiles = self._load_profiles()
        metrics = {"profiles": len(profiles), "finalized": 0, "skipped": 0, "failed": 0}

        for profile in profiles:
            try:
                result = self._finalize_brief(
                    profile["profile_id"], profile["profile_name"], kind, brief_date
                )
                if result == "finalized":
                    metrics["finalized"] += 1
                else:
                    metrics["skipped"] += 1
            except Exception as exc:
                _LOG.error("news.brief_finalizer | profile=%s kind=%s | %s",
                           profile["profile_name"], kind, exc, exc_info=True)
                metrics["failed"] += 1

        return metrics

    def _finalize_brief(
        self, profile_id: str, profile_name: str, kind: str, brief_date: date
    ) -> str:
        """Finalize one brief. Returns 'finalized' or 'skipped'."""

        # Check concurrency lock.
        with conn_ctx(self._dsn) as conn:
            lock_row = conn.execute(
                _CHECK_LOCK_SQL,
                [profile_id, self._cfg.finalizer_lock_window_minutes],
            ).fetchone()
        if lock_row:
            _LOG.info("news.brief_finalizer | lock held | profile=%s kind=%s",
                      profile_name, kind)
            return "skipped"

        # Check if a final brief already exists.
        with conn_ctx(self._dsn) as conn:
            existing = conn.execute(
                _CHECK_EXISTING_FINAL_SQL,
                [profile_id, brief_date, kind],
            ).fetchone()
        if existing:
            _LOG.info("news.brief_finalizer | already final | profile=%s kind=%s date=%s",
                      profile_name, kind, brief_date)
            return "skipped"

        run_id = str(uuid.uuid4())
        self._begin_run(run_id, "brief_finalization", "brief_finalizer",
                        profile_id=profile_id)

        try:
            result = self._do_finalize(run_id, profile_id, profile_name, kind, brief_date)
        except Exception as exc:
            self._finish_run(run_id, "failed", {}, str(exc))
            raise

        self._finish_run(run_id, "completed",
                         {"profile": profile_name, "kind": kind})
        return result

    def _do_finalize(
        self, run_id: str, profile_id: str, profile_name: str,
        kind: str, brief_date: date
    ) -> str:
        """Build and persist the brief."""

        today_start = datetime.combine(brief_date, datetime.min.time(), tzinfo=timezone.utc)

        if kind == "daily":
            window_start = today_start
            window_end   = today_start + timedelta(days=1)
        else:  # rolling_7_day
            window_start = today_start - timedelta(days=7)
            window_end   = today_start + timedelta(days=1)

        with conn_ctx(self._dsn) as conn:
            clusters = conn.execute(
                _LOAD_TOP_CLUSTERS_SQL,
                [window_start, window_end, profile_id],
            ).fetchall()

        if not clusters:
            _LOG.info("news.brief_finalizer | no clusters | profile=%s kind=%s date=%s",
                      profile_name, kind, brief_date)
            return "skipped"

        clusters_block = _build_clusters_block(clusters)

        if kind == "daily":
            prompt = _DAILY_PROMPT_TMPL.format(
                brief_date=str(brief_date),
                clusters_block=clusters_block,
            )
        else:
            prompt = _ROLLING_PROMPT_TMPL.format(
                start_date=str(brief_date - timedelta(days=7)),
                end_date=str(brief_date),
                clusters_block=clusters_block,
            )

        model = self._get_model(self._cfg.architect_brain_url, "architect-brain")
        try:
            raw = self._call_brain(
                self._cfg.architect_brain_url,
                model,
                system=_SYSTEM_PROMPT,
                prompt=prompt,
                max_tokens=2048,
                timeouts=_BRAIN_TIMEOUTS,
            )
        except LlamaServerError as exc:
            _LOG.warning("news.brief_finalizer | brain error | %s", exc)
            return "skipped"

        parsed = self._parse_json(raw)
        if not parsed:
            _LOG.warning("news.brief_finalizer | bad JSON | profile=%s", profile_name)
            return "skipped"

        cluster_ids = [c["cluster_id"] for c in clusters]
        brief_id    = str(uuid.uuid4())

        # Build a PostgreSQL array literal for uuid[].
        pg_array = "{" + ",".join(cluster_ids) + "}"

        with conn_ctx(self._dsn) as conn:
            conn.execute(_INSERT_BRIEF_SQL, [
                brief_id,
                profile_id,
                run_id,
                brief_date,
                window_start,
                window_end,
                kind,
                parsed.get("headline_summary") or "",
                parsed.get("summary_short") or "",
                parsed.get("summary_long") or "",
                pg_array,
                model,
            ])

        _LOG.info("news.brief_finalizer | finalized | profile=%s kind=%s date=%s brief_id=%s",
                  profile_name, kind, brief_date, brief_id)
        return "finalized"

    def _load_profiles(self) -> list[dict]:
        with conn_ctx(self._dsn) as conn:
            return conn.execute(_LOAD_PROFILES_SQL).fetchall()
