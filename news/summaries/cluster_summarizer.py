"""
news/summaries/cluster_summarizer.py

Cluster-level summarization.

For each news_story_cluster that has ≥ 2 summarized member articles but no
active cluster summary, this module produces a cluster-level synthesis:
  - summary_short   (2–3 sentences, event-level)
  - summary_medium  (one paragraph with source diversity notes)
  - top_facts_json  (list of 3–5 key facts across sources)
  - source_diversity_json (dict of source_name → count)
  - confidence_score (0.0–1.0, based on source count and agreement)

Brain selection:
  ≤ 5 articles in cluster → FAST brain (port 8011)
  > 5 articles in cluster → ARCHITECT brain (port 8012, richer synthesis)

Off-peak guard identical to article_summarizer.
  Pass force=True to bypass (used by the "Get News" manual pipeline trigger).

Shared infrastructure:
  Inherits _is_off_peak(), _get_model(), _call_brain(), _parse_json(),
  _begin_run(), _finish_run() from BasePipelineJob.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Optional

from openai_client import HttpTimeouts, LlamaServerError
from news.news_settings import get_news_settings
from news.summaries.base_summarizer import BasePipelineJob
from rag_v1.db.pg import conn_ctx
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.cluster_summarizer")

# Timeout for cluster summarization calls.
# read_s=180 allows time for ARCHITECT to synthesize large clusters (>5 articles)
# with multi-paragraph output.
_BRAIN_TIMEOUTS = HttpTimeouts(connect_s=5.0, read_s=180.0)

# Clusters with more than this many articles use the ARCHITECT brain for
# richer multi-source synthesis; smaller clusters use the faster FAST brain.
_FAST_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a senior news analyst. "
    "Synthesize multiple news articles about the same story into a concise, "
    "accurate cluster summary. Respond only in English. Be factual and neutral."
)

_CLUSTER_PROMPT_TMPL = """\
The following {n} news articles all cover the same story or event.
Synthesize them into a cluster summary.

{articles_block}

Respond with valid JSON only — no markdown — in this exact structure:
{{
  "summary_short": "2-3 sentence event summary.",
  "summary_medium": "One paragraph covering all key angles, including source diversity if present.",
  "top_facts": ["fact 1", "fact 2", "fact 3"],
  "confidence_score": 0.85
}}"""

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_FETCH_CLUSTERS_SQL = """
SELECT
    c.cluster_id::text,
    c.topic_id::text,
    c.article_count,
    c.cluster_title
FROM news_story_clusters c
WHERE c.article_count >= 2
  AND NOT EXISTS (
      SELECT 1 FROM news_cluster_summaries cs
      WHERE cs.cluster_id = c.cluster_id
        AND cs.is_active   = true
  )
ORDER BY c.importance_score DESC
LIMIT %s
"""

_FETCH_CLUSTER_ARTICLES_SQL = """
SELECT
    d.article_id::text,
    d.headline,
    s.summary_short,
    d.news_source
FROM daily_news d
LEFT JOIN news_article_summaries s
       ON s.article_id = d.article_id AND s.is_active = true
WHERE d.cluster_id = %s::uuid
  AND d.summary_status = 'summarized'
ORDER BY d.rank_score DESC NULLS LAST
LIMIT 12
"""

_DEACTIVATE_OLD_CLUSTER_SUMMARY_SQL = """
UPDATE news_cluster_summaries
SET is_active = false
WHERE cluster_id = %s::uuid AND is_active = true
"""

_INSERT_CLUSTER_SUMMARY_SQL = """
INSERT INTO news_cluster_summaries (
    cluster_id, run_id, summary_kind,
    summary_short, summary_medium,
    top_facts_json, source_diversity_json,
    confidence_score, model_name, prompt_version, is_active
) VALUES (
    %s::uuid, %s::uuid, 'article_short',
    %s, %s,
    %s::jsonb, %s::jsonb,
    %s, %s, 'v1', true
)
"""


# ---------------------------------------------------------------------------
# ClusterSummarizer
# ---------------------------------------------------------------------------

class ClusterSummarizer(BasePipelineJob):
    """Generates cluster-level summaries (off-peak only by default)."""

    # __init__ is fully provided by BasePipelineJob; no instance state to add.

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_once(self, force: bool = False) -> dict:
        """Summarize unsummarized story clusters (off-peak only by default).

        Args:
            force: When True, bypass the off-peak guard and run unconditionally.
                   Used by the manual "Get News" pipeline trigger.
        """
        if not force and not self._is_off_peak():
            return {"skipped": "chat_active"}

        run_id = str(uuid.uuid4())
        t0 = time.monotonic()
        self._begin_run(run_id, "cluster_summarization", "cluster_summarizer")

        clusters = self._fetch_clusters()
        if not clusters:
            self._finish_run(run_id, "completed", {"summarized": 0})
            return {"summarized": 0}

        _LOG.info("news.cluster_summarizer | start | run=%s | clusters=%d",
                  run_id, len(clusters))
        metrics = {"summarized": 0, "failed": 0}

        for row in clusters:
            if not force and not self._is_off_peak():
                _LOG.info("news.cluster_summarizer | chat active — stopping early")
                break
            try:
                self._summarize_cluster(row, run_id, metrics)
            except Exception as exc:
                _LOG.error("news.cluster_summarizer | cluster=%s | %s",
                           row["cluster_id"], exc, exc_info=True)
                metrics["failed"] += 1

        metrics["duration_s"] = round(time.monotonic() - t0, 2)
        self._finish_run(run_id, "completed", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_clusters(self) -> list[dict]:
        with conn_ctx(self._dsn) as conn:
            return conn.execute(_FETCH_CLUSTERS_SQL, [20]).fetchall()

    def _summarize_cluster(self, row: dict, run_id: str, metrics: dict) -> None:
        cluster_id = row["cluster_id"]
        n_articles = int(row.get("article_count") or 0)

        with conn_ctx(self._dsn) as conn:
            articles = conn.execute(
                _FETCH_CLUSTER_ARTICLES_SQL, [cluster_id]
            ).fetchall()

        if not articles:
            metrics["failed"] += 1
            return

        # Build article block for the prompt.
        lines = []
        for i, a in enumerate(articles, 1):
            hl  = a.get("headline") or "(no headline)"
            sm  = a.get("summary_short") or ""
            src = a.get("news_source") or "unknown"
            lines.append(f"Article {i} [{src}]: {hl}. {sm}".strip())
        articles_block = "\n".join(lines)

        prompt = _CLUSTER_PROMPT_TMPL.format(
            n=len(articles), articles_block=articles_block
        )

        # Select brain and token budget based on cluster size.
        if n_articles <= _FAST_THRESHOLD:
            brain_url  = self._cfg.fast_brain_url
            fallback   = "fast-brain"
            max_tokens = 768
        else:
            brain_url  = self._cfg.architect_brain_url
            fallback   = "architect-brain"
            max_tokens = 1024

        model = self._get_model(brain_url, fallback)

        try:
            raw = self._call_brain(
                brain_url,
                model,
                system=_SYSTEM_PROMPT,
                prompt=prompt,
                max_tokens=max_tokens,
                timeouts=_BRAIN_TIMEOUTS,
            )
        except LlamaServerError as exc:
            _LOG.warning("news.cluster_summarizer | brain error | %s", exc)
            metrics["failed"] += 1
            return

        parsed = self._parse_json(raw)
        if not parsed:
            metrics["failed"] += 1
            return

        src_diversity = _source_diversity(articles)
        confidence    = min(1.0, len(src_diversity) / max(n_articles, 1) * 2.0)

        with conn_ctx(self._dsn) as conn:
            with conn.transaction():
                conn.execute(_DEACTIVATE_OLD_CLUSTER_SUMMARY_SQL, [cluster_id])
                conn.execute(_INSERT_CLUSTER_SUMMARY_SQL, [
                    cluster_id,
                    run_id,
                    parsed.get("summary_short") or "",
                    parsed.get("summary_medium") or "",
                    json.dumps(parsed.get("top_facts") or []),
                    json.dumps(src_diversity),
                    float(parsed.get("confidence_score") or confidence),
                    model,
                ])

        metrics["summarized"] += 1


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _source_diversity(article_rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in article_rows:
        src = (r.get("news_source") or "unknown").strip()
        counts[src] = counts.get(src, 0) + 1
    return counts
