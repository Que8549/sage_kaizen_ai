"""
news/summaries/article_summarizer.py

Article-level summarization using the FAST brain.

For each article with summary_status='pending' and fetch_status='fetched',
this module calls the FAST brain (port 8011) directly — bypassing the chat
router — to produce:
  - summary_short  (1–2 sentences)
  - summary_medium (3–5 sentences)
  - key_points_json (list of 3–5 bullet facts)
  - entities_json  (list of named entities)

Output is written to news_article_summaries (is_active=True).  If a prior
active summary exists for that article+kind, it is set to is_active=False
before insertion (versioning).

Off-peak guard:
  Summarization only runs when last_chat_activity_ts() shows the user has
  been idle for at least cfg.off_peak_idle_seconds.  This prevents the FAST
  brain from being saturated during live chat sessions.
  Pass force=True to bypass (used by the "Get News" manual pipeline trigger).

Shared infrastructure:
  Inherits _is_off_peak(), _get_model(), _call_brain(), _parse_json(),
  _begin_run(), _finish_run() from BasePipelineJob.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Optional

from openai_client import HttpTimeouts, LlamaServerError
from news.news_settings import get_news_settings
from news.summaries.base_summarizer import BasePipelineJob
from rag_v1.db.pg import conn_ctx
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.article_summarizer")

# Timeout for article summarization calls.
# read_s=120 allows ~1000-token articles with some headroom; individual
# summaries are short (512 max_tokens) but prefill takes time on long content.
_BRAIN_TIMEOUTS = HttpTimeouts(connect_s=5.0, read_s=120.0)

# Maximum article content characters passed to the brain.
# Keeps prompt size predictable and avoids blowing the context budget on
# verbose articles (3000 chars ≈ 750 tokens, leaving ~15 K for ctx padding).
_MAX_CONTENT_CHARS = 3000

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise news summarization assistant. "
    "Respond only in English. Be factual, concise, and neutral. "
    "Never add opinions, speculation, or filler."
)

_ARTICLE_PROMPT_TMPL = """\
Summarize the following news article.

HEADLINE: {headline}

CONTENT:
{content}

Respond with valid JSON only — no markdown, no explanation — in this exact structure:
{{
  "summary_short": "1-2 sentence summary.",
  "summary_medium": "3-5 sentence summary with key facts.",
  "key_points": ["fact 1", "fact 2", "fact 3"],
  "entities": ["Entity A", "Entity B"]
}}"""

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_FETCH_PENDING_SQL = """
SELECT
    article_id::text,
    headline,
    snippet,
    article_content
FROM daily_news
WHERE summary_status = 'pending'
  AND fetch_status   = 'fetched'
ORDER BY first_seen_at DESC
LIMIT %s
"""

_DEACTIVATE_OLD_SUMMARY_SQL = """
UPDATE news_article_summaries
SET is_active = false
WHERE article_id = %s::uuid
  AND is_active  = true
"""

_INSERT_SUMMARY_SQL = """
INSERT INTO news_article_summaries (
    article_id, run_id, summary_kind,
    summary_short, summary_medium,
    key_points_json, entities_json,
    model_name, prompt_version,
    is_active
) VALUES (
    %s::uuid, %s::uuid, 'article_short',
    %s, %s,
    %s::jsonb, %s::jsonb,
    %s, 'v1',
    true
)
"""

_SET_SUMMARY_STATUS_SQL = """
UPDATE daily_news
SET summary_status = %s, updated_at = now()
WHERE article_id = %s::uuid
"""


# ---------------------------------------------------------------------------
# ArticleSummarizer
# ---------------------------------------------------------------------------

class ArticleSummarizer(BasePipelineJob):
    """
    Generates article-level summaries using the FAST brain.

    Off-peak guard enforced by default: summarization skips if the user has
    been active within cfg.off_peak_idle_seconds.  Pass force=True to bypass.
    """

    def __init__(self) -> None:
        super().__init__()
        self._semaphore = threading.Semaphore(self._cfg.summarization_concurrency)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_once(self, force: bool = False) -> dict:
        """Summarize one batch of pending articles (off-peak only by default).

        Args:
            force: When True, bypass the off-peak guard and run unconditionally.
                   Used by the manual "Get News" pipeline trigger.
        """
        if not force and not self._is_off_peak():
            _LOG.debug("news.article_summarizer | chat active — skipping")
            return {"skipped": "chat_active"}

        run_id = str(uuid.uuid4())
        t0 = time.monotonic()
        self._begin_run(run_id, "article_summarization", "article_summarizer")

        rows = self._fetch_pending()
        if not rows:
            self._finish_run(run_id, "completed", {"summarized": 0})
            return {"summarized": 0}

        _LOG.info("news.article_summarizer | start | run=%s | articles=%d",
                  run_id, len(rows))

        model   = self._get_model(self._cfg.fast_brain_url, "fast-brain")
        metrics = {"summarized": 0, "failed": 0, "skipped_chat": 0}

        for row in rows:
            # Re-check off-peak before each article — long batches can span
            # a chat session starting.  When force=True, skip this check too.
            if not force and not self._is_off_peak():
                remaining = len(rows) - metrics["summarized"] - metrics["failed"]
                metrics["skipped_chat"] += remaining
                _LOG.info("news.article_summarizer | chat became active — stopping early")
                break
            self._summarize_one(row, run_id, model, metrics)

        metrics["duration_s"] = round(time.monotonic() - t0, 2)
        self._finish_run(run_id, "completed", metrics)
        _LOG.info("news.article_summarizer | done | %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_pending(self) -> list[dict]:
        batch = self._cfg.fetch_batch_size
        with conn_ctx(self._dsn) as conn:
            return conn.execute(_FETCH_PENDING_SQL, [batch]).fetchall()

    def _summarize_one(self, row: dict, run_id: str, model: str, metrics: dict) -> None:
        article_id = row["article_id"]
        headline   = row.get("headline") or ""
        content    = (row.get("article_content") or row.get("snippet") or "")
        content    = content[:_MAX_CONTENT_CHARS]

        if not content.strip():
            self._set_status(article_id, "skipped")
            return

        prompt = _ARTICLE_PROMPT_TMPL.format(
            headline=headline or "(no headline)",
            content=content,
        )

        try:
            with self._semaphore:
                raw = self._call_brain(
                    self._cfg.fast_brain_url,
                    model,
                    system=_SYSTEM_PROMPT,
                    prompt=prompt,
                    max_tokens=512,
                    timeouts=_BRAIN_TIMEOUTS,
                    temperature=0.1,   # lower than default — factual extraction
                )
        except LlamaServerError as exc:
            _LOG.warning("news.article_summarizer | brain error | article=%s | %s",
                         article_id, exc)
            self._set_status(article_id, "failed_summary")
            metrics["failed"] += 1
            return
        except Exception as exc:
            _LOG.warning("news.article_summarizer | unexpected error | article=%s | %s",
                         article_id, exc)
            self._set_status(article_id, "failed_summary")
            metrics["failed"] += 1
            return

        parsed = self._parse_json(raw)
        if not parsed:
            _LOG.debug("news.article_summarizer | bad JSON | article=%s | raw=%r",
                       article_id, raw[:200])
            self._set_status(article_id, "failed_summary")
            metrics["failed"] += 1
            return

        summary_short  = parsed.get("summary_short") or ""
        summary_medium = parsed.get("summary_medium") or ""
        key_points     = json.dumps(parsed.get("key_points") or [])
        entities       = json.dumps(parsed.get("entities") or [])

        with conn_ctx(self._dsn) as conn:
            with conn.transaction():
                conn.execute(_DEACTIVATE_OLD_SUMMARY_SQL, [article_id])
                conn.execute(_INSERT_SUMMARY_SQL, [
                    article_id, run_id,
                    summary_short, summary_medium,
                    key_points, entities,
                    model,
                ])
        self._set_status(article_id, "summarized")
        metrics["summarized"] += 1

    def _set_status(self, article_id: str, status: str) -> None:
        with conn_ctx(self._dsn) as conn:
            conn.execute(_SET_SUMMARY_STATUS_SQL, [status, article_id])
