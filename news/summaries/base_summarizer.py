"""
news/summaries/base_summarizer.py

Shared infrastructure for all news pipeline summarization jobs.

Provides a concrete abstract base class (BasePipelineJob) that eliminates the
copy-paste duplication across ArticleSummarizer, ClusterSummarizer, and
BriefFinalizer.  Each subclass inherits:

  _cfg / _dsn         — NewsSettings and PostgreSQL DSN
  _is_off_peak()      — off-peak guard (chat-idle check)
  _get_model()        — brain model discovery with process-wide cache (M3 fix)
  _call_brain()       — stream brain, strip <think> blocks, return text
  _parse_json()       — extract first JSON object from brain output
  _begin_run()        — INSERT a running news_runs row
  _finish_run()       — UPDATE news_runs to completed/failed

Design notes
------------
ABC is used here because:
  1. Every subclass MUST implement run_once() — failing to do so is a
     programming error, not a runtime condition.  ABC raises at instantiation
     time rather than at first call, catching mistakes earlier.
  2. The other methods are fully concrete and should never be overridden
     without reason; ABC's @abstractmethod on run_once() makes that contract
     explicit without blocking inheritance of the shared implementation.

Compiled regex
--------------
_THINK_RE is compiled once at module import.  All three subclasses previously
called re.sub(..., flags=re.DOTALL) with a string pattern on every brain
response.  The re module caches compiled patterns internally, but calling
re.sub() with a string still pays a dict-lookup on every call.  Binding a
compiled pattern at module level eliminates that overhead across thousands of
summarization calls per day.

Process-wide model cache
------------------------
_get_cached_model_id() is a module-level function that caches the brain's
model alias in _MODEL_CACHE keyed by URL.  Previously each summarizer instance
called discover_model_id() on first use.  Because the scheduler creates a
fresh instance per job execution, this was an HTTP round-trip (3-5 s) on every
collection cycle.  With the module-level cache, it fires once per URL per
process lifetime regardless of how many instances are created.

Thread safety: _MODEL_CACHE uses double-checked locking.  Dict reads are
atomic under CPython's GIL (dict.__getitem__ is a single bytecode op).
Writes are protected by _MODEL_CACHE_LOCK to prevent duplicate HTTP calls
during parallel job starts.  This pattern remains safe under Python 3.13+
free-threading because dict operations have internal per-bucket locks.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional

from openai_client import HttpTimeouts, discover_model_id, stream_chat_completions
from news.news_settings import get_news_settings
from rag_v1.db.pg import conn_ctx
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.base_summarizer", file_name="news_agent.log")

# ---------------------------------------------------------------------------
# Compiled regex — strips <think>…</think> blocks from all brain responses.
# Compiled once at module load; reused by every _call_brain() invocation.
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# ---------------------------------------------------------------------------
# Process-wide model ID cache
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, str] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _get_cached_model_id(url: str, fallback: str) -> str:
    """
    Return the model alias for the brain at *url*, discovering it on first call.

    The result is cached in _MODEL_CACHE for the lifetime of the process.
    Subsequent calls for the same URL return immediately without any I/O.

    Args:
        url:      Base URL of the llama-server instance (e.g. "http://127.0.0.1:8011").
        fallback: Alias to use when discovery returns nothing (server unreachable
                  or /v1/models returns an empty list).
    """
    if url in _MODEL_CACHE:
        return _MODEL_CACHE[url]
    with _MODEL_CACHE_LOCK:
        if url not in _MODEL_CACHE:  # double-checked
            _MODEL_CACHE[url] = (
                discover_model_id(url, timeouts=HttpTimeouts(3.0, 5.0)) or fallback
            )
    return _MODEL_CACHE[url]


# ---------------------------------------------------------------------------
# Shared SQL — run tracking
# ---------------------------------------------------------------------------

_BEGIN_RUN_SQL = """
INSERT INTO news_runs (run_id, run_type, scheduled_for, started_at, status, worker_id)
VALUES (%s::uuid, %s, now(), now(), 'running', %s)
"""

_BEGIN_RUN_WITH_PROFILE_SQL = """
INSERT INTO news_runs (
    run_id, run_type, profile_id, scheduled_for, started_at, status, worker_id
) VALUES (
    %s::uuid, %s, %s::uuid, now(), now(), 'running', %s
)
"""

_FINISH_RUN_SQL = """
UPDATE news_runs
SET finished_at = now(), status = %s, metrics_json = %s::jsonb, error_text = %s
WHERE run_id = %s::uuid
"""


# ---------------------------------------------------------------------------
# BasePipelineJob
# ---------------------------------------------------------------------------

class BasePipelineJob(ABC):
    """
    Abstract base class for news pipeline summarization and finalization jobs.

    Subclasses must implement:
        run_once(force=False) -> dict

    All other methods are concrete and intended to be used as-is.  Override
    only if a specific job has a genuinely different requirement.
    """

    def __init__(self) -> None:
        self._cfg = get_news_settings()
        self._dsn = self._cfg.pg_dsn

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run_once(self, force: bool = False) -> dict:
        """
        Execute one job cycle.

        Args:
            force: When True, bypass the off-peak guard and run regardless of
                   current chat activity.  Used by the manual "Get News"
                   pipeline trigger in the Streamlit UI.

        Returns:
            Metrics dict (e.g. {"summarized": 5, "failed": 1, "duration_s": 12.3}).
        """
        ...

    # ------------------------------------------------------------------
    # Off-peak guard
    # ------------------------------------------------------------------

    def _is_off_peak(self) -> bool:
        """
        Return True when the user has been idle long enough for background work.

        Checks last_chat_activity_ts() from chat_service.  Defaults to True
        (off-peak) when chat_service is unavailable (e.g. during unit tests
        or if the import fails), so pipeline jobs run rather than silently
        blocking.

        The import is deferred to avoid a circular dependency:
          chat_service → context_injector → news_resolver → news pipeline
        """
        try:
            from chat_service import last_chat_activity_ts
            return (time.time() - last_chat_activity_ts()) > self._cfg.off_peak_idle_seconds
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Brain interaction
    # ------------------------------------------------------------------

    def _get_model(self, url: str, fallback: str = "brain") -> str:
        """Return the model alias for the brain at *url* (cached process-wide)."""
        return _get_cached_model_id(url, fallback)

    def _call_brain(
        self,
        url: str,
        model: str,
        system: str,
        prompt: str,
        *,
        max_tokens: int,
        timeouts: HttpTimeouts,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        """
        Stream a completion from the brain at *url* and return the full text.

        <think>…</think> blocks are stripped from the output so callers always
        receive clean content regardless of which brain (FAST or ARCHITECT) was
        invoked.  The FAST brain (Qwen2.5-Omni) does not emit thinking tokens,
        but the stripping is applied defensively.

        Args:
            url:         Base URL of the llama-server instance.
            model:       Model alias (from _get_model / _get_cached_model_id).
            system:      System prompt string.
            prompt:      User-turn content.
            max_tokens:  Hard ceiling on generated tokens.
            timeouts:    HttpTimeouts(connect_s, read_s) for this request.
            temperature: Sampling temperature (default 0.2 — factual/low-entropy).
            top_p:       Nucleus sampling threshold (default 0.9).

        Raises:
            LlamaServerError: on HTTP 4xx/5xx from llama-server.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
        raw = "".join(stream_chat_completions(
            base_url=url,
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeouts=timeouts,
        ))
        return _THINK_RE.sub("", raw).strip()

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        """
        Extract the first JSON object from a brain response.

        Tolerates surrounding prose, markdown code fences, and minor formatting
        noise by scanning for the outermost { … } span and parsing only that.
        Returns None if no valid JSON object is found.

        This method is a @staticmethod — it does not touch instance or class
        state and can be called as BasePipelineJob._parse_json(raw) in tests.
        """
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Run tracking
    # ------------------------------------------------------------------

    def _begin_run(
        self,
        run_id: str,
        run_type: str,
        worker_id: str,
        profile_id: Optional[str] = None,
    ) -> None:
        """
        INSERT a 'running' row into news_runs.

        Args:
            run_id:     UUID string for this run.
            run_type:   e.g. 'article_summarization', 'cluster_summarization',
                        'brief_finalization'.
            worker_id:  Identifies the worker class (e.g. 'article_summarizer').
            profile_id: Optional UUID string — only used by brief_finalization
                        runs which are scoped to a single news_profiles row.
        """
        if profile_id is not None:
            sql    = _BEGIN_RUN_WITH_PROFILE_SQL
            params = [run_id, run_type, profile_id, worker_id]
        else:
            sql    = _BEGIN_RUN_SQL
            params = [run_id, run_type, worker_id]
        with conn_ctx(self._dsn) as conn:
            conn.execute(sql, params)

    def _finish_run(
        self,
        run_id: str,
        status: str,
        metrics: dict,
        error: Optional[str] = None,
    ) -> None:
        """
        Mark a news_runs row as completed or failed.

        Args:
            run_id:  UUID string of the run to update.
            status:  'completed' or 'failed'.
            metrics: Arbitrary metrics dict — serialised to JSONB.
            error:   Error message string if status='failed', else None.
        """
        with conn_ctx(self._dsn) as conn:
            conn.execute(
                _FINISH_RUN_SQL,
                [status, json.dumps(metrics), error, run_id],
            )
