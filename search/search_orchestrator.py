"""
search/search_orchestrator.py

Sits between the context injector and the SearXNG HTTP client.
Responsibilities:
  - Apply min-score filter (with a safe fallback so results are never silently lost).
  - Sort by score descending.
  - Truncate to max_results based on which brain is handling the turn.
  - Auto-select time_range: news queries default to "week"; others are all-time.
  - Expose a lazy singleton so the client is constructed once and reused.

Env vars
--------
SAGE_SEARCH_ENABLED          true/false, default true
SAGE_SEARCH_URL              SearXNG base URL, default http://localhost:8080
SAGE_SEARCH_TIMEOUT_S        HTTP timeout in seconds, default 8
SAGE_SEARCH_SNIPPET_CHARS    max chars per snippet, default 300
SAGE_SEARCH_MAX_RESULTS_FAST max results for FAST brain, default 6
SAGE_SEARCH_MAX_RESULTS_ARCH max results for ARCHITECT brain, default 12
SAGE_SEARCH_MIN_SCORE        min score threshold (0.0 = no filter), default 0.0
SAGE_SEARCH_NEWS_TIME_RANGE  time_range value for news queries, default week
"""
from __future__ import annotations

import os

from search.models import SearchEvidence, WebResult
from search.searxng_client import SearXNGClient
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.search.orchestrator")


def _env_bool(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v.strip())
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v.strip())
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


class SearchOrchestrator:
    """
    High-level search coordinator.

    Usage (lazy singleton via get_orchestrator()):

        evidence = get_orchestrator().search(
            query="latest AI news",
            categories=["news", "technology"],
            brain="FAST",
        )
        if not evidence.empty:
            # pass to Summarizer then inject into prompt
    """

    def __init__(
        self,
        searxng_url: str | None = None,
        timeout_seconds: float | None = None,
        min_score: float | None = None,
        max_results_fast: int | None = None,
        max_results_architect: int | None = None,
        snippet_max_chars: int | None = None,
        news_time_range: str | None = None,
    ) -> None:
        self._client = SearXNGClient(
            base_url        = searxng_url,
            timeout_s       = timeout_seconds,
            snippet_max_chars = snippet_max_chars,
        )
        self._min_score        = min_score        or _env_float("SAGE_SEARCH_MIN_SCORE",        0.0)
        self._max_fast         = max_results_fast or _env_int("SAGE_SEARCH_MAX_RESULTS_FAST",   6)
        self._max_arch         = max_results_architect or _env_int("SAGE_SEARCH_MAX_RESULTS_ARCH", 12)
        self._news_time_range  = news_time_range  or _env_str("SAGE_SEARCH_NEWS_TIME_RANGE",    "week")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        categories: list[str],
        brain: str = "FAST",
        time_range: str | None = None,
    ) -> SearchEvidence:
        """
        Fetch, filter, rank, and truncate search results for one turn.

        Parameters
        ----------
        query      : Raw user query text.
        categories : SearXNG category names to search.
        brain      : "FAST" | "ARCHITECT" — controls max_results ceiling.
        time_range : Explicit override. None = auto (news→week, else all-time).

        Returns SearchEvidence; evidence.empty == True on any failure.
        """
        if not query.strip() or not categories:
            return self._empty(query, categories)

        # Auto time_range: default news category to "week" for recency
        if time_range is None and "news" in categories:
            time_range = self._news_time_range

        evidence = self._client.search(query, categories, time_range=time_range)

        if evidence.empty:
            return evidence

        # Filter by min_score; if everything is filtered out keep top-3
        # (SearXNG legitimately returns 0.0 for some engines, so a hard
        # threshold would silently discard valid results.)
        filtered: list[WebResult] = [r for r in evidence.results if r.score >= self._min_score]
        if not filtered:
            filtered = list(evidence.results)

        # Sort by score descending and apply per-brain ceiling
        max_k = self._max_fast if brain == "FAST" else self._max_arch
        filtered.sort(key=lambda r: r.score, reverse=True)
        filtered = filtered[:max_k]

        _LOG.info(
            "orchestrator | brain=%s | categories=%s | time_range=%s | "
            "raw=%d filtered=%d returned=%d",
            brain, categories, time_range,
            len(evidence.results), len(filtered), len(filtered),
        )

        return SearchEvidence(
            query             = evidence.query,
            results           = tuple(filtered),
            fetched_at        = evidence.fetched_at,
            categories_queried= evidence.categories_queried,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty(query: str, categories: list[str]) -> SearchEvidence:
        from datetime import datetime, timezone
        return SearchEvidence(
            query             = query,
            results           = (),
            fetched_at        = datetime.now(timezone.utc).isoformat(),
            categories_queried= tuple(categories),
        )


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_orchestrator: SearchOrchestrator | None = None


def get_orchestrator() -> SearchOrchestrator:
    """Return the process-wide SearchOrchestrator, creating it on first call."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SearchOrchestrator()
    return _orchestrator
