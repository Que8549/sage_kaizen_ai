"""
search/searxng_client.py

Thin HTTP client for a private SearXNG instance's JSON search API.

Endpoint: GET {base_url}/search?q=...&format=json&categories=...
Response: {"results": [{url, title, content, engine, category, score,
                        publishedDate?, author?}], ...}

SearXNG notes
-------------
- JSON format must be enabled in settings.yml under search.formats.
- method: POST in settings.yml is for the browser UI form; this client
  uses GET which SearXNG always accepts for the /search endpoint.
- The `score` field varies by engine; 0.0 is valid (not an error).
- `publishedDate` is ISO-8601 when present; absent for non-news results.

Env vars (override via environment or .env)
-------------------------------------------
SAGE_SEARCH_URL          base URL of the SearXNG instance (default: http://localhost:8080)
SAGE_SEARCH_TIMEOUT_S    per-request timeout in seconds   (default: 8)
SAGE_SEARCH_SNIPPET_CHARS max chars per snippet           (default: 300)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from search.models import SearchEvidence, WebResult
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.search.client")


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


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


class SearXNGClient:
    """
    Calls the SearXNG JSON search API and returns a normalized SearchEvidence.

    Deduplicates results by URL within the response.
    Truncates snippets to snippet_max_chars.
    Retries once on transient network errors (httpx.TransportError).
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float | None = None,
        snippet_max_chars: int | None = None,
    ) -> None:
        self._base_url = (base_url or _env_str("SAGE_SEARCH_URL", "http://localhost:8080")).rstrip("/")
        self._timeout  = timeout_s or _env_float("SAGE_SEARCH_TIMEOUT_S", 8.0)
        self._snip_max = snippet_max_chars or _env_int("SAGE_SEARCH_SNIPPET_CHARS", 300)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        categories: list[str],
        time_range: str | None = None,
        engines: list[str] | None = None,
    ) -> SearchEvidence:
        """
        Query SearXNG and return a SearchEvidence with normalized results.

        Parameters
        ----------
        query       : User query string.
        categories  : SearXNG category names (e.g. ["news", "technology"]).
        time_range  : "day" | "week" | "month" | "year" | None (all-time).
        engines     : Restrict to specific engine names (None = use all
                      engines assigned to the requested categories).

        Returns an empty SearchEvidence (evidence.empty == True) on any error
        so callers never need to guard against exceptions.
        """
        fetched_at = datetime.now(timezone.utc).isoformat()
        try:
            return self._search_with_retry(query, categories, time_range, engines, fetched_at)
        except Exception:
            _LOG.exception("SearXNG search failed for query=%r", query[:80])
            return SearchEvidence(
                query=query,
                results=(),
                fetched_at=fetched_at,
                categories_queried=tuple(categories),
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(2),
        wait=wait_fixed(1),
        reraise=True,
    )
    def _search_with_retry(
        self,
        query: str,
        categories: list[str],
        time_range: str | None,
        engines: list[str] | None,
        fetched_at: str,
    ) -> SearchEvidence:
        params: dict[str, str] = {
            "q":          query,
            "format":     "json",
            "categories": ",".join(categories),
            "language":   "en",
        }
        if time_range:
            params["time_range"] = time_range
        if engines:
            params["engines"] = ",".join(engines)

        resp = httpx.get(
            f"{self._base_url}/search",
            params=params,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        raw = data.get("results", [])
        _LOG.info(
            "searxng | query=%r | categories=%s | time_range=%s | raw_results=%d",
            query[:60], categories, time_range, len(raw),
        )

        seen_urls: set[str] = set()
        results: list[WebResult] = []

        for r in raw:
            url = (r.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            snippet = (r.get("content") or "").strip()
            if len(snippet) > self._snip_max:
                snippet = snippet[:self._snip_max] + "..."

            results.append(WebResult(
                title         = (r.get("title") or "").strip(),
                url           = url,
                snippet       = snippet,
                source_engine = r.get("engine", ""),
                category      = r.get("category", categories[0] if categories else "general"),
                score         = float(r.get("score") or 0.0),
                published_date= r.get("publishedDate"),
            ))

        return SearchEvidence(
            query             = query,
            results           = tuple(results),
            fetched_at        = fetched_at,
            categories_queried= tuple(categories),
        )
