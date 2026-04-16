"""
news/collectors/searxng_news_client.py

SearXNG HTTP client for the news collection pipeline.

Separate from search/searxng_client.py, which serves ad-hoc chat searches
and must not be modified.  This client is designed for the scheduled
collection pipeline and adds:

  - pageno support for deeper result sets
  - per-request categories and engine overrides
  - thread-safe rate limiting (0.5 s minimum gap between requests)
  - NewsArticleCandidate output type ready for DB upsert
  - raw SearXNG result fragment stored in raw_metadata for JSONB
  - URL canonicalization (strips tracking params, normalises scheme/host)
  - SHA-256 url_hash and simhash dedupe_fingerprint computed here

Public API:
    client = SearXNGNewsClient()
    candidates = client.search(query_text, categories=["news"], time_range="day")
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from news.news_settings import get_news_settings
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.searxng_client", file_name="news_agent.log")

# ---------------------------------------------------------------------------
# Tracking / analytics query parameters stripped before hashing.
# ---------------------------------------------------------------------------
_STRIP_PARAMS: frozenset[str] = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_reader", "fbclid", "gclid", "msclkid", "yclid",
    "wickedid", "ref", "source", "campaign", "mc_cid", "mc_eid", "_ga",
    "igshid", "s_cid", "ncid", "cmpid", "mbid",
})

# SearXNG result fields preserved verbatim in raw_metadata for JSONB.
_RAW_KEEP_FIELDS: tuple[str, ...] = (
    "url", "title", "content", "score", "publishedDate",
    "publisherName", "engines", "category", "thumbnail",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NewsArticleCandidate:
    """
    Normalised article candidate produced by one SearXNG result row.

    All fields map directly to daily_news columns.  url_hash and
    dedupe_fingerprint are computed here so the upsert layer never
    needs to re-derive them.
    """
    canonical_url: str
    url_hash: bytes                  # SHA-256(canonical_url.encode()) — bytea in DB
    headline: Optional[str]
    snippet: Optional[str]
    news_source: str                 # human-readable publisher name
    news_source_url: str             # original (non-canonical) URL from SearXNG
    published_at: Optional[datetime]
    language_code: str
    search_query: str
    search_category: str
    rank_score: float
    dedupe_fingerprint: str          # simhash hex of headline + first 300 chars of snippet
    raw_metadata: dict = field(default_factory=dict)  # raw SearXNG fragment → JSONB


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------

def canonicalize_url(raw_url: str) -> str:
    """
    Return a normalised URL for deduplication.

    Rules applied:
      - Lowercase scheme and hostname
      - Strip all tracking query parameters (_STRIP_PARAMS)
      - Remove trailing slash from path (except bare '/')
      - Drop fragment (#...)
      - Upgrade http → https when the rest of the URL is unchanged
    """
    try:
        parsed = urlparse(raw_url.strip())
        scheme = (parsed.scheme or "https").lower()
        if scheme == "http":
            scheme = "https"
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/") or "/"

        qs = parse_qs(parsed.query, keep_blank_values=False)
        clean_qs = {k: v for k, v in qs.items() if k.lower() not in _STRIP_PARAMS}
        query = urlencode(clean_qs, doseq=True) if clean_qs else ""

        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return raw_url.strip()


def url_sha256(canonical_url: str) -> bytes:
    """Return the SHA-256 digest of the canonical URL as raw bytes (for bytea)."""
    return hashlib.sha256(canonical_url.encode("utf-8")).digest()


def make_dedupe_fingerprint(headline: Optional[str], snippet: Optional[str]) -> str:
    """
    Compute a simhash fingerprint of headline + snippet prefix.

    Used for near-duplicate detection across different sources covering
    the same story.  Returns a hex string of the 64-bit simhash value.
    """
    from simhash import Simhash  # imported lazily — small startup cost
    parts = [headline or "", (snippet or "")[:300]]
    text = " ".join(p for p in parts if p)
    return hex(Simhash(text).value)


def _parse_published_at(raw: Optional[str]) -> Optional[datetime]:
    """Best-effort parse of SearXNG publishedDate into a timezone-aware datetime."""
    if not raw:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(raw.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    _LOG.debug("searxng_news | unparseable publishedDate=%r", raw)
    return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class SearXNGNewsClient:
    """
    Thread-safe synchronous HTTP client for news collection from a local
    SearXNG instance.

    One instance is typically shared across all topic-collector threads.
    The internal rate limiter (threading.Lock + monotonic timer) ensures
    the minimum gap between consecutive requests is respected even when
    multiple threads call search() concurrently.

    Usage:
        client = SearXNGNewsClient()
        candidates = client.search(
            query_text="AI news today",
            categories=["technology", "news"],
            time_range="day",
            max_results=20,
        )
    """

    def __init__(self) -> None:
        cfg = get_news_settings()
        self._base_url = cfg.searxng_base_url.rstrip("/")
        self._timeout = cfg.searxng_timeout_s
        self._min_gap = cfg.searxng_request_gap_s
        self._rate_lock = threading.Lock()
        self._last_request_ts: float = 0.0
        self._http = httpx.Client(
            timeout=self._timeout,
            follow_redirects=True,
            headers={"Accept": "application/json"},
        )
        _LOG.debug("searxng_news | client init | base_url=%s", self._base_url)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        try:
            self._http.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_rate_limit(self) -> None:
        """Block the calling thread until the minimum inter-request gap has elapsed."""
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_ts
            gap = self._min_gap - elapsed
            if gap > 0:
                time.sleep(gap)
            self._last_request_ts = time.monotonic()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    def _do_get(self, params: dict) -> dict:
        """Issue one GET /search request; retries on transient transport errors."""
        self._wait_for_rate_limit()
        resp = self._http.get(f"{self._base_url}/search", params=params)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_source(result: dict, canonical: str) -> str:
        """Best-effort publisher name from a SearXNG result dict."""
        for key in ("publisherName", "source", "engine"):
            val = result.get(key)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        return urlparse(canonical).netloc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        categories: list[str] | None = None,
        engines: list[str] | None = None,
        time_range: str = "day",
        max_results: int = 20,
        pageno: int = 1,
        language: str = "en",
    ) -> list[NewsArticleCandidate]:
        """
        Submit one search to SearXNG and return normalised candidates.

        Args:
            query_text:   The search query string.
            categories:   SearXNG category list, e.g. ["news", "technology"].
            engines:      Optional engine override list, e.g. ["bing news"].
            time_range:   SearXNG time filter: "day" | "week" | "month" | "year" | "".
            max_results:  Cap on candidates returned (SearXNG may return fewer).
            pageno:       Result page number (1-indexed).
            language:     Two-letter language code passed to SearXNG.

        Returns:
            List of NewsArticleCandidate objects, possibly empty on error.
        """
        params: dict[str, object] = {
            "q": query_text,
            "format": "json",
            "pageno": pageno,
            "language": language,
        }
        if categories:
            params["categories"] = ",".join(categories)
        if engines:
            params["engines"] = ",".join(engines)
        if time_range:
            params["time_range"] = time_range

        try:
            raw = self._do_get(params)
        except Exception as exc:
            _LOG.warning(
                "searxng_news | search failed | query=%r | error=%s",
                query_text, exc,
            )
            return []

        results: list[dict] = raw.get("results") or []
        candidates: list[NewsArticleCandidate] = []
        seen_hashes: set[bytes] = set()

        for i, r in enumerate(results[:max_results]):
            raw_url: str = (r.get("url") or r.get("href") or "").strip()
            if not raw_url:
                continue

            canonical = canonicalize_url(raw_url)
            uhash = url_sha256(canonical)

            # Deduplicate within a single search response.
            if uhash in seen_hashes:
                continue
            seen_hashes.add(uhash)

            headline = (r.get("title") or "").strip() or None
            snippet  = (r.get("content") or r.get("snippet") or "").strip() or None

            candidates.append(NewsArticleCandidate(
                canonical_url      = canonical,
                url_hash           = uhash,
                headline           = headline,
                snippet            = snippet,
                news_source        = self._extract_source(r, canonical),
                news_source_url    = raw_url,
                published_at       = _parse_published_at(r.get("publishedDate")),
                language_code      = language,
                search_query       = query_text,
                search_category    = (categories[0] if categories else "general"),
                rank_score         = float(r.get("score") or (max_results - i)),
                dedupe_fingerprint = make_dedupe_fingerprint(headline, snippet),
                raw_metadata       = {k: r[k] for k in _RAW_KEEP_FIELDS if k in r},
            ))

        _LOG.debug(
            "searxng_news | query=%r | time_range=%s | page=%d | returned=%d | deduped=%d",
            query_text, time_range, pageno, len(results), len(candidates),
        )
        return candidates
