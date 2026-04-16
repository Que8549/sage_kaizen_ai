"""
news/enrichment/article_enricher.py

Article enrichment pipeline.

For each article with fetch_status='pending', this module:
  1. Fetches the full HTML with httpx (10 s timeout, 3 retries)
  2. Extracts headline (OG/title tag) and published_at (OG/meta) if missing
  3. Converts body HTML to plain text with html2text
  4. Discovers image candidate URLs (og:image, twitter:image, first large img)
     and stores them in metadata.image_urls for the image pipeline
  5. Embeds headline + snippet via BGE-M3 (port 8020) → article_embedding
  6. Updates daily_news and advances fetch_status

State machine:
    pending → fetching → fetched       (success)
                       → failed_fetch  (retryable; retry_count < max_retries)
                       → skipped       (paywall / bot-block / empty body)

A news_runs row (run_type='enrichment') is created per batch run for auditing.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from news.news_settings import get_news_settings
from pg_settings import PgSettings
from rag_v1.db.pg import conn_ctx
from rag_v1.embed.embed_client import EmbedClient
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.enricher", file_name="news_agent.log")

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_FETCH_PENDING_SQL = """
SELECT
    article_id::text,
    canonical_url,
    headline,
    snippet,
    news_source,
    (metadata->>'fetch_retry_count')::int AS retry_count
FROM daily_news
WHERE fetch_status = 'pending'
   OR (fetch_status = 'failed_fetch'
       AND (metadata->>'fetch_retry_count')::int < %s)
ORDER BY first_seen_at DESC
LIMIT %s
"""

_SET_FETCHING_SQL = """
UPDATE daily_news
SET fetch_status = 'fetching', updated_at = now()
WHERE article_id = %s::uuid
"""

_UPDATE_ARTICLE_SQL = """
UPDATE daily_news SET
    fetch_status        = %s,
    article_content     = %s,
    headline            = COALESCE(%s, headline),
    published_at        = COALESCE(%s, published_at),
    article_embedding   = %s,
    image_status        = CASE
                            WHEN %s = 'fetched' AND jsonb_array_length(COALESCE(%s::jsonb, '[]'::jsonb)) > 0
                            THEN 'pending'
                            WHEN %s = 'fetched'
                            THEN 'no_images'
                            ELSE image_status
                          END,
    metadata            = metadata || %s::jsonb,
    updated_at          = now()
WHERE article_id = %s::uuid
"""

_INSERT_RUN_SQL = """
INSERT INTO news_runs (run_id, run_type, scheduled_for, started_at, status, worker_id)
VALUES (%s::uuid, 'enrichment', now(), now(), 'running', 'article_enricher')
"""

_UPDATE_RUN_SQL = """
UPDATE news_runs
SET finished_at = now(), status = %s, metrics_json = %s::jsonb, error_text = %s
WHERE run_id = %s::uuid
"""

# Minimum non-trivial body text length after extraction.
_MIN_BODY_CHARS = 300
# Image tags we look for (in priority order).
_OG_IMAGE_PROPS = ("og:image", "twitter:image", "og:image:secure_url")
# Headers that look like a browser to reduce bot-blocks.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _extract_og_tag(soup: BeautifulSoup, prop: str) -> Optional[str]:
    tag = soup.find("meta", attrs={"property": prop}) or \
          soup.find("meta", attrs={"name": prop})
    if tag:
        return (tag.get("content") or "").strip() or None
    return None


def _extract_headline(soup: BeautifulSoup) -> Optional[str]:
    for prop in ("og:title", "twitter:title"):
        v = _extract_og_tag(soup, prop)
        if v:
            return v
    tag = soup.find("h1")
    if tag:
        return tag.get_text(strip=True) or None
    title = soup.find("title")
    if title:
        return title.get_text(strip=True) or None
    return None


def _extract_published_at(soup: BeautifulSoup) -> Optional[datetime]:
    for prop in ("article:published_time", "og:article:published_time",
                 "datePublished", "pubdate"):
        v = _extract_og_tag(soup, prop)
        if v:
            for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z",
                        "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(v[:25], fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue
    return None


def _extract_image_urls(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Return up to 3 candidate image URLs (OG first, then first large img)."""
    urls: list[str] = []

    for prop in _OG_IMAGE_PROPS:
        v = _extract_og_tag(soup, prop)
        if v:
            urls.append(v if v.startswith("http") else urljoin(base_url, v))
            break

    if len(urls) < 2:
        for img in soup.find_all("img", src=True):
            src = img.get("src", "").strip()
            if not src or src.startswith("data:"):
                continue
            try:
                w = int(img.get("width") or 0)
                h = int(img.get("height") or 0)
                if w > 0 and w < 150:
                    continue
                if h > 0 and h < 100:
                    continue
            except (ValueError, TypeError):
                pass
            full = src if src.startswith("http") else urljoin(base_url, src)
            if full not in urls:
                urls.append(full)
            if len(urls) >= 3:
                break

    return urls


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text via html2text, stripping links and images."""
    try:
        import html2text as h2t
        h = h2t.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.ignore_emphasis = False
        h.body_width = 0
        return h.handle(html).strip()
    except Exception:
        # Fallback: BeautifulSoup plain text extraction
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed_text(text: str, embed_client: EmbedClient) -> Optional[list[float]]:
    """Return a 1024-dim BGE-M3 embedding, or None on failure."""
    if not text.strip():
        return None
    try:
        vecs = embed_client.embed([text[:2048]])
        return vecs[0] if vecs else None
    except Exception as exc:
        _LOG.debug("enricher | embed failed: %s", exc)
        return None


def _vec_to_pg_literal(vec: list[float]) -> str:
    """Convert a float list to a pgvector string literal '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


# ---------------------------------------------------------------------------
# ArticleEnricher
# ---------------------------------------------------------------------------

class ArticleEnricher:
    """
    Fetches full article text and metadata for pending daily_news rows.

    Typical call (from NewsScheduler):
        enricher = ArticleEnricher()
        result = enricher.run_once()
    """

    def __init__(self) -> None:
        self._cfg = get_news_settings()
        self._dsn = self._cfg.pg_dsn
        self._http = httpx.Client(
            timeout=self._cfg.fetch_timeout_s,
            follow_redirects=True,
            headers=_HEADERS,
        )
        self._embed = EmbedClient(
            base_url=self._cfg.bge_m3_embed_url,
            model=self._cfg.bge_m3_model,
            timeout_s=60.0,
        )

    def close(self) -> None:
        try:
            self._http.close()
            self._embed.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_once(self) -> dict:
        """
        Enrich one batch of pending articles.

        Returns a metrics dict suitable for news_runs.metrics_json.
        """
        run_id = str(uuid.uuid4())
        t0 = time.monotonic()

        with conn_ctx(self._dsn) as conn:
            conn.execute(_INSERT_RUN_SQL, [run_id])

        rows = self._fetch_pending()
        if not rows:
            self._update_run(run_id, "completed", {"processed": 0}, None)
            return {"processed": 0}

        _LOG.info("enricher | start | run=%s | articles=%d", run_id, len(rows))

        metrics = {"fetched": 0, "skipped": 0, "failed": 0, "embedded": 0}

        for row in rows:
            self._enrich_one(row, metrics)

        metrics["duration_s"] = round(time.monotonic() - t0, 2)
        self._update_run(run_id, "completed", metrics, None)
        _LOG.info("enricher | done | %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_pending(self) -> list[dict]:
        with conn_ctx(self._dsn) as conn:
            return conn.execute(
                _FETCH_PENDING_SQL,
                [self._cfg.fetch_max_retries, self._cfg.fetch_batch_size],
            ).fetchall()

    def _enrich_one(self, row: dict, metrics: dict) -> None:
        article_id = row["article_id"]
        url = row["canonical_url"]
        retry_count = int(row["retry_count"] or 0)

        # Mark as in-flight so a concurrent run doesn't pick it up.
        with conn_ctx(self._dsn) as conn:
            conn.execute(_SET_FETCHING_SQL, [article_id])

        try:
            resp = self._http.get(url)
        except Exception as exc:
            _LOG.debug("enricher | fetch error | url=%s | %s", url, exc)
            self._mark_failed(article_id, retry_count, str(exc))
            metrics["failed"] += 1
            return

        # Detect hard blocks before parsing.
        if resp.status_code in (403, 429, 451):
            self._mark_skipped(article_id, f"HTTP {resp.status_code}")
            metrics["skipped"] += 1
            return

        if resp.status_code >= 400:
            self._mark_failed(article_id, retry_count, f"HTTP {resp.status_code}")
            metrics["failed"] += 1
            return

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        body_text = _html_to_text(html)
        if len(body_text) < _MIN_BODY_CHARS:
            self._mark_skipped(article_id, "body_too_short")
            metrics["skipped"] += 1
            return

        headline    = _extract_headline(soup)
        published_at = _extract_published_at(soup)
        image_urls  = _extract_image_urls(soup, url)

        # Build embedding text from headline + existing snippet.
        embed_text = " ".join(filter(None, [
            headline or row.get("headline") or "",
            (row.get("snippet") or "")[:400],
        ])).strip()

        vec = _embed_text(embed_text, self._embed)
        vec_literal = _vec_to_pg_literal(vec) if vec else None
        if vec:
            metrics["embedded"] += 1

        img_json = json.dumps(image_urls)
        meta_patch = json.dumps({
            "fetch_retry_count": retry_count,
            "image_urls": image_urls,
        })

        with conn_ctx(self._dsn) as conn:
            conn.execute(
                _UPDATE_ARTICLE_SQL,
                [
                    "fetched",          # fetch_status
                    body_text,          # article_content
                    headline,           # headline (COALESCE)
                    published_at,       # published_at (COALESCE)
                    vec_literal,        # article_embedding
                    "fetched",          # for image_status CASE condition
                    img_json,           # jsonb_array_length check
                    "fetched",          # second reference in CASE
                    meta_patch,         # metadata patch
                    article_id,         # WHERE
                ],
            )

        metrics["fetched"] += 1

    def _mark_failed(self, article_id: str, retry_count: int, reason: str) -> None:
        meta = json.dumps({"fetch_retry_count": retry_count + 1, "fetch_error": reason})
        with conn_ctx(self._dsn) as conn:
            conn.execute(
                "UPDATE daily_news SET fetch_status='failed_fetch', "
                "metadata = metadata || %s::jsonb, updated_at=now() "
                "WHERE article_id = %s::uuid",
                [meta, article_id],
            )

    def _mark_skipped(self, article_id: str, reason: str) -> None:
        meta = json.dumps({"skip_reason": reason})
        with conn_ctx(self._dsn) as conn:
            conn.execute(
                "UPDATE daily_news SET fetch_status='skipped', "
                "metadata = metadata || %s::jsonb, updated_at=now() "
                "WHERE article_id = %s::uuid",
                [meta, article_id],
            )

    def _update_run(self, run_id: str, status: str,
                    metrics: dict, error: Optional[str]) -> None:
        with conn_ctx(self._dsn) as conn:
            conn.execute(
                _UPDATE_RUN_SQL,
                [status, json.dumps(metrics), error, run_id],
            )
