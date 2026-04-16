"""
news/images/news_image_pipeline.py

News image download, registration, and embedding pipeline.

For each article with image_status='pending', this module:
  1. Reads discovered image URLs from daily_news.metadata.image_urls
  2. Downloads each image (5 s timeout, skip on error)
  3. Computes SHA-256 of image bytes
  4. Saves to H:\\article_images\\<YYYY>\\<MM>\\<sha256[:16]>.<ext>
  5. Upserts into media_files (existing table; ON CONFLICT is idempotent)
  6. Inserts into news_article_images linking article_id → media_id
  7. Requests jina-clip-v2 embedding via mm_embed_client (port 8031)
  8. Upserts into news_image_embeddings with denormalized filter metadata
  9. Advances image_status → 'processed' | 'no_images' | 'failed_image'

Re-embedding without data loss:
  When re-processing an image that already has an embedding, the old row is
  set to is_active=False and a new row is inserted.  Article relationships
  are preserved because news_article_images is not touched.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

from news.news_settings import get_news_settings
from rag_v1.db.pg import conn_ctx
from rag_v1.wiki.mm_embed_client import MmEmbedClient
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.image_pipeline", file_name="news_agent.log")

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_FETCH_PENDING_SQL = """
SELECT
    d.article_id::text,
    d.topic_id::text,
    d.cluster_id::text,
    d.news_source,
    d.news_source_url,
    d.canonical_url,
    d.published_at,
    d.metadata
FROM daily_news d
WHERE d.image_status = 'pending'
  AND d.fetch_status IN ('fetched', 'skipped')
ORDER BY d.first_seen_at DESC
LIMIT %s
"""

_UPSERT_MEDIA_SQL = """
INSERT INTO media_files (file_path, modality, content_hash, file_size_b, width, height)
VALUES (%s, 'image', %s, %s, %s, %s)
ON CONFLICT (file_path, content_hash) DO UPDATE
    SET file_size_b = EXCLUDED.file_size_b
RETURNING media_id
"""

_UPSERT_ARTICLE_IMAGE_SQL = """
INSERT INTO news_article_images
    (article_id, media_id, source_image_url, image_role, position_index)
VALUES (%s::uuid, %s, %s, %s, %s)
ON CONFLICT (article_id, media_id) DO NOTHING
"""

_DEACTIVATE_OLD_EMBED_SQL = """
UPDATE news_image_embeddings
SET is_active = false, updated_at = now()
WHERE media_id = %s
  AND content_hash = %s
  AND is_active = true
"""

_INSERT_EMBED_SQL = """
INSERT INTO news_image_embeddings (
    media_id, article_id, cluster_id, topic_id,
    source_name, source_url, canonical_source_domain,
    image_role, published_at, embedding_model,
    embedding, content_hash, is_active
)
VALUES (
    %s, %s::uuid, %s::uuid, %s::uuid,
    %s, %s, %s,
    %s, %s, 'jina-clip-v2',
    %s, %s, true
)
"""

_SET_IMAGE_STATUS_SQL = """
UPDATE daily_news
SET image_status = %s, updated_at = now()
WHERE article_id = %s::uuid
"""

_INSERT_RUN_SQL = """
INSERT INTO news_runs (run_id, run_type, scheduled_for, started_at, status, worker_id)
VALUES (%s::uuid, 'image_processing', now(), now(), 'running', 'news_image_pipeline')
"""

_UPDATE_RUN_SQL = """
UPDATE news_runs
SET finished_at = now(), status = %s, metrics_json = %s::jsonb, error_text = %s
WHERE run_id = %s::uuid
"""

_ROLE_MAP = ["hero", "gallery", "gallery"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _image_ext(url: str, content_type: str) -> str:
    ct_map = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
              "image/gif": ".gif", "image/avif": ".avif"}
    if content_type:
        for k, v in ct_map.items():
            if k in content_type:
                return v
    path = urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"):
        if path.endswith(ext):
            return ext if ext != ".jpeg" else ".jpg"
    return ".jpg"


def _get_image_dimensions(data: bytes) -> tuple[Optional[int], Optional[int]]:
    try:
        from PIL import Image
        import io
        with Image.open(io.BytesIO(data)) as im:
            return im.width, im.height
    except Exception:
        return None, None


def _vec_to_pg_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


# ---------------------------------------------------------------------------
# NewsImagePipeline
# ---------------------------------------------------------------------------

class NewsImagePipeline:
    """
    Processes news images for pending articles.

    Typical call (from NewsScheduler):
        pipeline = NewsImagePipeline()
        result = pipeline.run_once()
    """

    def __init__(self) -> None:
        self._cfg = get_news_settings()
        self._dsn = self._cfg.pg_dsn
        self._storage_root = Path(self._cfg.image_storage_path)
        self._http = httpx.Client(
            timeout=self._cfg.image_download_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SageKaizen/1.0)"},
        )
        self._embed = MmEmbedClient(
            host="localhost",
            port=self._cfg.image_embed_port,
            timeout_s=60.0,
        )

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_once(self) -> dict:
        """Process one batch of pending articles."""
        run_id = str(uuid.uuid4())
        t0 = time.monotonic()

        with conn_ctx(self._dsn) as conn:
            conn.execute(_INSERT_RUN_SQL, [run_id])

        rows = self._fetch_pending()
        if not rows:
            self._update_run(run_id, "completed", {"processed": 0}, None)
            return {"processed": 0}

        _LOG.info("image_pipeline | start | run=%s | articles=%d", run_id, len(rows))

        metrics = {"articles_processed": 0, "images_downloaded": 0,
                   "images_embedded": 0, "no_images": 0, "errors": 0}

        for row in rows:
            try:
                self._process_article(row, metrics)
            except Exception as exc:
                _LOG.error("image_pipeline | article=%s | error=%s",
                           row["article_id"], exc, exc_info=True)
                self._set_status(row["article_id"], "failed_image")
                metrics["errors"] += 1

            metrics["articles_processed"] += 1

        metrics["duration_s"] = round(time.monotonic() - t0, 2)
        self._update_run(run_id, "completed", metrics, None)
        _LOG.info("image_pipeline | done | %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Internal: per-article
    # ------------------------------------------------------------------

    def _fetch_pending(self) -> list[dict]:
        with conn_ctx(self._dsn) as conn:
            return conn.execute(
                _FETCH_PENDING_SQL, [self._cfg.image_batch_size]
            ).fetchall()

    def _process_article(self, row: dict, metrics: dict) -> None:
        article_id = row["article_id"]
        metadata: dict = row["metadata"] or {}
        image_urls: list[str] = metadata.get("image_urls") or []

        if not image_urls:
            self._set_status(article_id, "no_images")
            metrics["no_images"] += 1
            return

        downloaded_any = False
        for i, img_url in enumerate(image_urls[:3]):
            success = self._process_one_image(row, img_url, i)
            if success:
                downloaded_any = True
                metrics["images_downloaded"] += 1
                metrics["images_embedded"] += 1

        self._set_status(article_id, "processed" if downloaded_any else "failed_image")
        if not downloaded_any:
            metrics["errors"] += 1

    def _process_one_image(self, row: dict, img_url: str, idx: int) -> bool:
        """Download, register, and embed one image. Returns True on success."""
        article_id = row["article_id"]

        try:
            resp = self._http.get(img_url)
            resp.raise_for_status()
        except Exception as exc:
            _LOG.debug("image_pipeline | download failed | url=%s | %s", img_url, exc)
            return False

        data = resp.content
        if len(data) < 1024:  # too small to be a real image
            return False

        content_hash = _sha256(data)
        sha_hex = _sha256_hex(data)
        content_type = resp.headers.get("content-type", "")
        ext = _image_ext(img_url, content_type)

        # Determine storage path.
        now = datetime.now(timezone.utc)
        rel_dir = Path(str(now.year)) / f"{now.month:02d}"
        filename = sha_hex[:16] + ext
        abs_dir = self._storage_root / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)
        file_path = str(abs_dir / filename)

        # Write to disk (idempotent: same hash → same path).
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(data)

        w, h = _get_image_dimensions(data)

        # Upsert media_files.
        with conn_ctx(self._dsn) as conn:
            mf_row = conn.execute(
                _UPSERT_MEDIA_SQL,
                [file_path, content_hash, len(data), w, h],
            ).fetchone()
            media_id = mf_row["media_id"]

            # Link article → image.
            role = _ROLE_MAP[min(idx, len(_ROLE_MAP) - 1)]
            conn.execute(
                _UPSERT_ARTICLE_IMAGE_SQL,
                [article_id, media_id, img_url, role, idx],
            )

        # Generate embedding via jina-clip-v2.
        try:
            vecs = self._embed.embed_image_bytes([data])
            vec = vecs[0] if vecs else None
        except Exception as exc:
            _LOG.debug("image_pipeline | embed failed | %s", exc)
            vec = None

        if vec:
            published_at = row.get("published_at")
            domain = urlparse(row.get("canonical_url") or "").netloc

            with conn_ctx(self._dsn) as conn:
                # Deactivate any existing embedding for this exact image.
                conn.execute(_DEACTIVATE_OLD_EMBED_SQL, [media_id, content_hash])
                conn.execute(
                    _INSERT_EMBED_SQL,
                    [
                        media_id,                       # media_id
                        article_id,                     # article_id
                        row.get("cluster_id"),          # cluster_id (nullable)
                        row.get("topic_id"),            # topic_id (nullable)
                        row.get("news_source") or "",   # source_name
                        row.get("news_source_url") or "",# source_url
                        domain,                         # canonical_source_domain
                        role,                           # image_role
                        published_at,                   # published_at
                        _vec_to_pg_literal(vec),        # embedding literal
                        content_hash,                   # content_hash
                    ],
                )

        return True

    def _set_status(self, article_id: str, status: str) -> None:
        with conn_ctx(self._dsn) as conn:
            conn.execute(_SET_IMAGE_STATUS_SQL, [status, article_id])

    def _update_run(self, run_id: str, status: str,
                    metrics: dict, error: Optional[str]) -> None:
        with conn_ctx(self._dsn) as conn:
            conn.execute(
                _UPDATE_RUN_SQL,
                [status, json.dumps(metrics), error, run_id],
            )
