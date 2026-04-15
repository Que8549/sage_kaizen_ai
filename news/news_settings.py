"""
news/news_settings.py

Centralised configuration for the Sage Kaizen News Runtime.

All values can be overridden via environment variables prefixed NEWS_
or via the project-root .env file (loaded by pydantic-settings).

Example .env additions:
    NEWS_IMAGE_STORAGE_PATH=H:\\article_images
    NEWS_COLLECTION_INTERVAL_MINUTES=60
    NEWS_BRIEF_FRESHNESS_HOURS=4
    NEWS_FETCH_FULL_TEXT=true
    NEWS_SUMMARIZATION_CONCURRENCY=2
    NEWS_CLUSTER_EPS=0.25

Usage:
    from news.news_settings import get_news_settings
    cfg = get_news_settings()
    print(cfg.image_storage_path)
"""
from __future__ import annotations

import threading
from functools import cached_property

from pg_settings import PgSettings
from pydantic import field_validator
from pydantic_settings import SettingsConfigDict


class NewsSettings(PgSettings):
    """
    News runtime configuration.

    Inherits PgSettings so PostgreSQL DSN is available on the same object.
    Add NEWS_ env vars to override any field without touching this file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="NEWS_",
    )

    # ── Image storage ──────────────────────────────────────────────────────────
    image_storage_path: str = r"H:\article_images"
    # Sub-directory layout: <image_storage_path>\<YYYY>\<MM>\<sha256_hex[:16]>.<ext>

    # ── SearXNG ────────────────────────────────────────────────────────────────
    searxng_base_url: str = "http://localhost:8080"
    searxng_timeout_s: float = 10.0
    # Minimum gap between consecutive SearXNG requests from the news collector.
    searxng_request_gap_s: float = 0.5

    # ── Scheduler intervals ────────────────────────────────────────────────────
    collection_interval_minutes: int = 60
    enrichment_interval_minutes: int = 30
    image_interval_minutes: int = 45
    clustering_interval_hours: int = 2
    summarization_interval_minutes: int = 30
    daily_brief_hour: int = 6       # 06:00 local time
    daily_brief_minute: int = 0
    rolling_brief_hour: int = 6     # 06:30 local time
    rolling_brief_minute: int = 30
    reconciliation_hour: int = 3    # 03:00 local time

    # ── Query serving ──────────────────────────────────────────────────────────
    # How long a finalised brief is considered fresh enough to serve from DB.
    brief_freshness_hours: int = 4

    # ── Article enrichment ─────────────────────────────────────────────────────
    fetch_full_text: bool = True
    fetch_timeout_s: float = 10.0
    fetch_max_retries: int = 3
    fetch_batch_size: int = 20      # max articles enriched per scheduler run

    # ── Clustering ────────────────────────────────────────────────────────────
    # DBSCAN cosine-distance threshold. 0.25 ≈ 0.75 cosine similarity.
    cluster_eps: float = 0.25
    cluster_min_samples: int = 2
    # Only cluster articles seen within this window (keeps DBSCAN input small).
    cluster_window_hours: int = 48

    # ── Summarization ──────────────────────────────────────────────────────────
    # Max concurrent FAST-brain summarization calls (avoids drowning live chat).
    summarization_concurrency: int = 2
    # Summarization only runs when no chat turn has fired within this window.
    off_peak_idle_seconds: int = 180

    # ── Image pipeline ────────────────────────────────────────────────────────
    image_download_timeout_s: float = 5.0
    image_embed_port: int = 8031    # jina-clip-v2 embed service (CUDA0)
    image_batch_size: int = 16

    # ── Scheduler safety ──────────────────────────────────────────────────────
    # If a brief_finalization run was started within this window, skip the new one.
    finalizer_lock_window_minutes: int = 120
    scheduler_max_workers: int = 4

    # ── Brain URLs (match brains.yaml ports) ─────────────────────────────────
    fast_brain_url: str = "http://127.0.0.1:8011"
    architect_brain_url: str = "http://127.0.0.1:8012"
    bge_m3_embed_url: str = "http://127.0.0.1:8020/v1"
    bge_m3_model: str = "bge-m3-embed"

    # ── Market data ───────────────────────────────────────────────────────────
    # yfinance: how many days of history to return for "recent price" questions.
    market_history_days: int = 7

    @field_validator("collection_interval_minutes", "enrichment_interval_minutes",
                     "image_interval_minutes", mode="before")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("interval must be >= 1")
        return v

    @field_validator("cluster_eps", mode="before")
    @classmethod
    def _valid_eps(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("cluster_eps must be between 0 and 1 (exclusive)")
        return v

    @cached_property
    def image_embed_base_url(self) -> str:
        return f"http://localhost:{self.image_embed_port}"


# ---------------------------------------------------------------------------
# Module-level lazy singleton — one instance per process.
# ---------------------------------------------------------------------------
_settings_instance: NewsSettings | None = None
_settings_lock = threading.Lock()


def get_news_settings() -> NewsSettings:
    """Return the process-wide NewsSettings singleton."""
    global _settings_instance
    if _settings_instance is None:
        with _settings_lock:
            if _settings_instance is None:
                _settings_instance = NewsSettings()
    return _settings_instance
