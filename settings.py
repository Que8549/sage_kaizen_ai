"""
settings.py

Startup configuration for the Sage Kaizen main application.

All values are loaded once at import time from:
  1. .env file at the project root  (highest-priority overrides)
  2. OS environment variables
  3. Default values defined below

This module uses pydantic-settings BaseSettings — the same pattern as
pg_settings.py and rag_v1/config/rag_settings.py — so all three startup-config
modules follow one consistent approach.

For per-call runtime flags (RAG on/off, top-k, budget caps) use env_utils.py
instead.  Those flags are re-read on every chat turn and should not be cached
at startup.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from prompt_library import sage_kaizen_system_prompt as _default_system_prompt


class ServerConfig(BaseSettings):
    """
    Runtime configuration for all Sage Kaizen server endpoints and networking.

    Fields map directly to SAGE_* environment variables (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Endpoints ────────────────────────────────────────────────────────────
    sage_q5_base_url: str    = "http://127.0.0.1:8011"
    sage_q6_base_url: str    = "http://127.0.0.1:8012"
    sage_embed_base_url: str = "http://127.0.0.1:8020"

    # ── Model IDs (optional; client can discover via /v1/models) ─────────────
    sage_q5_model_id: str    = "Q5"
    sage_q6_model_id: str    = "Q6"
    sage_embed_model_id: str = "EMBED"

    # ── History ───────────────────────────────────────────────────────────────
    sage_max_history_messages: int = 20

    # ── Networking ────────────────────────────────────────────────────────────
    sage_connect_timeout_s: float    = 3.0
    sage_read_timeout_s: float       = 900.0   # up to 15 minutes for long Architect turns
    sage_stream_keepalive_s: float   = 30.0

    # ── System prompt — source of truth is prompt_library.py ─────────────────
    # Not an env var — always sourced from prompt_library at import time.
    # Defined here so ChatService only needs to import CONFIG.
    @property
    def system_prompt(self) -> str:
        return _default_system_prompt

    # ── Convenience aliases (match old field names used across the codebase) ──
    @property
    def q5_base_url(self) -> str:
        return self.sage_q5_base_url

    @property
    def q6_base_url(self) -> str:
        return self.sage_q6_base_url

    @property
    def embed_base_url(self) -> str:
        return self.sage_embed_base_url

    @property
    def q5_model_id(self) -> str:
        return self.sage_q5_model_id

    @property
    def q6_model_id(self) -> str:
        return self.sage_q6_model_id

    @property
    def embedded_model_id(self) -> str:
        return self.sage_embed_model_id

    @property
    def max_history_messages(self) -> int:
        return self.sage_max_history_messages

    @property
    def connect_timeout_s(self) -> float:
        return self.sage_connect_timeout_s

    @property
    def read_timeout_s(self) -> float:
        return self.sage_read_timeout_s

    @property
    def stream_keepalive_s(self) -> float:
        return self.sage_stream_keepalive_s


CONFIG = ServerConfig()
