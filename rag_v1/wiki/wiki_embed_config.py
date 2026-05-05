"""
rag_v1/wiki/wiki_embed_config.py

Loads the wiki_embed: section from config/brains/brains.yaml into a typed
WikiEmbedConfig pydantic model.

Config-pattern rationale
------------------------
Three config layers exist in Sage Kaizen:

  1. Startup env config   — pydantic-settings BaseSettings (settings.py, pg_settings.py,
                            rag_settings.py).  Reads .env + environment variables once
                            at import time.  Use for credentials, base URLs, feature flags
                            that are set before the process starts.

  2. Runtime env flags    — env_utils.env_bool/int/float/str (context_injector.py etc.).
                            Re-read on every call.  Use for toggles that operators change
                            while the app is running without restarting.

  3. Structured YAML config — pydantic BaseModel loaded from brains.yaml (this file,
                            and server_manager.py).  Use for complex nested server
                            settings (model paths, CLI flags, batch sizes) that are
                            not env vars and are too structured for key=value .env syntax.

WikiEmbedConfig belongs to layer 3: it holds model paths, batch sizes, and
service ports that live in brains.yaml alongside other llama-server configs.
It uses BaseModel (not BaseSettings) because the source is YAML, not env vars.

Usage:
    from rag_v1.wiki.wiki_embed_config import load_wiki_embed_config

    cfg = load_wiki_embed_config()
    print(cfg.host, cfg.port, cfg.wiki_root)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator, model_validator

# ---------------------------------------------------------------------------
# Project-root-relative path to brains.yaml (resolved from this file's location)
# rag_v1/wiki/wiki_embed_config.py → rag_v1/wiki/ → rag_v1/ → project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BRAINS_YAML = _PROJECT_ROOT / "config" / "brains" / "brains.yaml"

_DEFAULT_EXCLUDE = (
    "References,External links,Further reading,Authority control"
)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class WikiEmbedServiceConfig(BaseModel):
    """Embed service settings (read by app.py / FastAPI)."""
    host: str          = "127.0.0.1"
    port: int          = 8031
    device: str        = "cuda:0"
    text_batch: int    = 32
    image_batch: int   = 8
    idle_timeout_s: float = 120.0


class WikiIngestConfig(BaseModel):
    """Ingest job settings (read by wiki_ingest.py)."""
    root: str                = ""
    chunk_chars: int         = 1200
    overlap: int             = 200
    log_every_pages: int     = 10000
    exclude_sections: str    = _DEFAULT_EXCLUDE


# ---------------------------------------------------------------------------
# WikiEmbedConfig
# ---------------------------------------------------------------------------

class WikiEmbedConfig(BaseModel):
    """
    Configuration for the wiki multimodal embed pipeline.

    Loaded from the wiki_embed: section of config/brains/brains.yaml.

    Top-level keys:
      model              — path to the local jina-clip-v2 directory
      log                — shared log file path for the service + ingest job
      startup_timeout_s  — seconds wiki_ingest.py waits for the service /health

    Sub-sections parsed into typed sub-models:
      service  — WikiEmbedServiceConfig (host, port, device, batch sizes, idle timeout)
      ingest   — WikiIngestConfig (root, chunk params, logging)
    """

    model: Path
    log: Path
    startup_timeout_s: float           = 300.0
    service: WikiEmbedServiceConfig    = WikiEmbedServiceConfig()
    ingest: WikiIngestConfig           = WikiIngestConfig()

    # ── Convenience properties (service) ─────────────────────────────────── #

    @property
    def host(self) -> str:
        return self.service.host

    @property
    def port(self) -> int:
        return self.service.port

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def device(self) -> str:
        return self.service.device

    @property
    def text_batch(self) -> int:
        return self.service.text_batch

    @property
    def image_batch(self) -> int:
        return self.service.image_batch

    @property
    def idle_timeout_s(self) -> float:
        """
        Seconds of inactivity before the embed service offloads jina-clip-v2
        from GPU to CPU, freeing ~2 GB on CUDA0.  0 = disabled (always resident).
        Override per-session via WIKI_EMBED_IDLE_TIMEOUT_S env var.
        """
        env_override = os.environ.get("WIKI_EMBED_IDLE_TIMEOUT_S")
        if env_override:
            try:
                return float(env_override)
            except ValueError:
                pass
        return self.service.idle_timeout_s

    # ── Convenience properties (ingest) ──────────────────────────────────── #

    @property
    def wiki_root(self) -> Path:
        if not self.ingest.root:
            raise KeyError("wiki_embed.ingest.root is not set in brains.yaml")
        return Path(self.ingest.root)

    @property
    def chunk_chars(self) -> int:
        return self.ingest.chunk_chars

    @property
    def overlap(self) -> int:
        return self.ingest.overlap

    @property
    def log_every_pages(self) -> int:
        return self.ingest.log_every_pages

    @property
    def exclude_sections(self) -> set[str]:
        raw = self.ingest.exclude_sections
        return {s.strip() for s in raw.split(",") if s.strip()}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_wiki_embed_config(yaml_path: Path = _BRAINS_YAML) -> WikiEmbedConfig:
    """
    Parse the wiki_embed: entry from brains.yaml into a WikiEmbedConfig.

    Raises FileNotFoundError if brains.yaml does not exist.
    Raises KeyError if the wiki_embed: section is missing.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"brains.yaml not found: {yaml_path}\n"
            "Expected at config/brains/brains.yaml relative to the project root."
        )
    raw_all: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    raw = raw_all["wiki_embed"]

    return WikiEmbedConfig(
        model             = Path(raw["model"]),
        log               = Path(raw["log"]),
        startup_timeout_s = float(raw.get("startup_timeout_s", 300.0)),
        service           = WikiEmbedServiceConfig(**raw.get("service", {})),
        ingest            = WikiIngestConfig(**raw.get("ingest", {})),
    )
