"""
rag_v1/wiki/wiki_embed_config.py

Loads the wiki_embed: section from config/brains/brains.yaml into a typed
WikiEmbedConfig dataclass.

Mirrors the BrainConfig / _load_brain_config pattern from server_manager.py:
  - brains.yaml is the single authoritative config source
  - No env vars, no hardcoded defaults in caller code
  - Shared by both mm_embed_service/app.py and wiki_ingest.py

Usage:
    from rag_v1.wiki.wiki_embed_config import load_wiki_embed_config

    cfg = load_wiki_embed_config()
    print(cfg.host, cfg.port, cfg.wiki_root)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

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
# WikiEmbedConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WikiEmbedConfig:
    """
    Configuration for the wiki multimodal embed pipeline.

    Loaded from the wiki_embed: section of config/brains/brains.yaml.

    Top-level keys:
      model              — path to the local jina-clip-v2 directory
      log                — shared log file path for the service + ingest job
      startup_timeout_s  — seconds wiki_ingest.py waits for the service /health

    Sub-sections (stored as plain dicts, exposed via typed properties):
      service:  embed service settings (host, port, device, batch sizes)
      ingest:   wiki_ingest.py job settings (root, chunk params, logging)
    """

    model: Path
    log: Path
    startup_timeout_s: float
    service: Dict[str, Any]
    ingest: Dict[str, Any]

    # ── Embed-service properties (used by app.py and wiki_ingest.py) ──── #

    @property
    def host(self) -> str:
        return str(self.service.get("host", "127.0.0.1"))

    @property
    def port(self) -> int:
        return int(self.service.get("port", 8031))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def device(self) -> str:
        return str(self.service.get("device", "cuda:0"))

    @property
    def text_batch(self) -> int:
        return int(self.service.get("text_batch", 32))

    @property
    def image_batch(self) -> int:
        return int(self.service.get("image_batch", 8))

    # ── Ingest-job properties (used by wiki_ingest.py) ────────────────── #

    @property
    def wiki_root(self) -> Path:
        root = self.ingest.get("root")
        if not root:
            raise KeyError("wiki_embed.ingest.root is not set in brains.yaml")
        return Path(str(root))

    @property
    def chunk_chars(self) -> int:
        return int(self.ingest.get("chunk_chars", 1200))

    @property
    def overlap(self) -> int:
        return int(self.ingest.get("overlap", 200))

    @property
    def log_every_pages(self) -> int:
        return int(self.ingest.get("log_every_pages", 10000))

    @property
    def exclude_sections(self) -> set[str]:
        raw = str(self.ingest.get("exclude_sections", _DEFAULT_EXCLUDE))
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
    data: Dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    raw = data["wiki_embed"]
    return WikiEmbedConfig(
        model=Path(raw["model"]),
        log=Path(raw["log"]),
        startup_timeout_s=float(raw.get("startup_timeout_s", 300.0)),
        service=dict(raw.get("service", {})),
        ingest=dict(raw.get("ingest", {})),
    )
