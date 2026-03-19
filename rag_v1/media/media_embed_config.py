"""
rag_v1/media/media_embed_config.py

Loads the media_embed: section from config/brains/brains.yaml into typed
MediaEmbedConfig dataclasses.

Architecture
------------
  image_service  — jina-clip-v2 running on port 8031 (wiki embed service reuse)
  audio_service  — CLAP (clap-htsat-unfused) running on port 8040

The CLAP model is loaded from a local directory (no HuggingFace download at
service start time).

Usage:
    from rag_v1.media.media_embed_config import load_media_embed_config
    cfg = load_media_embed_config()
    print(cfg.image_host, cfg.image_port)   # 127.0.0.1, 8031
    print(cfg.audio_host, cfg.audio_port)   # 127.0.0.1, 8040
    print(cfg.clap_model_dir)               # E:/clap-htsat-unfused
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BRAINS_YAML  = _PROJECT_ROOT / "config" / "brains" / "brains.yaml"


@dataclass(frozen=True)
class MediaEmbedConfig:
    """
    Configuration for the CLIP + CLAP dual-modal embed pipeline.

    Loaded from the media_embed: section of config/brains/brains.yaml.

    Fields:
      clap_model_dir     — local path to laion/clap-htsat-unfused weights
      log                — log file path for the CLAP embed service
      startup_timeout_s  — seconds to wait for CLAP /health on startup
      image_service      — config dict for the jina-clip-v2 service (port 8031)
      audio_service      — config dict for the CLAP service (port 8040)
    """

    clap_model_dir:    Path
    log:               Path
    startup_timeout_s: float
    image_service:     Dict[str, Any]
    audio_service:     Dict[str, Any]

    # ── Image (jina-clip-v2 / wiki embed service) ────────────────────────── #

    @property
    def image_host(self) -> str:
        return str(self.image_service.get("host", "127.0.0.1"))

    @property
    def image_port(self) -> int:
        return int(self.image_service.get("port", 8031))

    @property
    def image_base_url(self) -> str:
        return f"http://{self.image_host}:{self.image_port}"

    @property
    def image_batch(self) -> int:
        return int(self.image_service.get("image_batch", 32))

    # ── Audio (CLAP / clap-htsat-unfused) ───────────────────────────────── #

    @property
    def audio_host(self) -> str:
        return str(self.audio_service.get("host", "127.0.0.1"))

    @property
    def audio_port(self) -> int:
        return int(self.audio_service.get("port", 8040))

    @property
    def audio_base_url(self) -> str:
        return f"http://{self.audio_host}:{self.audio_port}"

    @property
    def audio_device(self) -> str:
        return str(self.audio_service.get("device", "cuda:1"))

    @property
    def audio_batch(self) -> int:
        return int(self.audio_service.get("audio_batch", 8))


def load_media_embed_config(yaml_path: Path = _BRAINS_YAML) -> MediaEmbedConfig:
    """
    Parse the media_embed: entry from brains.yaml into a MediaEmbedConfig.

    Raises FileNotFoundError if brains.yaml does not exist.
    Raises KeyError if the media_embed: section is missing.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"brains.yaml not found: {yaml_path}\n"
            "Expected at config/brains/brains.yaml relative to the project root."
        )
    data: Dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    raw = data["media_embed"]
    return MediaEmbedConfig(
        clap_model_dir    = Path(str(raw.get("clap_model_dir", "E:/clap-htsat-unfused"))),
        log               = Path(str(raw.get("log", "F:/Projects/sage_kaizen_ai/logs/clap_embed_service.log"))),
        startup_timeout_s = float(raw.get("startup_timeout_s", 120.0)),
        image_service     = dict(raw.get("image_service", {})),
        audio_service     = dict(raw.get("audio_service", {})),
    )
