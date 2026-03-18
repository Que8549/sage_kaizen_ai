"""
rag_v1/media/languagebind_embed_client.py

Thin HTTP client for the LanguageBind embed service (port 8040).

Mirrors the pattern of rag_v1/wiki/mm_embed_client.py.
Supports text, image, audio, and video embedding in a shared 768-dim space.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class MediaEmbedClient:
    """
    HTTP client for the LanguageBind FastAPI embed service.

    Usage:
        client = MediaEmbedClient(host="127.0.0.1", port=8040)
        text_vecs = client.embed_text(["a house by the sea"])
        img_vecs  = client.embed_image_bytes([open("photo.jpg", "rb").read()])
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8040) -> None:
        self._base = f"http://{host}:{port}"

    # ------------------------------------------------------------------ #
    # Health                                                               #
    # ------------------------------------------------------------------ #

    def ping(self, timeout_s: float = 2.0) -> bool:
        try:
            r = httpx.get(f"{self._base}/health", timeout=timeout_s)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Text                                                                 #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    def embed_text(
        self,
        texts: list[str],
        normalize: bool = True,
        timeout_s: float = 30.0,
    ) -> list[list[float]]:
        """Embed text strings into 768-dim vectors."""
        if not texts:
            return []
        r = httpx.post(
            f"{self._base}/embed/text",
            json={"texts": texts, "normalize": normalize},
            timeout=timeout_s,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    # ------------------------------------------------------------------ #
    # Image                                                                #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    def embed_image_bytes(
        self,
        images: list[bytes],
        normalize: bool = True,
        timeout_s: float = 60.0,
    ) -> list[list[float]]:
        """Embed raw image bytes (PNG/JPG) into 768-dim vectors."""
        if not images:
            return []
        b64s = [base64.b64encode(raw).decode("ascii") for raw in images]
        r = httpx.post(
            f"{self._base}/embed/image",
            json={"items_b64": b64s, "normalize": normalize},
            timeout=timeout_s,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    def embed_image_path(self, paths: list[Path | str], **kwargs) -> list[list[float]]:
        """Convenience wrapper: reads files from disk."""
        return self.embed_image_bytes(
            [Path(p).read_bytes() for p in paths], **kwargs
        )

    # ------------------------------------------------------------------ #
    # Audio                                                                #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    def embed_audio_bytes(
        self,
        clips: list[bytes],
        normalize: bool = True,
        timeout_s: float = 60.0,
    ) -> list[list[float]]:
        """Embed raw audio bytes (WAV/MP3/FLAC) into 768-dim vectors."""
        if not clips:
            return []
        b64s = [base64.b64encode(raw).decode("ascii") for raw in clips]
        r = httpx.post(
            f"{self._base}/embed/audio",
            json={"items_b64": b64s, "normalize": normalize},
            timeout=timeout_s,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    def embed_audio_path(self, paths: list[Path | str], **kwargs) -> list[list[float]]:
        return self.embed_audio_bytes(
            [Path(p).read_bytes() for p in paths], **kwargs
        )

    # ------------------------------------------------------------------ #
    # Video                                                                #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    def embed_video_bytes(
        self,
        clips: list[bytes],
        normalize: bool = True,
        timeout_s: float = 120.0,
    ) -> list[list[float]]:
        """Embed raw video bytes (MP4/MOV etc.) into 768-dim vectors."""
        if not clips:
            return []
        b64s = [base64.b64encode(raw).decode("ascii") for raw in clips]
        r = httpx.post(
            f"{self._base}/embed/video",
            json={"items_b64": b64s, "normalize": normalize},
            timeout=timeout_s,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    def embed_video_path(self, paths: list[Path | str], **kwargs) -> list[list[float]]:
        return self.embed_video_bytes(
            [Path(p).read_bytes() for p in paths], **kwargs
        )
