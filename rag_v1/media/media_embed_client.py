"""
rag_v1/media/media_embed_client.py

HTTP clients for the CLIP + CLAP dual-modal embed pipeline.

  ImageEmbedClient  — wraps the wiki jina-clip-v2 service (port 8031)
                      Returns 1024-dim L2-normalized float vectors.

  AudioEmbedClient  — wraps the CLAP clap-htsat-unfused service (port 8040)
                      Returns  512-dim L2-normalized float vectors.

Both clients use httpx with tenacity retries (3 attempts, 1 s wait).
Both raise httpx.HTTPStatusError on non-2xx responses after all retries.
"""
from __future__ import annotations

import base64
from typing import Sequence

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)


class ImageEmbedClient:
    """
    Client for the jina-clip-v2 embed service (reuses wiki embed service).

    The wiki service already supports:
      POST /embed/text  — list[str]  → list[list[float]] (1024-dim)
      POST /embed/image — list[bytes as base64] → list[list[float]] (1024-dim)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8031) -> None:
        self._base = f"http://{host}:{port}"

    def ping(self, timeout_s: float = 3.0) -> bool:
        try:
            r = httpx.get(f"{self._base}/health", timeout=timeout_s)
            if r.status_code != 200:
                return False
            data = r.json()
            # Wiki embed service returns {"status": "ok"};
            # any future service using the CLAP pattern returns {"loaded": true}.
            return data.get("loaded", False) or data.get("status") == "ok"
        except Exception:
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text strings for image similarity search (1024-dim)."""
        r = httpx.post(
            f"{self._base}/embed/text",
            json={"texts": list(texts)},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def embed_image_bytes(self, images: Sequence[bytes]) -> list[list[float]]:
        """Embed raw image bytes (1024-dim)."""
        r = httpx.post(
            f"{self._base}/embed/image",
            json={"images_b64": [base64.b64encode(b).decode() for b in images]},
            timeout=_TIMEOUT,
        )
        if not r.is_success:
            raise httpx.HTTPStatusError(
                f"{r.status_code} from {r.url}: {r.text[:500]}",
                request=r.request,
                response=r,
            )
        return r.json()["embeddings"]


class AudioEmbedClient:
    """
    Client for the CLAP (clap-htsat-unfused) embed service on port 8040.

    Endpoints:
      POST /embed/text  — list[str]   → list[list[float]] (512-dim)
      POST /embed/audio — list[bytes] as base64 → list[list[float]] (512-dim)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8040) -> None:
        self._base = f"http://{host}:{port}"

    def ping(self, timeout_s: float = 3.0) -> bool:
        try:
            r = httpx.get(f"{self._base}/health", timeout=timeout_s)
            return r.status_code == 200 and r.json().get("loaded", False)
        except Exception:
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text strings for audio similarity search (512-dim)."""
        r = httpx.post(
            f"{self._base}/embed/text",
            json={"texts": list(texts)},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def embed_audio_bytes(self, audios: Sequence[bytes]) -> list[list[float]]:
        """Embed raw audio file bytes (512-dim). Decoding happens server-side."""
        r = httpx.post(
            f"{self._base}/embed/audio",
            json={"audios_b64": [base64.b64encode(b).decode() for b in audios]},
            timeout=_TIMEOUT,
        )
        if not r.is_success:
            raise httpx.HTTPStatusError(
                f"{r.status_code} from {r.url}: {r.text[:500]}",
                request=r.request,
                response=r,
            )
        return r.json()["embeddings"]
