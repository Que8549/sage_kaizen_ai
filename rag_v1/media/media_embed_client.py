"""
rag_v1/media/media_embed_client.py

HTTP clients for the CLIP + CLAP dual-modal embed pipeline.

  ImageEmbedClient  — wraps the wiki jina-clip-v2 service (port 8031)
                      Returns 1024-dim L2-normalized float vectors.

  AudioEmbedClient  — wraps the CLAP clap-htsat-unfused service (port 8040)
                      Returns  512-dim L2-normalized float vectors.

Rewritten 2026-08-05 (see rag_v1/embed/base_client.py for the full rationale):

  * ImageEmbedClient was a second, independent implementation of
    MmEmbedClient — same host, same endpoints, same contract. It is now a thin
    subclass, so there is one implementation of the jina-clip-v2 protocol.
    The name is kept because sage_kaizen_ai_ingest's media_ingest.py imports
    it, and that project resolves this module from THIS repo (its CLAUDE.md
    §20).

  * Both clients previously called module-level ``httpx.post()``, building and
    discarding a TCP connection per call, while being driven at volume by the
    ingest media pipeline. Both are now pooled.

  * Both previously raised ``tenacity.RetryError`` after exhausting retries,
    unlike MmEmbedClient which re-raised the real error. All embed clients now
    re-raise ``httpx.HTTPStatusError``.
"""
from __future__ import annotations

import base64
from collections.abc import Sequence

from rag_v1.embed.base_client import BaseHttpEmbedClient
from rag_v1.wiki.mm_embed_client import MmEmbedClient


class ImageEmbedClient(MmEmbedClient):
    """
    Client for the jina-clip-v2 embed service (reuses the wiki embed service).

    Identical protocol to MmEmbedClient; this subclass exists only so the media
    pipeline reads with its own vocabulary and keeps a stable import path.

    Endpoints:
      POST /embed/text  — list[str]                → list[list[float]] (1024-dim)
      POST /embed/image — list[bytes] as base64    → list[list[float]] (1024-dim)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8031) -> None:
        super().__init__(host=host, port=port)

    def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text strings for image similarity search (1024-dim)."""
        return super().embed_text(list(texts))

    def embed_image_bytes(self, images_bytes: Sequence[bytes]) -> list[list[float]]:
        """Embed raw image bytes (1024-dim)."""
        # Parameter name matches MmEmbedClient.embed_image_bytes deliberately —
        # a mismatch is a Liskov violation pyright flags, and callers
        # (sage_kaizen_ai_ingest's media_ingest) pass it positionally anyway.
        return super().embed_image_bytes(list(images_bytes))


class AudioEmbedClient(BaseHttpEmbedClient):
    """
    Client for the CLAP (clap-htsat-unfused) embed service on port 8040.

    Endpoints:
      POST /embed/text  — list[str]                → list[list[float]] (512-dim)
      POST /embed/audio — list[bytes] as base64    → list[list[float]] (512-dim)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8040) -> None:
        super().__init__(host=host, port=port)

    def ping(self, timeout_s: float = 5.0) -> bool:
        """
        True only when CLAP reports the model actually loaded.

        Stricter than the inherited 2xx check, and deliberately kept that way:
        the CLAP service answers /health with 200 while the model is still
        loading, so plain reachability would report ready too early. The wiki
        service does not need this because it returns 503 until loaded.
        """
        payload = self.health(timeout_s=timeout_s)
        return bool(payload and payload.get("loaded", False))

    def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text strings for audio similarity search (512-dim)."""
        return self._post_embeddings("/embed/text", {"texts": list(texts)})

    def embed_audio_bytes(self, audios: Sequence[bytes]) -> list[list[float]]:
        """Embed raw audio file bytes (512-dim). Decoding happens server-side."""
        return self._post_embeddings(
            "/embed/audio",
            {"audios_b64": [base64.b64encode(b).decode() for b in audios]},
        )
