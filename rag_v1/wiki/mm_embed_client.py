"""
rag_v1/wiki/mm_embed_client.py

HTTP client for the jina-clip-v2 embed service (rag_v1.wiki.mm_embed_service.app).
Used by both wiki_ingest.py (batch job) and wiki_retriever.py (runtime).

Env vars (same as service):
    SAGE_WIKI_EMBED_HOST  — default 127.0.0.1
    SAGE_WIKI_EMBED_PORT  — default 8031
"""
from __future__ import annotations

import base64
import os

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed


class MmEmbedClient:
    """
    Thin HTTP client wrapping the /embed/text and /embed/image endpoints.

    All outputs are L2-normalised by the service (normalize=True is hardcoded),
    ready for cosine similarity via dot-product.
    """

    def __init__(
        self,
        host: str = os.getenv("SAGE_WIKI_EMBED_HOST", "127.0.0.1"),
        port: int = int(os.getenv("SAGE_WIKI_EMBED_PORT", "8031")),
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        self._client = httpx.Client(timeout=timeout_s)

    # ------------------------------------------------------------------ #
    # Health                                                               #
    # ------------------------------------------------------------------ #

    def ping(self, timeout_s: float = 5.0) -> bool:
        """Return True if the embed service is reachable and healthy."""
        try:
            r = self._client.get(f"{self.base_url}/health", timeout=timeout_s)
            return r.status_code < 500
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Text embeddings                                                      #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of strings.

        Returns a list of 1024-dim L2-normalised float vectors.
        The service enforces SAGE_WIKI_TEXT_BATCH as max batch size.
        """
        r = self._client.post(
            f"{self.base_url}/embed/text",
            json={"texts": texts, "normalize": True},
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    # ------------------------------------------------------------------ #
    # Image embeddings                                                     #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
    def embed_image_bytes(self, images_bytes: list[bytes]) -> list[list[float]]:
        """
        Embed a batch of raw image bytes (any PIL-compatible format).

        Returns a list of 1024-dim L2-normalised float vectors in the same
        shared vector space as embed_text(), enabling text ↔ image cosine search.
        """
        b64_list = [base64.b64encode(b).decode("ascii") for b in images_bytes]
        r = self._client.post(
            f"{self.base_url}/embed/image",
            json={"images_b64": b64_list, "normalize": True},
        )
        r.raise_for_status()
        return r.json()["embeddings"]
