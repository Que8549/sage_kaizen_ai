"""
rag_v1/wiki/mm_embed_client.py

HTTP client for the jina-clip-v2 embed service (rag_v1.wiki.mm_embed_service.app).
Used by both wiki_ingest.py (batch job) and wiki_retriever.py (runtime).

Host and port default to the values in config/brains/brains.yaml (wiki_embed.service).
Callers that already hold a WikiEmbedConfig should pass cfg.host and cfg.port explicitly.
"""
from __future__ import annotations

import base64

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed


class MmEmbedClient:
    """
    Thin HTTP client wrapping the /embed/text and /embed/image endpoints.

    All outputs are L2-normalised by the service (normalize=True is hardcoded),
    ready for cosine similarity via dot-product.

    When host/port are omitted, values are read from brains.yaml (wiki_embed.service)
    so that the single authoritative config source is always honoured.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        if host is None or port is None:
            from rag_v1.wiki.wiki_embed_config import load_wiki_embed_config
            _cfg = load_wiki_embed_config()
            host = host if host is not None else _cfg.host
            port = port if port is not None else _cfg.port
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
        The service enforces the batch limit configured in brains.yaml.
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
