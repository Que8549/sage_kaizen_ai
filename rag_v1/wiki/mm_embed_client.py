"""
rag_v1/wiki/mm_embed_client.py

HTTP client for the jina-clip-v2 embed service (rag_v1.wiki.mm_embed_service.app).
Used by both wiki_ingest.py (batch job) and wiki_retriever.py (runtime).

Host and port default to the values in config/brains/brains.yaml (wiki_embed.service).
Callers that already hold a WikiEmbedConfig should pass cfg.host and cfg.port explicitly.

Transport (pooling, retries, /health, close) comes from BaseHttpEmbedClient —
see rag_v1/embed/base_client.py for why the four embed clients were unified.
"""
from __future__ import annotations

import base64

from rag_v1.embed.base_client import BaseHttpEmbedClient


class MmEmbedClient(BaseHttpEmbedClient):
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
        super().__init__(host=host, port=port, timeout_s=timeout_s)

    # ------------------------------------------------------------------ #
    # Health                                                               #
    # ------------------------------------------------------------------ #
    #
    # health() / ping() are inherited.  health() returns the service payload:
    #
    #     {"status": "ok", "device": "cuda:1", "model": "jina-clip-v2",
    #      "loaded": true, "offloaded": false, "idle_timeout_s": 120.0}
    #
    # `device` is what WikiRetriever's display-GPU guard reads — ping() alone
    # proves only that *something* answered, never where the model is loaded.

    # ------------------------------------------------------------------ #
    # Text embeddings                                                      #
    # ------------------------------------------------------------------ #

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of strings.

        Returns a list of 1024-dim L2-normalised float vectors.
        The service enforces the batch limit configured in brains.yaml.
        """
        return self._post_embeddings(
            "/embed/text", {"texts": list(texts), "normalize": True}
        )

    # ------------------------------------------------------------------ #
    # Image embeddings                                                     #
    # ------------------------------------------------------------------ #

    def embed_image_bytes(self, images_bytes: list[bytes]) -> list[list[float]]:
        """
        Embed a batch of raw image bytes (any PIL-compatible format).

        Returns a list of 1024-dim L2-normalised float vectors in the same
        shared vector space as embed_text(), enabling text ↔ image cosine search.
        """
        b64_list = [base64.b64encode(b).decode("ascii") for b in images_bytes]
        return self._post_embeddings(
            "/embed/image", {"images_b64": b64_list, "normalize": True}
        )
