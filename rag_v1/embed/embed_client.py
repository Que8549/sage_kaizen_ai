"""
rag_v1/embed/embed_client.py

Client for the BGE-M3 embedding server (llama-server, port 8020).

Unlike the jina-clip-v2 and CLAP clients this speaks the OpenAI-compatible
``/embeddings`` shape — ``{"model": ..., "input": [...]}`` returning
``{"data": [{"index": i, "embedding": [...]}]}`` — so it overrides the request
methods rather than using BaseHttpEmbedClient._post_embeddings().  It shares
the base's pooling, close()/context-manager contract, and health handling.
"""
from __future__ import annotations

import httpx

from rag_v1.embed.base_client import BaseHttpEmbedClient


class EmbedClient(BaseHttpEmbedClient):
    def __init__(self, base_url: str, model: str, timeout_s: float = 60.0) -> None:
        # This client is constructed from a full base_url rather than host/port,
        # so it sets base_url directly instead of delegating to the base's
        # host/port constructor.
        super().__init__(host="127.0.0.1", port=0, timeout_s=timeout_s)
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Lazy-initialised on first aembed() call; one persistent connection per instance.
        self._aclient: httpx.AsyncClient | None = None

    def close(self) -> None:
        """Close the sync client and release its connections."""
        super().close()

    async def aclose(self) -> None:
        """Close the async client (call from async teardown if using aembed)."""
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None

    def ping(self, timeout_s: float = 5.0) -> bool:
        """
        True if the embedding server is reachable.

        Deliberately laxer than the base implementation: llama-server's embed
        endpoint does not serve a JSON /health, so anything below a 5xx counts
        as "the process is up and answering".
        """
        try:
            r = self._client.get(f"{self.base_url}/health", timeout=timeout_s)
            return r.status_code < 500
        except Exception:
            return False

    def _unpack(self, data, n: int) -> list[list[float]]:
        """
        Order embeddings by their ``index`` field.

        The server may return them out of order; callers zip results against
        their input list, so position matters.  Handles both
        ``{"data": [...]}`` and a bare ``[...]`` response.
        """
        items = data["data"] if isinstance(data, dict) else data
        out: list[list[float]] = [[] for _ in range(n)]
        for item in items:
            # JSON parsing already yields Python floats; list() copies without
            # re-converting each element (skips 1024 float() calls per embedding).
            out[item["index"]] = list(item["embedding"])
        return out

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Async embed — reuses a single AsyncClient per EmbedClient instance."""
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(timeout=self._client.timeout)
        r = await self._aclient.post(
            f"{self.base_url}/embeddings", json={"model": self.model, "input": texts}
        )
        r.raise_for_status()
        return self._unpack(r.json(), len(texts))

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self._client.post(
            f"{self.base_url}/embeddings", json={"model": self.model, "input": texts}
        )
        r.raise_for_status()
        return self._unpack(r.json(), len(texts))
