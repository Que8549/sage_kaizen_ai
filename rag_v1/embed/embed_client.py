from __future__ import annotations

import httpx


class EmbedClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout_s)
        # Lazy-initialised on first aembed() call; one persistent connection per instance.
        self._aclient: httpx.AsyncClient | None = None

    def close(self) -> None:
        """Close the underlying httpx.Client and release its connections."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the async client (call from async teardown if using aembed)."""
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None

    def __enter__(self) -> EmbedClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def ping(self, timeout_s: float = 5.0) -> bool:
        """Return True if the embedding server is reachable, False otherwise."""
        try:
            r = self._client.get(f"{self.base_url}/health", timeout=timeout_s)
            return r.status_code < 500
        except Exception:
            return False

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Async embed — reuses a single AsyncClient per EmbedClient instance."""
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(timeout=self._client.timeout)
        payload = {"model": self.model, "input": texts}
        r = await self._aclient.post(f"{self.base_url}/embeddings", json=payload)
        r.raise_for_status()
        data = r.json()
        items = data["data"] if isinstance(data, dict) else data
        out: list[list[float]] = [[] for _ in range(len(texts))]
        for item in items:
            out[item["index"]] = list(item["embedding"])
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        r = self._client.post(f"{self.base_url}/embeddings", json=payload)
        r.raise_for_status()
        data = r.json()
        # JSON parsing already yields Python floats; list() copies without
        # re-converting each element (skips 1024 float() calls per embedding).
        # Handle both {"data": [...]} and bare [...] response formats.
        items = data["data"] if isinstance(data, dict) else data
        out: list[list[float]] = [[] for _ in range(len(texts))]
        for item in items:
            out[item["index"]] = list(item["embedding"])
        return out
