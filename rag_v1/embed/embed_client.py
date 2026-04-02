from __future__ import annotations
import httpx
from typing import List

class EmbedClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        """Close the underlying httpx.Client and release its connections."""
        self._client.close()

    def __enter__(self) -> "EmbedClient":
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

    def embed(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model, 
            "input": texts
        }
        
        r = self._client.post(f"{self.base_url}/embeddings", json=payload)
        r.raise_for_status()
        data = r.json()

        # JSON parsing already yields Python floats; list() copies without
        # re-converting each element (skips 1024 float() calls per embedding).
        out: List[List[float]] = [[] for _ in range(len(texts))]
        for item in data["data"]:
            out[item["index"]] = list(item["embedding"])
        return out
