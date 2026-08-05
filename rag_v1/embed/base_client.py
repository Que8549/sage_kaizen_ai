"""
rag_v1/embed/base_client.py

One HTTP embed-client base, shared by every embed service in the stack.

Why this exists
---------------
Before 2026-08-05 there were four embed clients with three different
connection strategies and four different readings of ``/health``:

  ================= ================================ ======== =========================
  Class             Service                          Pooled?  On exhausted retries
  ================= ================================ ======== =========================
  MmEmbedClient     jina-clip-v2  :8031              yes      original exception
  ImageEmbedClient  jina-clip-v2  :8031  (duplicate) NO       tenacity.RetryError
  AudioEmbedClient  CLAP          :8040              NO       tenacity.RetryError
  EmbedClient       BGE-M3        :8020              yes      raw raise_for_status
  ================= ================================ ======== =========================

Two problems that fixed:

1. ``ImageEmbedClient`` was a second implementation of ``MmEmbedClient`` —
   same host, same ``/embed/text`` and ``/embed/image`` endpoints, same
   1024-dim contract. It is now a thin subclass, so there is one implementation.

2. The two media clients called module-level ``httpx.post()``, which builds and
   discards a client — and therefore a TCP connection — on every call. They are
   driven at volume by ``sage_kaizen_ai_ingest``'s media pipeline. That matters
   beyond efficiency: that project spent weeks on an unresolved TCP
   ephemeral-port-exhaustion lead (its CLAUDE.md §15, candidate 4) which
   recorded "MmEmbedClient was checked and is correctly pooled — the obvious
   leak isn't there". These two were never checked, and were not pooled.

Everything now shares one pooled ``httpx.Client``, one ``/health`` semantic,
one retry policy (``reraise=True``, so callers see the real
``httpx.HTTPStatusError`` rather than a ``tenacity.RetryError`` wrapper), and
one ``close()`` / context-manager contract.
"""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

__all__ = ["BaseHttpEmbedClient", "EMBED_RETRY"]

# Generous read timeout: a cold jina-clip-v2 or CLAP forward pass can take tens
# of seconds before torch.compile has warmed up.
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)

# One retry policy for every embed call.
#
# reraise=True is the important part: without it tenacity raises RetryError
# wrapping the real error, and the two media clients did exactly that while
# MmEmbedClient did not — so a caller catching httpx.HTTPStatusError worked
# against one client and silently missed the other.
EMBED_RETRY = retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)


class BaseHttpEmbedClient:
    """
    Shared transport for the embed services.

    Subclasses add the endpoint methods (``embed_text`` and friends) and call
    :meth:`_post_embeddings`. They inherit pooling, health, and cleanup.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        timeout_s: float | httpx.Timeout | None = None,
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        timeout = DEFAULT_TIMEOUT if timeout_s is None else timeout_s
        # One pooled client per instance, reused for the object's lifetime.
        self._client = httpx.Client(timeout=timeout)

    # -- lifecycle ------------------------------------------------------- #

    def close(self) -> None:
        """Close the underlying client and release its connection pool."""
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort: __del__ runs during interpreter shutdown when module
        # globals may already be torn down.
        try:
            self.close()
        except Exception:
            pass

    # -- health ---------------------------------------------------------- #

    def health(self, timeout_s: float = 5.0) -> dict[str, Any] | None:
        """
        Return the parsed ``/health`` payload, or None if unreachable/unhealthy.

        None covers every failure mode — connection refused, non-2xx (the wiki
        service returns 503 once its CUDA context has failed permanently), and
        an unparseable body. A non-dict payload yields ``{}`` rather than None,
        so "answered but said nothing useful" stays distinguishable from "did
        not answer".
        """
        try:
            r = self._client.get(f"{self.base_url}/health", timeout=timeout_s)
            if not r.is_success:
                return None
            payload = r.json()
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return None

    def ping(self, timeout_s: float = 5.0) -> bool:
        """True when the service is reachable and reports itself healthy."""
        return self.health(timeout_s=timeout_s) is not None

    # -- requests -------------------------------------------------------- #

    @EMBED_RETRY
    def _post_embeddings(self, path: str, payload: dict[str, Any]) -> list[list[float]]:
        """
        POST `payload` to `path` and return the ``embeddings`` array.

        Retries transient failures per EMBED_RETRY, then re-raises the original
        ``httpx.HTTPStatusError``.
        """
        r = self._client.post(f"{self.base_url}{path}", json=payload)
        r.raise_for_status()
        return r.json()["embeddings"]
