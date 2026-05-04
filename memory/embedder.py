"""
memory/embedder.py
Memory embedder — wraps the existing BGE-M3 embed client at port 8020.

Design rules:
- Reuses rag_v1/embed/embed_client.py.  Do NOT introduce a second embed client.
- Returns 1024-dimensional L2-normalized float vectors (BGE-M3 FP16).
- Sync on the hot path (called from repository retrieval queries).
- Module-level singleton avoids re-opening HTTP connections on every call.
"""
from __future__ import annotations


from rag_v1.embed.embed_client import EmbedClient
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.memory.embedder")

# BGE-M3 embed service — inherits the same port used by RAG and wiki retrieval.
_BGE_BASE_URL = "http://127.0.0.1:8020"
_BGE_MODEL    = "bge-m3"
_DIMS         = 1024

_singleton: EmbedClient | None = None


def _get_client() -> EmbedClient:
    global _singleton
    if _singleton is None:
        _singleton = EmbedClient(base_url=_BGE_BASE_URL, model=_BGE_MODEL, timeout_s=30.0)
        _LOG.debug("embedder | BGE-M3 client initialised at %s", _BGE_BASE_URL)
    return _singleton


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using BGE-M3 (sync).

    Returns a list of 1024-dimensional float vectors, one per input text.
    Raises httpx.HTTPError on service failure.
    """
    if not texts:
        return []
    client = _get_client()
    vecs = client.embed(texts)
    _LOG.debug("embedder | embedded %d texts → %d-dim vectors", len(texts), _DIMS)
    return vecs


def embed_one(text: str) -> list[float]:
    """Convenience wrapper for a single text."""
    return embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Async wrapper used by LangMem bridge and consolidator's background tasks
# ---------------------------------------------------------------------------

async def aembed_texts(texts: list[str]) -> list[list[float]]:
    """
    Async version — uses EmbedClient.aembed() (native httpx.AsyncClient)
    so the event loop is not blocked.  Used by langmem_bridge.py and the
    async consolidator path.
    """
    if not texts:
        return []
    return await _get_client().aembed(texts)


async def aembed_one(text: str) -> list[float]:
    result = await aembed_texts([text])
    return result[0]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def embed_service_alive() -> bool:
    """Return True if the BGE-M3 embed service is reachable."""
    try:
        return _get_client().ping(timeout_s=3.0)
    except Exception:
        return False
