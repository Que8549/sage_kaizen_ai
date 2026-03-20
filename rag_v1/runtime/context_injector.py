"""
rag_v1/runtime/context_injector.py

RAG context injection for chat turns.

apply_rag()      — injects retrieved document chunks into the last user message.
apply_wiki_rag() — injects Wikipedia chunks + images into the last user message.

Both are called exclusively from chat_service.prepare_messages() and have no
dependency on the routing logic in router.py.

Env-var helpers
---------------
env_bool / env_int live here (not in settings.py) because they read runtime
flags that change behaviour per-call (RAG on/off, topK) rather than startup
configuration.  settings.py._env() handles the startup-string case; these
handle the typed-runtime-flag case.  Two distinct purposes, one place each.

Lazy singletons
---------------
_rag_settings, _rag_injector, and _wiki_retriever are initialised on first
call rather than at import time, so a misconfigured DB or missing wiki package
never crashes the app at startup.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from rag_v1.config.rag_settings import RagSettings
from rag_v1.runtime.router_integration import RagInjector, _prepend_context
from sk_logging import get_logger

# Wiki multimodal RAG — optional dependency
try:
    from rag_v1.wiki.wiki_retriever import WikiRetriever
    _WIKI_AVAILABLE = True
except ImportError:
    _WIKI_AVAILABLE = False

if TYPE_CHECKING:
    from rag_v1.wiki.wiki_retriever import WikiRetriever

_LOG = get_logger("sage_kaizen.context_injector")


# ---------------------------------------------------------------------------
# Env-var helpers
# ---------------------------------------------------------------------------

def env_bool(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v.strip())
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_rag_settings: RagSettings | None = None
_rag_injector: RagInjector | None = None
_wiki_retriever: "WikiRetriever | None" = None


def _ensure_rag() -> tuple[RagInjector, RagSettings]:
    global _rag_settings, _rag_injector
    if _rag_injector is None:
        _rag_settings = RagSettings()
        _rag_injector = RagInjector(_rag_settings)
    return _rag_injector, _rag_settings  # type: ignore[return-value]


def _get_wiki_retriever() -> "WikiRetriever | None":
    global _wiki_retriever
    if not _WIKI_AVAILABLE:
        return None
    if _wiki_retriever is None:
        _, settings = _ensure_rag()
        _wiki_retriever = WikiRetriever(pg_dsn=settings.pg_dsn)
    return _wiki_retriever


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_rag(
    messages: List[Dict[str, Any]],
    user_text: str,
    decision: Any,  # RouteDecision — duck-typed to avoid cross-package import
    rag_enabled: bool | None = None,
) -> Tuple[List[Dict[str, Any]], list]:
    """
    Drop-in RAG enrichment.  Call AFTER building messages, BEFORE sending to llama-server.

    Returns:
        (messages, rag_sources) — messages has RAG context injected into the last
        user turn; rag_sources is a list[RetrievedChunk] for citation rendering.
        Both early-exit paths return (original_messages, []).

    Controls:
      - rag_enabled: if None, reads env SAGE_RAG_ENABLED (default True)
      - top_k: FAST uses SAGE_RAG_FAST_TOPK (default 4)
               ARCHITECT uses SAGE_RAG_ARCH_TOPK (default 10)
    """
    if not user_text:
        return messages, []

    enabled = env_bool("SAGE_RAG_ENABLED", default=True) if rag_enabled is None else rag_enabled
    if not enabled:
        return messages, []

    min_chars = env_int("SAGE_RAG_MIN_CHARS", default=12)
    if len(user_text.strip()) < min_chars:
        return messages, []

    fast_k = env_int("SAGE_RAG_FAST_TOPK", default=4)
    arch_k = env_int("SAGE_RAG_ARCH_TOPK", default=10)
    top_k  = fast_k if decision.brain == "FAST" else arch_k

    injector, _ = _ensure_rag()
    try:
        out, sources = injector.maybe_inject(
            messages=messages,
            user_text=user_text,
            brain=decision.brain,
            enabled=True,
            top_k=top_k,
        )
        return out, sources
    except Exception:
        _LOG.exception("RAG injection failed; continuing without RAG")
        return messages, []


def apply_wiki_rag(
    messages: List[Dict[str, Any]],
    user_text: str,
    decision: Any,  # RouteDecision — duck-typed to avoid cross-package import
    wiki_enabled: bool | None = None,
) -> Tuple[List[Dict[str, Any]], list]:
    """
    Always-on Wikipedia RAG enrichment.  Call after apply_rag(), before
    sending to llama-server.

    Mirrors apply_rag() pattern:
      - Injects a <wiki_context> block into the last user turn.
      - Returns (messages, wiki_images) where wiki_images is a
        list[WikiImage] for Streamlit image rendering.

    Controls:
      - wiki_enabled: if None, reads env SAGE_WIKI_RAG_ENABLED (default True)
      - top_k:        same env vars as regular RAG (SAGE_RAG_FAST_TOPK / ARCH_TOPK)
    """
    enabled = env_bool("SAGE_WIKI_RAG_ENABLED", default=True) if wiki_enabled is None else wiki_enabled
    if not enabled or not user_text:
        return messages, []

    min_chars = env_int("SAGE_RAG_MIN_CHARS", default=12)
    if len(user_text.strip()) < min_chars:
        return messages, []

    retriever = _get_wiki_retriever()
    if retriever is None:
        return messages, []

    fast_k = env_int("SAGE_RAG_FAST_TOPK", default=4)
    arch_k = env_int("SAGE_RAG_ARCH_TOPK", default=10)
    top_k  = fast_k if decision.brain == "FAST" else arch_k

    try:
        result = retriever.search(user_text, top_k_chunks=top_k, top_images=3)
        if result.empty or not result.chunks:
            return messages, []

        lines: List[str] = []
        for c in result.chunks:
            section = " > ".join(c.section_path) if c.section_path else "Introduction"
            lines.append(f"[{c.title} / {section} | score={c.score:.3f}]\n{c.text}")
        ctx = "\n\n---\n\n".join(lines)

        out = list(messages)
        for i in reversed(range(len(out))):
            if out[i].get("role") == "user":
                prefix = f"<wiki_context>\n{ctx}\n</wiki_context>\n\n"
                out[i] = {**out[i], "content": _prepend_context(out[i]["content"], prefix)}
                break

        _LOG.info(
            "wiki_rag | chunks=%d images=%d | query_chars=%d",
            len(result.chunks), len(result.images), len(user_text),
        )
        return out, result.images

    except Exception:
        _LOG.exception("Wiki RAG injection failed; continuing without wiki context")
        return messages, []
