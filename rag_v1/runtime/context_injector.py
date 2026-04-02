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
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Module-level executor — threads are kept alive for the process lifetime so
# thread creation cost is paid once, not on every chat turn.
# max_workers=4 covers the four parallel fetches: doc-RAG, wiki-RAG, search, music.
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag_par")

from input_guard import sanitize_chunk
from rag_v1.config.rag_settings import RagSettings
from rag_v1.runtime.router_integration import RagInjector, _prepend_context
from sk_logging import get_logger

# Wiki multimodal RAG — optional dependency
try:
    from rag_v1.wiki.wiki_retriever import WikiRetriever
    _WIKI_AVAILABLE = True
except ImportError:
    _WIKI_AVAILABLE = False

# Live web search — optional dependency
try:
    from search.models import SearchEvidence
    from search.search_orchestrator import get_orchestrator
    from search.summarizer import build_raw_context, summarize_evidence
    _SEARCH_AVAILABLE = True
except ImportError:
    _SEARCH_AVAILABLE = False

# Music retrieval — optional dependency
try:
    from rag_v1.media.music_retriever import (
        MusicRetriever,
        detect_intent as _detect_music_intent,
        format_music_context,
    )
    _MUSIC_AVAILABLE = True
except ImportError:
    _MUSIC_AVAILABLE = False

if TYPE_CHECKING:
    from rag_v1.wiki.wiki_retriever import WikiRetriever
    from search.models import SearchEvidence

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
_music_retriever: "MusicRetriever | None" = None


def _ensure_rag() -> tuple[RagInjector, RagSettings]:
    global _rag_settings, _rag_injector
    if _rag_injector is None:
        _rag_settings = RagSettings()
        _rag_injector = RagInjector(_rag_settings)
    return _rag_injector, _rag_settings  # type: ignore[return-value]


def _get_music_retriever() -> "MusicRetriever | None":
    global _music_retriever
    if not _MUSIC_AVAILABLE:
        return None
    if _music_retriever is None:
        _, settings = _ensure_rag()
        _music_retriever = MusicRetriever(pg_dsn=settings.pg_dsn)
    return _music_retriever


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
            lines.append(f"[{c.title} / {section} | score={c.score:.3f}]\n{sanitize_chunk(c.text, max_chars=None)}")
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


def _fetch_wiki_result(
    user_text: str,
    decision: Any,
    wiki_enabled: bool | None = None,
) -> tuple:
    """
    Run the wiki DB query and return (chunks_text, wiki_images) without injecting
    into messages.  Designed to run concurrently with apply_rag().

    Returns:
        (ctx_block, wiki_images) where ctx_block is the formatted <wiki_context>
        string (or "" if nothing found) and wiki_images is list[WikiImage].
    """
    enabled = env_bool("SAGE_WIKI_RAG_ENABLED", default=True) if wiki_enabled is None else wiki_enabled
    if not enabled or not user_text:
        return "", []

    min_chars = env_int("SAGE_RAG_MIN_CHARS", default=12)
    if len(user_text.strip()) < min_chars:
        return "", []

    retriever = _get_wiki_retriever()
    if retriever is None:
        return "", []

    fast_k = env_int("SAGE_RAG_FAST_TOPK", default=4)
    arch_k = env_int("SAGE_RAG_ARCH_TOPK", default=10)
    top_k  = fast_k if decision.brain == "FAST" else arch_k

    try:
        result = retriever.search(user_text, top_k_chunks=top_k, top_images=3)
        if result.empty or not result.chunks:
            return "", []

        lines: List[str] = []
        for c in result.chunks:
            section = " > ".join(c.section_path) if c.section_path else "Introduction"
            lines.append(f"[{c.title} / {section} | score={c.score:.3f}]\n{sanitize_chunk(c.text, max_chars=None)}")
        ctx_block = "\n\n---\n\n".join(lines)

        _LOG.info(
            "wiki_rag | chunks=%d images=%d | query_chars=%d",
            len(result.chunks), len(result.images), len(user_text),
        )
        return ctx_block, result.images

    except Exception:
        _LOG.exception("Wiki RAG fetch failed; continuing without wiki context")
        return "", []


def _fetch_search_result(
    user_text: str,
    decision: Any,
    fast_base_url: str | None = None,
    fast_model_id: str | None = None,
) -> "tuple[str, SearchEvidence | None]":
    """
    Run a live SearXNG search and return (ctx_block, evidence).

    Only executes when decision.needs_search is True and the search
    module is available.  Falls back to raw snippet formatting when the
    FAST brain is unavailable for the summarization pass.

    Returns:
        (ctx_block, evidence)
        ctx_block — formatted <search_context> string, or "" if nothing found.
        evidence  — SearchEvidence for citation rendering in the UI, or None.
    """
    if not _SEARCH_AVAILABLE:
        return "", None

    if not getattr(decision, "needs_search", False):
        return "", None

    search_enabled = env_bool("SAGE_SEARCH_ENABLED", default=True)
    if not search_enabled or not user_text:
        return "", None

    min_chars = env_int("SAGE_RAG_MIN_CHARS", default=12)
    if len(user_text.strip()) < min_chars:
        return "", None

    categories = list(getattr(decision, "search_categories", ()) or ("general", "news"))

    try:
        orchestrator = get_orchestrator()
        evidence = orchestrator.search(
            query=user_text,
            categories=categories,
            brain=decision.brain,
        )
    except Exception:
        _LOG.exception("SearchOrchestrator.search() failed; skipping live search")
        return "", None

    if evidence.empty:
        _LOG.info("search_rag | no results returned for query_chars=%d", len(user_text))
        return "", None

    # Summarization pass using the FAST brain
    summary = ""
    if fast_base_url and fast_model_id:
        try:
            summary = summarize_evidence(
                evidence=evidence,
                fast_base_url=fast_base_url,
                fast_model_id=fast_model_id,
            )
        except Exception:
            _LOG.warning("Summarization failed; injecting raw search snippets")

    # Build context block — prefer summary, fall back to raw snippets
    from datetime import datetime, timezone
    fetched = evidence.fetched_at[:16].replace("T", " ") + " UTC"
    cats    = ", ".join(evidence.categories_queried)

    if summary:
        body = summary
        # Attach source list after the summary for traceability
        source_lines = []
        for i, r in enumerate(evidence.results, 1):
            date_part = f" | {r.published_date}" if r.published_date else ""
            source_lines.append(f"[{i}] {r.title} | {r.source_engine}{date_part} — {r.url}")
        body += "\n\nSources:\n" + "\n".join(source_lines)
    else:
        body = build_raw_context(evidence)

    ctx_block = (
        f'<search_context fetched="{fetched}" categories="{cats}">\n'
        f"{body}\n"
        f"</search_context>"
    )

    _LOG.info(
        "search_rag | results=%d | summarized=%s | ctx_chars=%d",
        len(evidence.results), bool(summary), len(ctx_block),
    )

    # Return a new evidence with the summary attached so the UI can display it
    from search.models import SearchEvidence as _SE
    enriched_evidence = _SE(
        query             = evidence.query,
        results           = evidence.results,
        fetched_at        = evidence.fetched_at,
        categories_queried= evidence.categories_queried,
        summarized_text   = summary,
    )
    return ctx_block, enriched_evidence


def _fetch_music_result(
    user_text: str,
    decision: Any,
) -> str:
    """
    Detect music intent, run the appropriate MusicRetriever method, and
    return a formatted <music_context> block.

    Only executes when decision.needs_music is True and the music module
    is available.  Returns "" on any failure so the turn continues normally.
    """
    if not _MUSIC_AVAILABLE:
        return ""

    if not getattr(decision, "needs_music", False):
        return ""

    if not user_text:
        return ""

    retriever = _get_music_retriever()
    if retriever is None:
        return ""

    try:
        intent = _detect_music_intent(user_text)
        if intent is None:
            return ""

        results = retriever.dispatch(intent, top_k=10)
        ctx_block = format_music_context(intent, results, top_k=10)

        _LOG.info(
            "music_rag | intent=%s | results=%d | ctx_chars=%d",
            intent.intent, len(results), len(ctx_block),
        )
        return ctx_block

    except Exception:
        _LOG.exception("Music retrieval failed; continuing without music context")
        return ""


def apply_rag_and_wiki_parallel(
    messages: List[Dict[str, Any]],
    user_text: str,
    decision: Any,
    wiki_enabled: bool | None = None,
    fast_base_url: str | None = None,
    fast_model_id: str | None = None,
) -> Tuple[List[Dict[str, Any]], list, list, "Optional[SearchEvidence]", str]:
    """
    Run document RAG, wiki RAG, live web search, and music retrieval
    concurrently, then inject all into messages.

    The four fetches are independent and safe to run in parallel.
    Injection order (innermost → outermost in the user message):
        1. doc-RAG   → <context>
        2. wiki-RAG  → <wiki_context>
        3. search    → <search_context>   (only when decision.needs_search)
        4. music     → <music_context>    (only when decision.needs_music)

    Parameters
    ----------
    fast_base_url : FAST brain base URL — required for the summarization pass.
                    Pass None to skip summarization and inject raw snippets.
    fast_model_id : FAST brain model alias (from InferenceSession.q5_model_id).

    Returns
    -------
    (messages, rag_sources, wiki_images, search_evidence, music_context)
    search_evidence is None when live search was not triggered.
    music_context is "" when music retrieval was not triggered.
    """
    rag_fut    = _POOL.submit(apply_rag, messages, user_text, decision)
    wiki_fut   = _POOL.submit(_fetch_wiki_result, user_text, decision, wiki_enabled)
    search_fut = _POOL.submit(_fetch_search_result, user_text, decision,
                              fast_base_url, fast_model_id)
    music_fut  = _POOL.submit(_fetch_music_result, user_text, decision)

    rag_messages, rag_sources         = rag_fut.result()
    wiki_ctx_block, wiki_images       = wiki_fut.result()
    search_ctx_block, search_evidence = search_fut.result()
    music_ctx_block                   = music_fut.result()

    # Inject wiki context into already-RAG-enriched messages
    out = list(rag_messages)
    if wiki_ctx_block:
        for i in reversed(range(len(out))):
            if out[i].get("role") == "user":
                prefix = f"<wiki_context>\n{wiki_ctx_block}\n</wiki_context>\n\n"
                out[i] = {**out[i], "content": _prepend_context(out[i]["content"], prefix)}
                break

    # Inject search context outermost (closest to the user question)
    if search_ctx_block:
        for i in reversed(range(len(out))):
            if out[i].get("role") == "user":
                prefix = f"{search_ctx_block}\n\n"
                out[i] = {**out[i], "content": _prepend_context(out[i]["content"], prefix)}
                break

    # Inject music context outermost (closest to the user question)
    if music_ctx_block:
        for i in reversed(range(len(out))):
            if out[i].get("role") == "user":
                prefix = f"{music_ctx_block}\n\n"
                out[i] = {**out[i], "content": _prepend_context(out[i]["content"], prefix)}
                break

    return out, rag_sources, wiki_images if wiki_ctx_block else [], search_evidence, music_ctx_block
