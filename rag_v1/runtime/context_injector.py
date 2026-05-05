"""
rag_v1/runtime/context_injector.py

RAG context injection for chat turns.

Public API (called from chat_service.prepare_messages()):
  apply_rag_and_wiki_parallel() — runs all four fetches concurrently and
                                  injects the results into the messages list.
  apply_rag()                   — doc-RAG only (used as a sub-task of the
                                  parallel function; also useful for testing).

Internal fetch workers (thread-pool tasks, not part of the public API):
  _fetch_wiki_result()   — Wikipedia multimodal retrieval
  _fetch_search_result() — live SearXNG web search + optional summarization
  _fetch_music_result()  — music library retrieval

Runtime env flags
-----------------
env_bool / env_int are imported from env_utils.py.  They are re-read on every
chat turn so that flags like SAGE_RAG_ENABLED can be toggled at runtime without
restarting the app.  Startup configuration uses pydantic-settings BaseSettings
(settings.py / pg_settings.py) — one approach per purpose.

Lazy singletons
---------------
_rag_settings, _rag_injector, and _wiki_retriever are initialised on first
call rather than at import time, so a misconfigured DB or missing wiki package
never crashes the app at startup.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

# Module-level executor — threads are kept alive for the process lifetime so
# thread creation cost is paid once, not on every chat turn.
# max_workers=5 covers the five parallel fetches: doc-RAG, wiki-RAG, search, music, news.
_POOL = ThreadPoolExecutor(max_workers=5, thread_name_prefix="rag_par")

from env_utils import env_bool, env_int
from input_guard import sanitize_chunk
from rag_v1.config.rag_settings import RagSettings
from rag_v1.runtime.router_integration import RagInjector, _prepend_context
from sk_logging import get_logger

# Wiki multimodal RAG — optional dependency
try:
    from rag_v1.wiki.wiki_retriever import WikiRetriever
    _WIKI_AVAILABLE = True
except ImportError as _wiki_err:
    _WIKI_AVAILABLE = False
    import logging as _l; _l.getLogger("sage_kaizen.context_injector").warning(
        "Wiki RAG unavailable (ImportError: %s) — wiki retrieval disabled", _wiki_err
    )

# Live web search — optional dependency
try:
    from search.models import SearchEvidence
    from search.search_orchestrator import get_orchestrator
    from search.summarizer import build_raw_context, summarize_evidence
    _SEARCH_AVAILABLE = True
except ImportError as _search_err:
    _SEARCH_AVAILABLE = False
    import logging as _l; _l.getLogger("sage_kaizen.context_injector").warning(
        "Live search unavailable (ImportError: %s) — search disabled", _search_err
    )

# Music retrieval — optional dependency
try:
    from rag_v1.media.music_retriever import (
        MusicRetriever,
        detect_intent as _detect_music_intent,
        format_music_context,
    )
    _MUSIC_AVAILABLE = True
except ImportError as _music_err:
    _MUSIC_AVAILABLE = False
    import logging as _l; _l.getLogger("sage_kaizen.context_injector").warning(
        "Music retrieval unavailable (ImportError: %s) — music search disabled", _music_err
    )

# News resolver — optional dependency
try:
    from news.retrieval.news_resolver import resolve_news_context
    _NEWS_AVAILABLE = True
except ImportError as _news_err:
    _NEWS_AVAILABLE = False
    import logging as _l; _l.getLogger("sage_kaizen.context_injector").warning(
        "News resolver unavailable (ImportError: %s) — news context disabled", _news_err
    )

if TYPE_CHECKING:
    from search.models import SearchEvidence

_LOG = get_logger("sage_kaizen.context_injector")


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
    messages: list[dict[str, Any]],
    user_text: str,
    decision: Any,  # RouteDecision — duck-typed to avoid cross-package import
    rag_enabled: bool | None = None,
) -> tuple[list[dict[str, Any]], list]:
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

        lines: list[str] = []
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
    summarizer_base_url: str | None = None,
    summarizer_model_id: str | None = None,
) -> "tuple[str, SearchEvidence | None]":
    """
    Run a live SearXNG search and return (ctx_block, evidence).

    Only executes when decision.needs_search is True and the search
    module is available.  Falls back to raw snippet formatting when the
    summarization brain is unavailable.

    Summarizer priority:
      1. summarizer_base_url (dedicated CPU brain, port 8013)  — preferred
      2. fast_base_url       (FAST brain, port 8011)           — fallback
      3. raw snippets        (SAGE_SEARCH_SUMMARIZE=false or both unavailable)

    Controls:
      SAGE_SEARCH_ENABLED     (default true)  — master search on/off
      SAGE_SEARCH_SUMMARIZE   (default true)  — skip to raw snippets when false

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

    # Summarization pass — prefer dedicated summarizer brain when configured.
    # Set SAGE_SEARCH_SUMMARIZE=false to skip entirely and inject raw snippets.
    summary = ""
    summarize_enabled = env_bool("SAGE_SEARCH_SUMMARIZE", default=True)
    if summarize_enabled:
        # Pick the best available summarization endpoint
        _sum_url   = summarizer_base_url or fast_base_url
        _sum_model = summarizer_model_id or fast_model_id
        _sum_source = "summarizer" if summarizer_base_url else "fast_brain"
        if _sum_url and _sum_model:
            try:
                summary = summarize_evidence(
                    evidence=evidence,
                    fast_base_url=_sum_url,
                    fast_model_id=_sum_model,
                )
                if summary:
                    _LOG.info("search_rag | summarization via %s", _sum_source)
            except Exception:
                _LOG.warning(
                    "Summarization failed (source=%s url=%s); injecting raw snippets",
                    _sum_source, _sum_url,
                )
    else:
        _LOG.info("search_rag | SAGE_SEARCH_SUMMARIZE=false — injecting raw snippets")

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


def _fetch_news_result(user_text: str) -> str:
    """
    Resolve news context for the user message.

    Checks for market queries (yfinance point-in-time lookup) and news-intent
    queries (DB-first brief retrieval with stale fallback).  Returns a formatted
    <news_context> XML block, or "" if the query is not news-related or the
    news pipeline is unavailable.
    """
    if not _NEWS_AVAILABLE:
        return ""

    if not user_text:
        return ""

    try:
        ctx = resolve_news_context(user_text)
        if ctx is None:
            return ""
        block = ctx.to_xml_block()
        _LOG.info(
            "news_rag | source=%s stale=%s chars=%d",
            ctx.source, ctx.is_stale, len(block),
        )
        return block
    except Exception:
        _LOG.exception("News resolver failed; continuing without news context")
        return ""


def apply_rag_and_wiki_parallel(
    messages: list[dict[str, Any]],
    user_text: str,
    decision: Any,
    wiki_enabled: bool | None = None,
    fast_base_url: str | None = None,
    fast_model_id: str | None = None,
    summarizer_base_url: str | None = None,
    summarizer_model_id: str | None = None,
) -> tuple[list[dict[str, Any]], list, list, "SearchEvidence | None", str]:
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
    fast_base_url         : FAST brain base URL — fallback summarization endpoint.
    fast_model_id         : FAST brain model alias.
    summarizer_base_url   : Dedicated CPU summarizer base URL (port 8013).
                            When set, used instead of fast_base_url for search
                            summarization, freeing the FAST brain slot for
                            the main conversation response.
    summarizer_model_id   : Summarizer model alias (from brains.yaml alias key).

    Returns
    -------
    (messages, rag_sources, wiki_images, search_evidence, music_context)
    search_evidence is None when live search was not triggered.
    music_context is "" when music retrieval was not triggered.
    """
    rag_fut    = _POOL.submit(apply_rag, messages, user_text, decision)
    wiki_fut   = _POOL.submit(_fetch_wiki_result, user_text, decision, wiki_enabled)
    search_fut = _POOL.submit(
        _fetch_search_result, user_text, decision,
        fast_base_url, fast_model_id,
        summarizer_base_url, summarizer_model_id,
    )
    music_fut  = _POOL.submit(_fetch_music_result, user_text, decision)
    news_fut   = _POOL.submit(_fetch_news_result, user_text)

    # Per-worker timeout: prevents a hung SearXNG fetch, slow jina-clip GPU
    # restore, or stalled DB query from blocking the turn indefinitely.
    # Values are generous (doc-RAG: 15 s, wiki: 20 s, search+summarize: 30 s,
    # music: 10 s, news: 10 s) — real work is much faster; these are safety ceilings.
    # TimeoutError is caught and logged; the turn continues with partial context.
    _WORKER_TIMEOUTS = {"rag": 15, "wiki": 20, "search": 30, "music": 10, "news": 10}

    try:
        rag_messages, rag_sources = rag_fut.result(timeout=_WORKER_TIMEOUTS["rag"])
    except Exception:
        _LOG.exception("RAG worker timed out or failed; continuing without doc-RAG")
        rag_messages, rag_sources = messages, []

    try:
        wiki_ctx_block, wiki_images = wiki_fut.result(timeout=_WORKER_TIMEOUTS["wiki"])
    except Exception:
        _LOG.exception("Wiki worker timed out or failed; continuing without wiki context")
        wiki_ctx_block, wiki_images = "", []

    try:
        search_ctx_block, search_evidence = search_fut.result(timeout=_WORKER_TIMEOUTS["search"])
    except Exception:
        _LOG.exception("Search worker timed out or failed; continuing without search context")
        search_ctx_block, search_evidence = "", None

    try:
        music_ctx_block = music_fut.result(timeout=_WORKER_TIMEOUTS["music"])
    except Exception:
        _LOG.exception("Music worker timed out or failed; continuing without music context")
        music_ctx_block = ""

    try:
        news_ctx_block = news_fut.result(timeout=_WORKER_TIMEOUTS["news"])
    except Exception:
        _LOG.exception("News worker timed out or failed; continuing without news context")
        news_ctx_block = ""

    # ── Token budget guardrails ────────────────────────────────────────────
    # Trim wiki and search context blocks before injection so that a single
    # source cannot consume more than its per-brain char budget.
    # Defaults: FAST 4 000 wiki / 2 000 search; ARCH 16 000 wiki / 6 000 search.
    # Override via env vars without touching this file.
    is_fast = getattr(decision, "brain", "FAST") == "FAST"

    wiki_max = env_int(
        "SAGE_RAG_WIKI_FAST_MAX_CHARS" if is_fast else "SAGE_RAG_WIKI_ARCH_MAX_CHARS",
        default=4_000 if is_fast else 16_000,
    )
    if wiki_ctx_block and len(wiki_ctx_block) > wiki_max:
        wiki_ctx_block = wiki_ctx_block[:wiki_max] + "\n[... wiki context trimmed to budget ...]"
        _LOG.info("wiki_rag | trimmed to budget | brain=%s max_chars=%d", decision.brain, wiki_max)

    search_max = env_int(
        "SAGE_SEARCH_FAST_MAX_CHARS" if is_fast else "SAGE_SEARCH_ARCH_MAX_CHARS",
        default=2_000 if is_fast else 6_000,
    )
    if search_ctx_block and len(search_ctx_block) > search_max:
        search_ctx_block = search_ctx_block[:search_max] + "\n[... search context trimmed to budget ...]"
        _LOG.info("search_rag | trimmed to budget | brain=%s max_chars=%d", decision.brain, search_max)

    # Structured injection summary for post-turn analysis
    _LOG.info(
        "context_injection_json %s",
        json.dumps({
            "brain":             getattr(decision, "brain", "FAST"),
            "rag_chunks":        len(rag_sources),
            "wiki_chars":        len(wiki_ctx_block) if wiki_ctx_block else 0,
            "search_chars":      len(search_ctx_block) if search_ctx_block else 0,
            "music_chars":       len(music_ctx_block) if music_ctx_block else 0,
            "news_chars":        len(news_ctx_block) if news_ctx_block else 0,
            "wiki_images":       len(wiki_images),
            "wiki_trimmed":      bool(wiki_ctx_block and len(wiki_ctx_block) >= wiki_max),
            "search_trimmed":    bool(search_ctx_block and len(search_ctx_block) >= search_max),
        }),
    )

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

    # Inject news context outermost (closest to the user question; only when triggered)
    if news_ctx_block:
        for i in reversed(range(len(out))):
            if out[i].get("role") == "user":
                prefix = f"{news_ctx_block}\n\n"
                out[i] = {**out[i], "content": _prepend_context(out[i]["content"], prefix)}
                break

    return out, rag_sources, wiki_images if wiki_ctx_block else [], search_evidence, music_ctx_block
