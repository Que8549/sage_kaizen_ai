"""
review_service/nodes/web_researcher.py — Web Research Node (no LLM)

Queries SearXNG for up to 4 targeted searches derived from the changed
modules and framework context. Results are injected as a <web_research>
block in the architect_reviewer prompt.

Search topics are auto-generated based on:
  - Python framework/library imports in changed files
  - llama.cpp if brains.yaml changed
  - LangGraph/LangChain if review_service/ changed
  - Performance keywords for modules touching inference or RAG
  - General "latest performance" for the primary changed module

If SearXNG is unavailable, this node logs a warning and continues
with an empty web_research field — the review does not fail.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from sk_logging import get_logger
from ..state import ReviewState

_LOG = get_logger("sage_kaizen.review_service.web_researcher")

_MAIN_ROOT = Path("F:/Projects/sage_kaizen_ai")
_MAX_QUERIES = 4
_MAX_SNIPPET_CHARS = 400
_MAX_WEB_BLOCK_CHARS = 6_000


async def web_researcher_node(state: ReviewState) -> dict:
    """
    Generate search queries from scope, run them via SearXNG, return
    formatted web context for the ARCHITECT reviewer.
    """
    changed_files = state.get("changed_files", [])
    brains_yaml   = state.get("brains_yaml", "")

    queries = _build_queries(changed_files, brains_yaml)
    if not queries:
        _LOG.info("review.web_research | no queries generated")
        return {"web_research": ""}

    _LOG.info("review.web_research | queries=%s", queries)

    # Run all searches in parallel via asyncio.to_thread (httpx is synchronous)
    tasks = [
        asyncio.to_thread(_searxng_search, q)
        for q in queries[:_MAX_QUERIES]
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    parts: list[str] = []
    for query, result in zip(queries, results):
        if isinstance(result, Exception):
            _LOG.warning("review.web_research.fail | query=%r error=%s", query[:60], result)
            continue
        if result:
            parts.append(f"### {query}\n{result}")

    if not parts:
        return {"web_research": ""}

    web_block = "<web_research>\n" + "\n\n".join(parts) + "\n</web_research>"
    web_block = web_block[:_MAX_WEB_BLOCK_CHARS]

    _LOG.info("review.web_research | block_chars=%d", len(web_block))
    return {"web_research": web_block}


# ── Query generation ──────────────────────────────────────────────────────

def _build_queries(changed_files: list[str], brains_yaml: str) -> list[str]:
    queries: list[str] = []

    # llama.cpp updates if brains.yaml changed
    if any("brains" in f or "server_manager" in f for f in changed_files):
        queries.append("llama.cpp latest performance improvements 2025 2026")
        queries.append("llama-server flags optimization Qwen CUDA")

    # LangGraph updates if review_service changed
    if any("review_service" in f for f in changed_files):
        queries.append("LangGraph 1.x stateful agents performance best practices 2026")

    # RAG/embedding updates
    if any("rag_v1" in f or "embed" in f for f in changed_files):
        queries.append("pgvector HNSW performance optimization 2025 2026")
        queries.append("BGE-M3 embedding inference optimization")

    # Router / inference changes
    if any(f in ("router.py", "chat_service.py", "inference_session.py") for f in changed_files):
        queries.append("OpenAI compatible API streaming latency optimization Python 2025")

    # Streamlit performance
    if any("streamlit" in f for f in changed_files):
        queries.append("Streamlit performance optimization session state 2025")

    # General: always add a "current best practices" query for primary changed module
    if changed_files:
        primary = _primary_module(changed_files)
        if primary and not queries:
            queries.append(f"{primary} Python performance best practices 2025 2026")

    return queries[:_MAX_QUERIES]


def _primary_module(files: list[str]) -> str:
    """Pick the most important changed file as the primary search subject."""
    priority_names = ["chat_service", "router", "inference_session", "voice_pipeline"]
    for name in priority_names:
        for f in files:
            if name in f:
                return name.replace("_", " ")
    # Fallback: first .py file's stem
    for f in files:
        if f.endswith(".py") and not f.startswith("[voice]"):
            return Path(f).stem.replace("_", " ")
    return ""


# ── SearXNG search (synchronous — called via asyncio.to_thread) ───────────

def _searxng_search(query: str) -> str:
    """
    Call the existing SearXNG search client and format results as markdown.
    Returns empty string on any failure so the node can continue gracefully.
    """
    try:
        from search.search_orchestrator import get_orchestrator
        evidence = get_orchestrator().search(
            query=query,
            categories=["it", "science"],
            brain="ARCHITECT",
        )
        if evidence.empty or not evidence.results:
            return ""
        lines: list[str] = []
        for r in evidence.results[:5]:
            snippet = (r.content or "")[:_MAX_SNIPPET_CHARS].replace("\n", " ")
            lines.append(f"- **[{r.title}]({r.url})**\n  {snippet}")
        return "\n".join(lines)
    except Exception as exc:
        _LOG.warning("web_researcher._searxng_search failed: %s", exc)
        return ""
