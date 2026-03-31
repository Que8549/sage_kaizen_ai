"""
search/

Live web search pipeline for Sage Kaizen.

Public surface
--------------
SearchEvidence          — normalized result schema
WebResult               — single deduplicated result
get_orchestrator()      — lazy singleton SearchOrchestrator
summarize_evidence()    — FAST-brain summarization pass
build_raw_context()     — fallback raw snippet formatter
format_search_sources_markdown() — UI citation block
"""
from search.citations import format_search_sources_markdown
from search.models import SearchEvidence, WebResult
from search.search_orchestrator import SearchOrchestrator, get_orchestrator
from search.summarizer import build_raw_context, summarize_evidence

__all__ = [
    "SearchEvidence",
    "WebResult",
    "SearchOrchestrator",
    "get_orchestrator",
    "summarize_evidence",
    "build_raw_context",
    "format_search_sources_markdown",
]
