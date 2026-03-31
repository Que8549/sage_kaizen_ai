"""
search/citations.py

Citation formatting for live web search results.
Matches the visual style of rag_v1/retrieve/citations.py so all three
source types (doc-RAG, wiki-RAG, live-web) look consistent in the UI.
"""
from __future__ import annotations

from search.models import SearchEvidence


def format_search_sources_markdown(evidence: SearchEvidence) -> str:
    """
    Return a markdown block for display below the assistant response.

    Output format (mirrors format_sources_markdown in rag_v1/retrieve/citations.py):

        ---
        **Live Web** · news, technology · fetched 2026-03-30 22:15 UTC
        - [Title](url) · brave · 2026-03-29
        - [Title2](url2) · duckduckgo news

    Returns "" when evidence is empty or has no results.
    """
    if evidence.empty:
        return ""

    cats    = ", ".join(evidence.categories_queried)
    fetched = evidence.fetched_at[:16].replace("T", " ") + " UTC"

    lines = [
        "---",
        f"**Live Web** \u00b7 {cats} \u00b7 fetched {fetched}",
    ]
    for r in evidence.results:
        date_part = f" \u00b7 {r.published_date[:10]}" if r.published_date else ""
        title     = r.title or r.url
        lines.append(f"- [{title}]({r.url}) \u00b7 {r.source_engine}{date_part}")

    return "\n".join(lines)
