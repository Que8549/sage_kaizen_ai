"""
search/models.py

Normalized, citation-aware schema for live web search results.

WebResult      — a single deduplicated result from SearXNG.
SearchEvidence — the full result set for one query turn; passed from
                 SearchOrchestrator → Summarizer → context injector → UI.

These types are intentionally independent of SearXNG internals so that
swapping the search backend later requires changes only in searxng_client.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WebResult:
    """
    One normalized search result.

    Fields
    ------
    title          : Headline or page title.
    url            : Canonical result URL (deduplicated key).
    snippet        : Content excerpt, truncated to snippet_max_chars.
    source_engine  : Primary engine that returned this result (e.g. "brave").
    category       : SearXNG category (general | news | science | technology).
    score          : SearXNG relevance score, normalized to 0.0–1.0.
    published_date : ISO-8601 date string when available, None otherwise.
    """
    title: str
    url: str
    snippet: str
    source_engine: str
    category: str
    score: float
    published_date: str | None = None


@dataclass(frozen=True)
class SearchEvidence:
    """
    Aggregate result for one search call.

    Passed from SearchOrchestrator through Summarizer to context_injector.
    The summarized_text field is populated by Summarizer after the raw
    results are fetched; if summarization fails it remains "".

    Fields
    ------
    query              : The original user query text.
    results            : Filtered, ranked WebResult tuple.
    fetched_at         : ISO-8601 UTC timestamp of the SearXNG call.
    categories_queried : Categories passed to SearXNG (e.g. ("news", "technology")).
    summarized_text    : FAST-brain condensed summary; "" if skipped or failed.
    """
    query: str
    results: tuple[WebResult, ...]
    fetched_at: str
    categories_queried: tuple[str, ...]
    summarized_text: str = ""

    @property
    def empty(self) -> bool:
        return len(self.results) == 0
