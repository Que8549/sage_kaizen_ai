"""
search/summarizer.py

Lightweight pre-summarization pass: takes raw SearXNG results and calls the
FAST brain to extract key facts before context injection.

Why summarize before injection?
  Raw snippets contain repetition, boilerplate, and off-topic text.
  A short FAST-brain pass (max_tokens=350) produces a clean, cited paragraph
  that fits ~70 tokens instead of ~600 raw tokens — better signal, lower
  context cost.

Fallback behaviour
  If the FAST brain is unavailable, times out, or returns an empty string,
  summarize_evidence() returns "" and context_injector.py falls back to
  injecting raw snippets directly.  The turn is never blocked.
"""
from __future__ import annotations

from openai_client import HttpTimeouts, stream_chat_completions
from search.models import SearchEvidence
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.search.summarizer")

# Short timeout specifically for the summarization sub-call.
# It runs concurrently with doc-RAG and wiki-RAG in a thread pool,
# so a generous read timeout (15 s) is acceptable without blocking the turn.
_SUMMARIZER_TIMEOUTS = HttpTimeouts(connect_s=2.0, read_s=15.0)

_SUMMARIZE_SYSTEM = (
    "You are a precise research assistant. Given web search results and a user query, "
    "extract only the key facts directly relevant to the query. "
    "Be concise: 3–5 sentences maximum. "
    "Cite sources inline as [Source Name] or [Source Name, YYYY-MM-DD]. "
    "Do not invent or infer facts beyond what the search results state. "
    "Do not repeat the user's question."
)


def summarize_evidence(
    evidence: SearchEvidence,
    fast_base_url: str,
    fast_model_id: str,
    timeouts: HttpTimeouts | None = None,
) -> str:
    """
    Call the FAST brain to distil search results into a short cited paragraph.

    Parameters
    ----------
    evidence       : SearchEvidence returned by SearchOrchestrator.
    fast_base_url  : e.g. "http://127.0.0.1:8011"
    fast_model_id  : Model alias reported by the FAST brain /v1/models.
    timeouts       : Override default _SUMMARIZER_TIMEOUTS if needed.

    Returns
    -------
    A non-empty summary string, or "" on any failure.
    """
    if evidence.empty:
        return ""

    to = timeouts or _SUMMARIZER_TIMEOUTS

    # Build numbered results block for the prompt
    lines: list[str] = []
    for i, r in enumerate(evidence.results, 1):
        date_part = f" | {r.published_date}" if r.published_date else ""
        lines.append(
            f"[{i}] {r.title} | {r.source_engine}{date_part}\n"
            f"URL: {r.url}\n"
            f"Snippet: {r.snippet}"
        )
    results_block = "\n\n".join(lines)

    user_content = (
        f"User query: {evidence.query}\n\n"
        f"Search results:\n{results_block}\n\n"
        "Provide a concise summary of the key facts, with inline source citations."
    )

    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    try:
        chunks = list(stream_chat_completions(
            base_url  = fast_base_url,
            model     = fast_model_id,
            messages  = messages,
            temperature = 0.1,
            top_p     = 1.0,
            max_tokens = 350,
            timeouts  = to,
        ))
        summary = "".join(chunks).strip()
        _LOG.info(
            "summarizer | results=%d | summary_chars=%d",
            len(evidence.results), len(summary),
        )
        return summary
    except Exception:
        _LOG.warning(
            "Search summarization failed (fast_url=%s); injecting raw snippets instead",
            fast_base_url,
        )
        return ""


def build_raw_context(evidence: SearchEvidence, snippet_max_chars: int = 300) -> str:
    """
    Fallback: format raw results as a plain text block for direct injection.
    Used when summarization is skipped or fails.

    Format mirrors the <context> blocks used by doc-RAG and wiki-RAG:
        [1] Title | engine | date
        URL: ...
        Snippet: ...

        ---

        [2] ...
    """
    lines: list[str] = []
    for i, r in enumerate(evidence.results, 1):
        date_part = f" | {r.published_date}" if r.published_date else ""
        snippet = r.snippet
        if len(snippet) > snippet_max_chars:
            snippet = snippet[:snippet_max_chars] + "..."
        lines.append(
            f"[{i}] {r.title} | {r.source_engine}{date_part}\n"
            f"URL: {r.url}\n"
            f"Snippet: {snippet}"
        )
    return "\n\n---\n\n".join(lines)
