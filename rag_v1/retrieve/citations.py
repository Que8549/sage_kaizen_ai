from __future__ import annotations


from rag_v1.config.rag_settings import RetrievedChunk


def format_sources_markdown(chunks: list[RetrievedChunk]) -> str:
    """Return an inline markdown sources block for appending after a streamed response.

    Example output:
        ---
        **Sources**
        - `my_doc` · chunk 3 · relevance: 91%
        - `other_doc` · chunk 7 · relevance: 84%
    """
    if not chunks:
        return ""
    lines = ["---", "**Sources**"]
    for c in chunks:
        pct = int(c.score * 100)
        lines.append(f"- `{c.source_id}` \u00b7 chunk {c.chunk_id} \u00b7 relevance: {pct}%")
    return "\n".join(lines)
