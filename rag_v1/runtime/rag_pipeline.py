from __future__ import annotations

from typing import List, Tuple

from rag_v1.config.rag_settings import RetrievedChunk
from rag_v1.retrieve.retriever import PgvectorRetriever


class RagPipeline:

    def __init__(self, cfg):
        self.retriever = PgvectorRetriever(cfg)

    def build_context(self, user_query: str, top_k: int) -> Tuple[str, List[RetrievedChunk]]:
        """Retrieve chunks and build a context string.

        Returns:
            (context_str, kept_chunks) — context_str is empty string when nothing
            passes the min_score filter; kept_chunks is the parallel list used to
            render citations.
        """
        chunks = self.retriever.retrieve(user_query, top_k)

        min_score: float = getattr(self.retriever.cfg, "min_score", 0.0)
        kept: List[RetrievedChunk] = []
        lines: List[str] = []
        for c in chunks:
            if c.score >= min_score:
                kept.append(c)
                lines.append(
                    f"[{c.source_id}#chunk{c.chunk_id} | score={c.score:.3f}]\n{c.content}"
                )

        ctx = "\n\n---\n\n".join(lines)
        return ctx, kept
