from __future__ import annotations


from rag_v1.config.rag_settings import RetrievedChunk
from rag_v1.retrieve.retriever import PgvectorRetriever
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.rag_pipeline")


class RagPipeline:

    def __init__(self, cfg):
        self.retriever = PgvectorRetriever(cfg)

    def build_context(self, user_query: str, top_k: int) -> tuple[str, list[RetrievedChunk]]:
        """Retrieve chunks and build a context string.

        Returns:
            (context_str, kept_chunks) -- context_str is empty string when nothing
            passes the min_score filter or the noise-cluster gate rejects all results;
            kept_chunks is the parallel list used to render citations.

        Noise-cluster gate:
            When all returned chunks cluster in a tight score band AND the best result
            is mediocre, the query has no true semantic neighbor in the index (generic
            word-bleed). All results are discarded rather than injecting irrelevant
            context into the LLM prompt.
        """
        chunks = self.retriever.retrieve(user_query, top_k)

        min_score: float = getattr(self.retriever.cfg, "min_score", 0.0)
        kept: list[RetrievedChunk] = []
        lines: list[str] = []
        for c in chunks:
            if c.score >= min_score:
                kept.append(c)
                lines.append(
                    f"[{c.source_id}#chunk{c.chunk_id} | score={c.score:.3f}]\n{c.content}"
                )

        # Noise-cluster gate -- detects generic word-bleed false positives.
        # Triggers when: enough results returned AND scores cluster tightly
        # AND the best result is still mediocre (no true semantic neighbor found).
        cluster_min: int = getattr(self.retriever.cfg, "cluster_min_size", 3)
        cluster_spread: float = getattr(self.retriever.cfg, "cluster_max_spread", 0.030)
        cluster_floor: float = getattr(self.retriever.cfg, "cluster_top1_floor", 0.800)

        if len(kept) >= cluster_min:
            scores = [c.score for c in kept]
            spread = max(scores) - min(scores)
            top1 = max(scores)
            if spread < cluster_spread and top1 < cluster_floor:
                _LOG.info(
                    "RAG noise-cluster gate: rejected %d chunks "
                    "(spread=%.4f < %.4f, top1=%.4f < %.4f) query=%.80r",
                    len(kept), spread, cluster_spread, top1, cluster_floor, user_query,
                )
                return "", []

        ctx = "\n\n---\n\n".join(lines)
        return ctx, kept
