from rag_v1.retrieve.retriever import PgvectorRetriever

class RagPipeline:

    def __init__(self, cfg):
        self.retriever = PgvectorRetriever(cfg)

    def build_context(self, user_query, top_k):
        chunks = self.retriever.retrieve(user_query, top_k)

        min_score: float = getattr(self.retriever.cfg, "min_score", 0.0)
        lines = []
        for c in chunks:
            if c.score >= min_score:
                lines.append(
                    f"[{c.source_id}#chunk{c.chunk_id} | score={c.score:.3f}]\n{c.content}"
                )

        return "\n\n---\n\n".join(lines)
