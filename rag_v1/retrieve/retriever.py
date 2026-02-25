from rag_v1.db.pg import get_conn
from rag_v1.embed.embed_client import EmbedClient
from rag_v1.config.rag_settings import RetrievedChunk

class PgvectorRetriever:

    def __init__(self, cfg):
        self.cfg = cfg
        self.embed = EmbedClient(cfg.embed_base_url, cfg.embed_model)

    def retrieve(self, query, top_k=None):
        k = top_k or self.cfg.top_k
        q_emb = self.embed.embed([query])[0]

        max_dist = getattr(self.cfg, "max_distance", 0.5)
        sql = """
        SELECT source_id, chunk_id, content, metadata,
               (embedding <=> %s::vector) AS distance
        FROM rag_chunks
        WHERE embedding IS NOT NULL
          AND (embedding <=> %s::vector) < %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """

        with get_conn(self.cfg.pg_dsn) as conn:
            rows = conn.execute(sql, (q_emb, q_emb, max_dist, q_emb, k)).fetchall()

        results = []
        for r in rows:
            dist = float(r["distance"])
            results.append(
                RetrievedChunk(
                    source_id=r["source_id"],
                    chunk_id=int(r["chunk_id"]),
                    content=r["content"],
                    metadata=dict(r["metadata"]),
                    score=1.0 / (1.0 + dist),
                )
            )

        return results
