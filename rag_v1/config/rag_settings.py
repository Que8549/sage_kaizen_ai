from __future__ import annotations

from pydantic import BaseModel

from pg_settings import PgSettings


class RetrievedChunk(BaseModel):
    source_id: str
    chunk_id: int
    content: str
    score: float
    metadata: dict

class RagSettings(PgSettings):
    """
    RAG configuration.

    PG_* fields (pg_user, pg_password, pg_host, pg_port, pg_db, pg_dsn)
    are inherited from PgSettings and loaded from .env / environment variables.
    """

    # -------------------------
    # Embedding server
    # -------------------------
    embed_base_url: str = "http://127.0.0.1:8020/v1"
    embed_model: str = "bge-m3-embed"

    # -------------------------
    # Retrieval tuning
    # -------------------------
    top_k: int = 6
    chunk_chars: int = 1200
    chunk_overlap: int = 200
    # Minimum relevance score to inject a chunk (0.0 = no filter, 1.0 = exact match only).
    # Score = 1 / (1 + cosine_distance).  A value of ~0.40 filters out near-noise results.
    # Set via env var SAGE_RAG_MIN_SCORE or .env file.
    min_score: float = 0.0



