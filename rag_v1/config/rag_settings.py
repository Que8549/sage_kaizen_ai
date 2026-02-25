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
    # Maximum cosine distance allowed in the SQL WHERE clause (pre-filters before Python).
    # Cosine distance range: 0.0 (identical) -> 2.0 (opposite).  0.40 ~= score > 0.71.
    # Env: MAX_DISTANCE
    max_distance: float = 0.40

    # -------------------------
    # Noise-cluster gate
    # -------------------------
    # Rejects result sets where all chunks cluster in a tight score band with a mediocre
    # top-1 score -- the signature of "no true semantic neighbor, just word-bleed noise."
    #
    # Gate triggers when ALL three conditions hold:
    #   len(kept) >= cluster_min_size
    #   AND (max_score - min_score) < cluster_max_spread   (scores bunched together)
    #   AND max_score < cluster_top1_floor                 (best result is mediocre)
    #
    # Env: CLUSTER_MIN_SIZE  (minimum results returned before gate evaluates)
    cluster_min_size: int = 3
    # Env: CLUSTER_MAX_SPREAD  (score-space spread; scores range 0.0-1.0)
    cluster_max_spread: float = 0.030
    # Env: CLUSTER_TOP1_FLOOR  (top-1 score must exceed this to skip the gate)
    cluster_top1_floor: float = 0.800



