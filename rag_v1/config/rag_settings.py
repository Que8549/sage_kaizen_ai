from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    source_id: str
    chunk_id: int
    content: str
    score: float
    metadata: dict

class RagSettings(BaseSettings):
    """
    RAG configuration.

    Values are populated in this order:
        1. .env file (env_file below)
        2. OS environment variables
        3. Default values defined in this class

    Example .env file (project root):

        PG_USER=sage
        PG_PASSWORD=YourRealPassword
        PG_DB=sage_kaizen
    """

    # Tell pydantic to load environment variables from .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",   # ✅ KEY FIX: ignore unrelated .env keys
    )

    # -------------------------
    # PostgreSQL settings
    # -------------------------
    # These values are automatically populated by pydantic:
    # - First from .env file
    # - Then from system environment variables
    # - Otherwise default values below are used
    #
    # So:
    # self.pg_user       ← loaded from PG_USER in .env
    # self.pg_password   ← loaded from PG_PASSWORD in .env
    # self.pg_db         ← loaded from PG_DB in .env
    #
    # If not found, defaults below apply.

    pg_user: str = "my_user"
    pg_password: str = "my_pwd"
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_db: str = "my_db"

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

    # -------------------------
    # Derived DSN (built dynamically)
    # -------------------------
    @property
    def pg_dsn(self) -> str:
        """
        This builds the DSN using values populated above.

        IMPORTANT:
        self.pg_user and self.pg_password are already populated
        by pydantic from:
            - .env
            - OR environment variables
            - OR defaults

        Nothing manual is required here.
        """
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )



