from pydantic_settings import BaseSettings
from pydantic import BaseModel

class RagSettings(BaseSettings):
    pg_dsn: str = "postgresql://sage:sage@127.0.0.1:5432/sage_kaizen"
    embed_base_url: str = "http://127.0.0.1:8020/v1"
    embed_model: str = "bge-m3-embed"
    top_k: int = 6

class RetrievedChunk(BaseModel):
    source_id: str
    chunk_id: int
    content: str
    score: float
    metadata: dict
