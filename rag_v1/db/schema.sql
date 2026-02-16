
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
  id          bigserial PRIMARY KEY,
  source_id   text NOT NULL,
  chunk_id    int  NOT NULL,
  content     text NOT NULL,
  metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
  embedding   vector(1024),  -- adjust to model dimension
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (source_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS rag_chunks_source_id_idx
ON rag_chunks(source_id);

CREATE INDEX IF NOT EXISTS rag_chunks_metadata_gin
ON rag_chunks USING gin (metadata);

CREATE INDEX IF NOT EXISTS rag_chunks_hnsw_cos_idx
ON rag_chunks
USING hnsw (embedding vector_cosine_ops);




