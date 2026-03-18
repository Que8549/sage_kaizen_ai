-- rag_v1/db/media_schema.sql
--
-- Cross-modal media embedding schema for Sage Kaizen.
-- Uses pgvector for 768-dim LanguageBind cosine similarity search.
--
-- Run once against your PostgreSQL database:
--   psql -U sage -d sage_kaizen -f rag_v1/db/media_schema.sql
--
-- Prerequisites:
--   CREATE EXTENSION IF NOT EXISTS vector;  (pgvector)
--   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────────────────────────────────────── --
-- media_files: one row per ingested file                                       --
-- ─────────────────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS media_files (
    media_id     uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path    text        NOT NULL,       -- absolute path on disk
    modality     text        NOT NULL CHECK (modality IN ('image', 'audio', 'video')),
    content_hash bytea       NOT NULL,       -- SHA-256 of raw file bytes; used for dedup
    file_size_b  bigint,                     -- file size in bytes
    duration_s   float,                      -- for audio / video (seconds)
    width        int,                        -- for image / video (pixels)
    height       int,
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    metadata     jsonb       NOT NULL DEFAULT '{}'
);

-- Uniqueness constraint: one row per (path, hash) pair.
-- Re-ingesting the same file is idempotent via ON CONFLICT DO NOTHING.
CREATE UNIQUE INDEX IF NOT EXISTS media_files_path_hash
    ON media_files (file_path, content_hash);

-- Fast lookup by modality for modality-filtered searches
CREATE INDEX IF NOT EXISTS media_files_modality
    ON media_files (modality);


-- ─────────────────────────────────────────────────────────────────────────── --
-- media_embeddings: one or more embeddings per file                            --
-- Images:  one row (frame_index = NULL)                                        --
-- Audio:   one row (frame_index = NULL)                                        --
-- Video:   one row per sampled frame (frame_index = 0, 1, 2, …)               --
-- ─────────────────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS media_embeddings (
    embed_id     bigserial   PRIMARY KEY,
    media_id     uuid        NOT NULL REFERENCES media_files (media_id) ON DELETE CASCADE,
    frame_index  int,                        -- NULL for images and audio
    time_s       float,                      -- timestamp in source video (seconds)
    embedding    vector(768) NOT NULL,       -- LanguageBind 768-dim L2-normalized vector
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- HNSW index for fast approximate cosine similarity search.
-- Covers all modalities in a single index; modality filter is pushed as a
-- WHERE clause on the joined media_files.modality column.
CREATE INDEX IF NOT EXISTS media_embed_hnsw
    ON media_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Lookup by source file (used for re-ingest dedup check and cascade deletes)
CREATE INDEX IF NOT EXISTS media_embed_media_id
    ON media_embeddings (media_id);

-- ─────────────────────────────────────────────────────────────────────────── --
-- Convenience view: join file metadata with each embedding row                 --
-- ─────────────────────────────────────────────────────────────────────────── --

CREATE OR REPLACE VIEW media_embed_view AS
SELECT
    me.embed_id,
    mf.media_id,
    mf.file_path,
    mf.modality,
    mf.duration_s,
    mf.width,
    mf.height,
    me.frame_index,
    me.time_s,
    mf.metadata,
    me.embedding,
    me.created_at
FROM media_embeddings me
JOIN media_files mf ON mf.media_id = me.media_id;
