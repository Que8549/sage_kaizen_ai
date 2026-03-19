-- rag_v1/db/media_schema.sql
--
-- CLIP + CLAP dual-modal media embedding schema for Sage Kaizen.
--
--   image_embeddings: 1024-dim jinaai/jina-clip-v2 vectors (cosine)
--   audio_embeddings:  512-dim laion/clap-htsat-unfused vectors (cosine)
--
-- Run once against your PostgreSQL database:
--   psql -U sage -d sage_kaizen -f rag_v1/db/media_schema.sql
--
-- Prerequisites:
--   CREATE EXTENSION IF NOT EXISTS vector;       (pgvector)
--   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- --------------------------------------------------------------------------- --
-- media_files: one row per ingested file                                        --
-- --------------------------------------------------------------------------- --

CREATE TABLE IF NOT EXISTS media_files (
    media_id     uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path    text        NOT NULL,       -- absolute path on disk
    modality     text        NOT NULL CHECK (modality IN ('image', 'audio', 'video')),
    content_hash bytea       NOT NULL,       -- SHA-256 of raw file bytes; used for dedup
    file_size_b  bigint,                     -- file size in bytes
    duration_s   float,                      -- for audio (seconds)
    width        int,                        -- for image (pixels)
    height       int,
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    metadata     jsonb       NOT NULL DEFAULT '{}'
);

-- Uniqueness constraint: one row per (path, hash) pair.
-- Re-ingesting the same file is idempotent via ON CONFLICT DO NOTHING.
CREATE UNIQUE INDEX IF NOT EXISTS media_files_path_hash
    ON media_files (file_path, content_hash);

CREATE INDEX IF NOT EXISTS media_files_modality
    ON media_files (modality);


-- --------------------------------------------------------------------------- --
-- image_embeddings: one row per image file                                      --
-- Populated by jina-clip-v2 (port 8031, reuses wiki embed service).            --
-- --------------------------------------------------------------------------- --

CREATE TABLE IF NOT EXISTS image_embeddings (
    embed_id   bigserial    PRIMARY KEY,
    media_id   uuid         NOT NULL REFERENCES media_files (media_id) ON DELETE CASCADE,
    embedding  vector(1024) NOT NULL,        -- jina-clip-v2 1024-dim L2-normalized
    created_at timestamptz  NOT NULL DEFAULT now()
);

-- HNSW index for fast approximate cosine similarity search.
CREATE INDEX IF NOT EXISTS image_embed_hnsw
    ON image_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS image_embed_media_id
    ON image_embeddings (media_id);


-- --------------------------------------------------------------------------- --
-- audio_embeddings: one row per audio file                                      --
-- Populated by laion/clap-htsat-unfused (port 8040, CLAP embed service).       --
-- --------------------------------------------------------------------------- --

CREATE TABLE IF NOT EXISTS audio_embeddings (
    embed_id   bigserial   PRIMARY KEY,
    media_id   uuid        NOT NULL REFERENCES media_files (media_id) ON DELETE CASCADE,
    embedding  vector(512) NOT NULL,         -- CLAP clap-htsat-unfused 512-dim L2-normalized
    created_at timestamptz NOT NULL DEFAULT now()
);

-- HNSW index for fast approximate cosine similarity search.
CREATE INDEX IF NOT EXISTS audio_embed_hnsw
    ON audio_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS audio_embed_media_id
    ON audio_embeddings (media_id);


-- --------------------------------------------------------------------------- --
-- Convenience views                                                              --
-- --------------------------------------------------------------------------- --

CREATE OR REPLACE VIEW image_embed_view AS
SELECT
    ie.embed_id,
    mf.media_id,
    mf.file_path,
    mf.width,
    mf.height,
    mf.metadata,
    ie.embedding,
    ie.created_at
FROM image_embeddings ie
JOIN media_files mf ON mf.media_id = ie.media_id;

CREATE OR REPLACE VIEW audio_embed_view AS
SELECT
    ae.embed_id,
    mf.media_id,
    mf.file_path,
    mf.duration_s,
    mf.metadata,
    ae.embedding,
    ae.created_at
FROM audio_embeddings ae
JOIN media_files mf ON mf.media_id = ae.media_id;
