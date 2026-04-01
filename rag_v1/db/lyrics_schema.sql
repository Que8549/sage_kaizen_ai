-- rag_v1/db/lyrics_schema.sql
--
-- Lyrics embedding schema for Sage Kaizen.
-- Enables "find MP3 by song lyric" via BGE-M3 text embeddings (1024-dim).
--
-- Run once against your PostgreSQL database:
--   psql -U sage -d sage_kaizen -f rag_v1/db/lyrics_schema.sql
--
-- Prerequisites:
--   CREATE EXTENSION IF NOT EXISTS vector;       (pgvector)
--   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
--   media_files table must already exist (rag_v1/db/media_schema.sql)


-- --------------------------------------------------------------------------- --
-- lyrics_fetch_log: one row per audio file — tracks every fetch attempt        --
--                                                                               --
-- status values:                                                                --
--   'ok'        — lyrics found and embedded; skip on re-run                    --
--   'not_found' — Genius confirmed nothing there; skip on re-run               --
--   'error'     — network/API failure; retry on next run                       --
-- --------------------------------------------------------------------------- --

CREATE TABLE IF NOT EXISTS lyrics_fetch_log (
    media_id     uuid        PRIMARY KEY REFERENCES media_files (media_id) ON DELETE CASCADE,
    status       text        NOT NULL CHECK (status IN ('ok', 'not_found', 'error')),
    source       text,                      -- 'genius' | 'uslt_tag' | NULL
    content_hash text,                      -- sha256 hex of full lyrics text; used for dedup
    attempted_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS lyrics_fetch_log_status
    ON lyrics_fetch_log (status);


-- --------------------------------------------------------------------------- --
-- lyrics: chunked lyrics text + BGE-M3 embeddings                              --
-- One row per chunk; a single song typically has 1-3 rows.                     --
-- chunk_id=0 is always the first chunk (used for dedup via lyrics_fetch_log).  --
-- --------------------------------------------------------------------------- --

CREATE TABLE IF NOT EXISTS lyrics (
    lyric_id    bigserial    PRIMARY KEY,
    media_id    uuid         NOT NULL REFERENCES media_files (media_id) ON DELETE CASCADE,
    chunk_id    int          NOT NULL DEFAULT 0,
    chunk_text  text         NOT NULL,
    embedding   vector(1024) NOT NULL,      -- BGE-M3 bge-m3 FP16 1024-dim L2-normalized
    created_at  timestamptz  NOT NULL DEFAULT now(),
    UNIQUE (media_id, chunk_id)
);

-- HNSW index for fast approximate cosine similarity search.
-- Matches the pattern used in rag_chunks, wiki_chunks, and image_embeddings.
CREATE INDEX IF NOT EXISTS lyrics_hnsw
    ON lyrics
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS lyrics_media_id
    ON lyrics (media_id);


-- --------------------------------------------------------------------------- --
-- Convenience view: join lyrics chunks with file metadata                       --
-- --------------------------------------------------------------------------- --

CREATE OR REPLACE VIEW lyrics_view AS
SELECT
    l.lyric_id,
    mf.media_id,
    mf.file_path,
    mf.metadata,
    lfl.source       AS lyrics_source,
    l.chunk_id,
    l.chunk_text,
    l.embedding,
    l.created_at
FROM lyrics l
JOIN media_files       mf  ON mf.media_id  = l.media_id
JOIN lyrics_fetch_log  lfl ON lfl.media_id = l.media_id;


-- --------------------------------------------------------------------------- --
-- Grants                                                                        --
-- Run these as a superuser if the tables were created by a different role.      --
-- --------------------------------------------------------------------------- --

GRANT ALL ON TABLE lyrics_fetch_log TO sage;
GRANT ALL ON TABLE lyrics           TO sage;
GRANT USAGE, SELECT ON SEQUENCE lyrics_lyric_id_seq TO sage;
