-- =========================
-- Sage Kaizen: Wikipedia RAG (multimodal) schema
-- vector(1024), dedicated wiki_* tables
-- =========================

BEGIN;

-- 1) pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) Bundles (family grouping, e.g., Einstein + Einstein_(disambiguation))
CREATE TABLE IF NOT EXISTS wiki_bundles (
  bundle_id    UUID PRIMARY KEY,
  bundle_key   TEXT UNIQUE NOT NULL,   -- e.g., 'wiki:Albert_Einstein'
  family_title TEXT NOT NULL,          -- e.g., 'Albert_Einstein'
  first_letter CHAR(1) NOT NULL,       -- 'a'..'z' or '#'
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3) Pages (each markdown file)
CREATE TABLE IF NOT EXISTS wiki_pages (
  page_id       UUID PRIMARY KEY,
  bundle_id     UUID NOT NULL REFERENCES wiki_bundles(bundle_id) ON DELETE CASCADE,
  page_source_id TEXT UNIQUE NOT NULL, -- MUST equal folder_source_id(md_path): localfile:<abs_path>
  title         TEXT NOT NULL,
  page_type     TEXT NOT NULL CHECK (page_type IN ('article','disambiguation','redirect','list')),
  oldid         BIGINT NULL,
  md_path       TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_wiki_pages_bundle_id ON wiki_pages(bundle_id);

-- 4) Chunks (embedded text)
CREATE TABLE IF NOT EXISTS wiki_chunks (
  chunk_id     BIGSERIAL PRIMARY KEY,
  page_id      UUID NOT NULL REFERENCES wiki_pages(page_id) ON DELETE CASCADE,
  bundle_id    UUID NOT NULL REFERENCES wiki_bundles(bundle_id) ON DELETE CASCADE,
  title        TEXT NOT NULL,
  first_letter CHAR(1) NOT NULL,
  section_path TEXT[] NULL,
  chunk_index  INT NOT NULL,
  text         TEXT NOT NULL,
  chunk_hash   TEXT NOT NULL,
  embedding    VECTOR(1024) NOT NULL,
  CONSTRAINT uq_wiki_chunks_page_hash UNIQUE (page_id, chunk_hash)
);

CREATE INDEX IF NOT EXISTS idx_wiki_chunks_bundle_id ON wiki_chunks(bundle_id);
CREATE INDEX IF NOT EXISTS idx_wiki_chunks_page_id   ON wiki_chunks(page_id);
CREATE INDEX IF NOT EXISTS idx_wiki_chunks_first_letter ON wiki_chunks(first_letter);

-- HNSW for cosine distance
-- Tune m/ef_construction later; these are strong defaults for large corpora
CREATE INDEX IF NOT EXISTS hnsw_wiki_chunks_embedding_cos
ON wiki_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 100);

-- 5) Images (embedded pixels + embedded caption text)
CREATE TABLE IF NOT EXISTS wiki_images (
  image_id         BIGSERIAL PRIMARY KEY,
  bundle_id        UUID NOT NULL REFERENCES wiki_bundles(bundle_id) ON DELETE CASCADE,
  first_letter     CHAR(1) NOT NULL,
  relative_path    TEXT NOT NULL,      -- relative to the bundle directory or wiki root
  byte_hash        TEXT NOT NULL,      -- sha256 of bytes
  mime             TEXT NULL,
  caption_text     TEXT NOT NULL,
  image_embedding  VECTOR(1024) NOT NULL,
  caption_embedding VECTOR(1024) NOT NULL,
  is_hero          BOOLEAN NOT NULL DEFAULT false,
  hero_rank        INT NOT NULL DEFAULT 9999,
  CONSTRAINT uq_wiki_images_bundle_path UNIQUE (bundle_id, relative_path),
  CONSTRAINT uq_wiki_images_byte_hash   UNIQUE (byte_hash)
);

CREATE INDEX IF NOT EXISTS idx_wiki_images_bundle_id ON wiki_images(bundle_id);
CREATE INDEX IF NOT EXISTS idx_wiki_images_first_letter ON wiki_images(first_letter);

CREATE INDEX IF NOT EXISTS hnsw_wiki_images_image_embedding_cos
ON wiki_images
USING hnsw (image_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 100);

CREATE INDEX IF NOT EXISTS hnsw_wiki_images_caption_embedding_cos
ON wiki_images
USING hnsw (caption_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 100);

COMMIT;

