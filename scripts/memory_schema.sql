-- =============================================================================
-- memory_schema.sql
-- Sage Kaizen Memory Service — PostgreSQL schema
--
-- Purpose:
--   Creates all tables, indexes, and extensions for the Sage Kaizen Memory
--   Service.  All objects live in the 'memory' schema to avoid conflicts with
--   the 'public' schema used by the RAG tables and the 'langgraph' schema used
--   by the Review Service checkpointer.
--
-- Required extensions:
--   vector    — pgvector >= 0.8.2 (CVE-2026-3172 security fix)
--   pg_trgm   — trigram similarity for exact-ish technical term matching
--
-- Safe to re-run:
--   All DDL uses IF NOT EXISTS / DO $$ ... IF NOT EXISTS $$ patterns.
--   Re-running this file on an existing database is safe and idempotent.
--
-- Apply:
--   psql -U <user> -d <db> -f scripts/memory_schema.sql
--
-- pgvector iterative scan (0.8.0+):
--   All filtered vector queries in retriever.py must SET LOCAL:
--     SET LOCAL hnsw.iterative_scan = relaxed_order;
--     SET LOCAL hnsw.max_scan_tuples = 20000;
--   This prevents silent under-retrieval when user_id / project_id filters
--   are highly selective (one user's memories out of many).
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS memory;

-- ---------------------------------------------------------------------------
-- 1. memory.profiles
--    Stable user and project profile facts (always-on; injected every turn).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory.profiles (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          TEXT        NOT NULL,
    project_id       TEXT,
    workspace_id     TEXT,
    scope            TEXT        NOT NULL CHECK (scope IN ('user','project','workspace','global_system')),
    profile_type     TEXT        NOT NULL,   -- e.g. 'tone', 'format', 'environment', 'prohibition'
    key              TEXT        NOT NULL,
    value_text       TEXT        NOT NULL,
    value_json       JSONB,
    confidence       REAL        NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
    source_type      TEXT        NOT NULL,   -- 'explicit_user', 'inferred', 'system_default'
    is_pinned        BOOLEAN     NOT NULL DEFAULT TRUE,
    is_locked        BOOLEAN     NOT NULL DEFAULT FALSE,
    is_active        BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_confirmed_at TIMESTAMPTZ,
    expires_at       TIMESTAMPTZ,
    metadata         JSONB       NOT NULL DEFAULT '{}'::jsonb
);

-- Expression-based unique index replaces the inline UNIQUE constraint.
-- PostgreSQL does not allow COALESCE() inside table-level UNIQUE (...),
-- but does allow it in CREATE UNIQUE INDEX.
-- The ON CONFLICT clause in repository.py references this same expression.
CREATE UNIQUE INDEX IF NOT EXISTS idx_profiles_unique_key
    ON memory.profiles (
        user_id,
        COALESCE(project_id, ''),
        COALESCE(workspace_id, ''),
        scope,
        profile_type,
        key
    );

CREATE INDEX IF NOT EXISTS idx_profiles_user_project
    ON memory.profiles (user_id, project_id, scope, is_active);

-- ---------------------------------------------------------------------------
-- 2. memory.episodes
--    Prior interaction events and lessons (retrieved on demand).
--    Embeddings use BGE-M3 1024-dimensional L2-normalized vectors.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory.episodes (
    id                      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                 TEXT        NOT NULL,
    project_id              TEXT,
    workspace_id            TEXT,
    session_id              TEXT,
    scope                   TEXT        NOT NULL CHECK (scope IN ('user','project','workspace','session')),
    event_type              TEXT        NOT NULL,   -- 'correction', 'preference', 'decision', 'approval', 'bug_report'
    intent_label            TEXT,
    summary_text            TEXT        NOT NULL,
    raw_excerpt             TEXT,
    tags                    JSONB       NOT NULL DEFAULT '[]'::jsonb,
    importance              REAL        NOT NULL DEFAULT 0.5 CHECK (importance BETWEEN 0.0 AND 1.0),
    confidence              REAL        NOT NULL DEFAULT 0.6 CHECK (confidence BETWEEN 0.0 AND 1.0),
    sentiment               REAL,                   -- -1.0 (negative) to 1.0 (positive)
    was_user_correction     BOOLEAN     NOT NULL DEFAULT FALSE,
    was_explicit_preference BOOLEAN     NOT NULL DEFAULT FALSE,
    contradiction_group     TEXT,
    -- BGE-M3 FP16, 1024-dim, L2-normalized → cosine distance via <=>
    embedding               vector(1024),
    -- Generated tsvector for lexical search (FTS)
    search_tsv              tsvector    GENERATED ALWAYS AS (
        to_tsvector('english',
            coalesce(summary_text, '') || ' ' ||
            coalesce(raw_excerpt, '') || ' ' ||
            coalesce(intent_label, '')
        )
    ) STORED,
    metadata                JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at        TIMESTAMPTZ,
    last_retrieved_at       TIMESTAMPTZ,
    expires_at              TIMESTAMPTZ
);

-- HNSW vector index (pgvector 0.8.x)
-- m=16: connectivity suitable for 1024-dim memory embeddings
-- ef_construction=128: better recall at build time; slower build
-- Enable iterative scan at query time: SET LOCAL hnsw.iterative_scan = relaxed_order;
CREATE INDEX IF NOT EXISTS idx_episodes_embedding_hnsw
    ON memory.episodes
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- GIN index on generated tsvector for lexical search
CREATE INDEX IF NOT EXISTS idx_episodes_search_tsv
    ON memory.episodes
    USING GIN (search_tsv);

-- Composite metadata filter index (always applied before vector search).
-- NOW() is STABLE not IMMUTABLE, so it cannot appear in a partial index predicate.
-- The expires_at filter is applied at query time in repository.py instead.
CREATE INDEX IF NOT EXISTS idx_episodes_user_project_active
    ON memory.episodes (user_id, project_id, workspace_id, scope, expires_at);

CREATE INDEX IF NOT EXISTS idx_episodes_created_at
    ON memory.episodes (user_id, created_at DESC);

-- Trigram index for fuzzy technical term matching (model names, flags, ports)
CREATE INDEX IF NOT EXISTS idx_episodes_summary_trgm
    ON memory.episodes
    USING GIN (summary_text gin_trgm_ops);

-- ---------------------------------------------------------------------------
-- 3. memory.rules
--    Procedural rules and operational norms (retrieved + selectively pinned).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory.rules (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           TEXT,
    project_id        TEXT,
    workspace_id      TEXT,
    scope             TEXT        NOT NULL CHECK (scope IN ('user','project','workspace','global_system')),
    rule_kind         TEXT        NOT NULL,   -- 'coding_norm', 'tool_preference', 'documentation_policy', 'safety'
    rule_text         TEXT        NOT NULL,
    rationale         TEXT,
    confidence        REAL        NOT NULL DEFAULT 0.7 CHECK (confidence BETWEEN 0.0 AND 1.0),
    promotion_count   INT         NOT NULL DEFAULT 0,
    source_type       TEXT        NOT NULL,   -- 'explicit_user', 'promoted_from_episode', 'system'
    source_memory_id  UUID,                   -- FK to memory.episodes (soft reference)
    is_locked         BOOLEAN     NOT NULL DEFAULT FALSE,
    is_active         BOOLEAN     NOT NULL DEFAULT TRUE,
    review_status     TEXT        NOT NULL DEFAULT 'proposed'
                                  CHECK (review_status IN ('proposed','approved','demoted','archived')),
    metadata          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at        TIMESTAMPTZ,
    -- FTS on rule_text
    search_tsv        tsvector    GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(rule_text, '') || ' ' || coalesce(rationale, ''))
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_rules_user_project_active
    ON memory.rules (user_id, project_id, scope, is_active, review_status);

CREATE INDEX IF NOT EXISTS idx_rules_search_tsv
    ON memory.rules USING GIN (search_tsv);

-- ---------------------------------------------------------------------------
-- 4. memory.reflections
--    Session-level and batch consolidation outputs (not injected directly).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory.reflections (
    id                          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                     TEXT        NOT NULL,
    project_id                  TEXT,
    workspace_id                TEXT,
    session_id                  TEXT,
    reflection_type             TEXT        NOT NULL,   -- 'session', 'nightly', 'manual'
    summary_text                TEXT        NOT NULL,
    extracted_profile_candidates JSONB      NOT NULL DEFAULT '[]'::jsonb,
    extracted_rule_candidates   JSONB       NOT NULL DEFAULT '[]'::jsonb,
    contradictions              JSONB       NOT NULL DEFAULT '[]'::jsonb,
    pruning_suggestions         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    confidence                  REAL        NOT NULL DEFAULT 0.7,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata                    JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_reflections_user_session
    ON memory.reflections (user_id, session_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- 5. memory.links
--    Optional relation table linking memories to each other.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory.links (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    from_memory_id  UUID        NOT NULL,
    to_memory_id    UUID        NOT NULL,
    relation_type   TEXT        NOT NULL,   -- 'supersedes', 'contradicts', 'supports', 'derived_from'
    strength        REAL        NOT NULL DEFAULT 0.5,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (from_memory_id, to_memory_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_links_from ON memory.links (from_memory_id);
CREATE INDEX IF NOT EXISTS idx_links_to   ON memory.links (to_memory_id);

-- ---------------------------------------------------------------------------
-- 6. memory.audit_log
--    Full change history for governance and reversibility.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory.audit_log (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_table  TEXT        NOT NULL,   -- 'profiles', 'episodes', 'rules', 'reflections'
    memory_id     UUID        NOT NULL,
    action_type   TEXT        NOT NULL,   -- 'insert', 'update', 'delete', 'promote', 'demote', 'lock', 'forget'
    actor_type    TEXT        NOT NULL,   -- 'user', 'fast_brain', 'architect_brain', 'system', 'langmem'
    actor_id      TEXT,
    old_value     JSONB,
    new_value     JSONB,
    reason        TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_memory_id
    ON memory.audit_log (memory_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_log_table_action
    ON memory.audit_log (memory_table, action_type, created_at DESC);

-- (End of schema)
