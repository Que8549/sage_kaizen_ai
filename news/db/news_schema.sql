-- =============================================================================
-- news/db/news_schema.sql
--
-- Sage Kaizen Daily News Runtime — production schema.
--
-- Applies cleanly to a database that already has:
--   - pgvector extension (>= 0.8.0)
--   - media_files table (from rag_v1/db/media_schema.sql)
--
-- Run once as the schema owner:
--   psql -U sage -d sage_kaizen -f news/db/news_schema.sql
--
-- All objects live in the public schema with a news_ prefix, consistent
-- with the existing Sage Kaizen convention.
--
-- Table creation order respects FK dependencies:
--   news_topics
--   news_profiles
--   news_topic_queries
--   news_profile_topics
--   news_runs
--   daily_news                  (article store; replaces broken draft)
--   news_story_clusters
--   news_article_summaries
--   news_cluster_summaries
--   news_briefs
--   news_article_images
--   news_image_embeddings
-- =============================================================================


-- ---------------------------------------------------------------------------
-- news_topics
-- Defines the subjects Sage Kaizen monitors.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS news_topics (
    topic_id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_slug          text        NOT NULL,
    display_name        text        NOT NULL,
    description         text,
    is_enabled          boolean     NOT NULL DEFAULT true,
    priority_weight     float       NOT NULL DEFAULT 1.0,
    default_category    text        NOT NULL DEFAULT 'news',
    default_time_range  text        NOT NULL DEFAULT 'day'
                            CHECK (default_time_range IN ('day', 'week', 'month', 'year', '')),
    collection_strategy text        NOT NULL DEFAULT 'standard'
                            CHECK (collection_strategy IN ('standard', 'deep', 'minimal')),
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now(),
    metadata            jsonb       NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS news_topics_slug_uq
    ON news_topics (topic_slug);

CREATE INDEX IF NOT EXISTS news_topics_enabled
    ON news_topics (is_enabled);

COMMENT ON TABLE news_topics IS
    'Topics Sage Kaizen tracks. Seed with news/db/news_seed_data.sql.';


-- ---------------------------------------------------------------------------
-- news_profiles
-- End-user briefing scopes (e.g. general_brief, ai_brief).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS news_profiles (
    profile_id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_name            text        NOT NULL,
    description             text,
    is_enabled              boolean     NOT NULL DEFAULT true,
    summary_window_default  int         NOT NULL DEFAULT 1     -- days
                                CHECK (summary_window_default >= 1),
    top_n_default           int         NOT NULL DEFAULT 10
                                CHECK (top_n_default >= 1),
    include_market_data     boolean     NOT NULL DEFAULT false,
    created_at              timestamptz NOT NULL DEFAULT now(),
    updated_at              timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS news_profiles_name_uq
    ON news_profiles (profile_name);

COMMENT ON TABLE news_profiles IS
    'Briefing scope definitions. A profile maps to one or more topics via news_profile_topics.';


-- ---------------------------------------------------------------------------
-- news_topic_queries
-- One or more search query templates per topic.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS news_topic_queries (
    topic_query_id      uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_id            uuid        NOT NULL REFERENCES news_topics (topic_id) ON DELETE CASCADE,
    query_text          text        NOT NULL,
    searxng_categories  text[]      NOT NULL DEFAULT '{news}',
    preferred_engines   text[]      NOT NULL DEFAULT '{}',
    language_code       text        NOT NULL DEFAULT 'en',
    region_code         text        NOT NULL DEFAULT '',
    time_range          text        NOT NULL DEFAULT 'day'
                            CHECK (time_range IN ('day', 'week', 'month', 'year', '')),
    is_enabled          boolean     NOT NULL DEFAULT true,
    rank_weight         float       NOT NULL DEFAULT 1.0,
    max_results         int         NOT NULL DEFAULT 20
                            CHECK (max_results BETWEEN 1 AND 100),
    notes               text,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS news_topic_queries_topic_id
    ON news_topic_queries (topic_id);

CREATE INDEX IF NOT EXISTS news_topic_queries_enabled
    ON news_topic_queries (topic_id, is_enabled);

COMMENT ON TABLE news_topic_queries IS
    'Search query templates per topic. Multiple queries per topic are allowed and run in parallel.';


-- ---------------------------------------------------------------------------
-- news_profile_topics
-- Maps topics into profiles with per-profile weight overrides.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS news_profile_topics (
    profile_id      uuid    NOT NULL REFERENCES news_profiles (profile_id) ON DELETE CASCADE,
    topic_id        uuid    NOT NULL REFERENCES news_topics   (topic_id)   ON DELETE CASCADE,
    weight_override float,                    -- NULL = use topic.priority_weight
    is_required     boolean NOT NULL DEFAULT false,
    sort_order      int     NOT NULL DEFAULT 0,
    PRIMARY KEY (profile_id, topic_id)
);

CREATE INDEX IF NOT EXISTS news_profile_topics_topic_id
    ON news_profile_topics (topic_id);

COMMENT ON TABLE news_profile_topics IS
    'Many-to-many mapping of topics to profiles with optional weight overrides.';


-- ---------------------------------------------------------------------------
-- news_runs
-- Tracks every scheduled execution for auditing, retries, and concurrency
-- control. Finalizer jobs use this table as a lock before starting.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS news_runs (
    run_id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_type        text        NOT NULL
                        CHECK (run_type IN (
                            'collection',
                            'enrichment',
                            'image_processing',
                            'article_summarization',
                            'cluster_summarization',
                            'brief_finalization',
                            'reconciliation'
                        )),
    profile_id      uuid        REFERENCES news_profiles (profile_id) ON DELETE SET NULL,
    topic_id        uuid        REFERENCES news_topics   (topic_id)   ON DELETE SET NULL,
    scheduled_for   timestamptz,
    started_at      timestamptz,
    finished_at     timestamptz,
    status          text        NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    retry_count     int         NOT NULL DEFAULT 0,
    attempt_group_id uuid,                   -- groups retries of the same logical job
    worker_id       text,                    -- scheduler job id or thread name
    metrics_json    jsonb       NOT NULL DEFAULT '{}',
    error_text      text,
    metadata        jsonb       NOT NULL DEFAULT '{}',
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS news_runs_type_status
    ON news_runs (run_type, status);

CREATE INDEX IF NOT EXISTS news_runs_profile_date
    ON news_runs (profile_id, started_at DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS news_runs_topic_date
    ON news_runs (topic_id, started_at DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS news_runs_created_at
    ON news_runs (created_at DESC);

COMMENT ON TABLE news_runs IS
    'Audit log and concurrency lock for all scheduled news jobs. '
    'Before starting a brief_finalization job, query this table for any '
    'running row with the same profile_id + brief_date to prevent overlap.';


-- ---------------------------------------------------------------------------
-- daily_news
-- Normalized article store. Replaces the broken rag_v1/db/public.daily_news_template.sql
-- draft. One row per canonical article URL.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS daily_news (
    article_id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    first_seen_run_id   uuid        REFERENCES news_runs   (run_id)   ON DELETE SET NULL,
    topic_id            uuid        REFERENCES news_topics (topic_id) ON DELETE SET NULL,

    -- Core article fields
    headline            text,
    snippet             text,
    article_content     text,                -- full text; NULL until enrichment runs

    -- Source / URL
    news_source         text        NOT NULL DEFAULT '',
    news_source_url     text        NOT NULL DEFAULT '',
    canonical_url       text        NOT NULL,
    url_hash            bytea       NOT NULL, -- SHA-256 of canonical_url; primary dedup key
    published_at        timestamptz,
    first_seen_at       timestamptz NOT NULL DEFAULT now(),
    last_seen_at        timestamptz NOT NULL DEFAULT now(),

    -- Language / search metadata
    language_code       text        NOT NULL DEFAULT 'en',
    search_query        text,
    search_category     text,
    rank_score          float,

    -- Deduplication
    dedupe_fingerprint  text,                -- simhash of headline + snippet; near-dup detection

    -- Clustering (set by article_clusterer.py)
    cluster_id          uuid,                -- FK added below after news_story_clusters is created

    -- Embedding for clustering (BGE-M3, 1024-dim)
    article_embedding   vector(1024),        -- populated during enrichment

    -- Processing state machine
    fetch_status        text        NOT NULL DEFAULT 'pending'
                            CHECK (fetch_status IN (
                                'pending', 'fetching', 'fetched', 'failed_fetch', 'skipped'
                            )),
    summary_status      text        NOT NULL DEFAULT 'pending'
                            CHECK (summary_status IN (
                                'pending', 'summarizing', 'summarized', 'failed_summary', 'skipped'
                            )),
    image_status        text        NOT NULL DEFAULT 'pending'
                            CHECK (image_status IN (
                                'pending', 'processing', 'processed', 'failed_image', 'no_images'
                            )),

    -- Raw SearXNG response fragment; useful for debugging / reprocessing
    metadata            jsonb       NOT NULL DEFAULT '{}',
    ingested_at         timestamptz NOT NULL DEFAULT now()
);

-- Primary dedup index — upsert uses ON CONFLICT (url_hash)
CREATE UNIQUE INDEX IF NOT EXISTS daily_news_url_hash_uq
    ON daily_news (url_hash);

-- Common query patterns
CREATE INDEX IF NOT EXISTS daily_news_topic_id
    ON daily_news (topic_id);

CREATE INDEX IF NOT EXISTS daily_news_cluster_id
    ON daily_news (cluster_id);

CREATE INDEX IF NOT EXISTS daily_news_published_at
    ON daily_news (published_at DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS daily_news_first_seen_at
    ON daily_news (first_seen_at DESC);

CREATE INDEX IF NOT EXISTS daily_news_fetch_status
    ON daily_news (fetch_status)
    WHERE fetch_status IN ('pending', 'failed_fetch');

CREATE INDEX IF NOT EXISTS daily_news_summary_status
    ON daily_news (summary_status)
    WHERE summary_status IN ('pending', 'failed_summary');

CREATE INDEX IF NOT EXISTS daily_news_image_status
    ON daily_news (image_status)
    WHERE image_status IN ('pending', 'failed_image');

-- HNSW index for article-level similarity and clustering
-- Built after article_embedding is populated by the enrichment pipeline.
CREATE INDEX IF NOT EXISTS daily_news_article_embed_hnsw
    ON daily_news
    USING hnsw (article_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

COMMENT ON TABLE daily_news IS
    'Normalized article store. One row per canonical URL. '
    'url_hash (SHA-256) is the idempotency key. '
    'article_embedding (BGE-M3) powers DBSCAN clustering.';


-- ---------------------------------------------------------------------------
-- news_story_clusters
-- Groups related daily_news rows into one event or story.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS news_story_clusters (
    cluster_id                  uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_id                    uuid        REFERENCES news_topics   (topic_id)   ON DELETE SET NULL,
    profile_id                  uuid        REFERENCES news_profiles (profile_id) ON DELETE SET NULL,
    cluster_title               text,
    representative_article_id   uuid        REFERENCES daily_news    (article_id) ON DELETE SET NULL,
    importance_score            float       NOT NULL DEFAULT 0.0,
    story_start_at              timestamptz,
    story_end_at                timestamptz,
    article_count               int         NOT NULL DEFAULT 0,
    created_at                  timestamptz NOT NULL DEFAULT now(),
    updated_at                  timestamptz NOT NULL DEFAULT now(),
    metadata                    jsonb       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS news_story_clusters_topic_id
    ON news_story_clusters (topic_id);

CREATE INDEX IF NOT EXISTS news_story_clusters_profile_id
    ON news_story_clusters (profile_id);

CREATE INDEX IF NOT EXISTS news_story_clusters_story_start
    ON news_story_clusters (story_start_at DESC NULLS LAST);

COMMENT ON TABLE news_story_clusters IS
    'Story clusters produced by BGE-M3 + DBSCAN. '
    'daily_news.cluster_id references this table.';


-- Now that news_story_clusters exists, add the FK from daily_news.cluster_id.
-- Uses a deferred ALTER to avoid ordering issues; safe to re-run (IF NOT EXISTS
-- is not available for constraints, so we use a DO block).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'daily_news_cluster_id_fkey'
          AND table_name = 'daily_news'
    ) THEN
        ALTER TABLE daily_news
            ADD CONSTRAINT daily_news_cluster_id_fkey
            FOREIGN KEY (cluster_id)
            REFERENCES news_story_clusters (cluster_id)
            ON DELETE SET NULL;
    END IF;
END;
$$;


-- ---------------------------------------------------------------------------
-- news_article_summaries
-- Article-level summaries with model + prompt provenance and versioning.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS news_article_summaries (
    article_summary_id  uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id          uuid        NOT NULL REFERENCES daily_news (article_id) ON DELETE CASCADE,
    run_id              uuid        REFERENCES news_runs (run_id) ON DELETE SET NULL,
    summary_kind        text        NOT NULL
                            CHECK (summary_kind IN (
                                'article_short', 'article_medium', 'article_long',
                                'key_points', 'entities'
                            )),
    summary_short       text,
    summary_medium      text,
    summary_long        text,
    key_points_json     jsonb       NOT NULL DEFAULT '[]',
    entities_json       jsonb       NOT NULL DEFAULT '[]',
    sentiment_label     text        CHECK (sentiment_label IN ('positive', 'negative', 'neutral', 'mixed')),
    model_name          text,
    model_version       text,
    prompt_version      text,
    generated_at        timestamptz NOT NULL DEFAULT now(),
    is_active           boolean     NOT NULL DEFAULT true,
    metadata            jsonb       NOT NULL DEFAULT '{}'
);

-- Only one active summary of each kind per article.
CREATE UNIQUE INDEX IF NOT EXISTS news_article_summaries_active_kind_uq
    ON news_article_summaries (article_id, summary_kind)
    WHERE is_active = true;

CREATE INDEX IF NOT EXISTS news_article_summaries_article_id
    ON news_article_summaries (article_id);

CREATE INDEX IF NOT EXISTS news_article_summaries_is_active
    ON news_article_summaries (article_id, is_active);

COMMENT ON TABLE news_article_summaries IS
    'Per-article summaries. Re-summarization sets old row is_active=false and inserts new row. '
    'The partial unique index enforces at most one active summary per (article, kind).';


-- ---------------------------------------------------------------------------
-- news_cluster_summaries
-- Cluster-level summaries with source-diversity and confidence metadata.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS news_cluster_summaries (
    cluster_summary_id      uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id              uuid        NOT NULL REFERENCES news_story_clusters (cluster_id) ON DELETE CASCADE,
    run_id                  uuid        REFERENCES news_runs (run_id) ON DELETE SET NULL,
    summary_kind            text        NOT NULL
                                CHECK (summary_kind IN (
                                    'article_short', 'article_medium', 'article_long',
                                    'key_points', 'entities'
                                )),
    summary_short           text,
    summary_medium          text,
    summary_long            text,
    top_facts_json          jsonb       NOT NULL DEFAULT '[]',
    source_diversity_json   jsonb       NOT NULL DEFAULT '{}',
    confidence_score        float       CHECK (confidence_score BETWEEN 0.0 AND 1.0),
    model_name              text,
    model_version           text,
    prompt_version          text,
    generated_at            timestamptz NOT NULL DEFAULT now(),
    is_active               boolean     NOT NULL DEFAULT true,
    metadata                jsonb       NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS news_cluster_summaries_active_kind_uq
    ON news_cluster_summaries (cluster_id, summary_kind)
    WHERE is_active = true;

CREATE INDEX IF NOT EXISTS news_cluster_summaries_cluster_id
    ON news_cluster_summaries (cluster_id);

COMMENT ON TABLE news_cluster_summaries IS
    'Per-cluster summaries. Same versioning pattern as news_article_summaries.';


-- ---------------------------------------------------------------------------
-- news_briefs
-- Final DB-native summaries for user-facing retrieval.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS news_briefs (
    brief_id                uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              uuid        REFERENCES news_profiles (profile_id) ON DELETE SET NULL,
    run_id                  uuid        REFERENCES news_runs     (run_id)     ON DELETE SET NULL,
    brief_date              date        NOT NULL,
    window_start_at         timestamptz NOT NULL,
    window_end_at           timestamptz NOT NULL,
    brief_kind              text        NOT NULL
                                CHECK (brief_kind IN (
                                    'daily', 'rolling_7_day', 'topic', 'market_context'
                                )),
    headline_summary        text,
    summary_short           text,
    summary_long            text,
    top_story_cluster_ids   uuid[]      NOT NULL DEFAULT '{}',
    model_name              text,
    model_version           text,
    generated_at            timestamptz NOT NULL DEFAULT now(),
    freshness_at            timestamptz NOT NULL DEFAULT now(),
    is_final                boolean     NOT NULL DEFAULT false,
    metadata                jsonb       NOT NULL DEFAULT '{}'
);

-- One finalised brief per (profile, date, kind).
-- is_final=false rows are drafts / in-progress; multiple can coexist.
CREATE UNIQUE INDEX IF NOT EXISTS news_briefs_final_uq
    ON news_briefs (profile_id, brief_date, brief_kind)
    WHERE is_final = true;

CREATE INDEX IF NOT EXISTS news_briefs_profile_date
    ON news_briefs (profile_id, brief_date DESC);

CREATE INDEX IF NOT EXISTS news_briefs_freshness
    ON news_briefs (freshness_at DESC);

COMMENT ON TABLE news_briefs IS
    'Final user-facing briefs. The partial unique index prevents duplicate finalised '
    'briefs for the same profile/date/kind. Finalizer jobs check news_runs for '
    'in-progress runs before inserting.';


-- ---------------------------------------------------------------------------
-- news_article_images
-- Maps articles to one or more images (one article → many images).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS news_article_images (
    article_image_id    uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id          uuid        NOT NULL REFERENCES daily_news   (article_id) ON DELETE CASCADE,
    media_id            uuid        NOT NULL REFERENCES media_files  (media_id)   ON DELETE CASCADE,
    source_image_url    text        NOT NULL,
    image_role          text        NOT NULL DEFAULT 'hero'
                            CHECK (image_role IN ('hero', 'thumbnail', 'inline', 'gallery')),
    caption_text        text,
    surrounding_text    text,
    position_index      int         NOT NULL DEFAULT 0,
    discovered_at       timestamptz NOT NULL DEFAULT now(),
    metadata            jsonb       NOT NULL DEFAULT '{}'
);

-- Prevent the same media_id being linked to the same article twice.
CREATE UNIQUE INDEX IF NOT EXISTS news_article_images_article_media_uq
    ON news_article_images (article_id, media_id);

CREATE INDEX IF NOT EXISTS news_article_images_article_id
    ON news_article_images (article_id);

CREATE INDEX IF NOT EXISTS news_article_images_media_id
    ON news_article_images (media_id);

COMMENT ON TABLE news_article_images IS
    'Article-to-image join table. media_id links to the existing media_files table '
    '(modality=''image'', file_path under H:\\article_images\\).';


-- ---------------------------------------------------------------------------
-- news_image_embeddings
-- Vector embeddings and retrieval metadata for news images.
-- Uses jina-clip-v2 (1024-dim, same as image_embeddings) so the same
-- embed service (port 8031) can be reused without modification.
--
-- Denormalized filter columns (topic_id, published_at, source_name,
-- canonical_source_domain) live here to avoid expensive joins during
-- filtered ANN queries.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS news_image_embeddings (
    news_image_embed_id     uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    media_id                uuid        NOT NULL REFERENCES media_files       (media_id)   ON DELETE CASCADE,
    article_id              uuid        REFERENCES daily_news                 (article_id) ON DELETE SET NULL,
    cluster_id              uuid        REFERENCES news_story_clusters        (cluster_id) ON DELETE SET NULL,
    topic_id                uuid        REFERENCES news_topics                (topic_id)   ON DELETE SET NULL,
    profile_id              uuid        REFERENCES news_profiles              (profile_id) ON DELETE SET NULL,

    -- Denormalized source metadata for fast filtered ANN (avoids joins)
    source_name             text,
    source_url              text,
    canonical_source_domain text,
    image_role              text        CHECK (image_role IN ('hero', 'thumbnail', 'inline', 'gallery')),
    caption_text            text,
    surrounding_text        text,
    published_at            timestamptz,
    first_seen_at           timestamptz NOT NULL DEFAULT now(),

    -- Embedding provenance
    embedding_model         text        NOT NULL DEFAULT 'jina-clip-v2',
    embedding_version       text        NOT NULL DEFAULT '1',
    embedding_metric        text        NOT NULL DEFAULT 'cosine',

    -- The vector (jina-clip-v2, 1024-dim, L2-normalised)
    embedding               vector(1024) NOT NULL,

    -- Dedup / re-embedding support
    content_hash            bytea,              -- SHA-256 of image bytes; detect unchanged images
    is_active               boolean     NOT NULL DEFAULT true,

    created_at              timestamptz NOT NULL DEFAULT now(),
    updated_at              timestamptz NOT NULL DEFAULT now(),
    metadata                jsonb       NOT NULL DEFAULT '{}'
);

-- Primary ANN index.
-- ef_construction=128 (vs 64 in media_schema) for better recall on high-turnover news images.
CREATE INDEX IF NOT EXISTS news_image_embed_hnsw
    ON news_image_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- B-tree indexes on every common filter column.
-- These work with the HNSW iterative scan (SET hnsw.iterative_scan = relaxed_order).
CREATE INDEX IF NOT EXISTS news_image_embed_topic_id
    ON news_image_embeddings (topic_id);

CREATE INDEX IF NOT EXISTS news_image_embed_published_at
    ON news_image_embeddings (published_at DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS news_image_embed_first_seen_at
    ON news_image_embeddings (first_seen_at DESC);

CREATE INDEX IF NOT EXISTS news_image_embed_source_name
    ON news_image_embeddings (source_name);

CREATE INDEX IF NOT EXISTS news_image_embed_source_domain
    ON news_image_embeddings (canonical_source_domain);

CREATE INDEX IF NOT EXISTS news_image_embed_article_id
    ON news_image_embeddings (article_id);

CREATE INDEX IF NOT EXISTS news_image_embed_cluster_id
    ON news_image_embeddings (cluster_id);

CREATE INDEX IF NOT EXISTS news_image_embed_is_active
    ON news_image_embeddings (is_active)
    WHERE is_active = true;

COMMENT ON TABLE news_image_embeddings IS
    'jina-clip-v2 embeddings for news images. Denormalized filter columns '
    '(topic_id, published_at, source_name, canonical_source_domain) support '
    'filtered ANN without joins. Use SET hnsw.iterative_scan = relaxed_order '
    'in the query session for narrow filter sets (pgvector >= 0.8.0).';


-- =============================================================================
-- Grants — mirror the pattern used in rag_v1/db/media_schema.sql
-- =============================================================================

GRANT ALL ON TABLE news_topics              TO sage;
GRANT ALL ON TABLE news_profiles            TO sage;
GRANT ALL ON TABLE news_topic_queries       TO sage;
GRANT ALL ON TABLE news_profile_topics      TO sage;
GRANT ALL ON TABLE news_runs                TO sage;
GRANT ALL ON TABLE daily_news               TO sage;
GRANT ALL ON TABLE news_story_clusters      TO sage;
GRANT ALL ON TABLE news_article_summaries   TO sage;
GRANT ALL ON TABLE news_cluster_summaries   TO sage;
GRANT ALL ON TABLE news_briefs              TO sage;
GRANT ALL ON TABLE news_article_images      TO sage;
GRANT ALL ON TABLE news_image_embeddings    TO sage;

-- Sequences for BIGSERIAL columns (none here — all PKs are UUID via gen_random_uuid()).
-- No explicit sequence grants required.
