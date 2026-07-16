-- =============================================================================
-- log/db/log_schema.sql
--
-- Sage Kaizen structured application logging — production schema.
--
-- Prerequisite: run scripts/setup_log_schema.sql first (creates the `log`
-- schema and grants it to `sage`) — this file only creates tables inside it.
--
-- Run once as the schema owner:
--   psql -U sage -d sage_kaizen -f log/db/log_schema.sql
--
-- All objects live in the dedicated `log` schema, one table per source
-- .log file across sage_kaizen_ai / sage_kaizen_ai_ingest / sage_kaizen_ai_voice
-- (structured, logging-module-based files only — see CLAUDE.md for the raw
-- subprocess / llama-server logs that are deliberately out of scope).
--
-- Populated by an in-process buffered PostgresLogHandler (sk_logging.py, one
-- local copy per project) — never by application code directly. Rows are
-- appended only; there is no UPDATE/DELETE path in normal operation.
--
-- Column notes:
--   log_date   — from the LogRecord's own `created` timestamp (UTC), NOT
--                parsed from the formatted log line text. File logs remain
--                in local naive time; DB rows are UTC — this offset is
--                expected, not a bug.
--   log_type   — named to match the requesting spec; holds the log LEVEL
--                (INFO/WARNING/ERROR/...), not a taxonomy of message types.
--   run_id     — process-level correlation ID (see log.all_logs below for
--                the cross-table join this enables). Nullable: not every
--                code path is wired up yet, and some log lines predate any
--                run context (e.g. module import time).
--   exception  — populated only when the LogRecord carried exc_info; without
--                this, ERROR rows in the DB would be strictly less useful
--                than the file line they mirror.
--
-- No native partitioning yet (Postgres best-practice guidance: partition at
-- ~50-100GB or 100M+ rows — not there). Converting to a partitioned table
-- later is a mechanical migration once volume actually warrants it.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- Table creation macro (repeated per source .log file — see per-table blocks
-- below for the six sources). Column/index/comment shape is identical across
-- all six; only the table name and source_project default differ.
-- ---------------------------------------------------------------------------

-- log.sage_kaizen — main project's sage_kaizen.log (chat/RAG/review-service/memory)
CREATE TABLE IF NOT EXISTS log.sage_kaizen (
    id             bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    log_date       timestamptz NOT NULL,
    log_type       text        NOT NULL,
    log_name       text        NOT NULL,
    description    text        NOT NULL,
    exception      text,
    run_id         uuid,
    source_project text        NOT NULL DEFAULT 'sage_kaizen_ai'
                       CHECK (source_project IN ('sage_kaizen_ai', 'sage_kaizen_ai_ingest', 'sage_kaizen_ai_voice')),
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sage_kaizen_log_date ON log.sage_kaizen (log_date DESC);
CREATE INDEX IF NOT EXISTS sage_kaizen_run_id   ON log.sage_kaizen (run_id, log_date) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS sage_kaizen_errors   ON log.sage_kaizen (log_date DESC) WHERE log_type IN ('ERROR', 'WARNING', 'CRITICAL');

COMMENT ON TABLE log.sage_kaizen IS
    'Structured logs from sage_kaizen_ai''s logs/sage_kaizen.log (chat service, router, RAG runtime, review service, memory). Populated by sk_logging.py''s PostgresLogHandler.';


-- log.sage_kaizen_ingest — ingest project's sage_kaizen_ingest.log
CREATE TABLE IF NOT EXISTS log.sage_kaizen_ingest (
    id             bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    log_date       timestamptz NOT NULL,
    log_type       text        NOT NULL,
    log_name       text        NOT NULL,
    description    text        NOT NULL,
    exception      text,
    run_id         uuid,
    source_project text        NOT NULL DEFAULT 'sage_kaizen_ai_ingest'
                       CHECK (source_project IN ('sage_kaizen_ai', 'sage_kaizen_ai_ingest', 'sage_kaizen_ai_voice')),
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sage_kaizen_ingest_log_date ON log.sage_kaizen_ingest (log_date DESC);
CREATE INDEX IF NOT EXISTS sage_kaizen_ingest_run_id   ON log.sage_kaizen_ingest (run_id, log_date) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS sage_kaizen_ingest_errors   ON log.sage_kaizen_ingest (log_date DESC) WHERE log_type IN ('ERROR', 'WARNING', 'CRITICAL');

COMMENT ON TABLE log.sage_kaizen_ingest IS
    'Structured logs from sage_kaizen_ai_ingest''s logs/sage_kaizen_ingest.log (default sk_logging.py sink, e.g. mm_embed_service lifecycle). Populated by sk_logging.py''s PostgresLogHandler.';


-- log.sage_kaizen_voice — voice project's sage_kaizen_voice.log
CREATE TABLE IF NOT EXISTS log.sage_kaizen_voice (
    id             bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    log_date       timestamptz NOT NULL,
    log_type       text        NOT NULL,
    log_name       text        NOT NULL,
    description    text        NOT NULL,
    exception      text,
    run_id         uuid,
    source_project text        NOT NULL DEFAULT 'sage_kaizen_ai_voice'
                       CHECK (source_project IN ('sage_kaizen_ai', 'sage_kaizen_ai_ingest', 'sage_kaizen_ai_voice')),
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sage_kaizen_voice_log_date ON log.sage_kaizen_voice (log_date DESC);
CREATE INDEX IF NOT EXISTS sage_kaizen_voice_run_id   ON log.sage_kaizen_voice (run_id, log_date) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS sage_kaizen_voice_errors   ON log.sage_kaizen_voice (log_date DESC) WHERE log_type IN ('ERROR', 'WARNING', 'CRITICAL');

COMMENT ON TABLE log.sage_kaizen_voice IS
    'Structured logs from sage_kaizen_ai_voice''s logs/sage_kaizen_voice.log (STT/TTS pipeline, ZMQ handlers). Populated by sk_logging.py''s PostgresLogHandler.';


-- log.news_agent — news_agent.log, written by BOTH sage_kaizen_ai (query-time
-- news_resolver.py/market_client.py) and sage_kaizen_ai_ingest (the full
-- collection/enrichment/clustering/summarization pipeline). Same logical
-- concern split across a process/deployment boundary — one table,
-- disambiguated by source_project (no default here; always supplied).
CREATE TABLE IF NOT EXISTS log.news_agent (
    id             bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    log_date       timestamptz NOT NULL,
    log_type       text        NOT NULL,
    log_name       text        NOT NULL,
    description    text        NOT NULL,
    exception      text,
    run_id         uuid,
    source_project text        NOT NULL
                       CHECK (source_project IN ('sage_kaizen_ai', 'sage_kaizen_ai_ingest', 'sage_kaizen_ai_voice')),
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS news_agent_log_date ON log.news_agent (log_date DESC);
CREATE INDEX IF NOT EXISTS news_agent_run_id   ON log.news_agent (run_id, log_date) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS news_agent_errors   ON log.news_agent (log_date DESC) WHERE log_type IN ('ERROR', 'WARNING', 'CRITICAL');
CREATE INDEX IF NOT EXISTS news_agent_source   ON log.news_agent (source_project, log_date DESC);

COMMENT ON TABLE log.news_agent IS
    'Structured logs from news_agent.log, written by BOTH sage_kaizen_ai (query-time news_resolver.py/market_client.py) and sage_kaizen_ai_ingest (the full news pipeline) — disambiguate with source_project. Populated by sk_logging.py''s PostgresLogHandler.';


-- log.media_ingest — ingest project's media_ingest.log
CREATE TABLE IF NOT EXISTS log.media_ingest (
    id             bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    log_date       timestamptz NOT NULL,
    log_type       text        NOT NULL,
    log_name       text        NOT NULL,
    description    text        NOT NULL,
    exception      text,
    run_id         uuid,
    source_project text        NOT NULL DEFAULT 'sage_kaizen_ai_ingest'
                       CHECK (source_project IN ('sage_kaizen_ai', 'sage_kaizen_ai_ingest', 'sage_kaizen_ai_voice')),
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS media_ingest_log_date ON log.media_ingest (log_date DESC);
CREATE INDEX IF NOT EXISTS media_ingest_run_id   ON log.media_ingest (run_id, log_date) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS media_ingest_errors   ON log.media_ingest (log_date DESC) WHERE log_type IN ('ERROR', 'WARNING', 'CRITICAL');

COMMENT ON TABLE log.media_ingest IS
    'Structured logs from sage_kaizen_ai_ingest''s logs/media_ingest.log (media_ingest.py). Populated by sk_logging.py''s PostgresLogHandler.';


-- log.wiki_ingest — ingest project's wiki_ingest.log
CREATE TABLE IF NOT EXISTS log.wiki_ingest (
    id             bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    log_date       timestamptz NOT NULL,
    log_type       text        NOT NULL,
    log_name       text        NOT NULL,
    description    text        NOT NULL,
    exception      text,
    run_id         uuid,
    source_project text        NOT NULL DEFAULT 'sage_kaizen_ai_ingest'
                       CHECK (source_project IN ('sage_kaizen_ai', 'sage_kaizen_ai_ingest', 'sage_kaizen_ai_voice')),
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS wiki_ingest_log_date ON log.wiki_ingest (log_date DESC);
CREATE INDEX IF NOT EXISTS wiki_ingest_run_id   ON log.wiki_ingest (run_id, log_date) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS wiki_ingest_errors   ON log.wiki_ingest (log_date DESC) WHERE log_type IN ('ERROR', 'WARNING', 'CRITICAL');

COMMENT ON TABLE log.wiki_ingest IS
    'Structured logs from sage_kaizen_ai_ingest''s logs/wiki_ingest.log (wiki_ingest.py root logger, fixed 2026-07 to match the standard sk_logging format). Populated by sk_logging.py''s PostgresLogHandler.';


-- ---------------------------------------------------------------------------
-- log.all_logs — convenience view unioning all six tables with a
-- table_name discriminator, so a single run's log lines across every
-- component can be pulled with one query:
--
--   SELECT * FROM log.all_logs WHERE run_id = '...' ORDER BY log_date;
--
-- Known limits: adding/removing a source table means editing this view; a
-- wide unscoped time-range query across all six tables cannot merge-sort via
-- indexes efficiently (each branch is independently indexed, not the union).
-- Designed for the run_id-scoped case above, not general-purpose log search.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW log.all_logs AS
    SELECT 'sage_kaizen'        AS table_name, id, log_date, log_type, log_name, description, exception, run_id, source_project, created_at FROM log.sage_kaizen
    UNION ALL
    SELECT 'sage_kaizen_ingest' AS table_name, id, log_date, log_type, log_name, description, exception, run_id, source_project, created_at FROM log.sage_kaizen_ingest
    UNION ALL
    SELECT 'sage_kaizen_voice'  AS table_name, id, log_date, log_type, log_name, description, exception, run_id, source_project, created_at FROM log.sage_kaizen_voice
    UNION ALL
    SELECT 'news_agent'         AS table_name, id, log_date, log_type, log_name, description, exception, run_id, source_project, created_at FROM log.news_agent
    UNION ALL
    SELECT 'media_ingest'       AS table_name, id, log_date, log_type, log_name, description, exception, run_id, source_project, created_at FROM log.media_ingest
    UNION ALL
    SELECT 'wiki_ingest'        AS table_name, id, log_date, log_type, log_name, description, exception, run_id, source_project, created_at FROM log.wiki_ingest;

COMMENT ON VIEW log.all_logs IS
    'UNION ALL convenience view across all log.* tables for run_id-scoped cross-component queries. See table comment header in log_schema.sql for limits.';


-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------

GRANT ALL ON TABLE log.sage_kaizen        TO sage;
GRANT ALL ON TABLE log.sage_kaizen_ingest TO sage;
GRANT ALL ON TABLE log.sage_kaizen_voice  TO sage;
GRANT ALL ON TABLE log.news_agent         TO sage;
GRANT ALL ON TABLE log.media_ingest       TO sage;
GRANT ALL ON TABLE log.wiki_ingest        TO sage;
GRANT SELECT ON log.all_logs              TO sage;
