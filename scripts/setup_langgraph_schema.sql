-- scripts/setup_langgraph_schema.sql
--
-- Run this ONCE as a PostgreSQL superuser (postgres) to create the dedicated
-- langgraph schema and grant the sage user full ownership.
--
-- How to run (in pgAdmin query tool, or psql as postgres):
--
--   Option A — pgAdmin:
--     1. Open pgAdmin → connect to the sage_kaizen database
--     2. Tools → Query Tool
--     3. Paste this file and Execute (F5)
--
--   Option B — psql on the command line (run as postgres Windows user):
--     "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -d sage_kaizen -f scripts/setup_langgraph_schema.sql
--
-- This script is idempotent — safe to re-run.
-- ─────────────────────────────────────────────────────────────────────────────

-- 1. Create the dedicated schema for LangGraph checkpoint tables
--    (checkpoints, checkpoint_blobs, checkpoint_migrations)
CREATE SCHEMA IF NOT EXISTS langgraph AUTHORIZATION sage;

-- 2. Grant full privileges on the schema to sage
--    (CREATE + USAGE — sage needs to create tables and access them)
GRANT ALL PRIVILEGES ON SCHEMA langgraph TO sage;

-- 3. Search path: PUBLIC FIRST.  Corrected 2026-08-24.
--
--    This used to read `TO langgraph, public`, which made langgraph the
--    default creation target for the entire application role, not just for
--    LangGraph.  Every unqualified CREATE TABLE by `sage` — in either
--    project — landed in the checkpoint schema.  That is not hypothetical:
--    migrate_wiki_chunks_partitioned.py created its state table there and
--    now carries an explicit `SET search_path = public` to work around it.
--
--    It was never needed.  review_service/checkpointer.py appends
--    `options=-csearch_path=langgraph` to its own DSN, so LangGraph's
--    unqualified references already resolve correctly on the connection that
--    makes them, independently of any role default.
--
--    langgraph is kept on the path (second) so a manual psql session can
--    still reach those tables unqualified; it simply no longer wins.
ALTER ROLE sage SET search_path TO public, langgraph;

-- Verify
SELECT
    nspname        AS schema_name,
    nspowner::regrole AS owner,
    nspacl         AS acl
FROM pg_namespace
WHERE nspname = 'langgraph';
