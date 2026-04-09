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

-- 3. Set the default schema search path for sage so LangGraph's
--    un-qualified table references resolve to langgraph, not public.
ALTER ROLE sage SET search_path TO langgraph, public;

-- Verify
SELECT
    nspname        AS schema_name,
    nspowner::regrole AS owner,
    nspacl         AS acl
FROM pg_namespace
WHERE nspname = 'langgraph';
