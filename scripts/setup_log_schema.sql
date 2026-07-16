-- scripts/setup_log_schema.sql
--
-- Run this ONCE as a PostgreSQL superuser (postgres) to create the dedicated
-- `log` schema and grant the sage user full ownership. Apply this BEFORE
-- log/db/log_schema.sql (which creates the tables) and before rolling out
-- the PostgresLogHandler code in any project's sk_logging.py — the handler
-- degrades silently to file-only logging when the schema/tables are missing,
-- so getting this sequencing right matters more than the safety net.
--
-- How to run (in pgAdmin query tool, or psql as postgres):
--
--   Option A — pgAdmin:
--     1. Open pgAdmin -> connect to the sage_kaizen database
--     2. Tools -> Query Tool
--     3. Paste this file and Execute (F5)
--
--   Option B — psql on the command line (run as postgres Windows user):
--     "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -d sage_kaizen -f scripts/setup_log_schema.sql
--
-- This script is idempotent — safe to re-run.
--
-- Note: unlike scripts/setup_langgraph_schema.sql, this deliberately does NOT
-- add `log` to the `sage` role's search_path. Application code always
-- schema-qualifies log inserts/selects (log.<table>), so there is no need to
-- risk unqualified-name ambiguity as more dedicated schemas accumulate on
-- the shared `sage` role over time.
-- ─────────────────────────────────────────────────────────────────────────────

-- 1. Create the dedicated schema for structured application log tables
CREATE SCHEMA IF NOT EXISTS log AUTHORIZATION sage;

-- 2. Grant full privileges on the schema to sage
--    (CREATE + USAGE — sage needs to create tables and read/write them)
GRANT ALL PRIVILEGES ON SCHEMA log TO sage;

-- Verify
SELECT
    nspname           AS schema_name,
    nspowner::regrole AS owner,
    nspacl            AS acl
FROM pg_namespace
WHERE nspname = 'log';
