-- =============================================================================
-- Migration 001 — Add updated_at to daily_news
-- =============================================================================
-- Root cause:
--   news_schema.sql defined daily_news without an updated_at column.
--   Five pipeline modules write  SET updated_at = now()  on every state
--   transition, causing a hard UndefinedColumn error on the first enrichment
--   run:
--     article_enricher.py   (_SET_FETCHING_SQL, _UPDATE_ARTICLE_SQL,
--                            _mark_failed, _mark_skipped)
--     article_clusterer.py  (cluster assignment + reset)
--     article_summarizer.py (_SET_SUMMARY_STATUS_SQL)
--     news_image_pipeline.py (_SET_IMAGE_STATUS_SQL)
--     news_scheduler.py     (reconciliation queries — reads AND writes)
--
-- Apply once as the postgres superuser (table owner):
--
--   "C:\Program Files\PostgreSQL\18\bin\psql.exe" ^
--       -U postgres -d sage_kaizen ^
--       -f "F:\Projects\sage_kaizen_ai\news\db\migrations\001_add_daily_news_updated_at.sql"
--
-- Safe to re-run: ADD COLUMN IF NOT EXISTS is idempotent.
-- =============================================================================

-- Step 1: Add the column with a server-side default so existing rows are
-- filled immediately.  PostgreSQL applies the DEFAULT to all current rows
-- in a single pass before releasing the lock.
ALTER TABLE daily_news
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();

-- Step 2: Backfill existing rows with last_seen_at rather than the migration
-- timestamp.  last_seen_at is the best available proxy for "last time this row
-- was touched" (set by the upsert in topic_collector.py).
-- Only update rows that still carry the migration default (i.e. updated_at
-- equals their ingested_at, which is what PostgreSQL sets when ADD COLUMN runs).
-- Rows added after the migration will already have correct values.
UPDATE daily_news
SET    updated_at = last_seen_at
WHERE  updated_at >= (SELECT MIN(ingested_at) FROM daily_news)  -- safety guard
  AND  last_seen_at IS NOT NULL;

-- Step 3: Index to support the scheduler reconciliation queries:
--   WHERE updated_at < now() - INTERVAL '30 minutes'
CREATE INDEX IF NOT EXISTS daily_news_updated_at
    ON daily_news (updated_at);

-- Verify
SELECT
    COUNT(*)                          AS total_rows,
    COUNT(updated_at)                 AS rows_with_updated_at,
    MIN(updated_at)::date             AS oldest,
    MAX(updated_at)::date             AS newest
FROM daily_news;
