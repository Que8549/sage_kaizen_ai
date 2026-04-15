# Runbook: Daily News Pipeline

## Overview

The Daily News Pipeline collects, enriches, clusters, summarizes, and finalizes news articles from SearXNG into `news_briefs` rows that are injected into Sage Kaizen chat turns as `<news_context>` blocks.

All pipeline state lives in PostgreSQL. The scheduler runs inside the Streamlit process via APScheduler 3.11.2 `BackgroundScheduler`.

---

## Prerequisites

| Requirement | Details |
|---|---|
| SearXNG | Running at `http://localhost:8080` (see `F:\Projects\searxng`) |
| PostgreSQL | Tables applied from `news/db/news_schema.sql` |
| Seed data | Topics and queries applied from `news/db/news_seed_data.sql` |
| BGE-M3 embed server | Port 8020 (for article embeddings) |
| jina-clip-v2 embed server | Port 8031 (for image embeddings) |
| H:\ drive | Mounted and writable (image storage: `H:\article_images`) |
| APScheduler | `pip install apscheduler==3.11.2` |
| simhash | `pip install simhash==2.1.2` |
| yfinance | `pip install yfinance==1.2.2` |
| html2text | `pip install html2text` |

---

## One-Time Schema Setup

Run these once as a PostgreSQL superuser (or the app DB user if it has CREATE TABLE privileges):

```sql
-- Apply the news schema
\i news/db/news_schema.sql

-- Seed the 11 topics, 5 profiles, and 33 query templates
\i news/db/news_seed_data.sql
```

Verify seed data:

```sql
SELECT topic_slug, is_enabled FROM news_topics ORDER BY topic_slug;
SELECT profile_name, is_enabled FROM news_profiles ORDER BY profile_name;
SELECT COUNT(*) FROM news_topic_queries;
```

Expected output:
- 11 rows in `news_topics`, all `is_enabled = true`
- 5 rows in `news_profiles`, all `is_enabled = true`
- 33 rows in `news_topic_queries`

---

## Starting the Scheduler

The scheduler starts automatically when the Streamlit UI loads via `@st.cache_resource` in `ui_streamlit_server.py`. No manual step is required.

To verify it started, check the log:

```
logs/sage_kaizen.log
```

Look for:

```
sage_kaizen.news.scheduler | started
sage_kaizen.news.scheduler | registered 9 jobs
```

---

## Job Schedule

| Job ID | Trigger | Default Interval | What It Does |
|---|---|---|---|
| `collect_all_topics` | interval | every 60 min | SearXNG search for all enabled topics |
| `enrich_articles` | interval | every 30 min | Full-text fetch + BGE-M3 embed for pending articles |
| `process_images` | interval | every 45 min | Download + jina-clip-v2 embed for article images |
| `cluster_articles` | interval | every 2 hours | DBSCAN clustering of embedded articles |
| `summarize_articles` | interval | every 30 min (off-peak) | FAST brain summaries for unclustered articles |
| `summarize_clusters` | interval | every 30 min (off-peak) | Cluster-level synthesis (FAST ≤5, ARCHITECT >5 articles) |
| `finalize_daily_brief` | cron | 06:00 UTC | ARCHITECT daily brief for all enabled profiles |
| `finalize_rolling_brief` | cron | 06:30 UTC | ARCHITECT 7-day rolling brief for all enabled profiles |
| `reconcile_failed` | cron | 03:00 UTC | Reset stuck/failed articles back to `pending` |

**Off-peak guard**: `summarize_articles` and `summarize_clusters` check `last_chat_activity_ts()` in `chat_service.py`. If the last chat turn was within `off_peak_idle_seconds` (default 180 s), the job skips. This prevents summarization from competing with the active user's response stream.

---

## Configuration

All settings live in `news/news_settings.py` and are overridable via environment variables with the `NEWS_` prefix.

Key settings:

| Env Var | Default | Purpose |
|---|---|---|
| `NEWS_SEARXNG_BASE_URL` | `http://localhost:8080` | SearXNG instance URL |
| `NEWS_BRIEF_FRESHNESS_HOURS` | `4` | Hours before a brief is considered stale |
| `NEWS_OFF_PEAK_IDLE_SECONDS` | `180` | Seconds of chat inactivity before summarization runs |
| `NEWS_IMAGE_STORAGE_PATH` | `H:\article_images` | Local path for downloaded images |
| `NEWS_COLLECTION_INTERVAL_MINUTES` | `60` | Collection job frequency |
| `NEWS_SUMMARIZATION_INTERVAL_MINUTES` | `30` | Summarization job frequency |
| `NEWS_DAILY_BRIEF_HOUR` | `6` | UTC hour for daily brief cron |
| `NEWS_ROLLING_BRIEF_HOUR` | `6` | UTC hour for rolling-7-day brief cron |
| `NEWS_ROLLING_BRIEF_MINUTE` | `30` | UTC minute for rolling-7-day brief cron |

---

## Manually Triggering a Collection Run

To trigger an immediate collection without waiting for the scheduler:

```python
from news.collectors.topic_collector import TopicCollector
result = TopicCollector().run_once()
print(result)
```

To collect only specific topics:

```python
result = TopicCollector().run_once(topic_slugs=["ai", "technology"])
```

---

## Manually Generating a Brief

To generate today's daily brief immediately (requires ARCHITECT brain running):

```python
from news.summaries.brief_finalizer import BriefFinalizer
result = BriefFinalizer().run_daily()
print(result)  # {"profiles": 5, "finalized": 3, "skipped": 1, "failed": 1}
```

To generate the 7-day rolling brief:

```python
result = BriefFinalizer().run_rolling_7day()
```

---

## Inspecting Pipeline State

### Check recent collection runs

```sql
SELECT run_type, status, started_at, finished_at,
       metrics_json->>'articles_new' AS new_articles,
       error_text
FROM news_runs
WHERE run_type = 'collection'
ORDER BY started_at DESC
LIMIT 20;
```

### Check article fetch status distribution

```sql
SELECT fetch_status, COUNT(*) AS n
FROM daily_news
WHERE created_at > now() - INTERVAL '24 hours'
GROUP BY fetch_status
ORDER BY n DESC;
```

### Check today's briefs

```sql
SELECT p.profile_name, b.brief_kind, b.is_final,
       b.freshness_at, LEFT(b.headline_summary, 80) AS headline
FROM news_briefs b
JOIN news_profiles p ON p.profile_id = b.profile_id
WHERE b.brief_date = CURRENT_DATE
ORDER BY p.profile_name, b.brief_kind;
```

### Find stuck articles

```sql
-- Articles stuck in 'fetching' for > 30 minutes (worker crashed)
SELECT article_id, url, updated_at
FROM daily_news
WHERE fetch_status = 'fetching'
  AND updated_at < now() - INTERVAL '30 minutes';

-- Articles with failed fetch but retries remaining
SELECT article_id, url,
       (metadata->>'fetch_retry_count')::int AS retries,
       updated_at
FROM daily_news
WHERE fetch_status = 'failed_fetch'
  AND (metadata->>'fetch_retry_count')::int < 3;
```

The `reconcile_failed` job (03:00 UTC) resets both of these automatically. To reset manually:

```sql
UPDATE daily_news
SET fetch_status = 'pending', updated_at = now()
WHERE fetch_status IN ('fetching', 'failed_fetch')
  AND updated_at < now() - INTERVAL '30 minutes';
```

---

## Disabling/Enabling Topics

To pause collection for a topic without deleting it:

```sql
UPDATE news_topics SET is_enabled = false WHERE topic_slug = 'united_kingdom';
```

To re-enable:

```sql
UPDATE news_topics SET is_enabled = true WHERE topic_slug = 'united_kingdom';
```

The scheduler picks up the change on the next collection run (no restart needed).

---

## Adding a New Topic

1. Insert the topic:
   ```sql
   INSERT INTO news_topics (topic_slug, topic_name, description, is_enabled)
   VALUES ('gaming', 'Gaming', 'Video games and gaming industry news', true);
   ```

2. Add query templates:
   ```sql
   INSERT INTO news_topic_queries (topic_id, query_template, searxng_categories, time_range, is_enabled)
   SELECT topic_id, 'latest gaming news', ARRAY['news'], 'day', true
   FROM news_topics WHERE topic_slug = 'gaming';
   ```

3. Associate with a profile:
   ```sql
   INSERT INTO news_profile_topics (profile_id, topic_id, weight)
   SELECT p.profile_id, t.topic_id, 1.0
   FROM news_profiles p, news_topics t
   WHERE p.profile_name = 'general_brief' AND t.topic_slug = 'gaming';
   ```

---

## Log Locations

All news pipeline logs go to `logs/sage_kaizen.log` (rotating, 10 MB / 5 backups).

Key logger names:

| Logger | What It Covers |
|---|---|
| `sage_kaizen.news.scheduler` | Job registration and lifecycle |
| `sage_kaizen.news.collector` | SearXNG fetches and article upserts |
| `sage_kaizen.news.enricher` | Full-text fetch and BGE-M3 embedding |
| `sage_kaizen.news.image_pipeline` | Image download and jina-clip-v2 embedding |
| `sage_kaizen.news.clusterer` | DBSCAN clustering runs |
| `sage_kaizen.news.article_summarizer` | FAST brain article summaries |
| `sage_kaizen.news.cluster_summarizer` | Cluster-level synthesis |
| `sage_kaizen.news.brief_finalizer` | Daily and rolling-7-day brief generation |
| `sage_kaizen.news.resolver` | DB-first query resolution (per chat turn) |
| `sage_kaizen.news.market_client` | yfinance market lookups |

---

## Troubleshooting

### Briefs are not being generated

1. Check that ARCHITECT brain (port 8012) is running: `curl http://localhost:8012/health`
2. Check `news_runs` for `brief_finalization` rows with `status = 'failed'` and inspect `error_text`
3. Check that story clusters exist with active summaries:
   ```sql
   SELECT COUNT(*) FROM news_story_clusters c
   JOIN news_cluster_summaries cs ON cs.cluster_id = c.cluster_id AND cs.is_active = true
   WHERE c.story_start_at >= CURRENT_DATE;
   ```

### No articles are being collected

1. Check SearXNG is running: `curl http://localhost:8080/search?q=test&format=json`
2. Check `news_runs` for `collection` rows with `status = 'failed'`
3. Check that `news_topics` has enabled topics and `news_topic_queries` has enabled queries

### Summarization never runs

1. The off-peak guard may be preventing it. Check chat activity: if Streamlit is receiving chat turns continuously, the guard stays active.
2. Lower `NEWS_OFF_PEAK_IDLE_SECONDS` to `60` temporarily to test.
3. Verify `last_chat_activity_ts()` is importable from `chat_service`:
   ```python
   from chat_service import last_chat_activity_ts
   print(last_chat_activity_ts())
   ```

### Image storage errors

1. Verify `H:\` is mounted and writable
2. Check `NEWS_IMAGE_STORAGE_PATH` env var or default `H:\article_images`
3. The pipeline creates subdirectories `H:\article_images\YYYY\MM\` automatically
