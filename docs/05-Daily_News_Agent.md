# 05-Daily_News_Agent.md

> **Implementation location**: All news ingest pipelines (`collectors/`, `enrichment/`, `clustering/`,
> `images/`, `summaries/`, `scheduler/`) live in the **`sage_kaizen_ai_ingest`** project at
> `F:\Projects\sage_kaizen_ai_ingest\news\`.  
> Only the query-time retrieval layer (`news/retrieval/news_resolver.py`, `news/retrieval/market_client.py`)
> remains in this project. Run the scheduler with `python scripts/run_scheduler.py` from the ingest project.

## Purpose

This document defines a production-ready design for the Sage Kaizen Daily News Runtime.

The goal is to make Sage Kaizen capable of answering questions such as:

- "What's in the news today?"
- "What are today's top stories?"
- "Summarize the top stories from the last 7 days."
- "What was the stock price for Nvidia yesterday?"
- "Find similar images for today's AI news only."
- "Find images for science category from the last 7 days."
- "Find related images from Reuters-like sources only."
- "Summarize the news of the war with Iran since the war sstarted."

This design assumes:

- Sage Kaizen runs on the user's Windows 11 Pro workstation.
- Sage Kaizen runs on the user's rig also known as "my rig":
  - OS: Windows 11 Professional
  - CPU: AMD Ryzen 9 9950X3D
  - RAM: 192 GB DDR5
  - GPU0: RTX 5090 (32 GB VRAM)
  - GPU1: RTX 5080 (16 GB VRAM)
  - Storage: 40 TB mixed SSD/HDD
- SearXNG is already running locally in Docker instance (http://localhost:8080).
- APScheduler is used inside Python for recurring jobs and retry logic.
- PostgreSQL with pgvector is the source of truth for articles, summaries, runs, and image vectors.
- No dated markdown report artifacts are generated.
- News-related images are stored in the `H:\article_images\` directory, managed through `media_files`, and embedded into a dedicated `news_image_embeddings` table.
- The architecture must remain modular and replaceable.

---

## Architectural Summary

Use one **News Runtime** composed of:

1. **Topic collectors**
   - Parallel jobs that query local SearXNG for configured topics and query templates.
   - These are I/O-bound and safe to run concurrently.

2. **Normalization and enrichment pipeline**
   - Canonicalize URLs.
   - Deduplicate repeated articles.
   - Fetch article text when enabled.
   - Discover and register news images.
   - Persist normalized article records.

3. **Clustering and summary pipeline**
   - Group related articles into story clusters.
   - Generate article summaries.
   - Generate cluster summaries.
   - Generate final daily and multi-day briefs.

4. **Query-serving layer**
   - Answer DB-first questions from stored summaries and articles.
   - Route live-data questions to search/market lookups when needed.
   - Support hybrid queries that combine stored news with current external data.

This is intentionally not "many autonomous full-stack agents all doing everything." It is one runtime with controlled parallel collection and controlled heavy synthesis.

---

## Why This Design Fits Sage Kaizen

- Collection work is mostly network-bound and lightweight, so multiple topic collectors can run in parallel without stressing the machine.
- Heavy summarization and synthesis should remain controlled to avoid interfering with the fast and architect brains used for live chat.
- Database-first persistence makes the system queryable, auditable, and rebuildable.
- Topic-driven configuration lets the user directly control what Sage Kaizen tracks.
- Image embeddings are treated as first-class retrieval assets for multimodal news questions.

---

## Runtime Model

### Parallel Tier

Safe to run simultaneously:

- SearXNG topic searches
- Article fetches
- Image metadata discovery
- URL normalization
- Exact deduplication
- Basic metadata extraction

### Controlled Tier

Limit concurrency:

- Semantic deduplication
- Story clustering
- Article summarization
- Cluster summarization
- Image embedding generation

### Serialized Tier

Allow only one active finalization per profile/window:

- Daily brief generation
- 7-day summary generation
- Rebuild operations for a given brief/profile

---

## Scheduler Design

Use APScheduler inside Python as the runtime scheduler.

Recommended jobs:

- topic collection jobs
- article enrichment jobs
- image processing jobs
- article summarization jobs
- cluster summarization jobs
- brief finalization jobs
- reconciliation / retry jobs

Operational rules:

- prevent overlapping finalizers for the same profile and date window
- keep ingestion idempotent
- allow retries without duplicating logical articles
- persist every run in the database
- distinguish between "collection complete" and "brief finalized"

---

## Core Data Model

The original `daily_news` draft should be treated as a starter concept only. The production model should contain the following logical tables.

### A. `news_topics`

Purpose:
Defines the topics Sage Kaizen should track.

Suggested columns:

- `topic_id`
- `topic_slug`
- `display_name`
- `description`
- `is_enabled`
- `priority_weight`
- `default_category`
- `default_time_range`
- `collection_strategy`
- `created_at`
- `updated_at`
- `metadata`

Examples:

- ai
- technology
- science
- world
- business
- cybersecurity
- semiconductors
- nvidia
- robotics

### B. `news_topic_queries`

Purpose:
Stores one or more search templates per topic.

Suggested columns:

- `topic_query_id`
- `topic_id`
- `query_text`
- `searxng_categories`
- `preferred_engines`
- `language_code`
- `region_code`
- `time_range`
- `is_enabled`
- `rank_weight`
- `max_results`
- `notes`
- `created_at`
- `updated_at`

Why this exists:
A single topic often needs multiple query formulations.

### C. `news_profiles`

Purpose:
Defines end-user briefing scopes.

Suggested columns:

- `profile_id`
- `profile_name`
- `description`
- `is_enabled`
- `summary_window_default`
- `top_n_default`
- `include_market_data`
- `created_at`
- `updated_at`

Examples:

- general_brief
- ai_brief
- science_brief
- markets_brief

### D. `news_profile_topics`

Purpose:
Maps topics into profiles.

Suggested columns:

- `profile_id`
- `topic_id`
- `weight_override`
- `is_required`
- `sort_order`

### E. `news_runs`

Purpose:
Tracks each scheduled execution.

Suggested columns:

- `run_id`
- `run_type`
- `profile_id`
- `topic_id`
- `scheduled_for`
- `started_at`
- `finished_at`
- `status`
- `retry_count`
- `attempt_group_id`
- `worker_id`
- `metrics_json`
- `error_text`
- `metadata`
- `created_at`

Use cases:

- auditing
- retries
- monitoring freshness
- diagnosing failed collection vs failed summarization

### F. `daily_news` (repurposed as normalized article store)

Purpose:
Stores normalized article records.

Suggested columns:

- `article_id`
- `first_seen_run_id`
- `topic_id`
- `headline`
- `snippet`
- `article_content`
- `news_source`
- `news_source_url`
- `canonical_url`
- `url_hash`
- `published_at`
- `first_seen_at`
- `last_seen_at`
- `language_code`
- `search_query`
- `search_category`
- `rank_score`
- `dedupe_fingerprint`
- `cluster_id`
- `fetch_status`
- `summary_status`
- `image_status`
- `metadata`
- `ingested_at`

Important notes:

- Do not keep a single `article_images` UUID column on this table.
- One article may have multiple images.
- Use separate relationship tables for article/image linkage.

### G. `news_story_clusters`

Purpose:
Groups related articles into one event or story.

Suggested columns:

- `cluster_id`
- `topic_id`
- `profile_id`
- `cluster_title`
- `representative_article_id`
- `importance_score`
- `story_start_at`
- `story_end_at`
- `article_count`
- `created_at`
- `updated_at`
- `metadata`

### H. `news_article_summaries`

Purpose:
Stores article-level extracted summaries.

Suggested columns:

- `article_summary_id`
- `article_id`
- `run_id`
- `summary_kind`
- `summary_short`
- `summary_medium`
- `summary_long`
- `key_points_json`
- `entities_json`
- `sentiment_label`
- `model_name`
- `model_version`
- `prompt_version`
- `generated_at`
- `is_active`
- `metadata`

Why this table matters:

- supports rebuilds without re-running article extraction
- enables fast retrieval for article-level questions
- preserves model provenance and allows re-summarization later

### I. `news_cluster_summaries`

Purpose:
Stores cluster-level summaries.

Suggested columns:

- `cluster_summary_id`
- `cluster_id`
- `run_id`
- `summary_kind`
- `summary_short`
- `summary_medium`
- `summary_long`
- `top_facts_json`
- `source_diversity_json`
- `confidence_score`
- `model_name`
- `model_version`
- `prompt_version`
- `generated_at`
- `is_active`
- `metadata`

Why this table matters:

- daily briefs should usually summarize clusters, not raw articles
- cluster-level summaries make 7-day rollups much better
- source-diversity fields help explain multi-publisher consensus

### J. `news_briefs`

Purpose:
Stores final DB-native summaries for user-facing retrieval.

Suggested columns:

- `brief_id`
- `profile_id`
- `run_id`
- `brief_date`
- `window_start_at`
- `window_end_at`
- `brief_kind`
- `headline_summary`
- `summary_short`
- `summary_long`
- `top_story_cluster_ids`
- `model_name`
- `model_version`
- `generated_at`
- `freshness_at`
- `is_final`
- `metadata`

Brief kinds:

- daily
- rolling_7_day
- topic
- market_context

### K. `news_article_images`

Purpose:
Maps articles to one or more images.

Suggested columns:

- `article_image_id`
- `article_id`
- `media_id`
- `source_image_url`
- `image_role`
- `caption_text`
- `surrounding_text`
- `position_index`
- `discovered_at`
- `metadata`

Image roles:

- hero
- thumbnail
- inline
- gallery

---

## News Image Vector Design

The original `image_embeddings` pattern is a useful starting point but is too minimal for news retrieval. A dedicated table is recommended.

### `news_image_embeddings`

Purpose:
Stores vector embeddings and retrieval metadata for news-related images.

Suggested columns:

- `news_image_embed_id`
- `media_id`
- `article_id`
- `cluster_id`
- `topic_id`
- `profile_id`
- `source_name`
- `source_url`
- `canonical_source_domain`
- `image_role`
- `caption_text`
- `surrounding_text`
- `published_at`
- `first_seen_at`
- `embedding_model`
- `embedding_version`
- `embedding_metric`
- `embedding`
- `content_hash`
- `is_active`
- `created_at`
- `updated_at`
- `metadata`

Design reasons:

- supports topic-aware filtering
- supports source-aware filtering
- supports date-window filtering
- supports re-embedding/versioning
- avoids expensive deep joins for common image-search workflows

---

## Retrieval Plan for News Images

### Required index strategy

1. **HNSW** on `news_image_embeddings.embedding`
2. **B-tree indexes** on common filter columns
3. **Careful filtered-query design** when recall matters

### Common filter columns to index

At minimum:

- `topic_id`
- `published_at`
- `first_seen_at`
- `source_name`
- `canonical_source_domain`
- `article_id`
- `cluster_id`
- `is_active`

### Retrieval principles

For queries such as:

- "find similar images for today's AI news only"
- "find images for science category from the last 7 days"
- "find related images from Reuters-like sources only"

the system should:

1. apply metadata filters first
2. rank within the filtered candidate set by vector distance
3. increase candidate breadth when filters are narrow
4. use iterative scan behavior when filtered ANN recall matters
5. prefer denormalized filter fields in `news_image_embeddings` for speed

### Denormalization guidance

Keep these directly in `news_image_embeddings`:

- `topic_id`
- `published_at`
- `source_name`
- `canonical_source_domain`

That is the best tradeoff for filtered similarity search.

---

## SearXNG Collection Strategy

SearXNG runs locally in Docker and is the primary search source.

### Collector rules

- use JSON output
- query by configured topic templates
- use topic/category-aware requests
- prefer time windows such as "day" for daily runs and "week" for reconciliation jobs
- record query metadata in the database
- preserve raw response fragments in JSONB where useful

### Recommended workflow

For each enabled topic:

1. load enabled query templates
2. submit SearXNG search requests
3. normalize result items
4. canonicalize URLs
5. upsert articles
6. schedule enrichment for new or materially changed items

---

## Summarization Strategy

### Article level

Use `news_article_summaries` for:

- concise extracted summary
- medium summary
- detailed summary
- key facts
- named entities
- provenance

### Cluster level

Use `news_cluster_summaries` for:

- event-level summary
- source diversity
- consensus / disagreement framing
- confidence score

### Brief level

Use `news_briefs` for:

- "today's top stories"
- "top stories from the last 7 days"
- topic-scoped summaries
- optional market-context summaries

### Brain usage

Recommended split:

- Fast brain:
  - lightweight extraction
  - article summary generation
  - preliminary cluster condensation

- Architect brain:
  - final cluster synthesis when needed
  - daily brief generation
  - 7-day rollups
  - hybrid analysis that combines stored news with live data

---

## Query-Serving Model

User requests should be classified into one of these conceptual groups:

- `news_today`
- `news_top_stories_today`
- `news_summary_last_n_days`
- `news_topic_summary`
- `market_point_lookup`
- `market_history_lookup`
- `news_image_similarity`
- `news_source_filtered_similarity`
- `hybrid_news_plus_market`

### DB-first queries

Answer from Postgres first:

- "What's in the news today?"
- "What are today's top stories?"
- "Summarize the top stories from the last 7 days."

### Live-data queries

Use live market/search retrieval:

- "What was the stock price for Nvidia yesterday?"

### Hybrid queries

Use both stored news and live lookups:

- "How did today's top AI stories affect Nvidia stock?"

---

## Router Integration Guidance

The existing router already detects many live-data and search-triggering phrases. The next implementation should add a second-stage interpretation that decides among:

- DB-only
- live-only
- hybrid

The runtime should not treat all search-worthy requests the same.

Recommended path:

1. existing router decides `needs_search` and broad category hints
2. a news-aware retrieval layer determines:
   - whether the answer can come from stored news data
   - whether it also requires live market/news lookup
   - whether it should use image retrieval
3. the orchestrator composes the final answer path

---

## Operational Requirements

- all ingestion must be idempotent
- retries must not create duplicate logical articles
- final briefs must be locked per profile/date window
- failures must be visible in `news_runs`
- article/content/image/summarization states must be explicit
- image embeddings must support re-processing without data loss
- model provenance must be stored for every summary layer

---

## Production Rollout Order

### Phase 1
- topic tables
- profile tables
- run tracking
- refactor `daily_news` into normalized article storage

### Phase 2
- parallel SearXNG collectors
- URL canonicalization
- exact deduplication
- article upsert path

### Phase 3
- article enrichment
- image discovery
- `news_article_images`
- `news_image_embeddings`

### Phase 4
- story clustering
- `news_article_summaries`
- `news_cluster_summaries`

### Phase 5
- `news_briefs`
- daily and 7-day finalizers
- DB-first query serving

### Phase 6
- router-aware DB vs live vs hybrid selection
- filtered image retrieval
- operational dashboards and retry tooling

---

## Non-Goals for v1

- autonomous code changes by the news runtime
- uncontrolled self-directed agents
- filesystem report artifacts
- unconstrained concurrent final summarizers
- unsupported direct reliance on image vectors without metadata filtering

---

## Final Recommendation

The production-ready design for Sage Kaizen is:

- one News Runtime
- multiple parallel topic collectors
- one normalized article store
- one story-cluster layer
- one article summary layer
- one cluster summary layer
- one brief layer
- one dedicated news-image vector table
- one hybrid query path that can answer from DB, live sources, or both

This structure is the best fit for:
- your hardware
- your local SearXNG deployment
- APScheduler-based recurring jobs
- PostgreSQL with pgvector
- Sage Kaizen's long-term modular architecture
