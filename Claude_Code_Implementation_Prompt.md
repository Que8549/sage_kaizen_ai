# Claude_Code_Implementation_Prompt.md

You are working inside the Sage Kaizen repository.

Your job is to implement the Daily News Runtime described in `05-Daily_News_Agent.md`.

Search the internet for the latest information to validate your recommendations and verify your implementation.

Ask questions if necessary.

## Critical constraints

Follow these rules exactly:

1. Do not replace the overall Sage Kaizen architecture.
2. Keep the design modular and production-oriented.
3. Use PostgreSQL with pgvector as the source of truth.
4. SearXNG is already running locally in Docker and must be used as the primary news collection source.
5. Use APScheduler inside Python for recurring jobs and retry logic.
6. Do not generate dated markdown news report artifacts.
7. Store summaries in Postgres only.
8. Treat news images as first-class assets using `media_files`, `news_article_images`, and `news_image_embeddings`.
9. Respect the existing Sage Kaizen router intent-hinting approach and integrate rather than replace it.
10. The implementation must support article summaries, cluster summaries, daily briefs, 7-day summaries, and filtered image similarity retrieval.
11. Keep the code consistent with the Sage Kaizen project structure and prior architecture patterns.
12. Prefer production-safe behavior over shortcuts.

## Deliverables

Implement the Daily News Runtime in stages, with clear commits or patch groups per stage.

### Stage 1 — Repository review and plan

First, inspect the repository and produce a concise implementation plan that identifies:

- the best folder locations for the news runtime
- the best place to integrate APScheduler startup
- where to hook DB-first and hybrid query resolution into the existing chat/query path
- where to integrate SearXNG client logic
- where media/image pipeline code should live
- any conflicts with existing ingestion, router, or retrieval modules

Do not start by making broad speculative edits. Ground the implementation in the actual repo structure.

### Stage 2 — Schema implementation

Add production-ready schema support for the following logical tables:

- `news_topics`
- `news_topic_queries`
- `news_profiles`
- `news_profile_topics`
- `news_runs`
- `daily_news` (refactor into normalized article storage while preserving safe migration behavior)
- `news_story_clusters`
- `news_article_summaries`
- `news_cluster_summaries`
- `news_briefs`
- `news_article_images`
- `news_image_embeddings`

Requirements:

- preserve safe foreign-key relationships
- include status/provenance fields
- support idempotent ingestion
- support re-summarization and re-embedding
- support filtered vector search on news images
- use pgvector-compatible indexing choices appropriate for production

Also include migration-safe notes if the existing `daily_news` table or related objects already exist.

### Stage 3 — SearXNG collection layer

Implement a production-safe local SearXNG client.

Requirements:

- configurable base URL for local Docker-hosted SearXNG
- JSON search requests
- support for categories, time ranges, engine preferences, paging, and query templates
- retries and timeout handling
- structured normalization of results into article candidates
- storage of raw or partial response metadata in JSONB when useful

Then implement topic-driven collectors that:

- load enabled topics
- load enabled topic queries
- collect results in parallel
- normalize and canonicalize URLs
- upsert article records idempotently
- queue enrichment for new or materially changed articles

### Stage 4 — Enrichment pipeline

Implement article enrichment.

Requirements:

- optional fetching of article text when enabled
- extraction of headline, snippet, source, canonical URL, language, timestamps, and related metadata
- explicit article processing states
- safe failure handling
- no duplicate logical articles

### Stage 5 — Image pipeline

Implement the news image pipeline.

Requirements:

- discover image URLs associated with news articles
- register/download images through the existing media pipeline as appropriate for the repo
- populate `news_article_images`
- generate embeddings into `news_image_embeddings`
- persist topic/date/source/domain/image-role metadata directly in `news_image_embeddings`
- support image reprocessing without breaking article relationships

Design for future filtered image retrieval such as:

- similar images for today's AI news
- science-category images from the last 7 days
- Reuters-like source filtered image retrieval

### Stage 6 — Clustering and summaries

Implement story clustering and both summary layers.

#### A. Article summary layer

Use `news_article_summaries` for:

- short summary
- medium summary
- long summary
- key points
- entities
- model provenance
- prompt provenance
- active/inactive versioning

#### B. Cluster summary layer

Use `news_cluster_summaries` for:

- short summary
- medium summary
- long summary
- top facts
- source diversity / consensus metadata
- confidence score
- model provenance
- active/inactive versioning

#### C. Brief layer

Use `news_briefs` for:

- daily top stories
- rolling 7-day summaries
- topic-scoped summaries

The implementation should avoid regenerating everything from scratch when only the final brief needs refresh.

### Stage 7 — APScheduler runtime

Implement the in-process scheduler.

Requirements:

- recurring topic collection jobs
- enrichment jobs
- image processing jobs
- summarization jobs
- brief finalization jobs
- reconciliation/retry jobs

Operational safeguards:

- prevent overlapping finalizers for the same profile/window
- prevent duplicate logical work when jobs retry
- store run status and metrics in `news_runs`
- ensure the runtime is restart-safe

### Stage 8 — Query integration

Extend Sage Kaizen's retrieval path so user questions can be resolved as:

- DB-only
- live-only
- hybrid

The system must support these user-facing behaviors:

- "What's in the news today?"
- "What are today's top stories?"
- "Summarize the top stories from the last 7 days."
- "What was the stock price for Nvidia yesterday?"
- image-related filtered retrieval requests

Use the existing router style and intent hints as a starting point. Do not rip it out. Add a news-aware resolution layer after routing.

The retrieval layer should decide whether to answer from:

- stored articles and summaries in Postgres
- live search / live market data
- or both

### Stage 9 — Filtered image retrieval

Implement retrieval logic consistent with pgvector best practices.

Requirements:

- HNSW on `news_image_embeddings.embedding`
- B-tree indexes on common filter columns
- careful filtered-query planning when recall matters
- support metadata filters on topic, date window, source, and source domain

The code should be structured so that filtered ANN behavior can be tuned later without invasive rewrites.

### Stage 10 — Tests, validation, and docs

Add:

- focused tests for URL dedupe and article upsert behavior
- tests for topic query loading
- tests for summary versioning behavior
- tests for image/article relationships
- tests for DB-first vs live vs hybrid query selection
- implementation notes for operators
- a short runbook for how to seed topics, run the scheduler, and inspect failures

## Implementation style

- Make small, well-scoped patches.
- Prefer explicit types and clean interfaces.
- Reuse existing project patterns where they already fit.
- Do not add unnecessary frameworks.
- Keep configuration centralized.
- Log important runtime decisions.
- Preserve future replacement paths for search, embedding, or summarization components.

## Important design expectations

### Topic-driven control
The user must be able to define tracked topics and query templates in the database. Do not hardcode all tracked topics into Python.

### Summary layering
Do not collapse article summaries, cluster summaries, and briefs into a single generic blob table. Keep them distinct.

### Image retrieval
Do not reuse the minimal `image_embeddings` concept unchanged for news. Use the dedicated `news_image_embeddings` model with provenance/filter metadata.

### Query-serving behavior
Do not answer all news-like questions by always calling live search. Use stored summaries for routine daily/weekly news questions unless freshness rules require a refresh.

### Market questions
Questions like "What was the stock price for Nvidia yesterday?" must go through a live/trusted market path rather than trying to infer the answer from news summaries alone.

## Output expectations

When you work:

1. First produce an implementation plan grounded in the repo.
2. Then implement in stages.
3. After each stage, summarize:
   - what was changed
   - what files were added or modified
   - any migration or operational risks
   - what remains next

When uncertain, choose the safer and more maintainable implementation.

Do not stop at only schema design. Implement the actual runtime path through scheduling, ingestion, summarization, image handling, and query integration.
