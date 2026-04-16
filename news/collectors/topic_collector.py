"""
news/collectors/topic_collector.py

Parallel topic-driven news collector.

Loads all enabled topics and their query templates from PostgreSQL, submits
searches to SearXNG concurrently (one worker thread per topic), normalises
results, and upserts article candidates into daily_news.

Key design points:
  - One news_runs row is created per topic execution for auditing.
  - Upsert key is url_hash (SHA-256 of canonical URL); ON CONFLICT updates
    last_seen_at and improves headline/snippet/rank_score if better.
  - New articles land with fetch_status='pending' so the enrichment pipeline
    picks them up automatically.
  - The xmax trick (xmax = 0 → new insert) lets us count new vs updated rows.
  - All DB writes use psycopg3 thread-local connections (rag_v1/db/pg.py).
  - Thread safety: SearXNGNewsClient is shared; its internal rate limiter is
    thread-safe.  DB connections are thread-local — no sharing needed.

Public API:
    collector = TopicCollector()
    result = collector.run_once()          # collect all enabled topics
    result = collector.run_once(["ai"])    # collect specific topics by slug
"""
from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from news.collectors.searxng_news_client import NewsArticleCandidate, SearXNGNewsClient
from news.news_settings import get_news_settings
from pg_settings import PgSettings
from rag_v1.db.pg import conn_ctx
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.topic_collector", file_name="news_agent.log")

# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_LOAD_TOPICS_SQL = """
SELECT
    t.topic_id::text,
    t.topic_slug,
    t.display_name,
    t.default_time_range
FROM news_topics t
WHERE t.is_enabled = true
{slug_filter}
ORDER BY t.priority_weight DESC, t.topic_slug
"""

_LOAD_QUERIES_SQL = """
SELECT
    q.topic_query_id::text,
    q.query_text,
    q.searxng_categories,
    q.preferred_engines,
    q.time_range,
    q.max_results,
    q.rank_weight
FROM news_topic_queries q
WHERE q.topic_id = %s::uuid
  AND q.is_enabled = true
ORDER BY q.rank_weight DESC, q.topic_query_id
"""

_INSERT_RUN_SQL = """
INSERT INTO news_runs (
    run_id, run_type, topic_id, scheduled_for, started_at,
    status, worker_id, attempt_group_id, metadata
)
VALUES (
    %s::uuid, 'collection', %s::uuid, now(), now(),
    'running', %s, %s::uuid, '{}'::jsonb
)
"""

_UPDATE_RUN_SQL = """
UPDATE news_runs
SET
    finished_at  = now(),
    status       = %s,
    metrics_json = %s::jsonb,
    error_text   = %s
WHERE run_id = %s::uuid
"""

_UPSERT_ARTICLE_SQL = """
INSERT INTO daily_news (
    first_seen_run_id,
    topic_id,
    headline,
    snippet,
    news_source,
    news_source_url,
    canonical_url,
    url_hash,
    published_at,
    language_code,
    search_query,
    search_category,
    rank_score,
    dedupe_fingerprint,
    fetch_status,
    summary_status,
    image_status,
    metadata
) VALUES (
    %s::uuid,
    %s::uuid,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    'pending',
    'pending',
    'pending',
    %s::jsonb
)
ON CONFLICT (url_hash) DO UPDATE SET
    last_seen_at    = now(),
    headline        = COALESCE(EXCLUDED.headline,  daily_news.headline),
    snippet         = COALESCE(EXCLUDED.snippet,   daily_news.snippet),
    rank_score      = GREATEST(EXCLUDED.rank_score, daily_news.rank_score),
    search_query    = EXCLUDED.search_query,
    search_category = EXCLUDED.search_category
RETURNING article_id, (xmax = 0) AS is_new
"""


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _TopicQueryRow:
    topic_query_id: str
    query_text: str
    searxng_categories: list[str]
    preferred_engines: list[str]
    time_range: str
    max_results: int
    rank_weight: float


@dataclass
class _TopicRow:
    topic_id: str
    topic_slug: str
    display_name: str
    default_time_range: str
    queries: list[_TopicQueryRow] = field(default_factory=list)


@dataclass
class TopicResult:
    """Per-topic collection outcome."""
    topic_id: str
    topic_slug: str
    run_id: str
    new_articles: int = 0
    updated_articles: int = 0
    query_errors: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class CollectionResult:
    """Aggregated outcome from a full run_once() call."""
    group_id: str
    topics_run: int = 0
    topics_ok: int = 0
    topics_failed: int = 0
    total_new: int = 0
    total_updated: int = 0
    duration_s: float = 0.0
    topic_results: list[TopicResult] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"CollectionResult(topics={self.topics_run}, ok={self.topics_ok}, "
            f"failed={self.topics_failed}, new={self.total_new}, "
            f"updated={self.total_updated}, duration={self.duration_s:.1f}s)"
        )


# ---------------------------------------------------------------------------
# TopicCollector
# ---------------------------------------------------------------------------

class TopicCollector:
    """
    Orchestrates parallel news collection for all enabled topics.

    Typical usage (called by NewsScheduler every N minutes):
        collector = TopicCollector()
        result = collector.run_once()

    The collector is stateless between calls; each run_once() is independent.
    A single SearXNGNewsClient instance is shared across threads for its
    built-in rate limiter.
    """

    def __init__(self, client: SearXNGNewsClient | None = None) -> None:
        self._client = client or SearXNGNewsClient()
        self._cfg = get_news_settings()
        self._dsn = self._cfg.pg_dsn

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_once(self, topic_slugs: list[str] | None = None) -> CollectionResult:
        """
        Collect news for all enabled topics (or a specific subset by slug).

        Runs topic collections in parallel (max_workers = scheduler concurrency).
        Each topic gets its own news_runs row for individual auditing.

        Args:
            topic_slugs: If provided, only collect these topic slugs.
                         If None, all enabled topics are collected.

        Returns:
            CollectionResult with aggregated stats.
        """
        group_id = str(uuid.uuid4())
        t0 = time.monotonic()

        topics = self._load_topics(topic_slugs)
        if not topics:
            _LOG.info("collector | no enabled topics found | slugs_filter=%s", topic_slugs)
            return CollectionResult(group_id=group_id)

        _LOG.info(
            "collector | start | group=%s | topics=%d | filter=%s",
            group_id, len(topics), topic_slugs or "all",
        )

        result = CollectionResult(group_id=group_id, topics_run=len(topics))
        max_workers = min(self._cfg.scheduler_max_workers, len(topics))

        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix="news_collect") as pool:
            futures: dict[Future[TopicResult], _TopicRow] = {
                pool.submit(self._collect_topic, topic, group_id): topic
                for topic in topics
            }
            for future in as_completed(futures):
                topic = futures[future]
                try:
                    tr = future.result()
                except Exception as exc:
                    _LOG.error(
                        "collector | topic=%s | unhandled exception: %s",
                        topic.topic_slug, exc, exc_info=True,
                    )
                    tr = TopicResult(
                        topic_id=topic.topic_id,
                        topic_slug=topic.topic_slug,
                        run_id="",
                        error=str(exc),
                    )

                result.topic_results.append(tr)
                if tr.succeeded:
                    result.topics_ok += 1
                    result.total_new += tr.new_articles
                    result.total_updated += tr.updated_articles
                else:
                    result.topics_failed += 1

        result.duration_s = time.monotonic() - t0
        _LOG.info("collector | done | %s", result)
        return result

    # ------------------------------------------------------------------
    # Internal: data loading
    # ------------------------------------------------------------------

    def _load_topics(self, slugs: list[str] | None) -> list[_TopicRow]:
        """Load enabled topics and their active query templates from DB."""
        slug_filter = ""
        params: list = []
        if slugs:
            placeholders = ", ".join("%s" for _ in slugs)
            slug_filter = f"AND t.topic_slug IN ({placeholders})"
            params = list(slugs)

        sql = _LOAD_TOPICS_SQL.format(slug_filter=slug_filter)

        with conn_ctx(self._dsn) as conn:
            topic_rows = conn.execute(sql, params).fetchall()

        topics: list[_TopicRow] = []
        for row in topic_rows:
            queries = self._load_queries(row["topic_id"])
            if not queries:
                _LOG.warning(
                    "collector | topic=%s has no enabled queries — skipping",
                    row["topic_slug"],
                )
                continue
            topics.append(_TopicRow(
                topic_id=row["topic_id"],
                topic_slug=row["topic_slug"],
                display_name=row["display_name"],
                default_time_range=row["default_time_range"],
                queries=queries,
            ))

        return topics

    def _load_queries(self, topic_id: str) -> list[_TopicQueryRow]:
        """Load enabled query templates for one topic."""
        with conn_ctx(self._dsn) as conn:
            rows = conn.execute(_LOAD_QUERIES_SQL, [topic_id]).fetchall()

        return [
            _TopicQueryRow(
                topic_query_id    = r["topic_query_id"],
                query_text        = r["query_text"],
                searxng_categories= list(r["searxng_categories"] or []),
                preferred_engines = list(r["preferred_engines"] or []),
                time_range        = r["time_range"] or "day",
                max_results       = int(r["max_results"]),
                rank_weight       = float(r["rank_weight"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Internal: per-topic collection
    # ------------------------------------------------------------------

    def _collect_topic(self, topic: _TopicRow, group_id: str) -> TopicResult:
        """
        Run all queries for one topic, upsert results, track in news_runs.

        Designed to run inside a ThreadPoolExecutor worker.
        """
        run_id = str(uuid.uuid4())
        worker_id = f"topic_collector:{topic.topic_slug}"
        t0 = time.monotonic()

        self._insert_run(run_id, topic.topic_id, group_id, worker_id)

        tr = TopicResult(
            topic_id=topic.topic_id,
            topic_slug=topic.topic_slug,
            run_id=run_id,
        )

        try:
            for q in topic.queries:
                candidates = self._client.search(
                    query_text = q.query_text,
                    categories = q.searxng_categories or None,
                    engines    = q.preferred_engines or None,
                    time_range = q.time_range,
                    max_results= q.max_results,
                    language   = "en",
                )
                if not candidates:
                    tr.query_errors += 1
                    continue

                new_c, updated_c = self._upsert_candidates(
                    candidates, topic.topic_id, run_id
                )
                tr.new_articles     += new_c
                tr.updated_articles += updated_c

            tr.duration_s = time.monotonic() - t0
            metrics = {
                "new_articles":     tr.new_articles,
                "updated_articles": tr.updated_articles,
                "query_errors":     tr.query_errors,
                "queries_run":      len(topic.queries),
                "duration_s":       round(tr.duration_s, 2),
            }
            self._update_run(run_id, "completed", metrics, None)
            _LOG.info(
                "collector | topic=%s | new=%d | updated=%d | errors=%d | %.1fs",
                topic.topic_slug, tr.new_articles, tr.updated_articles,
                tr.query_errors, tr.duration_s,
            )

        except Exception as exc:
            tr.error = str(exc)
            tr.duration_s = time.monotonic() - t0
            _LOG.error(
                "collector | topic=%s | failed | error=%s",
                topic.topic_slug, exc, exc_info=True,
            )
            self._update_run(run_id, "failed", {}, str(exc))

        return tr

    # ------------------------------------------------------------------
    # Internal: DB writes
    # ------------------------------------------------------------------

    def _insert_run(
        self,
        run_id: str,
        topic_id: str,
        group_id: str,
        worker_id: str,
    ) -> None:
        with conn_ctx(self._dsn) as conn:
            conn.execute(_INSERT_RUN_SQL, [run_id, topic_id, worker_id, group_id])

    def _update_run(
        self,
        run_id: str,
        status: str,
        metrics: dict,
        error_text: Optional[str],
    ) -> None:
        with conn_ctx(self._dsn) as conn:
            conn.execute(
                _UPDATE_RUN_SQL,
                [status, json.dumps(metrics), error_text, run_id],
            )

    def _upsert_candidates(
        self,
        candidates: list[NewsArticleCandidate],
        topic_id: str,
        run_id: str,
    ) -> tuple[int, int]:
        """
        Upsert a batch of candidates into daily_news within one transaction.

        Returns (new_count, updated_count).

        Upsert logic:
          - New URL  → insert with fetch/summary/image status = 'pending'
          - Known URL → update last_seen_at; keep existing statuses intact
                        (do NOT reset in-progress enrichment)
          - headline/snippet: COALESCE keeps existing non-null values
          - rank_score: GREATEST keeps the best score seen across runs
        """
        if not candidates:
            return 0, 0

        new_count = 0
        updated_count = 0

        with conn_ctx(self._dsn) as conn:
            with conn.transaction():
                for c in candidates:
                    params = [
                        run_id,                        # first_seen_run_id
                        topic_id,                      # topic_id
                        c.headline,                    # headline
                        c.snippet,                     # snippet
                        c.news_source,                 # news_source
                        c.news_source_url,             # news_source_url
                        c.canonical_url,               # canonical_url
                        c.url_hash,                    # url_hash  (bytes → bytea)
                        c.published_at,                # published_at
                        c.language_code,               # language_code
                        c.search_query,                # search_query
                        c.search_category,             # search_category
                        c.rank_score,                  # rank_score
                        c.dedupe_fingerprint,          # dedupe_fingerprint
                        json.dumps(c.raw_metadata),    # metadata (JSONB)
                    ]
                    row = conn.execute(_UPSERT_ARTICLE_SQL, params).fetchone()
                    if row and row["is_new"]:
                        new_count += 1
                    else:
                        updated_count += 1

        return new_count, updated_count
