"""
news/clustering/article_clusterer.py

Story clustering via BGE-M3 embeddings + DBSCAN.

Groups recently collected articles into story clusters so that the
summarization layer can produce cluster-level (event-level) summaries
rather than individual article summaries.

Algorithm:
  1. Load articles from the last N hours that have article_embedding set
  2. L2-normalize the embedding matrix
  3. Run sklearn DBSCAN(metric='cosine', eps=cfg.cluster_eps, min_samples=2)
  4. Label=-1 articles remain unclustered (standalone)
  5. Upsert news_story_clusters rows, one per DBSCAN label
  6. Update daily_news.cluster_id for every article

Idempotency:
  Re-running produces new cluster rows for the current window; old cluster
  references on articles outside the window are not disturbed.  Within the
  window, cluster_id is overwritten — this is intentional since the cluster
  composition may change as new articles arrive.

Concurrency:
  Designed to run in a single BackgroundScheduler worker (not parallel).
  The DBSCAN step is CPU-bound but fast for hundreds of articles.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

from news.news_settings import get_news_settings
from rag_v1.db.pg import conn_ctx
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.clusterer", file_name="news_agent.log")

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_LOAD_ARTICLES_SQL = """
SELECT
    article_id::text,
    topic_id::text,
    headline,
    article_embedding::text AS embedding_text
FROM daily_news
WHERE article_embedding IS NOT NULL
  AND first_seen_at >= now() - (%(window_hours)s || ' hours')::interval
ORDER BY first_seen_at DESC
"""

_UPSERT_CLUSTER_SQL = """
INSERT INTO news_story_clusters
    (cluster_id, topic_id, cluster_title, importance_score,
     article_count, story_start_at, story_end_at)
VALUES
    (%s::uuid, %s::uuid, %s, %s, %s, %s, %s)
ON CONFLICT DO NOTHING
RETURNING cluster_id
"""

_UPDATE_ARTICLE_CLUSTER_SQL = """
UPDATE daily_news
SET cluster_id = %s::uuid, updated_at = now()
WHERE article_id = %s::uuid
"""

_CLEAR_ARTICLE_CLUSTER_SQL = """
UPDATE daily_news
SET cluster_id = NULL, updated_at = now()
WHERE article_id = %s::uuid
"""

_INSERT_RUN_SQL = """
INSERT INTO news_runs (run_id, run_type, scheduled_for, started_at, status, worker_id)
VALUES (%s::uuid, 'cluster_summarization', now(), now(), 'running', 'article_clusterer')
"""

_UPDATE_RUN_SQL = """
UPDATE news_runs
SET finished_at = now(), status = %s, metrics_json = %s::jsonb, error_text = %s
WHERE run_id = %s::uuid
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_vec(text: str) -> Optional[np.ndarray]:
    """Parse a pgvector '[f,f,f,...]' string into a numpy float32 array."""
    try:
        text = text.strip().lstrip("[").rstrip("]")
        return np.fromstring(text, dtype=np.float32, sep=",")
    except Exception:
        return None


def _cluster_title(headlines: list[str]) -> str:
    """Generate a simple cluster title from member headlines."""
    if not headlines:
        return "Untitled cluster"
    # Use the shortest headline as a proxy for the most concise title.
    return min(headlines, key=len)[:120]


# ---------------------------------------------------------------------------
# ArticleClusterer
# ---------------------------------------------------------------------------

class ArticleClusterer:
    """
    Clusters articles using BGE-M3 embeddings and DBSCAN.

    Typical call (from NewsScheduler, off-peak):
        clusterer = ArticleClusterer()
        result = clusterer.run_once()
    """

    def __init__(self) -> None:
        self._cfg = get_news_settings()
        self._dsn = self._cfg.pg_dsn

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_once(self) -> dict:
        """Cluster articles from the last N hours. Returns metrics dict."""
        run_id = str(uuid.uuid4())
        t0 = time.monotonic()

        with conn_ctx(self._dsn) as conn:
            conn.execute(_INSERT_RUN_SQL, [run_id])

        articles = self._load_articles()
        if len(articles) < 2:
            self._update_run(run_id, "completed",
                             {"articles": len(articles), "clusters": 0}, None)
            return {"articles": len(articles), "clusters": 0}

        _LOG.info("clusterer | start | run=%s | articles=%d", run_id, len(articles))

        ids      = [a["article_id"] for a in articles]
        topics   = [a.get("topic_id") for a in articles]
        headlines = [a.get("headline") or "" for a in articles]
        vecs_raw = [_parse_vec(a["embedding_text"]) for a in articles]

        # Drop articles with unparseable embeddings.
        valid = [(i, v) for i, v in enumerate(vecs_raw) if v is not None and v.shape[0] == 1024]
        if len(valid) < 2:
            self._update_run(run_id, "completed",
                             {"articles": len(articles), "clusters": 0}, None)
            return {"articles": len(articles), "clusters": 0}

        valid_idx  = [i for i, _ in valid]
        matrix     = np.stack([v for _, v in valid], axis=0).astype(np.float32)
        matrix     = normalize(matrix, norm="l2")

        labels = DBSCAN(
            eps=self._cfg.cluster_eps,
            min_samples=self._cfg.cluster_min_samples,
            metric="cosine",
            n_jobs=1,
        ).fit_predict(matrix)

        # Build cluster → article-index mapping.
        cluster_map: dict[int, list[int]] = {}
        for mat_i, label in enumerate(labels):
            if label == -1:
                continue  # noise; stays unclustered
            cluster_map.setdefault(label, []).append(valid_idx[mat_i])

        n_clusters = len(cluster_map)
        _LOG.info("clusterer | labels=%d | clusters=%d | noise=%d",
                  len(labels), n_clusters, sum(1 for l in labels if l == -1))

        # Persist clusters and update article cluster_ids.
        with conn_ctx(self._dsn) as conn:
            for label, art_indices in cluster_map.items():
                cluster_id = str(uuid.uuid4())
                member_headlines = [headlines[i] for i in art_indices]
                title = _cluster_title(member_headlines)

                # Compute a simple importance score (larger clusters = higher score).
                importance = float(len(art_indices)) / max(len(articles), 1)

                # Derive the most common topic in this cluster.
                topic_counts: dict = {}
                for i in art_indices:
                    t = topics[i]
                    if t:
                        topic_counts[t] = topic_counts.get(t, 0) + 1
                dominant_topic = max(topic_counts, key=topic_counts.get) if topic_counts else None

                with conn.transaction():
                    conn.execute(_UPSERT_CLUSTER_SQL, [
                        cluster_id,
                        dominant_topic,
                        title,
                        importance,
                        len(art_indices),
                        datetime.now(timezone.utc),  # story_start_at (approximate)
                        datetime.now(timezone.utc),  # story_end_at (updated as articles arrive)
                    ])
                    for i in art_indices:
                        conn.execute(_UPDATE_ARTICLE_CLUSTER_SQL, [cluster_id, ids[i]])

            # Clear cluster_id for noise articles (label=-1) within the window.
            for mat_i, label in enumerate(labels):
                if label == -1:
                    conn.execute(_CLEAR_ARTICLE_CLUSTER_SQL, [ids[valid_idx[mat_i]]])

        metrics = {
            "articles": len(articles),
            "valid_embeddings": len(valid),
            "clusters": n_clusters,
            "noise": int(sum(1 for l in labels if l == -1)),
            "duration_s": round(time.monotonic() - t0, 2),
        }
        self._update_run(run_id, "completed", metrics, None)
        _LOG.info("clusterer | done | %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_articles(self) -> list[dict]:
        with conn_ctx(self._dsn) as conn:
            return conn.execute(
                _LOAD_ARTICLES_SQL,
                {"window_hours": self._cfg.cluster_window_hours},
            ).fetchall()

    def _update_run(self, run_id: str, status: str,
                    metrics: dict, error: Optional[str]) -> None:
        with conn_ctx(self._dsn) as conn:
            conn.execute(
                _UPDATE_RUN_SQL,
                [status, json.dumps(metrics), error, run_id],
            )
