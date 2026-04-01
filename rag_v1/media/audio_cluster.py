"""
rag_v1/media/audio_cluster.py

KMeans clustering of audio_embeddings for "songs that sound similar to each
other" and artist/era grouping queries.

Algorithm
---------
  1. Load all (media_id, embedding) rows from audio_embeddings.
  2. Run sklearn KMeans(n_clusters) on the 512-dim vectors.
  3. Write cluster assignments to the audio_clusters table.

The cluster_label is set to "Cluster N" by default.  The MusicRetriever uses
these labels when answering "find songs that sound like each other".

Run standalone:
    python -m rag_v1.media.audio_cluster [--n-clusters 50]

Or call run_clustering() from media_ingest.py Phase 4 (--cluster).
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import psycopg
from psycopg.rows import DictRow

from rag_v1.db.pg import conn_ctx

_LOG = logging.getLogger("sage_kaizen.audio_cluster")

_SQL_LOAD_EMBEDDINGS = """
SELECT mf.media_id::text, ae.embedding
FROM   audio_embeddings ae
JOIN   media_files mf ON mf.media_id = ae.media_id
WHERE  mf.modality = 'audio'
ORDER  BY mf.file_path;
"""

_SQL_UPSERT_CLUSTER = """
INSERT INTO audio_clusters (media_id, cluster_id, cluster_label, clustered_at)
VALUES (%s::uuid, %s, %s, now())
ON CONFLICT (media_id) DO UPDATE SET
    cluster_id    = EXCLUDED.cluster_id,
    cluster_label = EXCLUDED.cluster_label,
    clustered_at  = EXCLUDED.clustered_at;
"""


@dataclass
class ClusterStats:
    n_files:    int = 0
    n_clusters: int = 0

    def report(self) -> str:
        return f"files={self.n_files}  clusters={self.n_clusters}"


def run_clustering(
    dsn: str,
    n_clusters: int = 50,
    random_state: int = 42,
) -> ClusterStats:
    """
    Load audio embeddings, fit KMeans, and write assignments to audio_clusters.

    n_clusters is automatically capped at the number of files (handles small
    test libraries gracefully).
    """
    try:
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for clustering. Run: pip install scikit-learn"
        ) from exc

    stats = ClusterStats()

    with conn_ctx(dsn) as conn:
        # ── 1. Load embeddings ─────────────────────────────────────────────
        with conn.cursor() as cur:
            rows = cur.execute(_SQL_LOAD_EMBEDDINGS).fetchall()

        if not rows:
            _LOG.warning("No audio embeddings found — clustering skipped.")
            return stats

        import json as _json  # noqa: PLC0415
        media_ids = [row["media_id"] for row in rows]
        # pgvector returns embeddings as Python lists via psycopg
        X = np.array([
            _json.loads(row["embedding"]) if isinstance(row["embedding"], str)
            else list(row["embedding"])
            for row in rows
        ], dtype=np.float32)

        n_actual = min(n_clusters, len(media_ids))
        stats.n_files    = len(media_ids)
        stats.n_clusters = n_actual

        _LOG.info(
            "Clustering %d audio files into %d clusters …", len(media_ids), n_actual
        )

        # ── 2. MiniBatchKMeans (faster than KMeans for large N) ────────────
        km = MiniBatchKMeans(
            n_clusters=n_actual,
            random_state=random_state,
            batch_size=1024,
            n_init=3,
        )
        labels = km.fit_predict(X)

        # ── 3. Write to audio_clusters ─────────────────────────────────────
        with conn.cursor() as cur:
            for media_id, label in zip(media_ids, labels):
                cluster_label = f"Cluster {int(label)}"
                cur.execute(_SQL_UPSERT_CLUSTER, (media_id, int(label), cluster_label))

        conn.commit()

    _LOG.info("Clustering complete. %s", stats.report())
    return stats


# ─────────────────────────────────────────────────────────────────────────── #
# Standalone CLI                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Cluster audio embeddings with KMeans and store in audio_clusters."
    )
    parser.add_argument("--n-clusters", type=int, default=50,
                        help="Number of KMeans clusters (default: 50)")
    args = parser.parse_args()

    from pg_settings import PgSettings  # noqa: PLC0415
    pg = PgSettings()
    stats = run_clustering(dsn=pg.pg_dsn, n_clusters=args.n_clusters)
    print(f"\n=== Clustering complete === {stats.report()}", flush=True)


if __name__ == "__main__":
    main()
