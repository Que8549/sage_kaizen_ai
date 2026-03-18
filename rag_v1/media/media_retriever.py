"""
rag_v1/media/media_retriever.py

Cross-modal retrieval using LanguageBind 768-dim embeddings stored in pgvector.

Query "house" (text) → returns matching images, audio clips, and video segments
from the media_embeddings table using a single cosine similarity search.

Prerequisites:
  1. Run media_schema.sql against your PostgreSQL database.
  2. Ingest media files using media_ingest.py (or your own pipeline) to
     populate the media_files and media_embeddings tables.
  3. Ensure the LanguageBind embed service (port 8040) is running.

Usage:
    retriever = MediaRetriever(pg_dsn="postgresql://...")
    results   = retriever.search("a red house by the sea", top_k=10)
    for r in results:
        print(r.modality, r.file_path, r.score)
"""
from __future__ import annotations

import atexit
import subprocess
import sys
import time
from dataclasses import dataclass, field

import psycopg
from psycopg.rows import dict_row, DictRow

from rag_v1.db.pg import get_conn
from rag_v1.media.languagebind_embed_client import MediaEmbedClient
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.media_retriever")


# ──────────────────────────────────────────────────────────────────────────── #
# Result dataclass                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class MediaResult:
    media_id:    str        # UUID of the source file
    file_path:   str        # absolute path on disk
    modality:    str        # "image", "audio", "video"
    score:       float      # cosine similarity (higher = more similar)
    frame_index: int | None = None  # for video: which frame matched
    time_s:      float | None = None
    metadata:    dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────── #
# SQL                                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

_SQL_SEARCH = """
SELECT
    mf.media_id::text,
    mf.file_path,
    mf.modality,
    me.frame_index,
    me.time_s,
    mf.metadata,
    1.0 - (me.embedding <=> %s::vector) AS score
FROM media_embeddings me
JOIN media_files mf ON mf.media_id = me.media_id
WHERE (me.embedding <=> %s::vector) < %s
ORDER BY me.embedding <=> %s::vector
LIMIT %s;
"""

# Filter to a specific modality (e.g. only images)
_SQL_SEARCH_MODALITY = """
SELECT
    mf.media_id::text,
    mf.file_path,
    mf.modality,
    me.frame_index,
    me.time_s,
    mf.metadata,
    1.0 - (me.embedding <=> %s::vector) AS score
FROM media_embeddings me
JOIN media_files mf ON mf.media_id = me.media_id
WHERE mf.modality = %s
  AND (me.embedding <=> %s::vector) < %s
ORDER BY me.embedding <=> %s::vector
LIMIT %s;
"""


# ──────────────────────────────────────────────────────────────────────────── #
# MediaRetriever                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

class MediaRetriever:
    """
    Cross-modal similarity search over the media_embeddings table.

    Embeds a text query with LanguageBind then performs HNSW cosine search
    across all ingested media files (images, audio, video) simultaneously.

    On first search(), auto-starts the embed service subprocess if it is not
    already running (same pattern as WikiRetriever).
    """

    def __init__(
        self,
        pg_dsn: str,
        host: str = "127.0.0.1",
        port: int = 8040,
        max_distance: float = 0.45,
    ) -> None:
        self._pg_dsn      = pg_dsn
        self._max_dist    = max_distance
        self._client      = MediaEmbedClient(host=host, port=port)
        self._embed_proc: subprocess.Popen | None = None
        self._atexit_reg: bool = False

    # ------------------------------------------------------------------ #
    # Service lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_service(self) -> bool:
        if self._client.ping(timeout_s=2.0):
            return True

        if self._embed_proc is not None and self._embed_proc.poll() is not None:
            _LOG.warning("Media embed service exited (rc=%s).", self._embed_proc.returncode)
            self._embed_proc = None

        _LOG.info("Media embed service not running — auto-starting …")
        self._embed_proc = subprocess.Popen(
            [sys.executable, "-m", "rag_v1.media.languagebind_embed_service.app"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not self._atexit_reg:
            atexit.register(self._shutdown)
            self._atexit_reg = True

        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            if self._client.ping(timeout_s=2.0):
                _LOG.info("Media embed service ready.")
                return True
            time.sleep(2.0)

        _LOG.warning("Media embed service did not start within 180 s.")
        return False

    def _shutdown(self) -> None:
        if self._embed_proc and self._embed_proc.poll() is None:
            try:
                self._embed_proc.terminate()
                self._embed_proc.wait(timeout=5)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int = 10,
        modality: str | None = None,
    ) -> list[MediaResult]:
        """
        Embed a text query and return the top-k most similar media assets.

        Parameters
        ----------
        query    : Text search query (e.g. "a house by the sea").
        top_k    : Maximum number of results to return.
        modality : Optional filter — "image", "audio", or "video".
                   If None, returns results across all modalities.

        Returns
        -------
        List of MediaResult sorted by descending cosine similarity.
        Returns [] on any failure (graceful degradation).
        """
        if not self._ensure_service():
            return []

        try:
            vecs = self._client.embed_text([query])
            if not vecs:
                return []
            qvec = vecs[0]
        except Exception:
            _LOG.exception("MediaRetriever: text embed failed for query %r", query)
            return []

        try:
            return self._query_db(qvec, top_k, modality)
        except Exception:
            _LOG.exception("MediaRetriever: DB query failed")
            return []

    # ------------------------------------------------------------------ #
    # DB helpers                                                           #
    # ------------------------------------------------------------------ #

    def _query_db(
        self,
        qvec: list[float],
        top_k: int,
        modality: str | None,
    ) -> list[MediaResult]:
        conn: psycopg.Connection[DictRow] = get_conn(self._pg_dsn)
        try:
            with conn.cursor(row_factory=dict_row) as cur:
                if modality:
                    rows = cur.execute(
                        _SQL_SEARCH_MODALITY,
                        (qvec, modality, qvec, self._max_dist, qvec, top_k),
                    ).fetchall()
                else:
                    rows = cur.execute(
                        _SQL_SEARCH,
                        (qvec, qvec, self._max_dist, qvec, top_k),
                    ).fetchall()
        finally:
            conn.close()

        return [
            MediaResult(
                media_id    = row["media_id"],
                file_path   = row["file_path"],
                modality    = row["modality"],
                score       = float(row["score"]),
                frame_index = row.get("frame_index"),
                time_s      = row.get("time_s"),
                metadata    = dict(row.get("metadata") or {}),
            )
            for row in rows
        ]
