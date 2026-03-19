"""
rag_v1/media/media_retriever.py

Dual-modal retrieval using CLIP + CLAP embeddings stored in pgvector.

  search_images(query) — text -> jina-clip-v2 (port 8031) -> image_embeddings (1024-dim)
  search_audio(query)  — text -> CLAP (port 8040)          -> audio_embeddings  (512-dim)

Prerequisites:
  1. Run media_schema.sql against your PostgreSQL database.
  2. Ingest media files using media_ingest.py to populate media_files,
     image_embeddings, and audio_embeddings.
  3. Ensure the wiki embed service (port 8031) and CLAP service (port 8040)
     are running before calling search_images / search_audio.

Usage:
    retriever = MediaRetriever(pg_dsn="postgresql://...")
    images = retriever.search_images("a red house by the sea", top_k=10)
    audio  = retriever.search_audio("waves crashing on shore", top_k=5)
    for r in images:
        print(r.file_path, r.score)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import psycopg
from psycopg.rows import dict_row, DictRow

from rag_v1.db.pg import get_conn
from rag_v1.media.media_embed_client import AudioEmbedClient, ImageEmbedClient
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.media_retriever")


# ──────────────────────────────────────────────────────────────────────────── #
# Result dataclass                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class MediaResult:
    media_id:  str          # UUID of the source file
    file_path: str          # absolute path on disk
    modality:  str          # "image" or "audio"
    score:     float        # cosine similarity (higher = more similar)
    metadata:  dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────── #
# SQL                                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

_SQL_SEARCH_IMAGES = """
SELECT
    mf.media_id::text,
    mf.file_path,
    mf.modality,
    mf.metadata,
    1.0 - (ie.embedding <=> %s::vector) AS score
FROM image_embeddings ie
JOIN media_files mf ON mf.media_id = ie.media_id
WHERE (ie.embedding <=> %s::vector) < %s
ORDER BY ie.embedding <=> %s::vector
LIMIT %s;
"""

_SQL_SEARCH_AUDIO = """
SELECT
    mf.media_id::text,
    mf.file_path,
    mf.modality,
    mf.metadata,
    1.0 - (ae.embedding <=> %s::vector) AS score
FROM audio_embeddings ae
JOIN media_files mf ON mf.media_id = ae.media_id
WHERE (ae.embedding <=> %s::vector) < %s
ORDER BY ae.embedding <=> %s::vector
LIMIT %s;
"""


# ──────────────────────────────────────────────────────────────────────────── #
# MediaRetriever                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

class MediaRetriever:
    """
    Dual-modal similarity search over image_embeddings and audio_embeddings.

    Each modality has its own embedding service:
      Images — jina-clip-v2 on port 8031 (wiki embed service, always running)
      Audio  — CLAP clap-htsat-unfused on port 8040 (CLAP embed service)

    Both services must be running before calling search_images / search_audio.
    Graceful degradation: returns [] if the relevant service is unreachable.
    """

    def __init__(
        self,
        pg_dsn: str,
        image_host: str = "127.0.0.1",
        image_port: int = 8031,
        audio_host: str = "127.0.0.1",
        audio_port: int = 8040,
        max_distance: float = 0.45,
    ) -> None:
        self._pg_dsn    = pg_dsn
        self._max_dist  = max_distance
        self._img_client = ImageEmbedClient(host=image_host, port=image_port)
        self._aud_client = AudioEmbedClient(host=audio_host, port=audio_port)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def search_images(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[MediaResult]:
        """
        Embed a text query with jina-clip-v2 and return the most similar images.

        Parameters
        ----------
        query  : Text search query (e.g. "a house by the sea").
        top_k  : Maximum number of results to return.

        Returns
        -------
        List of MediaResult (modality="image") sorted by descending cosine similarity.
        Returns [] on any failure (graceful degradation).
        """
        try:
            vecs = self._img_client.embed_text([query])
            if not vecs:
                return []
            qvec = vecs[0]
        except Exception:
            _LOG.exception("MediaRetriever: image text embed failed for query %r", query)
            return []

        try:
            return self._query_images(qvec, top_k)
        except Exception:
            _LOG.exception("MediaRetriever: image DB query failed")
            return []

    def search_audio(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[MediaResult]:
        """
        Embed a text query with CLAP and return the most similar audio files.

        Parameters
        ----------
        query  : Text search query (e.g. "waves crashing on shore").
        top_k  : Maximum number of results to return.

        Returns
        -------
        List of MediaResult (modality="audio") sorted by descending cosine similarity.
        Returns [] on any failure (graceful degradation).
        """
        try:
            vecs = self._aud_client.embed_text([query])
            if not vecs:
                return []
            qvec = vecs[0]
        except Exception:
            _LOG.exception("MediaRetriever: audio text embed failed for query %r", query)
            return []

        try:
            return self._query_audio(qvec, top_k)
        except Exception:
            _LOG.exception("MediaRetriever: audio DB query failed")
            return []

    # ------------------------------------------------------------------ #
    # DB helpers                                                           #
    # ------------------------------------------------------------------ #

    def _query_images(self, qvec: list[float], top_k: int) -> list[MediaResult]:
        conn: psycopg.Connection[DictRow] = get_conn(self._pg_dsn)
        try:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    _SQL_SEARCH_IMAGES,
                    (qvec, qvec, self._max_dist, qvec, top_k),
                ).fetchall()
        finally:
            conn.close()

        return [
            MediaResult(
                media_id  = row["media_id"],
                file_path = row["file_path"],
                modality  = row["modality"],
                score     = float(row["score"]),
                metadata  = dict(row.get("metadata") or {}),
            )
            for row in rows
        ]

    def _query_audio(self, qvec: list[float], top_k: int) -> list[MediaResult]:
        conn: psycopg.Connection[DictRow] = get_conn(self._pg_dsn)
        try:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    _SQL_SEARCH_AUDIO,
                    (qvec, qvec, self._max_dist, qvec, top_k),
                ).fetchall()
        finally:
            conn.close()

        return [
            MediaResult(
                media_id  = row["media_id"],
                file_path = row["file_path"],
                modality  = row["modality"],
                score     = float(row["score"]),
                metadata  = dict(row.get("metadata") or {}),
            )
            for row in rows
        ]
