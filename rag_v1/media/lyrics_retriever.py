"""
rag_v1/media/lyrics_retriever.py

Semantic lyric search via BGE-M3 embeddings stored in the lyrics table.

Usage
-----
    retriever = LyricsRetriever(pg_dsn="postgresql://...")
    results = retriever.search("California Love", top_k=5)
    for r in results:
        print(r.file_path, r.title, r.artist, r.score)

Requires:
  - lyrics_schema.sql applied to the database
  - lyrics_ingest.py run to populate the lyrics table
  - BGE-M3 embed service running on port 8020
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import psycopg
from psycopg.rows import dict_row, DictRow

from rag_v1.db.pg import get_conn
from rag_v1.embed.embed_client import EmbedClient

_LOG = logging.getLogger("sage_kaizen.lyrics_retriever")

_SQL_SEARCH = """
SELECT
    l.lyric_id,
    mf.media_id::text,
    mf.file_path,
    mf.metadata->>'title'  AS title,
    mf.metadata->>'artist' AS artist,
    l.chunk_id,
    l.chunk_text,
    1.0 - (l.embedding <=> %s::vector) AS score
FROM lyrics l
JOIN media_files mf ON mf.media_id = l.media_id
WHERE (l.embedding <=> %s::vector) < %s
ORDER BY l.embedding <=> %s::vector
LIMIT %s;
"""


@dataclass
class LyricsResult:
    media_id:     str
    file_path:    str
    title:        str
    artist:       str
    matched_text: str
    chunk_id:     int
    score:        float
    metadata:     dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        if self.title and self.artist:
            return f"{self.title} — {self.artist}"
        return self.title or self.artist or self.file_path


class LyricsRetriever:
    """
    Semantic lyric search over the lyrics table using BGE-M3 (port 8020).

    Gracefully returns [] if the embed service is unreachable or the
    lyrics table is empty.
    """

    def __init__(
        self,
        pg_dsn: str,
        embed_base_url: str = "http://127.0.0.1:8020/v1",
        embed_model: str = "bge-m3-embed",
        max_distance: float = 0.50,
    ) -> None:
        self._pg_dsn       = pg_dsn
        self._max_distance = max_distance
        self._embed        = EmbedClient(embed_base_url, embed_model)

    def search(self, query: str, top_k: int = 10) -> list[LyricsResult]:
        """
        Embed query with BGE-M3 and return the best-matching lyric chunks.

        Multiple chunks from the same song may be returned; callers that
        want one result per song should deduplicate on media_id.
        """
        if not query.strip():
            return []

        try:
            vecs = self._embed.embed([query])
            if not vecs:
                return []
            qvec = vecs[0]
        except Exception:
            _LOG.exception("LyricsRetriever: embed failed for query %r", query)
            return []

        try:
            return self._query(qvec, top_k)
        except Exception:
            _LOG.exception("LyricsRetriever: DB query failed")
            return []

    def search_deduplicated(self, query: str, top_k: int = 10) -> list[LyricsResult]:
        """
        Like search() but returns at most one result per song (best chunk).
        """
        raw = self.search(query, top_k=top_k * 3)
        seen: set[str] = set()
        out: list[LyricsResult] = []
        for r in raw:
            if r.media_id not in seen:
                seen.add(r.media_id)
                out.append(r)
            if len(out) >= top_k:
                break
        return out

    def _query(self, qvec: list[float], top_k: int) -> list[LyricsResult]:
        conn: psycopg.Connection[DictRow] = get_conn(self._pg_dsn)
        try:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    _SQL_SEARCH,
                    (qvec, qvec, self._max_distance, qvec, top_k),
                ).fetchall()
        finally:
            conn.close()

        return [
            LyricsResult(
                media_id     = row["media_id"],
                file_path    = row["file_path"],
                title        = row.get("title") or "",
                artist       = row.get("artist") or "",
                matched_text = row.get("chunk_text") or "",
                chunk_id     = int(row.get("chunk_id") or 0),
                score        = float(row["score"]),
            )
            for row in rows
        ]
