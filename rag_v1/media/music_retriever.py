"""
rag_v1/media/music_retriever.py

Unified music query hub.  All music-related retrieval passes through here.

Supported query types
---------------------
  search_by_mood(query)         — CLAP text → audio_embeddings cosine search
                                  "energetic dance music", "slow jazz ballad"

  search_by_lyrics(query)       — BGE-M3 text → lyrics cosine search
                                  "song about California", "miss you much"

  find_similar(file_path)       — audio embedding lookup → cosine search
                                  "find more like this song"

  build_playlist(vibe, count)   — CLAP mood search + optional BPM/attribute filter
                                  "playlist for a road trip, fast songs"

  filter_by_attributes(...)     — pure SQL filter on metadata jsonb
                                  has_vocals, min_bpm, max_bpm, key, is_explicit

  get_similar_cluster(file_path)— return songs in the same KMeans cluster
                                  "songs that sound similar to each other"

  detect_intent(query)          — parse query text → MusicIntent dataclass
                                  used by context_injector to dispatch

All methods return list[MusicResult] and degrade gracefully to [] on failure.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psycopg
from psycopg.rows import dict_row, DictRow

from rag_v1.db.pg import get_conn
from rag_v1.embed.embed_client import EmbedClient
from rag_v1.media.media_embed_client import AudioEmbedClient
from rag_v1.media.lyrics_retriever import LyricsResult, LyricsRetriever

_LOG = logging.getLogger("sage_kaizen.music_retriever")


# ─────────────────────────────────────────────────────────────────────────── #
# Result                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class MusicResult:
    media_id:   str
    file_path:  str
    title:      str
    artist:     str
    score:      float
    bpm:        float | None = None
    key:        str   | None = None
    has_vocals: bool  | None = None
    is_explicit:bool  | None = None
    cluster_id: int   | None = None
    matched_lyric: str | None = None   # set by search_by_lyrics

    @property
    def display_name(self) -> str:
        if self.title and self.artist:
            return f"{self.title} — {self.artist}"
        return self.title or self.artist or Path(self.file_path).stem

    def format_line(self) -> str:
        parts = [self.display_name]
        if self.bpm is not None:
            parts.append(f"BPM={self.bpm:.0f}")
        if self.key:
            parts.append(f"key={self.key}")
        if self.has_vocals is not None:
            parts.append("vocals" if self.has_vocals else "instrumental")
        if self.is_explicit:
            parts.append("explicit")
        parts.append(f"score={self.score:.2f}")
        return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────── #
# Intent detection                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class MusicIntent:
    """
    Parsed music query intent.

    intent values
    -------------
    mood        — "energetic dance music", "sad slow songs"
    lyrics      — "song that says California", "tracks about missing someone"
    similar     — "find more like Dear Mama", "songs similar to California Love"
    playlist    — "make a playlist for a road trip"
    attribute   — "instrumental songs", "songs with BPM over 100", "songs in C major"
    cluster     — "songs that sound similar to each other", "group my music"
    explicit    — "show explicit songs" / "show clean songs"
    """
    intent: str
    query:  str
    params: dict = field(default_factory=dict)


# Hint patterns for intent detection (checked in order)
_LYRIC_RE = re.compile(
    r"\b(song\s+that\s+says|lyric|lyrics\s+about|contains\s+the\s+(word|phrase)|"
    r"sings?\s+about|song\s+where\s+someone\s+sings?|find\s+the\s+song\s+with|"
    r"tracks?\s+with\s+the\s+word|has\s+the\s+(word|line))\b",
    re.IGNORECASE,
)
_SIMILAR_RE = re.compile(
    r"\b(more\s+like(\s+this)?|similar\s+to|sounds?\s+like|find\s+(more|songs?)\s+like|"
    r"songs?\s+similar|like\s+this\s+(song|track)|just\s+like)\b",
    re.IGNORECASE,
)
_CLUSTER_RE = re.compile(
    r"\b(sound\s+similar\s+to\s+each\s+other|group\s+(my\s+)?music|"
    r"cluster|songs?\s+that\s+sound\s+alike|music\s+groups?|"
    r"era\s+cluster|artist\s+cluster)\b",
    re.IGNORECASE,
)
_PLAYLIST_RE = re.compile(
    r"\b(make|create|generate|build)\s+a?\s*playlist\b|playlist\s+(for|of|with)\b",
    re.IGNORECASE,
)
_ATTRIBUTE_RE = re.compile(
    r"\b(instrumental|no\s+vocals?|with\s+vocals?|bpm|beats?\s+per\s+minute|"
    r"tempo|key\s+of|in\s+the\s+key|explicit|clean\s+(version|songs?)|"
    r"fast\s+(songs?|tracks?)|slow\s+(songs?|tracks?)|upbeat|"
    r"over\s+\d+\s+bpm|under\s+\d+\s+bpm|greater\s+than|less\s+than)\b",
    re.IGNORECASE,
)
_MOOD_HINTS = (
    "play something", "play me", "find songs", "find a song", "find me",
    "list songs", "list tracks", "music for", "songs about", "tracks about",
    "something to dance", "something chill", "something upbeat", "something slow",
    "what songs", "which songs", "any songs",
)

# BPM filter extraction
_BPM_GT_RE = re.compile(r"(?:over|above|greater\s+than|>|bpm\s*>?\s*)(\d+)", re.IGNORECASE)
_BPM_LT_RE = re.compile(r"(?:under|below|less\s+than|<|bpm\s*<?\s*)(\d+)", re.IGNORECASE)


def detect_intent(query: str) -> MusicIntent | None:
    """
    Parse query text and return a MusicIntent, or None if not music-related.
    """
    txt = query.strip()
    low = txt.lower()

    if _CLUSTER_RE.search(txt):
        return MusicIntent(intent="cluster", query=txt)

    if _SIMILAR_RE.search(txt):
        return MusicIntent(intent="similar", query=txt)

    if _LYRIC_RE.search(txt):
        return MusicIntent(intent="lyrics", query=txt)

    if _PLAYLIST_RE.search(txt):
        params: dict = {}
        gt = _BPM_GT_RE.search(txt)
        lt = _BPM_LT_RE.search(txt)
        if gt:
            params["min_bpm"] = int(gt.group(1))
        if lt:
            params["max_bpm"] = int(lt.group(1))
        return MusicIntent(intent="playlist", query=txt, params=params)

    if _ATTRIBUTE_RE.search(txt):
        params = {}
        if re.search(r"\binstrumental\b|\bno\s+vocals?\b", low):
            params["has_vocals"] = False
        if re.search(r"\bwith\s+vocals?\b|\bhas\s+vocals?\b|\bwith\s+singing\b", low):
            params["has_vocals"] = True
        if re.search(r"\bexplicit\b", low):
            params["is_explicit"] = True
        if re.search(r"\bclean\b", low):
            params["is_explicit"] = False
        gt = _BPM_GT_RE.search(txt)
        lt = _BPM_LT_RE.search(txt)
        if gt:
            params["min_bpm"] = int(gt.group(1))
        if lt:
            params["max_bpm"] = int(lt.group(1))
        key_m = re.search(
            r"\b(?:key\s+of|in\s+the\s+key)\s+([A-G][b#]?\s+(?:major|minor))\b",
            txt, re.IGNORECASE
        )
        if key_m:
            params["key"] = key_m.group(1).strip()
        return MusicIntent(intent="attribute", query=txt, params=params)

    # Mood: check broad music hints as last resort
    if any(h in low for h in _MOOD_HINTS):
        return MusicIntent(intent="mood", query=txt)

    return None


# ─────────────────────────────────────────────────────────────────────────── #
# SQL                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

_SQL_MOOD_SEARCH = """
SELECT
    mf.media_id::text,
    mf.file_path,
    mf.metadata->>'title'               AS title,
    mf.metadata->>'artist'              AS artist,
    (mf.metadata->>'bpm')::float        AS bpm,
    mf.metadata->>'key'                 AS key,
    (mf.metadata->>'has_vocals')::bool  AS has_vocals,
    (mf.metadata->>'is_explicit')::bool AS is_explicit,
    ac.cluster_id,
    1.0 - (ae.embedding <=> %s::vector) AS score
FROM audio_embeddings ae
JOIN media_files mf ON mf.media_id = ae.media_id
LEFT JOIN audio_clusters ac ON ac.media_id = ae.media_id
WHERE (ae.embedding <=> %s::vector) < %s
  AND (%s IS NULL OR (mf.metadata->>'bpm')::float >= %s)
  AND (%s IS NULL OR (mf.metadata->>'bpm')::float <= %s)
  AND (%s IS NULL OR (mf.metadata->>'has_vocals')::bool = %s)
  AND (%s IS NULL OR (mf.metadata->>'is_explicit')::bool = %s)
ORDER BY ae.embedding <=> %s::vector
LIMIT %s;
"""

_SQL_SIMILAR_BY_PATH = """
SELECT ae.embedding, mf.media_id::text
FROM audio_embeddings ae
JOIN media_files mf ON mf.media_id = ae.media_id
WHERE mf.file_path ILIKE %s
LIMIT 1;
"""

_SQL_SIMILAR_BY_TITLE = """
SELECT ae.embedding, mf.media_id::text
FROM audio_embeddings ae
JOIN media_files mf ON mf.media_id = ae.media_id
WHERE mf.metadata->>'title' ILIKE %s
LIMIT 1;
"""

_SQL_SIMILAR_SEARCH = """
SELECT
    mf.media_id::text,
    mf.file_path,
    mf.metadata->>'title'               AS title,
    mf.metadata->>'artist'              AS artist,
    (mf.metadata->>'bpm')::float        AS bpm,
    mf.metadata->>'key'                 AS key,
    (mf.metadata->>'has_vocals')::bool  AS has_vocals,
    (mf.metadata->>'is_explicit')::bool AS is_explicit,
    ac.cluster_id,
    1.0 - (ae.embedding <=> %s::vector) AS score
FROM audio_embeddings ae
JOIN media_files mf ON mf.media_id = ae.media_id
LEFT JOIN audio_clusters ac ON ac.media_id = ae.media_id
WHERE mf.media_id != %s::uuid
  AND (ae.embedding <=> %s::vector) < 0.45
ORDER BY ae.embedding <=> %s::vector
LIMIT %s;
"""

_SQL_ATTRIBUTE_FILTER = """
SELECT
    mf.media_id::text,
    mf.file_path,
    mf.metadata->>'title'               AS title,
    mf.metadata->>'artist'              AS artist,
    (mf.metadata->>'bpm')::float        AS bpm,
    mf.metadata->>'key'                 AS key,
    (mf.metadata->>'has_vocals')::bool  AS has_vocals,
    (mf.metadata->>'is_explicit')::bool AS is_explicit,
    ac.cluster_id
FROM media_files mf
LEFT JOIN audio_clusters ac ON ac.media_id = mf.media_id
WHERE mf.modality = 'audio'
  AND (%s IS NULL OR (mf.metadata->>'bpm')::float >= %s)
  AND (%s IS NULL OR (mf.metadata->>'bpm')::float <= %s)
  AND (%s IS NULL OR (mf.metadata->>'has_vocals')::bool = %s)
  AND (%s IS NULL OR (mf.metadata->>'is_explicit')::bool = %s)
  AND (%s IS NULL OR LOWER(mf.metadata->>'key') = LOWER(%s))
ORDER BY (mf.metadata->>'bpm')::float DESC NULLS LAST
LIMIT %s;
"""

_SQL_CLUSTER_MEMBERS = """
SELECT
    mf.media_id::text,
    mf.file_path,
    mf.metadata->>'title'               AS title,
    mf.metadata->>'artist'              AS artist,
    (mf.metadata->>'bpm')::float        AS bpm,
    mf.metadata->>'key'                 AS key,
    (mf.metadata->>'has_vocals')::bool  AS has_vocals,
    (mf.metadata->>'is_explicit')::bool AS is_explicit,
    ac.cluster_id
FROM audio_clusters ac
JOIN media_files mf ON mf.media_id = ac.media_id
WHERE ac.cluster_id = (
    SELECT cluster_id FROM audio_clusters WHERE media_id = %s::uuid LIMIT 1
)
  AND ac.media_id != %s::uuid
ORDER BY mf.file_path
LIMIT %s;
"""


# ─────────────────────────────────────────────────────────────────────────── #
# MusicRetriever                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class MusicRetriever:
    """
    Unified music query hub.

    Instantiate once (lazy singleton in context_injector) and reuse.
    All methods return list[MusicResult] and fail gracefully.
    """

    def __init__(
        self,
        pg_dsn: str,
        clap_host: str = "127.0.0.1",
        clap_port: int = 8040,
        embed_base_url: str = "http://127.0.0.1:8020/v1",
        embed_model: str = "bge-m3-embed",
        max_distance: float = 0.45,
    ) -> None:
        self._pg_dsn      = pg_dsn
        self._max_dist    = max_distance
        self._clap        = AudioEmbedClient(host=clap_host, port=clap_port)
        self._lyrics_ret  = LyricsRetriever(
            pg_dsn=pg_dsn,
            embed_base_url=embed_base_url,
            embed_model=embed_model,
        )

    # ------------------------------------------------------------------ #
    # Mood search (CLAP text → audio_embeddings)                          #
    # ------------------------------------------------------------------ #

    def search_by_mood(
        self,
        query: str,
        top_k: int = 10,
        min_bpm: float | None = None,
        max_bpm: float | None = None,
        has_vocals: bool | None = None,
        is_explicit: bool | None = None,
    ) -> list[MusicResult]:
        try:
            vecs = self._clap.embed_text([query])
            if not vecs:
                return []
            qvec = vecs[0]
        except Exception:
            _LOG.exception("search_by_mood: CLAP embed failed for %r", query)
            return []

        try:
            conn = get_conn(self._pg_dsn)
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(_SQL_MOOD_SEARCH, (
                    qvec, qvec, self._max_dist,
                    min_bpm, min_bpm,
                    max_bpm, max_bpm,
                    has_vocals, has_vocals,
                    is_explicit, is_explicit,
                    qvec, top_k,
                )).fetchall()
            conn.close()
        except Exception:
            _LOG.exception("search_by_mood: DB query failed")
            return []

        return [_row_to_result(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Lyric search (BGE-M3 text → lyrics table)                          #
    # ------------------------------------------------------------------ #

    def search_by_lyrics(self, query: str, top_k: int = 10) -> list[MusicResult]:
        try:
            results = self._lyrics_ret.search_deduplicated(query, top_k=top_k)
        except Exception:
            _LOG.exception("search_by_lyrics failed for %r", query)
            return []

        return [
            MusicResult(
                media_id      = r.media_id,
                file_path     = r.file_path,
                title         = r.title,
                artist        = r.artist,
                score         = r.score,
                matched_lyric = r.matched_text[:200] if r.matched_text else None,
            )
            for r in results
        ]

    # ------------------------------------------------------------------ #
    # Find similar (audio embedding cosine search)                        #
    # ------------------------------------------------------------------ #

    def find_similar(self, query: str, top_k: int = 10) -> list[MusicResult]:
        """
        query can be a file path, a song title, or a free-form description.
        If it looks like an existing path or title, look up its stored
        embedding; otherwise fall back to CLAP text embed.
        """
        conn = get_conn(self._pg_dsn)
        try:
            # 1. Try exact file path match
            source_vec = None
            source_id  = None
            with conn.cursor(row_factory=dict_row) as cur:
                row = cur.execute(
                    _SQL_SIMILAR_BY_PATH, (f"%{query.strip()}%",)
                ).fetchone()
                if row:
                    source_vec = row["embedding"]
                    source_id  = row["media_id"]

            # 2. Try title match
            if source_vec is None:
                with conn.cursor(row_factory=dict_row) as cur:
                    row = cur.execute(
                        _SQL_SIMILAR_BY_TITLE, (f"%{query.strip()}%",)
                    ).fetchone()
                    if row:
                        source_vec = row["embedding"]
                        source_id  = row["media_id"]

            # 3. Fall back to CLAP text embed
            if source_vec is None:
                try:
                    vecs = self._clap.embed_text([query])
                    source_vec = vecs[0] if vecs else None
                except Exception:
                    pass

            if source_vec is None:
                return []

            # Convert pgvector return type to list[float]
            import json as _json  # noqa: PLC0415
            if isinstance(source_vec, str):
                source_vec = _json.loads(source_vec)

            # 4. Cosine search
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(_SQL_SIMILAR_SEARCH, (
                    source_vec,
                    source_id or "00000000-0000-0000-0000-000000000000",
                    source_vec, source_vec, top_k,
                )).fetchall()

        except Exception:
            _LOG.exception("find_similar: DB query failed")
            return []
        finally:
            conn.close()

        return [_row_to_result(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Playlist (mood + BPM/attribute filter)                              #
    # ------------------------------------------------------------------ #

    def build_playlist(
        self,
        vibe: str,
        count: int = 20,
        min_bpm: float | None = None,
        max_bpm: float | None = None,
        has_vocals: bool | None = None,
        is_explicit: bool | None = None,
    ) -> list[MusicResult]:
        return self.search_by_mood(
            query=vibe,
            top_k=count,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            has_vocals=has_vocals,
            is_explicit=is_explicit,
        )

    # ------------------------------------------------------------------ #
    # Attribute filter (pure SQL, no embedding)                           #
    # ------------------------------------------------------------------ #

    def filter_by_attributes(
        self,
        has_vocals: bool | None = None,
        min_bpm: float | None = None,
        max_bpm: float | None = None,
        key: str | None = None,
        is_explicit: bool | None = None,
        top_k: int = 20,
    ) -> list[MusicResult]:
        try:
            conn = get_conn(self._pg_dsn)
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(_SQL_ATTRIBUTE_FILTER, (
                    min_bpm, min_bpm,
                    max_bpm, max_bpm,
                    has_vocals, has_vocals,
                    is_explicit, is_explicit,
                    key, key,
                    top_k,
                )).fetchall()
            conn.close()
        except Exception:
            _LOG.exception("filter_by_attributes: DB query failed")
            return []

        return [_row_to_result(row, score=1.0) for row in rows]

    # ------------------------------------------------------------------ #
    # Cluster-based similarity ("songs that sound like each other")       #
    # ------------------------------------------------------------------ #

    def get_similar_cluster(self, query: str, top_k: int = 20) -> list[MusicResult]:
        """
        Return songs in the same acoustic cluster as the matched file/title.
        Falls back to [] if no cluster data exists.
        """
        conn = get_conn(self._pg_dsn)
        try:
            # Resolve media_id from path or title
            source_id: str | None = None
            with conn.cursor(row_factory=dict_row) as cur:
                row = cur.execute(
                    _SQL_SIMILAR_BY_PATH, (f"%{query.strip()}%",)
                ).fetchone()
                if row:
                    source_id = row["media_id"]
            if source_id is None:
                with conn.cursor(row_factory=dict_row) as cur:
                    row = cur.execute(
                        _SQL_SIMILAR_BY_TITLE, (f"%{query.strip()}%",)
                    ).fetchone()
                    if row:
                        source_id = row["media_id"]

            if source_id is None:
                return []

            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    _SQL_CLUSTER_MEMBERS, (source_id, source_id, top_k)
                ).fetchall()

        except Exception:
            _LOG.exception("get_similar_cluster: DB query failed")
            return []
        finally:
            conn.close()

        return [_row_to_result(row, score=1.0) for row in rows]

    # ------------------------------------------------------------------ #
    # Dispatch from MusicIntent                                           #
    # ------------------------------------------------------------------ #

    def dispatch(self, intent: MusicIntent, top_k: int = 10) -> list[MusicResult]:
        """
        Route a parsed MusicIntent to the correct retrieval method.
        This is the single entry point called from context_injector.
        """
        p = intent.params

        if intent.intent == "mood":
            return self.search_by_mood(
                intent.query, top_k=top_k,
                min_bpm=p.get("min_bpm"), max_bpm=p.get("max_bpm"),
                has_vocals=p.get("has_vocals"), is_explicit=p.get("is_explicit"),
            )

        if intent.intent == "lyrics":
            return self.search_by_lyrics(intent.query, top_k=top_k)

        if intent.intent == "similar":
            return self.find_similar(intent.query, top_k=top_k)

        if intent.intent == "playlist":
            return self.build_playlist(
                intent.query, count=max(top_k, 20),
                min_bpm=p.get("min_bpm"), max_bpm=p.get("max_bpm"),
                has_vocals=p.get("has_vocals"), is_explicit=p.get("is_explicit"),
            )

        if intent.intent == "attribute":
            return self.filter_by_attributes(
                has_vocals=p.get("has_vocals"),
                min_bpm=p.get("min_bpm"),
                max_bpm=p.get("max_bpm"),
                key=p.get("key"),
                is_explicit=p.get("is_explicit"),
                top_k=top_k,
            )

        if intent.intent == "cluster":
            return self.get_similar_cluster(intent.query, top_k=top_k)

        return []


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def _row_to_result(row: dict, score: float | None = None) -> MusicResult:
    return MusicResult(
        media_id    = str(row.get("media_id") or ""),
        file_path   = str(row.get("file_path") or ""),
        title       = str(row.get("title") or ""),
        artist      = str(row.get("artist") or ""),
        score       = float(row.get("score") or score or 0.0),
        bpm         = float(row["bpm"]) if row.get("bpm") is not None else None,
        key         = row.get("key"),
        has_vocals  = row.get("has_vocals"),
        is_explicit = row.get("is_explicit"),
        cluster_id  = int(row["cluster_id"]) if row.get("cluster_id") is not None else None,
    )


def format_music_context(
    intent: MusicIntent,
    results: list[MusicResult],
    top_k: int = 10,
) -> str:
    """
    Format music results as a <music_context> block for LLM injection.
    """
    if not results:
        return ""

    lines = [
        f'<music_context intent="{intent.intent}" query="{intent.query[:120]}">',
        f"Found {len(results)} track(s):\n",
    ]
    for i, r in enumerate(results[:top_k], 1):
        line = f"{i}. {r.format_line()}"
        if r.matched_lyric:
            line += f'\n   Lyric: "…{r.matched_lyric.strip()[:120]}…"'
        lines.append(line)
    lines.append("</music_context>")
    return "\n".join(lines)
