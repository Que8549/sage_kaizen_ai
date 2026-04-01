"""
rag_v1/media/lyrics_fetch.py

Fetches lyrics for a given (file_path, title, artist) combination.

Priority:
  1. USLT ID3 tag embedded in the MP3 (zero network cost)
  2. Genius API via lyricsgenius

Returns (lyrics_text, source) or None if lyrics cannot be found.
Raises on network/API errors so the caller can record status='error'
and retry on the next run.

Thread safety:
  Uses threading.local() to create one Genius instance per worker thread.
  lyricsgenius wraps requests.Session which is NOT thread-safe; one instance
  per thread avoids contention across the ThreadPoolExecutor.

Config:
  GENIUS_API_TOKEN — set in .env (loaded by pydantic-settings at startup)
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

_LOG = logging.getLogger("sage_kaizen.lyrics_fetch")

# One Genius client per thread (requests.Session is not thread-safe)
_thread_local = threading.local()


def _get_genius():
    """Return a thread-local lyricsgenius.Genius instance."""
    if hasattr(_thread_local, "genius"):
        return _thread_local.genius

    try:
        import lyricsgenius  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "lyricsgenius is not installed. Run: pip install lyricsgenius"
        ) from exc

    token = os.environ.get("GENIUS_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "GENIUS_API_TOKEN is not set. Add it to your .env file."
        )

    _thread_local.genius = lyricsgenius.Genius(
        token,
        verbose=False,
        timeout=15,
        retries=2,
        remove_section_headers=True,
        skip_non_songs=True,
    )
    return _thread_local.genius


def _lyrics_from_tag(path: Path) -> Optional[str]:
    """Extract USLT lyrics from an MP3's ID3 tags. Returns text or None."""
    try:
        import mutagen.id3 as id3  # type: ignore
        tags = id3.ID3(str(path))
        for key in tags.keys():
            if key.startswith("USLT"):
                text = str(tags[key]).strip()
                if text:
                    return text
    except Exception:
        pass
    return None


def get_lyrics(
    path: Path,
    title: str,
    artist: str,
) -> Optional[tuple[str, str]]:
    """
    Return (lyrics_text, source) or None.

    source is one of: 'uslt_tag' | 'genius'

    Raises on Genius network/API errors so the caller can log status='error'
    and retry on the next ingest run. Returns None for clean not-found cases.
    """
    # 1. Embedded USLT tag — zero network cost, check first
    text = _lyrics_from_tag(path)
    if text:
        return text, "uslt_tag"

    # 2. Genius API — requires title at minimum
    if not title:
        return None

    genius = _get_genius()
    # May raise lyricsgenius.exceptions.* or requests.* on network/API errors.
    # Intentionally not caught here — caller records status='error' for retry.
    song = genius.search_song(title, artist=artist or "", get_full_info=False)
    if song and song.lyrics and song.lyrics.strip():
        return song.lyrics.strip(), "genius"

    return None
