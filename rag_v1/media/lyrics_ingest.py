"""
rag_v1/media/lyrics_ingest.py

Phase 2 of media ingest: fetch lyrics for all audio files in media_files,
chunk them, embed via BGE-M3 (port 8020), and store in the lyrics table.

Optimized for large libraries (~32k files):
  - Loads all pending DB rows upfront, then streams through a thread pool
  - ThreadPoolExecutor(max_workers=N) limits Genius API concurrency
  - Per-worker sleep(delay_s) rate-limits API calls across the pool
  - lyrics_fetch_log deduplicates across runs:
      status='ok'        -> skip (lyrics + embeddings already stored)
      status='not_found' -> skip (Genius confirmed nothing; don't re-hammer)
      status='error'     -> retry (network timeout, 5xx, etc.)
      absent             -> process
  - Batch BGE-M3 embed calls (lyrics_batch songs per batch) for throughput

Called by media_ingest.py main() as Phase 2 after audio/image ingest.
Can also be invoked standalone via --lyrics-only flag on media_ingest.py.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psycopg
from psycopg.rows import DictRow

from rag_v1.db.pg import conn_ctx
from rag_v1.embed.embed_client import EmbedClient
from rag_v1.ingest.ingest_utils import chunk_text, sha256_text

_LOG = logging.getLogger("sage_kaizen.lyrics_ingest")


# ──────────────────────────────────────────────────────────────────────────── #
# SQL                                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

# Load all audio rows that have no fetch log entry OR previously errored.
# 'ok' and 'not_found' rows are intentionally excluded — they are done.
_SQL_PENDING = """
SELECT mf.media_id::text, mf.file_path
FROM   media_files mf
LEFT JOIN lyrics_fetch_log lfl ON lfl.media_id = mf.media_id
WHERE  mf.modality = 'audio'
  AND  (lfl.media_id IS NULL OR lfl.status = 'error')
ORDER BY mf.ingested_at;
"""

_SQL_UPSERT_LOG = """
INSERT INTO lyrics_fetch_log (media_id, status, source, content_hash, attempted_at)
VALUES (%s::uuid, %s, %s, %s, now())
ON CONFLICT (media_id) DO UPDATE SET
    status       = EXCLUDED.status,
    source       = EXCLUDED.source,
    content_hash = EXCLUDED.content_hash,
    attempted_at = EXCLUDED.attempted_at;
"""

_SQL_DELETE_CHUNKS = "DELETE FROM lyrics WHERE media_id = %s::uuid;"

_SQL_INSERT_CHUNK = """
INSERT INTO lyrics (media_id, chunk_id, chunk_text, embedding)
VALUES (%s::uuid, %s, %s, %s::vector)
ON CONFLICT (media_id, chunk_id) DO UPDATE SET
    chunk_text = EXCLUDED.chunk_text,
    embedding  = EXCLUDED.embedding;
"""


# ──────────────────────────────────────────────────────────────────────────── #
# Data containers                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class _FetchJob:
    media_id:  str
    file_path: Path
    title:     str
    artist:    str


@dataclass
class _FetchResult:
    media_id:     str
    file_path:    Path
    status:       str            # 'ok' | 'not_found' | 'error'
    source:       Optional[str] = None
    lyrics_text:  Optional[str] = None
    content_hash: Optional[str] = None
    error:        Optional[BaseException] = None


@dataclass
class LyricsIngestStats:
    fetched:       int = 0
    not_found:     int = 0
    errors:        int = 0
    chunks_written: int = 0

    def report(self) -> str:
        return (
            f"fetched={self.fetched}  not_found={self.not_found}  "
            f"errors={self.errors}  chunks={self.chunks_written}"
        )


# ──────────────────────────────────────────────────────────────────────────── #
# ID3 tag extraction                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

def _read_id3_tags(path: Path) -> tuple[str, str]:
    """Return (title, artist) from MP3 ID3 tags. Falls back to (stem, '')."""
    try:
        import mutagen.id3 as id3  # type: ignore
        tags = id3.ID3(str(path))
        title  = str(tags.get("TIT2", "")).strip()
        artist = str(tags.get("TPE1", "")).strip()
        return title, artist
    except Exception:
        return path.stem, ""


# ──────────────────────────────────────────────────────────────────────────── #
# Worker (runs inside the thread pool)                                          #
# ──────────────────────────────────────────────────────────────────────────── #

def _fetch_worker(job: _FetchJob, delay_s: float) -> _FetchResult:
    """
    Fetch lyrics for one file. Always sleeps delay_s before returning
    to rate-limit Genius API calls across all pool workers.
    """
    from rag_v1.media.lyrics_fetch import get_lyrics  # noqa: PLC0415

    try:
        result = get_lyrics(job.file_path, job.title, job.artist)
    except Exception as exc:
        # Network/API error — return error so caller retries next run
        return _FetchResult(
            media_id=job.media_id,
            file_path=job.file_path,
            status="error",
            error=exc,
        )
    finally:
        # Rate-limit: sleep regardless of success/failure/exception
        time.sleep(delay_s)

    if result is None:
        return _FetchResult(
            media_id=job.media_id,
            file_path=job.file_path,
            status="not_found",
        )

    lyrics_text, source = result
    return _FetchResult(
        media_id=job.media_id,
        file_path=job.file_path,
        status="ok",
        source=source,
        lyrics_text=lyrics_text,
        content_hash=sha256_text(lyrics_text),
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Batch writer (always called from the main thread)                            #
# ──────────────────────────────────────────────────────────────────────────── #

def _write_batch(
    results: list[_FetchResult],
    conn: psycopg.Connection[DictRow],
    embed: EmbedClient,
    chunk_chars: int,
    chunk_overlap: int,
    stats: LyricsIngestStats,
) -> None:
    """Embed all ok lyrics in the batch and write lyrics + fetch_log rows."""
    ok = [r for r in results if r.status == "ok" and r.lyrics_text]

    if ok:
        # Compute chunks once per song, build a flat list for one embed call
        song_chunks: list[tuple[str, list[str]]] = []
        all_texts:   list[str] = []
        for r in ok:
            chunks = chunk_text(r.lyrics_text, chunk_chars, chunk_overlap)  # type: ignore[arg-type]
            song_chunks.append((r.media_id, chunks))
            all_texts.extend(chunks)

        try:
            vecs = embed.embed(all_texts)
        except Exception:
            _LOG.exception(
                "BGE-M3 embed failed for batch of %d chunks — marking %d songs as error.",
                len(all_texts), len(ok),
            )
            for r in ok:
                r.status = "error"
                r.error = RuntimeError("embed call failed")
            song_chunks = []
            vecs = []

        if vecs:
            with conn.cursor() as cur:
                vec_idx = 0
                for media_id, chunks in song_chunks:
                    cur.execute(_SQL_DELETE_CHUNKS, (media_id,))
                    for chunk_id, chunk_text_val in enumerate(chunks):
                        cur.execute(
                            _SQL_INSERT_CHUNK,
                            (media_id, chunk_id, chunk_text_val, vecs[vec_idx]),
                        )
                        vec_idx += 1
                    stats.chunks_written += len(chunks)

    # Write fetch_log for ALL results in this batch (ok / not_found / error)
    with conn.cursor() as cur:
        for r in results:
            cur.execute(
                _SQL_UPSERT_LOG,
                (r.media_id, r.status, r.source, r.content_hash),
            )
            if r.status == "ok":
                stats.fetched += 1
            elif r.status == "not_found":
                stats.not_found += 1
            else:
                stats.errors += 1

    conn.commit()


# ──────────────────────────────────────────────────────────────────────────── #
# Public entry point                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

def run_lyrics_ingest(
    dsn: str,
    embed_base_url: str,
    embed_model: str,
    workers:       int   = 8,
    lyrics_batch:  int   = 50,
    delay_s:       float = 0.3,
    chunk_chars:   int   = 1200,
    chunk_overlap: int   = 200,
    log_every:     int   = 100,
) -> LyricsIngestStats:
    """
    Fetch lyrics for all unprocessed audio rows in media_files,
    embed via BGE-M3, and store in the lyrics table.

    Skips rows already in lyrics_fetch_log with status='ok' or 'not_found'.
    Retries rows with status='error'.
    """
    stats = LyricsIngestStats()
    embed = EmbedClient(embed_base_url, embed_model)

    with conn_ctx(dsn) as conn:
        with conn.cursor() as cur:
            rows = cur.execute(_SQL_PENDING).fetchall()

        total = len(rows)
        _LOG.info("Lyrics ingest: %d audio files pending.", total)
        print(f"Lyrics ingest: {total} audio files pending.", flush=True)

        if total == 0:
            return stats

        # Read ID3 tags upfront (local disk, fast) before starting the pool
        jobs: list[_FetchJob] = []
        for row in rows:
            path = Path(row["file_path"])
            title, artist = _read_id3_tags(path)
            jobs.append(_FetchJob(
                media_id=row["media_id"],
                file_path=path,
                title=title,
                artist=artist,
            ))

        pending:   list[_FetchResult] = []
        processed: int = 0

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="lyrics",
        ) as pool:
            future_to_job = {
                pool.submit(_fetch_worker, job, delay_s): job
                for job in jobs
            }

            for future in as_completed(future_to_job):
                result = future.result()
                job    = future_to_job[future]
                pending.append(result)
                processed += 1

                if result.status == "error":
                    _LOG.warning(
                        "Lyrics error [%s — %s] %s: %s",
                        job.artist, job.title, job.file_path.name, result.error,
                    )
                elif result.status == "not_found":
                    _LOG.debug("Not found: %s — %s", job.artist, job.title)

                if processed % log_every == 0:
                    msg = f"[{processed}/{total}] {stats.report()}"
                    _LOG.info(msg)
                    print(msg, flush=True)

                if len(pending) >= lyrics_batch:
                    _write_batch(pending, conn, embed, chunk_chars, chunk_overlap, stats)
                    pending = []

        # Flush any remaining results
        if pending:
            _write_batch(pending, conn, embed, chunk_chars, chunk_overlap, stats)

    _LOG.info("Lyrics ingest complete. %s", stats.report())
    return stats
