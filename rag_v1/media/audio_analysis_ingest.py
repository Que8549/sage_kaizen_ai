"""
rag_v1/media/audio_analysis_ingest.py

Phase 3 of media ingest: extract BPM, key, vocal detection, and explicit
flagging for all audio files in media_files, then merge results into
media_files.metadata (jsonb).

Resume-safe: files that already have a 'bpm' key in metadata are skipped
unless --force-reanalysis is passed.

Vocal detection uses the CLAP embed service (port 8040) zero-shot approach:
  - Embed two text probes once per run (no audio re-loading needed)
  - Compare stored audio_embeddings against the probes via cosine sim
  - Flag has_vocals=True when vocal probe wins by > threshold

BPM + Key use librosa on raw audio bytes:
  - ThreadPoolExecutor(workers) parallelises the CPU-bound librosa work

Explicit flagging reads existing lyrics rows; files without lyrics entries
get is_explicit=None (unknown) rather than False.

Run standalone:
    python -m rag_v1.media.audio_analysis_ingest [--force-reanalysis]

Integrated via media_ingest.py main() as Phase 3 (--no-analysis to skip).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psycopg
from psycopg.rows import DictRow

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_v1.db.pg import conn_ctx
from rag_v1.media.audio_analysis import (
    classify_vocals,
    clap_vocal_probes,
    extract_bpm_key,
    is_explicit_from_lyrics,
)

_LOG = logging.getLogger("sage_kaizen.audio_analysis_ingest")

# ─────────────────────────────────────────────────────────────────────────── #
# SQL                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

_SQL_PENDING = """
SELECT mf.media_id::text, mf.file_path
FROM   media_files mf
WHERE  mf.modality = 'audio'
  AND  (%(force)s OR NOT (mf.metadata ? 'bpm'))
ORDER BY mf.file_path;
"""

# Fetch stored audio embedding for vocal classification
_SQL_AUDIO_EMB = """
SELECT ae.embedding::text
FROM   audio_embeddings ae
WHERE  ae.media_id = %s::uuid
LIMIT 1;
"""

# Fetch concatenated lyrics for explicit scan
_SQL_LYRICS_TEXT = """
SELECT string_agg(l.chunk_text, ' ' ORDER BY l.chunk_id) AS lyrics
FROM   lyrics l
WHERE  l.media_id = %s::uuid;
"""

# Merge new keys into existing jsonb metadata (PostgreSQL ||  operator)
_SQL_UPDATE_META = """
UPDATE media_files
SET    metadata = metadata || %s::jsonb
WHERE  media_id = %s::uuid;
"""


# ─────────────────────────────────────────────────────────────────────────── #
# Data containers                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class _AnalysisJob:
    media_id:  str
    file_path: Path


@dataclass
class _AnalysisResult:
    media_id: str
    bpm:      float | None
    key:      str | None
    error:    bool = False


@dataclass
class AudioAnalysisStats:
    processed: int = 0
    skipped:   int = 0
    errors:    int = 0

    def report(self) -> str:
        return (
            f"processed={self.processed}  "
            f"skipped={self.skipped}  "
            f"errors={self.errors}"
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Worker                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def _analyse_worker(job: _AnalysisJob) -> _AnalysisResult:
    """CPU-bound: load audio and extract BPM + key with librosa."""
    if not job.file_path.exists():
        _LOG.debug("File not found on disk: %s", job.file_path)
        return _AnalysisResult(media_id=job.media_id, bpm=None, key=None, error=True)

    bpm, key = extract_bpm_key(job.file_path)
    return _AnalysisResult(media_id=job.media_id, bpm=bpm, key=key)


# ─────────────────────────────────────────────────────────────────────────── #
# Public entry point                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def run_audio_analysis_ingest(
    dsn: str,
    clap_host: str    = "127.0.0.1",
    clap_port: int    = 8040,
    workers: int      = 4,
    force: bool       = False,
    log_every: int    = 100,
) -> AudioAnalysisStats:
    """
    Main entry point.  Called from media_ingest.main() as Phase 3, and
    available standalone via __main__.

    Parameters
    ----------
    dsn        PostgreSQL DSN.
    clap_host  CLAP embed service host (for vocal probe embeds).
    clap_port  CLAP embed service port.
    workers    Thread pool size for librosa BPM/key extraction.
    force      Re-analyse files that already have 'bpm' in metadata.
    log_every  Print progress every N files.
    """
    stats = AudioAnalysisStats()

    # ── 1. Fetch CLAP vocal probe vectors (one HTTP call, reused for all files)
    _LOG.info("Fetching CLAP vocal probe embeddings …")
    vocal_probes = clap_vocal_probes(clap_host=clap_host, clap_port=clap_port)
    if vocal_probes is None:
        _LOG.warning(
            "CLAP service unreachable — has_vocals will not be set. "
            "Start the service and re-run to fill in vocal detection."
        )

    with conn_ctx(dsn) as conn:
        # ── 2. Load pending audio files
        with conn.cursor() as cur:
            rows = cur.execute(_SQL_PENDING, {"force": force}).fetchall()

        total = len(rows)
        _LOG.info("Audio analysis: %d files pending (force=%s).", total, force)
        print(f"Audio analysis: {total} files pending.", flush=True)

        if total == 0:
            return stats

        jobs = [
            _AnalysisJob(
                media_id  = row["media_id"],
                file_path = Path(row["file_path"]),
            )
            for row in rows
        ]

        # ── 3. BPM + key extraction (ThreadPoolExecutor, CPU-bound)
        bpm_key_map: dict[str, _AnalysisResult] = {}
        extracted = 0   # Phase 1 counter — separate from stats (which count Phase 2 DB writes)
        extract_errors = 0

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="audio_analysis",
        ) as pool:
            future_to_job = {pool.submit(_analyse_worker, job): job for job in jobs}
            for future in as_completed(future_to_job):
                result = future.result()
                bpm_key_map[result.media_id] = result
                extracted += 1
                if result.error:
                    extract_errors += 1
                if extracted % log_every == 0:
                    msg = (
                        f"[{extracted}/{total}] BPM/key extracted"
                        f" — decode_errors={extract_errors}"
                    )
                    _LOG.info(msg)
                    print(msg, flush=True)

        stats.errors = extract_errors  # carry Phase-1 errors into final report

        # ── 4. Vocal detection + explicit scan + DB write (single thread)
        with conn.cursor() as cur:
            for idx, job in enumerate(jobs):
                bk = bpm_key_map.get(job.media_id)
                if bk is None or bk.error:
                    continue

                # Vocal detection from stored audio embedding
                has_vocals: bool | None = None
                if vocal_probes is not None:
                    emb_row = cur.execute(
                        _SQL_AUDIO_EMB, (job.media_id,)
                    ).fetchone()
                    if emb_row and emb_row["embedding"]:
                        import json as _json  # noqa: PLC0415
                        audio_vec = _json.loads(emb_row["embedding"])
                        has_vocals = classify_vocals(
                            audio_vec, vocal_probes[0], vocal_probes[1]
                        )

                # Explicit flagging from lyrics text
                is_explicit: bool | None = None
                lyrics_row = cur.execute(
                    _SQL_LYRICS_TEXT, (job.media_id,)
                ).fetchone()
                if lyrics_row and lyrics_row["lyrics"]:
                    is_explicit = is_explicit_from_lyrics(lyrics_row["lyrics"])

                # Build metadata patch
                import json as _json  # noqa: PLC0415
                patch: dict = {}
                if bk.bpm is not None:
                    patch["bpm"] = round(bk.bpm, 1)
                if bk.key is not None:
                    patch["key"] = bk.key
                if has_vocals is not None:
                    patch["has_vocals"] = has_vocals
                if is_explicit is not None:
                    patch["is_explicit"] = is_explicit

                if patch:
                    cur.execute(_SQL_UPDATE_META, (_json.dumps(patch), job.media_id))
                    stats.processed += 1
                else:
                    stats.skipped += 1

                if (idx + 1) % log_every == 0:
                    conn.commit()
                    msg = f"[{idx+1}/{total}] metadata written — {stats.report()}"
                    _LOG.info(msg)
                    print(msg, flush=True)

        conn.commit()

    _LOG.info("Audio analysis ingest complete. %s", stats.report())
    return stats


# ─────────────────────────────────────────────────────────────────────────── #
# Standalone CLI                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Extract BPM, key, vocal detection, and explicit flags for all audio files."
    )
    parser.add_argument("--force-reanalysis", action="store_true",
                        help="Re-analyse files that already have BPM in metadata")
    parser.add_argument("--workers", type=int, default=4,
                        help="ThreadPool workers for librosa extraction (default: 4)")
    parser.add_argument("--clap-host", default="127.0.0.1")
    parser.add_argument("--clap-port", type=int, default=8040)
    args = parser.parse_args()

    from pg_settings import PgSettings  # noqa: PLC0415
    pg = PgSettings()

    stats = run_audio_analysis_ingest(
        dsn=pg.pg_dsn,
        clap_host=args.clap_host,
        clap_port=args.clap_port,
        workers=args.workers,
        force=args.force_reanalysis,
    )
    print(f"\n=== Audio analysis complete === {stats.report()}", flush=True)


if __name__ == "__main__":
    main()
