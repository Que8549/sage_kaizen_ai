"""
rag_v1/media/media_ingest.py

Batch ingest job: walk a directory of media files, embed them with
LanguageBind (image + audio), and store 768-dim vectors in PostgreSQL.

Supported:  images (.png .jpg .jpeg .webp .gif .bmp .tiff)
            audio  (.wav .mp3 .flac .ogg .m4a .aac)
Deferred:   video  (.mp4 .mov .avi .mkv .webm) — logged and skipped

Resume-safe: files already in media_files (matched by path + SHA-256)
are skipped automatically.  Re-running is fully idempotent.

Prerequisites:
  1. Apply the DB schema once:
       psql -U sage -d sage_kaizen -f rag_v1/db/media_schema.sql
  2. Clone LanguageBind and install soundfile:
       git clone https://github.com/PKU-YuanGroup/LanguageBind F:/Projects/sage_kaizen_ai/languagebind_repo
       pip install soundfile
  3. Set PG_USER / PG_PASSWORD / PG_DB in .env (or as env vars)

Run:
    python -m rag_v1.media.media_ingest --root /path/to/media/directory

CLI flags:
    --root          Root directory to scan recursively (required)
    --device        Override MEDIA_EMBED_DEVICE (e.g. "cuda:0")
    --image-batch   Images per embed call  (default: from brains.yaml)
    --audio-batch   Audio clips per embed call  (default: from brains.yaml)
    --log-every     Print progress every N files scanned (default: 50)
    --no-service    Skip auto-starting the embed service (assumes port 8040 is up)
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import io
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import psycopg
import yaml
from psycopg.rows import dict_row, DictRow
from psycopg.types.json import Jsonb

from pg_settings import PgSettings
from rag_v1.db.pg import get_conn
from rag_v1.media.languagebind_embed_client import MediaEmbedClient

_LOG = logging.getLogger("sage_kaizen.media_ingest")

# ──────────────────────────────────────────────────────────────────────────── #
# File type sets                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"})
_AUDIO_EXTS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


# ──────────────────────────────────────────────────────────────────────────── #
# Config                                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

def _load_cfg() -> dict:
    """Return the media_embed section from brains.yaml."""
    root = Path(__file__).resolve().parents[2]   # rag_v1/media → rag_v1 → project root
    data = yaml.safe_load(
        (root / "config" / "brains" / "brains.yaml").read_text(encoding="utf-8")
    )
    return data["media_embed"]


# ──────────────────────────────────────────────────────────────────────────── #
# Stats                                                                          #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class IngestStats:
    written: int = 0
    skipped: int = 0
    errors: int = 0
    skipped_video: int = 0

    def report(self) -> str:
        return (
            f"written={self.written}  skipped={self.skipped}  "
            f"errors={self.errors}  video_deferred={self.skipped_video}"
        )


# ──────────────────────────────────────────────────────────────────────────── #
# File scanning                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

def _iter_media(root: Path) -> Iterator[tuple[Path, str]]:
    """Yield (path, modality) for every supported media file under root."""
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in _IMAGE_EXTS:
            yield p, "image"
        elif ext in _AUDIO_EXTS:
            yield p, "audio"
        elif ext in _VIDEO_EXTS:
            yield p, "video"


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


# ──────────────────────────────────────────────────────────────────────────── #
# Metadata extraction                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

def _image_meta(raw: bytes) -> tuple[int | None, int | None]:
    """Return (width, height) from raw image bytes, or (None, None) on error."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw))
        return img.width, img.height
    except Exception:
        return None, None


def _audio_duration(path: Path) -> float | None:
    """Return audio duration in seconds, or None on error."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.duration)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────── #
# SQL                                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

_SQL_INSERT_FILE = """
INSERT INTO media_files
    (file_path, modality, content_hash, file_size_b, width, height, duration_s, metadata)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (file_path, content_hash) DO NOTHING
RETURNING media_id::text
"""

_SQL_SELECT_FILE = """
SELECT media_id::text FROM media_files
WHERE file_path = %s AND content_hash = %s
"""

_SQL_CHECK_EMBED = """
SELECT 1 FROM media_embeddings me
JOIN media_files mf ON mf.media_id = me.media_id
WHERE mf.file_path = %s AND mf.content_hash = %s
LIMIT 1
"""

_SQL_INSERT_EMBED = """
INSERT INTO media_embeddings (media_id, embedding)
VALUES (%s, %s::vector)
"""


# ──────────────────────────────────────────────────────────────────────────── #
# DB write helpers                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _upsert_file(
    cur: psycopg.Cursor[DictRow],
    path: Path,
    modality: str,
    content_hash: bytes,
    file_size: int,
    width: int | None,
    height: int | None,
    duration_s: float | None,
) -> tuple[str | None, bool]:
    """
    Insert a media_files row.

    Returns (media_id, is_new):
        is_new=True  → row was just inserted; caller should write embedding.
        is_new=False → row already existed; caller should check embedding.
    """
    rows = cur.execute(
        _SQL_INSERT_FILE,
        (str(path), modality, content_hash, file_size,
         width, height, duration_s, Jsonb({})),
    ).fetchall()

    if rows:
        return str(rows[0]["media_id"]), True

    # ON CONFLICT — row exists; fetch the existing id
    row = cur.execute(_SQL_SELECT_FILE, (str(path), content_hash)).fetchone()
    if row:
        return str(row["media_id"]), False
    return None, False


# ──────────────────────────────────────────────────────────────────────────── #
# Embed service lifecycle                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

def _ensure_service(
    client: MediaEmbedClient,
    timeout_s: int,
    device: str | None,
) -> bool:
    if client.ping(timeout_s=3.0):
        return True

    _LOG.info("Media embed service not running — auto-starting …")
    env = os.environ.copy()
    if device:
        env["MEDIA_EMBED_DEVICE"] = device

    proc = subprocess.Popen(
        [sys.executable, "-m", "rag_v1.media.languagebind_embed_service.app"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(_terminate, proc)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if client.ping(timeout_s=2.0):
            _LOG.info("Media embed service ready.")
            return True
        time.sleep(2.0)

    _LOG.error("Embed service did not become healthy within %d s.", timeout_s)
    return False


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────── #
# Per-job data container                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class _FileJob:
    path: Path
    modality: str
    raw: bytes
    content_hash: bytes
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None


# ──────────────────────────────────────────────────────────────────────────── #
# Batch processing                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _process_batch(
    jobs: list[_FileJob],
    client: MediaEmbedClient,
    conn: psycopg.Connection[DictRow],
    stats: IngestStats,
) -> None:
    """Embed a same-modality batch and write results to the DB."""
    if not jobs:
        return

    modality = jobs[0].modality

    # ── Embed ─────────────────────────────────────────────────────────────── #
    try:
        if modality == "image":
            vecs = client.embed_image_bytes([j.raw for j in jobs])
        else:
            vecs = client.embed_audio_bytes([j.raw for j in jobs])
    except Exception:
        _LOG.exception("Embed call failed for %d %s file(s).", len(jobs), modality)
        stats.errors += len(jobs)
        return

    if len(vecs) != len(jobs):
        _LOG.error(
            "Embed returned %d vectors for %d inputs — dropping batch.",
            len(vecs), len(jobs),
        )
        stats.errors += len(jobs)
        return

    # ── Write to DB ───────────────────────────────────────────────────────── #
    with conn.cursor(row_factory=dict_row) as cur:
        for job, vec in zip(jobs, vecs):
            try:
                media_id, is_new = _upsert_file(
                    cur,
                    path=job.path,
                    modality=job.modality,
                    content_hash=job.content_hash,
                    file_size=len(job.raw),
                    width=job.width,
                    height=job.height,
                    duration_s=job.duration_s,
                )
                if media_id is None:
                    _LOG.error("Could not resolve media_id for %s", job.path)
                    stats.errors += 1
                    continue

                if not is_new:
                    # File row existed — check if embedding also exists
                    existing = cur.execute(
                        _SQL_CHECK_EMBED, (str(job.path), job.content_hash)
                    ).fetchone()
                    if existing:
                        stats.skipped += 1
                        continue
                    # File exists but embedding is missing — write it
                    _LOG.info("Backfilling missing embedding for %s", job.path)

                cur.execute(_SQL_INSERT_EMBED, (media_id, vec))
                stats.written += 1

            except Exception:
                _LOG.exception("DB write failed for %s", job.path)
                stats.errors += 1

    conn.commit()


# ──────────────────────────────────────────────────────────────────────────── #
# Main ingest loop                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def run_ingest(
    root: Path,
    dsn: str,
    host: str,
    port: int,
    image_batch: int,
    audio_batch: int,
    startup_timeout: int,
    device: str | None,
    log_every: int,
    auto_service: bool,
) -> IngestStats:
    client = MediaEmbedClient(host=host, port=port)
    stats = IngestStats()

    if auto_service and not _ensure_service(client, startup_timeout, device):
        _LOG.error("Cannot proceed without the embed service.")
        return stats

    conn = get_conn(dsn)
    img_buf: list[_FileJob] = []
    aud_buf: list[_FileJob] = []
    total = 0

    for path, modality in _iter_media(root):
        total += 1

        if modality == "video":
            _LOG.debug("Skipping video (deferred): %s", path)
            stats.skipped_video += 1
            if total % log_every == 0:
                _print_progress(total, stats)
            continue

        try:
            raw = path.read_bytes()
        except OSError:
            _LOG.exception("Cannot read %s — skipping.", path)
            stats.errors += 1
            continue

        job = _FileJob(
            path=path,
            modality=modality,
            raw=raw,
            content_hash=_sha256(raw),
        )
        if modality == "image":
            job.width, job.height = _image_meta(raw)
            img_buf.append(job)
            if len(img_buf) >= image_batch:
                _process_batch(img_buf, client, conn, stats)
                img_buf = []
        else:
            job.duration_s = _audio_duration(path)
            aud_buf.append(job)
            if len(aud_buf) >= audio_batch:
                _process_batch(aud_buf, client, conn, stats)
                aud_buf = []

        if total % log_every == 0:
            _print_progress(total, stats)

    # Flush partial batches
    _process_batch(img_buf, client, conn, stats)
    _process_batch(aud_buf, client, conn, stats)
    conn.close()

    return stats


def _print_progress(total: int, stats: IngestStats) -> None:
    msg = f"[{total:>6} scanned] {stats.report()}"
    print(msg, flush=True)
    _LOG.info(msg)


# ──────────────────────────────────────────────────────────────────────────── #
# CLI                                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = _load_cfg()
    svc = cfg["service"]

    parser = argparse.ArgumentParser(
        description="Ingest media files into media_files + media_embeddings tables."
    )
    parser.add_argument("--root",        required=True,  help="Root directory to scan recursively")
    parser.add_argument("--device",      default=None,   help="Override MEDIA_EMBED_DEVICE (e.g. cuda:0)")
    parser.add_argument("--image-batch", type=int,       default=int(svc.get("image_batch", 8)))
    parser.add_argument("--audio-batch", type=int,       default=int(svc.get("audio_batch", 4)))
    parser.add_argument("--log-every",   type=int,       default=50,  help="Print stats every N files")
    parser.add_argument("--no-service",  action="store_true",         help="Skip auto-starting embed service")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: --root is not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    pg = PgSettings()
    stats = run_ingest(
        root=root,
        dsn=pg.pg_dsn,
        host=svc.get("host", "127.0.0.1"),
        port=int(svc.get("port", 8040)),
        image_batch=args.image_batch,
        audio_batch=args.audio_batch,
        startup_timeout=int(cfg.get("startup_timeout_s", 180)),
        device=args.device,
        log_every=args.log_every,
        auto_service=not args.no_service,
    )

    print(f"\n=== Ingest complete === {stats.report()}", flush=True)
    _LOG.info("Ingest complete. %s", stats.report())


if __name__ == "__main__":
    main()
