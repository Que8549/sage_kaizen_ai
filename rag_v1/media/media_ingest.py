"""
rag_v1/media/media_ingest.py

Batch ingest job: walk a directory of media files, embed them, and store
vectors in PostgreSQL.

  Images  → jina-clip-v2 (port 8031, wiki embed service) → image_embeddings (1024-dim)
  Audio   → CLAP clap-htsat-unfused (port 8040)          → audio_embeddings  (512-dim)

Supported:  images (.png .jpg .jpeg .webp .gif .bmp .tiff .heic)
            audio  (.wav .mp3 .flac .ogg .m4a .aac)
Deferred:   video  (.mp4 .mov .avi .mkv .webm .3gp) — logged and skipped

Resume-safe: files already in media_files (matched by path + SHA-256) and
already embedded are skipped automatically.  Re-running is fully idempotent.

Prerequisites:
  1. Apply the DB schema once:
       psql -U sage -d sage_kaizen -f rag_v1/db/media_schema.sql
  2. Ensure the wiki embed service is running on port 8031 (for images).
  3. Ensure the CLAP embed service is running on port 8040 (for audio).
       python -m rag_v1.media.clap_embed_service.app
  4. Set PG_USER / PG_PASSWORD / PG_DB in .env (or as env vars).

Run:
    python -m rag_v1.media.media_ingest --root /path/to/media/directory
    
CLI flags:
    --root          Root directory to scan recursively (required)
    --image-batch   Images per embed call  (default: from brains.yaml)
    --audio-batch   Audio clips per embed call  (default: from brains.yaml)
    --log-every     Print progress every N files scanned (default: 50)
    --no-service    Skip auto-starting the CLAP embed service

Usage: 
    python.exe -m rag_v1.media.media_ingest [-h] [--root ROOT]
                                               [--audio-device AUDIO_DEVICE]
                                               [--image-batch IMAGE_BATCH]
                                               [--audio-batch AUDIO_BATCH]
                                               [--log-every LOG_EVERY]
                                               [--no-service] [--no-lyrics]
                                               [--lyrics-only]
                                               [--lyrics-workers LYRICS_WORKERS]
                                               [--lyrics-batch LYRICS_BATCH]
                                               [--lyrics-delay LYRICS_DELAY]
                                               [--no-analysis]
                                               [--analysis-workers ANALYSIS_WORKERS]
                                               [--cluster]
                                               [--n-clusters N_CLUSTERS]

Ingest media files into media_files + image_embeddings / audio_embeddings
tables. Images use the wiki embed service (port 8031). Audio uses the CLAP
embed service (port 8040).

options:
  -h, --help            show this help message and exit
  --root ROOT           Root directory to scan recursively (required unless
                        --lyrics-only)
  --audio-device AUDIO_DEVICE
                        Override CLAP_DEVICE (e.g. cuda:1)
  --image-batch IMAGE_BATCH
  --audio-batch AUDIO_BATCH
  --log-every LOG_EVERY
                        Print stats every N files
  --no-service          Skip auto-starting CLAP/wiki embed services
  --no-lyrics           Skip lyrics ingest phase
  --lyrics-only         Run only lyrics phase; skip audio/image ingest
  --lyrics-workers LYRICS_WORKERS
                        Genius API concurrent workers (default: 8)
  --lyrics-batch LYRICS_BATCH
                        Songs between DB commits in lyrics phase (default: 50)
  --lyrics-delay LYRICS_DELAY
                        Seconds to sleep between Genius calls per worker
                        (default: 0.3)
  --no-analysis         Skip audio analysis phase (BPM, key, vocal detection,
                        explicit flagging)
  --analysis-workers ANALYSIS_WORKERS
                        Thread pool size for librosa BPM/key extraction
                        (default: 4)
  --cluster             Run KMeans clustering on audio embeddings after
                        analysis
  --n-clusters N_CLUSTERS
                        Number of KMeans clusters (default: 50)    

"""
from __future__ import annotations

import argparse
import atexit
import io
import logging
from logging.handlers import RotatingFileHandler
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import psycopg
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb

from rag_v1.db.pg import conn_ctx
from rag_v1.ingest.ingest_utils import sha256_bytes
from rag_v1.media.media_embed_client import AudioEmbedClient, ImageEmbedClient
from rag_v1.media.media_embed_config import load_media_embed_config

_LOG = logging.getLogger("sage_kaizen.media_ingest")

# Register pillow-heif so PIL.Image.open() can decode .heic files
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    _LOG.warning("pillow-heif not installed — .heic files will fail metadata extraction")

# ──────────────────────────────────────────────────────────────────────────── #
# File type sets                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".heic"})
_AUDIO_EXTS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"})


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
        elif ext:
            _LOG.debug("Skipping unrecognized extension %s: %s", ext, p)


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
# SQL                                                                          #
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

_SQL_CHECK_IMAGE_EMBED = """
SELECT 1 FROM image_embeddings ie
JOIN media_files mf ON mf.media_id = ie.media_id
WHERE mf.file_path = %s AND mf.content_hash = %s
LIMIT 1
"""

_SQL_CHECK_AUDIO_EMBED = """
SELECT 1 FROM audio_embeddings ae
JOIN media_files mf ON mf.media_id = ae.media_id
WHERE mf.file_path = %s AND mf.content_hash = %s
LIMIT 1
"""

_SQL_INSERT_IMAGE_EMBED = """
INSERT INTO image_embeddings (media_id, embedding)
VALUES (%s, %s::vector)
"""

_SQL_INSERT_AUDIO_EMBED = """
INSERT INTO audio_embeddings (media_id, embedding)
VALUES (%s, %s::vector)
"""


# ──────────────────────────────────────────────────────────────────────────── #
# DB write helpers                                                             #
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
        is_new=True  -> row was just inserted; caller should write embedding.
        is_new=False -> row already existed; caller should check embedding.
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
# Service lifecycle helpers                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

def _make_subprocess_env(**overrides: str) -> dict[str, str]:
    """
    Build an env dict for child service processes.

    Starts from os.environ, then layers in HF_TOKEN from .env (if not already
    set in the OS env), then applies any caller-supplied overrides.

    HuggingFace Hub (huggingface_hub >= 0.20) reads HF_TOKEN.  The legacy
    HUGGING_FACE_HUB_TOKEN alias is also set for compatibility with older
    transformers builds.
    """
    from dotenv import dotenv_values  # noqa: PLC0415

    env = os.environ.copy()

    # Load .env values without touching os.environ in the current process.
    env_file = _PROJECT_ROOT / ".env"
    if env_file.exists():
        dotenv_vals = dotenv_values(str(env_file))
        hf_token = env.get("HF_TOKEN") or dotenv_vals.get("HF_TOKEN", "")
        if hf_token:
            env["HF_TOKEN"] = hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token  # legacy alias

    env.update(overrides)
    return env


def _ensure_wiki_service(
    client: ImageEmbedClient,
    timeout_s: int,
    service_log: Path,
) -> bool:
    """Auto-start the wiki embed service (jina-clip-v2) if not already running."""
    if client.ping(timeout_s=3.0):
        return True

    _LOG.info("Wiki embed service not running — auto-starting ...")
    _LOG.info("Service stdout/stderr -> %s", service_log)

    svc_log_fh = open(service_log, "ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", "rag_v1.wiki.mm_embed_service.app"],
        env=_make_subprocess_env(),
        stdout=svc_log_fh,
        stderr=svc_log_fh,
    )
    atexit.register(_terminate, proc)
    atexit.register(svc_log_fh.close)

    deadline = time.monotonic() + timeout_s
    next_log = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if client.ping(timeout_s=2.0):
            _LOG.info("Wiki embed service ready.")
            return True
        now = time.monotonic()
        if now >= next_log:
            remaining = int(deadline - now)
            _LOG.info("Waiting for jina-clip-v2 to load ... (%d s remaining)", remaining)
            next_log = now + 15.0
        time.sleep(2.0)

    _LOG.error("Wiki embed service did not become healthy within %d s.", timeout_s)
    return False


def _ensure_clap_service(
    client: AudioEmbedClient,
    timeout_s: int,
    device: str | None,
    service_log: Path,
) -> bool:
    if client.ping(timeout_s=3.0):
        return True

    _LOG.info("CLAP embed service not running — auto-starting ...")
    _LOG.info("Service stdout/stderr -> %s", service_log)
    clap_overrides: dict[str, str] = {}
    if device:
        clap_overrides["CLAP_DEVICE"] = device
    env = _make_subprocess_env(**clap_overrides)

    svc_log_fh = open(service_log, "ab")  # append; multiple runs accumulate
    proc = subprocess.Popen(
        [sys.executable, "-m", "rag_v1.media.clap_embed_service.app"],
        env=env,
        stdout=svc_log_fh,
        stderr=svc_log_fh,
    )
    atexit.register(_terminate, proc)
    atexit.register(svc_log_fh.close)

    deadline = time.monotonic() + timeout_s
    next_log = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if client.ping(timeout_s=2.0):
            _LOG.info("CLAP embed service ready.")
            return True
        now = time.monotonic()
        if now >= next_log:
            remaining = int(deadline - now)
            _LOG.info("Waiting for CLAP model to load ... (%d s remaining)", remaining)
            next_log = now + 15.0
        time.sleep(2.0)

    _LOG.error("CLAP embed service did not become healthy within %d s.", timeout_s)
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

def _process_image_batch(
    jobs: list[_FileJob],
    client: ImageEmbedClient,
    conn: psycopg.Connection[DictRow],
    stats: IngestStats,
) -> None:
    if not jobs:
        return

    try:
        vecs = client.embed_image_bytes([j.raw for j in jobs])
    except Exception:
        _LOG.exception("Image embed call failed for %d file(s).", len(jobs))
        stats.errors += len(jobs)
        return

    if len(vecs) != len(jobs):
        _LOG.error(
            "Image embed returned %d vectors for %d inputs — dropping batch.",
            len(vecs), len(jobs),
        )
        stats.errors += len(jobs)
        return

    with conn.cursor() as cur:
        for job, vec in zip(jobs, vecs):
            cur.execute("SAVEPOINT sp_img")
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
                    cur.execute("ROLLBACK TO SAVEPOINT sp_img")
                    stats.errors += 1
                    continue

                if not is_new:
                    existing = cur.execute(
                        _SQL_CHECK_IMAGE_EMBED, (str(job.path), job.content_hash)
                    ).fetchone()
                    if existing:
                        cur.execute("RELEASE SAVEPOINT sp_img")
                        stats.skipped += 1
                        continue
                    _LOG.info("Backfilling missing image embedding for %s", job.path)

                cur.execute(_SQL_INSERT_IMAGE_EMBED, (media_id, vec))
                cur.execute("RELEASE SAVEPOINT sp_img")
                stats.written += 1

            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT sp_img")
                _LOG.exception("DB write failed for %s", job.path)
                stats.errors += 1

    conn.commit()


def _process_audio_batch(
    jobs: list[_FileJob],
    client: AudioEmbedClient,
    conn: psycopg.Connection[DictRow],
    stats: IngestStats,
) -> None:
    if not jobs:
        return

    try:
        vecs = client.embed_audio_bytes([j.raw for j in jobs])
    except Exception:
        _LOG.exception("Audio embed call failed for %d file(s).", len(jobs))
        stats.errors += len(jobs)
        return

    if len(vecs) != len(jobs):
        _LOG.error(
            "Audio embed returned %d vectors for %d inputs — dropping batch.",
            len(vecs), len(jobs),
        )
        stats.errors += len(jobs)
        return

    with conn.cursor() as cur:
        for job, vec in zip(jobs, vecs):
            cur.execute("SAVEPOINT sp_aud")
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
                    cur.execute("ROLLBACK TO SAVEPOINT sp_aud")
                    stats.errors += 1
                    continue

                if not is_new:
                    existing = cur.execute(
                        _SQL_CHECK_AUDIO_EMBED, (str(job.path), job.content_hash)
                    ).fetchone()
                    if existing:
                        cur.execute("RELEASE SAVEPOINT sp_aud")
                        stats.skipped += 1
                        continue
                    _LOG.info("Backfilling missing audio embedding for %s", job.path)

                cur.execute(_SQL_INSERT_AUDIO_EMBED, (media_id, vec))
                cur.execute("RELEASE SAVEPOINT sp_aud")
                stats.written += 1

            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT sp_aud")
                _LOG.exception("DB write failed for %s", job.path)
                stats.errors += 1

    conn.commit()


# ──────────────────────────────────────────────────────────────────────────── #
# Main ingest loop                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def run_ingest(
    root: Path,
    dsn: str,
    image_host: str,
    image_port: int,
    audio_host: str,
    audio_port: int,
    image_batch: int,
    audio_batch: int,
    startup_timeout: int,
    audio_device: str | None,
    log_every: int,
    auto_service: bool,
    clap_service_log: Path | None = None,
    wiki_service_log: Path | None = None,
) -> IngestStats:
    img_client = ImageEmbedClient(host=image_host, port=image_port)
    aud_client = AudioEmbedClient(host=audio_host, port=audio_port)
    stats = IngestStats()

    _logs_dir = Path(__file__).resolve().parents[2] / "logs"
    _clap_log = clap_service_log or (_logs_dir / "clap_embed_service.log")
    _wiki_log = wiki_service_log or (_logs_dir / "wiki_embed_service.log")

    if auto_service:
        # Auto-start wiki embed service (jina-clip-v2) for image embedding
        if not _ensure_wiki_service(img_client, startup_timeout, _wiki_log):
            _LOG.error("Cannot proceed with image ingest without the wiki embed service.")
            # Continue anyway — audio will still be ingested

        # Auto-start CLAP service for audio embedding
        if not _ensure_clap_service(aud_client, startup_timeout, audio_device, _clap_log):
            _LOG.error("Cannot proceed with audio ingest without the CLAP embed service.")
            # Continue anyway — images will still be ingested

    img_buf: list[_FileJob] = []
    aud_buf: list[_FileJob] = []
    total = 0

    with conn_ctx(dsn) as conn:
        for path, modality in _iter_media(root):
            total += 1

            if modality == "video":
                _LOG.info("Skipping video (deferred): %s", path)
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
                content_hash=sha256_bytes(raw),
            )
            if modality == "image":
                job.width, job.height = _image_meta(raw)
                img_buf.append(job)
                if len(img_buf) >= image_batch:
                    _process_image_batch(img_buf, img_client, conn, stats)
                    img_buf = []
            else:  # audio
                job.duration_s = _audio_duration(path)
                aud_buf.append(job)
                if len(aud_buf) >= audio_batch:
                    _process_audio_batch(aud_buf, aud_client, conn, stats)
                    aud_buf = []

            if total % log_every == 0:
                _print_progress(total, stats)

        # Flush partial batches
        _process_image_batch(img_buf, img_client, conn, stats)
        _process_audio_batch(aud_buf, aud_client, conn, stats)

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
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = load_media_embed_config()

    _log_dir = Path(__file__).resolve().parents[2] / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _fh = RotatingFileHandler(
        filename=str(_log_dir / "media_ingest.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    _fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(_fh)

    parser = argparse.ArgumentParser(
        description=(
            "Ingest media files into media_files + image_embeddings / audio_embeddings tables.\n"
            "Images use the wiki embed service (port 8031).\n"
            "Audio uses the CLAP embed service (port 8040)."
        )
    )
    
    parser.add_argument("--root",            default=None,        help="Root directory to scan recursively (required unless --lyrics-only)")
    parser.add_argument("--audio-device",    default=None,        help="Override CLAP_DEVICE (e.g. cuda:1)")
    parser.add_argument("--image-batch",     type=int,            default=cfg.image_batch)
    parser.add_argument("--audio-batch",     type=int,            default=cfg.audio_batch)
    parser.add_argument("--log-every",       type=int,            default=50,   help="Print stats every N files")
    parser.add_argument("--no-service",      action="store_true",               help="Skip auto-starting CLAP/wiki embed services")
    parser.add_argument("--no-lyrics",       action="store_true",               help="Skip lyrics ingest phase")
    parser.add_argument("--lyrics-only",     action="store_true",               help="Run only lyrics phase; skip audio/image ingest")
    parser.add_argument("--lyrics-workers",  type=int,            default=8,    help="Genius API concurrent workers (default: 8)")
    parser.add_argument("--lyrics-batch",    type=int,            default=50,   help="Songs between DB commits in lyrics phase (default: 50)")
    parser.add_argument("--lyrics-delay",    type=float,          default=0.3,  help="Seconds to sleep between Genius calls per worker (default: 0.3)")
    parser.add_argument("--no-analysis",     action="store_true",               help="Skip audio analysis phase (BPM, key, vocal detection, explicit flagging)")
    parser.add_argument("--analysis-workers",type=int,            default=4,    help="Thread pool size for librosa BPM/key extraction (default: 4)")
    parser.add_argument("--cluster",         action="store_true",               help="Run KMeans clustering on audio embeddings after analysis")
    parser.add_argument("--n-clusters",      type=int,            default=50,   help="Number of KMeans clusters (default: 50)")
    args = parser.parse_args()

    if not args.lyrics_only and args.root is None:
        parser.error("--root is required unless --lyrics-only is specified")

    from pg_settings import PgSettings  # noqa: PLC0415
    pg = PgSettings()

    # ── Phase 1: audio + image ingest ────────────────────────────────────── #
    if not args.lyrics_only:
        root = Path(args.root).expanduser().resolve()  # type: ignore[arg-type]
        if not root.is_dir():
            print(f"ERROR: --root is not a directory: {root}", file=sys.stderr)
            sys.exit(1)

        stats = run_ingest(
            root=root,
            dsn=pg.pg_dsn,
            image_host=cfg.image_host,
            image_port=cfg.image_port,
            audio_host=cfg.audio_host,
            audio_port=cfg.audio_port,
            image_batch=args.image_batch,
            audio_batch=args.audio_batch,
            startup_timeout=int(cfg.startup_timeout_s),
            audio_device=args.audio_device,
            log_every=args.log_every,
            auto_service=not args.no_service,
        )
        print(f"\n=== Audio/image ingest complete === {stats.report()}", flush=True)
        _LOG.info("Audio/image ingest complete. %s", stats.report())

    # ── Phase 2: lyrics fetch + embed ────────────────────────────────────── #
    if not args.no_lyrics:
        from rag_v1.config.rag_settings import RagSettings      # noqa: PLC0415
        from rag_v1.media.lyrics_ingest import run_lyrics_ingest  # noqa: PLC0415

        rag_cfg = RagSettings()
        lyrics_stats = run_lyrics_ingest(
            dsn=pg.pg_dsn,
            embed_base_url=rag_cfg.embed_base_url,
            embed_model=rag_cfg.embed_model,
            workers=args.lyrics_workers,
            lyrics_batch=args.lyrics_batch,
            delay_s=args.lyrics_delay,
            chunk_chars=rag_cfg.chunk_chars,
            chunk_overlap=rag_cfg.chunk_overlap,
        )
        print(f"\n=== Lyrics ingest complete === {lyrics_stats.report()}", flush=True)
        _LOG.info("Lyrics ingest complete. %s", lyrics_stats.report())

    # ── Phase 3: audio analysis (BPM, key, vocal detection, explicit) ────── #
    if not args.no_analysis and not args.lyrics_only:
        from rag_v1.media.audio_analysis_ingest import run_audio_analysis_ingest  # noqa: PLC0415

        analysis_stats = run_audio_analysis_ingest(
            dsn=pg.pg_dsn,
            clap_host=cfg.audio_host,
            clap_port=cfg.audio_port,
            workers=args.analysis_workers,
        )
        print(f"\n=== Audio analysis complete === {analysis_stats.report()}", flush=True)
        _LOG.info("Audio analysis complete. %s", analysis_stats.report())

    # ── Phase 4: acoustic clustering (optional, only when --cluster) ──────── #
    if args.cluster and not args.lyrics_only:
        from rag_v1.media.audio_cluster import run_clustering  # noqa: PLC0415

        cluster_stats = run_clustering(
            dsn=pg.pg_dsn,
            n_clusters=args.n_clusters,
        )
        print(f"\n=== Clustering complete === {cluster_stats.report()}", flush=True)
        _LOG.info("Clustering complete. %s", cluster_stats.report())


if __name__ == "__main__":
    main()
