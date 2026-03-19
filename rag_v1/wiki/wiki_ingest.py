"""
rag_v1/wiki/wiki_ingest.py

Offline batch job: ingest a Wikipedia ZIM dump (extracted by zim_dump.py)
into the wiki_* PostgreSQL tables using jinaai/jina-clip-v2 embeddings.

Pipeline (3 stages, all running concurrently):
  Stage 1 — IO Scanner (1 thread):
      Walks the dump tree sequentially (HDD-friendly), computes content_hash,
      skips unchanged pages, pre-reads image bytes, and feeds PageRaw objects
      into a bounded work_queue (maxsize=200).
  Stage 2 — Embed+Write Workers (2 threads):
      Worker A → embed service A on port 8031 (cuda:0 / RTX 5090)
      Worker B → embed service B on port 8032 (cuda:1 / RTX 5080)
      Each worker pulls PageRaw from the shared queue, chunks text, calls its
      embed service, and writes chunks + images to Postgres.
  Stage 3 — Progress Reporter (1 thread):
      Logs written/skipped/errors every 30 seconds.

All configuration is read from config/brains/brains.yaml (wiki_embed: section).

Run command:
    python -m rag_v1.wiki.wiki_ingest

Optional overrides:
    python -m rag_v1.wiki.wiki_ingest --root "I:\\other\\dump_root"
    python -m rag_v1.wiki.wiki_ingest --no-embed-service

CLI flags:
    --root              Override the wiki root from brains.yaml (ingest.root)
    --no-embed-service  Skip auto-spawning embed services (assumes BOTH
                        port-8031 and port-8032 instances are already running)

Config keys consumed from brains.yaml wiki_embed:
    model                    local jina-clip-v2 directory (used by app.py)
    log                      log file path (service A); service B uses log_b suffix
    startup_timeout_s        seconds to wait for each embed service /health
    service.host             embed service host
    service.port             embed service A port (default 8031; B uses 8032)
    service.text_batch       texts per embed call (recommend 128)
    service.image_batch      images per embed call (recommend 32)
    ingest.root              root directory of the extracted ZIM dump
    ingest.chunk_chars       target chunk size in characters
    ingest.overlap           overlap between consecutive chunks
    ingest.exclude_sections  comma-separated heading names to skip

Note: WIKI_EMBED_PORT and WIKI_EMBED_DEVICE are injected internally into the
embed service subprocesses — they are not intended to be set by the user.

Postgres credentials:
    PG_USER / PG_PASSWORD / PG_HOST / PG_PORT / PG_DB  (via .env or env vars)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import logging
import mimetypes
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, LiteralString

import psycopg
from psycopg.rows import dict_row, DictRow

from rag_v1.db.pg import get_conn
from rag_v1.ingest.ingest_utils import chunk_text, folder_source_id, sha256_text
from rag_v1.wiki.mm_embed_client import MmEmbedClient
from rag_v1.wiki.wiki_embed_config import WikiEmbedConfig, load_wiki_embed_config

# ──────────────────────────────────────────────────────────────────────────── #
# Logger (handlers added in main() after config is loaded)                      #
# ──────────────────────────────────────────────────────────────────────────── #

_LOG = logging.getLogger("sage_kaizen.wiki_ingest")

# ──────────────────────────────────────────────────────────────────────────── #
# Constants / patterns                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

_IMAGE_EXTS  = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_HEADING_RE  = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_ABBREV_RE   = re.compile(r"^\*\[.+?\]:.+$", re.MULTILINE)
_LINK_RE     = re.compile(r"\[([^\]]+)\]\([^)]*\)")   # [text](url) → text
_IMAGE_MD_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_FAMILY_STRIP = re.compile(
    r"_\(disambiguation\)$|_\(unknown\)$|_\(list\)$|_\(redirect\)$",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────────────── #
# Data classes                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class SectionChunk:
    section_path: list[str]
    chunk_index: int
    text: str        # raw markdown stored in DB
    embed_text: str  # cleaned text sent to embedding model
    chunk_hash: str  # sha256 of embed_text


@dataclass
class ImageRef:
    name: str
    caption: str
    hero_rank: int


@dataclass
class PageRaw:
    """
    All data for one page, pre-fetched by the IO scanner thread.
    Passed via work_queue to embed+write worker threads.
    """
    md_path: Path
    raw_text: str
    content_hash: str
    family_title: str
    first_letter: str
    bundle_key: str
    page_source_id: str
    img_files: list[Path]        # pre-validated (suffix in _IMAGE_EXTS)
    img_bytes_list: list[bytes]  # parallel index to img_files


@dataclass
class IngestStats:
    """Thread-safe counters shared across the scanner, workers, and reporter."""
    written: int = 0
    skipped: int = 0
    errors: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_written(self) -> None:
        with self._lock:
            self.written += 1

    def add_skipped(self) -> None:
        with self._lock:
            self.skipped += 1

    def add_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> tuple[int, int, int]:
        """Return (written, skipped, errors) atomically."""
        with self._lock:
            return self.written, self.skipped, self.errors


# ──────────────────────────────────────────────────────────────────────────── #
# Markdown utilities                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

def clean_for_embed(text: str) -> str:
    """Strip markdown link syntax and abbreviation blocks for cleaner embeddings."""
    text = _LINK_RE.sub(r"\1", text)      # [text](url) → text
    text = _ABBREV_RE.sub("", text)       # remove *[ABBR]: ... lines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_section_chunks(
    raw_text: str,
    excluded: set[str],
    chunk_chars: int,
    overlap: int,
) -> list[SectionChunk]:
    """
    Section-aware chunking:
      1. Strip abbreviation blocks.
      2. Split on headings; preamble (before first heading) = section_path=[].
      3. Skip excluded sections.
      4. chunk_text() each segment; build SectionChunk list.
    """
    # 1. Strip abbreviation blocks
    text = _ABBREV_RE.sub("", raw_text).strip()

    # 2. Collect segments: list of (section_path, content_text)
    segments: list[tuple[list[str], str]] = []
    path_stack: list[str] = []
    last_end = 0

    for m in _HEADING_RE.finditer(text):
        content_before = text[last_end:m.start()].strip()
        if content_before:
            segments.append((list(path_stack), content_before))
        level   = len(m.group(1))      # # → 1, ## → 2, ### → 3, #### → 4
        heading = m.group(2).strip()
        path_stack = path_stack[:level - 1] + [heading]
        last_end = m.end()

    trailing = text[last_end:].strip()
    if trailing:
        segments.append((list(path_stack), trailing))

    # 3. Filter excluded sections (last element of path matches excluded set)
    kept = [
        (path, content)
        for path, content in segments
        if not path or path[-1] not in excluded
    ]

    # 4. Sub-chunk each segment
    chunks: list[SectionChunk] = []
    global_idx = 0
    for path, content in kept:
        for sub in chunk_text(content, chunk_chars, overlap):
            sub = sub.strip()
            if not sub:
                continue
            cleaned = clean_for_embed(sub)
            if not cleaned:
                continue
            chunks.append(SectionChunk(
                section_path=path,
                chunk_index=global_idx,
                text=sub,
                embed_text=cleaned,
                chunk_hash=sha256_text(cleaned),
            ))
            global_idx += 1
    return chunks


def _prettify_filename(name: str) -> str:
    stem = Path(name).stem.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", stem).strip()


def extract_image_refs(md_text: str) -> dict[str, ImageRef]:
    """
    Parse image references from markdown: ![caption](filename).
    Returns {filename: ImageRef}. First occurrence = hero_rank 0.
    """
    result: dict[str, ImageRef] = {}
    rank = 0
    for m in _IMAGE_MD_RE.finditer(md_text):
        raw_path = m.group(2).split("?")[0]
        name     = Path(raw_path).name
        if not name:
            continue
        if name not in result:
            caption = m.group(1).strip() or _prettify_filename(name)
            result[name] = ImageRef(name=name, caption=caption, hero_rank=rank)
            rank += 1
    return result


def _detect_page_type(folder_name: str, content: str) -> str:
    if folder_name.lower().endswith("_(disambiguation)"):
        return "disambiguation"
    if folder_name.lower().endswith("_(redirect)"):
        return "redirect"
    fl = folder_name.lower()
    if fl.startswith("list_of") or fl.startswith("list of"):
        return "list"
    # Check content for redirect markers
    if content.strip().lower().startswith("#redirect"):
        return "redirect"
    return "article"


def _mime_for(suffix: str) -> str | None:
    guess, _ = mimetypes.guess_type(f"x{suffix}")
    return guess


# ──────────────────────────────────────────────────────────────────────────── #
# Database helpers                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

_SQL_GET_PAGE_HASH: LiteralString = """
SELECT content_hash FROM wiki_pages
WHERE page_source_id = %s
LIMIT 1;
"""

_SQL_UPSERT_BUNDLE: LiteralString = """
INSERT INTO wiki_bundles (bundle_id, bundle_key, family_title, first_letter, updated_at)
VALUES (%s, %s, %s, %s, now())
ON CONFLICT (bundle_key) DO UPDATE
    SET family_title = EXCLUDED.family_title,
        updated_at   = now()
RETURNING bundle_id;
"""

_SQL_UPSERT_PAGE: LiteralString = """
INSERT INTO wiki_pages
    (page_id, bundle_id, page_source_id, title, page_type, oldid,
     md_path, content_hash, updated_at)
VALUES (%s, %s, %s, %s, %s, NULL, %s, %s, now())
ON CONFLICT (page_source_id) DO UPDATE
    SET bundle_id    = EXCLUDED.bundle_id,
        title        = EXCLUDED.title,
        page_type    = EXCLUDED.page_type,
        md_path      = EXCLUDED.md_path,
        content_hash = EXCLUDED.content_hash,
        updated_at   = now()
RETURNING page_id;
"""

_SQL_DELETE_CHUNKS: LiteralString = "DELETE FROM wiki_chunks WHERE page_id = %s;"

_SQL_INSERT_CHUNK: LiteralString = """
INSERT INTO wiki_chunks
    (page_id, bundle_id, title, first_letter, section_path,
     chunk_index, text, chunk_hash, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (page_id, chunk_hash) DO NOTHING;
"""

_SQL_CHECK_IMAGE: LiteralString = "SELECT 1 FROM wiki_images WHERE byte_hash = %s LIMIT 1;"

_SQL_INSERT_IMAGE: LiteralString = """
INSERT INTO wiki_images
    (bundle_id, first_letter, relative_path, byte_hash, mime,
     caption_text, image_embedding, caption_embedding,
     is_hero, hero_rank)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT DO NOTHING;
"""


def _get_existing_hash(conn: psycopg.Connection[DictRow], source_id: str) -> str | None:
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(_SQL_GET_PAGE_HASH, (source_id,)).fetchone()
        return row["content_hash"] if row else None


def _upsert_bundle(
    conn: psycopg.Connection[DictRow],
    bundle_key: str,
    family_title: str,
    first_letter: str,
) -> str:
    new_id = str(uuid.uuid4())
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(
            _SQL_UPSERT_BUNDLE, (new_id, bundle_key, family_title, first_letter)
        ).fetchone()
    if row is None:
        raise RuntimeError(f"UPSERT RETURNING returned no row for bundle_key={bundle_key!r}")
    return str(row["bundle_id"])


def _upsert_page(
    conn: psycopg.Connection[DictRow],
    page_source_id: str,
    bundle_id: str,
    title: str,
    page_type: str,
    md_path: str,
    content_hash: str,
) -> str:
    new_id = str(uuid.uuid4())
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(
            _SQL_UPSERT_PAGE,
            (new_id, bundle_id, page_source_id, title, page_type, md_path, content_hash),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"UPSERT RETURNING returned no row for page_source_id={page_source_id!r}")
    return str(row["page_id"])


# ──────────────────────────────────────────────────────────────────────────── #
# Logging setup                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

def _setup_logging(log_path: Path) -> None:
    """
    Configure the root logger to write to both stdout and the log file
    specified in brains.yaml (wiki_embed.log).

    Called once at the start of main() after config is loaded.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # httpx logs every HTTP request at INFO level; suppress to WARNING so the
    # embed-call traffic doesn't flood the ingest log.
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────── #
# File discovery                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

def _iter_md_files(root: Path) -> Generator[Path, None, None]:
    """
    Walk the 3-level Wikipedia dump tree without rglob and without sorting.

    Structure:  root / {L1} / {L2_3chars} / {article_dir} / *.md

    Avoids materialising millions of Paths in RAM.  Resume safety is handled by
    content_hash in the DB — processing order does not matter.

    Yields Path objects for every *.md file found.
    """
    for l1 in root.iterdir():
        if not l1.is_dir():
            continue
        for l2 in l1.iterdir():
            if not l2.is_dir():
                continue
            for article_dir in l2.iterdir():
                if not article_dir.is_dir():
                    continue
                for f in article_dir.iterdir():
                    if f.suffix.lower() == ".md":
                        yield f


# ──────────────────────────────────────────────────────────────────────────── #
# Embed service lifecycle                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

def _start_embed_service(
    cfg: WikiEmbedConfig,
    port: int | None = None,
    device: str | None = None,
    log_suffix: str = "",
) -> subprocess.Popen:
    """
    Spawn mm_embed_service.app as a subprocess and wait until /health responds.

    port and device override cfg.port / cfg.device when provided, allowing a
    second instance to be started on a different GPU and port (e.g. port=8032,
    device="cuda:1" for the RTX 5080 during ingest).

    log_suffix is appended to the log file stem to disambiguate the two instances
    (e.g. log_suffix="_b" → sage_wiki_embed_b.log).

    First-time jina-clip-v2 load (GPU init + model weights) can take 30–90 s.
    startup_timeout_s from brains.yaml provides ample headroom.
    """
    effective_port   = port   if port   is not None else cfg.port
    effective_device = device if device is not None else cfg.device

    base_log = cfg.log
    log_path = base_log.with_stem(base_log.stem + log_suffix) if log_suffix else base_log
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _LOG.info(
        "Starting wiki embed service (port=%s, device=%s) — log → %s",
        effective_port, effective_device, log_path,
    )

    # Inject port and device overrides as env vars for the subprocess.
    # The service reads WIKI_EMBED_PORT / WIKI_EMBED_DEVICE in preference to
    # brains.yaml when these are set.
    env = os.environ.copy()
    env["WIKI_EMBED_PORT"]   = str(effective_port)
    env["WIKI_EMBED_DEVICE"] = effective_device

    log_fh = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "rag_v1.wiki.mm_embed_service.app"],
        stdout=log_fh,
        stderr=log_fh,
        env=env,
    )

    client    = MmEmbedClient(host=cfg.host, port=effective_port)
    timeout_s = cfg.startup_timeout_s
    deadline  = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        # Surface crash immediately instead of waiting for full timeout.
        if proc.poll() is not None:
            log_fh.flush()
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                tail = "(log unreadable)"
            raise RuntimeError(
                f"Wiki embed service (port={effective_port}) exited "
                f"(rc={proc.returncode}) before becoming healthy.\n"
                f"Last output from {log_path}:\n{tail}"
            )
        if client.ping(timeout_s=2.0):
            _LOG.info("Wiki embed service ready on port %s.", effective_port)
            return proc
        elapsed = time.monotonic() - (deadline - timeout_s)
        _LOG.info(
            "  Waiting for embed service (port=%s) … %.0f s elapsed",
            effective_port, elapsed,
        )
        time.sleep(5.0)

    proc.terminate()
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    except OSError:
        tail = "(log unreadable)"
    raise RuntimeError(
        f"Wiki embed service (port={effective_port}) did not become healthy "
        f"within {timeout_s:.0f} s.\nLast output from {log_path}:\n{tail}"
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Pipeline: IO scanner                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

def _scan_pages(
    root: Path,
    scan_conn: psycopg.Connection[DictRow],
    work_queue: "queue.Queue[PageRaw | None]",
    stats: IngestStats,
    num_workers: int,
) -> None:
    """
    IO Scanner — runs in a single daemon thread.

    Walks the dump tree sequentially (HDD-friendly), computes content_hash for
    each .md file, skips pages whose hash matches the DB, pre-reads all image
    bytes for new/changed pages, and puts PageRaw objects onto work_queue.

    On completion puts num_workers sentinel (None) values onto the queue so
    each worker thread receives exactly one stop signal.
    """
    for md_path in _iter_md_files(root):
        folder_name    = md_path.parent.name
        page_source_id = folder_source_id(md_path)

        try:
            raw_text = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            _LOG.warning("Scanner: cannot read %s — skipping.", md_path)
            stats.add_error()
            continue

        content_hash  = sha256_text(raw_text)
        existing_hash = _get_existing_hash(scan_conn, page_source_id)
        if existing_hash == content_hash:
            stats.add_skipped()
            continue

        # Pre-compute bundle metadata (mirrors _ingest_page logic)
        family_title = _FAMILY_STRIP.sub("", folder_name)
        bundle_key   = "wiki:" + family_title
        first_letter = family_title[0].lower() if family_title[0].isalpha() else "#"

        # Pre-read image bytes from the same directory (sequential on HDD)
        img_files_validated: list[Path] = []
        img_bytes_list: list[bytes] = []
        for img_file in sorted(md_path.parent.iterdir(), key=lambda x: x.name):
            if img_file.suffix.lower() not in _IMAGE_EXTS:
                continue
            try:
                img_bytes_list.append(img_file.read_bytes())
                img_files_validated.append(img_file)
            except OSError:
                _LOG.warning("Scanner: cannot read image %s — skipping.", img_file)

        work_queue.put(PageRaw(
            md_path=md_path,
            raw_text=raw_text,
            content_hash=content_hash,
            family_title=family_title,
            first_letter=first_letter,
            bundle_key=bundle_key,
            page_source_id=page_source_id,
            img_files=img_files_validated,
            img_bytes_list=img_bytes_list,
        ))

    # Signal each worker to stop
    for _ in range(num_workers):
        work_queue.put(None)


# ──────────────────────────────────────────────────────────────────────────── #
# Per-page processing                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

def _ingest_page(
    md_path: Path,
    root: Path,
    conn: psycopg.Connection[DictRow],
    client: MmEmbedClient,
    excluded: set[str],
    chunk_chars: int,
    overlap: int,
    text_batch: int,
    image_batch: int,
) -> tuple[bool, bool]:
    """
    Process a single markdown page.

    Returns (written: bool, skipped: bool).
    Raises on unrecoverable error (caller counts errors).
    """
    folder_name = md_path.parent.name

    # Source ID — MUST match ingest_utils convention
    page_source_id = folder_source_id(md_path)

    # Family / bundle metadata
    family_title = _FAMILY_STRIP.sub("", folder_name)
    bundle_key   = "wiki:" + family_title
    first_letter = family_title[0].lower() if family_title[0].isalpha() else "#"

    # Read content
    raw_text = md_path.read_text(encoding="utf-8", errors="replace")
    page_type = _detect_page_type(folder_name, raw_text)
    content_hash = sha256_text(raw_text)

    # Dedupe by content hash
    existing_hash = _get_existing_hash(conn, page_source_id)
    if existing_hash == content_hash:
        return False, True  # skipped

    # Upsert bundle + page
    bundle_id = _upsert_bundle(conn, bundle_key, family_title, first_letter)
    page_id   = _upsert_page(
        conn, page_source_id, bundle_id, family_title,
        page_type, str(md_path), content_hash,
    )

    # Delete stale chunks (re-embed on content change)
    with conn.cursor() as cur:
        cur.execute(_SQL_DELETE_CHUNKS, (page_id,))

    # ── Text chunks ──────────────────────────────────────────────────────── #
    section_chunks = split_into_section_chunks(raw_text, excluded, chunk_chars, overlap)

    for batch in itertools.batched(section_chunks, text_batch):
        texts_to_embed = [c.embed_text for c in batch]
        embs = client.embed_text(texts_to_embed)
        rows = [
            (page_id, bundle_id, family_title, first_letter,
             c.section_path if c.section_path else None,
             c.chunk_index, c.text, c.chunk_hash, embs[i])
            for i, c in enumerate(batch)
        ]
        with conn.cursor() as cur:
            cur.executemany(_SQL_INSERT_CHUNK, rows)

    conn.commit()

    # ── Images ───────────────────────────────────────────────────────────── #
    md_refs = extract_image_refs(raw_text)
    img_files = sorted(
        (f for f in md_path.parent.iterdir() if f.suffix.lower() in _IMAGE_EXTS),
        key=lambda f: f.name,
    )

    pending_imgs: list[tuple[Path, ImageRef | None]] = []
    for img_file in img_files:
        ref = md_refs.get(img_file.name)
        pending_imgs.append((img_file, ref))

    for batch_imgs in itertools.batched(pending_imgs, image_batch):
        img_bytes_list: list[bytes] = []
        img_meta_list: list[tuple[Path, ImageRef | None]] = []

        for img_file, ref in batch_imgs:
            try:
                img_bytes = img_file.read_bytes()
            except OSError:
                continue
            byte_hash = hashlib.sha256(img_bytes).hexdigest()

            # Skip if already in DB (covers cross-page duplicates of shared
            # Wikipedia icons after the previous page's commit)
            with conn.cursor() as cur:
                if cur.execute(_SQL_CHECK_IMAGE, (byte_hash,)).fetchone():
                    continue

            img_bytes_list.append(img_bytes)
            img_meta_list.append((img_file, ref))

        if not img_bytes_list:
            continue

        captions: list[str] = []
        for img_file, ref in img_meta_list:
            if ref:
                captions.append(ref.caption)
            else:
                captions.append(_prettify_filename(img_file.name))

        image_embs   = client.embed_image_bytes(img_bytes_list)
        caption_embs = client.embed_text(captions)

        for idx, (img_file, ref) in enumerate(img_meta_list):
            img_bytes     = img_bytes_list[idx]
            byte_hash     = hashlib.sha256(img_bytes).hexdigest()
            caption       = captions[idx]
            hero_rank     = ref.hero_rank if ref else 9999
            is_hero       = hero_rank == 0
            rel_path      = img_file.relative_to(root).as_posix()
            mime          = _mime_for(img_file.suffix)
            image_emb     = image_embs[idx]
            caption_emb   = caption_embs[idx]

            with conn.cursor() as cur:
                cur.execute(
                    _SQL_INSERT_IMAGE,
                    (bundle_id, first_letter, rel_path, byte_hash, mime,
                     caption, image_emb, caption_emb, is_hero, hero_rank),
                )

    conn.commit()
    return True, False  # written


def _ingest_page_from_raw(
    page: PageRaw,
    root: Path,
    conn: psycopg.Connection[DictRow],
    client: MmEmbedClient,
    excluded: set[str],
    chunk_chars: int,
    overlap: int,
    text_batch: int,
    image_batch: int,
) -> None:
    """
    Process a pre-scanned PageRaw object.

    Called by embed+write worker threads in the parallel pipeline.
    The IO scanner has already done the dedup check; this function always writes.
    Raises on unrecoverable error (caller increments stats.errors and rolls back).
    """
    page_type = _detect_page_type(page.md_path.parent.name, page.raw_text)

    # Upsert bundle + page
    bundle_id = _upsert_bundle(conn, page.bundle_key, page.family_title, page.first_letter)
    page_id   = _upsert_page(
        conn, page.page_source_id, bundle_id, page.family_title,
        page_type, str(page.md_path), page.content_hash,
    )

    # Delete stale chunks (re-embed on content change)
    with conn.cursor() as cur:
        cur.execute(_SQL_DELETE_CHUNKS, (page_id,))

    # ── Text chunks ──────────────────────────────────────────────────────── #
    section_chunks = split_into_section_chunks(page.raw_text, excluded, chunk_chars, overlap)

    for batch in itertools.batched(section_chunks, text_batch):
        texts_to_embed = [c.embed_text for c in batch]
        embs = client.embed_text(texts_to_embed)
        rows = [
            (page_id, bundle_id, page.family_title, page.first_letter,
             c.section_path if c.section_path else None,
             c.chunk_index, c.text, c.chunk_hash, embs[i])
            for i, c in enumerate(batch)
        ]
        with conn.cursor() as cur:
            cur.executemany(_SQL_INSERT_CHUNK, rows)

    conn.commit()

    # ── Images (bytes already pre-read by IO scanner) ─────────────────────── #
    md_refs = extract_image_refs(page.raw_text)

    pending_imgs: list[tuple[Path, bytes, ImageRef | None]] = []
    for img_file, img_bytes in zip(page.img_files, page.img_bytes_list):
        ref       = md_refs.get(img_file.name)
        byte_hash = hashlib.sha256(img_bytes).hexdigest()
        # Skip if already in DB (cross-page shared Wikipedia icons)
        with conn.cursor() as cur:
            if cur.execute(_SQL_CHECK_IMAGE, (byte_hash,)).fetchone():
                continue
        pending_imgs.append((img_file, img_bytes, ref))

    for batch_imgs in itertools.batched(pending_imgs, image_batch):
        img_bytes_batch: list[bytes] = [b for _, b, _ in batch_imgs]
        captions: list[str] = [
            r.caption if r else _prettify_filename(f.name)
            for f, _, r in batch_imgs
        ]

        image_embs   = client.embed_image_bytes(img_bytes_batch)
        caption_embs = client.embed_text(captions)

        for idx, (img_file, img_bytes, ref) in enumerate(batch_imgs):
            byte_hash   = hashlib.sha256(img_bytes).hexdigest()
            caption     = captions[idx]
            hero_rank   = ref.hero_rank if ref else 9999
            is_hero     = hero_rank == 0
            rel_path    = img_file.relative_to(root).as_posix()
            mime        = _mime_for(img_file.suffix)
            image_emb   = image_embs[idx]
            caption_emb = caption_embs[idx]

            with conn.cursor() as cur:
                cur.execute(
                    _SQL_INSERT_IMAGE,
                    (bundle_id, page.first_letter, rel_path, byte_hash, mime,
                     caption, image_emb, caption_emb, is_hero, hero_rank),
                )

    conn.commit()


# ──────────────────────────────────────────────────────────────────────────── #
# Pipeline: embed+write worker and progress reporter                             #
# ──────────────────────────────────────────────────────────────────────────── #

def _run_worker(
    worker_id: str,
    work_queue: "queue.Queue[PageRaw | None]",
    root: Path,
    conn: psycopg.Connection[DictRow],
    client: MmEmbedClient,
    excluded: set[str],
    chunk_chars: int,
    overlap: int,
    text_batch: int,
    image_batch: int,
    stats: IngestStats,
) -> None:
    """
    Embed+Write worker — runs in a daemon thread.

    Pulls PageRaw items from work_queue, embeds and writes each page to
    Postgres, then signals task_done().  Exits when it receives a sentinel
    (None).  Errors are logged and counted; the worker never crashes.
    """
    _LOG.info("Worker %s started.", worker_id)
    while True:
        item = work_queue.get()
        if item is None:
            # Sentinel — this worker is done
            work_queue.task_done()
            _LOG.info("Worker %s received sentinel — exiting.", worker_id)
            return
        try:
            _ingest_page_from_raw(
                page=item,
                root=root,
                conn=conn,
                client=client,
                excluded=excluded,
                chunk_chars=chunk_chars,
                overlap=overlap,
                text_batch=text_batch,
                image_batch=image_batch,
            )
            stats.add_written()
        except Exception:
            _LOG.exception("Worker %s: error processing %s", worker_id, item.md_path)
            stats.add_error()
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            work_queue.task_done()


def _run_reporter(
    stats: IngestStats,
    start_time: float,
    stop_event: threading.Event,
    interval_s: float = 30.0,
) -> None:
    """
    Progress reporter — runs in a daemon thread.

    Wakes every interval_s, logs written/skipped/errors and pages-per-minute.
    Exits immediately when stop_event is set.
    """
    while not stop_event.wait(timeout=interval_s):
        written, skipped, errors = stats.snapshot()
        total      = written + skipped + errors
        elapsed_s  = time.monotonic() - start_time
        elapsed_m  = elapsed_s / 60.0
        rate       = total / elapsed_m if elapsed_m > 0 else 0.0
        _LOG.info(
            "[wiki_ingest] written=%d  skipped=%d  errors=%d  "
            "rate=%.0f pg/min  elapsed=%.0fm",
            written, skipped, errors, rate, elapsed_m,
        )


# ──────────────────────────────────────────────────────────────────────────── #
# Main                                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sage Kaizen — Wikipedia multimodal ingest (parallel pipeline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Override the wiki root from brains.yaml (ingest.root).",
    )
    parser.add_argument(
        "--no-embed-service",
        action="store_true",
        help=(
            "Do not auto-spawn embed services "
            "(assumes BOTH port-8031 and port-8032 instances are already running)."
        ),
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────── #
    wiki_cfg = load_wiki_embed_config()
    _setup_logging(wiki_cfg.log)

    root = Path(args.root).resolve() if args.root else wiki_cfg.wiki_root
    if not root.exists():
        _LOG.error("Wiki root does not exist: %s", root)
        sys.exit(1)

    excluded    = wiki_cfg.exclude_sections
    chunk_chars = wiki_cfg.chunk_chars
    overlap     = wiki_cfg.overlap
    text_batch  = wiki_cfg.text_batch
    image_batch = wiki_cfg.image_batch

    from pg_settings import PgSettings
    pg_dsn = PgSettings().pg_dsn

    _LOG.info("Wiki ingest (parallel pipeline) starting.")
    _LOG.info("  Root:         %s", root)
    _LOG.info("  Embed A:      %s:8031  (cuda:0 / RTX 5090)", wiki_cfg.host)
    _LOG.info("  Embed B:      %s:8032  (cuda:1 / RTX 5080)", wiki_cfg.host)
    _LOG.info("  text_batch:   %d   image_batch: %d", text_batch, image_batch)
    _LOG.info("  Log:          %s", wiki_cfg.log)
    _LOG.info("  Excluded:     %s", excluded)

    embed_proc_a: subprocess.Popen | None = None
    embed_proc_b: subprocess.Popen | None = None
    scan_conn:    psycopg.Connection[DictRow] | None = None
    conn_a:       psycopg.Connection[DictRow] | None = None
    conn_b:       psycopg.Connection[DictRow] | None = None

    try:
        # ── Start embed services ─────────────────────────────────────────── #
        if not args.no_embed_service:
            embed_proc_a = _start_embed_service(
                wiki_cfg, port=8031, device="cuda:0", log_suffix="",
            )
            embed_proc_b = _start_embed_service(
                wiki_cfg, port=8032, device="cuda:1", log_suffix="_b",
            )
        else:
            _LOG.info("--no-embed-service: assuming both services are already running.")

        client_a = MmEmbedClient(host=wiki_cfg.host, port=8031)
        client_b = MmEmbedClient(host=wiki_cfg.host, port=8032)
        for label, client in [("A (8031)", client_a), ("B (8032)", client_b)]:
            if not client.ping(timeout_s=5.0):
                raise RuntimeError(f"Embed service {label} is not reachable.")

        # ── Open DB connections ───────────────────────────────────────────── #
        # scan_conn: read-only hash checks; autocommit avoids a long idle
        # transaction being held open across the full multi-hour scan.
        scan_conn = get_conn(pg_dsn)
        scan_conn.autocommit = True
        conn_a    = get_conn(pg_dsn)
        conn_b    = get_conn(pg_dsn)

        # ── Shared pipeline objects ──────────────────────────────────────── #
        stats:      IngestStats                      = IngestStats()
        work_queue: queue.Queue[PageRaw | None]      = queue.Queue(maxsize=200)
        stop_event: threading.Event                  = threading.Event()
        start_time: float                            = time.monotonic()

        # ── Launch threads ────────────────────────────────────────────────── #
        scanner_thread = threading.Thread(
            target=_scan_pages,
            args=(root, scan_conn, work_queue, stats, 2),
            name="wiki-scanner",
            daemon=True,
        )
        worker_a_thread = threading.Thread(
            target=_run_worker,
            args=("A", work_queue, root, conn_a, client_a,
                  excluded, chunk_chars, overlap, text_batch, image_batch, stats),
            name="wiki-worker-A",
            daemon=True,
        )
        worker_b_thread = threading.Thread(
            target=_run_worker,
            args=("B", work_queue, root, conn_b, client_b,
                  excluded, chunk_chars, overlap, text_batch, image_batch, stats),
            name="wiki-worker-B",
            daemon=True,
        )
        reporter_thread = threading.Thread(
            target=_run_reporter,
            args=(stats, start_time, stop_event, 30.0),
            name="wiki-reporter",
            daemon=True,
        )

        scanner_thread.start()
        worker_a_thread.start()
        worker_b_thread.start()
        reporter_thread.start()

        # ── Wait for completion ───────────────────────────────────────────── #
        scanner_thread.join()       # scanner has enqueued all sentinels
        work_queue.join()           # all items (including sentinels) task_done'd
        stop_event.set()            # wake reporter so it logs final stats and exits
        reporter_thread.join(timeout=5.0)

        written, skipped, errors = stats.snapshot()
        elapsed_s = time.monotonic() - start_time
        _LOG.info(
            "Ingest complete. written=%d  skipped=%d  errors=%d  elapsed=%.0fs",
            written, skipped, errors, elapsed_s,
        )

    finally:
        for conn_obj in (scan_conn, conn_a, conn_b):
            if conn_obj is not None:
                try:
                    conn_obj.close()
                except Exception:
                    pass
        for label, proc in [("A", embed_proc_a), ("B", embed_proc_b)]:
            if proc is not None:
                _LOG.info("Shutting down embed service %s (pid=%s)…", label, proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _LOG.warning(
                        "Embed service %s did not exit in 10 s — killing.", label
                    )
                    proc.kill()
        _LOG.info("Embed services stopped.")


if __name__ == "__main__":
    main()
