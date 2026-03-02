"""
rag_v1/wiki/wiki_ingest.py

Offline batch job: ingest a Wikipedia ZIM dump (extracted by zim_dump.py)
into the wiki_* PostgreSQL tables using jinaai/jina-clip-v2 embeddings.

Run command (Windows cmd.exe):
    set SAGE_WIKI_ROOT=I:\\llm_data\\wikipedia_maxi_2025_08
    set SAGE_WIKI_EMBED_DEVICE=cuda:0
    set SAGE_WIKI_EMBED_PORT=8031
    python -m rag_v1.wiki.wiki_ingest --root "%SAGE_WIKI_ROOT%"

Optional: skip auto-spawning the embed service (if already running):
    python -m rag_v1.wiki.wiki_ingest --root "%SAGE_WIKI_ROOT%" --no-embed-service

Env vars:
    SAGE_WIKI_ROOT              — path to extracted ZIM dump
    SAGE_WIKI_EMBED_HOST        — embed service host   (default 127.0.0.1)
    SAGE_WIKI_EMBED_PORT        — embed service port   (default 8031)
    SAGE_WIKI_EMBED_DEVICE      — PyTorch device       (default cuda:0)
    SAGE_WIKI_TEXT_BATCH        — texts per embed call (default 32)
    SAGE_WIKI_IMAGE_BATCH       — images per embed call(default 8)
    SAGE_WIKI_COMMIT_EVERY      — rows per commit      (default 200)
    SAGE_WIKI_LOG_EVERY_PAGES   — progress log period  (default 10000)
    SAGE_WIKI_EXCLUDE_SECTIONS  — CSV of heading names to skip
                                   (default: References,External links,
                                    Further reading,Authority control)
    PG_USER / PG_PASSWORD / PG_HOST / PG_PORT / PG_DB  — Postgres creds
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import logging
import mimetypes
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import LiteralString

import psycopg
from psycopg.rows import dict_row, DictRow

from rag_v1.db.pg import get_conn
from rag_v1.ingest.ingest_utils import chunk_text, folder_source_id, sha256_text
from rag_v1.wiki.mm_embed_client import MmEmbedClient

# ──────────────────────────────────────────────────────────────────────────── #
# Logging                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
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

# Default excluded heading names (case-sensitive match against heading text)
_DEFAULT_EXCLUDED = "References,External links,Further reading,Authority control"

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
ON CONFLICT (bundle_id, relative_path) DO NOTHING;
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
# Embed service lifecycle                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

def _start_embed_service(host: str, port: int, timeout_s: float = 300.0) -> subprocess.Popen:
    """
    Spawn the embed service as a subprocess and wait up to timeout_s for /health.

    stderr is written to a temp log file so startup errors are visible.
    First-time jina-clip-v2 load (GPU init + model weights) can take 2-5 min.
    """
    import tempfile

    log_path = Path(tempfile.gettempdir()) / f"sage_wiki_embed_{port}.log"
    _LOG.info("Starting wiki embed service on %s:%s …", host, port)
    _LOG.info("  Embed service log → %s", log_path)

    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "rag_v1.wiki.mm_embed_service.app",
         "--host", host, "--port", str(port)],
        stdout=log_fh,
        stderr=log_fh,
    )
    client  = MmEmbedClient(host=host, port=port)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        # If the process died, surface its log immediately.
        if proc.poll() is not None:
            log_fh.flush()
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                tail = "(log unreadable)"
            raise RuntimeError(
                f"Wiki embed service exited (rc={proc.returncode}) before becoming healthy.\n"
                f"Last output from {log_path}:\n{tail}"
            )
        if client.ping(timeout_s=2.0):
            _LOG.info("Wiki embed service ready on port %s.", port)
            return proc
        elapsed = time.monotonic() - (deadline - timeout_s)
        _LOG.info("  Waiting for embed service … %.0f s elapsed", elapsed)
        time.sleep(5.0)

    proc.terminate()
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    except OSError:
        tail = "(log unreadable)"
    raise RuntimeError(
        f"Wiki embed service did not become healthy within {timeout_s:.0f} s (port {port}).\n"
        f"Last output from {log_path}:\n{tail}"
    )


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

    # Process images one or in small batches (image_batch)
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

            # Skip if already in DB (UNIQUE on byte_hash)
            with conn.cursor() as cur:
                if cur.execute(_SQL_CHECK_IMAGE, (byte_hash,)).fetchone():
                    continue

            img_bytes_list.append(img_bytes)
            img_meta_list.append((img_file, ref))

        if not img_bytes_list:
            continue

        # Build caption list for batch text embedding
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


# ──────────────────────────────────────────────────────────────────────────── #
# Main                                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sage Kaizen — Wikipedia multimodal ingest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default=os.getenv("SAGE_WIKI_ROOT", r"I:\llm_data\wikipedia_maxi_2025_08"),
        help="Root directory of the extracted Wikipedia ZIM dump.",
    )
    parser.add_argument(
        "--no-embed-service",
        action="store_true",
        help="Do not auto-spawn the embed service (assumes it is already running).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        _LOG.error("SAGE_WIKI_ROOT does not exist: %s", root)
        sys.exit(1)

    embed_host = os.getenv("SAGE_WIKI_EMBED_HOST", "127.0.0.1")
    embed_port = int(os.getenv("SAGE_WIKI_EMBED_PORT", "8031"))
    chunk_chars  = 1200
    overlap      = 200
    text_batch   = int(os.getenv("SAGE_WIKI_TEXT_BATCH", "32"))
    image_batch  = int(os.getenv("SAGE_WIKI_IMAGE_BATCH", "8"))
    log_every    = int(os.getenv("SAGE_WIKI_LOG_EVERY_PAGES", "10000"))
    excluded_raw = os.getenv("SAGE_WIKI_EXCLUDE_SECTIONS", _DEFAULT_EXCLUDED)
    excluded     = {s.strip() for s in excluded_raw.split(",") if s.strip()}

    # Postgres DSN (from PgSettings env vars / .env file)
    from pg_settings import PgSettings
    pg_settings = PgSettings()
    pg_dsn = pg_settings.pg_dsn

    _LOG.info("Wiki ingest starting.")
    _LOG.info("  Root:     %s", root)
    _LOG.info("  Embed:    %s:%s", embed_host, embed_port)
    _LOG.info("  Excluded: %s", excluded)

    embed_proc: subprocess.Popen | None = None
    conn: psycopg.Connection[DictRow] | None = None

    try:
        # ── Start embed service ────────────────────────────────────────── #
        if not args.no_embed_service:
            embed_proc = _start_embed_service(embed_host, embed_port)
        else:
            _LOG.info("--no-embed-service: assuming service is already running.")

        client = MmEmbedClient(host=embed_host, port=embed_port)
        if not client.ping(timeout_s=5.0):
            raise RuntimeError("Embed service is not reachable. Is it running?")

        # ── Connect to Postgres ─────────────────────────────────────────── #
        conn = get_conn(pg_dsn)

        # ── Scan for markdown files ─────────────────────────────────────── #
        md_files = sorted(root.rglob("*.md"))
        total = len(md_files)
        _LOG.info("Found %d markdown files to process.", total)

        written = 0
        skipped = 0
        errors  = 0
        start_time = time.monotonic()

        for page_num, md_path in enumerate(md_files, start=1):
            try:
                did_write, did_skip = _ingest_page(
                    md_path=md_path,
                    root=root,
                    conn=conn,
                    client=client,
                    excluded=excluded,
                    chunk_chars=chunk_chars,
                    overlap=overlap,
                    text_batch=text_batch,
                    image_batch=image_batch,
                )
                if did_write:
                    written += 1
                if did_skip:
                    skipped += 1
            except Exception:
                errors += 1
                _LOG.exception("Error processing %s", md_path)

            if page_num % log_every == 0 or page_num == total:
                elapsed_s = time.monotonic() - start_time
                elapsed_m = elapsed_s / 60.0
                rate = page_num / elapsed_m if elapsed_m > 0 else 0.0
                remaining = total - page_num
                eta_m = remaining / rate if rate > 0 else 0.0
                _LOG.info(
                    "[wiki_ingest] pages=%d/%d  written=%d  skipped=%d  errors=%d"
                    "  |  rate=%.0f pg/min  elapsed=%.0fm  ETA=%.0fm",
                    page_num, total, written, skipped, errors,
                    rate, elapsed_m, eta_m,
                )

        _LOG.info(
            "Ingest complete. written=%d  skipped=%d  errors=%d",
            written, skipped, errors,
        )

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if embed_proc is not None:
            _LOG.info("Shutting down embed service (pid=%s)…", embed_proc.pid)
            embed_proc.terminate()
            try:
                embed_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _LOG.warning("Embed service did not exit in 10 s — killing.")
                embed_proc.kill()
            _LOG.info("Embed service stopped.")


if __name__ == "__main__":
    main()
