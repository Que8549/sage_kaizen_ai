from __future__ import annotations

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row

from rag_v1.embed.embed_client import EmbedClient
from rag_v1.config.rag_settings import RagSettings

from rag_v1.ingest.ingest_utils import (
    sha256_text,
    chunk_text,
    folder_source_id,
    get_existing_content_hash,
    upsert_chunks_executemany,
)


def normalize_path(p: Path) -> str:
    # Stable ID across runs: absolute, normalized, backslashes, no trailing spaces
    return str(p.resolve())


# -----------------------------
# File iteration + chunking
# -----------------------------
def iter_text_files(root: Path) -> Iterable[Path]:
    exts = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


# -----------------------------
# Main workflow
# -----------------------------
def main() -> None:
    cfg = RagSettings()

    folder = os.environ.get("SAGE_RAG_INGEST_DIR", r"F:\Projects\sage_kaizen_ai\docs")
    root = Path(folder).resolve()
    print(f"Ingesting from: {root}")

    embed = EmbedClient(cfg.embed_base_url, cfg.embed_model)

    # Commit batching controls
    commit_every = int(os.environ.get("SAGE_RAG_COMMIT_EVERY", "200"))  # chunks
    total_chunks = 0
    pending_chunks = 0
    files_processed = 0
    files_skipped = 0

    with psycopg.connect(cfg.pg_dsn, row_factory=dict_row) as conn:  # type: ignore[arg-type]
        for path in iter_text_files(root):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if not text.strip():
                continue

            source_id = folder_source_id(path)

            # Build a stable content hash (include path metadata lightly)
            # NOTE: Hash only content is fine; adding path is optional. We'll hash content only.
            content_hash = sha256_text(text)

            # DEDUPE: skip unchanged files
            existing = get_existing_content_hash(conn, source_id)
            if existing and existing == content_hash:
                files_skipped += 1
                continue

            chunks = chunk_text(text, chunk_chars=cfg.chunk_chars, overlap=cfg.chunk_overlap)
            if not chunks:
                continue

            # Embed (batch per file)
            embs = embed.embed(chunks)

            meta = {
                "source_type": "localfile",
                "path": str(path),
                "ext": path.suffix.lower(),
                "bytes": path.stat().st_size,
                "mtime": path.stat().st_mtime,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "content_hash": content_hash,
            }

            n = upsert_chunks_executemany(
                conn,
                source_id=source_id,
                chunks=chunks,
                embeddings=embs,
                metadata=meta,
            )
            total_chunks += n
            pending_chunks += n
            files_processed += 1

            if pending_chunks >= commit_every:
                conn.commit()
                pending_chunks = 0

            print(f"Upserted {n:4d} chunks | {path}")

        if pending_chunks > 0:
            conn.commit()

    print(
        "Done. "
        f"Files processed: {files_processed}, files skipped (dedupe): {files_skipped}, "
        f"total chunks upserted: {total_chunks}"
    )


if __name__ == "__main__":
    main()
