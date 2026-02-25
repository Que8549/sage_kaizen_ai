from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import psycopg
from psycopg.rows import dict_row
from docx import Document

from rag_v1.embed.embed_client import EmbedClient
from rag_v1.config.rag_settings import RagSettings
from server_manager import ManagedServers, ensure_embed_running

from rag_v1.ingest.ingest_utils import (
    sha256_text,
    chunk_text,
    folder_source_id,
    get_existing_content_hash,
    upsert_chunks_executemany,
)
from rag_v1.ingest.ingest_runtime import CommitBatcher


def iter_text_files(root: Path) -> Iterable[Path]:
    exts = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml", ".docx"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


_DOCX_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_file_text(path: Path) -> str:
    if path.suffix.lower() != ".docx":
        return path.read_text(encoding="utf-8", errors="ignore")

    doc = Document(str(path))
    parts: list[str] = []
    W = _DOCX_W

    for child in doc.element.body:
        if child.tag == W + "p":
            text = "".join(node.text or "" for node in child.iter() if node.tag == W + "t")
            if text.strip():
                parts.append(text)
        elif child.tag == W + "tbl":
            for tr in child.findall(".//" + W + "tr"):
                cells = [
                    "".join(node.text or "" for node in tc.iter() if node.tag == W + "t").strip()
                    for tc in tr.findall(W + "tc")
                ]
                row = " | ".join(c for c in cells if c)
                if row:
                    parts.append(row)

    return "\n".join(parts)


def exclude_file(path: Path) -> bool:
    ignore_files = {"baseline_benchmark_prompts.txt", "raspberry pi commands.txt"}

    for file_name in ignore_files:
        if file_name in str(path):
            print(f"Skipping {file_name}...")
            return True
        
    return False


def main() -> None:
    cfg = RagSettings()

    folder = os.environ.get("SAGE_RAG_INGEST_DIR", r"F:\Projects\sage_kaizen_ai\docs")
    root = Path(folder).resolve()
    print(f"Ingesting from: {root}")

    servers = ManagedServers.from_yaml()
    ok, msg = ensure_embed_running(servers)
    if not ok:
        raise RuntimeError(f"Embedding server failed to start: {msg}")
    print(f"Embedding server: {msg}")

    embed = EmbedClient(cfg.embed_base_url, cfg.embed_model)

    commit_every = int(os.environ.get("SAGE_RAG_COMMIT_EVERY", "200"))  # in chunks
    batcher = CommitBatcher(commit_every=commit_every)

    total_chunks = 0
    files_processed = 0
    files_skipped = 0

    with psycopg.connect(cfg.pg_dsn, row_factory=dict_row) as conn:  # type: ignore[arg-type]
        for path in iter_text_files(root):
            try:
                if exclude_file(path):
                    continue

                text = read_file_text(path)
            except Exception:
                continue

            if not text.strip():
                continue

            source_id = folder_source_id(path)
            content_hash = sha256_text(text)

            # DEDUPE: skip unchanged
            existing = get_existing_content_hash(conn, source_id)
            if existing and existing == content_hash:
                files_skipped += 1
                continue

            chunks = chunk_text(text, chunk_chars=cfg.chunk_chars, overlap=cfg.chunk_overlap)
            if not chunks:
                continue

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

            n = upsert_chunks_executemany(conn, source_id=source_id, chunks=chunks, embeddings=embs, metadata=meta)
            total_chunks += n
            files_processed += 1

            batcher.add(n)
            if batcher.should_commit():
                conn.commit()
                batcher.reset()

            print(f"Upserted {n:4d} chunks | {path}")

        batcher.commit_if_needed(conn)

    print(
        "Done. "
        f"Files processed: {files_processed}, "
        f"files skipped (dedupe): {files_skipped}, "
        f"total chunks upserted: {total_chunks}"
    )


if __name__ == "__main__":
    main()
