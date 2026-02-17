from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple
import json

import psycopg
from psycopg.rows import dict_row

from rag_v1.embed.embed_client import EmbedClient
from rag_v1.config.rag_settings import RagSettings


def iter_text_files(root: Path) -> Iterable[Path]:
    exts = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_chars, n)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


def upsert_chunks(
    conn: psycopg.Connection,
    source_id: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: dict,
) -> int:
    sql = """
    INSERT INTO rag_chunks (source_id, chunk_id, content, metadata, embedding)
    VALUES (%s, %s, %s, %s::jsonb, %s)
    ON CONFLICT (source_id, chunk_id)
    DO UPDATE SET
        content = EXCLUDED.content,
        metadata = EXCLUDED.metadata,
        embedding = EXCLUDED.embedding,
        updated_at = now()
    ;
    """
    with conn.cursor() as cur:
        for idx, (content, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(sql, (source_id, idx, content, json.dumps(metadata), emb))
    return len(chunks)


def main() -> None:
    cfg = RagSettings()

    folder = os.environ.get("SAGE_RAG_INGEST_DIR", r"F:\Projects\sage_kaizen_ai")
    root = Path(folder).resolve()
    print(f"Ingesting from: {root}")

    embed = EmbedClient(cfg.embed_base_url, cfg.embed_model)

    total = 0
    with psycopg.connect(cfg.pg_dsn, row_factory=dict_row) as conn: # type: ignore[arg-type]
        for path in iter_text_files(root):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            chunks = chunk_text(text, chunk_chars=cfg.chunk_chars, overlap=cfg.chunk_overlap)
            if not chunks:
                continue

            # Embeddings call (batch by file; you can micro-batch later)
            embs = embed.embed(chunks)

            meta = {
                "path": str(path),
                "ext": path.suffix.lower(),
                "bytes": path.stat().st_size,
            }

            n = upsert_chunks(conn, source_id=str(path), chunks=chunks, embeddings=embs, metadata=meta)
            conn.commit()
            total += n
            print(f"Upserted {n:4d} chunks | {path}")

    print(f"Done. Total chunks upserted: {total}")


if __name__ == "__main__":
    main()
