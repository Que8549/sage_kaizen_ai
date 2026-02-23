from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import psycopg
from psycopg.rows import dict_row


# -----------------------------
# Hashing
# -----------------------------
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


# -----------------------------
# Chunking
# -----------------------------
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



# -----------------------------
# Minimal HTML stripping (fast fallback)
# -----------------------------
_TAG_RE = re.compile(r"<[^>]+>")

def strip_html(html: str) -> str:
    txt = _TAG_RE.sub(" ", html)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


# -----------------------------
# Source ID conventions
# -----------------------------
def normalize_path(p: Path) -> str:
    # Stable ID across runs: absolute, normalized
    return str(p.resolve())

def folder_source_id(path: Path) -> str:
    # localfile:<absolute_path>
    return f"localfile:{normalize_path(path)}"

def normalize_url(url: str) -> str:
    u = url.strip()
    if "#" in u:
        u = u.split("#", 1)[0]
    return u

def web_source_id(url: str) -> str:
    # web:<normalized_url>
    return f"web:{normalize_url(url)}"

def rss_item_source_id(item_key: str) -> str:
    # rss_item:<sha1(guid|link|title)>
    return f"rss_item:{sha1_text(item_key)}"

def absolutize_url(base_url: str, maybe_relative: str) -> str:
    if maybe_relative and not urlparse(maybe_relative).scheme:
        return urljoin(base_url, maybe_relative)
    return maybe_relative


# -----------------------------
# Dedupe helper
# -----------------------------
def get_existing_content_hash(conn: psycopg.Connection, source_id: str) -> Optional[str]:
    """
    Dedupe strategy used across ingesters:
      - store 'content_hash' in metadata
      - read chunk 0 metadata->>'content_hash' for that source_id
    """
    sql = """
    SELECT metadata->>'content_hash' AS content_hash
    FROM rag_chunks
    WHERE source_id = %s AND chunk_id = 0
    LIMIT 1;
    """
    with conn.cursor(row_factory=dict_row) as cur:  # type: ignore[arg-type]
        row = cur.execute(sql, (source_id,)).fetchone()
        if not row:
            return None
        return row.get("content_hash")


# -----------------------------
# Batch UPSERT helper
# -----------------------------
def upsert_chunks_executemany(
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

    meta_json = json.dumps(metadata)

    rows: List[Tuple[str, int, str, str, List[float]]] = []
    for idx, (content, emb) in enumerate(zip(chunks, embeddings)):
        rows.append((source_id, idx, content, meta_json, emb))

    with conn.cursor() as cur:
        try:
            cur.executemany(sql, rows)
        except psycopg.errors.InsufficientPrivilege as e:
            raise RuntimeError(
                "Postgres user lacks permissions on rag_chunks. "
                "Grant INSERT/UPDATE on public.rag_chunks to the configured PG_USER."
            ) from e

    return len(rows)
