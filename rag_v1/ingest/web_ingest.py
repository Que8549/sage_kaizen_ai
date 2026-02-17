from __future__ import annotations

import os
import json
import time
import hashlib
import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import psycopg
from psycopg.rows import dict_row

from rag_v1.embed.embed_client import EmbedClient
from rag_v1.config.rag_settings import RagSettings

from rag_v1.ingest.ingest_utils import (
    sha256_text,
    web_source_id,
    strip_html,
    chunk_text,
    get_existing_content_hash,
    upsert_chunks_executemany,
)


try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore



def normalize_url(url: str) -> str:
    u = url.strip()
    # remove trailing fragment for stability
    if "#" in u:
        u = u.split("#", 1)[0]
    return u


# -----------------------------
# HTML to text extraction
# -----------------------------
_TAG_RE = re.compile(r"<[^>]+>")


def extract_main_text(html: str, url: str) -> str:
    # 1) readability-lxml (best)
    try:
        from readability import Document  # type: ignore
        doc = Document(html)
        title = (doc.short_title() or "").strip()
        content_html = doc.summary(html_partial=True)
        body = strip_html(content_html)
        if title:
            return f"{title}\n\nURL: {url}\n\n{body}".strip()
        return f"URL: {url}\n\n{body}".strip()
    except Exception:
        pass

    # 2) BeautifulSoup fallback (ok)
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, "html.parser")

        # drop scripts/styles/nav/footer
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = (soup.title.string.strip() if soup.title and soup.title.string else "")
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()

        if title:
            return f"{title}\n\nURL: {url}\n\n{text}".strip()
        return f"URL: {url}\n\n{text}".strip()
    except Exception:
        pass

    # 3) minimal stripper
    txt = strip_html(html)
    return f"URL: {url}\n\n{txt}".strip()


# -----------------------------
# Fetch URLs
# -----------------------------
def fetch_url(url: str, timeout_s: float = 30.0) -> str:
    if httpx is None:
        from urllib.request import Request, urlopen
        req = Request(url, headers={"User-Agent": "SageKaizenRAG/1.0"})
        with urlopen(req, timeout=timeout_s) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    else:
        with httpx.Client(timeout=timeout_s, headers={"User-Agent": "SageKaizenRAG/1.0"}) as client:
            r = client.get(url, follow_redirects=True)
            r.raise_for_status()
            return r.text


def load_url_list_from_env() -> List[str]:
    """
    Provide URLs via:
      - SAGE_RAG_URLS="https://a,https://b"
      - OR SAGE_RAG_URLS_FILE="F:\\path\\urls.txt" (one URL per line)
    """
    urls_env = (os.environ.get("SAGE_RAG_URLS") or "").strip()
    urls_file = (os.environ.get("SAGE_RAG_URLS_FILE") or "").strip()

    urls: List[str] = []
    if urls_file:
        p = os.path.abspath(urls_file)
        for line in open(p, "r", encoding="utf-8", errors="ignore").read().splitlines():
            u = line.strip()
            if u and not u.startswith("#"):
                urls.append(u)
    if urls_env:
        for part in urls_env.split(","):
            u = part.strip()
            if u:
                urls.append(u)

    # de-dupe preserving order
    seen = set()
    out: List[str] = []
    for u in urls:
        u2 = normalize_url(u)
        if u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out


def main() -> None:
    cfg = RagSettings()
    urls = load_url_list_from_env()
    if not urls:
        print("No URLs provided. Set SAGE_RAG_URLS or SAGE_RAG_URLS_FILE.")
        return

    embed = EmbedClient(cfg.embed_base_url, cfg.embed_model)
    commit_every = int(os.environ.get("SAGE_RAG_COMMIT_EVERY", "200"))

    total_pages = 0
    total_chunks = 0
    pending = 0

    with psycopg.connect(cfg.pg_dsn, row_factory=dict_row) as conn:  # type: ignore[arg-type]
        for url in urls:
            source_id = web_source_id(url)

            try:
                html = fetch_url(url, timeout_s=45.0)
            except Exception as e:
                print(f"!! fetch failed: {url} ({e})")
                continue

            doc_text = extract_main_text(html, url)
            if not doc_text:
                continue

            content_hash = sha256_text(doc_text)

            existing = get_existing_content_hash(conn, source_id)
            if existing and existing == content_hash:
                # unchanged; skip embed + write
                continue

            chunks = chunk_text(doc_text, cfg.chunk_chars, cfg.chunk_overlap)
            if not chunks:
                continue

            embs = embed.embed(chunks)

            meta = {
                "source_type": "web",
                "url": url,
                "host": urlparse(url).netloc,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "content_hash": content_hash,
            }

            n = upsert_chunks_executemany(conn, source_id=source_id, chunks=chunks, embeddings=embs, metadata=meta)
            total_pages += 1
            total_chunks += n
            pending += n

            if pending >= commit_every:
                conn.commit()
                pending = 0

            print(f"Upserted {n:4d} chunks | {url}")

        if pending > 0:
            conn.commit()

    print(f"Done. Pages upserted: {total_pages}, chunks upserted: {total_chunks}")


if __name__ == "__main__":
    main()
