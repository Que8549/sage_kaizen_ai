from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse

import psycopg
from psycopg.rows import dict_row

from rag_v1.embed.embed_client import EmbedClient
from rag_v1.config.rag_settings import RagSettings

from rag_v1.ingest.ingest_utils import (
    sha256_text,
    rss_item_source_id,
    strip_html,
    chunk_text,
    get_existing_content_hash,
    upsert_chunks_executemany,
    absolutize_url,
)
from rag_v1.ingest.ingest_runtime import load_list_from_env_and_file, CommitBatcher

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore


@dataclass
class FeedItem:
    title: str
    link: str
    guid: str
    published: str
    summary: str
    content: str


def _fetch_url(url: str, timeout_s: float = 20.0) -> str:
    if httpx is None:
        from urllib.request import Request, urlopen
        req = Request(url, headers={"User-Agent": "SageKaizenRAG/1.0"})
        with urlopen(req, timeout=timeout_s) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    else:
        with httpx.Client(timeout=timeout_s, headers={"User-Agent": "SageKaizenRAG/1.0"}) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.text


def parse_feed_xml(feed_url: str, xml_text: str) -> List[FeedItem]:
    import xml.etree.ElementTree as ET

    items: List[FeedItem] = []
    root = ET.fromstring(xml_text)

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "content": "http://purl.org/rss/1.0/modules/content/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }

    # Atom entries
    for entry in root.findall(".//atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        link_el = entry.find("atom:link", ns)
        link = ""
        if link_el is not None:
            link = link_el.attrib.get("href", "") or ""
        guid = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        content = (entry.findtext("atom:content", default="", namespaces=ns) or "").strip()

        if link:
            link = absolutize_url(feed_url, link)

        items.append(FeedItem(title=title, link=link, guid=guid, published=published, summary=summary, content=content))

    # RSS items
    for it in root.findall(".//item"):
        title = (it.findtext("title", default="") or "").strip()
        link = (it.findtext("link", default="") or "").strip()
        guid = (it.findtext("guid", default="") or "").strip()
        published = (it.findtext("pubDate", default="") or "").strip()

        content_encoded = ""
        ce = it.find("{http://purl.org/rss/1.0/modules/content/}encoded")
        if ce is not None and ce.text:
            content_encoded = ce.text

        summary = (it.findtext("description", default="") or "").strip()

        if link:
            link = absolutize_url(feed_url, link)

        items.append(
            FeedItem(
                title=title,
                link=link,
                guid=guid,
                published=published,
                summary=summary,
                content=content_encoded or "",
            )
        )

    return items


def main() -> None:
    cfg = RagSettings()

    feeds = load_list_from_env_and_file(
        env_csv_var="SAGE_RAG_RSS_FEEDS",
        env_file_var="SAGE_RAG_RSS_FILE",
        normalize=None,
    )
    if not feeds:
        print("No feeds provided. Set SAGE_RAG_RSS_FEEDS or SAGE_RAG_RSS_FILE.")
        return

    embed = EmbedClient(cfg.embed_base_url, cfg.embed_model)

    max_items = int(os.environ.get("SAGE_RAG_RSS_MAX_ITEMS", "20"))
    fetch_article_pages = (os.environ.get("SAGE_RAG_RSS_FETCH_ARTICLE", "0").strip().lower() in ("1", "true", "yes", "on"))

    commit_every = int(os.environ.get("SAGE_RAG_COMMIT_EVERY", "200"))
    batcher = CommitBatcher(commit_every=commit_every)

    total_chunks = 0
    total_items = 0

    with psycopg.connect(cfg.pg_dsn, row_factory=dict_row) as conn:  # type: ignore[arg-type]
        for feed_url in feeds:
            print(f"\n== Feed: {feed_url}")
            try:
                xml_text = _fetch_url(feed_url, timeout_s=30.0)
            except Exception as e:
                print(f"  !! failed to fetch feed: {e}")
                continue

            try:
                items = parse_feed_xml(feed_url, xml_text)
            except Exception as e:
                print(f"  !! failed to parse feed: {e}")
                continue

            for item in items[:max_items]:
                item_key = item.guid or item.link or item.title
                source_id = rss_item_source_id(item_key)

                doc_parts: List[str] = []
                if item.title:
                    doc_parts.append(item.title.strip())
                if item.published:
                    doc_parts.append(f"Published: {item.published.strip()}")
                if item.link:
                    doc_parts.append(f"URL: {item.link.strip()}")

                body = item.content or item.summary or ""
                body_txt = strip_html(body)

                if fetch_article_pages and item.link:
                    try:
                        page_html = _fetch_url(item.link, timeout_s=30.0)
                        page_txt = strip_html(page_html)
                        if len(page_txt) > len(body_txt) * 1.2:
                            body_txt = page_txt
                    except Exception:
                        pass

                if body_txt:
                    doc_parts.append(body_txt)

                doc_text = "\n\n".join(doc_parts).strip()
                if not doc_text:
                    continue

                content_hash = sha256_text(doc_text)

                existing = get_existing_content_hash(conn, source_id)
                if existing and existing == content_hash:
                    continue

                chunks = chunk_text(doc_text, cfg.chunk_chars, cfg.chunk_overlap)
                if not chunks:
                    continue

                embs = embed.embed(chunks)

                meta = {
                    "source_type": "rss",
                    "feed_url": feed_url,
                    "title": item.title,
                    "url": item.link,
                    "guid": item.guid,
                    "published": item.published,
                    "host": urlparse(item.link).netloc if item.link else "",
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "content_hash": content_hash,
                }

                n = upsert_chunks_executemany(conn, source_id=source_id, chunks=chunks, embeddings=embs, metadata=meta)
                total_chunks += n
                total_items += 1

                batcher.add(n)
                if batcher.should_commit():
                    conn.commit()
                    batcher.reset()

                print(f"  Upserted {n:4d} chunks | {item.title[:80] if item.title else source_id}")

        batcher.commit_if_needed(conn)

    print(f"\nDone. Items upserted: {total_items}, chunks upserted: {total_chunks}")


if __name__ == "__main__":
    main()
