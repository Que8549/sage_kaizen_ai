"""
rag_v1/wiki/wiki_retriever.py

Runtime retriever for the Wikipedia multimodal index.

Used by router.apply_wiki_rag() on every chat turn when wiki retrieval
is enabled.  On first call, auto-starts the jina-clip-v2 embed service
as a subprocess (if not already running) and registers atexit cleanup.

Configuration (wiki root, embed host/port) is read from
config/brains/brains.yaml (wiki_embed: section).
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import psycopg
from psycopg.rows import dict_row, DictRow

from rag_v1.db.pg import get_conn
from rag_v1.wiki.mm_embed_client import MmEmbedClient
from rag_v1.wiki.wiki_embed_config import load_wiki_embed_config
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.wiki_retriever")

# ──────────────────────────────────────────────────────────────────────────── #
# Result dataclasses                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class WikiChunk:
    chunk_id: int
    bundle_id: str
    title: str
    section_path: list[str] | None
    chunk_index: int
    text: str
    score: float       # 1 - cosine_distance  (higher = more similar)


@dataclass
class WikiImage:
    image_id: int
    bundle_id: str
    absolute_path: str    # full path for st.image() / display
    caption_text: str
    is_hero: bool
    hero_rank: int
    sim_score: float


@dataclass
class WikiSearchResult:
    chunks: list[WikiChunk] = field(default_factory=list)
    images: list[WikiImage] = field(default_factory=list)
    empty: bool = False


# ──────────────────────────────────────────────────────────────────────────── #
# SQL                                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

_SQL_TOP_CHUNKS = """
SELECT
    wc.chunk_id,
    wc.bundle_id::text,
    wc.title,
    wc.section_path,
    wc.chunk_index,
    wc.text,
    (wc.embedding <=> %s::vector) AS distance
FROM wiki_chunks wc
WHERE (wc.embedding <=> %s::vector) < %s
ORDER BY wc.embedding <=> %s::vector
LIMIT %s;
"""

_SQL_TOP_IMAGES = """
SELECT
    wi.image_id,
    wi.bundle_id::text,
    wi.relative_path,
    wi.caption_text,
    wi.is_hero,
    wi.hero_rank,
    GREATEST(
        1.0 - (wi.image_embedding   <=> %s::vector),
        1.0 - (wi.caption_embedding <=> %s::vector)
    ) AS sim_score
FROM wiki_images wi
WHERE wi.bundle_id = ANY(%s::uuid[])
ORDER BY wi.is_hero DESC, wi.hero_rank ASC, sim_score DESC
LIMIT %s;
"""


# ──────────────────────────────────────────────────────────────────────────── #
# WikiRetriever                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

class WikiRetriever:
    """
    Retrieves Wikipedia chunks and images for a user query.

    Wiki root, embed host, and embed port are loaded from
    config/brains/brains.yaml (wiki_embed: section).

    On first call to search(), auto-starts the embed service if it is not
    already running, and registers atexit cleanup.

    Gracefully returns WikiSearchResult(empty=True) on any failure so the
    chat pipeline is never blocked by wiki retrieval issues.
    """

    def __init__(
        self,
        pg_dsn: str,
        max_distance: float       = 0.40,
        cluster_min_size: int     = 3,
        cluster_max_spread: float = 0.030,
        cluster_top1_floor: float = 0.800,
    ) -> None:
        wiki_cfg = load_wiki_embed_config()

        self._pg_dsn              = pg_dsn
        self._wiki_root           = wiki_cfg.wiki_root
        self._embed_host          = wiki_cfg.host
        self._embed_port          = wiki_cfg.port
        self._startup_timeout_s   = wiki_cfg.startup_timeout_s   # from brains.yaml (300 s)
        self._embed_log           = wiki_cfg.log                  # for subprocess stderr
        self._max_distance        = max_distance
        self._cluster_min         = cluster_min_size
        self._cluster_spread      = cluster_max_spread
        self._cluster_floor       = cluster_top1_floor
        self._client              = MmEmbedClient(host=wiki_cfg.host, port=wiki_cfg.port)
        self._embed_proc: subprocess.Popen | None = None
        self._atexit_registered: bool = False

    # ------------------------------------------------------------------ #
    # Service lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_service(self) -> bool:
        """
        Auto-start embed service if not running.
        Returns True if the service is up and ready.
        """
        if self._client.ping(timeout_s=2.0):
            return True

        # Check if a previously started proc has died
        if self._embed_proc is not None and self._embed_proc.poll() is not None:
            _LOG.warning("Wiki embed service process exited unexpectedly (rc=%s).",
                         self._embed_proc.returncode)
            self._embed_proc = None

        _LOG.info(
            "Wiki embed service not detected — auto-starting on %s:%s …",
            self._embed_host, self._embed_port,
        )
        # Redirect stdout+stderr to the wiki embed log so startup errors
        # (model load failures, CUDA errors, import errors) are captured.
        self._embed_log.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(self._embed_log, "ab", buffering=0)
        # Forward WIKI_EMBED_VERBOSE to the subprocess so the verbosity setting
        # propagates from the parent process.  If the parent has not set it,
        # the subprocess defaults to quiet mode (no tqdm bars, no access logs).
        env = os.environ.copy()
        if "WIKI_EMBED_VERBOSE" not in env:
            env["WIKI_EMBED_VERBOSE"] = "0"
        try:
            # The service reads host/port/device from brains.yaml at startup.
            # cwd=_PROJECT_ROOT ensures rag_v1 is importable as a package.
            self._embed_proc = subprocess.Popen(
                [sys.executable, "-m", "rag_v1.wiki.mm_embed_service.app"],
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=str(_PROJECT_ROOT),
                env=env,
            )
        finally:
            log_fh.close()  # parent closes its copy; child keeps its own fd

        if not self._atexit_registered:
            atexit.register(self._shutdown_service)
            self._atexit_registered = True

        # Use startup_timeout_s from brains.yaml (300 s) — the service needs
        # model load + torch.compile warmup which can exceed 60 s on first run.
        deadline = time.monotonic() + self._startup_timeout_s
        while time.monotonic() < deadline:
            if self._client.ping(timeout_s=2.0):
                _LOG.info("Wiki embed service ready (port %s).", self._embed_port)
                return True
            time.sleep(1.0)

        _LOG.warning(
            "Wiki embed service did not start within %.0f s — "
            "wiki retrieval disabled for this query.",
            self._startup_timeout_s,
        )
        return False

    def _shutdown_service(self) -> None:
        if self._embed_proc and self._embed_proc.poll() is None:
            try:
                self._embed_proc.terminate()
                self._embed_proc.wait(timeout=5)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k_chunks: int = 6,
        top_images: int   = 3,
    ) -> WikiSearchResult:
        """
        Embed the query with jina-clip-v2, retrieve top wiki chunks and images.

        Returns WikiSearchResult(empty=True) when:
          - embed service is unreachable
          - no chunks pass the distance threshold
          - noise-cluster gate fires (all chunks cluster tightly at mediocre scores)
          - any unexpected error
        """
        if not self._ensure_service():
            return WikiSearchResult(empty=True)

        try:
            qvec = self._client.embed_text([query])[0]
        except Exception:
            _LOG.exception("Wiki embed_text failed for query")
            return WikiSearchResult(empty=True)

        try:
            chunks = self._get_chunks(qvec, top_k_chunks)
        except Exception:
            _LOG.exception("Wiki chunk retrieval failed")
            return WikiSearchResult(empty=True)

        if not chunks:
            return WikiSearchResult(empty=True)

        # Noise-cluster gate — same logic as RagPipeline
        scores = [c.score for c in chunks]
        if (len(chunks) >= self._cluster_min
                and max(scores) - min(scores) < self._cluster_spread
                and max(scores) < self._cluster_floor):
            _LOG.info(
                "Wiki noise-cluster gate: rejected %d chunks "
                "(spread=%.4f, top1=%.4f)",
                len(chunks), max(scores) - min(scores), max(scores),
            )
            return WikiSearchResult(empty=True)

        # Deduplicated top bundle IDs (preserving chunk rank order)
        bundle_ids = list(dict.fromkeys(c.bundle_id for c in chunks))[:3]

        try:
            images = self._get_images(qvec, bundle_ids, top_images)
        except Exception:
            _LOG.exception("Wiki image retrieval failed; returning chunks only")
            images = []

        return WikiSearchResult(chunks=chunks, images=images)

    # ------------------------------------------------------------------ #
    # Private SQL helpers                                                  #
    # ------------------------------------------------------------------ #

    def _get_chunks(self, qvec: list[float], top_k: int) -> list[WikiChunk]:
        with get_conn(self._pg_dsn) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    _SQL_TOP_CHUNKS,
                    (qvec, qvec, self._max_distance, qvec, top_k),
                ).fetchall()

        return [
            WikiChunk(
                chunk_id     = row["chunk_id"],
                bundle_id    = row["bundle_id"],
                title        = row["title"],
                section_path = list(row["section_path"]) if row["section_path"] else None,
                chunk_index  = row["chunk_index"],
                text         = row["text"],
                score        = float(1.0 - row["distance"]),
            )
            for row in rows
        ]

    def _get_images(
        self,
        qvec: list[float],
        bundle_ids: list[str],
        top_images: int,
    ) -> list[WikiImage]:
        if not bundle_ids:
            return []

        with get_conn(self._pg_dsn) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    _SQL_TOP_IMAGES,
                    (qvec, qvec, bundle_ids, top_images),
                ).fetchall()

        images: list[WikiImage] = []
        for row in rows:
            rel = row["relative_path"].replace("/", os.sep)
            abs_path = str(self._wiki_root / rel)
            images.append(WikiImage(
                image_id      = row["image_id"],
                bundle_id     = row["bundle_id"],
                absolute_path = abs_path,
                caption_text  = row["caption_text"],
                is_hero       = bool(row["is_hero"]),
                hero_rank     = int(row["hero_rank"]),
                sim_score     = float(row["sim_score"]),
            ))
        return images
