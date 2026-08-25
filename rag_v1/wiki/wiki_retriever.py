"""
rag_v1/wiki/wiki_retriever.py

Runtime retriever for the Wikipedia multimodal index.

Used by apply_rag_and_wiki_parallel() (context_injector.py) on every chat
turn when wiki retrieval is enabled.  On first call, auto-starts the jina-clip-v2 embed service
as a subprocess (if not already running) and registers atexit cleanup.

Configuration (wiki root, embed host/port) is read from
config/brains/brains.yaml (wiki_embed: section).
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from psycopg import sql
from psycopg.rows import dict_row

from rag_v1.db.pg import conn_ctx
from rag_v1.db.vector_index import (
    DEFAULT_HNSW_EF_SEARCH,
    VECTOR_INDEX_AMS,
    WIKI_EMBED_DIMS,
    WIKI_IVFFLAT_PROBES,
    apply_vector_tuning,
)
from rag_v1.wiki.mm_embed_client import MmEmbedClient
from rag_v1.wiki.wiki_embed_config import load_wiki_embed_config
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.wiki_retriever")

# ──────────────────────────────────────────────────────────────────────────── #
# Display-GPU guard                                                              #
# ──────────────────────────────────────────────────────────────────────────── #
# cuda:0 is the RTX 5090 at PCI 01:00.0 that drives the three monitors.  It is
# display-only: cuda:1 (RTX 5090 OC) and cuda:2 (RTX 5080 eGPU) are the compute
# GPUs.  Sustained CUDA work on the display card is the documented Windows TDR /
# display-driver-reset trigger, and a PnP-level fault there blanks the desktop.
#
# sage_kaizen_ai_ingest closed the equivalent holes on its side (its CLAUDE.md
# §19).  This module is the main app's own path onto a GPU — WikiRetriever
# spawns mm_embed_service at *chat* time, never passing through wiki_ingest.py's
# guards — so it needs its own.  Two ways it could previously have landed on
# cuda:0, both closed below:
#
#   1. WikiEmbedServiceConfig.device defaulted to "cuda:0" (fixed 2026-08-04),
#      and _start of the service falls back to it when brains.yaml is silent.
#   2. _ensure_service() passed os.environ.copy() straight through, so a
#      WIKI_EMBED_DEVICE=cuda:0 inherited from any parent shell won — the
#      service reads `os.environ.get("WIKI_EMBED_DEVICE") or cfg.device`.
#
# Named constants rather than inline literals: several guards depend on these
# and a wrong value silently disables all of them.
_DISPLAY_GPU_DEVICE = "cuda:0"
_DISPLAY_GPU_INDEX = 0


# ──────────────────────────────────────────────────────────────────────────── #
# Vector-index guard                                                             #
# ──────────────────────────────────────────────────────────────────────────── #
# wiki_chunks is ~3.5 TB / 508M rows. Without an ANN index on `embedding`, the
# ORDER BY <=> below is a full sequential scan of that table, and every row pays
# detoasting because a 1024-dim vector is 4104 bytes and therefore always
# TOASTed.
#
# This is not hypothetical and not rare: sage_kaizen_ai_ingest's
# `--manage-indexes` DROPS the HNSW index for the duration of a bulk ingest and
# rebuilds it afterwards, so the index is absent for as long as an ingest run
# lasts — days, currently.
#
# Measured 2026-08-06, with the index absent: one search() call had not returned
# after 15 minutes. Worse, context_injector collects its five workers under a
# single shared 30 s deadline, so the caller walks away at 30 s while Postgres
# keeps scanning — three parallel backends were still burning disk I/O 24
# minutes later and had to be killed with pg_cancel_backend(). Every wiki-routed
# turn would start another one.
#
# So the failure mode was: wiki-RAG contributes nothing (looks exactly like "no
# hits"), while quietly saturating the disk the whole app and the concurrent
# ingest share. Fail fast and say so instead.
# Defined once in rag_v1/db/vector_index.py — see that module for why these
# values are what they are. Local aliases keep this file readable.
_WIKI_VECTOR_INDEX_AM = VECTOR_INDEX_AMS

# jina-clip-v2 output dimensionality. Needed literally in the halfvec cast,
# because the cast in the query must match the index expression exactly.
_WIKI_EMBED_DIMS = WIKI_EMBED_DIMS

# NOT sqrt(lists) — wiki_chunks is 32 partitions and a nearest-neighbour query
# cannot prune them, so the cost multiplies by 32. The cold-page measurements
# that produced 5 are recorded in rag_v1/db/vector_index.py.
_IVFFLAT_PROBES = WIKI_IVFFLAT_PROBES

# How long to wait before re-checking after finding no index. A rebuild takes
# days, so polling the catalog often is pointless — but a fixed negative cache
# would keep wiki-RAG dark until the app restarts, which is worse.
_INDEX_RECHECK_INTERVAL_S = 300.0

# Backstop for the vector query itself. The index guard covers the known cause,
# but any future plan regression that turns this into a seq scan must not be
# able to outlive its caller again. Slightly under context_injector's 30 s
# shared deadline so Postgres gives up before the consumer does, rather than
# leaving an orphaned scan behind.
_WIKI_QUERY_TIMEOUT_MS = 25_000


class DisplayGpuRefused(RuntimeError):
    """Raised when wiki retrieval would place inference on the display GPU."""


def _is_display_gpu(device: str) -> bool:
    """True when `device` names the display GPU (cuda:0), tolerating whitespace/case."""
    return (device or "").strip().lower() == _DISPLAY_GPU_DEVICE


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

# Two variants, because pgvector will only use a halfvec index when the query
# casts EXACTLY as the index expression does ("Get the nearest neighbors:
# SELECT * FROM items ORDER BY embedding::halfvec(3) <-> ..." — pgvector README).
# Query the wrong one and the index is silently ignored: the planner falls back
# to a sequential scan of 3.5 TB, which is the failure this whole guard exists
# to prevent — except now with an index present to make it look fine.
# _vector_index_ready() detects which exists and search() picks the match.
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
ORDER BY wc.embedding <=> %s::vector
LIMIT %s;
"""

# The dimension is written out rather than interpolated so this stays a
# LiteralString — psycopg only accepts those as a bare query, which is the type
# system enforcing its injection guard. test_wiki_retriever asserts it matches
# _WIKI_EMBED_DIMS, so the two cannot drift apart silently.
_SQL_TOP_CHUNKS_HALFVEC = """
SELECT
    wc.chunk_id,
    wc.bundle_id::text,
    wc.title,
    wc.section_path,
    wc.chunk_index,
    wc.text,
    (wc.embedding::halfvec(1024) <=> %s::halfvec(1024)) AS distance
FROM wiki_chunks wc
ORDER BY wc.embedding::halfvec(1024) <=> %s::halfvec(1024)
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
        allow_display_gpu: bool   = False,
    ) -> None:
        wiki_cfg = load_wiki_embed_config()

        self._pg_dsn              = pg_dsn
        self._wiki_root           = wiki_cfg.wiki_root
        self._embed_host          = wiki_cfg.host
        self._embed_port          = wiki_cfg.port
        self._startup_timeout_s   = wiki_cfg.startup_timeout_s   # from brains.yaml (300 s)
        self._embed_log           = wiki_cfg.log                  # for subprocess stderr
        self._config_device       = wiki_cfg.device               # brains.yaml fallback
        self._max_distance        = max_distance
        self._cluster_min         = cluster_min_size
        self._cluster_spread      = cluster_max_spread
        self._cluster_floor       = cluster_top1_floor
        self._client              = MmEmbedClient(host=wiki_cfg.host, port=wiki_cfg.port)
        self._embed_proc: subprocess.Popen | None = None
        self._atexit_registered: bool = False
        self._warmed_up: bool = False
        # Opt-in consent to run on the display GPU.  Nothing in the chat path
        # passes this; it exists so a deliberate operator override is possible
        # without widening the guard for everyone.
        self._allow_display_gpu = allow_display_gpu
        # Serialises _ensure_service().  Without it two threads from the
        # context_injector pool could each miss the ping and each spawn an
        # mm_embed_service, putting two cold torch/CUDA initialisations on the
        # same physical GPU simultaneously — a failure mode the ingest project
        # hit twice (its CLAUDE.md §15, 2026-05-28 and 2026-07-18) and had to
        # serialise service startup to stop.
        self._service_lock = threading.Lock()
        # Vector-index guard state. None = not yet checked.
        self._index_present: bool | None = None
        # "halfvec" | "vector" | None — decides which SQL variant search() emits.
        self._index_kind: str | None = None
        # "ivfflat" | "hnsw" — decides which query-time tuning GUC to set.
        self._index_am: str = "hnsw"
        self._index_checked_at: float = 0.0
        self._index_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Vector-index guard                                                   #
    # ------------------------------------------------------------------ #

    def _query_index_kind(self) -> str | None:
        """
        Ask the catalog which ANN index exists on wiki_chunks' embedding, if any.

        Returns "halfvec", "vector", or None.

        Matched by access method rather than by index name: ingest creates
        `hnsw_wiki_chunks_embedding_cos`, but a hand-built index, a rename, or a
        switch to ivfflat would all still be usable, and a name check would call
        those missing.

        The definition is read with pg_get_indexdef() rather than by joining
        pg_attribute on indkey. An EXPRESSION index — which is exactly what
        `(embedding::halfvec(1024))` is — stores 0 in indkey for the expression
        column, so an attribute join finds nothing and would report the halfvec
        index as missing. That is the index the migration builds.

        The pg_inherits join matters just as much. wiki_chunks is now HASH
        partitioned, and the ivfflat indexes were built on each PARTITION
        individually — there is no parent-level ivfflat index to find, because
        CREATE INDEX CONCURRENTLY is unsupported on partitioned tables and the
        migration therefore builds per partition. Matching only
        `tbl.relname = 'wiki_chunks'` found nothing and reported wiki-RAG
        disabled with every index in place; caught by the post-swap end-to-end
        check on 2026-08-24. Both shapes are matched so this works before and
        after the swap.
        """
        with conn_ctx(self._pg_dsn) as conn, conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(
                """
                SELECT pg_get_indexdef(i.indexrelid) AS def
                FROM pg_index i
                JOIN pg_class idx    ON idx.oid = i.indexrelid
                JOIN pg_class tbl    ON tbl.oid = i.indrelid
                JOIN pg_am    am     ON am.oid  = idx.relam
                LEFT JOIN pg_inherits inh   ON inh.inhrelid = tbl.oid
                LEFT JOIN pg_class    parent ON parent.oid  = inh.inhparent
                WHERE (tbl.relname = 'wiki_chunks' OR parent.relname = 'wiki_chunks')
                  AND am.amname   = ANY(%s)
                  AND i.indisvalid
                """,
                (list(_WIKI_VECTOR_INDEX_AM),),
            ).fetchall()

        defs = [r["def"] for r in rows if r["def"] and "embedding" in r["def"]]
        if not defs:
            return None

        # Record the access method too: ivfflat and hnsw need DIFFERENT
        # query-time tuning GUCs, and setting the wrong one is silent. An
        # ivfflat scan left at its default probes = 1 examines a single list
        # and returns almost nothing — it looks like "no matches", not an error.
        self._index_am = "ivfflat" if any("ivfflat" in d for d in defs) else "hnsw"

        # Prefer halfvec when both exist: it is the cheaper probe, and a
        # migration that has built halfvec is the intended path.
        if any("halfvec" in d for d in defs):
            return "halfvec"
        return "vector"

    def _vector_index_ready(self) -> bool:
        """
        True when the wiki vector index exists; False (loudly, once) when not.

        `indisvalid` is part of the check on purpose: the rebuild uses CREATE
        INDEX CONCURRENTLY, which leaves an INVALID index behind if it fails or
        is interrupted. An invalid index is not used by the planner, so treating
        its mere presence as "ready" would reintroduce the full scan.

        On a catalog error this returns False without caching — the database is
        probably unreachable, in which case the search would fail anyway, and
        guessing "present" would risk the very scan this exists to prevent.
        """
        now = time.monotonic()
        with self._index_lock:
            if self._index_present is True:
                return True
            if (self._index_present is False
                    and now - self._index_checked_at < _INDEX_RECHECK_INTERVAL_S):
                return False

            try:
                kind = self._query_index_kind()
            except Exception:
                _LOG.exception(
                    "Wiki vector-index check failed; skipping wiki retrieval for "
                    "this turn rather than risking a full scan of wiki_chunks"
                )
                return False

            present = kind is not None
            self._index_checked_at = now
            first_answer = self._index_present is None
            changed = self._index_present is not None and self._index_present != present
            self._index_present = present
            self._index_kind = kind

            if present:
                if first_answer or changed:
                    _LOG.info(
                        "Wiki vector index present (%s) — wiki retrieval enabled.", kind
                    )
            elif first_answer or changed:
                # Logged at most once per transition, not once per turn: this is
                # a days-long steady state during an ingest run, not an incident.
                _LOG.warning(
                    "Wiki retrieval DISABLED: no valid hnsw/ivfflat index on "
                    "wiki_chunks.embedding. Without it each query is a full scan "
                    "of a ~3.5 TB table. sage_kaizen_ai_ingest drops this index "
                    "for the duration of a bulk ingest (--manage-indexes) and "
                    "rebuilds it afterwards, so this is expected mid-ingest. "
                    "Re-checking every %.0f s. See CLAUDE.md §17.",
                    _INDEX_RECHECK_INTERVAL_S,
                )
            return present

    # ------------------------------------------------------------------ #
    # Display-GPU guard                                                    #
    # ------------------------------------------------------------------ #

    def _effective_device(self) -> tuple[str, str]:
        """
        Resolve the device the service *will* load on, and where that came from.

        Mirrors mm_embed_service/app.py's own precedence exactly:
            os.environ["WIKI_EMBED_DEVICE"] or cfg.device
        Checking the effective value (rather than just the config) is what makes
        the guard hold even when a config default regresses or an env var leaks
        in from a parent shell.
        """
        env_device = os.environ.get("WIKI_EMBED_DEVICE")
        if env_device:
            return env_device.strip(), "WIKI_EMBED_DEVICE env var"
        return self._config_device, "brains.yaml wiki_embed.service.device"

    def _assert_device_allowed(self, device: str, source: str) -> None:
        """Raise DisplayGpuRefused if `device` is the display GPU without consent."""
        if _is_display_gpu(device) and not self._allow_display_gpu:
            raise DisplayGpuRefused(
                f"Refusing to run wiki embed inference on {device} (the display GPU, "
                f"index {_DISPLAY_GPU_INDEX}) — requested via {source}. "
                f"cuda:0 drives the monitors; sustained CUDA work there resets the "
                f"display driver. Set the device to cuda:1, or construct "
                f"WikiRetriever(allow_display_gpu=True) if this is deliberate."
            )

    # ------------------------------------------------------------------ #
    # Service lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def _maybe_warmup(self) -> None:
        """Send one dummy embed to absorb torch.compile/CUDA JIT on first forward pass."""
        if self._warmed_up:
            return
        try:
            self._client.embed_text(["warmup"])
            _LOG.info("Wiki embed service warmed up (port %s).", self._embed_port)
        except Exception:
            _LOG.warning("Wiki embed warmup failed; first real query may be slower")
        self._warmed_up = True

    def _check_running_service_device(self, health: dict) -> bool:
        """
        Validate the device reported by an already-running service.

        Returns True if it is safe to use.  Raises DisplayGpuRefused when the
        model is loaded on the display GPU without consent.

        A bare ping only proves that *something* answered on the port — never
        where the model actually is.  This is the main app's equivalent of the
        ingest project's §19 gap 2: a service started by any other path (an
        ingest session, a manual `python -m …`, a stale process from before a
        config fix) could be sitting on cuda:0 with nothing in our logs.
        """
        device = str(health.get("device") or "").strip()
        if not device:
            # Older service build with no `device` key. Nothing to check
            # against; don't fail the turn over a missing diagnostic field.
            _LOG.debug("Wiki embed /health reported no device field — device check skipped.")
            return True

        if _is_display_gpu(device) and not self._allow_display_gpu:
            raise DisplayGpuRefused(
                f"The wiki embed service already listening on port {self._embed_port} "
                f"has jina-clip-v2 loaded on {device} — the display GPU (index "
                f"{_DISPLAY_GPU_INDEX}, drives the monitors). This process did not "
                f"start it. Stop that service and restart it on cuda:1, or construct "
                f"WikiRetriever(allow_display_gpu=True) if this is deliberate."
            )

        expected, _ = self._effective_device()
        if device != expected:
            # Not the display GPU, just not what we'd have chosen. A deliberate
            # manual override elsewhere is unusual but legitimate, and this
            # process didn't start the service — so it doesn't get a veto.
            _LOG.warning(
                "Wiki embed service on port %s is on %s, expected %s — "
                "continuing (service was not started by this process).",
                self._embed_port, device, expected,
            )
        return True

    def _ensure_service(self) -> bool:
        """
        Auto-start embed service if not running.
        Returns True if the service is up and ready.

        Raises DisplayGpuRefused if the service is, or would be, running
        inference on the display GPU. Callers in the chat path
        (context_injector._fetch_wiki_result) already catch every exception and
        degrade to no wiki context, so this fails loudly in the logs without
        breaking the turn.

        Serialised by _service_lock so concurrent chat turns cannot each decide
        the service is missing and each spawn one.
        """
        with self._service_lock:
            return self._ensure_service_locked()

    def _ensure_service_locked(self) -> bool:
        health = self._client.health(timeout_s=2.0)
        if health is not None:
            self._check_running_service_device(health)
            self._maybe_warmup()
            return True

        # Check if a previously started proc has died
        if self._embed_proc is not None and self._embed_proc.poll() is not None:
            _LOG.warning("Wiki embed service process exited unexpectedly (rc=%s).",
                         self._embed_proc.returncode)
            self._embed_proc = None

        # Guard BEFORE spawning: refuse to create the process at all rather
        # than starting it and discovering the device afterwards.
        device, source = self._effective_device()
        self._assert_device_allowed(device, source)

        _LOG.info(
            "Wiki embed service not detected — auto-starting on %s:%s (device=%s, via %s) …",
            self._embed_host, self._embed_port, device, source,
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
        # Pin the device explicitly rather than letting the child re-resolve it.
        # The child reads `os.environ.get("WIKI_EMBED_DEVICE") or cfg.device`, so
        # passing os.environ through unmodified meant an inherited
        # WIKI_EMBED_DEVICE silently beat brains.yaml. We already validated
        # `device` above; writing it back makes the value we checked the value
        # the child uses, with no second resolution step to disagree with.
        env["WIKI_EMBED_DEVICE"] = device
        try:
            # The service reads host/port/batch sizes from brains.yaml at startup.
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
            health = self._client.health(timeout_s=2.0)
            if health is not None:
                # Verify what we spawned actually landed where we told it to.
                self._check_running_service_device(health)
                _LOG.info(
                    "Wiki embed service ready (port %s, device %s).",
                    self._embed_port, health.get("device", "unknown"),
                )
                self._maybe_warmup()
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
                self._embed_proc.wait(timeout=3)
            except Exception:
                try:
                    self._embed_proc.kill()
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
          - the wiki vector index is missing (see _vector_index_ready)
          - embed service is unreachable
          - no chunks pass the distance threshold
          - noise-cluster gate fires (all chunks cluster tightly at mediocre scores)
          - any unexpected error
        """
        # Checked BEFORE _ensure_service() deliberately. Starting the embed
        # service costs ~3.2 GB of VRAM and up to 90 s of cold torch.compile
        # warmup; doing that to serve a query that cannot run is pure waste, and
        # it would occupy the GPU that a concurrent ingest is using.
        if not self._vector_index_ready():
            return WikiSearchResult(empty=True)

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
        with conn_ctx(self._pg_dsn) as conn:
            # One definition of the recall knob and the timeout, shared with
            # every other pgvector caller. Picking the wrong knob is silent:
            # an ivfflat scan left at the default probes = 1 examines a single
            # list out of 4000 and looks exactly like "no matches".
            apply_vector_tuning(
                conn,
                index_am=self._index_am,
                probes=_IVFFLAT_PROBES,
                ef_search=DEFAULT_HNSW_EF_SEARCH,
                timeout_ms=_WIKI_QUERY_TIMEOUT_MS,
            )
            chunk_sql = (_SQL_TOP_CHUNKS_HALFVEC if self._index_kind == "halfvec"
                         else _SQL_TOP_CHUNKS)
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    chunk_sql,
                    (qvec, qvec, top_k),
                ).fetchall()
        # Filter by distance threshold in Python; keeps the HNSW index path clean
        rows = [r for r in rows if float(r["distance"]) < self._max_distance]

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

        with conn_ctx(self._pg_dsn) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    _SQL_TOP_IMAGES,
                    (qvec, qvec, bundle_ids, top_images),
                ).fetchall()

        if not rows:
            _LOG.debug("Wiki image query returned 0 rows for %d bundle_id(s)", len(bundle_ids))
            return []

        images: list[WikiImage] = []
        for row in rows:
            rel = row["relative_path"].replace("/", os.sep)
            abs_path = str(self._wiki_root / rel)
            if not os.path.isfile(abs_path):
                _LOG.warning("Wiki image file missing on disk: %s", abs_path)
                continue
            images.append(WikiImage(
                image_id      = row["image_id"],
                bundle_id     = row["bundle_id"],
                absolute_path = abs_path,
                caption_text  = row["caption_text"],
                is_hero       = bool(row["is_hero"]),
                hero_rank     = int(row["hero_rank"]),
                sim_score     = float(row["sim_score"]) if row["sim_score"] is not None else 0.0,
            ))
        _LOG.debug("Wiki image retrieval: %d/%d images valid on disk", len(images), len(rows))
        return images
