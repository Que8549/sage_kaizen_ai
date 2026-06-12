"""
memory/service.py
MemoryService facade — the single entry point for all memory operations.

Usage in router.py (hot path):
    from memory.service import MemoryService
    _MEMORY = MemoryService()

    bundle = _MEMORY.get_memory_bundle(request)
    prompt_segment = bundle_builder.format_bundle_prompt(bundle)

Usage post-turn (background / end of handler):
    _MEMORY.write_episode(episode_req)

Usage in maintenance runner:
    result = _MEMORY.run_reflection(user_id="alquin", session_id=sid)
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as _FutureTimeout

from sk_logging import get_logger
from .bundle_builder import build_bundle, format_bundle_prompt
from .consolidator import run_reflection
from .models import (
    EpisodeWriteRequest,
    MemoryBundle,
    MemoryContextRequest,
    ProfileWriteRequest,
    PromotionDecision,
    ReflectionResult,
)
from .retriever import retrieve_episodes, retrieve_profiles, retrieve_rules
from .writer import write_episode, write_explicit_profile

_LOG = get_logger("sage_kaizen.memory.service")

# Per-brain token caps (04-Memory_Service.md)
_FAST_MAX_TOKENS      = 600
_ARCHITECT_MAX_TOKENS = 1500

# Module-level pool: avoids 3-thread creation/teardown (~3–15 ms on Windows) on
# every chat turn.  max_workers=3 matches the three parallel memory fetches.
_POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="memory-")


class MemoryService:
    """
    Facade over retrieval, writing, and consolidation.

    Thread-safe: the underlying connection pool (psycopg3 ConnectionPool)
    handles concurrent access.  A single shared instance is safe to use
    from the Streamlit session thread and any background daemon threads.
    """

    def get_memory_bundle(self, request: MemoryContextRequest) -> MemoryBundle:
        """
        Retrieve a token-budgeted MemoryBundle for the current turn.

        Called by router.py between route selection and prompt assembly.
        """
        t0 = time.monotonic()

        # Resolve brain-specific default; caller override takes precedence when set.
        default_tokens = (
            _ARCHITECT_MAX_TOKENS
            if request.route_target == "architect"
            else _FAST_MAX_TOKENS
        )
        max_tokens = request.max_bundle_tokens if request.max_bundle_tokens is not None else default_tokens

        # Fetch all three memory sources in parallel — psycopg3 ConnectionPool
        # is thread-safe and each function acquires its own connection.
        _f_profiles: Future = _POOL.submit(
            retrieve_profiles, request.user_id, request.project_id
        )
        _f_rules: Future = _POOL.submit(
            retrieve_rules,
            user_id=request.user_id,
            project_id=request.project_id,
            query_text=request.query_text,
            limit=6,
        )
        _f_episodes: Future = _POOL.submit(
            retrieve_episodes,
            user_id=request.user_id,
            project_id=request.project_id,
            query_text=request.query_text,
            top_k=8,
        )
        try:
            profiles = _f_profiles.result(timeout=2.0)
        except (_FutureTimeout, Exception):
            _LOG.warning("memory | profiles retrieval timed out or failed; using empty list")
            profiles = []
        try:
            rules = _f_rules.result(timeout=2.0)
        except (_FutureTimeout, Exception):
            _LOG.warning("memory | rules retrieval timed out or failed; using empty list")
            rules = []
        try:
            episodes = _f_episodes.result(timeout=2.0)
        except (_FutureTimeout, Exception):
            _LOG.warning("memory | episodes retrieval timed out or failed; using empty list")
            episodes = []

        bundle = build_bundle(
            profiles=profiles,
            rules=rules,
            episodes=episodes,
            max_tokens=max_tokens,
        )

        latency_ms = (time.monotonic() - t0) * 1000
        _LOG.info(
            "service | bundle: profiles=%d rules=%d episodes=%d tokens=%d/%d "
            "truncated=%s latency_ms=%.1f user=%s",
            len(bundle.profiles), len(bundle.rules), len(bundle.episodes),
            bundle.estimated_tokens, max_tokens, bundle.was_truncated,
            latency_ms, request.user_id,
        )
        return bundle

    def format_bundle(self, bundle: MemoryBundle) -> str:
        """Format a bundle into the prompt segment string."""
        return format_bundle_prompt(bundle)

    def write_episode(self, req: EpisodeWriteRequest) -> str | None:
        """
        Write a post-turn episode (Path B — selective).
        Returns the new episode id, or None if policy skipped the write.
        """
        return write_episode(req)

    def write_explicit_profile(self, req: ProfileWriteRequest) -> str:
        """Write an explicit user preference to profile memory (Path A)."""
        return write_explicit_profile(req)

    def run_reflection(
        self,
        user_id: str,
        project_id: str = "sage_kaizen",
        session_id: str | None = None,
        mode: str = "deep",
        lookback_hours: int = 24,
    ) -> ReflectionResult:
        """
        Run a consolidation pass (Path C + D).
        mode="lightweight" → Fast brain, ~1s
        mode="deep"        → Architect brain, ~30–120s
        """
        return run_reflection(
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            mode=mode,
            lookback_hours=lookback_hours,
        )

    def explain_bundle(self, bundle: MemoryBundle) -> str:
        """Return a human-readable explanation of why each item was included."""
        lines: list[str] = [
            f"Memory bundle — {bundle.total_items} items, "
            f"~{bundle.estimated_tokens} tokens "
            f"({'truncated' if bundle.was_truncated else 'within budget'}):\n"
        ]
        for p in bundle.profiles:
            lines.append(f"  [PROFILE] {p.key} (scope={p.scope}, confidence={p.confidence:.2f})")
        for r in bundle.rules:
            lock = " LOCKED" if r.is_locked else ""
            lines.append(f"  [RULE{lock}] {r.rule_text[:80]} (confidence={r.confidence:.2f})")
        for e in bundle.episodes:
            correction = " CORRECTION" if e.was_user_correction else ""
            lines.append(
                f"  [EPISODE{correction}] {e.summary_text[:80]} "
                f"(score={e.retrieval_score:.3f}, importance={e.importance:.2f})"
            )
        return "\n".join(lines)
