"""
memory/retriever.py
Hybrid memory retrieval: metadata filter → FTS → HNSW → RRF → scoring → top-k.

pgvector 0.8.x note:
  All filtered vector queries use SET LOCAL hnsw.iterative_scan = relaxed_order
  (applied inside repository.fetch_episodes_vector) to prevent silent under-
  retrieval when user_id / project_id filters are highly selective.
"""
from __future__ import annotations

import time

from sk_logging import get_logger
from .embedder import embed_one
from .models import EpisodeMemoryItem, ProfileMemoryItem, RuleMemoryItem
from .ranker import deduplicate, filter_contradictions, rrf_fuse, score_episodes
from .repository import (
    fetch_active_profiles,
    fetch_active_rules,
    fetch_episodes_lexical,
    fetch_episodes_vector,
    touch_episode_retrieved,
)
from .schemas import EpisodeRow, ProfileRow, RuleRow

_LOG = get_logger("sage_kaizen.memory.retriever")

# Over-fetch multiplier for RRF (pull more candidates, fuse, trim to limit).
# Research shows pulling 2–3x from each source before fusion improves recall.
_OVERFETCH = 3


def retrieve_profiles(
    user_id: str,
    project_id: str | None,
) -> list[ProfileMemoryItem]:
    """Load always-on profile facts."""
    rows: list[ProfileRow] = fetch_active_profiles(user_id, project_id)
    return [
        ProfileMemoryItem(
            id=row.id,
            profile_type=row.profile_type,
            key=row.key,
            value_text=row.value_text,
            confidence=row.confidence,
            scope=row.scope,
            is_pinned=row.is_pinned,
            source_type=row.source_type,
        )
        for row in rows
    ]


def retrieve_rules(
    user_id: str | None,
    project_id: str | None,
    query_text: str | None,
    limit: int = 6,
) -> list[RuleMemoryItem]:
    """Retrieve procedural rules relevant to the current intent."""
    rows: list[RuleRow] = fetch_active_rules(
        user_id=user_id,
        project_id=project_id,
        query_text=query_text,
        limit=limit,
    )
    return [
        RuleMemoryItem(
            id=row.id,
            rule_kind=row.rule_kind,
            rule_text=row.rule_text,
            confidence=row.confidence,
            is_locked=row.is_locked,
            review_status=row.review_status,
        )
        for row in rows
    ]


def retrieve_episodes(
    user_id: str,
    project_id: str | None,
    query_text: str,
    scope_filter: str | None = None,
    top_k: int = 6,
) -> list[EpisodeMemoryItem]:
    """
    Hybrid episode retrieval.

    1. Embed the query using BGE-M3 (port 8020).
    2. Run lexical FTS and vector HNSW in parallel (sequential here for simplicity;
       parallelise in Phase 2 if latency budget is tight).
    3. Fuse via RRF.
    4. Score, filter contradictions, deduplicate, trim to top_k.
    5. Touch retrieved timestamps.
    """
    t0 = time.monotonic()

    fetch_limit = top_k * _OVERFETCH

    # Embed query (sync — runs in the same thread as the router)
    try:
        query_vec = embed_one(query_text)
    except Exception as exc:
        _LOG.warning("retriever | embed failed: %s — falling back to lexical only", exc)
        query_vec = None

    # FTS retrieval
    lexical_results: list[tuple[EpisodeRow, float]] = fetch_episodes_lexical(
        user_id=user_id,
        project_id=project_id,
        query_text=query_text,
        limit=fetch_limit,
    )

    # Vector retrieval (skipped if embed failed)
    vector_results: list[tuple[EpisodeRow, float]] = []
    if query_vec is not None:
        vector_results = fetch_episodes_vector(
            user_id=user_id,
            project_id=project_id,
            embedding=query_vec,
            limit=fetch_limit,
        )

    # Fuse
    fused = rrf_fuse(lexical_results, vector_results)

    # Score, filter, dedup, trim
    scored = score_episodes(fused, scope_filter=scope_filter)
    scored = filter_contradictions(scored)
    scored = deduplicate(scored)
    top = scored[:top_k]

    # Touch timestamps in background (non-blocking — fire-and-forget)
    for row, _ in top:
        try:
            touch_episode_retrieved(row.id)
        except Exception:
            pass  # non-critical

    latency_ms = (time.monotonic() - t0) * 1000
    _LOG.info(
        "retriever | lexical=%d vector=%d fused=%d top=%d latency_ms=%.1f",
        len(lexical_results), len(vector_results), len(fused), len(top), latency_ms,
    )

    return [
        EpisodeMemoryItem(
            id=row.id,
            event_type=row.event_type,
            summary_text=row.summary_text,
            tags=row.tags,
            importance=row.importance,
            confidence=row.confidence,
            was_user_correction=row.was_user_correction,
            was_explicit_preference=row.was_explicit_preference,
            created_at=row.created_at,
            retrieval_score=score,
        )
        for row, score in top
    ]
