"""
memory/writer.py
Write paths A–D for the Sage Kaizen Memory Service.

Path A — explicit profile write (user stated a stable preference).
Path B — episodic write (selective, post-turn).
Path C — reflection write (consolidator output).
Path D — rule promotion (called by consolidator / policy check).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from sk_logging import get_logger
from .audit import log_action
from .embedder import embed_one
from .models import EpisodeWriteRequest, ProfileWriteRequest, PromotionDecision
from .policy import should_write_episode
from .repository import (
    insert_episode,
    insert_reflection,
    insert_rule,
    upsert_profile,
)

_LOG = get_logger("sage_kaizen.memory.writer")


# ---------------------------------------------------------------------------
# Path A — explicit profile
# ---------------------------------------------------------------------------

def write_explicit_profile(req: ProfileWriteRequest) -> str:
    """
    Write a user-stated stable preference to memory.profiles.
    Returns the row id.
    """
    t0 = time.monotonic()
    row_id = upsert_profile(
        user_id=req.user_id,
        project_id=req.project_id,
        workspace_id=req.workspace_id,
        scope=req.scope,
        profile_type=req.profile_type,
        key=req.key,
        value_text=req.value_text,
        confidence=req.confidence,
        source_type=req.source_type,
        is_locked=req.is_locked,
    )
    log_action(
        memory_table="profiles",
        memory_id=row_id,
        action_type="insert",
        actor_type="user" if req.source_type == "explicit_user" else "system",
        new_value={"key": req.key, "value": req.value_text},
        reason="explicit_profile_write",
    )
    _LOG.info(
        "writer | profile upserted key=%s user=%s latency_ms=%.1f",
        req.key, req.user_id, (time.monotonic() - t0) * 1000,
    )
    return row_id


# ---------------------------------------------------------------------------
# Path B — episodic write (selective)
# ---------------------------------------------------------------------------

def write_episode(req: EpisodeWriteRequest) -> Optional[str]:
    """
    Write an episode if it passes the selective write policy.
    Embeds the summary_text with BGE-M3.
    Returns the row id, or None if the policy skipped the write.
    """
    if not should_write_episode(
        event_type=req.event_type,
        user_text=req.summary_text,      # proxy for filtering
        assistant_text=req.raw_excerpt or "",
        was_user_correction=req.was_user_correction,
        was_explicit_preference=req.was_explicit_preference,
        estimated_importance=req.importance,
    ):
        _LOG.debug(
            "writer | episode skipped by policy event=%s importance=%.2f user=%s",
            req.event_type, req.importance, req.user_id,
        )
        return None

    t0 = time.monotonic()

    # Embed the summary — skip write entirely if BGE-M3 is unavailable.
    # An episode without an embedding is unfindable by vector search and
    # pollutes the index without adding retrieval value.
    try:
        embedding = embed_one(req.summary_text)
    except Exception as exc:
        _LOG.warning(
            "writer | embed failed (BGE-M3 down?), episode skipped to avoid unfindable row: %s", exc
        )
        return None

    if embedding is None:
        _LOG.warning("writer | embed_one returned None, episode skipped")
        return None

    row_id = insert_episode(
        user_id=req.user_id,
        project_id=req.project_id,
        workspace_id=req.workspace_id,
        session_id=req.session_id,
        scope=req.scope,
        event_type=req.event_type,
        summary_text=req.summary_text,
        embedding=embedding,
        intent_label=req.intent_label,
        raw_excerpt=req.raw_excerpt,
        tags=req.tags,
        importance=req.importance,
        confidence=req.confidence,
        was_user_correction=req.was_user_correction,
        was_explicit_preference=req.was_explicit_preference,
    )
    log_action(
        memory_table="episodes",
        memory_id=row_id,
        action_type="insert",
        actor_type="fast_brain",
        new_value={"event_type": req.event_type, "summary": req.summary_text[:120]},
        reason="post_turn_episodic_write",
    )
    _LOG.info(
        "writer | episode written event=%s user=%s latency_ms=%.1f",
        req.event_type, req.user_id, (time.monotonic() - t0) * 1000,
    )
    return row_id


# ---------------------------------------------------------------------------
# Path C — reflection write
# ---------------------------------------------------------------------------

def write_reflection(
    user_id: str,
    project_id: Optional[str],
    session_id: Optional[str],
    reflection_type: str,
    summary_text: str,
    profile_candidates: Optional[List[Dict[str, Any]]] = None,
    rule_candidates: Optional[List[Dict[str, Any]]] = None,
    contradictions: Optional[List[Dict[str, Any]]] = None,
    pruning_suggestions: Optional[List[Dict[str, Any]]] = None,
    confidence: float = 0.7,
) -> str:
    """Write a reflection record (output of consolidator).  Returns the row id."""
    row_id = insert_reflection(
        user_id=user_id,
        project_id=project_id,
        session_id=session_id,
        reflection_type=reflection_type,
        summary_text=summary_text,
        profile_candidates=profile_candidates,
        rule_candidates=rule_candidates,
        contradictions=contradictions,
        pruning_suggestions=pruning_suggestions,
        confidence=confidence,
    )
    log_action(
        memory_table="reflections",
        memory_id=row_id,
        action_type="insert",
        actor_type="architect_brain",
        new_value={"type": reflection_type, "summary": summary_text[:120]},
        reason="consolidation_run",
    )
    _LOG.info("writer | reflection written type=%s user=%s id=%s", reflection_type, user_id, row_id)
    return row_id


# ---------------------------------------------------------------------------
# Path D — rule promotion
# ---------------------------------------------------------------------------

def write_promoted_rule(decision: PromotionDecision, user_id: Optional[str], project_id: Optional[str]) -> str:
    """Write a promoted rule from a PromotionDecision.  Returns the rule id."""
    row_id = insert_rule(
        user_id=user_id,
        project_id=project_id,
        scope=decision.scope,
        rule_kind=decision.rule_kind,
        rule_text=decision.rule_text,
        rationale=decision.rationale,
        confidence=decision.confidence,
        source_type="promoted_from_episode",
        source_memory_id=decision.source_memory_id,
        is_locked=False,
        review_status="approved" if decision.approved else "proposed",
    )
    log_action(
        memory_table="rules",
        memory_id=row_id,
        action_type="promote",
        actor_type="architect_brain",
        new_value={"rule": decision.rule_text[:120], "confidence": decision.confidence},
        reason=decision.rationale,
    )
    _LOG.info(
        "writer | rule promoted kind=%s confidence=%.2f user=%s id=%s",
        decision.rule_kind, decision.confidence, user_id, row_id,
    )
    return row_id
