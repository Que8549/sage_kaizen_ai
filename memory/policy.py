"""
memory/policy.py
Promotion thresholds, decay rules, and selective episode write policy.

All thresholds are defined here as module-level constants so they can be
tuned in one place without touching logic.
"""
from __future__ import annotations

from typing import List, Optional

from sk_logging import get_logger
from .models import EpisodeWriteRequest, PromotionDecision
from .schemas import EpisodeRow

_LOG = get_logger("sage_kaizen.memory.policy")

# ---------------------------------------------------------------------------
# Selective write policy (Path B — episodic write)
# ---------------------------------------------------------------------------

# Minimum turn length (tokens, estimated) to consider writing an episode
# when no explicit signal is present.
_MIN_AUTO_TOKENS = 200

# Minimum importance score to auto-write a long turn.
_MIN_AUTO_IMPORTANCE = 0.4

# Event types that always trigger a write regardless of length.
_ALWAYS_WRITE_EVENTS = frozenset({
    "correction",
    "preference",
    "decision",
    "approval",
    "architecture_choice",
    "model_selection",
    "code_change",
    "bug_report",
})

# Event types that never auto-write (greeting, trivial, ack, etc.)
_NEVER_WRITE_EVENTS = frozenset({
    "greeting",
    "acknowledgement",
    "trivial",
    "clarification",
})


def should_write_episode(
    event_type: str,
    user_text: str,
    assistant_text: str,
    was_user_correction: bool = False,
    was_explicit_preference: bool = False,
    estimated_importance: float = 0.5,
) -> bool:
    """
    Selective write policy from 04-Memory_Service.md.

    Returns True if this turn is worth persisting as an episode.
    """
    if event_type in _NEVER_WRITE_EVENTS:
        return False

    if was_user_correction or was_explicit_preference:
        return True

    if event_type in _ALWAYS_WRITE_EVENTS:
        return True

    # Long turn with meaningful content
    combined_tokens = (len(user_text) + len(assistant_text)) / 3.5  # rough est.
    if combined_tokens >= _MIN_AUTO_TOKENS and estimated_importance >= _MIN_AUTO_IMPORTANCE:
        return True

    return False


# ---------------------------------------------------------------------------
# Promotion policy thresholds
# ---------------------------------------------------------------------------

# Minimum confidence from Architect reflection to promote to profile memory.
PROFILE_PROMOTE_CONFIDENCE = 0.90

# Minimum number of consistent observations across sessions for auto-promotion.
PROFILE_PROMOTE_MIN_EPISODES = 3

# Minimum confidence to promote to a procedural rule.
RULE_PROMOTE_CONFIDENCE = 0.85

# Minimum promotion_count before a rule becomes 'approved'.
RULE_APPROVE_COUNT = 2

# Maximum rule promotions per consolidation run (prevents bulk noisy promotions).
MAX_PROMOTIONS_PER_RUN = 5


def check_rule_promotion(
    rule_text: str,
    rule_kind: str,
    confidence: float,
    rationale: str,
    source_memory_id: Optional[str] = None,
    scope: str = "project",
) -> Optional[PromotionDecision]:
    """
    Evaluate whether a candidate rule should be promoted.

    Returns a PromotionDecision (not yet approved) if the candidate passes
    minimum thresholds, else None.
    """
    if confidence < RULE_PROMOTE_CONFIDENCE:
        _LOG.debug(
            "policy | rule rejected (confidence %.2f < %.2f): %s",
            confidence, RULE_PROMOTE_CONFIDENCE, rule_text[:60],
        )
        return None

    if not rule_text.strip():
        return None

    return PromotionDecision(
        source_memory_id=source_memory_id,
        rule_text=rule_text,
        rule_kind=rule_kind,
        confidence=confidence,
        rationale=rationale,
        approved=False,
        scope=scope,
    )


# ---------------------------------------------------------------------------
# Decay helpers
# ---------------------------------------------------------------------------

def compute_decay_weight(age_days: int, importance: float) -> float:
    """
    Decayed importance score for pruning decisions.

    High-importance events (importance >= 0.8) decay slowly (half-life ~180 days).
    Low-importance events (importance < 0.4) decay quickly (half-life ~30 days).
    """
    if importance >= 0.8:
        half_life = 180.0
    elif importance >= 0.6:
        half_life = 90.0
    elif importance >= 0.4:
        half_life = 60.0
    else:
        half_life = 30.0

    import math
    return importance * math.exp(-0.693 * age_days / half_life)


def should_prune_episode(row: EpisodeRow, age_days: int) -> bool:
    """
    Return True if this episode should be archived/deleted.

    Never prune:
    - user corrections
    - explicit preferences
    - high-importance decisions (importance >= 0.8)

    Prune if:
    - decayed weight drops below 0.05
    - age > 365 days and importance < 0.3
    """
    if row.was_user_correction or row.was_explicit_preference:
        return False
    if row.importance >= 0.8:
        return False

    decayed = compute_decay_weight(age_days, row.importance)
    if decayed < 0.05:
        return True
    if age_days > 365 and row.importance < 0.3:
        return True
    return False
