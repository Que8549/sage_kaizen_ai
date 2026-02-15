from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List
from sk_logging import get_logger

DEPTH_HINTS = (
    "explain", "analyze", "compare", "why", "how", "history", "philosophy", "theology",
    "deep", "in depth", "detailed", "step-by-step", "teach", "tutor", "architecture",
    "design", "tradeoff", "pros and cons", "evaluate", "optimize", "tune", "take time to think",
    "double check your answer",
)

CODE_HINTS = ("code", "python", "c#", "typescript", "debug", "stack trace", "error", "traceback", "exception")

FAST_HINTS = ("summarize", "tl;dr", "quick", "brief", "short", "bullet", "one sentence", "in one paragraph")

_WORD = r"(?:^|[\s\W]){w}(?:$|[\s\W])"

_LOG = get_logger("sage_kaizen.router")

@dataclass(frozen=True)
class RouteDecision:
    brain: str                 # "FAST" (5080) or "ARCHITECT" (5090)
    reasons: List[str]
    score: int

def _log_decision(decision: "RouteDecision", user_text: str) -> None:
    reasons = ",".join(decision.reasons[:8]) if decision.reasons else ""
    _LOG.info(
        "route | brain=%s | score=%s | reasons=[%s] | input_chars=%s",
        decision.brain, decision.score, reasons, len(user_text)
    )

def _has_word(txt: str, word: str) -> bool:
    return re.search(_WORD.format(w=re.escape(word)), txt) is not None


def route(user_text: str, force_architect: bool = False) -> RouteDecision:
    """
    Returns a routing decision:
      - FAST      -> 5080 (Qwen2.5-14B Q6_K)
      - ARCHITECT -> 5090 (Qwen2.5-32B Q6_K_L)

    Also logs the decision to logs/sage_kaizen.log.
    """
    if not user_text:
        decision = RouteDecision(brain="FAST", reasons=["empty_input"], score=0)
        _log_decision(decision, user_text)
        return decision

    if force_architect:
        decision = RouteDecision(brain="ARCHITECT", reasons=["force_architect"], score=999)
        _log_decision(decision, user_text)
        return decision

    txt = user_text.lower()
    score = 0
    reasons: List[str] = []

    # Length heuristics
    n = len(txt)
    if n > 2000:
        score += 4
        reasons.append("very_long_input")
    elif n > 800:
        score += 2
        reasons.append("long_input")

    # Depth hints (moderate)
    for k in DEPTH_HINTS:
        if k in txt:
            score += 2
            reasons.append(f"depth:{k}")
            break

    # Code hints (strong)
    for k in CODE_HINTS:
        if k in txt:
            score += 3
            reasons.append(f"code:{k}")
            break

    # Multi-part markers (weak)
    if " and " in txt or " also " in txt:
        score += 1
        reasons.append("multi_part_marker")

    if _has_word(txt, "vs") or _has_word(txt, "versus"):
        score += 1
        reasons.append("comparison_marker")

    # Fast intent markers (counterweight)
    for k in FAST_HINTS:
        if k in txt:
            score -= 2
            reasons.append(f"fast_intent:{k}")
            break

    # Final threshold
    if score >= 3:
        decision = RouteDecision(brain="ARCHITECT", reasons=reasons or ["score_threshold"], score=score)
        _log_decision(decision, user_text)
        return decision

    decision = RouteDecision(brain="FAST", reasons=reasons or ["default_fast"], score=score)
    _log_decision(decision, user_text)
    return decision
