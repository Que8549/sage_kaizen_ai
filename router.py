from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Dict, List

from sk_logging import get_logger

# RAG v1 integration
from rag_v1.runtime.router_integration import RagInjector
from rag_v1.config.rag_settings import RagSettings

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

# ----------------------------
# RAG globals (safe + simple)
# ----------------------------
# Uses env vars if you set them (recommended), otherwise defaults from RagSettings
# Example:
#   set SAGE_RAG_ENABLED=1
#   set SAGE_RAG_FAST_TOPK=4
#   set SAGE_RAG_ARCH_TOPK=10
#
# NOTE: RagSettings itself can also read env vars (pydantic settings), so you can centralize there too.
rag_settings = RagSettings()
rag_injector = RagInjector(rag_settings)

def _env_bool(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v.strip())
    except ValueError:
        return default


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


# ---------------------------------------------------
# NEW: RAG hook you call before llama-server request
# ---------------------------------------------------
def apply_rag(
    messages: List[Dict[str, Any]],
    user_text: str,
    decision: RouteDecision,
    rag_enabled: bool | None = None,
) -> List[Dict[str, Any]]:
    """
    Drop-in RAG enrichment:
      - Call this AFTER you build messages (prompt library + templates),
        but BEFORE you send to llama-server.

    Example usage in UI/server code:
        decision = router.route(user_text)
        messages = build_messages(...)
        messages = router.apply_rag(messages, user_text, decision)
        resp = call_llama_server(decision.brain, messages)

    Controls:
      - rag_enabled: if None, reads env SAGE_RAG_ENABLED (default True)
      - top_k: FAST uses SAGE_RAG_FAST_TOPK (default 4)
               ARCHITECT uses SAGE_RAG_ARCH_TOPK (default 10)
    """
    
    if not user_text:
        return messages

    enabled = _env_bool("SAGE_RAG_ENABLED", default=True) if rag_enabled is None else rag_enabled
    if not enabled:
        return messages

    # Optionally, skip RAG for ultra-short inputs (reduces needless calls)
    min_chars = _env_int("SAGE_RAG_MIN_CHARS", default=12)
    if len(user_text.strip()) < min_chars:
        return messages

    fast_k = _env_int("SAGE_RAG_FAST_TOPK", default=4)
    arch_k = _env_int("SAGE_RAG_ARCH_TOPK", default=10)

    top_k = fast_k if decision.brain == "FAST" else arch_k

    try:
        out = rag_injector.maybe_inject(
            messages=messages,
            user_text=user_text,
            brain=decision.brain,
            enabled=True,
        )
        # NOTE: current RagInjector decides its own top_k (FAST=4 / ARCH=10).
        # If you want ENV-driven top_k, update RagInjector to accept top_k,
        # or add a second method. For now, keep behavior stable and simple.
        return out
    except Exception:
        _LOG.exception("RAG injection failed; continuing without RAG")
        return messages
