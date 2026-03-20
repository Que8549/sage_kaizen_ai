from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List

from openai_client import HttpTimeouts, stream_chat_completions
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


# ---------------------------------------------------
# LLM-assisted routing
# ---------------------------------------------------

_CLASSIFY_SYSTEM = (
    "You are a query router. Your only job is to classify queries.\n\n"
    "Reply with exactly one word: FAST or ARCHITECT.\n\n"
    "FAST — simple questions, quick lookups, summaries, casual chat, "
    "basic calculations, short creative tasks\n"
    "ARCHITECT — deep technical analysis, code review, architecture design, "
    "complex multi-step reasoning, long-form writing, advanced tutoring, "
    "hardware tuning, system design"
)


def llm_route(
    user_text: str,
    fast_base_url: str,
    model_id: str,
    timeouts: HttpTimeouts,
    force_architect: bool = False,
) -> RouteDecision:
    """
    Ask the FAST brain to classify query complexity in ~1 token.

    Requires the FAST brain (Q5) to already be running.  If the call fails
    for any reason (server error, timeout, unexpected output) this function
    falls back to the keyword-scoring heuristic route() automatically.

    Parameters
    ----------
    user_text:       The raw user query.
    fast_base_url:   Base URL of the FAST brain (e.g. http://127.0.0.1:8011).
    model_id:        Model alias reported by /v1/models.
    timeouts:        HttpTimeouts — use short values (e.g. connect=2s, read=5s).
    force_architect: If True, skip classification and return ARCHITECT immediately.
    """
    if force_architect:
        decision = RouteDecision(brain="ARCHITECT", reasons=["force_architect"], score=999)
        _log_decision(decision, user_text)
        return decision

    if not user_text:
        decision = RouteDecision(brain="FAST", reasons=["empty_input"], score=0)
        _log_decision(decision, user_text)
        return decision

    classify_messages = [
        {"role": "system", "content": _CLASSIFY_SYSTEM},
        # Cap at 500 chars — the classifier only needs enough context to judge
        {"role": "user", "content": user_text[:500]},
    ]

    try:
        chunks = list(stream_chat_completions(
            base_url=fast_base_url,
            model=model_id,
            messages=classify_messages,
            temperature=0.0,   # deterministic classification
            top_p=1.0,
            max_tokens=10,     # we only need "FAST" or "ARCHITECT" (1–2 tokens)
            timeouts=timeouts,
        ))
        label = "".join(chunks).strip().upper()
        brain = "ARCHITECT" if "ARCHITECT" in label else "FAST"
        _LOG.info(
            "llm_route | brain=%s | label=%r | input_chars=%s",
            brain, label, len(user_text),
        )
        score = 999 if brain == "ARCHITECT" else 0
        decision = RouteDecision(brain=brain, reasons=["llm_classification"], score=score)
        return decision

    except Exception:
        _LOG.warning("llm_route failed; falling back to heuristic route()")
        return route(user_text, force_architect=False)

