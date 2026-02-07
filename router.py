
from __future__ import annotations

from typing import Tuple


DEPTH_HINTS = (
    "explain", "analyze", "compare", "why", "how", "history", "philosophy", "theology",
    "deep", "in depth", "detailed", "step-by-step", "teach", "tutor", "architecture",
    "design", "tradeoff", "pros and cons", "evaluate",
)
CODE_HINTS = ("code", "python", "c#", "typescript", "debug", "stack trace", "error")


def should_escalate_to_q6(user_text: str, force_q6: bool) -> bool:
    if force_q6:
        return True
    txt = user_text.lower()
    if any(k in txt for k in DEPTH_HINTS):
        return True
    if any(k in txt for k in CODE_HINTS):
        return True
    if " and " in txt or " also " in txt or " vs " in txt or "compare" in txt:
        return True
    return False
