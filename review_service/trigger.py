"""
review_service/trigger.py — Trigger detection and command parsing.

Matches review phrases typed into the Sage Kaizen chat input.
Bypasses the injection guard — these are known-safe internal commands.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# Matches all supported review trigger phrases (case-insensitive, verbose)
REVIEW_TRIGGER_RE = re.compile(
    r"""(?xi)
    (?:
        review \s+ (?:your\s+|the\s+)? codebase  |
        codebase \s+ review                       |
        architect \s+ review                      |
        (?:run\s+)? code \s+ review               |
        review \s+ (?:the\s+)? staged
              (?:\s+ changes?)?                   |
        review \s+ staged                         |
        review \s+ (?:the\s+)? file               |
        review \s+ (?:the\s+)? module             |
        regression \s+ audit                      |
        review \s+ mode
    )
    """,
)


@dataclass
class ReviewCommand:
    mode: str         # "full" | "staged" | "file" | "regression"
    target: str = ""  # file path for "file" mode; base ref for "regression"


def is_review_command(text: str) -> bool:
    """Return True if the input should be handled as a review trigger."""
    return bool(REVIEW_TRIGGER_RE.search(text.strip()))


def parse_review_command(text: str) -> ReviewCommand:
    """
    Parse a trigger phrase into a ReviewCommand.

    Examples
    --------
    "Review your codebase"            → ReviewCommand(mode="full")
    "Review staged changes"           → ReviewCommand(mode="staged")
    "Review the file chat_service.py" → ReviewCommand(mode="file",  target="chat_service.py")
    "Regression audit after HEAD~2"   → ReviewCommand(mode="regression", target="HEAD~2")
    """
    lower = text.lower()

    # File mode: "review the file X" or "review the module X"
    m = re.search(r"(?:review\s+(?:the\s+)?(?:file|module))\s+(\S+)", lower)
    if m:
        return ReviewCommand(mode="file", target=m.group(1))

    # Staged mode
    if "staged" in lower:
        return ReviewCommand(mode="staged")

    # Regression mode: optional ref extraction
    if "regression" in lower or "audit" in lower:
        ref_m = re.search(r"after\s+(\S+)", lower)
        base = ref_m.group(1) if ref_m else "HEAD~1"
        return ReviewCommand(mode="regression", target=base)

    # Default: full repo review
    return ReviewCommand(mode="full")
