"""
input_guard.py — Prompt-injection defense for Sage Kaizen.

Protects against indirect injection via retrieved content (RAG chunks,
Wikipedia passages, web search snippets) — the primary attack surface for
a local-only system with untrusted document/web ingestion.

Also provides check_user_input() for direct-input hard-rejection of
structural injection tokens.

Public API
----------
sanitize_chunk(text, max_chars)          — clean a retrieved doc/wiki chunk;
                                           strips template tokens and fake
                                           instruction headers; returns
                                           sanitized text (never raises).
sanitize_search_snippet(text, max_chars) — same as sanitize_chunk but also
                                           strips HTML tags; shorter default.
check_user_input(text)                   — hard-reject user input containing
                                           structural injection patterns.
                                           Raises InjectionDetectedError.
"""
from __future__ import annotations

import re

from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.input_guard")


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class InjectionDetectedError(ValueError):
    """Raised by check_user_input() when a hard-reject pattern is matched."""

    def __init__(self, pattern: str) -> None:
        super().__init__(
            f"Input blocked: suspected prompt-injection pattern ({pattern!r}). "
            "Please rephrase your request."
        )
        self.pattern = pattern


# ---------------------------------------------------------------------------
# Chat template structural tokens
#
# These markers are used by LLM chat templates to delimit role boundaries.
# They have no legitimate use in plain-text document or web content, making
# them unambiguous injection vectors when found in retrieved material.
# ---------------------------------------------------------------------------

_CHAT_TOKENS: tuple[str, ...] = (
    "<|im_start|>", "<|im_end|>",              # Qwen / ChatML
    "[INST]", "[/INST]",                        # Llama-2
    "<<SYS>>", "<</SYS>>",                     # Llama-2 system block
    "<|system|>", "<|user|>", "<|assistant|>", # generic role tags
    "<|eot_id|>", "<|end_of_turn|>",            # Llama-3 / Gemma
    "<|endoftext|>", "<|begin_of_text|>",       # GPT-2 / Llama-3 BOS
)

# Pre-compiled as exact-match patterns (case-sensitive — tokens are exact)
_CHAT_TOKEN_RES: tuple[re.Pattern, ...] = tuple(
    re.compile(re.escape(tok)) for tok in _CHAT_TOKENS
)


# ---------------------------------------------------------------------------
# Fake instruction headers
#
# Lines that attempt to masquerade as authoritative instruction headers
# inside retrieved document or web content. Matched at line start; the
# matched line is replaced with a [REMOVED] marker so surrounding text
# stays coherent.
# ---------------------------------------------------------------------------

_FAKE_HEADER_RE = re.compile(
    r"^[ \t]*(?:"
    r"system\s*prompt\s*[:：]"
    r"|new\s+(?:system\s+)?instructions?\s*[:：]"
    r"|override\s+instructions?\s*[:：]"
    r"|ignore\s+(?:all\s+)?previous\s+instructions?"
    r"|disregard\s+(?:all\s+)?(?:previous\s+)?instructions?"
    r"|you\s+(?:must\s+)?(?:now\s+)?(?:ignore|forget)\s+"
    r"(?:your\s+)?(?:previous\s+)?(?:instructions?|guidelines?|rules?|training)"
    r")\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# HTML tag stripper (for web search snippets)
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]{0,200}>")


# ---------------------------------------------------------------------------
# Hard-reject patterns for direct user input
#
# Conservative: only match structural chat-template tokens and the most
# unambiguous explicit-override phrases. These have essentially zero
# legitimate use in plain natural-language queries.
# ---------------------------------------------------------------------------

_HARD_REJECT: tuple[tuple[re.Pattern, str], ...] = (
    # Chat template structural tokens
    (re.compile(r"<\|im_start\|>",  re.IGNORECASE), "<|im_start|>"),
    (re.compile(r"<\|im_end\|>",    re.IGNORECASE), "<|im_end|>"),
    (re.compile(r"\[INST\]",        re.IGNORECASE), "[INST]"),
    (re.compile(r"<<SYS>>",         re.IGNORECASE), "<<SYS>>"),
    (re.compile(r"<\|eot_id\|>",    re.IGNORECASE), "<|eot_id|>"),
    (re.compile(r"<\|system\|>",    re.IGNORECASE), "<|system|>"),
    # Explicit instruction-override phrases (only the most unambiguous forms)
    (
        re.compile(r"ignore\s+all\s+previous\s+instructions?", re.IGNORECASE),
        "ignore all previous instructions",
    ),
    (
        re.compile(r"new\s+system\s+prompt\s*:", re.IGNORECASE),
        "new system prompt:",
    ),
    (
        re.compile(r"override\s+(?:your\s+)?(?:system\s+)?instructions?\s*:", re.IGNORECASE),
        "override instructions:",
    ),
)

# Hard cap on user input length (chars). Context-stuffing via extremely long
# inputs is a known vector for diluting or overwriting the system prompt.
_MAX_INPUT_CHARS = 32_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sanitize_chunk(text: str, max_chars: int | None = 2000) -> str:
    """
    Clean a retrieved document or Wikipedia chunk before context injection.

    Steps
    -----
    1. Enforce max_chars limit (truncate with marker). Pass None to skip.
    2. Strip chat template structural tokens.
    3. Replace lines that look like fake system-prompt headers with [REMOVED].

    Never raises. Returns the sanitized text (may be shorter than input).
    """
    if not text:
        return text

    original_len = len(text)

    # 1. Length cap
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + " [truncated]"

    # 2. Strip chat template tokens
    for pat in _CHAT_TOKEN_RES:
        text = pat.sub("", text)

    # 3. Replace fake instruction headers
    cleaned, n_headers = _FAKE_HEADER_RE.subn("[REMOVED]", text)
    if n_headers:
        _LOG.warning(
            "input_guard | sanitize_chunk | removed %d fake header(s) | original_len=%d",
            n_headers,
            original_len,
        )
        text = cleaned

    return text.strip()


def sanitize_search_snippet(text: str, max_chars: int | None = 500) -> str:
    """
    Clean a web search snippet before context injection.

    More aggressive than sanitize_chunk:
      - Strips HTML tags first (search engines often include raw markup).
      - Shorter default max_chars (web snippets should be brief summaries).

    Never raises. Returns the sanitized text.
    """
    if not text:
        return text

    # Strip HTML tags before length check — tags inflate character count
    text = _HTML_TAG_RE.sub("", text)

    return sanitize_chunk(text, max_chars=max_chars)


def check_user_input(text: str) -> None:
    """
    Hard-reject gate for direct user input.

    Scans for structural chat-template tokens and explicit instruction-override
    patterns with no legitimate use in plain natural-language queries.

    Also enforces _MAX_INPUT_CHARS to defend against context-stuffing.

    Raises InjectionDetectedError if any pattern matches.
    Returns None silently when input is clean.
    """
    if not text:
        return

    if len(text) > _MAX_INPUT_CHARS:
        raise InjectionDetectedError(f"input exceeds {_MAX_INPUT_CHARS:,} character limit")

    for pat, label in _HARD_REJECT:
        if pat.search(text):
            _LOG.warning(
                "input_guard | check_user_input | BLOCKED | pattern=%r | input_chars=%d",
                label,
                len(text),
            )
            raise InjectionDetectedError(label)
