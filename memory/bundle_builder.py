"""
memory/bundle_builder.py
Assembles a MemoryBundle from retrieved items within the per-brain token cap.

Token budget (from 04-Memory_Service.md):
  FAST brain:      max 600 tokens
  ARCHITECT brain: max 1,500 tokens

Priority order when trimming: profiles → rules → episodes (drop lowest-scored first).
"""
from __future__ import annotations

from typing import List

from sk_logging import get_logger
from .models import EpisodeMemoryItem, MemoryBundle, ProfileMemoryItem, RuleMemoryItem

_LOG = get_logger("sage_kaizen.memory.bundle_builder")

# Rough token estimation constants.
# Calibrated for the compact prompt format in 04-Memory_Service.md.
_CHARS_PER_TOKEN = 3.5
_HEADER_TOKENS   = 40   # overhead for section headers + blank lines


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _profile_tokens(p: ProfileMemoryItem) -> int:
    return _estimate_tokens(f"- {p.key}: {p.value_text}")


def _rule_tokens(r: RuleMemoryItem) -> int:
    return _estimate_tokens(f"- {r.rule_text}")


def _episode_tokens(e: EpisodeMemoryItem) -> int:
    return _estimate_tokens(f"- [{e.event_type}] {e.summary_text}")


def build_bundle(
    profiles: List[ProfileMemoryItem],
    rules: List[RuleMemoryItem],
    episodes: List[EpisodeMemoryItem],
    max_tokens: int,
) -> MemoryBundle:
    """
    Greedily fill the bundle within max_tokens.

    Order: all profiles → all rules → episodes by descending retrieval_score.
    Truncates episodes first (lowest score dropped), then rules, then profiles.
    """
    used = _HEADER_TOKENS
    was_truncated = False

    kept_profiles: List[ProfileMemoryItem] = []
    for p in profiles:
        cost = _profile_tokens(p)
        if used + cost <= max_tokens:
            kept_profiles.append(p)
            used += cost
        else:
            was_truncated = True

    kept_rules: List[RuleMemoryItem] = []
    for r in rules:
        cost = _rule_tokens(r)
        if used + cost <= max_tokens:
            kept_rules.append(r)
            used += cost
        else:
            was_truncated = True

    # Episodes sorted by retrieval_score descending (already sorted by retriever,
    # but re-sort here to be safe in case caller mixed the order)
    sorted_eps = sorted(episodes, key=lambda e: e.retrieval_score, reverse=True)
    kept_episodes: List[EpisodeMemoryItem] = []
    for e in sorted_eps:
        cost = _episode_tokens(e)
        if used + cost <= max_tokens:
            kept_episodes.append(e)
            used += cost
        else:
            was_truncated = True

    total = len(kept_profiles) + len(kept_rules) + len(kept_episodes)

    if was_truncated:
        _LOG.debug(
            "bundle_builder | truncated: profiles=%d rules=%d episodes=%d tokens=%d/%d",
            len(kept_profiles), len(kept_rules), len(kept_episodes), used, max_tokens,
        )

    return MemoryBundle(
        profiles=kept_profiles,
        rules=kept_rules,
        episodes=kept_episodes,
        total_items=total,
        estimated_tokens=used,
        was_truncated=was_truncated,
    )


def format_bundle_prompt(bundle: MemoryBundle) -> str:
    """
    Format a MemoryBundle into the compact prompt segment defined in
    04-Memory_Service.md.  Returns an empty string if the bundle is empty.
    """
    if bundle.total_items == 0:
        return ""

    lines: List[str] = []

    if bundle.profiles:
        lines.append("[USER PROFILE]")
        for p in bundle.profiles:
            lines.append(f"- {p.key}: {p.value_text}")
        lines.append("")

    if bundle.rules:
        lines.append("[PROJECT NORMS]")
        for r in bundle.rules:
            lock_marker = " [LOCKED]" if r.is_locked else ""
            lines.append(f"- {r.rule_text}{lock_marker}")
        lines.append("")

    if bundle.episodes:
        lines.append("[RELEVANT PRIOR EPISODES]")
        for e in bundle.episodes:
            correction = " [USER CORRECTION]" if e.was_user_correction else ""
            lines.append(f"- [{e.event_type}]{correction} {e.summary_text}")
        lines.append("")

    header = (
        "You have access to structured prior memory about this user and project.\n"
        "Use the following memories only as guidance if relevant to the current request.\n"
        "Prefer current user instructions over past memory.  If current instructions\n"
        "conflict with past memory, follow the current instructions.\n\n"
    )

    return header + "\n".join(lines)
