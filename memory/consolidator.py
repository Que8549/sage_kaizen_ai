"""
memory/consolidator.py
Reflection and consolidation — runs Architect brain over recent episodes to
extract stable preferences, detect contradictions, and propose rule promotions.

Two modes:
  lightweight  — Fast brain via lightweight prompt (end-of-session, low cost)
  deep         — Architect brain with full reasoning (nightly / manual)

Uses the existing openai_client pattern to call llama-server.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from openai_client import stream_chat_completions, HttpTimeouts
from pg_settings import PgSettings
from sk_logging import get_logger
from .models import ProfileWriteRequest, PromotionDecision, ReflectionResult
from .policy import (
    MAX_PROMOTIONS_PER_RUN,
    RULE_PROMOTE_CONFIDENCE,
    check_rule_promotion,
)
from .repository import fetch_episodes_since, insert_reflection
from .writer import write_reflection, write_promoted_rule

_LOG = get_logger("sage_kaizen.memory.consolidator")

# Architect brain endpoint
_ARCHITECT_BASE = "http://127.0.0.1:8012"
_FAST_BASE      = "http://127.0.0.1:8011"


def _call_brain(
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """Synchronous non-streaming call to a llama-server brain.  Returns the response text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    chunks: List[str] = []
    timeouts = HttpTimeouts(connect=5.0, read=120.0, write=10.0, pool=5.0)
    for chunk in stream_chat_completions(
        base_url=base_url,
        messages=messages,
        model="",        # llama-server ignores model field for local inference
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_tokens=max_tokens,
        stream=True,
        timeouts=timeouts,
    ):
        chunks.append(chunk)
    return "".join(chunks).strip()


_REFLECTION_SYSTEM = """You are a memory consolidation agent for Sage Kaizen AI.
Your job is to analyse a list of recent interaction episodes and extract:
1. Stable user preferences (profile candidates)
2. Operational rules that should guide future behavior (rule candidates)
3. Contradictions between episodes
4. Episodes that are likely stale or low-value and can be pruned

Respond ONLY with a valid JSON object matching this schema exactly:
{
  "summary": "one sentence summary of the session patterns",
  "profile_candidates": [
    {"profile_type": "...", "key": "...", "value_text": "...", "confidence": 0.0}
  ],
  "rule_candidates": [
    {"rule_kind": "...", "rule_text": "...", "confidence": 0.0, "rationale": "..."}
  ],
  "contradictions": [
    {"episode_ids": ["..."], "description": "..."}
  ],
  "pruning_suggestions": [
    {"episode_id": "...", "reason": "..."}
  ]
}

Rules:
- Only include profile candidates with confidence >= 0.85
- Only include rule candidates with confidence >= 0.80
- Be conservative — fewer high-confidence items is better than many uncertain ones
- Never include sensitive personal data in rule_text
"""


def run_reflection(
    user_id: str,
    project_id: str = "sage_kaizen",
    session_id: Optional[str] = None,
    mode: str = "deep",   # "lightweight" | "deep"
    lookback_hours: int = 24,
) -> ReflectionResult:
    """
    Run a reflection consolidation pass for a user.

    mode="lightweight": Fast brain, short prompt, ~1s latency
    mode="deep":        Architect brain, full reasoning, ~30–120s latency
    """
    t0 = time.monotonic()

    # Fetch recent episodes via the public repository API
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=lookback_hours)
    rows = fetch_episodes_since(
        user_id=user_id,
        project_id=project_id,
        cutoff=cutoff,
        limit=40,
    )

    if not rows:
        _LOG.info("consolidator | no episodes in window, skipping reflection user=%s", user_id)
        return ReflectionResult(
            reflection_id="",
            summary_text="No episodes in the lookback window.",
        )

    # Build episode list for the prompt
    episode_list = "\n".join(
        f"[{r['event_type']}] id={r['id']} importance={r['importance']:.2f}: {r['summary_text']}"
        for r in rows
    )

    user_prompt = (
        f"User: {user_id}  Project: {project_id}\n"
        f"Lookback: {lookback_hours}h  Episodes: {len(rows)}\n\n"
        f"EPISODES:\n{episode_list}\n\n"
        "Extract memory consolidation JSON now."
    )

    base_url = _ARCHITECT_BASE if mode == "deep" else _FAST_BASE
    max_tokens = 1500 if mode == "deep" else 800

    parse_error = False
    try:
        raw = _call_brain(
            base_url=base_url,
            system_prompt=_REFLECTION_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
        # Strip markdown fences (``` or ```json) — regex is more robust than split
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        data: Dict[str, Any] = json.loads(raw)
    except Exception as exc:
        _LOG.warning("consolidator | LLM parse failed: %s", exc)
        parse_error = True
        data = {"summary": "Reflection failed — LLM parse error.", "profile_candidates": [],
                "rule_candidates": [], "contradictions": [], "pruning_suggestions": []}

    # Write the reflection record
    reflection_id = write_reflection(
        user_id=user_id,
        project_id=project_id,
        session_id=session_id,
        reflection_type=mode,
        summary_text=data.get("summary", ""),
        profile_candidates=data.get("profile_candidates", []),
        rule_candidates=data.get("rule_candidates", []),
        contradictions=data.get("contradictions", []),
        pruning_suggestions=data.get("pruning_suggestions", []),
        confidence=0.7,
    )

    # Evaluate and write approved rule candidates (up to MAX_PROMOTIONS_PER_RUN)
    promotions_written = 0
    written_promotions: List[PromotionDecision] = []
    for rc in data.get("rule_candidates", []):
        if promotions_written >= MAX_PROMOTIONS_PER_RUN:
            break
        decision = check_rule_promotion(
            rule_text=rc.get("rule_text", ""),
            rule_kind=rc.get("rule_kind", "general"),
            confidence=float(rc.get("confidence", 0.0)),
            rationale=rc.get("rationale", ""),
        )
        if decision:
            write_promoted_rule(decision, user_id, project_id)
            written_promotions.append(decision)
            promotions_written += 1

    # Build profile candidates for the caller (not auto-written — require user or threshold gate)
    profile_candidates: List[ProfileWriteRequest] = []
    for pc in data.get("profile_candidates", []):
        profile_candidates.append(ProfileWriteRequest(
            user_id=user_id,
            project_id=project_id,
            scope="user",
            profile_type=pc.get("profile_type", "general"),
            key=pc.get("key", ""),
            value_text=pc.get("value_text", ""),
            confidence=float(pc.get("confidence", 0.7)),
            source_type="inferred",
        ))

    latency_ms = (time.monotonic() - t0) * 1000
    _LOG.info(
        "consolidator | reflection done mode=%s user=%s episodes=%d promotions=%d latency_ms=%.1f",
        mode, user_id, len(rows), promotions_written, latency_ms,
    )

    return ReflectionResult(
        reflection_id=reflection_id,
        summary_text=data.get("summary", ""),
        profile_candidates=profile_candidates,
        rule_candidates=written_promotions,
        contradictions_found=len(data.get("contradictions", [])),
        pruning_suggestions=len(data.get("pruning_suggestions", [])),
        parse_error=parse_error,
    )
