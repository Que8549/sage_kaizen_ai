"""
memory/schemas.py
Dataclasses that map directly to memory.* table rows.

These are used internally by repository.py to represent raw DB results before
they are converted to Pydantic DTOs (models.py) for the service layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ProfileRow:
    id: str
    user_id: str
    project_id: str | None
    workspace_id: str | None
    scope: str
    profile_type: str
    key: str
    value_text: str
    value_json: dict[str, Any] | None
    confidence: float
    source_type: str
    is_pinned: bool
    is_locked: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_confirmed_at: datetime | None
    expires_at: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeRow:
    id: str
    user_id: str
    project_id: str | None
    workspace_id: str | None
    session_id: str | None
    scope: str
    event_type: str
    intent_label: str | None
    summary_text: str
    raw_excerpt: str | None
    tags: list[str]
    importance: float
    confidence: float
    sentiment: float | None
    was_user_correction: bool
    was_explicit_preference: bool
    contradiction_group: str | None
    embedding: list[float] | None
    created_at: datetime
    last_accessed_at: datetime | None
    last_retrieved_at: datetime | None
    expires_at: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleRow:
    id: str
    user_id: str | None
    project_id: str | None
    workspace_id: str | None
    scope: str
    rule_kind: str
    rule_text: str
    rationale: str | None
    confidence: float
    promotion_count: int
    source_type: str
    source_memory_id: str | None
    is_locked: bool
    is_active: bool
    review_status: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionRow:
    id: str
    user_id: str
    project_id: str | None
    workspace_id: str | None
    session_id: str | None
    reflection_type: str
    summary_text: str
    extracted_profile_candidates: list[dict[str, Any]]
    extracted_rule_candidates: list[dict[str, Any]]
    contradictions: list[dict[str, Any]]
    pruning_suggestions: list[dict[str, Any]]
    confidence: float
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
