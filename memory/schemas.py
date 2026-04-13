"""
memory/schemas.py
Dataclasses that map directly to memory.* table rows.

These are used internally by repository.py to represent raw DB results before
they are converted to Pydantic DTOs (models.py) for the service layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ProfileRow:
    id: str
    user_id: str
    project_id: Optional[str]
    workspace_id: Optional[str]
    scope: str
    profile_type: str
    key: str
    value_text: str
    value_json: Optional[Dict[str, Any]]
    confidence: float
    source_type: str
    is_pinned: bool
    is_locked: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_confirmed_at: Optional[datetime]
    expires_at: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeRow:
    id: str
    user_id: str
    project_id: Optional[str]
    workspace_id: Optional[str]
    session_id: Optional[str]
    scope: str
    event_type: str
    intent_label: Optional[str]
    summary_text: str
    raw_excerpt: Optional[str]
    tags: List[str]
    importance: float
    confidence: float
    sentiment: Optional[float]
    was_user_correction: bool
    was_explicit_preference: bool
    contradiction_group: Optional[str]
    embedding: Optional[List[float]]
    created_at: datetime
    last_accessed_at: Optional[datetime]
    last_retrieved_at: Optional[datetime]
    expires_at: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleRow:
    id: str
    user_id: Optional[str]
    project_id: Optional[str]
    workspace_id: Optional[str]
    scope: str
    rule_kind: str
    rule_text: str
    rationale: Optional[str]
    confidence: float
    promotion_count: int
    source_type: str
    source_memory_id: Optional[str]
    is_locked: bool
    is_active: bool
    review_status: str
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionRow:
    id: str
    user_id: str
    project_id: Optional[str]
    workspace_id: Optional[str]
    session_id: Optional[str]
    reflection_type: str
    summary_text: str
    extracted_profile_candidates: List[Dict[str, Any]]
    extracted_rule_candidates: List[Dict[str, Any]]
    contradictions: List[Dict[str, Any]]
    pruning_suggestions: List[Dict[str, Any]]
    confidence: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
