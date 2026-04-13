"""
memory/models.py
Pydantic DTOs for the Sage Kaizen Memory Service.

These are the types passed between service boundaries (service layer → router,
bundle_builder → prompt assembly, etc.).  They do NOT map 1:1 to DB columns;
for DB row types see schemas.py.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request / Context
# ---------------------------------------------------------------------------

class MemoryContextRequest(BaseModel):
    """Everything the retriever needs to build a memory bundle for one turn."""
    user_id: str
    project_id: str = "sage_kaizen"
    workspace_id: Optional[str] = None
    session_id: Optional[str] = None
    query_text: str                          # raw or normalized user message
    intent_label: Optional[str] = None      # e.g. "code_help", "creative", "tutor"
    route_target: str = "fast"              # "fast" | "architect"
    tags: List[str] = Field(default_factory=list)
    # Per-brain budget cap (tokens).  None = use the brain-specific default
    # (FAST=600, ARCHITECT=1500) resolved in service.py.
    max_bundle_tokens: Optional[int] = None


# ---------------------------------------------------------------------------
# Individual memory items (what gets injected into the prompt)
# ---------------------------------------------------------------------------

class ProfileMemoryItem(BaseModel):
    id: str
    profile_type: str
    key: str
    value_text: str
    confidence: float
    scope: str
    is_pinned: bool
    source_type: str


class EpisodeMemoryItem(BaseModel):
    id: str
    event_type: str
    summary_text: str
    tags: List[str]
    importance: float
    confidence: float
    was_user_correction: bool
    was_explicit_preference: bool
    created_at: datetime
    # Score assigned by the ranker (not persisted)
    retrieval_score: float = 0.0


class RuleMemoryItem(BaseModel):
    id: str
    rule_kind: str
    rule_text: str
    confidence: float
    is_locked: bool
    review_status: str


# ---------------------------------------------------------------------------
# Bundle — what gets injected before model inference
# ---------------------------------------------------------------------------

class MemoryBundle(BaseModel):
    """A fully assembled, token-budgeted memory context for one turn."""
    profiles: List[ProfileMemoryItem] = Field(default_factory=list)
    rules: List[RuleMemoryItem] = Field(default_factory=list)
    episodes: List[EpisodeMemoryItem] = Field(default_factory=list)
    total_items: int = 0
    estimated_tokens: int = 0
    # Set True when the bundle was truncated to fit the token cap
    was_truncated: bool = False


# ---------------------------------------------------------------------------
# Write inputs
# ---------------------------------------------------------------------------

class EpisodeWriteRequest(BaseModel):
    """Input for Path B (episodic write) in writer.py."""
    user_id: str
    project_id: str = "sage_kaizen"
    workspace_id: Optional[str] = None
    session_id: Optional[str] = None
    scope: str = "project"
    event_type: str
    intent_label: Optional[str] = None
    summary_text: str
    raw_excerpt: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.6
    was_user_correction: bool = False
    was_explicit_preference: bool = False


class ProfileWriteRequest(BaseModel):
    """Input for Path A (explicit profile write) in writer.py."""
    user_id: str
    project_id: str = "sage_kaizen"
    workspace_id: Optional[str] = None
    scope: str = "user"
    profile_type: str
    key: str
    value_text: str
    confidence: float = 1.0
    source_type: str = "explicit_user"
    is_locked: bool = False


# ---------------------------------------------------------------------------
# Consolidation / Reflection outputs
# ---------------------------------------------------------------------------

class ReflectionResult(BaseModel):
    """Output from consolidator.run_reflection()."""
    reflection_id: str
    summary_text: str
    profile_candidates: List[ProfileWriteRequest] = Field(default_factory=list)
    rule_candidates: List["PromotionDecision"] = Field(default_factory=list)
    contradictions_found: int = 0
    pruning_suggestions: int = 0
    parse_error: bool = False   # True when the LLM returned unparseable JSON


class PromotionDecision(BaseModel):
    """A candidate rule promotion, returned by policy.check_promotions()."""
    source_memory_id: Optional[str] = None
    rule_text: str
    rule_kind: str
    confidence: float
    rationale: str
    approved: bool = False             # True after human or architect confirmation
    scope: str = "project"
