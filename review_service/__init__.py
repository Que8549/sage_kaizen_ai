"""
review_service — Sage Kaizen Architect Review Service

Triggered by chat phrases like "Review your codebase".
Runs entirely on the ARCHITECT brain (port 8012, CUDA0).
Human-gated: no files are written without explicit approval.

Entry points:
  ReviewRunner  — background thread orchestration (used by Streamlit UI)
  parse_review_command / is_review_command  — trigger detection
"""
from .trigger import is_review_command, parse_review_command, ReviewCommand
from .runner import ReviewRunner

__all__ = [
    "is_review_command",
    "parse_review_command",
    "ReviewCommand",
    "ReviewRunner",
]
