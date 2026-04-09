"""
review_service — Sage Kaizen Architect Review Service

Triggered by chat phrases like "Review your codebase".
Runs entirely on the ARCHITECT brain (port 8012, CUDA0).
Human-gated: no files are written without explicit approval.

Entry points (import directly from sub-modules to keep startup cost zero):
  from review_service.trigger import is_review_command, parse_review_command
  from review_service.runner  import ReviewRunner   # lazy — pulls langchain_core

DO NOT import ReviewRunner here at package level. langchain_core takes ~2.6s
to import on Python 3.14 and must not run during Streamlit startup.
"""
# Intentionally empty — no eager imports.
# Import trigger symbols directly; import ReviewRunner lazily at point of use.

