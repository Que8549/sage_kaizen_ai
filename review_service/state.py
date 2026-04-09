"""
review_service/state.py — ReviewState TypedDict

Single source of truth for all state fields that flow through the
LangGraph review pipeline. Every node reads from and writes to this shape.
"""
from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class ReviewState(TypedDict):
    # ── Inputs (set at invocation) ─────────────────────────────────────────
    mode: str           # "full" | "staged" | "file" | "regression"
    target: str         # file path for "file" mode; base ref for "regression"

    # ── Scope collector output ────────────────────────────────────────────
    git_diff: str                  # raw unified diff text
    changed_files: list[str]       # list of changed file paths (relative)
    todo_markers: str              # grep output for TODO/FIXME across changed files
    arch_docs: str                 # docs/01-ARCHITECTURE.md + 02-ARCH-PATTERNS.md
    brains_yaml: str               # raw content of config/brains/brains.yaml
    file_tree: str                 # top-level dir listing of all projects
    overflow_files: list[str]      # files excluded from full-mode due to budget cap
    scope_char_count: int          # total chars collected (for budget tracking log)

    # ── Subprocess checks output (no LLM) ────────────────────────────────
    pyright_output: str
    ruff_output: str
    pytest_collect: str
    # full-mode only — set by subprocess_checks, consumed by code_quality_reviewer
    vulture_output: str       # dead code detected across entire source trees
    ruff_quality_output: str  # extended ruff rules: C90,B,SIM,UP,PERF,RUF,PIE

    # ── Web research output ───────────────────────────────────────────────
    web_research: str              # SearXNG results formatted as context block

    # ── ARCHITECT pass outputs ────────────────────────────────────────────
    architect_findings: str        # structured: risks, design, naming, GPU, RAG/schema
    flags_findings: str            # brains.yaml flag sanity
    docs_findings: str             # docs/ drift vs code
    code_quality_findings: str     # ARCHITECT: smells, dead code, optimizations (full only)

    # ── Synthesis + human gate ────────────────────────────────────────────
    synthesis: str                 # final merged markdown from synthesizer
    approved: bool                 # set True by human_gate after interrupt() resumes

    # ── Output ────────────────────────────────────────────────────────────
    output_paths: list[str]        # written file paths (review.md, adr.md, .patch files)
    error: Optional[str]           # fatal error message if any node fails


def default_state(mode: str, target: str = "") -> ReviewState:
    """Return a ReviewState pre-filled with safe defaults for a new run."""
    return ReviewState(
        mode=mode,
        target=target,
        git_diff="",
        changed_files=[],
        todo_markers="",
        arch_docs="",
        brains_yaml="",
        file_tree="",
        overflow_files=[],
        scope_char_count=0,
        pyright_output="",
        ruff_output="",
        pytest_collect="",
        vulture_output="",
        ruff_quality_output="",
        web_research="",
        architect_findings="",
        flags_findings="",
        docs_findings="",
        code_quality_findings="",
        synthesis="",
        approved=False,
        output_paths=[],
        error=None,
    )
