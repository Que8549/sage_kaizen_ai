"""
review_service/nodes/synthesizer.py — Final synthesis node.

Merges all findings from architect_reviewer, flags_sanity, docs_drift,
and subprocess checks into a single coherent markdown report.

This is the final ARCHITECT call before the human gate.
The output (state["synthesis"]) is displayed to the user verbatim.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sk_logging import get_logger
from ..state import ReviewState

_LOG = get_logger("sage_kaizen.review_service.synthesizer")

_SYSTEM_PROMPT = """\
You are writing the final Sage Kaizen Architect Review report.
Merge all findings below into a single well-structured markdown document.

The user will read this report and decide whether to approve writing it to disk.
Be precise, actionable, and prioritized. Avoid repetition.

Required structure:

## Executive Summary
2–3 sentences: what changed, overall risk level, key concerns.
Include verdict: PASS | PASS_WITH_NOTES | NEEDS_WORK | BLOCK

## Critical & High Findings
Each finding:
  ### [SEVERITY] Finding Title
  **File**: path/to/file.py (line N if known)
  **Issue**: description
  **Action**: specific recommended fix

## Medium Findings
Same format as above.

## Low / Style Findings
Brief bullets only.

## Performance & Latency Notes
Specific latency improvements identified. Include any version upgrades
or flag changes recommended from web research (cite URLs).

## Documentation Updates Required
List doc files that need updating with specific sections.

## Patch Suggestions
For each patch in suggested_patches from the architect findings:
  ### Patch: <description>
  **File**: path
  **Priority**: CRITICAL|HIGH|MEDIUM|LOW
  ```diff
  <before/after code block — only if actual code was in context>
  ```

## Verdict
PASS            — No significant issues; safe to merge.
PASS_WITH_NOTES — Minor issues; document and continue.
NEEDS_WORK      — Several issues require addressing before merge.
BLOCK           — Critical architectural issues; do not merge.

Include a one-line justification for the verdict.
"""


def make_synthesizer_node(llm: ChatOpenAI):
    async def synthesizer_node(state: ReviewState) -> dict:
        context = _build_context(state)
        _LOG.info("review.architect_call | node=synthesizer | context_chars=%d", len(context))
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
        try:
            response = await llm.ainvoke(messages)
            synthesis = response.content
        except Exception as exc:
            _LOG.exception("synthesizer failed: %s", exc)
            synthesis = f"[synthesizer ERROR: {exc}]\n\nRaw findings follow:\n\n{_raw_dump(state)}"

        _LOG.info("review.architect_call | node=synthesizer | synthesis_chars=%d", len(synthesis))
        return {"synthesis": synthesis}

    return synthesizer_node


def _build_context(state: ReviewState) -> str:
    parts: list[str] = []

    if state.get("architect_findings"):
        parts.append(f"<architect_findings>\n{state['architect_findings']}\n</architect_findings>")

    if state.get("flags_findings"):
        parts.append(f"<flags_findings>\n{state['flags_findings']}\n</flags_findings>")

    if state.get("docs_findings"):
        parts.append(f"<docs_findings>\n{state['docs_findings']}\n</docs_findings>")

    static_parts: list[str] = []
    if state.get("pyright_output") and "error" in state["pyright_output"].lower():
        static_parts.append(f"pyright:\n{state['pyright_output'][:1500]}")
    if state.get("ruff_output") and state["ruff_output"].strip() and "not installed" not in state["ruff_output"]:
        static_parts.append(f"ruff:\n{state['ruff_output'][:1000]}")
    if static_parts:
        parts.append("<static_analysis>\n" + "\n\n".join(static_parts) + "\n</static_analysis>")

    overflow = state.get("overflow_files", [])
    if overflow:
        parts.append(
            f"<note>Full-mode review: {len(overflow)} files not reviewed due to context budget. "
            f"List: {', '.join(overflow[:10])}{'...' if len(overflow) > 10 else ''}</note>"
        )

    if state.get("web_research"):
        parts.append(state["web_research"])

    return "\n\n".join(parts)


def _raw_dump(state: ReviewState) -> str:
    """Fallback: dump raw findings if synthesizer LLM call fails."""
    return (
        f"architect_findings:\n{state.get('architect_findings', '[empty]')}\n\n"
        f"flags_findings:\n{state.get('flags_findings', '[empty]')}\n\n"
        f"docs_findings:\n{state.get('docs_findings', '[empty]')}\n"
    )
