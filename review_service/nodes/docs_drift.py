"""
review_service/nodes/docs_drift.py — Documentation drift check.

Compares the architecture documentation and CLAUDE.md against the recent
code changes to find sections that no longer accurately describe the system.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sk_logging import get_logger
from ..state import ReviewState

_LOG = get_logger("sage_kaizen.review_service.docs_drift")

_SYSTEM_PROMPT = """\
You are checking whether Sage Kaizen's documentation accurately reflects the current code.

You will receive the current architecture docs (CLAUDE.md, 01-ARCHITECTURE.md,
02-ARCH-PATTERNS.md), the list of recently changed files, and the git diff.

Check for documentation drift:

1. Module responsibilities:
   Did any changed file alter a module's responsibilities as described in 01-ARCHITECTURE.md?
   (e.g. chat_service.py now handles X but the doc says it doesn't)

2. Service/Port inventory in CLAUDE.md:
   Were any ports, services, or models added/changed/removed without updating the table?

3. New public functions or classes:
   Are new public APIs in changed files absent from the docs?

4. CLAUDE.md invariants:
   Were any of the hard invariants (section 5 of CLAUDE.md) potentially violated
   by the changed code? (bat files, cmd.exe, stdout redirect for llama-server, etc.)

5. Memory file:
   Should .claude/projects/f--Projects-sage-kaizen-ai/memory/MEMORY.md be updated
   to reflect the changes? List the specific memory entry(ies) that need updating.

6. Architect_Reviewer.md:
   Does docs/Architect_Reviewer.md need any updates to reflect code changes?

Output format: markdown bullet list with:
  - The specific doc file and section that needs updating
  - What is wrong / outdated
  - Suggested correction (brief)

If docs are current, say "Documentation appears current — no drift detected."
"""


def make_docs_drift_node(llm: ChatOpenAI):
    async def docs_drift_node(state: ReviewState) -> dict:
        context = _build_context(state)
        _LOG.info("review.architect_call | node=docs_drift | context_chars=%d", len(context))
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
        try:
            response = await llm.ainvoke(messages)
            findings = response.content
        except Exception as exc:
            _LOG.exception("docs_drift failed: %s", exc)
            findings = f"[docs_drift ERROR: {exc}]"

        _LOG.info("review.architect_call | node=docs_drift | response_chars=%d", len(findings))
        return {"docs_findings": findings}

    return docs_drift_node


def _build_context(state: ReviewState) -> str:
    parts: list[str] = []

    if state.get("arch_docs"):
        parts.append(f"<arch_docs>\n{state['arch_docs'][:12000]}\n</arch_docs>")

    if state.get("changed_files"):
        files_str = "\n".join(state["changed_files"])
        parts.append(f"<changed_files>\n{files_str}\n</changed_files>")

    if state.get("git_diff"):
        # Send a condensed version — docs drift doesn't need full diff content
        diff_preview = state["git_diff"][:8000]
        parts.append(f"<diff_preview>\n{diff_preview}\n</diff_preview>")

    return "\n\n".join(parts)
