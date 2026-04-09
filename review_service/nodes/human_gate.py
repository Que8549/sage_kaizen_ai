"""
review_service/nodes/human_gate.py — Human approval gate.

Pauses LangGraph execution via interrupt(). State is checkpointed to
PostgreSQL at this exact point. The Streamlit UI reads the interrupt
payload and shows Approve / Reject buttons.

Execution resumes only when ReviewRunner.resume(approved) delivers
Command(resume=True/False) on the same thread_id.

INVARIANT:
  - interrupt() is NOT wrapped in try/except — the interrupt exception
    must propagate to LangGraph's runtime.
  - output_writer is only reachable when approved == True.
    The conditional edge in graph.py enforces this independently.
"""
from __future__ import annotations

from langgraph.types import interrupt

from sk_logging import get_logger
from ..state import ReviewState

_LOG = get_logger("sage_kaizen.review_service.human_gate")


def human_gate_node(state: ReviewState) -> dict:
    """
    Pause for human approval.

    The interrupt() payload is the dict surfaced in the Streamlit
    review status widget (synthesis text + prompt).

    Returns {"approved": bool} — the value passed to Command(resume=...).
    """
    _LOG.info("review.interrupt | awaiting human approval")

    # interrupt() raises a special LangGraph exception caught by the runtime.
    # The return value is whatever was passed to Command(resume=<value>).
    approved = interrupt({
        "synthesis": state.get("synthesis", ""),
        "prompt": (
            "Architect review complete. "
            "Approve to write findings to reviews/ and docs/adr/?"
        ),
    })

    # approved is the raw value from Command(resume=...) — coerce to bool
    result = bool(approved)
    _LOG.info("review.gate_response | approved=%s", result)
    return {"approved": result}
