"""
review_service/graph.py — LangGraph state machine for the Architect Review Service.

Graph topology (all edges sequential — ARCHITECT port 8012 has parallel: 1):

    scope_collector
        → subprocess_checks   (pyright / ruff / pytest-collect — no LLM)
        → web_researcher      (SearXNG: performance, library updates)
        → architect_reviewer  (ARCHITECT: risks, design, naming, GPU, RAG/schema)
        → flags_sanity        (ARCHITECT: brains.yaml flag correctness)
        → docs_drift          (ARCHITECT: docs/ vs code divergence)
        → synthesizer         (ARCHITECT: merge all findings → final markdown)
        → human_gate          (interrupt() — pauses for human approval)
        → output_writer       (writes reviews/, adr/, patches/ — only if approved)
        → END

Why sequential and not Send fan-out:
  All LLM nodes share the same ARCHITECT endpoint (parallel: 1). Parallel
  dispatch would queue at the HTTP server anyway. Sequential edges let each
  node's output enrich the next node's prompt — flags_sanity sees
  architect_findings, docs_drift sees both, synthesizer sees everything.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .state import ReviewState
from .nodes.scope_collector import scope_collector_node
from .nodes.subprocess_checks import subprocess_checks_node
from .nodes.web_researcher import web_researcher_node
from .nodes.architect_reviewer import make_architect_reviewer_node
from .nodes.flags_sanity import make_flags_sanity_node
from .nodes.docs_drift import make_docs_drift_node
from .nodes.synthesizer import make_synthesizer_node
from .nodes.human_gate import human_gate_node
from .nodes.output_writer import output_writer_node

# ARCHITECT brain endpoint — source of truth in brains.yaml (port 8012, CUDA0)
_ARCHITECT_BASE_URL = "http://127.0.0.1:8012/v1"
_ARCHITECT_MODEL    = "Qwen3.5-27B-Q6_K"


def _make_architect_llm() -> ChatOpenAI:
    """
    Build the LangChain ChatOpenAI binding for the ARCHITECT brain.

    streaming=False: all review passes request complete structured output.
    The llama-server OpenAI-compat endpoint is transparent to LangChain.
    """
    return ChatOpenAI(
        base_url=_ARCHITECT_BASE_URL,
        api_key="none",
        model=_ARCHITECT_MODEL,
        temperature=0.2,     # low temperature for analytical / structured passes
        streaming=False,
        max_tokens=8192,
        timeout=300,         # 5 min — 27B reasoning pass can be slow on a large diff
    )


def build_review_graph(checkpointer) -> "CompiledStateGraph":  # type: ignore[name-defined]
    """
    Compile the review StateGraph with a PostgreSQL checkpointer.

    Called inside the ReviewRunner's isolated asyncio event loop — one graph
    per run so the checkpointer's connection pool is tied to the right loop.
    """
    architect = _make_architect_llm()

    builder: StateGraph = StateGraph(ReviewState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    builder.add_node("scope_collector",   scope_collector_node)
    builder.add_node("subprocess_checks", subprocess_checks_node)
    builder.add_node("web_researcher",    web_researcher_node)
    builder.add_node("architect_reviewer", make_architect_reviewer_node(architect))
    builder.add_node("flags_sanity",      make_flags_sanity_node(architect))
    builder.add_node("docs_drift",        make_docs_drift_node(architect))
    builder.add_node("synthesizer",       make_synthesizer_node(architect))
    builder.add_node("human_gate",        human_gate_node)
    builder.add_node("output_writer",     output_writer_node)

    # ── Edges — sequential pipeline ───────────────────────────────────────
    builder.set_entry_point("scope_collector")
    builder.add_edge("scope_collector",    "subprocess_checks")
    builder.add_edge("subprocess_checks",  "web_researcher")
    builder.add_edge("web_researcher",     "architect_reviewer")
    builder.add_edge("architect_reviewer", "flags_sanity")
    builder.add_edge("flags_sanity",       "docs_drift")
    builder.add_edge("docs_drift",         "synthesizer")
    builder.add_edge("synthesizer",        "human_gate")

    # human_gate calls interrupt() — execution pauses here until resume.
    # After resume, route: approved → output_writer; rejected → END.
    builder.add_conditional_edges(
        "human_gate",
        lambda s: "output_writer" if s.get("approved") else END,
        {"output_writer": "output_writer", END: END},
    )
    builder.add_edge("output_writer", END)

    return builder.compile(checkpointer=checkpointer)
