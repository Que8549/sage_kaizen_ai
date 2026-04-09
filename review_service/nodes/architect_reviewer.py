"""
review_service/nodes/architect_reviewer.py — Main ARCHITECT analysis node.

Single ARCHITECT call that produces a structured review covering:
  - architecture_risks
  - design_consistency
  - naming_api_issues
  - gpu_placement_concerns (always includes VRAM budget + latency audit)
  - rag_schema_drift
  - performance_and_latency   (mandatory every review)
  - suggested_patches         (file:line references; no hallucinated fixes)

The response is stored as raw text in state["architect_findings"].
If ARCHITECT returns structured JSON, it is preserved as-is.
If it returns prose markdown, it is also preserved — synthesizer handles both.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sk_logging import get_logger
from ..state import ReviewState

_LOG = get_logger("sage_kaizen.review_service.architect_reviewer")

_SYSTEM_PROMPT = """\
You are reviewing the Sage Kaizen AI codebase as its lead architect.
Sage Kaizen is a local AI system with:
  - FAST brain: Qwen2.5-Omni-7B-Q8_0, port 8011, CUDA1 (RTX 5080, 16 GB)
  - ARCHITECT brain: Qwen3.5-27B-Q6_K, port 8012, CUDA0 (RTX 5090, 32 GB)
  - BGE-M3 embed: port 8020, CUDA0
  - Streamlit UI + ZeroMQ voice bridge + PostgreSQL RAG (pgvector)
  - All servers launched via server_manager.py reading brains.yaml — no .bat files, no cmd.exe

You will receive:
  <git_diff>     — recent code changes (unified diff or --stat)
  <arch_docs>    — architecture docs (CLAUDE.md, 01-ARCHITECTURE.md, 02-ARCH-PATTERNS.md)
  <brains_yaml>  — llama-server configuration
  <todos>        — TODO/FIXME markers in changed files
  <static>       — pyright + ruff output
  <web_research> — SearXNG results for current best practices (may be empty)

Produce a structured review with these exact sections.
Tag every finding: [CRITICAL] / [HIGH] / [MEDIUM] / [LOW].

## architecture_risks
Module boundary violations, tight coupling, layering issues, service contract breakage.

## design_consistency
Deviations from established patterns: dual-brain routing, prompt_library single source,
RAG injection via context_injector, server_manager + brains.yaml config authority.

## naming_api_issues
Inconsistent naming, public function signature changes without updating callers,
TypedDict field changes, return type mismatches.

## gpu_placement_concerns
Wrong CUDA device assignments, VRAM budget overflows, flash_attn + mmproj interaction hazards,
KV cache sizing vs ctx_size, split_mode conflicts, thread count vs CPU core count.
Reference specific brains.yaml entries by name.

## rag_schema_drift
pgvector schema changes, embedding dimension mismatches, ingest/retrieve contract drift,
HNSW index parameter changes, table additions without migration scripts.

## performance_and_latency
MANDATORY — include in every review regardless of what changed.
Check:
  - Blocking calls on Streamlit main thread
  - Connection reuse (HTTP sessions, psycopg pools)
  - Async/await usage — are asyncio.gather opportunities missed?
  - Import-time cost — heavy ML imports at module level?
  - RAG retrieval path latency
  - KV cache effectiveness (slot_prompt_similarity, cache_ram settings)
  - Thread counts vs AMD Ryzen 9 9950X3D physical cores (16c / 32t)
  - Any newer llama.cpp flags or Qwen model config from web research
  Use web_research results to cite current best practices if available.

## suggested_patches
For each actionable finding, provide:
  - file: path/to/file.py
  - line: approximate line number (if known from diff context)
  - description: what to change and why
  - severity: CRITICAL|HIGH|MEDIUM|LOW
  - code_block: short before/after diff (ONLY if you have the actual code in context)
    Mark with "# APPROXIMATE" if line numbers are estimated.

## severity_summary
Counts: critical=N, high=N, medium=N, low=N

Instructions:
- Reference file:line where visible in the diff.
- Do NOT hallucinate fixes for code you have not seen in context.
- Do NOT suggest re-adding flash_attn Python package (SM_120 Blackwell unsupported).
- Do NOT suggest .bat files or cmd.exe for llama-server.
- Cite web_research URLs when recommending specific versions or flags.
"""


def make_architect_reviewer_node(llm: ChatOpenAI):
    async def architect_reviewer_node(state: ReviewState) -> dict:
        context = _build_context(state)
        _LOG.info(
            "review.architect_call | node=architect_reviewer | context_chars=%d",
            len(context),
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
        try:
            response = await llm.ainvoke(messages)
            findings = response.content
        except Exception as exc:
            _LOG.exception("architect_reviewer failed: %s", exc)
            findings = f"[architect_reviewer ERROR: {exc}]"

        _LOG.info("review.architect_call | node=architect_reviewer | response_chars=%d", len(findings))
        return {"architect_findings": findings}

    return architect_reviewer_node


def _build_context(state: ReviewState) -> str:
    parts: list[str] = []

    if state.get("git_diff"):
        parts.append(f"<git_diff>\n{state['git_diff']}\n</git_diff>")

    if state.get("arch_docs"):
        parts.append(f"<arch_docs>\n{state['arch_docs']}\n</arch_docs>")

    if state.get("brains_yaml"):
        parts.append(f"<brains_yaml>\n{state['brains_yaml']}\n</brains_yaml>")

    if state.get("todo_markers"):
        parts.append(f"<todos>\n{state['todo_markers']}\n</todos>")

    static_parts: list[str] = []
    if state.get("pyright_output"):
        static_parts.append(f"pyright:\n{state['pyright_output']}")
    if state.get("ruff_output"):
        static_parts.append(f"ruff:\n{state['ruff_output']}")
    if static_parts:
        parts.append("<static>\n" + "\n\n".join(static_parts) + "\n</static>")

    if state.get("web_research"):
        parts.append(state["web_research"])

    overflow = state.get("overflow_files", [])
    if overflow:
        parts.append(
            f"<overflow_note>\n"
            f"{len(overflow)} files were excluded from this review due to context budget limits.\n"
            f"Excluded: {', '.join(overflow[:20])}\n"
            f"</overflow_note>"
        )

    return "\n\n".join(parts)
