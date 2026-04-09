"""
review_service/nodes/code_quality_reviewer.py — Python code quality analysis node.

Full-mode only. Skips immediately (returns empty findings) for all other modes.

Runs AFTER subprocess_checks (which supplies vulture_output and ruff_quality_output)
and BEFORE web_researcher. Produces state["code_quality_findings"] consumed by
the synthesizer.

What this node covers
---------------------
  dead_code              — unused functions/classes/methods/variables (vulture)
  code_smells            — high complexity, long functions, magic numbers, God classes
  optimization_opportunities — asyncio.gather gaps, import-time cost, redundant work
  best_practice_violations   — type-hint gaps, bare except:, mutable defaults, deprecated patterns
  performance_antipatterns   — blocking calls in async, missing connection reuse, tensor ops

Why a separate node (not merged into architect_reviewer)
---------------------------------------------------------
  - architect_reviewer focuses on architectural risk and design — different concern level
  - Keeps prompts focused; each LLM call has a single well-defined responsibility
  - Code quality output is long enough (vulture + ruff) to warrant its own context window
  - Synthesizer merges all findings; separation does not cost a second review pass
    at the UI level
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sk_logging import get_logger
from ..state import ReviewState

_LOG = get_logger("sage_kaizen.review_service.code_quality_reviewer")

_SYSTEM_PROMPT = """\
You are a Python code quality specialist reviewing the Sage Kaizen AI codebase.

Sage Kaizen context:
  - Python 3.14 on Windows 11 / CUDA 13.1
  - Asyncio throughout (Streamlit main thread + background daemon threads for reviews)
  - PostgreSQL via psycopg3 (async), httpx for HTTP, LangChain / LangGraph for review orchestration
  - Heavy ML imports (torch, transformers, langchain_core) that carry significant import-time cost

You will receive:
  <vulture_output>    — dead code detected by vulture --min-confidence 80 across both project roots
  <ruff_quality>      — extended ruff rules: C90 (complexity), B (bugbear), SIM (simplify),
                        UP (pyupgrade), PERF (performance antipatterns), RUF, PIE
  <git_diff_preview>  — recent code changes for context (may be truncated)

Produce a structured code quality report with EXACTLY these sections.
Tag every finding: [CRITICAL] / [HIGH] / [MEDIUM] / [LOW].

## dead_code
Unused functions, classes, methods, and variables identified by vulture.
Group by file. For each item include:
  - name and type (function / class / method / variable / attribute)
  - file:line
  - confidence %
  - recommendation: DELETE or KEEP (if KEEP, explain why — e.g. LangGraph node callback,
    TypedDict field used as dict key, pytest fixture, Streamlit callback, __dunder__ method,
    __all__ export, dataclass field)

Known false-positive patterns to exclude from the report:
  - Functions passed to builder.add_node() or passed as callbacks — they appear unused to vulture
  - TypedDict fields — accessed as dict["key"], not as attributes
  - pytest fixture functions
  - Streamlit widget callback functions (on_change=, on_click=)
  - __dunder__ methods and __all__ lists
  - Fields decorated with @property or @staticmethod

## code_smells
Structural issues that reduce maintainability:
  - Functions with cyclomatic complexity > 10 (cite ruff C90 output: file:line function complexity=N)
  - Functions estimated > 50 lines (note as estimate if not confirmed by diff)
  - Magic numbers: hardcoded numeric literals that should be named constants
    (exclude obvious ones: 0, 1, -1, port numbers already defined in brains.yaml)
  - God classes: classes with > 8 public methods or mixing unrelated responsibilities
  - Deeply nested blocks: > 3 levels of indentation for non-trivial logic
  - Long parameter lists: > 5 non-self parameters on a single function

## optimization_opportunities
  - Multiple sequential awaits on independent coroutines that could use asyncio.gather()
  - Repeated identical dict/list comprehension or computation inside a loop
  - Import-time cost: heavy ML packages (torch, transformers, langchain_core, langchain_openai)
    imported at module level in files that are always imported at Streamlit startup
    (flag only if the import is NOT guarded by lazy loading or TYPE_CHECKING)
  - Pure deterministic functions called in hot paths that could use functools.lru_cache
  - Intermediate list() or list comprehension wrapping a generator where simple iteration suffices

## best_practice_violations
  - Public functions / methods missing type annotations on parameters or return type
    (do NOT flag private _ methods or test helpers)
  - Bare except: clauses — catching BaseException silently; cite file:line
  - Mutable default arguments: def f(x=[], ...) or def f(x={}, ...)
  - String concatenation in loops that should use str.join()
  - Deprecated Python patterns flagged by UP rules (cite the ruff UP code)
  - isinstance() calls that could be replaced with structural pattern matching (Python 3.10+)
  - Missing context managers for resources (file handles, DB cursors opened without with:)

## performance_antipatterns
  - Blocking I/O inside async def: open(), requests.get(), subprocess.run() without
    asyncio.to_thread() or run_in_executor()
  - httpx.Client (sync) used where httpx.AsyncClient is appropriate
  - psycopg connection created per-call rather than reused from a pool
  - Torch / numpy operations on CPU when GPU tensors are already loaded
  - Streamlit: heavy computation in the main render body without @st.cache_data or @st.cache_resource
  - Items flagged by ruff PERF rules (cite the PERF code and file:line)
  - Large string concatenations in tight loops (should use io.StringIO or list+join)

## severity_summary
Counts: critical=N, high=N, medium=N, low=N
One-sentence overall assessment.

Instructions:
  - Reference file:line from static analysis output wherever available.
  - Vulture false positives are common — apply the exclusion rules above rigorously.
    Do NOT report an item as dead code unless you are confident it is genuinely unused.
  - Do NOT suggest removing code without confirming it is unreachable.
  - Keep each finding to 1–2 sentences: what the problem is + what to do.
  - Do NOT suggest adding flash_attn Python package (SM_120 Blackwell unsupported on Windows).
  - Do NOT suggest .bat files or cmd.exe launches.
"""


def make_code_quality_reviewer_node(llm: ChatOpenAI):
    async def code_quality_reviewer_node(state: ReviewState) -> dict:
        # Only meaningful for full-mode reviews
        if state.get("mode") != "full":
            _LOG.info(
                "review.code_quality | skipped (mode=%s, full mode only)", state.get("mode")
            )
            return {"code_quality_findings": ""}

        vulture = state.get("vulture_output", "")
        ruff_q  = state.get("ruff_quality_output", "")

        if not vulture.strip() and not ruff_q.strip():
            _LOG.info("review.code_quality | skipped (no vulture/ruff-quality output)")
            return {"code_quality_findings": ""}

        context = _build_context(state)
        _LOG.info(
            "review.architect_call | node=code_quality_reviewer | context_chars=%d",
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
            _LOG.exception("code_quality_reviewer failed: %s", exc)
            findings = f"[code_quality_reviewer ERROR: {exc}]"

        _LOG.info(
            "review.architect_call | node=code_quality_reviewer | response_chars=%d",
            len(findings),
        )
        return {"code_quality_findings": findings}

    return code_quality_reviewer_node


def _build_context(state: ReviewState) -> str:
    parts: list[str] = []

    vulture = state.get("vulture_output", "").strip()
    if vulture:
        parts.append(f"<vulture_output>\n{vulture}\n</vulture_output>")
    else:
        parts.append("<vulture_output>\n[vulture not installed or no dead code detected]\n</vulture_output>")

    ruff_q = state.get("ruff_quality_output", "").strip()
    if ruff_q:
        parts.append(f"<ruff_quality>\n{ruff_q}\n</ruff_quality>")
    else:
        parts.append("<ruff_quality>\n[no extended ruff findings]\n</ruff_quality>")

    # Provide a short diff preview for code context (not the full diff)
    git_diff = state.get("git_diff", "")
    if git_diff:
        parts.append(f"<git_diff_preview>\n{git_diff[:3_000]}\n</git_diff_preview>")

    return "\n\n".join(parts)
