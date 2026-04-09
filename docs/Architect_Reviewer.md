# Architect Reviewer — Service Guide

**Version**: 1.0  
**Owner**: Sage Kaizen  
**Updated**: 2026-04-08  
**Module**: `review_service/`

---

## 1. Purpose

The Architect Reviewer is a stateful, human-gated code review service built on LangGraph. It runs the ARCHITECT brain (Qwen3.5-27B-Q6_K, port 8012, CUDA0) against the Sage Kaizen codebase and its associated projects, produces structured findings, pauses for human approval, then writes review artifacts to disk.

It is **not a continuous process**. It activates only when a review trigger phrase is detected in the chat input, executes its pipeline in a background thread, and terminates. No autonomous writes occur without explicit user approval.

---

## 2. Trigger Phrases

Type any of these into the Sage Kaizen chat:

| Phrase | Mode | Scope |
|---|---|---|
| `Review your codebase` | `full` | All three projects, `--stat` + chunked file reads |
| `Review staged changes` | `staged` | `git diff --staged` (main app only) |
| `Review the file <path>` | `file` | Single file + imports + arch context |
| `Regression audit` | `regression` | `git diff HEAD~1 HEAD` |
| `Code review` | `full` | Alias for full mode |
| `Architect review` | `full` | Alias for full mode |

Trigger phrases bypass the injection guard (`check_user_input`) — they are treated as known-safe internal commands.

---

## 3. Projects in Scope

The Architect Reviewer has access to all three Sage Kaizen projects:

| Project | Path | Git Repo | Scope |
|---|---|---|---|
| Main app | `F:\Projects\sage_kaizen_ai\` | Yes | Full diff, file tree, arch docs |
| Voice app | `F:\Projects\sage_kaizen_ai_voice\` | Yes | Full diff, file tree |
| SearXNG | `F:\Projects\searxng\` | No | `docker-compose.yml`, config files |

---

## 4. Pipeline Stages

```
scope_collector
    └─ subprocess_checks   (pyright, ruff, pytest-collect — no LLM)
        └─ web_researcher  (SearXNG: performance, library updates, known issues)
            └─ architect_reviewer   (ARCHITECT: risks, design, naming, GPU, RAG)
                └─ flags_sanity     (ARCHITECT: brains.yaml flag correctness)
                    └─ docs_drift   (ARCHITECT: docs/ vs code divergence)
                        └─ synthesizer  (ARCHITECT: merge → final markdown)
                            └─ human_gate   (interrupt — awaits approval)
                                └─ output_writer  (reviews/, adr/, patches/)
```

All ARCHITECT calls are **sequential** (port 8012 has `parallel: 1`). Each node's output is available as context for the next node.

---

## 5. Scope Collection Strategy

### Staged / File / Regression Modes
Full diff text is collected and sent directly to ARCHITECT. Typical staged diffs are 1–20K tokens — well within the 128K context window.

### Full Repo Mode — Chunking Strategy

Full repo mode cannot send the complete diff (potentially 500K+ chars). Instead:

**Phase 1 — Inventory** (no LLM):
- Run `git diff main HEAD --stat` for main and voice repos
- Collect all changed file paths
- Read `docker-compose.yml` for SearXNG
- Collect full content of `docs/01-ARCHITECTURE.md`, `docs/02-ARCH-PATTERNS.md`, `CLAUDE.md`
- Collect `config/brains/brains.yaml`

**Phase 2 — Priority Scoring** (no LLM):
Files are ranked for ARCHITECT analysis in this order:
1. Python files with `>20` changed lines
2. Python files with `5–20` changed lines
3. Config files (`*.yaml`, `*.json`, `*.toml`)
4. Documentation files (`*.md`)
5. All others (test files, scripts, etc.)

**Phase 3 — Budget Fill**:
Read file content up to a **70,000 char budget** for diff material:
- Each file: up to 3,000 chars of unified diff
- Architecture docs: up to 15,000 chars (always included)
- brains.yaml: always included (~3,000 chars)
- Remaining budget split across priority-ranked files

**Phase 4 — Overflow Chunk** (if changed files > budget):
- Store remaining files as `overflow_files` list in state
- Synthesizer notes overflow in the report: "N files not reviewed due to context budget"

---

## 6. Web Research Integration

Before the ARCHITECT review pass, the `web_researcher` node queries SearXNG for up to 4 targeted searches derived from the changed modules. Search topics are auto-generated based on:

- Frameworks and libraries imported in changed files
- llama.cpp if `brains.yaml` changed
- LangGraph/LangChain if `review_service/` changed
- Performance keywords for any module touching inference or RAG

**Search categories**: `["it", "science"]` (tech + science)  
**Max results**: 12 per query (ARCHITECT brain ceiling)  
**Results injected as**: `<web_research>` block in the architect_reviewer prompt

The ARCHITECT brain uses these results to:
- Flag outdated dependency versions
- Recommend newer llama.cpp flags or model configurations
- Cite known issues or CVEs in used libraries
- Suggest performance improvements based on current best practices

If SearXNG is unavailable, the `web_researcher` node logs a warning and continues without web context — review does not fail.

---

## 7. Performance & Latency Review (Mandatory)

Every ARCHITECT pass **must** include a performance and latency audit. This is non-optional and applies to all review modes.

The ARCHITECT reviewer checks:

### Inference Performance
- KV cache hit rates: Is `slot_prompt_similarity` set appropriately?
- Batch sizes: `batch_size` / `ubatch_size` vs context window usage
- `flash_attn` enabled on all supported brains?
- Thread counts vs physical core count (AMD Ryzen 9 9950X3D: 16 cores / 32 threads)
- GPU layer assignment: All layers on GPU (`n_gpu_layers: all`)?
- Context size vs actual usage patterns — is the window larger than needed?

### Application Performance
- Blocking calls on Streamlit's main thread
- RAG retrieval latency — are vector indexes appropriate (HNSW vs IVFFlat)?
- Connection pooling — HTTP sessions reused across turns?
- Import time — heavy ML imports at module level instead of deferred?
- Async patterns — `asyncio.gather` used where parallel IO is possible?

### VRAM Budget
- Current VRAM allocations vs model sizes vs headroom
- KV cache size vs context window — are they consistent?
- Any VRAM fragmentation risk from parallel loads?

---

## 8. ARCHITECT Prompt Style

All ARCHITECT prompts in this service follow these rules:

1. **Structured output first**: Every analytical prompt requests JSON or markdown with defined sections. Free-form prose is used only in the synthesizer.
2. **Reference specificity**: Prompts instruct ARCHITECT to cite `file:line` references where possible.
3. **Severity tagging**: All findings must be tagged `[CRITICAL]`, `[HIGH]`, `[MEDIUM]`, or `[LOW]`.
4. **No hallucinated fixes**: ARCHITECT is instructed to only suggest patches for code it has actually read in context.
5. **Think tokens**: The `<think>` mode is active on ARCHITECT (Qwen3.5-27B supports extended reasoning). Prompts are designed to benefit from multi-step reasoning.

---

## 9. Human Gate — Non-Negotiable

The graph **pauses** at `human_gate` after synthesis. Nothing is written to disk until you approve.

- The synthesis document is displayed in the Streamlit UI
- Two buttons appear: **Approve — Write Files** and **Reject — Discard**
- Approval triggers `output_writer`; rejection ends the graph at END
- State is checkpointed to PostgreSQL — approval can happen minutes later without losing results

**Invariant**: `output_writer` is only reachable via `human_gate → approved=True`. The conditional edge in `graph.py` enforces this. No code path bypasses this gate.

---

## 10. Output Artifacts

### `reviews/YYYY-MM-DD-HHMM-{mode}-review.md`
Full synthesis report. Always written on approval.

### `docs/03-DECISIONS/ADR-YYYY-MM-DD-HHMM-architect-review.md`
Written only when `architect_findings` contains **CRITICAL or HIGH** architectural risks. Uses the existing ADR template from `docs/03-DECISIONS/0001-adr-template.md`.

### `reviews/patches/YYYY-MM-DD-HHMM-{slug}.patch`
Written for each suggested patch in `suggested_patches`. Format is unified diff (`--- a/file`, `+++ b/file`). These are suggestions only — they are never applied automatically.

---

## 11. Checkpoint Persistence

LangGraph state is checkpointed to PostgreSQL using `langgraph-checkpoint-postgres`.

- Tables: `checkpoints`, `checkpoint_blobs`, `checkpoint_migrations` (created automatically on first run via `asetup()`)
- Thread ID format: `review-YYYYMMDD-HHMMSS-{mode}`
- State survives process restart — if Streamlit restarts between scope collection and approval, the synthesized report is not lost
- `thread_id` is shown in the review status widget for manual recovery if needed

Connection: uses `pg_settings.py` DSN (same database as feedback dataset).

---

## 12. Invariants

These are inherited from `CLAUDE.md` and must never be violated by this service:

1. Never launch llama-server via `cmd.exe`
2. Never use stdout/stderr redirection for llama-server — `--log-file` only
3. Never recreate `.bat` files
4. ARCHITECT brain auto-started via `server_manager.ensure_q6_running()` if not running
5. No autonomous git commits, no autonomous branch creation
6. No autonomous merge of patch files
7. `interrupt_before` human gate is non-negotiable — no writes without approval

---

## 13. Configuration

No new config keys are required. The service reads:

| Config | Source |
|---|---|
| ARCHITECT URL | `brains.yaml → architect.server.port` → hardcoded `http://127.0.0.1:8012/v1` |
| PostgreSQL DSN | `pg_settings.py` → `.env` → `PG_USER / PG_PASSWORD / PG_DB` |
| SearXNG URL | `SAGE_SEARCH_URL` env var → default `http://localhost:8080` |
| Project roots | Hardcoded constants in `review_service/nodes/scope_collector.py` |

---

## 14. Running Manually (CLI)

The review service can also be run directly without Streamlit:

```bash
# Full repo review
python -m review_service.cli --mode full

# Staged changes only
python -m review_service.cli --mode staged

# Single file
python -m review_service.cli --mode file --target chat_service.py

# Regression audit
python -m review_service.cli --mode regression --target HEAD~1
```

CLI does not support the human gate interactively — it prints the synthesis and prompts for `y/n` approval in the terminal.

---

## 15. Observability

All nodes log to the Sage Kaizen rotating log via `sk_logging.get_logger("sage_kaizen.review_service.*")`.

Key log events:
- `review.start`: mode, thread_id
- `review.scope`: char counts for diff, docs, todos
- `review.web_research`: query terms, result counts
- `review.architect_call`: node name, prompt_chars, response_chars
- `review.interrupt`: synthesis ready, awaiting approval
- `review.approved` / `review.rejected`
- `review.output`: files written
- `review.error`: any unhandled exception
