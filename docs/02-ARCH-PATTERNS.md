# 02 — Architectural Patterns Used in Sage Kaizen

This document describes the repeatable design patterns used across the system.

---

# 1. Dual-Brain Pattern

## Description

Two independent inference servers:
- FAST (default)
- ARCHITECT (deep reasoning)

## Why

- Performance isolation
- Deterministic escalation
- Maintainable complexity separation

## Rules

- FAST handles 80–90% of requests
- ARCHITECT only used intentionally
- Routing must be transparent

---

# 2. Template-First Prompting Pattern

Templates are applied before model invocation.

Examples:
- sage_fast_core
- sage_architect_core

Principles:
- Templates are composable
- Templates are auditable
- Per-turn override allowed
- No hidden prompt mutation

---

# 3. Tool Invocation Pattern

LLM may:
- Retrieve RAG context
- Send Pi agent commands
- Generate documentation

Constraints:
- Tool calls must be explicit
- Tool errors must be logged
- LLM never directly performs OS actions

---

# 4. Idempotent Ingestion Pattern

All ingestion:
- Has stable Source IDs
- Uses content hashing
- Supports safe re-runs
- Uses batch DB inserts

This prevents:
- Duplicate vector entries
- Partial failure corruption

---

# 5. Agent Transport Pattern (ZeroMQ)

Host:
- Master orchestrator

Pi:
- Worker nodes

Properties:
- Message schema versioned
- Timeouts explicit
- Commands validated before execution

---

# 6. Observability-First Pattern

Design assumption:
> If it’s not logged, it didn’t happen.

Startup readiness determined by:
- “server is listening”
- “slots idle”

No implicit readiness.

---

# 7. Replaceable Module Pattern

Each subsystem must be replaceable:
- STT
- TTS
- Vector DB
- LLM backend

Interfaces must remain stable.

---

# 8. Human-Gated State Machine Pattern

## Description

A LangGraph `StateGraph` workflow where execution pauses at a designated checkpoint
and requires explicit human approval before any file writes occur.

Used by: `review_service/`

## Why

- Reviews generate ADRs, patches, and reports — destructive-if-wrong artifacts
- The human gate ensures the ARCHITECT's synthesis is inspected before committing output
- Checkpoint persistence (PostgreSQL `langgraph` schema) survives a Streamlit reload
  between the interrupt and the resume

## Structure

```
scope_collector → subprocess_checks → web_researcher
  → architect_reviewer → flags_sanity → docs_drift
  → synthesizer → human_gate (interrupt())
                      ↓ approved=True      ↓ approved=False
                 output_writer → END       END (no files)
```

## Rules

- **interrupt() before all writes** — no file I/O before human approval
- **Checkpoint on every node step** — AsyncPostgresSaver writes state to PostgreSQL
  after each node; run is resumable after process restart
- **Isolated asyncio event loop per run** — ReviewRunner spawns a daemon thread with
  `asyncio.new_event_loop()` to avoid conflict with Streamlit's main thread
- **Sequential edges only** — ARCHITECT has `parallel: 1`; fan-out would queue at
  the HTTP server anyway; sequential edges let each node enrich the next node's prompt
- **Single checkpointer connection** — reuses `pg_settings.py` DSN; LangGraph tables
  live in the `langgraph` schema, not `public`

## Checkpoint Overhead - Monitor `pg_settings.py` DSN latency

Each node step issues one PostgreSQL round-trip (~1–5 ms).  With 8 nodes this adds
~8–40 ms per review run.  Review runs take several minutes (LLM inference dominates),
so checkpoint I/O is negligible.  Monitor `pg_settings.py` DSN latency if contention
with the feedback dataset is suspected.

## Checkpoint Overhead — No Code Change Needed As of April 9, 2026

The AsyncPostgresSaver is fully async (asyncpg), so each of the ~8 checkpoint writes per review is non-blocking and takes ~1–5 ms. Review runs are dominated by ARCHITECT inference (minutes per node). The only measurable overhead is connection pool open/close at start() and resume() (~50–200 ms total), which is architecturally correct as-is — persisting a pool across event loops would break the Streamlit isolation design. The analysis is now documented in Pattern #8.
