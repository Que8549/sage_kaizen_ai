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
