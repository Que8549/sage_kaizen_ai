# AGENTS.md — How to Develop Sage Kaizen (WHO / WHAT / WHY / HOW)
This file is the “README for agents” (human or AI) working in this repo.  
It is designed for **progressive disclosure**: quick start → validation → conventions → deep links.

---

## 1) WHO
You are an agent helping the project owner (Alquin) evolve Sage Kaizen safely.
Assume:
- Windows 11 dev environment in VS Code
- Python 3.14.3
- Dual Nvidia GPUs (5090 + 5080)
- llama.cpp servers are the inference backbone

---

## 2) WHAT (What you’ll be changing)
Typical work areas:
- Streamlit UI chat experience + server status indicators (`ui_streamlit_server.py`)
- Router logic (FAST vs ARCHITECT, template application, escalation) (`router.py`)
- Chat service turn lifecycle (memory, prompts, RAG, streaming) (`chat_service.py`)
- llama-server orchestration (process management, logging, GPU flags) (`server_manager.py`)
- RAG ingestion (folder/rss/web/wiki/media) + retrieval pipeline (`rag_v1/`)
- Memory system (episode retrieval, profiles, learned rules) (`memory/`)
- Live web search integration (SearXNG, scoring, summarization) (`search/`)
- News collection and enrichment (`news/`)
- Prompt-injection defense (`input_guard.py`)
- Multi-format document parsing (`document_parser.py`)
- Pi agent orchestration (ZeroMQ command schema + safety) (`agents/` — planned)
- Review service (LangGraph, human gate, ADR/patch generation) (`review_service/`)

---

## 3) WHY (What “good” looks like)
Sage Kaizen is meant to be:
- Local-first, modular, and production-minded
- Observable (logs), testable, reproducible
- Accurate by default; performance optimizations must be measured and safe

---

## 4) HOW (How to work here)
### Required workflow: RPI Loop (Research → Plan → Implement → Validate)
1. Research: locate the current implementation and read logs
2. Plan: propose minimal files touched + success criteria
3. Implement: small diffs, typed Python, clear logging
4. Validate: run checks; cite the log lines or observed behavior

### Non-negotiables (must comply)
See `CLAUDE.md` invariants — do not violate them.

---

## 5) Progressive “Start Here” Guide (for agents)
### If you are new to this repo
1) `docs/01-ARCHITECTURE.md`  
2) `docs/10-RUNBOOKS/01-LLAMA-SERVERS.md`  
3) `docs/10-RUNBOOKS/02-STREAMLIT-UI.md`  
4) `docs/20-PROMPTS.md`  
5) Then make changes.

### If you are fixing a bug
1) Reproduce it
2) Find the log evidence
3) Identify the smallest responsible fix
4) Add/adjust logging so it’s easier next time
5) Validate fix (commands + output)

---

## 6) Validation Checklists (pick what applies)
### UI / Router
- FAST brain responds
- ARCHITECT escalation works
- template keys visible/traceable
- UI state doesn’t get stuck after servers load

### llama-server orchestration
- Uses `--log-file`
- Doesn’t run via `cmd.exe`
- All paths are fully expanded
- Startup readiness detection uses explicit log markers (e.g., “server is listening”, “slots idle”)

### RAG ingest
- Idempotent reruns (no duplicates)
- Uses batching (executemany/batch inserts)
- Shared hashing/source-id conventions used via a utility module
- Clear failure logging

### Pi agents / ZeroMQ
- Schema is documented and backward compatible
- Safe timeouts/retries
- Malformed command handling is safe

---

## 7) Conventions to Respect
- Python typing is a first-class requirement (Pylance clean); no `# type: ignore` without explanation.
- Use Unix shell syntax in bash commands (forward slashes, `/dev/null`); PowerShell when explicitly needed.
- Avoid refactors that change architecture without adding an ADR (see `docs/03-DECISIONS/`).
- All external content (RAG chunks, web snippets) must pass through `input_guard.sanitize_chunk()` before context injection.
- New optional subsystems must degrade gracefully — never let a missing service abort a turn.

---

## 8) Deep Links
- Architecture: `docs/01-ARCHITECTURE.md`
- Patterns: `docs/02-ARCH-PATTERNS.md`
- ADRs: `docs/03-DECISIONS/`
- Runbooks: `docs/10-RUNBOOKS/`
- Prompts/Templates: `docs/20-PROMPTS.md`
- Quality: `docs/30-QUALITY.md`
- Contributing: `docs/40-CONTRIBUTING.md`
