# CLAUDE.md — Sage Kaizen Agent Index (WHO / WHAT / WHY / HOW)
This repository is **Sage Kaizen**: a modular, production-ready local AI system with a dual-brain inference stack, voice + device control, RAG, and self-documenting tooling.

This file is written for **Claude Code** and Claude-in-VS-Code usage as the always-on “project brain.”  
It uses **progressive disclosure**:
1) Quick orientation (who/what/why/how)  
2) Non-negotiable invariants  
3) How to work (workflow + definition of done)  
4) Deep links into repo documentation (patterns, ADRs, runbooks, etc.)

---

## 1) WHO (Stakeholders + Operating Context)
### Primary user / operator
- **Alquin Cook** (project owner), building Sage Kaizen on a high-end Windows rig.

### Primary developer environment
- VS Code on Windows 11 Pro
- Python 3.14.3
- CUDA 13.1
- Custom `llama.cpp` + custom `llama-cpp-python` linked to the custom build

### Target runtime environments
- Windows host: runs llama-server brains, Streamlit UI, RAG ingestion, orchestration services
- Raspberry Pi 4/5 fleet: runs ZeroMQ agents and physical-world modules (LED, sensors, audio, etc.)

---

## 2) WHAT (System Overview)
Sage Kaizen is a **local cognitive engine** made of replaceable modules:

### Core modules (v1)
- **Dual brains** (two llama-server instances):
  - FAST brain (default): Qwen\Qwen2.5-14B-Instruct-GGUF Q6_K (or equivalent)
  - ARCHITECT brain (on demand): bartowski Qwen2.5-32B-Instruct Q6_K_L (or equivalent)
- **Router**: selects brain, applies templates, escalates to ARCHITECT when needed
- **Streamlit UI**: chat interface, status, templates visible, debugging-friendly
- **Pi Agent Transport**: ZeroMQ messaging to Raspberry Pi agents (device orchestrator)
- **RAG v1**: ingest (folder + RSS + web) into a vector store; query-time retrieval
- **Docs Generator v1**: repo scan → README + Mermaid diagrams

### User-facing behaviors
- Creative writing (stories, poems)
- Tutoring grades 1–12 (tone + safety + pedagogy)
- Voice-driven tools (STT → LLM → Tool → TTS)
- Physical-world control (“set LED mode cosmic”)

---

## 3) WHY (Goals + Non-Goals)
### Goals
- **Local-first**: runs without cloud dependency by default
- **Modular**: components can be swapped/upgraded (models, STT/TTS, RAG backend)
- **Production-minded**: observable (logs), testable, reproducible
- **Accurate by default**: prioritize correctness over raw speed (unless performance tuning is the task)

### Non-goals (unless explicitly requested)
- Large rewrites that break conventions
- “Magic” behavior without logs/tests
- Coupling .bat scripts to runtime logic beyond config keys

---

## 4) HOW (How We Build Here)
### Default workflow: RPI Loop (Research → Plan → Implement → Validate)
For any non-trivial work:
1. **Research**: locate current behavior + logs + existing patterns
2. **Plan**: short plan with files touched + success criteria
3. **Implement**: minimal diffs, typed, well-logged
4. **Validate**: run checks, confirm via logs/tests, document results

### Definition of Done
A change is “done” when:
- It respects the **Non-Negotiable Invariants**
- It is testable (documented commands / checks)
- It doesn’t introduce new typing/Pylance errors
- It improves or preserves observability (logs)
- Docs are updated if architecture/behavior changes

---

## 5) NON-NEGOTIABLE INVARIANTS (Never regress)
These are **hard constraints**:

1. `.bat` files are **configuration only**
   - Only authoritative keys: `EXE=...` and `MODEL=...` (and other explicit config keys we list in docs)
   - Nothing else in `.bat` is authoritative

2. **Never** launch llama-server via `cmd.exe`
   - No `cmd /c ...`
   - Python must execute the EXE directly

3. **Always** use `--log-file` for llama-server
   - Never rely on `stdout/stderr` redirection (`>`, `>>`) for long-running servers

4. Paths must be **fully expanded** before Python uses them
   - No `%ROOT%`, no delayed expansion assumptions

---

## 6) CURRENT HARDWARE (Authoritative)
User rig:
- OS: Windows 11 Professional
- CPU: AMD Ryzen 9 9950X3D
- RAM: 192 GB DDR5
- GPU0: RTX 5090 (32 GB VRAM)
- GPU1: RTX 5080 (16 GB VRAM)
- Storage: 40 TB mixed SSD/HDD

---

## 7) REPO “INDEX” (Progressive Disclosure Links)
This section is the navigation hub. When uncertain, start with **01-ARCHITECTURE**.

### Architecture + Patterns
- `docs/01-ARCHITECTURE.md` — system overview, data/control flow, module boundaries
- `docs/02-ARCH-PATTERNS.md` — patterns used (dual brain, tool router, agent transport, RAG)
- `docs/03-DECISIONS/` — ADRs (architecture decision records)

### Runbooks + Operations
- `docs/10-RUNBOOKS/01-LLAMA-SERVERS.md` — starting/stopping, logs, flags, ports
- `docs/10-RUNBOOKS/02-STREAMLIT-UI.md` — UI troubleshooting + state model
- `docs/10-RUNBOOKS/03-RAG-INGEST.md` — ingest idempotency, hashing, batching
- `docs/10-RUNBOOKS/04-PI-AGENTS.md` — ZeroMQ schema, retries, safety

### Prompting + Templates
- `docs/20-PROMPTS.md` — prompt library overview, template keys, escalation rules

### Testing + Quality
- `docs/30-QUALITY.md` — typing, linting, smoke tests, performance checks

### Contribution Guides
- `docs/40-CONTRIBUTING.md` — PR checklist, commit hygiene, how to add modules safely

---

## 8) If You’re a Coding Agent: Start Here
1) Read `docs/01-ARCHITECTURE.md`  
2) Read `docs/10-RUNBOOKS/01-LLAMA-SERVERS.md`  
3) Read `AGENTS.md`  
4) Only then propose changes

---

## 9) Notes for Claude (behavioral guidance)
- Prefer small, incremental changes that preserve existing style.
- When adding new features, prefer adding a module rather than tangling existing modules.
- If a fact is uncertain (flags, versions, APIs), check local `--help` output or project docs.
- For fine tuning AI models using llama-server refer to local `--help` or https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md.
