# AGENTS.md
Think of this file as a “README for coding agents”: setup, commands, conventions, and how to work effectively in this repo. :contentReference[oaicite:6]{index=6}

## Primary Goal
Help agents ship correct, maintainable improvements to **Sage Kaizen**:
- Dual-brain local inference via llama.cpp servers
- Streamlit UI + router + prompt templates
- RAG ingest pipelines
- Pi orchestration via ZeroMQ
- Voice loop (STT/TTS)
- Self-documenting repo tooling

## Operating Principles
1. **Follow repo invariants** (see `CLAUDE.md`).
2. Use the **RPI loop**: Research → Plan → Implement → Validate.
3. Prefer **small, reversible diffs** and “prove it works” with logs/tests.
4. Accuracy > speed unless performance work is the explicit task.

## Quick Start (Windows / VS Code)
> Update these if your repo already has a canonical script/Makefile.

### Python env
- Create venv: `python -m venv .venv`
- Activate (cmd.exe): `.venv\Scripts\activate`
- Install deps (example): `python -m pip install -r requirements.txt`

### Run Streamlit UI (example)
- `python -m streamlit run ui_streamlit_server.py`

### Start local brains (example)
- Prefer running via the repo’s **Python process manager** (e.g., `server_manager.py`) rather than manually.
- `.bat` files are config-only; the Python manager reads them and launches the EXE directly (see `CLAUDE.md`).

## Validation Checklist (pick what applies)
### UI / Router changes
- Launch Streamlit and confirm:
  - FAST brain works (Q5)
  - ARCHITECT escalation works (Q6)
  - Model “loaded” statuses update correctly
  - Templates apply automatically but can be overridden

### llama-server orchestration changes
- Confirm:
  - Logs written via `--log-file`
  - Paths are fully expanded
  - GPU selection flags match intent
  - Startup waits for “server is listening …” and/or “slots idle” markers before enabling UI interactions

### RAG ingest changes
- Confirm:
  - Re-runs are idempotent (no duplicate rows)
  - Uses batching (`executemany` / batch inserts)
  - Source ID + hashing conventions are consistent across ingest scripts
  - Failure modes are logged clearly (with actionable error messages)

### Pi / ZeroMQ changes
- Confirm:
  - Command schema stays backward compatible
  - Timeouts/retries are explicit
  - Worker nodes handle malformed commands safely

## How to Work in This Codebase (recommended agent workflow)
This is modeled after “specialized agents” patterns used by HumanLayer-style Claude Code workflows. :contentReference[oaicite:7]{index=7}

### Phase 1 — Research
- Find where relevant code lives (router, UI, server manager, ingest scripts).
- Read the *current* implementation and logs before proposing changes.
- Identify existing patterns and match them (don’t invent a new style unless asked).

### Phase 2 — Plan
Write a short plan with:
- Files to touch
- Intended behavior change
- Validation steps (exact commands/log signals)

### Phase 3 — Implement
- Make minimal diffs
- Keep functions small and typed
- Add/adjust logs where it improves debuggability

### Phase 4 — Validate
- Run the specific checks for the area you touched
- Report results (what you ran, what happened, where logs are)

## Repo-Specific Conventions to Respect
- Python typing + Pylance correctness is a first-class requirement.
- Windows cmd.exe command compatibility (`^` continuations).
- Don’t “optimize away” safety checks, logging, or the Q5/Q6 routing intent.
- Don’t change the `.bat` contract; treat `.bat` as config-only.

## When You’re Unsure
If the task depends on repo-specific truth (exact file paths, script names, DB schema, port numbers):
- Prefer to **search the repo** and cite exact file references.
- If you can’t find it, propose a conservative change that doesn’t assume structure.
- Avoid speculative refactors; do the smallest safe step.
