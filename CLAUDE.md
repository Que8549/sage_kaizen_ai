# CLAUDE.md
This file provides guidance to Claude Code (code.claude.com) when working in this repository. Claude should treat this as **repo-level, always-on rules**. :contentReference[oaicite:4]{index=4}

## Project: Sage Kaizen (Dual-Mode Local Cognitive Engine)
Sage Kaizen is a modular, production-minded local AI system with:
- **Dual-brain llama.cpp stack**: “FAST” (Q5 default) + “ARCHITECT” (Q6 on demand)
- **Streamlit chat UI** (dual-brain router + model load status)
- **Voice loop**: STT → LLM → tools → TTS (Coqui XTTS / OpenVoice targets)
- **Pi Agent transport**: ZeroMQ orchestration controlling Raspberry Pi 4/5 devices (LED universe, sensors, etc.)
- **RAG v1**: local folder ingest + RSS/web ingest into a vector store
- **Self-documenting codebase generator**: repo scan → README + Mermaid diagrams
- Must support **creative writing** (stories/poems) and **tutoring grades 1–12** (tone + safety + pedagogy)

## Hardware + Environment (authoritative)
User rig:
- OS: Windows 11 Professional
- CPU: AMD Ryzen 9 9950X3D
- RAM: 192 GB DDR5
- GPU0: RTX 5090 (32 GB VRAM)
- GPU1: RTX 5080 (16 GB VRAM)
- Storage: 40 TB mixed SSD/HDD
Dev environment:
- VS Code
- Python 3.14.3
- CUDA 13.1
- PyTorch cu130
- Custom llama.cpp + custom llama-cpp-python linked against that build

## Repo Architecture Map (keep in sync as code evolves)
Common modules (names based on project conventions seen in this repo’s discussions):
- `ui_streamlit_server.py` (or `ui_streamlit.py`): Streamlit chat UI + status + template display
- `router.py`: routes requests to FAST vs ARCHITECT brain; applies prompt templates automatically
- `server_manager.py`: launches/manages llama-server processes
- `openai_client.py`: OpenAI-compatible client wrapper for llama-server endpoints
- `prompt_library.py` / `sage_kaizen_prompt_lib.py`: system prompts + templates (“sage_fast_core”, “sage_architect_core”, etc.)
- `ingest/` or `rag/`: `rss_ingest.py`, `web_ingest.py`, `folder_ingest.py`, shared `ingest_utils.py`, `ingest_runtime.py`
- `pi_agents/` (or similar): ZeroMQ-based device orchestrator and Pi-side workers
- `docs_tools/` (or similar): repo scan → docs + Mermaid diagrams

If repo differs, update this section first; it’s the agent’s “map”.

## Non-Negotiable Invariants (do not regress)
These rules are treated as **hard constraints**:
1. `.bat` files are **configuration only**:
   - Only authoritative keys are like `EXE=...` and `MODEL=...`
   - Nothing else in `.bat` is considered authoritative
2. Never launch llama-server via `cmd.exe` (no `cmd /c ...`).
   - Always execute the `llama-server.exe` **directly from Python**.
3. Always use `--log-file` for llama-server logs.
   - Never rely on `stdout/stderr` redirection (`>`, `>>`) for long-running servers.
4. All paths must be **fully expanded** in Python before use.
   - No `%ROOT%`, no delayed expansion assumptions.

## Work Style: RPI Loop (Research → Plan → Implement → Validate)
Default workflow for any non-trivial change:
1. **Research**: locate files + read current behavior before edits
2. **Plan**: write a short, testable plan (phases, success criteria)
3. **Implement**: small diffs; keep changes local and incremental
4. **Validate**: run the right checks; confirm behavior via logs/tests

This mirrors the “context engineering” approach used by HumanLayer-style workflows. :contentReference[oaicite:5]{index=5}

## Coding Standards (Python-first)
- Python 3.14+ compatible typing; keep Pylance happy.
- Prefer small modules, explicit types, and clear boundaries.
- Avoid hidden global side effects at import time (especially in Streamlit).
- Use structured logging; never swallow exceptions silently.
- Preserve existing conventions for naming, prompts, template keys, and file layout.

### Windows / cmd.exe rules
- Provide commands that work in **cmd.exe** (use `^` for line continuation).
- Don’t assume PowerShell unless explicitly requested.
- Be careful with quoting and backslashes in paths.

## llama.cpp / llama-server Expectations
- Treat llama-server flags as build-dependent; if unsure, consult the local `llama-server --help` output (this repo tracks a help dump).
- Prioritize **accuracy over speed** by default, but improve latency when safe.
- Multi-GPU goals: ensure GPU0 (5090) is meaningfully utilized when configured to be.

## Safety / Secrets
- Never commit secrets, tokens, private keys, or personal data.
- If a change requires credentials, add `.env.example` guidance instead.
- Prefer least-privilege settings and explicit allowlists.

## “Definition of Done” for changes
A change is done only when:
- It matches the architecture + invariants above
- It’s testable (commands or checks are documented)
- It doesn’t introduce new Pylance/type errors
- It includes or updates logging needed for debugging
- If it changes behavior: update docs or Mermaid diagrams accordingly
