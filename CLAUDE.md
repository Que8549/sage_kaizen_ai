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
### Primary user / operator / administrator / owner
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
  - FAST brain (default): `Qwen2.5-Omni-7B-Q8_0` (port 8011, RTX 5080/CUDA1) — multimodal: text + image + audio input via mmproj encoder
  - ARCHITECT brain (on demand): `Qwen3.5-27B-Q6_K` (port 8012, RTX 5090/CUDA0) — 64K context, reasoning mode (`<think>` tokens), hybrid DeltaNet+attention
- **Router**: selects brain, applies templates, escalates to ARCHITECT when needed
- **Streamlit UI**: chat interface, status, templates visible, debugging-friendly
- **Pi Agent Transport**: ZeroMQ messaging to Raspberry Pi agents (device orchestrator)
- **RAG v1**: ingest (folder + RSS + web) into PostgreSQL + pgvector; query-time retrieval
  - `rag_v1/wiki/` — Wikipedia multimodal RAG (jina-clip-v2 embeddings, text + image)
  - `rag_v1/media/` — Cross-modal ingest: images (jina-clip-v2, 1024-dim) + audio (CLAP, 512-dim)
  - `rag_v1/embed/` — BGE-M3 embed client (wraps port 8020)
  - `rag_v1/retrieve/` — retriever + citation formatting
- **Docs Generator v1**: repo scan → README + Mermaid diagrams (planned)
- Review `config/brains/brains.yaml` for latest AI models and all server settings

### Service / Port Inventory
| Service | Model | Port | GPU | Purpose |
|---------|-------|------|-----|---------|
| FAST brain | Qwen2.5-Omni-7B-Q8_0 | 8011 | CUDA1 (5080) | Multimodal chat |
| ARCHITECT brain | Qwen3.5-27B-Q6_K | 8012 | CUDA0 (5090) | Deep reasoning |
| BGE-M3 embed | bge-m3-FP16 | 8020 | CUDA0 (5090) | RAG text embeddings (1024-dim) |
| Wiki embed (jina-clip-v2) | jina-clip-v2 | 8031 | CUDA0 (5090) | Wikipedia multimodal embeddings (1024-dim) |
| CLAP embed | clap-htsat-unfused | 8040 | CUDA1 (5080) | Audio embeddings (512-dim) |
| SearXNG | (metasearch) | 8080 | Docker Desktop | Live web search JSON API |

### Live Web Search (`search/`)
- `search/models.py` — `WebResult` + `SearchEvidence` normalized citation schema
- `search/searxng_client.py` — httpx JSON client for private SearXNG instance (http://localhost:8080)
- `search/search_orchestrator.py` — dedup, score filter, time_range, per-brain result ceiling; lazy singleton `get_orchestrator()`
- `search/summarizer.py` — lightweight FAST-brain summarization pass before context injection; falls back to raw snippets if brain unavailable
- `search/citations.py` — `format_search_sources_markdown()` for UI display (matches doc-RAG + wiki-RAG citation style)
- Router sets `needs_search=True` + `search_categories` on `RouteDecision` when live data is needed
- `context_injector.apply_rag_and_wiki_parallel()` runs a 3rd parallel worker; injects `<search_context>` block; returns 4-tuple `(messages, rag_sources, wiki_images, search_evidence)`
- SearXNG Docker instance: `F:\Projects\searxng\` — configured with JSON format enabled, limiter disabled, CORS open

### Supporting modules (root-level)
- `mermaid_streamlit.py` — Mermaid diagram detection and rendering in the chat UI
- `sk_logging.py` — centralized rotating log configuration
- `pg_settings.py` — Pydantic BaseSettings for PostgreSQL DSN (feedback dataset DB)
- `voice_bridge.py` — ZMQ bridge binding ports 5790/5791/5792 for the voice app

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

1. **`config/brains/brains.yaml` is the single authoritative config source**
   - All server settings (exe, model, log, flags, ports) live in `brains.yaml`
   - No `.bat` files — they have been deleted; do not recreate them
   - `server_manager.py` reads YAML directly and spawns the EXE via `subprocess.Popen`

2. **Never** launch llama-server via `cmd.exe`
   - No `cmd /c ...`
   - Python must execute the EXE directly

3. **Always** use `--log-file` for llama-server
   - Never rely on `stdout/stderr` redirection (`>`, `>>`) for long-running servers

4. Paths must be **fully expanded** before Python uses them
   - No `%ROOT%`, no environment variable expansion assumptions

---

## 6) CURRENT HARDWARE (Authoritative)
User rig also known as "my rig":
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
- `docs/01-ARCHITECTURE.md` — system overview, data/control flow, module boundaries ✓
- `docs/02-ARCH-PATTERNS.md` — patterns used (dual brain, tool router, agent transport, RAG) ✓
- `docs/03-DECISIONS/` — ADRs (architecture decision records) ✓

### Runbooks + Operations (planned — not yet created)
- `docs/10-RUNBOOKS/01-LLAMA-SERVERS.md` — starting/stopping, logs, flags, ports
- `docs/10-RUNBOOKS/02-STREAMLIT-UI.md` — UI troubleshooting + state model
- `docs/10-RUNBOOKS/03-RAG-INGEST.md` — ingest idempotency, hashing, batching
- `docs/10-RUNBOOKS/04-PI-AGENTS.md` — ZeroMQ schema, retries, safety

### Prompting + Templates (planned — not yet created)
- `docs/20-PROMPTS.md` — prompt library overview, template keys, escalation rules

### Testing + Quality (planned — not yet created)
- `docs/30-QUALITY.md` — typing, linting, smoke tests, performance checks

### Contribution Guides (planned — not yet created)
- `docs/40-CONTRIBUTING.md` — PR checklist, commit hygiene, how to add modules safely

---

## 8) If You’re a Coding Agent: Start Here
1) Read `docs/01-ARCHITECTURE.md`
2) Read `docs/10-RUNBOOKS/01-LLAMA-SERVERS.md`
3) Read `AGENTS.md`
4) **Review recent git history** — run `git log --oneline -30` and inspect relevant diffs before proposing changes
5) Only then propose changes

---

## 9) Notes for Claude (behavioral guidance)
- Prefer small, incremental changes that preserve existing style.
- When adding new features, prefer adding a module rather than tangling existing modules.
- If a fact is uncertain (flags, versions, APIs), check local `--help` output or project docs.
- For fine tuning AI models using llama-server refer to local `--help` or https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md.

### Review Git History Before Implementing
Before adding or reinstating any package, library, or approach, run:
```
git log --oneline -30
git log --all --oneline --grep="<keyword>"
git show <commit>
```
Past commits document what was tried and abandoned. Key known failures in this repo:
- **`flash_attn`** — present in `requirements.txt` as a local path reference but intentionally non-functional at runtime for Python-level code. SM_120 (Blackwell, RTX 5090/5080) is unsupported by flash-attn 2.x/3.x/4.x on Windows; `flash_attn.ops.triton.rotary` requires the OpenAI Triton compiler (Linux-only). The llama-server `--flash-attn` flag is separate and works correctly (handled by the C++ runtime, not Python). For Python inference code use PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`) with cuDNN SDP backend (`torch.backends.cuda.enable_cudnn_sdp(True)`).
- **`cmd.exe` for llama-server** — never use; `server_manager.py` spawns the EXE directly via `subprocess.Popen`.
- **`stdout/stderr` redirection for llama-server** — never use; always `--log-file`.
- **`.bat` launch scripts** — deleted (`start_q5_server.bat`, `start_q6_server.bat`, `start_embedding_point.bat`). All config is in `config/brains/brains.yaml`. Do not recreate `.bat` files for server launch.

If a commit message says "reverted", "removed", "uninstalled", or describes a failure, read it before reimplementing the same approach.

---

## 10) Related and Associated Projects
 - Integrate with Sage Kaizen Voice (voice app) located at F:\Projects\sage_kaizen_ai_voice\
 - Sage Kaizen local-first AI assistant (main app) located at F:\Projects\sage_kaizen_ai\
 - SearXNG - local search engine running at http://localhost:8080/ located at F:\Projects\searxng

---

## 11) FAST Brain Model — Upgrade Research Log (2026-04-07)

### Current State (as of 2026-04-07)
- **Model**: `Qwen2.5-Omni-7B-Q8_0` — the only viable audio-capable model for the RTX 5080 in llama.cpp
- **llama.cpp build**: b8639 (early April 2025) — outdated; rebuild recommended
- **Known limitation**: Mid-response Chinese language code-switching during long-form generation (confirmed Qwen2.5-Omni-7B training data bias; see QwenLM/Qwen2.5 issue #347)
- **Workaround applied**: `router.py` now routes creative writing (`CREATIVE_HINTS`) to ARCHITECT (score +3); `prompt_library.py` `sage_fast_core` includes English-only instruction

### Why No Upgrade Is Possible Yet
Audio file upload support (`kind="audio"` in `chat_service.py`) depends on llama.cpp's mmproj audio encoder. In the llama.cpp ecosystem (as of 2026-04-07), **only Qwen2.5-Omni-7B** combines all three capabilities: audio input + image/video input + general text reasoning. Every other capable model fails on at least one requirement:

| Candidate | Blocker |
|---|---|
| Qwen3-8B / Qwen3-VL-8B | No audio encoder — audio uploads break |
| Gemma 3 12B | No audio encoder in llama.cpp |
| Qwen3-Omni-30B-A3B | Fits on 5090 but 5090 is fully occupied by ARCHITECT + BGE-M3 (~29 GB used of 32 GB) |
| Qwen3.5-Omni | llama.cpp audio support confirmed incomplete as of 2026-04-07 |
| Voxtral-Mini-3B | Known crash on audio encoding — llama.cpp issue #21080 |
| Ultravox v0.5/v0.6 (8B) | Audio-to-text only; no vision, no general reasoning |

### VRAM Budget — Current vs Proposed Q6_K Downquant
If English stability is needed on FAST without a model change, dropping to Q6_K saves ~1.85 GB with negligible quality loss (~0.1–0.2 PPL):

```
# Current (Q8_0)                    # Proposed (Q6_K)
Model weights:   ~8.10 GB           Model weights:   ~6.25 GB  (-1.85 GB)
mmproj F16:      ~2.64 GB           mmproj F16:      ~2.64 GB  (unchanged)
KV cache q8_0:   ~0.45 GB           KV cache q8_0:   ~0.45 GB  (unchanged)
Compute buffer:  ~0.50 GB           Compute buffer:  ~0.50 GB  (unchanged)
Total:          ~11.70 GB           Total:           ~9.84 GB
Headroom:        ~4.3 GB            Headroom:        ~6.2 GB  (+1.9 GB)
```
Q6_K GGUF: https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF  
`brains.yaml` change: update `model:` path and `alias:` only — all flags, mmproj, and ports unchanged.

### Functionality Checklist (What Must Be Preserved on Any FAST Upgrade)
Before proposing or applying a FAST brain model change, verify all of the following are maintained:

| Capability | How It Works | File |
|---|---|---|
| Audio file uploads (`.wav`, `.mp3`) | mmproj audio encoder; `kind="audio"` routes to FAST | `chat_service.py:194` |
| Image input | mmproj vision encoder; `kind="image"/"video_frame"` → ARCHITECT or FAST | `chat_service.py:183` |
| Video input | Client-side frame extraction → image attachments → ARCHITECT | `chat_service.py:183` |
| Flash attention | `flash_attn: true` in brains.yaml; C++ runtime only (not Python) | `brains.yaml:65` |
| KV prefix cache | `cache_ram: 512`, `slot_prompt_similarity: 0.10` | `brains.yaml:74,79` |
| 16K context | `ctx_size: 16384` — 1 image ≈ 1280 tokens, 15104 for conversation | `brains.yaml:53` |
| Port 8011, CUDA1 | Hard-coded in routing and inference session | `brains.yaml:38,43` |
| TTS voice pipeline | Audio output is text-only; Kokoro handles TTS separately | `voice_bridge.py` |

### Watch List — When to Revisit the FAST Brain Upgrade
Monitor these milestones; when any trigger is met, re-evaluate:

1. **Qwen3.5-Omni llama.cpp audio PR merges** — check https://github.com/ggml-org/llama.cpp/pulls for "omni" or "audio" PRs. This is the primary upgrade path when it lands. Model will need 5090 or GPU upgrade (30B+ size).

2. **Voxtral-Mini-3B crash fixed** — track llama.cpp issue #21080. If fixed, Mistral's 3B audio model could run as a lightweight audio-only companion on the 5080 alongside a stronger text model.

3. **Qwen2.5-Omni-14B or larger Omni release** — Alibaba has only released 3B and 7B Omni variants. A 14B would be a direct drop-in upgrade if it fits (~14 GB weights Q6_K = marginal, Q4_K_M = comfortable on 5080).

4. **Gemma 4 audio support in llama.cpp** — Gemma 4 natively supports audio but llama.cpp audio parsing is not yet implemented. Track https://github.com/ggml-org/llama.cpp/issues.

5. **llama.cpp rebuild** — Current build b8639 predates SM_120 Blackwell kernel optimizations. Rebuilding from latest release (https://github.com/ggml-org/llama.cpp/releases) improves token throughput on RTX 5080/5090 with no model changes needed.

### Qwen2.5-Omni-7B GGUF Sources
- Official: https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF (Q8_0, Q6_K, Q4_K_M, and others)
- Unsloth: https://huggingface.co/unsloth/Qwen2.5-Omni-7B-GGUF (extensive quant options including IQ variants)
