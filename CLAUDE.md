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
- CUDA 13.3
- Custom `llama.cpp` build (MSVC, CUDA SM_120 Blackwell, `ARCHS=1200`)


### Target runtime environments
- Windows host: runs llama-server brains, Streamlit UI, RAG ingestion, orchestration services
- Raspberry Pi 4/5 fleet: runs ZeroMQ agents and physical-world modules (LED, sensors, audio, etc.)

---

## 2) WHAT (System Overview)
Sage Kaizen is a **local cognitive engine** made of replaceable modules:

### Core modules (v1)
- **Dual brains** (two llama-server instances):
  - FAST brain (default): `Qwen2.5-Omni-7B-Q6_K` (port 8011, **CUDA0** — RTX 5090, display GPU) — multimodal: text + image + audio input via mmproj encoder
  - ARCHITECT brain (on demand): `Qwen3.6-27B-MTP-Q6_K` (port 8012, **CUDA1** — RTX 5090 OC) — **128K context**, reasoning mode (`<think>` tokens), MTP speculative decoding: draft-mtp, hybrid DeltaNet+attention. Measured 64.25 t/s effective, 1.41x over base decode (§15)
  - Summarizer (lightweight): `Qwen3-4B-Q8_0` (port 8013, **CUDA2** — RTX 5080 eGPU) — search evidence summarization before context injection. Moved off CPU 2026-08-24 (§16.1)
- **Router**: selects brain, applies templates, escalates to ARCHITECT when needed (`router.py`)
- **Streamlit UI**: chat interface, status, templates visible, debugging-friendly (`ui_streamlit_server.py`)
- **Chat Service**: full turn lifecycle — route → memory → prompt → parallel RAG → stream (`chat_service.py`)
- **Pi Agent Transport**: ZeroMQ messaging to Raspberry Pi agents (`voice_bridge.py` owns host ZMQ; `agents/` planned)
- **RAG v1**: ingest (folder + RSS + web + ZIM) into PostgreSQL + pgvector; parallel query-time retrieval
  - `rag_v1/wiki/` — Wikipedia multimodal RAG (jina-clip-v2 embeddings, text + image)
  - `rag_v1/media/` — Cross-modal ingest: images (jina-clip-v2, 1024-dim) + audio (CLAP, 512-dim)
  - `rag_v1/embed/` — `BaseHttpEmbedClient` (shared pooled transport for all four embed clients) + BGE-M3 client (port 8020)
  - `rag_v1/retrieve/` — retriever + citation formatting
  - `rag_v1/runtime/context_injector.py` — parallel 5-worker assembly (doc-RAG · wiki · search · music · news)
  - `rag_v1/runtime/router_integration.py` — `RagInjector` wires router decisions to context_injector
- **Memory Service**: persistent episode memory, user profiles, learned rules (`memory/service.py`); lazy singleton with graceful degradation
- **News Module**: news collection, clustering, enrichment, image pipeline (`news/`)
- **Docs Generator v1**: repo scan → README + Mermaid diagrams (planned)
- **Review Service**: LangGraph-based codebase review service triggered by chat phrases; generates ADRs, patches, and review reports using ARCHITECT brain
  - `review_service/graph.py` — sequential StateGraph: scope → subprocess_checks → web_researcher → architect_reviewer → flags_sanity → docs_drift → synthesizer → human_gate → output_writer
  - `review_service/runner.py` — ReviewRunner; background daemon thread with isolated asyncio event loop
  - `review_service/checkpointer.py` — AsyncPostgresSaver using pg_settings.py DSN; dedicated `langgraph` schema
  - `review_service/trigger.py` — `is_review_command()` heuristic; called by `ui_streamlit_server.py` (not router.py)
  - `review_service/output/` — review_writer, adr_writer, patch_writer (write to `reviews/`, `docs/03-DECISIONS/`, `patches/`)
- **Supporting root-level modules**:
  - `document_parser.py` — multi-format doc extraction (docx, xlsx, csv, code, txt, etc.)
  - `input_guard.py` — prompt-injection defense for all external content (RAG chunks, web snippets)
  - `env_utils.py` — per-call env var accessors (`env_bool`, `env_int`, `env_float`, `env_str`); re-read every turn
  - `lazy.py` — `@lazy_singleton`: the one thread-safe lazy singleton helper; every process-wide accessor uses it (see §14)
  - `evals/` — model evaluation harness: hard gates, `llama-bench` wrapper, deterministic quality scorers, golden-set miner (see §15)
  - `mermaid_streamlit.py` — Mermaid diagram detection and rendering
  - `sk_logging.py` — centralized rotating log configuration; also mirrors structured log records into PostgreSQL (`log` schema, best-effort, see §12)
  - `pg_settings.py` — Pydantic BaseSettings for PostgreSQL DSN
  - `voice_bridge.py` — ZMQ bridge binding ports 5790/5791/5792 for the voice app
- Review `config/brains/brains.yaml` for latest AI models and all server settings

### Service / Port Inventory
| Service | Model | Port | GPU | Purpose |
|---------|-------|------|-----|---------|
| FAST brain | Qwen2.5-Omni-7B-Q6_K | 8011 | **CUDA0 (5090, display)** | Multimodal chat. Sole service on the display GPU (§16.1) |
| ARCHITECT brain | Qwen3.6-27B-MTP-Q6_K | 8012 | **CUDA1 (5090 OC)** | Deep reasoning; 128K ctx; `<think>` tokens |
| Summarizer | Qwen3-4B-Q8_0 | 8013 | **CUDA2 (5080 eGPU)** | Search evidence summarization. Moved off CPU 2026-08-24 |
| BGE-M3 embed | bge-m3-FP16 | 8020 | CUDA1 (5090 OC) | RAG text embeddings (1024-dim). Moved off CUDA0 2026-08-06 — see §16 |
| Wiki embed A | jina-clip-v2 | 8031 | **CUDA2 (5080 eGPU)** | Wikipedia multimodal embeddings; also serves media image embeds |
| Wiki embed B | jina-clip-v2 | 8032 | CUDA2 (5080 eGPU) | Wikipedia ingest only. FAST no longer shares this GPU |
| CLAP embed | clap-htsat-unfused | 8040 | **CUDA2 (5080 eGPU)** | Audio embeddings (512-dim) |
| SearXNG | (metasearch) | 8080 | Docker Desktop | Live web search JSON API |
| Voice STT/TTS | Whisper distil-large-v3.5 + Kokoro-82M | ZMQ 5790/5791/5792 | CPU (ONNX) | Voice: transcript in, token stream out, barge-in |

### Live Web Search (`search/`)
- `search/models.py` — `WebResult` + `SearchEvidence` normalized citation schema
- `search/searxng_client.py` — httpx JSON client for private SearXNG instance (http://localhost:8080)
- `search/search_orchestrator.py` — dedup, score filter, time_range, per-brain result ceiling; lazy singleton `get_orchestrator()`
- `search/summarizer.py` — lightweight FAST-brain summarization pass before context injection; falls back to raw snippets if brain unavailable
- `search/citations.py` — `format_search_sources_markdown()` for UI display (matches doc-RAG + wiki-RAG citation style)
- Router sets `needs_search=True` + `search_categories` on `RouteDecision` when live data is needed
- `context_injector.apply_rag_and_wiki_parallel()` runs a 3rd parallel worker; injects `<search_context>` block; returns 4-tuple `(messages, rag_sources, wiki_images, search_evidence)`
- SearXNG Docker instance: `F:\Projects\searxng\` — configured with JSON format enabled, limiter disabled, CORS open

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

5. **Review service uses the existing PostgreSQL connection (`pg_settings.py`) for LangGraph checkpoint persistence**
   - Tables live in the `langgraph` schema (not `public`) — run `scripts/setup_langgraph_schema.sql` once as superuser before first review run
   - Do not introduce a separate database connection for the review service

6. **No PyTorch compute on `cuda:0`** (was worded "cuda:0 is display-only", which reality never matched)
   - cuda:0 drives three monitors. `torch.compile` autotune there is the documented Windows TDR trigger, so every PyTorch service (jina-clip-v2, CLAP) is barred from it.
   - llama-server is C++ and IS permitted: since 2026-08-24 the FAST brain runs on cuda:0 as its sole tenant, leaving ~20.8 GB free (§16.1).
   - Enforced by `WikiRetriever`'s `DisplayGpuRefused` guard, which checks the *effective* device, not just config
   - The only sanctioned cuda:0 compute is `wiki_ingest.py --gpu0-workers 1` in the ingest project
   - See §10 and sage_kaizen_ai_ingest CLAUDE.md §19

7. **Seven shared modules resolve to THIS repo for both projects** — changing them is a cross-project change (see §13)

8. **THE WRITER OWNS THE SCHEMA** (revised 2026-08-10 — this replaces the old
   "all schema lives in main" rule, which reality had already outgrown)
   - The app that **writes** a table owns its DDL, its migrations, and its indexes.
   - **Ingest writes all ingested data** — `wiki_*`, `news_*`/`daily_news`,
     `rag_chunks`, `image_embeddings`, `audio_embeddings`, `audio_clusters`,
     `media_files`, `lyrics*`. Ingest owns that schema.
   - **Main writes** `memory.*` and `public.ratings`. Main owns that schema.
   - **`log.*` is the one shared-write exception** — both projects insert into it,
     so main owns it (§12). Exceptions must be named here, never assumed.
   - **The main app must be able to READ every ingest-owned table at all times.**
     That is a hard contract, not a courtesy: see §19 for the expand/contract
     discipline that keeps it true, and the checklist that must be run before
     any PostgreSQL change.
   - `news/db/news_schema.sql` living in ingest is no longer an "exception" —
     it is the rule, correctly applied. Several files in **this** repo are now
     the misfiled ones; §19 lists them.

---

## 6) CURRENT HARDWARE (Authoritative)
User rig also known as "my rig":
- Motherboard: Gigabyte X870E AORUS XTREME AI TOP AMD AM5 eATX Motherboard
- CPU: AMD Ryzen 9 9950X3D
- RAM: 192 GB DDR5 Speed: 6400 MT/s
- GPU0: CUDA 0 - NVIDIA GeForce RTX 5090 (32GB VRAM) — primary display GPU (3 monitors); **FAST Brain only**, ~20.8 GB left free (§16.1)
- GPU1: CUDA 1 - Gigabyte GeForce RTX 5090 OC (32GB VRAM) — no display; **ARCHITECT Brain + BGE-M3 embed**
- GPU2: CUDA 2 - Gigabyte GeForce RTX 5080 (16GB VRAM) — no display; **summarizer + jina-clip-v2 + CLAP** (§16.1). 
  - Connected by MinisForum DEG2 OCuLink eGPU Dock via USB-C (https://www.minisforum.com/products/deg2). 
  - The MinisForum DEG2 OCuLink eGPU Dock has a 2TB SSD drive.
- CUDA: 13.3
- Storage: 40 TB mixed SSD/HDD
- OS: Windows 11 Professional
- Database: PostgreSQL with pgvector
- Power Supply 1600W https://seasonic.com/atx3-prime-tx/ 
- Python (this venv): 3.14.3

---

## 7) REPO “INDEX” (Progressive Disclosure Links)
This section is the navigation hub. When uncertain, start with **01-ARCHITECTURE**.

### Architecture + Patterns
- `docs/01-ARCHITECTURE.md` — system overview, data/control flow, module boundaries ✓
- `docs/02-ARCH-PATTERNS.md` — patterns used (dual brain, tool router, agent transport, RAG) ✓
- `docs/03-DECISIONS/` — ADRs (architecture decision records) ✓
- `docs/Architect_Reviewer.md` — Review Service: design, trigger phrases, workflow, output artifacts ✓
- `Benchmarking_Kaizen_Models.md` — **how to decide whether a new model is worth upgrading to** (method, decision rule, watch-list status) ✓

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
- **`flash_attn`** — present in `requirements.txt` as a local path reference but intentionally non-functional at runtime for Python-level code. SM_120 (Blackwell, RTX 5090) is unsupported by flash-attn 2.x/3.x/4.x on Windows; `flash_attn.ops.triton.rotary` requires the OpenAI Triton compiler (Linux-only). The llama-server `--flash-attn` flag is separate and works correctly (handled by the C++ runtime, not Python). For Python inference code use PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`) with cuDNN SDP backend (`torch.backends.cuda.enable_cudnn_sdp(True)`).
- **`cmd.exe` for llama-server** — never use; `server_manager.py` spawns the EXE directly via `subprocess.Popen`.
- **`stdout/stderr` redirection for llama-server** — never use; always `--log-file`.
- **`.bat` launch scripts** — deleted (`start_q5_server.bat`, `start_q6_server.bat`, `start_embedding_point.bat`). All config is in `config/brains/brains.yaml`. Do not recreate `.bat` files for server launch.

If a commit message says "reverted", "removed", "uninstalled", or describes a failure, read it before reimplementing the same approach.

---

## 10) Related and Associated Projects
 - Integrate with Sage Kaizen Voice (voice app) located at F:\Projects\sage_kaizen_ai_voice\
 - Sage Kaizen local-first AI assistant (main app) located at F:\Projects\sage_kaizen_ai\
 - Sage Kaizen ingest (ingest app) located at F:\Projects\sage_kaizen_ai_ingest
 - SearXNG - local search engine running at http://localhost:8080/ located at F:\Projects\searxng

### Wiki Ingest — GPU Layout and Thermal Management

**No PyTorch compute on cuda:0.** The RTX 5090 at CUDA0 drives three monitors, and
Windows TDR (2 s) resets the display driver if a CUDA kernel runs too long —
`torch.compile` autotune can freeze a GPU for 10–60 s, which means a black screen
and a reboot. llama-server is C++, does no such autotune, and **is** permitted
there: since 2026-08-24 the FAST brain is cuda:0's sole tenant (§16.1). The bar is
on PyTorch services — jina-clip-v2 and CLAP.

Wiki embed services A (port 8031) and B (port 8032) both run on **CUDA2
(RTX 5080 eGPU)** as of 2026-08-24; they were on CUDA1 before that, and service A
was documented as CUDA0 until 2026-08-05. That last error was never true of the
ingest path (`wiki_ingest.py` passes an explicit `device=` per service) but it
*was* the value in `brains.yaml`, which is what any other consumer falls back to
— including this app's chat-time `WikiRetriever`. Corrected in `brains.yaml`
(commit a1ac499), in `WikiEmbedServiceConfig.device`, and enforced by a hard
guard:

- `rag_v1/wiki/wiki_retriever.py` raises `DisplayGpuRefused` rather than
  starting `mm_embed_service` on cuda:0, checking the **effective** device
  (`WIKI_EMBED_DEVICE` env var, then `brains.yaml`) so it holds even if a
  config regresses.
- It pins the validated device into the child's environment, closing the hole
  where an inherited `WIKI_EMBED_DEVICE=cuda:0` silently beat `brains.yaml`.
- It reads `/health`'s `device` field rather than trusting a bare ping, so a
  service this process did not start is also checked.

The only sanctioned PyTorch use of cuda:0 is `wiki_ingest.py`'s optional C worker
under an explicit `--gpu0-workers 1` — a deliberate, per-run override, not a
default. Note that it now collides with the FAST brain, which lives on cuda:0;
stop FAST before passing that flag. See sage_kaizen_ai_ingest's CLAUDE.md §19 for
the audit this mirrors.

Before running a long wiki ingest session, set GPU power limits once as Administrator:
```powershell
F:\Projects\sage_kaizen_ai_ingest\scripts\set_gpu_limits.ps1
```
Limits: GPU0 RTX 5090 → 420 W (stock 575 W; display GPU), GPU1 RTX 5090 OC → 500 W (stock 575 W+; sustained ingest).  
Then run ingest with `--no-power-limits` (limits persist until reboot).  
If power limits cannot be applied, `wiki_ingest.py` auto-falls back to 2 workers + 75 ms throttle.  
Default throttle is 25 ms per worker even with power limits applied (protective baseline).  
**No longer required (2026-08-24):** stopping the FAST brain before a wiki ingest
was needed only while wiki-embed-B and FAST shared CUDA1. After the remap (§16.1)
both wiki embed services are alone on CUDA2 and FAST is alone on CUDA0, so chat
and ingest no longer contend for the same GPU. The power-limit script still
applies — it caps the two 5090s, not the 5080.

---

## 11) FAST Brain Model — Upgrade Research Log (2026-04-07, updated 2026-06-11)

### Current State (as of 2026-06-11)
- **Model**: `Qwen2.5-Omni-7B-Q6_K` — **Q6_K downquant applied**; saves ~1.85 GB VRAM vs Q8_0 with negligible quality loss (~0.1–0.2 PPL)
- **GPU**: Gigabyte GeForce RTX 5090 OC (CUDA1, 32 GB VRAM) — upgraded from RTX 5080 (16 GB) on 2026-05-24
- **llama.cpp build**: **b10298 (`15586e2d7`), rebuilt 2026-08-06** — custom MSVC build, `CMAKE_CUDA_ARCHITECTURES=120` (SM_120 Blackwell), `GGML_CUDA_FA=ON`, `GGML_CUDA_GRAPHS=ON`, Release. Costs 15–22% throughput vs b9598 (§15); kept for the capabilities it unlocked.
  - Note: `BLACKWELL_NATIVE_FP4=1` was previously recorded here as a build define. No such option exists in llama.cpp's CMake — the current cache has no FP4 knob. Treat the old note as stale, not as a setting to restore.
- **Context**: 32K (upgraded from 16K — new 32 GB VRAM budget provides ~21.7 GB headroom)
- **Known limitation**: Mid-response Chinese language code-switching during long-form generation (confirmed Qwen2.5-Omni-7B training data bias; see QwenLM/Qwen2.5 issue #347)
- **Workaround applied**: `router.py` routes creative writing (`CREATIVE_HINTS`) to ARCHITECT (score +3); `prompt_library.py` `sage_fast_core` includes English-only instruction

### Q6_K VRAM Budget (RTX 5090 OC, 32 GB)
```
Model weights:          ~6.25 GB
mmproj F16:             ~2.64 GB
KV cache q8_0 @32K:     ~0.90 GB  (doubled from 16K; 28 layers, 4 KV heads, head_dim=128)
Compute buffer:         ~0.50 GB
Total:                 ~10.29 GB  (headroom: ~21.7 GB on RTX 5090 OC's 32 GB)
```

### Why No Further Model Upgrade Is Possible Yet
Audio file upload support (`kind="audio"` in `chat_service.py`) depends on llama.cpp's mmproj audio encoder. In the llama.cpp ecosystem (as of 2026-04-19), **only Qwen2.5-Omni-7B** combines all three capabilities: audio input + image/video input + general text reasoning. Every other capable model fails on at least one requirement:

| Candidate | Blocker |
|---|---|
| Qwen3-8B / Qwen3-VL-8B | No audio encoder — audio uploads break |
| Gemma 3 12B | No audio encoder in llama.cpp |
| Qwen3-Omni-30B-A3B | Fits on CUDA1 (5090 OC, 32 GB) but would displace FAST brain; no audio upgrade path yet |
| Qwen3.5-Omni | llama.cpp audio support confirmed incomplete as of 2026-04-19 |
| Voxtral-Mini-3B | Known crash on audio encoding — llama.cpp issue #21080 |
| Ultravox v0.5/v0.6 (8B) | Audio-to-text only; no vision, no general reasoning |

### Functionality Checklist (What Must Be Preserved on Any FAST Upgrade)
Before proposing or applying a FAST brain model change, verify all of the following are maintained:

| Capability | How It Works | File |
|---|---|---|
| Audio file uploads (`.wav`, `.mp3`) | mmproj audio encoder; `kind="audio"` routes to FAST | `chat_service.py:194` |
| Image input | mmproj vision encoder; `kind="image"/"video_frame"` → ARCHITECT or FAST | `chat_service.py:183` |
| Video input | Client-side frame extraction → image attachments → ARCHITECT | `chat_service.py:183` |
| Flash attention | `flash_attn: true` in brains.yaml; C++ runtime only (not Python) | `brains.yaml:78` |
| KV prefix cache | `cache_ram: 512` — **unverified for FAST**; ARCHITECT's was dead at 256 (§15) | `brains.yaml:90` |
| 32K context | `ctx_size: 32768` — 1 image ≈ 1280 tokens, ~30K for conversation | `brains.yaml:61` |
| Port 8011, CUDA0 | `brains.yaml` is authoritative; do not hard-code | `brains.yaml:39,49` |
| TTS voice pipeline | Audio output is text-only; Kokoro handles TTS separately | `voice_bridge.py` |

### Watch List — When to Revisit the FAST Brain Upgrade

> **THREE OF THESE HAVE FIRED (verified 2026-08-06, re-verified on b10298).**
> Checked against the local build's projector table, not assumed. See
> `Benchmarking_Kaizen_Models.md` §6.

1. ✅ **FIRED — Qwen3-Omni audio support is in this build.** Merged upstream in
   b8769 (April 2026); the local build (b10298, and b9598 before it) contains
   `PROJECTOR_TYPE_QWEN3A` (audio) and `PROJECTOR_TYPE_QWEN3VL` (vision), and
   `clip.h` exposes `has_vision`/`has_audio` on one context, so a combined omni
   mmproj is representable. Official GGUFs exist:
   `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF` (Q4_K_M ≈ 18.6 GB).
   **This is the primary upgrade path and it is now open.**

   **Re-confirmed 2026-08-07 from upstream documentation, not inference:**
   llama.cpp's own `docs/multimodal.md` lists Qwen3-Omni under *"Mixed
   modalities — audio input, vision input"*. The repo ships **one** mmproj
   (Q8_0 1.33 GB / bf16 2.21 GB — smaller than the incumbent's 2.64 GB F16),
   with Q4_K_M at 18.56 GB and Q8_0 at 32.48 GB.

   Caveats, both now sharper than when first written:
   - **VRAM.** CUDA1 also hosts BGE-M3 since §16 moved it there. Estimated
     total with all co-tenants is **~28.4 GB of 32.6** — feasible, but the KV
     figure assumes Qwen3-30B-A3B geometry and is unverified for Omni. Run
     `python scripts/run_model_gate.py --brain fast --plan-co-tenants` first.
   - **Q4_K_M is the only quant that fits, and MoE is more quantisation-
     sensitive than dense** — fewer parameters per expert. This is
     Q6_K-on-7B-dense vs Q4_K_M-on-3B-active-MoE; more parameters is *not*
     automatically better quality. The CJK scorer decides it.
   - Cheapest first step: download **only the 1.33 GB mmproj** and gate it for
     audio+vision before committing to the 18.6 GB model.

   Take the **Instruct** variant, not Thinking: FAST must not think, or the
   routing latency model collapses. Full review: `Benchmarking_Kaizen_Models.md`
   §12. **Deferred until the wiki_chunks migration finishes** — see §17.

2. ✅ **FIRED — Voxtral is supported.** `PROJECTOR_TYPE_VOXTRAL` is present in
   this build. The original blocker was llama.cpp issue #21080 (crash on audio
   encoding). Worth re-testing only as a lightweight audio-only companion; it
   has no vision and no general reasoning, so it cannot replace FAST alone.

3. ⬜ **Qwen2.5-Omni-14B or larger Omni release** — Alibaba has only released 3B
   and 7B Omni variants. Largely superseded by trigger 1.

4. ✅ **FIRED — Gemma 4 audio is supported.** `PROJECTOR_TYPE_GEMMA4A` and
   `PROJECTOR_TYPE_GEMMA4UA` are present. Gemma 4 has no MTP and a different
   `<think>` contract, so it is a weaker fit than Qwen3-Omni for this stack, but
   it is no longer blocked.

5. ✅ **FIRED, with a caveat — rebuilt to b10298 on 2026-08-06.** SM_120 kernels
   present. It brought **Eagle3 speculative decoding for Qwen3.5/3.6**
   (upstream #24593) — the only identified route past the measured 1.41× MTP
   ceiling on ARCHITECT — and **Qwen3-TTS** projectors, a TTS path inside the
   brain stack that did not exist when the voice pipeline was designed. Both are
   unevaluated. The caveat: this build also costs 15–22% raw throughput (§15).

**How to act on a trigger:** do not swap a model on the strength of a leaderboard.
Run the harness — `Benchmarking_Kaizen_Models.md` has the method and the decision
rule, and §15 below has the commands.

### Qwen2.5-Omni-7B GGUF Sources
- Official: https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF (Q8_0, Q6_K, Q4_K_M, and others)
- Unsloth: https://huggingface.co/unsloth/Qwen2.5-Omni-7B-GGUF (extensive quant options including IQ variants)

---

## 12) Structured Logging → PostgreSQL (`log` schema)

Added 2026-07-16. Every structured, `logging`-module-based log file across
`sage_kaizen_ai`, `sage_kaizen_ai_ingest`, and `sage_kaizen_ai_voice` is
mirrored (best-effort, non-blocking) into a matching table in the dedicated
`log` Postgres schema.

**DB-only, then a crash-safety file was re-added — both same day, 2026-07-16**:
after applying the schema and verifying end-to-end that all six tables
receive every record (a live write-then-read check per table, not just "the
code looks right"), the rotating `.log` files these six loggers used to also
write were retired — explicit decision to make Postgres the sole source of
truth and eliminate the redundant on-disk copy. This immediately created a
real gap: `PostgresLogHandler` batches records in memory for up to ~2s/200
records before they reach Postgres, and a hard crash (e.g. a BSOD) gives no
chance to flush that buffer — so a crash could lose the last few seconds of
logs entirely, which was exactly the kind of loss this whole feature was
built to prevent. Fixed the same day: a **small** `RotatingFileHandler`
(`FALLBACK_MAX_BYTES` / `FALLBACK_BACKUP_CNT` in `sk_logging.py` — 1 MB × 2
backups) was re-attached alongside `PostgresLogHandler` for all six sources.
Every log call now writes to this file synchronously (same mechanism file
logging always used here), completely independent of the DB batching, so a
crash can't lose the data. It is deliberately small and NOT a second
permanent archive — Postgres remains the intended long-term store; if a
crash ever does cause a gap in the DB, reconciling this file back into
Postgres is a manual step, not automatic (no write-ahead-log/replay system
was built — that was considered and explicitly declined in favor of this
simpler, lower-risk mechanism). `file_names` not in the six-table map are
unaffected either way — they always had (and keep) the standard-size
rotating file as their only copy.

**Setup (one-time, run in order):**
```powershell
# 1. As postgres superuser (pgAdmin or psql -U postgres)
psql -U postgres -d sage_kaizen -f scripts/setup_log_schema.sql
# 2. As sage
psql -U sage -d sage_kaizen -f log/db/log_schema.sql
```
Apply BOTH before relying on any project's DB logging — the handler degrades
silently to file-only when the schema/tables are missing, so getting this
sequencing right matters more than the safety net.

**Tables** (one per source `.log` file, `id bigint GENERATED ALWAYS AS
IDENTITY PRIMARY KEY` for fast sequential indexing — a deliberate deviation
from the house `uuid PRIMARY KEY DEFAULT gen_random_uuid()` convention used
elsewhere, chosen specifically for this append-only high-volume workload):
`log.sage_kaizen`, `log.sage_kaizen_ingest`, `log.sage_kaizen_voice`,
`log.news_agent` (shared by main + ingest, disambiguated by
`source_project`), `log.media_ingest`, `log.wiki_ingest`. Plus
`log.all_logs`, a `UNION ALL` convenience view across all six for run_id-scoped
cross-component queries:
```sql
SELECT * FROM log.all_logs WHERE run_id = '...' ORDER BY log_date;
```

**Out of scope (deliberately, not yet implemented)**: raw subprocess-captured
stdout/stderr logs (embed-service crash tracebacks, uvicorn banners) and
llama-server's own native log format — neither is reliably parseable as
structured rows. Both remain file-only.

**Mechanism** — `sk_logging.py`'s `PostgresLogHandler` (one local copy per
project, same "N local copies" convention as the rest of this file): a
bounded, non-blocking queue + dedicated consumer thread batches records
(flush every ~2s or ~200 records) into the mapped `log.<table>` via a
dedicated psycopg3 connection, reading fields natively off each `LogRecord`
(never parsing formatted text). Never raises: missing psycopg, unset DSN,
unreachable DB, or a missing schema all degrade to a silent drop (bounded
queue, newest dropped — see `_BoundedQueueHandler.enqueue`), with at most one diagnostic notice per state
transition — written to a dedicated `sk_logging_internal.log`, never
stdout/stderr, never recursing into the DB handler that's failing. This
internal-diagnostics file exists purely to report on the DB path's own
health, not as a mirror of application log content — that role now belongs
to the small crash-safety `RotatingFileHandler` described above, which each
of the six sources also writes.
Flush-on-exit is registered via `atexit`, but DB writes are still batched up
to ~2s — see the DB-only tradeoff note above for what that means now that
there's no file behind it. `get_logger(name, file_name=...)` for any
`file_name` **not** in the six-table map still gets a plain
`RotatingFileHandler` as before — there's no DB destination for those, so
file logging remains their only copy and was never touched by this change.

**`run_id` correlation**: every process computes one `run_id` (UUID) at
`sk_logging` import time (`os.environ.get("SAGE_KAIZEN_RUN_ID") or
str(uuid.uuid4())`), stamped onto every `LogRecord` in that process via
`logging.setLogRecordFactory()` (process-global — covers root-logger/
third-party-library capture too, not just per-logger handlers). Subprocess
launch sites that spawn a sibling Python process using `sk_logging.py` (e.g.
wiki-embed services, CLAP) propagate `SAGE_KAIZEN_RUN_ID` in the child's env,
so one logical run's rows share a `run_id` across process boundaries. This is
a NEW, process-level correlation axis — deliberately separate from the
existing job-level `run_id` in `news_runs` (per-job, not per-process) and the
voice project's turn-level ZMQ `session_id`; not unifying those.

**No native partitioning yet** — Postgres best-practice guidance is to
partition at ~50-100GB or 100M+ rows; not there. Converting to a partitioned
table later is a well-documented, mechanical migration once volume actually
warrants it.

---

## 13) CROSS-PROJECT MODULE OWNERSHIP — this repo owns them outright (2026-08-24)

**RESOLVED.** The duplicate copies in `sage_kaizen_ai_ingest` have been
DELETED. These modules now exist once, here, and both projects import them
from this repo:

| Module | Note |
|---|---|
| `sk_logging` | |
| `pg_settings` | |
| `openai_client` | never duplicated; listed for completeness |
| `rag_v1.db.pg` | |
| `rag_v1.wiki.wiki_embed_config` | |
| `rag_v1.wiki.mm_embed_client` | |
| `rag_v1.media.media_embed_client` | driven at volume by ingest, not by this app |
| `news.news_settings` | **was missing from the old list** |
| `news.scheduler.news_scheduler` | **was missing from the old list** |

Still ingest-only, unchanged: `rag_v1.wiki.wiki_ingest`, `rag_v1.ingest.*`,
`rag_v1.media.media_ingest`, `news.enrichment.*`, `news.images.*`,
`news.summaries.*`, `news.base_job`.

### What the old arrangement actually cost

From 2026-08-04 until now this section said six (later seven) modules were
duplicated and that this repo's copy won. Both halves were understated.

- There were **nine**, not seven. `news_settings` and `news_scheduler` were
  never listed, and `openai_client` was listed despite not being duplicated.
- **Every single duplicated pair had diverged**, and the divergence ran in
  both directions — the dead copies were not stale, they were *different*.
  Three real fixes were sitting in unreachable code:

  | Dead copy held | Live copy had |
  |---|---|
  | `pg_settings`: `.env` resolved relative to the FILE | `env_file=".env"`, CWD-relative — every ingest script, launched from the ingest root, silently got the placeholder defaults |
  | `news_scheduler`: `COALESCE((metadata->>'fetch_retry_count')::int, 0)` | no COALESCE — `NULL < 3` is NULL, so an article that failed its FIRST fetch was **never** retried |
  | `news_scheduler`: singleton `_lock` + context-managed jobs | unsynchronised `start()`, leaked httpx clients |

  All three are now ported into the live copies. This is the real argument
  against "harmless" duplication: the tested, better-maintained copy was the
  one that never ran.
- **ingest's test suite was testing the dead copies.** Its `news_scheduler`
  reported 100% coverage while the live module in this repo reported **0%**.
  That test file now lives at `tests/test_news_scheduler.py` here.

**Cause of the shadowing.** `sage_kaizen_ai_ingest/_bootstrap.py` inserts each
project root only `if _s not in sys.path`. With both projects `pip install -e`'d
both roots are already present, so both inserts are skipped and `.pth` ordering
decides — putting `F:\Projects\sage_kaizen_ai` at `sys.path[0]`. Verified from
ingest's OWN venv, not the shared one; the result is the same either way.
`_bootstrap.py`'s docstring asserted the opposite and has been corrected.

**Working rule now.** These modules have one home. Changing one is still a
cross-project change — check ingest's call sites before you do — but there is
no longer a second copy to keep in sync or to mistake for the live one.

---

## 14) CODE-QUALITY REFACTOR — 2026-08-04/05

A senior-review pass over the whole app. Twelve defects fixed and five shared
abstractions extracted. Every item below has regression tests.

### Defects found and fixed

| # | Defect | Impact |
|---|---|---|
| 1 | `health_check()` probed four paths sequentially, each paying the full connect timeout | **8.03 s** to decide a server was down, on every ambiguous-score turn and every UI status refresh. Now short-circuits on transport failure: **2.00 s** (measured both) |
| 2 | `PostgresLogHandler` read `record.exc_info`, which `QueueHandler.prepare()` had already stripped | `log.<table>.exception` was **always NULL**; tracebacks were folded into `description` and truncated |
| 3 | `WikiEmbedServiceConfig.device` defaulted to `cuda:0` | Chat-time wiki retrieval could load jina-clip-v2 onto the display GPU. See §10 |
| 4 | 8 lazy singletons had no lock | Two context-injector workers could each construct a `WikiRetriever` → two cold CUDA inits on one GPU |
| 5 | Context collection used five sequential `.result(timeout=)` calls | Worst case was their **sum, 85 s**, not the max the docstring promised. Now one shared 30 s deadline |
| 6 | `llm_route()` returned without calling `_log_decision()` | `route_json` was missing for exactly the ambiguous turns |
| 7 | Dead imports, per-call constant rebuilds, a `trimmed` off-by-one, unpruned `_spawned_procs` | |
| 8 | `_TtsFilter` advanced one state transition per `feed()`, and discarded the buffer when a terminator was not in the current chunk | **Voice went silent at the start of every ARCHITECT turn and never recovered.** `</think>` almost never arrives whole from a token stream. Invisible in the UI, which renders unfiltered text |
| 9 | `web_researcher._searxng_search` read `r.content`; `WebResult` has `.snippet` | The `except Exception` swallowed the `AttributeError`, so **every successful SearXNG search was silently discarded**. The node had never contributed to a review |
| 10 | `_extract_ticker` substring-matched 3–4 letter aliases | `"something"` contains `eth`, `"goldfish"` contains `gold` — a turn mentioning a goldfish got a live gold-futures price injected. Now word-bounded, hyphen included in the boundary class |
| 11 | `_collect_diff("file", "")` → `MAIN_ROOT / ""` is a directory, `.exists()` passed, `read_text()` raised | Crash, reachable from `parse_review_command("review the file")` |
| 12 | `_is_market_query`'s comment named `"how much is this worth"` as the false positive it avoided — while `"how much is"` was in the list | Phrases are now tiered: weak ones additionally require a nameable instrument |

### Shared abstractions extracted

- **`lazy.py` → `@lazy_singleton`** — replaces ~10 hand-rolled
  `global X; if X is None` blocks, only two of which were locked. One lock per
  accessor; a `None` return is not cached (the optional-dependency accessors
  rely on that); `.reset()` for tests.
- **`rag_v1/embed/base_client.py` → `BaseHttpEmbedClient`** — one pooled
  `httpx.Client`, one `/health` semantic, one retry policy (`reraise=True`), one
  `close()` contract for all four embed clients. `ImageEmbedClient` was a second
  implementation of `MmEmbedClient` (same host, same endpoints) and is now a
  thin subclass; the name is kept because ingest imports it. **Both media
  clients were unpooled**, building and discarding a TCP connection per call
  while driven at volume by ingest — directly relevant to that project's
  unresolved ephemeral-port-exhaustion lead (its §15 candidate 4), which
  recorded that `MmEmbedClient` was checked and pooled but never checked these.
- **`server_manager._ensure_brain_running()`** — the four `ensure_*_running`
  functions were one body copied four times.
- **`context_injector._prepend_to_last_user()`** — the walk-backwards-find-user
  loop was written out once per context source.
- **`rag_v1/db/pg.py` onto `psycopg_pool.ConnectionPool`** — converges with
  `memory/db.py`; the project had two connection strategies. The old
  `threading.local()` cache only evicted on exception, so the first query after
  any server-side disconnect always failed; and three call sites called
  `conn.close()` on the shared cached connection, defeating the cache entirely.

### Test suite

239 tests → **1404**, coverage **27.7% → 82.9%**, `fail_under = 80` enforced.
`tests/test_coverage_config.py` guards the coverage config itself: `news/` and
`rag_v1/` are namespace packages and coverage only descends into *regular*
packages, so every directory has to be listed by hand — a list that rots
silently, because an unlisted directory is absent from the report rather than
reported as 0%. That understated the denominator by ~390 statements before it
was found.

`ui_streamlit_server.py` and `code_download.py` are deliberately unmeasured,
with the reason recorded in `pyproject.toml`.

---

## 15) MODEL EVALUATION HARNESS (`evals/`) — added 2026-08-06

**Before swapping any model, run the harness.** Full methodology and the decision
rule live in `Benchmarking_Kaizen_Models.md`; this section is the operating
summary.

### Why it exists

Public benchmarks cannot answer "should Sage Kaizen upgrade?" MMLU is saturated
and the leaders cluster inside measurement noise, while none of them measure the
things that decide it here: whether a 7B stays in English through a long story,
whether it emits clean router labels, whether `<think>` arrives in a shape
`_TtsFilter` can parse, whether the audio *and* vision encoders survive in one
mmproj. A leaderboard win that costs the audio encoder is a downgrade.

### Layers, cheapest first

| Layer | Module | Cost | Purpose |
|---|---|---|---|
| 1 — hard gates | `evals/gates.py` | seconds | pass/fail compatibility; run before downloading 20 GB |
| 2 — performance | `evals/bench.py` | ~30 min | `llama-bench` pp/tg, run comparison with tolerance |
| 2b — effective rate | `evals/mtp.py` | ~5 min | live-server decode **with** speculative decoding; reads `/metrics` spec counters |
| 3 — golden set | `evals/golden.py` | — | frozen prompts mined from `log.sage_kaizen` |
| 3 — scorers | `evals/scorers.py` | free | CJK, `<think>` leak, citations, refusal, length |
| 4 — live shadow | not built | 1 week | candidate on real traffic |

`evals/gguf_meta.py` is a dependency-free GGUF header reader underneath the
gates — it answers "does this mmproj carry both encoders?" with a few kilobytes
of I/O rather than loading a 20 GB file.

### Commands

```powershell
python scripts/run_model_gate.py --brain fast                     # incumbent
python scripts/run_model_gate.py --brain fast --plan-co-tenants   # + wiki/CLAP VRAM
python scripts/run_model_gate.py --brain architect --json

python scripts/run_bench.py --brain architect --label baseline    # servers must be STOPPED
python scripts/run_bench.py --compare baseline candidate

# effective decode rate WITH speculative decoding — server must be RUNNING,
# and the brain needs `metrics: true` in brains.yaml
python scripts/run_mtp.py --brain architect --label mtp-run --base-run baseline --warmup

python scripts/run_scorers.py --responses cand.jsonl --baseline base.jsonl
```

All three exit non-zero on failure, so they compose into a larger script.

### Two things that are easy to get wrong

1. **`llama-bench` does not measure MTP.** It feeds random tokens and exposes no
   `--spec-*` flags (re-verified on b10298), so its `tg` is the decode ceiling
   *without* speculative decoding. The effective rate must be measured against a
   running `llama-server` — that is what `scripts/run_mtp.py` is for. Comparing a
   candidate's llama-bench number to a production number compares two different
   quantities.
2. **`llama-bench` loads its own copy of the model.** Running it while
   `llama-server` holds the same weights double-allocates VRAM
   (2 × 21.3 GiB > 32 GiB on CUDA0). `run_bench.py` refuses to start when the
   brain's port is listening; `--force` overrides.

### Current state (llama.cpp **b10298 / `15586e2d7`** as of 2026-08-06)

- **ARCHITECT effective rate measured** → `benchmarks/results/mtp-b10298.mtp.json`.
  **64.25 t/s** with MTP against **45.62 t/s** base decode = **1.41× speedup** at
  **84.9% draft acceptance**. Acceptance is content-dependent: 90–94% on reasoning
  and code, ~78% on prose. The theoretical ceiling with one MTP head is 1.85×;
  the gap is the cost of running the head. The old 1.79× figure belongs to a
  different model on a different build — do not compare them.
- **The b10298 update cost 15–22% of raw throughput.** Reproduced twice, same
  flags, no power cap, CUDA graphs still working; upstream, not local. Not
  bisected — 700 commits at ~10 min/rebuild has not been paid for. Kept anyway:
  b10298 is what brought Qwen3-TTS, Eagle3-for-Qwen3.6, and the spec-decode
  metrics. See `Benchmarking_Kaizen_Models.md` §10.
- **ARCHITECT's prompt cache had never stored anything.** `cache_ram: 256` was
  sized for a model whose prompt state scales with tokens; Qwen3.6's 48 DeltaNet
  layers carry a fixed-size recurrent state costing ~320–470 MiB per snapshot, so
  every write was rejected — on 40-token prompts. Raised to 4096; verified
  1707.8 ms → 99.9 ms TTFT on a 3216-token prefix (§11). **FAST has not been
  checked** — grep its log for `exceeds cache size limit` while it serves.
- `scripts/run_bench.py` never ran end-to-end before this: it passed
  `llama-server.exe` to `llama-bench`. Fixed.
- **Golden set deliberately not frozen yet.** The log is currently dominated by
  wiki-ingest work rather than representative chat traffic, so a set mined today
  would over-weight whatever was being tested. `tests/test_evals_bench_and_scorers.py`
  asserts no set exists; delete that test when one is frozen.
- **Pairwise LLM judge and live shadow are not built** — Layer 3 tier 2 and
  Layer 4 respectively.

---

## 16) GPU ALLOCATION — measured and corrected 2026-08-06

The app was launched (`python -m streamlit run ui_streamlit_server.py`) and every
GPU sampled with `nvidia-smi` rather than reasoned about from config.

### What was wrong

**The display GPU had 106 MiB free.** CUDA0 drives three monitors and was
running ARCHITECT *and* BGE-M3, at 32081 / 32607 MiB — 99.7% full. Two causes:

1. **`--fit` is a startup-time target, not an enforced floor.** ARCHITECT
   reserves `fit_target` MiB of free VRAM as measured *at its own startup*.
   `_auto_start_servers` launches the embed chain and ARCHITECT in parallel
   threads — measured, both PIDs started in the same second — so BGE-M3 then
   allocated into the very reserve ARCHITECT had just set aside. The documented
   2 GB floor was never delivered.
2. **`--device CUDAn` restricts placement, not initialisation.** Every
   llama-server built a CUDA context on *every* visible GPU: three servers ×
   three GPUs, ~231 MiB each, including **694 MiB on the RTX 5080 this app never
   uses**.

### What changed

- **BGE-M3 (8020) moved CUDA0 → CUDA1.** Measured footprint 1152 MiB. On CUDA1
  the start-order problem cannot recur: `ensure_q5_running` already starts embed
  before FAST, so FAST's `fit` sees true free VRAM. ARCHITECT now owns CUDA0.
- **`server_manager._plan_cuda_isolation` + `_child_env`** set
  `CUDA_VISIBLE_DEVICES` per spawn and renumber `--device` accordingly, so
  brains.yaml keeps naming *physical* GPUs while each process sees exactly one.
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` is pinned into the child env rather than
  inherited — it is currently a user-level variable, and a correct GPU
  assignment must not depend on something outside this repo.

### Result (measured before → after)

| GPU | Free before | Free after | Contexts before → after |
|---|---|---|---|
| CUDA0 (display) | **106 MiB** | **3250 MiB** | 3 → 1 (ARCHITECT only) |
| CUDA1 (5090 OC) | 24002 MiB | 20498 MiB | 3 → 2 (FAST + BGE-M3) |
| CUDA2 (5080) | 15284 MiB | **15974 MiB, 4 MiB used** | 3 → **0** |

Verified per-process: each llama-server now holds exactly one CUDA context, on
the right card. Only ~1383 MiB of CUDA0's ~3.1 GB gain is attributable to these
changes (1152 embed + 231 context); the rest is desktop variance, which
`nvidia-smi` cannot decompose without elevation. Do not quote 3.1 GB as the
effect size.

### Measured VRAM footprints (use these, not estimates)

| Service | Measured |
|---|---|
| BGE-M3 (8020) | 1152 MiB |
| jina-clip-v2 wiki embed (8031) | 3234 MiB |
| llama-server CUDA context, per foreign GPU | ~231 MiB |

`evals/gates.py: DEVICE_CO_TENANTS` carried 2.0 GiB for jina-clip-v2 — a ~60%
understatement, in the wrong direction for a gate meant to refuse a model that
will not fit. Corrected to the measured values.

### What this repo does NOT copy from sage_kaizen_ai_ingest

Ingest keeps work off cuda:0 by application-level device selection plus hard
guards, and **explicitly rejected `CUDA_VISIBLE_DEVICES`**: it broke
jina-clip-v2's `trust_remote_code` paths, which passed `/health` and then
returned 500 on real inference (`wiki_ingest.py` documents this). That is a
PyTorch failure mode. llama-server is C++ and was verified here before the code
was written — a BGE-M3 server under `CUDA_VISIBLE_DEVICES=1 --device CUDA0`
served real embeddings holding a context on exactly one GPU.

**So the isolation applies only to llama-server spawns.** The wiki embed service
is PyTorch and is left alone; it was measured holding a context on cuda:1 only,
because PyTorch initialises the device it is told to use and no others.

### Placement decisions deliberately NOT taken

- **Swapping ARCHITECT to CUDA1.** CUDA1 must host the PyTorch embed services
  (torch.compile autotune on the display GPU is the TDR trigger), and
  ARCHITECT's ~27 GB plus jina-clip-v2 plus CLAP does not fit in 32 GB. The
  current split is forced, not preferred.
- **Moving anything onto the RTX 5080 (CUDA2).** It is idle and tempting, but
  ingest's own notes call the USB4/OCuLink tunnel unstable enough to restart
  service D every 2 h. BGE-M3 is on the critical path for every RAG turn.
  CUDA1 holds ~11 GB of 32 GB with everything resident, so there is no pressure
  to justify the risk. **The 5080 remains deliberately unused headroom.**

### Known defect, now worked around

`server_manager` writes a launch header (EXE / MODEL) to the brain's log before
spawning — and llama-server opens `--log-file` in **truncating** mode, so that
header has never survived a single start. The physical→visible GPU mapping is
therefore recorded via `sk_logging` (`log.sage_kaizen`), not the header. This
matters because after isolation each server's own log only ever says "CUDA0".

---

## 17) WIKI-RAG WAS SILENTLY DEAD — RESOLVED 2026-08-24

> **STATUS: FIXED.** `wiki_chunks` is now a 32-way HASH-partitioned table with
> 32 halfvec **ivfflat** indexes (1,304 GB on the NVMe tablespace), and wiki-RAG
> returns relevant results in 4-8.7 s. See §18 for the migration and §18.2 for
> the measured outcome. The section below is kept because the failure mode it
> describes is permanent: **it recurs for the duration of every bulk ingest**,
> since `--manage-indexes` still drops the vector indexes for the whole run.
> The `WikiRetriever` guard added here is what makes that degradation loud
> instead of silent.

**The original problem (2026-08-06):** `wiki_chunks` had no ANN index.
`sage_kaizen_ai_ingest`'s `--manage-indexes` drops it before a bulk run and
rebuilds it after, and ingest was mid-run. That is correct behaviour for ingest.
The problem is what it did to *this* app, which nobody had looked at.

### Symptom

`WikiRetriever.search()` still runs and still looks healthy. The embed step is
fine — 0.4 s to jina-clip-v2 on port 8031. The pgvector query then
**sequential-scans a 3.5 TB table**. Measured: one call had not returned after
15 minutes; killing the Python client did not stop it — three parallel backends
were still scanning 24 minutes later and had to be cancelled with
`pg_cancel_backend()`.

`context_injector.apply_rag_and_wiki_parallel()` collects all five workers under
one shared 30 s deadline, so the wiki worker **always** misses it. Net effect:

- wiki-RAG contributes **nothing** to any chat turn, and has not for the
  duration of the ingest run
- nothing in the UI or logs says so — it looks identical to "no wiki hits"
- each wiki-routed turn leaves an abandoned full-table scan running for minutes,
  competing with ingest and with every other query in the app

`rag_chunks`, `image_embeddings`, `audio_embeddings`, `lyrics`, `episodes`,
`daily_news` and `news_image_embeddings` all have HNSW indexes. `wiki_chunks`
and `wiki_images` are the only two without.

### The fix (IMPLEMENTED 2026-08-06, still active)

`WikiRetriever` checks for the index once and short-circuits when it is
absent — a single `pg_indexes` lookup, cached — logging a clear "wiki-RAG
disabled: no vector index" rather than firing a 3.5 TB scan per turn. That
converts a silent 30 s timeout plus minutes of stray I/O into an immediate,
visible no-op. It also makes the degradation honest: the retriever already has a
`DisplayGpuRefused`-style guard pattern to follow.

### Cost of the rebuild — SUPERSEDED by measurement, see §18.2

The estimate below was made before the work was done. What actually happened:
HNSW was abandoned after a ~44-day projection, and 32 **ivfflat** indexes were
built in **4.27 days** totalling **1,304 GB**. Keep the reasoning, ignore the
numbers.

Original estimate: 508,359,968 rows × 1024-dim float32, HNSW at
m=16/ef_construction=100: **~2.2 TB and days to ~2 weeks** — the graph needs ~2.2 TB of
`maintenance_work_mem` and the machine has 192 GB, so the build is almost
entirely on-disk. Full measurements, the growth trend (chunks per page rise
27 → 154 → 400 across the corpus, so the finished index may exceed 5 TB), and
the `halfvec` / Matryoshka / partitioning alternatives are documented in
**sage_kaizen_ai_ingest CLAUDE.md §21**. Do not rebuild mid-run.

### Also found: PostgreSQL was on stock defaults — TUNED 2026-08-06/24

Was `shared_buffers = 128MB`, `work_mem = 4MB`, `maintenance_work_mem = 64MB`,
`effective_cache_size = 4GB` on a 192 GB host. Now applied via `ALTER SYSTEM`
(most settings are sighup, so no restart was needed): `max_wal_size` 16GB,
`wal_compression` lz4, `effective_cache_size` 128GB, `work_mem` 64MB,
`checkpoint_timeout` 15min, plus `shared_buffers` 512MB and `wal_buffers` 64MB
on the 2026-08-11 restart. `pg_wal` also moved to NVMe (E:), which measured a
**+70% copy throughput** gain. Rationale for every value:
`config/postgres/sage_kaizen_tuning.conf`.

### Notes for anyone measuring this table

- `n_live_tup` reads **0** for `wiki_chunks`/`wiki_images` — the tables have
  never been ANALYZEd. They are not empty; use `pg_class.reltuples` and
  `pg_relation_size`, or ANALYZE first.
- The 1024-dim vectors are 4104 B each and therefore always TOASTed: 2640 GB of
  the table's 3503 GB is TOAST. Any per-row scan pays detoasting on every row.
- Always `SET statement_timeout` before querying this table interactively. A
  bare `SELECT` will not come back.

---

## 18) POSTGRESQL TUNING + THE wiki_chunks PARTITION MIGRATION (2026-08-06)

Two artifacts, neither applied automatically. Both exist because the host
reboots under load (~9 unclean shutdowns in 7 days; ingest CLAUDE.md §18 root
cause still open), which makes "how long can one unit of work be?" the design
constraint rather than raw throughput.

### `config/postgres/sage_kaizen_tuning.conf`

Every value carries its measured before-value and a justification. Apply with
`include_if_exists` as the LAST line of postgresql.conf, then restart.

Three that are counter-intuitive and should not be "corrected" later:

- **`shared_buffers = 512MB`, not 25% of RAM.** PostgreSQL's own wiki caps the
  useful Windows range at 64MB–512MB; large values are ineffective there.
- **`max_wal_size = 16GB`, not larger.** Bigger WAL means longer replay after
  each crash, and crashes are routine here. This is a deliberate compromise.
- **Durability is restated, not relaxed.** `fsync` / `full_page_writes` /
  `synchronous_commit` are pinned ON with a comment explaining why, so a future
  "speed up the ingest" pass has to consciously override an invariant. The
  standard bulk-load advice would risk a 3.5 TB database that costs weeks to
  rebuild.

`random_page_cost` and `effective_io_concurrency` stay at HDD values globally
and should be overridden **per tablespace** once the index moves to NVMe.

### `scripts/migrate_wiki_chunks_partitioned.py`

HASH-partitions `wiki_chunks` by `page_id` into 32 partitions and builds one
halfvec HNSW index per partition. Phases (`--create`, `--copy`, `--constraints`,
`--index`, `--verify`, `--swap`) are individually idempotent and resumable.

**Partition count is derived, not chosen for tidiness.** An HNSW build is fast
only while the graph fits in `maintenance_work_mem`. At halfvec ≈2.3 KB/element,
508M rows / 32 ≈ 15.9M rows ≈ **36 GB per partition**, which fits a 48 GB
session `maintenance_work_mem` on a 190 GB host. Full-precision `vector(1024)`
would be ~68 GB per partition and would not — which is the second reason for
halfvec, after the D: drive capacity argument.

HASH rather than the existing `first_letter` column because letter distribution
is heavily skewed and vector search probes every partition anyway.

Crash tolerance specifics: the copy commits per batch and stores its resume
point in `wiki_chunks_migration`; index builds skip partitions that already have
a **valid** index and **drop INVALID ones first** — that being exactly the
debris a crash mid-`CREATE INDEX` leaves. `--swap` renames rather than drops,
and requires `--i-understand-this-is-destructive`.

### The coupling that makes this a code change, not just a DDL change

**pgvector only uses a halfvec index when the query casts identically.** Build
the index and leave the query as `embedding <=> $1::vector` and the index is
silently ignored — a full scan, but now with an index present so the §17 guard
waves it through. That would be worse than having no index.

So `WikiRetriever` now detects *which* index exists (`_query_index_kind` →
`"halfvec"` / `"vector"` / `None`) and emits the matching SQL. Detection reads
`pg_get_indexdef()` rather than joining `pg_attribute` on `indkey`, because an
expression index stores 0 there — an attribute join reports the halfvec index,
the one this migration builds, as missing.

### Schema consequence — this is a CROSS-PROJECT change

PostgreSQL requires every UNIQUE/PK on a partitioned table to include all
partition key columns. With HASH(page_id):

| constraint | effect |
|---|---|
| `uq_wiki_chunks_page_hash (page_id, chunk_hash)` | unchanged — already includes `page_id`; ingest's `ON CONFLICT` keeps working |
| `wiki_chunks_pkey (chunk_id)` | **illegal** — becomes `(chunk_id, page_id)` |
| `DELETE ... WHERE page_id = %s` | unchanged, and gains partition pruning |

`sage_kaizen_ai_ingest` writes this table. Do not run the migration without
reading ingest CLAUDE.md §21.

### Not yet decided

The migration requires copying ~3.5 TB (hash partitioning cannot ATTACH an
existing unpartitioned table). A **partial-index** alternative achieves the same
per-unit crash tolerance with no rewrite, at the cost of a UNION-ALL query
shape. See the open question at the end of ingest §21.

### 18.1) `scripts/move_pg_wal_to_nvme.ps1` — REQUIRES ELEVATION

Moves `pg_wal` from the HDD to `E:\pgwal` via a directory junction. Written
because the copy phase measured **77% disk busy at only 33.7 MB/s with queue
depth 0.6** on drive I: — seek contention, not bandwidth, with source reads,
partition writes, TOAST and WAL all on one spindle.

**Must run in an Administrator PowerShell.** Stopping the service and writing
inside PGDATA both need elevation; an unelevated shell fails with "Cannot open
'postgresql-x64-18' service". The migration itself does not need elevation —
only this.

Safety properties: verifies a **clean** shutdown via `pg_controldata` before
touching anything (moving WAL after an unclean shutdown can destroy segments
crash recovery needs — non-optional on this host); **copies** before renaming;
never deletes, leaving `pg_wal_old` in place for manual removal after
verification; and restores + restarts the service on any failure.

---

## 19) POSTGRESQL CHANGES — KEEPING THE TWO PROJECTS IN SYNC (2026-08-10)

Both projects share **one** PostgreSQL database. That is a deliberate choice, not
an accident, but it means a schema change in one repo can silently break the
other — which is exactly what happened on 2026-08-10 (see the case study below).
This section is the governing process. **The identical section is ingest
CLAUDE.md §22; if you change one, change both.**

### 19.1 The rule: the writer owns the schema

The app that **writes** a table owns its DDL, migrations and indexes. The other
app is a **reader** and must never issue DDL against it.

| Owner | Tables | Schema files live in |
|---|---|---|
| **ingest** | `wiki_bundles`, `wiki_pages`, `wiki_chunks`, `wiki_images` | ingest |
| **ingest** | `daily_news`, `news_runs`, `news_story_clusters`, `news_briefs`, `news_article_summaries`, `news_cluster_summaries`, `news_article_images`, `news_image_embeddings` | ingest |
| **ingest** | `rag_chunks` | ingest |
| **ingest** | `image_embeddings`, `audio_embeddings`, `audio_clusters`, `media_files`, `lyrics`, `lyrics_fetch_log` | ingest |
| **main** | `memory.*` (episodes, profiles, rules, reflections, audit_log) | main |
| **main** | `public.ratings` | main |
| **main** | `langgraph.*` (LangGraph checkpoints) | main |
| **main** | `log.*` — **shared write**, main owns by exception (§12) | main |

Derived from what each project actually executes INSERT/UPDATE against, verified
2026-08-10 — not from where files happen to sit today.

### 19.2 Files currently misfiled (relocate; deliberately deferred)

These live in **main** but describe **ingest-owned** tables. Left in place for
now because the `wiki_chunks` partition migration is mid-copy and moving DDL
during it adds risk for no benefit. Move them once it completes:

```
rag_v1/db/schema.sql                 -> rag_chunks
rag_v1/db/wiki_schema.sql            -> wiki_*
rag_v1/db/image_embeddings.sql       -> image_embeddings
rag_v1/db/media_schema.sql           -> media_files
rag_v1/db/lyrics_schema.sql          -> lyrics
rag_v1/db/audio_clusters_schema.sql  -> audio_clusters
```

Correctly placed and staying: `log/db/log_schema.sql`, `scripts/memory_schema.sql`,
`feedback/schema.sql`, `scripts/setup_*.sql`.

`scripts/migrate_wiki_chunks_partitioned.py` is an ingest-owned-table migration
living in main. It stays here: it is running, it is a one-off, and relocating a
script mid-migration risks the resume path. Revisit after `--swap`.

### 19.3 Expand / contract — never break the reader in the same step

The standard pattern for schema change when writer and reader deploy
independently ([expand and contract / parallel change](https://www.prisma.io/dataguide/types/relational/expand-and-contract-pattern)).
Three phases, and **the reader must work after every one of them**:

1. **Expand** — add the new thing alongside the old. New column nullable, new
   table, new index. Never drop, never rename, never tighten in this step.
2. **Migrate** — backfill data; update the reader to handle both shapes; update
   the writer to produce the new shape.
3. **Contract** — only once the reader provably no longer needs the old shape,
   remove it.

A rename is never one step: it is add-new, dual-write, migrate-readers,
drop-old. The same applies to changing a column's type, tightening NOT NULL, or
changing an index's opclass.

### 19.4 Checklist — run before ANY PostgreSQL change

1. **Who writes this table?** That project owns the change. If it is the other
   project's table, stop and go do it there.
2. **Who reads it?** `grep` the *other* project for the table name. Every reader
   must survive each phase independently.
3. **Does it change a PK, unique constraint, or partition key?** Then check the
   other project's `ON CONFLICT`, `DELETE ... WHERE`, and any explicit column
   lists — those bind to constraint shape.
4. **Does it change an index?** Check the other project's index management. A
   `DROP INDEX IF EXISTS <name>` that no longer matches is a **silent no-op**,
   not an error.
5. **Does it change an opclass or add a cast?** (e.g. `vector` →
   `halfvec`.) The reader's query must cast identically or the index is ignored
   and the planner silently reverts to a sequential scan.
6. **Are constraints preserved?** Recreating a table drops NOT NULLs and foreign
   keys unless you restate them. Nothing fails; the constraint is just gone.
7. **Update BOTH CLAUDE.md files** — this section and ingest §22 are one
   document in two places.
8. **Run both test suites.** Main and ingest.

### 19.5 Case study: what this process would have caught

The `wiki_chunks` partition migration (§18) changed a table **ingest writes and
main reads**, and broke ingest three ways that no test would have caught:

- `DROP INDEX IF EXISTS hnsw_wiki_chunks_embedding_cos` stopped matching
  anything once the per-partition indexes were named
  `wiki_chunks_part_pNNN_hv_hnsw`. It logged success and left 32 HNSW indexes
  in place for a bulk ingest to fight — **checklist item 4**.
- `CREATE INDEX CONCURRENTLY` is rejected on partitioned tables, so the rebuild
  would hard-error — **item 4**.
- The new table's `CREATE TABLE` silently omitted 7 NOT NULLs and both
  `ON DELETE CASCADE` foreign keys — **item 6**.

All three were found by reading the other project, not by running tests. That is
why items 2-6 are manual greps rather than assertions.

### 18.2) OUTCOME — the migration as actually built (2026-08-24)

Every phase except `--swap` is complete. What follows is measured, not planned.

| Phase | Result |
|---|---|
| copy | 512,483,256 rows |
| dedupe | 999,970 duplicate rows removed |
| constraints | 5 btree indexes, 7 NOT NULLs, 2 FKs validated, ANALYZE (27,662 s) |
| index | **32/32 ivfflat, all valid, 1,304 GB** on `sage_nvme` (D:) |
| verify | **source 512,483,260 = target 512,483,256 + 4 quarantined** |
| swap | NOT RUN — destructive, gated |

### HNSW was abandoned on measurement

Partition 1 of 32 ran **8.24 h** and had written 8.6 GB of ~35 GB (~25%), which
projects to ~33 h/partition and **~44 days**. CPU was at 7% while the source HDD
sat at 0% idle delivering 23.4 MB/s: the bottleneck was never graph construction
or the NVMe, it was reading ~89 GB of heap + TOASTed vectors per partition off a
7200 RPM disk. ivfflat pays that read once instead of repeatedly.

ivfflat actual: first two partitions 7.8 h each (competing with leftover
autovacuum/ANALYZE), remaining 30 averaged **2.90 h**. Total **102.5 h = 4.27
days**. A 9.7-day projection made from the first two samples was 2.3x pessimistic.

### probes is NOT sqrt(lists) — the most important number here

pgvector's `probes ~= sqrt(lists)` assumes ONE index. A nearest-neighbour query
has no `page_id` predicate, so it **cannot prune partitions** — all 32 are
probed and the cost multiplies by 32. Measured 2026-08-24, 12 random queries per
setting on an identical sample:

```
probes=63 (sqrt)   66.3 s single query      -> past the 25 s statement_timeout
probes=20          p90 24.98 s  max 29.51 s -> at/over the timeout
probes=10          p90 10.03 s  max 10.19 s  recall@10 75%   <- chosen
probes=5           p90 36.45 s (cold cache)  recall@10 75%
```

**Recall was identical at 5/10/20**, so probing more buys latency and nothing.
At 63 every wiki-RAG query would have hit the timeout and returned nothing —
indistinguishable from "no matches", the exact failure §17 exists to prevent.
Two tests asserted the sqrt rule and therefore enforced the bug; both now assert
the partition-aware constraint instead.

The 75% figure is **self-recall** (does a vector retrieve itself), a floor not a
full measure. It did not improve at 20 probes, so it is not probe starvation —
most likely halfvec quantisation plus near-duplicate wiki boilerplate filling
the top-10. Real semantic recall needs the golden set (§15) and is unmeasured.

### A copy-phase bug worth remembering

`phase_copy` committed the batch INSERT and the resume marker as **two separate
autocommit transactions**. A process killed in that gap left the rows committed
with the marker unmoved, so the next run re-copied that batch. Seven-ish
interruptions produced 999,970 duplicates — 9.9997 batches of 100,000, which is
what identified the cause. The unique index caught it; `--dedupe` repaired it;
both statements now commit together, with bisection using SAVEPOINTs.

### Four source rows are permanently unreadable

`wiki_chunks` TOAST has one damaged 8 KB page per incident, found by
`data_checksums = on`: chunk_ids 173,810,706 / 173,810,708
(`Bavarian_Lower_Inn_Valley`) and 260,664,119 / 260,664,120
(`Catherine_(Black_Clover)`). Recorded in `wiki_chunks_corrupt` with page_id and
title, absent from the new table, **source never modified**. Re-ingest those two
pages to restore them.

### 16.1) GPU REMAP APPLIED — 2026-08-24 (Option 2)

The display GPU no longer carries the heavy model. Applied in brains.yaml:

| GPU | Before | After |
|---|---|---|
| CUDA0 (5090, 3 monitors) | ARCHITECT + BGE-M3, **106 MiB free** | **FAST only** (~10.3 GB), ~20.8 GB free |
| CUDA1 (5090 OC, headless) | FAST + embeds | ARCHITECT + BGE-M3 |
| CUDA2 (5080 eGPU) | **idle** | summarizer + jina-clip-v2 + CLAP |

**The summarizer moved off the CPU.** It sat on the search critical path doing
CPU inference while a 16 GB GPU idled. OCuLink Gen4 x4 costs 2-3% for a
fully-resident model, and it is the lowest-risk eGPU tenant: a tunnel hiccup
delays search summarisation, it does not break a chat turn. BGE-M3 stays on a
5090 precisely because it *is* on the critical path.

**`_auto_start_servers` now starts embed synchronously BEFORE the brains.**
BGE-M3 and ARCHITECT now share CUDA1, and `--fit` samples free VRAM at its own
startup — launched in parallel that is a race, and it is exactly how CUDA0 ended
up with 106 MiB free. `ensure_embed_running` is idempotent, so the later
`ensure_q5_running` call costs nothing.

`evals/gates.py: DEVICE_CO_TENANTS` and `scripts/run_model_gate.py: _DEVICE_INDEX`
were both stale after the remap and are corrected — the gate was still assuming
FAST lived on CUDA1 with all four embed services beside it.

### 16.2) ivfflat probes: FINAL value is 5, not 10 and definitely not 63

Cold-page measurement is the only one that matters — a warm repeat is ~1 s at
any setting. 10 random vectors scattered across the 1.3 TB index:

```
probes=10   median 15.36 s   p90 74.17 s   1/10 OVER the 25 s timeout
probes=5    median  4.10 s   p90  8.76 s   0/10 over      <- chosen
probes=3    median  1.49 s   p90  2.96 s   0/10 over
```

probes=3 is faster still but its recall is unmeasured (a single probe vector
found itself at 3; at 1 it did not), so 5 buys margin for ~2.6 s.

Post-swap end-to-end at probes=5: 4/5 real queries return relevant results in
4.0-8.7 s warm. The empty one was the distance/noise gate, not a timeout.
