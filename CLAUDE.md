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
  - FAST brain (default): `Qwen2.5-Omni-7B-Q6_K` (port 8011, RTX 5090 OC/CUDA1) — multimodal: text + image + audio input via mmproj encoder
  - ARCHITECT brain (on demand): `Qwen3.6-27B-MTP-Q6_K` (port 8012, RTX 5090/CUDA0) — **128K context**, reasoning mode (`<think>` tokens), MTP speculative decoding: draft-mtp, hybrid DeltaNet+attention
  - Summarizer (lightweight): `Qwen3-4B-Q8_0` (port 8013, CPU-only) — search evidence summarization before context injection
- **Router**: selects brain, applies templates, escalates to ARCHITECT when needed (`router.py`)
- **Streamlit UI**: chat interface, status, templates visible, debugging-friendly (`ui_streamlit_server.py`)
- **Chat Service**: full turn lifecycle — route → memory → prompt → parallel RAG → stream (`chat_service.py`)
- **Pi Agent Transport**: ZeroMQ messaging to Raspberry Pi agents (`voice_bridge.py` owns host ZMQ; `agents/` planned)
- **RAG v1**: ingest (folder + RSS + web + ZIM) into PostgreSQL + pgvector; parallel query-time retrieval
  - `rag_v1/wiki/` — Wikipedia multimodal RAG (jina-clip-v2 embeddings, text + image)
  - `rag_v1/media/` — Cross-modal ingest: images (jina-clip-v2, 1024-dim) + audio (CLAP, 512-dim)
  - `rag_v1/embed/` — BGE-M3 embed client (wraps port 8020)
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
  - `review_service/trigger.py` — `is_review_command()` heuristic; called by router.py
  - `review_service/output/` — review_writer, adr_writer, patch_writer (write to `reviews/`, `docs/03-DECISIONS/`, `patches/`)
- **Supporting root-level modules**:
  - `document_parser.py` — multi-format doc extraction (docx, xlsx, csv, code, txt, etc.)
  - `input_guard.py` — prompt-injection defense for all external content (RAG chunks, web snippets)
  - `env_utils.py` — per-call env var accessors (`env_bool`, `env_int`, `env_float`, `env_str`); re-read every turn
  - `mermaid_streamlit.py` — Mermaid diagram detection and rendering
  - `sk_logging.py` — centralized rotating log configuration; also mirrors structured log records into PostgreSQL (`log` schema, best-effort, see §12)
  - `pg_settings.py` — Pydantic BaseSettings for PostgreSQL DSN
  - `voice_bridge.py` — ZMQ bridge binding ports 5790/5791/5792 for the voice app
- Review `config/brains/brains.yaml` for latest AI models and all server settings

### Service / Port Inventory
| Service | Model | Port | GPU | Purpose |
|---------|-------|------|-----|---------|
| FAST brain | Qwen2.5-Omni-7B-Q6_K | 8011 | CUDA1 (5090 OC) | Multimodal chat (text + image + audio via mmproj) |
| ARCHITECT brain | Qwen3.6-27B-MTP-Q6_K | 8012 | CUDA0 (5090) | Deep reasoning; 128K ctx; `<think>` tokens |
| Summarizer | Qwen3-4B-Q8_0 | 8013 | CPU-only | Lightweight search evidence summarization |
| BGE-M3 embed | bge-m3-FP16 | 8020 | CUDA0 (5090) | RAG text embeddings (1024-dim) |
| Wiki embed A | jina-clip-v2 | 8031 | CUDA0 (5090) | Wikipedia multimodal embeddings (workers A1/A2) |
| Wiki embed B | jina-clip-v2 | 8032 | CUDA1 (5090 OC) | Wikipedia ingest only (workers B1/B2; stop FAST brain first) |
| CLAP embed | clap-htsat-unfused | 8040 | CUDA1 (5090 OC) | Audio embeddings (512-dim) |
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

6. **PostgreSQL schema migrations for cross-project concerns live in this (main) project**
   - `log/db/log_schema.sql` (structured logging, §12) follows this rule — run `scripts/setup_log_schema.sql` once as superuser first, then `log/db/log_schema.sql` as `sage`
   - **Known exception**: `news/db/news_schema.sql` and `news/db/migrations/` live in and are git-tracked by `sage_kaizen_ai_ingest`, not here — a pre-existing precedent from before this rule was made explicit. Don't treat it as license to add new schema files outside this project; it's flagged here rather than silently left inconsistent.

---

## 6) CURRENT HARDWARE (Authoritative)
User rig also known as "my rig":
- Motherboard: Gigabyte X870E AORUS XTREME AI TOP AMD AM5 eATX Motherboard
- CPU: AMD Ryzen 9 9950X3D
- RAM: 192 GB DDR5 Speed: 6400 MT/s
- GPU0: CUDA 0 - NVIDIA GeForce RTX 5090 (32GB VRAM) — primary display GPU (3 monitors); ARCHITECT Brain + BGE-M3 embed
- GPU1: CUDA 1 - Gigabyte GeForce RTX 5090 OC (32GB VRAM) — no display; FAST Brain + Wiki embed B + CLAP embed
- GPU2: CUDA 2 - Gigabyte GeForce RTX 5080 (16GB VRAM) — no display; 
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
Wiki embed service A (port 8031) runs on **CUDA0 (RTX 5090)**; wiki embed service B (port 8032) runs on **CUDA1 (RTX 5090 OC)**.  
The RTX 5090 (CUDA0) drives the display (3 monitors) and must not run sustained compute — Windows TDR (2 s) resets the display driver if a CUDA kernel runs too long, causing a black screen requiring reboot.

Before running a long wiki ingest session, set GPU power limits once as Administrator:
```powershell
F:\Projects\sage_kaizen_ai_ingest\scripts\set_gpu_limits.ps1
```
Limits: GPU0 RTX 5090 → 420 W (stock 575 W; display GPU), GPU1 RTX 5090 OC → 500 W (stock 575 W+; sustained ingest).  
Then run ingest with `--no-power-limits` (limits persist until reboot).  
If power limits cannot be applied, `wiki_ingest.py` auto-falls back to 2 workers + 75 ms throttle.  
Default throttle is 25 ms per worker even with power limits applied (protective baseline).  
Stop the FAST brain (port 8011) before starting wiki ingest — conservative: wiki-embed-B + FAST brain both run on CUDA1; stopping FAST avoids GPU compute contention during long ingest sessions.

---

## 11) FAST Brain Model — Upgrade Research Log (2026-04-07, updated 2026-06-11)

### Current State (as of 2026-06-11)
- **Model**: `Qwen2.5-Omni-7B-Q6_K` — **Q6_K downquant applied**; saves ~1.85 GB VRAM vs Q8_0 with negligible quality loss (~0.1–0.2 PPL)
- **GPU**: Gigabyte GeForce RTX 5090 OC (CUDA1, 32 GB VRAM) — upgraded from RTX 5080 (16 GB) on 2026-05-24
- **llama.cpp build**: b9598 (fdc3db9b6) — custom MSVC build with CUDA SM_120 Blackwell kernels (`ARCHS=1200`, `BLACKWELL_NATIVE_FP4=1`)
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
| Flash attention | `flash_attn: true` in brains.yaml; C++ runtime only (not Python) | `brains.yaml:67` |
| KV prefix cache | `cache_ram: 512`, `slot_prompt_similarity: 0.10` | `brains.yaml:79,84` |
| 32K context | `ctx_size: 32768` — 1 image ≈ 1280 tokens, ~30K for conversation | `brains.yaml:55` |
| Port 8011, CUDA1 | Hard-coded in routing and inference session | `brains.yaml:39,43` |
| TTS voice pipeline | Audio output is text-only; Kokoro handles TTS separately | `voice_bridge.py` |

### Watch List — When to Revisit the FAST Brain Upgrade
Monitor these milestones; when any trigger is met, re-evaluate:

1. **Qwen3.5-Omni llama.cpp audio PR merges** — check https://github.com/ggml-org/llama.cpp/pulls for "omni" or "audio" PRs. This is the primary upgrade path when it lands. Model will need 5090 or GPU upgrade (30B+ size).

2. **Voxtral-Mini-3B crash fixed** — track llama.cpp issue #21080. If fixed, Mistral's 3B audio model could run as a lightweight audio-only companion on CUDA1 alongside a stronger text model.

3. **Qwen2.5-Omni-14B or larger Omni release** — Alibaba has only released 3B and 7B Omni variants. A 14B would be a direct drop-in upgrade (~14 GB weights Q6_K = marginal, Q4_K_M = comfortable on the RTX 5090 OC's 32 GB; ~21.7 GB headroom available).

4. **Gemma 4 audio support in llama.cpp** — Gemma 4 natively supports audio but llama.cpp audio parsing is not yet implemented. Track https://github.com/ggml-org/llama.cpp/issues.

5. **llama.cpp rebuild** — Current build b9598 (fdc3db9b6) includes SM_120 Blackwell kernel optimizations. Monitor https://github.com/ggml-org/llama.cpp/releases for newer builds; a Chunk-fused GatedDeltaNet kernel for Blackwell (PR #21074) is in development and would benefit ARCHITECT throughput when merged.

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
queue, oldest dropped first), with at most one diagnostic notice per state
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
