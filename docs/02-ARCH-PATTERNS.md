# 02 — Architectural Patterns Used in Sage Kaizen

This document describes the repeatable design patterns used across the system.
When adding a new module, identify which patterns apply and follow their rules.

---

## 1. Dual-Brain Pattern

### Description
Two independent llama-server instances with distinct roles:

| Brain | Model | Port | GPU | Role |
|-------|-------|------|-----|------|
| FAST | Qwen2.5-Omni-7B-Q6_K | 8011 | CUDA0 (RTX 5090, display) | Default; multimodal (text + image + audio); low-latency |
| ARCHITECT | Qwen3.6-27B-MTP-Q6_K | 8012 | CUDA1 (RTX 5090 OC) | Deep reasoning; 128K context; `<think>` tokens; MTP speculative decoding |

### Why
- Performance isolation — FAST handles 80–90 % of requests without occupying the 5090
- Deterministic escalation — routing score ≥ 3 → ARCHITECT (heuristic) or LLM-assisted choice
- Complexity separation — ARCHITECT used for code review, philosophy, long-form analysis

### Rules
- FAST handles everything that doesn't require deep reasoning or 128K context
- Routing must be transparent (logged `RouteDecision` includes score + reason)
- Creative writing routes to ARCHITECT to avoid Qwen2.5-Omni language drift
- Thinking budget (`thinking_budget`) may be applied per turn for ARCHITECT

---

## 2. Template-First Prompting Pattern

### Description
All prompts are assembled from composable, auditable strings before any model call.
`TemplateKey` enum values map to full text blocks in `TEMPLATES` dict.

### Why
- Auditable — every prompt element is in `prompt_library.py` or in `.env` overrides
- Composable — `build_messages()` assembles `[system + memory + core + extra_templates]`
- No hidden mutation — prompt construction happens entirely in `chat_service.select_templates()` before the HTTP call

### Rules
- `prompt_library.py` is the single source of truth for all prompt text
- Per-turn overrides via `TurnConfig.extra_templates` — never mutate `TEMPLATES` dict
- `settings.py` imports `sage_kaizen_system_prompt` from `prompt_library` — never define it elsewhere

---

## 3. Idempotent Ingestion Pattern

### Description
All ingest pipelines use stable source IDs + content hashing to allow safe re-runs.

### Why
- Prevents duplicate vector entries when ingest is re-run after a crash
- Enables partial-failure recovery without corrupting the vector index

### Rules
- Source ID format: `localfile:<path>`, `rss_item:<url>`, `web:<url>`, `wiki:<title>`
- Content hash: `xxhash` or `sha256_text()` on raw text before chunking
- Upsert via `executemany` — `ON CONFLICT (source_id, chunk_id) DO UPDATE`
- Batch sizes from `brains.yaml` ingest config (`text_batch`, `image_batch`)

---

## 4. Lazy Singleton Pattern

### Description
Expensive resources (DB connections, ML model services, process spawning) are initialized on first use, not at import time.

**Implementation: `@lazy_singleton` from `lazy.py`.** Decorate a zero-argument
factory; the wrapper handles the locking. Do not hand-roll
`global X; if X is None: X = ...` — that is what this replaced.

```python
from lazy import lazy_singleton

@lazy_singleton
def get_orchestrator() -> SearchOrchestrator:
    return SearchOrchestrator()

get_orchestrator()          # constructs once, under a lock
get_orchestrator.reset()    # drops it — tests only
```

### Where Used
- `rag_v1/runtime/context_injector.py` — `_rag_pair`, `_get_wiki_retriever`, `_get_music_retriever`
- `search/search_orchestrator.py` — `get_orchestrator()`
- `news/retrieval/` — `get_news_resolver()`, `get_market_client()`
- `news/news_settings.py` — `get_news_settings()`, `_get_pg_settings()`
- `memory/embedder.py` — `_get_client()` (BGE-M3)
- `memory/langmem_bridge.py` — `get_langmem_bridge()`
- `MemoryService` (`chat_service._get_memory`) — hand-rolled, because it also latches a permanent "disabled" state on schema failure
- `VoiceBridge` — `@st.cache_resource`; Streamlit owns that lifecycle
- `rag_v1/db/pg.py`, `memory/db.py` — `psycopg_pool.ConnectionPool`, which does its own lazy open

### Why
- Streamlit reruns on every interaction — heavy init at import time would stall every rerun
- Some services depend on GPU availability; deferring init lets the app start even if a service is down

### Why the locking is not optional
These accessors are called from `context_injector`'s worker pool. An
unsynchronised initialiser is a real race, and for `WikiRetriever` a lost race
meant two cold torch/CUDA initialisations of jina-clip-v2 on the same physical
GPU at once — the failure mode `sage_kaizen_ai_ingest` hit twice (its CLAUDE.md
§15, 2026-05-28 and 2026-07-18) and had to serialise service startup to stop
machines freezing. Before 2026-08-04, 8 of the 10 singletons here were unlocked.

### Rules
- Use `@lazy_singleton`; it gives you double-checked locking with one lock per accessor
- A `None` return is deliberately **not** cached — the optional-dependency accessors rely on retrying
- An exception propagates and caches nothing, so a later call can retry
- `.reset()` is for tests; production code must not call it
- Log clearly when initialization succeeds or fails
- Always fall back gracefully — caller receives `None` or empty results, never an exception bubble

---

## 5. Graceful Degradation Pattern

### Description
Every optional subsystem (memory, wiki RAG, live search, summarizer, music retrieval) degrades to empty results rather than raising an exception that aborts the turn.

### Why
- Any of these services may be offline (server not running, schema not created)
- A missing wiki or search result is acceptable; a broken turn is not

### Rules Applied In
| Subsystem | Failure mode | Degraded behavior |
|-----------|-------------|-------------------|
| `MemoryService` | Schema not created / DB down | Returns `None`; `chat_service` skips memory injection |
| `WikiRetriever` | jina-clip-v2 service down | Returns empty `WikiSearchResult` |
| `SearchOrchestrator` | SearXNG unreachable | Returns empty `SearchEvidence` |
| `summarizer.py` | FAST brain not started | Returns raw snippets joined by newline |
| `MusicRetriever` | CLAP service down | Returns empty list |

### Rules
- Wrap every optional subsystem call in `try/except`; log the exception at WARNING level
- Return a typed empty object (not `None` where possible) so downstream code needs no null-checks
- Never let a degraded subsystem cause a streaming response to abort mid-turn

---

## 6. Prompt Injection Defense Pattern

### Description
All externally-sourced text (RAG chunks, web snippets, Wikipedia extracts, RSS content) is sanitized before injection into the LLM context.

### Why
- Retrieved content may contain adversarial strings targeting chat template tokens or instruction formats
- Web search results are fully attacker-controlled

### Implementation (`input_guard.py`)
- `sanitize_chunk(text, max_chars)` — strips chat template tokens (`<|im_start|>`, `[INST]`, `<<SYS>>`, `<|eot_id|>`, etc.), fake instruction headers ("ignore previous instructions"), HTML tags
- `check_user_input(text)` — raises `InjectionDetectedError` on structural injection in user input
- Applied to: every RAG chunk, every web snippet, every wiki extract before context injection

### Rules
- Never inject external content into `role: system` without sanitization
- `input_guard` must run BEFORE content is added to the message list
- `InjectionDetectedError` is surfaced to the user ("content filtered") — not silently swallowed

---

## 7. Parallel Context Assembly Pattern

### Description
RAG sources (doc-RAG, wiki, search, music, news) are fetched concurrently in a `ThreadPoolExecutor` before the streaming request begins.

### Why
- Each retrieval step involves I/O (PostgreSQL, HTTP to embed server, HTTP to SearXNG)
- Sequential fetching would add 500–2000 ms per turn
- Concurrent fetching keeps total retrieval latency near the slowest single source

### Implementation (`rag_v1/runtime/context_injector.py`)
- `apply_rag_and_wiki_parallel()` — 5-worker `ThreadPoolExecutor`
- Returns 4-tuple: `(messages, rag_sources, wiki_images, search_evidence)`
- Each worker fails gracefully (see Pattern 5)

### Rules
- Max 5 parallel DB/HTTP workers per turn (avoid overwhelming PostgreSQL)
- Each worker must complete within its own timeout; no global cancel on partial failure
- Search worker only fires when `RouteDecision.needs_search=True`

---

## 8. Observability-First Pattern

### Design Assumption
> If it's not logged, it didn't happen.

### Where Applied
- **Server readiness**: `_wait_for_ready()` in `server_manager.py` checks log for `"server is listening"` or `"slots idle"` — not an implicit timeout
- **Routing**: every `RouteDecision` logs brain, score, reason, template keys selected
- **RAG retrieval**: chunk count, distances, source IDs logged per turn
- **Memory**: episode write success/failure logged; silent failure on schema-missing
- **Review service**: each LangGraph node transition logged with state summary

### Rules
- `sk_logging.py` `get_logger()` is the only logger factory — no bare `print()` in service code
- `RotatingFileHandler(5 MB × 5 backups)` — all logs in `logs/`
- Server startup readiness **must** be detected by log marker, not by a fixed sleep

---

## 9. Agent Transport Pattern (ZeroMQ)

### Description
Host-to-Pi communication over ZeroMQ with versioned message schemas.

### Topology
- `tcp://127.0.0.1:5790` — PULL BIND (transcript in from voice app)
- `tcp://127.0.0.1:5791` — PUB BIND (token stream out to voice app)
- `tcp://127.0.0.1:5792` — PULL BIND (barge-in interrupt from voice app)
- Pi agents: PUSH → host; host: DEALER → Pi agents *(planned)*

### Rules
- Message schema must be versioned and backward-compatible
- Timeouts must be explicit on every socket operation
- Commands from Pi validated before execution — never trust agent input for OS actions
- `VoiceBridge` owns all ZMQ socket lifecycle; no raw socket creation outside this module

---

## 10. Replaceable Module Pattern

### Description
Each subsystem exposes a narrow interface; the implementation behind it can be swapped.

### Current Implementations
| Subsystem | Current | Replaceable with |
|-----------|---------|-----------------|
| STT | faster-whisper distil-large-v3.5 (ONNX, CPU) | any model implementing `transcribe(audio) -> str` |
| TTS | Kokoro-82M ONNX (CPU) | any model implementing `synthesize(text) -> audio_bytes` |
| Vector DB | pgvector (HNSW for rag_chunks/media; ivfflat for partitioned wiki_chunks) | any service returning `(source_id, chunk_id, score, content)` tuples |
| LLM backend | llama-server (GGUF) | any OpenAI-compatible `/v1/chat/completions` endpoint |
| Text embeddings | BGE-M3 FP16 via llama-server | any service returning float vectors via `/v1/embeddings` |
| Image embeddings | jina-clip-v2 (FastAPI) | any 1024-dim CLIP-style model |
| Audio embeddings | CLAP htsat-unfused (FastAPI) | any 512-dim audio embedding model |
| Web search | SearXNG | any service returning `WebResult` list |

### Rules
- Interfaces must remain stable when swapping implementations
- New implementations write to the same DB schema / expose the same HTTP contract
- No caller knows which concrete class is behind the interface (always through a typed facade)

---

## 11. Human-Gated State Machine Pattern

### Description
A LangGraph `StateGraph` workflow where execution pauses at a designated checkpoint
and requires explicit human approval before any file writes occur.

### Used By
`review_service/` — triggered by `is_review_command()` in `router.py`

### Graph Structure
```
scope_collector → subprocess_checks → web_researcher
  → architect_reviewer → flags_sanity → docs_drift
  → synthesizer → human_gate (interrupt())
                      ↓ approved=True      ↓ approved=False
                 output_writer → END       END (no files written)
```

### Why
- Reviews generate ADRs, patches, and reports — high-impact-if-wrong artifacts
- The human gate ensures ARCHITECT synthesis is inspected before any files are touched
- Checkpoint persistence (PostgreSQL `langgraph` schema) survives a Streamlit reload between interrupt and resume

### Rules
- `interrupt()` before all file writes — zero I/O before human approval
- `AsyncPostgresSaver` writes state after each node — run is resumable after process restart
- `ReviewRunner` spawns a daemon thread with `asyncio.new_event_loop()` — isolated from Streamlit's main thread
- Sequential edges only — `parallel: 1` on ARCHITECT; fan-out queues at the HTTP server anyway
- All checkpoint tables live in the `langgraph` schema, never in `public`

### Checkpoint Overhead
Each node step: ~1–5 ms PostgreSQL round-trip × 8 nodes ≈ 8–40 ms per review run.
Review runs are dominated by ARCHITECT inference (minutes per node). Overhead is negligible.

---

## 12. Off-Peak Coordination Pattern

### Description
Long-running background tasks (Wiki ingest, model consolidation) coordinate with active chat sessions to avoid GPU contention.

### Why
- Wiki ingest moved to CUDA2 (RTX 5080 eGPU) on 2026-08-24; it no longer shares a GPU with FAST (now CUDA0)
- Running ingest during a live chat session causes inference timeouts

### Implementation
- `chat_service.record_chat_activity()` — updates a shared timestamp on every turn
- `chat_service.last_chat_activity_ts()` — read by background tasks to check idle time
- Wiki ingest: services A/B (ports 8031/8032) run on CUDA2, so chat and ingest no longer contend for the same GPU
- Background memory consolidation: runs only when no turn has fired in the last N seconds

---

## 13. Shared Transport Base Pattern

### Description
Every client that talks to the same *kind* of service shares one base class
holding the transport concerns — connection pooling, health, retry policy,
cleanup — and subclasses add only the endpoint methods.

### Where Used
- `rag_v1/embed/base_client.py` → `BaseHttpEmbedClient`, subclassed by
  `EmbedClient` (BGE-M3, 8020), `MmEmbedClient` (jina-clip-v2, 8031),
  `ImageEmbedClient` (a thin subclass of `MmEmbedClient`), and
  `AudioEmbedClient` (CLAP, 8040)
- `server_manager._ensure_brain_running()` — the same idea for process
  lifecycle: one body for all four llama-server brains
- `rag_v1/db/pg.py` and `memory/db.py` — both on `psycopg_pool.ConnectionPool`

### Why
Four independently-written HTTP clients had drifted into three connection
strategies and four readings of `/health`:

| | Pooled? | On exhausted retries |
|---|---|---|
| `MmEmbedClient` | yes | original exception |
| `ImageEmbedClient` | **no** | `tenacity.RetryError` |
| `AudioEmbedClient` | **no** | `tenacity.RetryError` |
| `EmbedClient` | yes | raw `raise_for_status` |

Concretely, that cost:

- **A duplicate implementation.** `ImageEmbedClient` and `MmEmbedClient` spoke
  the same protocol to the same service on the same port.
- **Unpooled connections on a hot path.** The two media clients called
  module-level `httpx.post()`, building and discarding a TCP connection per
  call, while driven at volume by `sage_kaizen_ai_ingest`'s media pipeline —
  a project with an unresolved TCP ephemeral-port-exhaustion investigation
  whose notes recorded that `MmEmbedClient` had been checked and was pooled.
  Nobody had checked these two.
- **A retry contract that differed between siblings.** Code catching
  `httpx.HTTPStatusError` worked against one client and silently missed the
  other.

### Rules
- Transport belongs in the base; protocol belongs in the subclass
- One retry policy (`EMBED_RETRY`), applied at exactly one level — decorating
  both the base's request helper and the subclass's method gives 3 × 3 attempts
- `health()` returns the parsed payload or `None`; `{}` means "answered, said
  nothing useful", which is not the same as "did not answer"
- A subclass may tighten readiness (`AudioEmbedClient.ping()` additionally
  requires `loaded`, because CLAP answers 200 while still loading) but must not
  loosen the transport contract
- Every client is a context manager and has `close()`; a pooled client that is
  never closed leaks its connection pool
