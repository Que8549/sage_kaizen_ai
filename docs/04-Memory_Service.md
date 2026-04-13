# Memory_Service.md

## Purpose

This document defines a production-ready **Memory Service** for Sage Kaizen: a fully local, modular, governed long-term memory architecture that helps the system adapt to the user's norms, preferences, workflows, and project context over time without changing model weights.

The design is optimized for the current Sage Kaizen baseline:

- Windows 11 Pro
- Python 3.14.3
- PostgreSQL + pgvector **>= 0.8.2** (see Security Note below)
- psycopg3 (`psycopg[binary]`) — already installed; do NOT add asyncpg
- local llama.cpp / llama-server inference
- dual-brain architecture
  - **Fast Brain**: low-latency routing / extraction / lightweight reflection
  - **Architect Brain**: deep reasoning / memory consolidation / policy promotion
- existing RAG and retrieval pipeline
- existing project invariants around modularity, official docs first, and non-beta / non-deprecated components

This Memory Service is designed to be:
- **fully local**
- **auditable**
- **typed**
- **governed**
- **replaceable**
- **safe against uncontrolled persona drift**

---

## pgvector Version Notes

### pgvector 0.8.0 (October 2024)
- Added **iterative index scan** (`hnsw.iterative_scan`, `ivfflat.iterative_scan`) — prevents silent under-retrieval when WHERE filters are selective. **Required for memory retrieval** because queries always filter by `user_id`, `project_id`, and `is_active`.
- Improved query planner's index selection under filters.

### pgvector 0.8.1 (September 2025)
- Bug-fix release. No schema DDL changes.

### pgvector 0.8.2 (2026)
- **Security fix: CVE-2026-3172** — buffer overflow in parallel HNSW index builds; can leak data or crash the server.
- **Upgrade to 0.8.2+ before running any HNSW index builds in production.**
- No schema DDL changes; all 0.8.x schemas are compatible.

### Iterative scan — required configuration
Enable per-query in retriever code before filtered vector searches:

```sql
SET LOCAL hnsw.iterative_scan = relaxed_order;
SET LOCAL hnsw.max_scan_tuples = 20000;
```

`relaxed_order` returns approximate results ordered by score (fastest). `strict_order` returns exact top-k but scans more tuples. For memory retrieval, `relaxed_order` is correct — we rerank anyway.

---

## Design goals

1. Persist useful user-specific knowledge across sessions.
2. Retrieve only the most relevant history for the current turn.
3. Adapt output style and persona to the user's specific norms.
4. Avoid dumping raw transcript history into every prompt.
5. Separate stable preferences from transient episodes.
6. Add a controlled "self-improving" loop via reflection and consolidation.
7. Preserve explainability, provenance, and reversibility.
8. Integrate cleanly into the current Sage Kaizen router and RAG flow.

---

## Key design decision

**Sage Kaizen should use a native local memory service built on PostgreSQL + pgvector rather than relying on a hosted memory platform.**

Reason:
- aligns with current Sage Kaizen stack
- keeps all user memory local
- avoids cloud lock-in
- supports strict filtering by user / project / scope
- supports hybrid retrieval with vector + lexical search
- makes governance, pruning, and review easier

### Phase 1 shortcut: LangMem + LangGraph

For rapid iteration before the full custom service is built, a production-ready **LangMem bridge** is provided at `memory/langmem_bridge.py`. It uses:

- `langmem` SDK (`create_memory_store_manager`, `ReflectionExecutor`)
- `AsyncPostgresStore` from `langgraph.store.postgres` (already installed via `langgraph-checkpoint-postgres`)
- Local BGE-M3 embed service (port 8020) as the embedding function
- ARCHITECT brain (port 8012) as the memory extraction LLM

**Trade-offs vs. full custom service:**

| Aspect | LangMem bridge | Custom service |
|--------|----------------|----------------|
| Time to implement | Hours | Days–weeks |
| Governance / promotion thresholds | LLM-driven, less controlled | Fully explicit thresholds |
| Four-class memory model | Single `store` table, JSON docs | Typed tables per class |
| Contradiction detection | Not built-in | Explicit in policy.py |
| Audit log | Not built-in | Full audit_log table |
| Hybrid retrieval (FTS + vector) | Vector only | Vector + FTS + RRF |
| Token budget per brain | Manual | Explicit per-brain caps |

**Decision**: Use LangMem bridge for Phase 1 evaluation. Migrate to full custom service in Phase 2 once usage patterns are understood.

---

## Architectural model

The Memory Service uses four memory classes.

### 1. Core Profile Memory (always-on)

Small, curated, structured memory that is injected into every turn.

Examples:
- preferred answer depth
- tone preferences
- formatting preferences
- project invariants
- architecture preferences
- hard prohibitions
- preferred documentation sources
- environment assumptions

This is the closest Sage Kaizen equivalent to "core memory" / "memory blocks".

### 2. Episodic Memory (retrieved)

Semantically searchable event-like memories derived from prior interactions.

Examples:
- user corrected a recommendation
- user rejected a framework
- user preferred one model over another
- user approved a design direction
- user reported a bug and its resolution
- prior architectural decision for this project

These are retrieved on demand.

**Selective write policy** — write an episode only when:
- user corrected or rejected a recommendation
- user stated an explicit preference
- user approved a design decision
- turn involved a code change, architecture decision, or model selection
- turn length > 200 tokens AND estimated importance > 0.4

Do NOT write a new episode for short acknowledgements, greetings, or trivial clarifications.

### 3. Procedural Rule Memory (retrieved + selectively pinned)

Learned operational rules that guide behavior.

Examples:
- browse official docs first for current technical questions
- prefer Windows commands
- avoid beta and deprecated components
- prefer Hugging Face HauhauCS models for Sage Kaizen recommendations
- use production-ready modular design language
- use the fast brain for extraction and the architect brain for consolidation

These may be promoted from repeated episodes or explicit user instructions.

### 4. Reflection / Consolidation Memory

Background summaries and judgments generated from sessions.

Examples:
- stable preference candidates
- contradictions detected
- stale memories identified
- confidence changes
- possible rule promotions
- risk flags for drift or ambiguity

These are not injected directly into prompts unless selected by policy.

---

## Memory scopes

Every memory item must have a scope.

- `user`: applies to the user across projects
- `project`: applies only to Sage Kaizen
- `workspace`: applies to the current repository / working tree
- `session`: applies only to the current conversation or session
- `global_system`: maintained by the system, read-only for normal memory promotion

For Sage Kaizen, the common default is:
- project and workspace scoped for engineering decisions
- user scoped for stable stylistic preferences

---

## Token budget per brain

Memory bundles must be sized for the receiving brain's context window.

| Brain | Max bundle tokens | Reason |
|-------|-------------------|--------|
| FAST (Qwen2.5-Omni-7B) | 600 | 16K total; 1 image ≈ 1,280 tokens; need headroom |
| ARCHITECT (Qwen3.5-27B) | 1,500 | 64K total; deep context is fine |

The bundle builder must enforce these caps by priority: profiles → rules → episodes (drop lowest-scored episodes first).

---

## Governance principles

The system must never behave like an uncontrolled self-editing agent.

### Required controls

Each memory row must support:
- provenance
- confidence
- source type
- created_at
- last_accessed_at
- last_confirmed_at
- decay policy
- expiration policy
- contradiction group
- lock status
- promotion source
- review status

### Critical rules

1. **Nothing becomes permanent just because it was mentioned once.**
2. **Stable profile updates require either explicit user confirmation or repeated consistent evidence.**
3. **Procedural rule promotion must be gated by thresholds.**
4. **Sensitive or ambiguous memories must not auto-promote.**
5. **Every memory must be reversible.**
6. **Memory retrieval must be filtered by user and scope before ranking.**

---

## Production retrieval strategy

For each incoming user turn:

### Step 1 — build memory context request

Inputs:
- user_id
- project_id
- workspace_id
- session_id
- raw user message
- normalized user intent
- route target (fast / architect)
- top-level tags

### Step 2 — load always-on profile

Retrieve:
- active `memory.profiles` rows for the user and project
- locked procedural rules
- user-confirmed norms

Budget target:
- 300 to 600 tokens after formatting (FAST brain limit)

### Step 3 — episodic candidate retrieval

Use hybrid retrieval:
1. metadata filter first (`user_id`, `project_id`, `is_active`)
2. lexical / full-text retrieval (generated tsvector)
3. vector retrieval (HNSW with **iterative scan enabled**)
4. reciprocal rank fusion
5. optional rerank
6. policy trimming

Return:
- top 3 to 6 episodic items

### Step 4 — procedural memory retrieval

Retrieve:
- matching operational rules relevant to the current intent
- workspace-specific coding patterns
- project-specific architecture norms

Return:
- top 2 to 5 rules

### Step 5 — contradiction and freshness filtering

Drop or demote:
- expired items
- contradicted items
- low-confidence items
- overly stale items unless reinforced
- duplicate items

### Step 6 — build memory bundle

Bundle structure:
1. stable profile facts
2. relevant project invariants
3. relevant procedural rules
4. top episodic lessons
5. optional recent corrections

Enforce per-brain token caps (see Token budget section above).

### Step 7 — inject into prompt

Inject as a compact structured memory segment, not raw transcript text.

Recommended prompt format:

```text
[MEMORY PROFILE]
- ...
- ...

[MEMORY RULES]
- ...
- ...

[RELEVANT PRIOR EPISODES]
- ...
- ...
```

---

## Why hybrid retrieval is required

Pure vector search is not enough for this use case.

Sage Kaizen must support:
- strict lexical matches for model names, flags, filenames, commands, ports, and versions
- semantic matches for "preferred architecture style", "tone", "project norms"
- metadata filters for user / project / workspace / confidence / freshness

Therefore, retrieval should combine:
- PostgreSQL full-text search (generated tsvector)
- pgvector similarity (HNSW with iterative scan)
- metadata filters
- rank fusion (RRF)
- optional reranking

**Important**: Always apply metadata filters before vector search. pgvector 0.8.0+ improves planner behavior under filters, and enabling `hnsw.iterative_scan = relaxed_order` prevents silent under-retrieval when the filter is highly selective (e.g., a single user's memories).

**Future optimization**: BGE-M3 natively supports sparse vector retrieval (SPLADE-style), which can replace `pg_trgm` for exact technical term matching (model names, ports, flags). Defer to Phase 3.

---

## Storage model

The design uses five primary tables and one optional audit table, all in the `memory` schema.

### 1. `memory.profiles`

Stores stable user and project profile facts.

### 2. `memory.episodes`

Stores prior interaction events and lessons.

### 3. `memory.rules`

Stores procedural rules and operational norms.

### 4. `memory.reflections`

Stores session-level and batch consolidation outputs.

### 5. `memory.links`

Optional relation table linking memories to each other.

### 6. `memory.audit_log` (recommended)

Tracks writes, updates, merges, promotions, demotions, and deletes.

---

## PostgreSQL schema

Use UUID keys, JSONB metadata, generated tsvector for lexical search, and vector columns for semantic retrieval.

### Extension requirements

```sql
CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector >= 0.8.2 recommended
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- trigram similarity for exact-ish matching
```

See `scripts/memory_schema.sql` for full DDL.

---

## Indexing

### Vector search — HNSW (pgvector 0.8.x)

```sql
CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);
```

- `m = 16`: graph connectivity; 16 is a good balance for 1024-dim memory embeddings
- `ef_construction = 128`: build-time search depth; higher = better recall, slower build
- `vector_cosine_ops`: correct for BGE-M3 L2-normalized 1024-dim embeddings
- Enable iterative scan per-query with `SET LOCAL hnsw.iterative_scan = relaxed_order`

### Lexical search — generated tsvector

```sql
ADD COLUMN search_tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(summary_text,'') || ' ' || coalesce(raw_excerpt,''))
) STORED;
CREATE INDEX ... USING GIN (search_tsv);
```

### Metadata filters

Composite B-tree indexes on `(user_id, project_id, scope, is_active)` for fast pre-filtering before vector search.

---

## Embedding source

**The memory embedder (`memory/embedder.py`) must reuse the existing BGE-M3 embed client.**

- Do not introduce a second embedding client or a second embedding model.
- Wraps `rag_v1/embed/embed_client.py` — calls BGE-M3 at `http://127.0.0.1:8020`.
- 1024-dimensional L2-normalized vectors.
- Use `psycopg3` (sync) on the hot retrieval path; `httpx.AsyncClient` for batch async consolidation.

---

## Connection pooling

The memory service is called on every turn. Without a connection pool, each retrieval opens a new PostgreSQL connection, adding 5–20 ms latency.

**The `MemoryService` must use a shared connection pool.**

- Use psycopg3 `ConnectionPool` (sync) for the hot path.
- Initialize the pool once at `MemoryService` construction.
- Share the pool across all repository calls.

Do **not** use asyncpg. psycopg3 is already installed and used by the review service.

---

## Write-path design

### Path A — explicit profile write

Trigger: user explicitly states a stable preference or rule.

Action: write directly to `memory.profiles`; if operational, also create a locked `memory.rules` row.

### Path B — episodic write (selective)

Trigger: completed turn that passes the selective write policy (see Episodic Memory section).

Action: summarize the interaction into an event; assign type, tags, importance, confidence; embed and store.

### Path C — reflection write

Trigger: end of session, idle window, nightly batch, or explicit maintenance run.

Action: generate consolidated findings; propose profile updates; propose rule promotions; suggest contradictions and pruning; write to `memory.reflections`.

### Path D — rule promotion

Trigger: repeated evidence, user confirmation, or high-confidence reflection outcome.

Action: create or update `memory.rules`; increment promotion count; optionally lock if user-approved.

---

## Promotion policy

### Promote to profile memory when

Any of the following is true:
- explicit user instruction
- same preference observed 3+ times across 2+ sessions
- architect reflection confidence >= 0.90 and no contradiction found

### Promote to procedural rule when

All are true:
- repeated behavioral pattern or correction
- relevance to future agent behavior
- confidence >= 0.85
- not contradicted
- scope can be determined safely

### Never auto-promote when

- memory is sensitive or ambiguous
- preference appears temporary
- evidence is single-shot and weak
- tone appears situational rather than stable
- the user appears uncertain or hypothetical

---

## Decay and pruning policy

### Episodes
- score decays over time
- stronger recency decay for low-importance events
- retain high-importance decisions longer

### Rules
- periodic review every 30–90 days
- demote if contradicted repeatedly
- keep locked rules until manually changed

### Deletion policy

Delete or archive:
- duplicate episodes
- low-confidence stale reflections
- contradicted or superseded rules
- expired session-scoped entries

---

## Scoring model

```text
final_score =
    0.35 * semantic_score
  + 0.20 * lexical_score
  + 0.15 * recency_score
  + 0.10 * importance_score
  + 0.10 * confidence_score
  + 0.10 * scope_match_score
```

Additional penalties for contradiction, staleness, duplication, oversaturation from the same session.

---

## API design

Module path: `memory/` at the project root (following existing conventions: `search/`, `rag_v1/`, `review_service/`).

```text
memory/
  __init__.py
  models.py          # Pydantic DTOs — MemoryBundle, MemoryContextRequest, etc.
  schemas.py         # DB row types (dataclasses matching table columns)
  repository.py      # psycopg3 CRUD — profiles, episodes, rules, reflections
  embedder.py        # wraps rag_v1/embed/embed_client.py (BGE-M3 port 8020)
  retriever.py       # hybrid FTS + HNSW + RRF; iterative scan enabled
  ranker.py          # scoring formula, contradiction filter, duplicate suppression
  bundle_builder.py  # assembles prompt-ready MemoryBundle within token budget
  writer.py          # write paths A–D (explicit profile, episodic, reflection, rule)
  consolidator.py    # reflection job, promotion decision, pruning suggestions
  policy.py          # promotion thresholds, decay rules, selective write policy
  audit.py           # audit_log writes
  service.py         # MemoryService facade
  langmem_bridge.py  # Phase 1: LangMem + LangGraph shortcut
```

### Core service interface

```python
class MemoryService:
    def get_memory_bundle(self, request: MemoryContextRequest) -> MemoryBundle
    def write_explicit_profile(self, user_id: str, key: str, value: str, ...) -> None
    def write_episode(self, user_id: str, summary: str, ...) -> None
    def run_reflection(self, user_id: str, session_id: str) -> ReflectionResult
    def promote_rules(self, user_id: str) -> list[PromotionDecision]
    def prune_memories(self, user_id: str) -> int
    def explain_bundle(self, bundle: MemoryBundle) -> str
```

---

## Integration points in Sage Kaizen

### Router integration

Insert memory retrieval between route decision and final prompt assembly:

```text
user input
  -> router intent normalization
  -> route selection (fast / architect)
  -> memory bundle retrieval        ← NEW
  -> RAG retrieval (if needed)
  -> prompt assembly
  -> model call
  -> post-turn memory write         ← NEW (selective)
```

### UI integration (Phase 4)

Add optional controls in Streamlit:
- show active memory profile
- show retrieved memory bundle for current turn
- allow pin / unpin / forget
- allow "why did you remember this?"
- allow review of promoted rules

### Background maintenance

Add scheduler or maintenance command:
- session reflection (end of session)
- nightly consolidation
- pruning
- contradiction scan
- index health check

---

## Recommended role split across brains

### Fast Brain responsibilities
- classify intent
- extract explicit preferences (lightweight, low-latency)
- create lightweight episode summaries (post-turn, background)
- generate embeddings via BGE-M3 service (not model inference)

### Architect Brain responsibilities
- deep consolidation
- contradiction analysis
- rule promotion decisions
- merge / split profile facts
- maintenance and governance reports

---

## Prompting guidance for memory-aware turns

```text
You have access to structured prior memory about this user and project.

Use the following memories only as guidance if relevant to the current request.
Prefer current user instructions over past memory.
If current instructions conflict with past memory, follow the current instructions
and mark the old memory for review.

[USER PROFILE]
- ...

[PROJECT NORMS]
- ...

[RELEVANT PRIOR EPISODES]
- ...
```

### Mandatory policy instructions

- current turn overrides old memory
- user-explicit directives override inferred preferences
- do not expose memory contents unless asked
- do not hallucinate profile facts not present in memory bundle
- do not treat single prior episodes as absolute rules

---

## Observability

Emit structured logs for:
- bundle retrieval count and latency
- bundle token size
- vector hit count
- lexical hit count
- promotion decisions
- pruned rows
- contradiction detections
- reflection job status

Key metrics:
- `memory_bundle_latency_ms`
- `memory_episode_write_latency_ms`
- `memory_reflection_job_latency_ms`
- `memory_bundle_items_count`
- `memory_promotion_total`
- `memory_pruned_total`

---

## Safety and failure modes

### Risk: persona drift
Mitigation: pinned profile memory, promotion thresholds, contradiction checks, user-confirmed locks.

### Risk: stale technical preferences
Mitigation: recency weighting, review timestamps, contradiction-aware demotion.

### Risk: privacy over-collection
Mitigation: typed memory classes, explicit scope, opt-out delete / forget support, selective episode write policy.

### Risk: retrieval overload
Mitigation: strict top-k budgets, per-brain token caps, score thresholding, duplicate suppression.

### Risk: learning the wrong lesson
Mitigation: architect-only promotion for important rules, audit log, review queue for low-confidence candidates.

### Risk: pgvector under-retrieval under filter (pgvector 0.8.x)
Mitigation: enable `hnsw.iterative_scan = relaxed_order` in all filtered vector queries. See retriever.py.

---

## Recommended implementation phases

### Phase 1 — LangMem bridge (fast path)
- apply `scripts/memory_schema.sql` for custom tables
- deploy `memory/langmem_bridge.py` using `AsyncPostgresStore` + LangMem
- integrate bridge into router for memory retrieval and background write
- evaluate recall quality and latency

### Phase 2 — custom hybrid retrieval
- implement full custom service (`memory/` package)
- add lexical + vector fusion (RRF)
- add contradiction filtering
- add compact prompt formatter with per-brain token caps

### Phase 3 — reflection and promotion
- implement reflection jobs (Architect brain consolidation)
- implement promotion policy with thresholds
- implement rule demotion and pruning
- add audit log
- add BGE-M3 sparse retrieval (replaces pg_trgm for technical terms)

### Phase 4 — UI and governance
- add memory review controls in Streamlit
- add explainability panel
- add manual forget / pin / lock
- add maintenance dashboard

---

## File and code conventions for this project

- Python 3.14.3
- Windows-safe paths and commands
- modular components, no beta or deprecated dependencies
- `memory/` at project root (not `sage/memory/`)
- SQL files at `scripts/memory_schema.sql` (not `sql/`)
- psycopg3 only (no asyncpg)
- reuse `rag_v1/embed/embed_client.py` for BGE-M3 embeddings
- pgvector >= 0.8.2 (CVE-2026-3172 security fix)

---

## Minimum acceptance criteria

A production-acceptable v1 must:

1. persist explicit user preferences
2. persist episodic memories with embeddings (selective write policy enforced)
3. retrieve hybrid top-k memory per turn
4. inject memory bundle before model inference (within per-brain token caps)
5. perform post-turn episodic write (selective)
6. support nightly reflection
7. support rule promotion with thresholds
8. support forget / disable / lock
9. support audit logging
10. expose tests and metrics

---

## Suggested tests

### Unit tests
- profile upsert
- episodic insert
- lexical retrieval
- vector retrieval
- fusion ranking
- contradiction filtering
- rule promotion thresholds
- decay scoring
- prompt bundle formatting
- token budget enforcement per brain

### Integration tests
- router + memory bundle path
- post-turn selective write
- reflection job path
- forget flow
- locked rule protection
- iterative scan fallback behavior

---

## Recommended next deliverables

1. `scripts/memory_schema.sql`
2. `memory/models.py`
3. `memory/schemas.py`
4. `memory/repository.py`
5. `memory/embedder.py`
6. `memory/retriever.py`
7. `memory/ranker.py`
8. `memory/bundle_builder.py`
9. `memory/writer.py`
10. `memory/policy.py`
11. `memory/audit.py`
12. `memory/consolidator.py`
13. `memory/service.py`
14. `memory/langmem_bridge.py` ← Phase 1 LangMem shortcut
15. router integration patch
16. tests
17. Streamlit memory review panel (Phase 4)

---

## Notes for Claude Code

When implementing:
- preserve existing RAG behavior
- integrate memory before final prompt assembly
- do not break current brains.yaml or server orchestration invariants
- keep memory service independent from model runtime so it remains reusable
- use deterministic repository methods and typed DTOs
- add migration-safe SQL and idempotent schema creation
- include logging and tests from the first implementation pass
- enable `hnsw.iterative_scan = relaxed_order` in all filtered vector queries
- use `memory/` at project root, SQL at `scripts/`
- use psycopg3 connection pool — not asyncpg

---

## References

- Anthropic Claude Code memory model: `CLAUDE.md` + auto memory
- LangGraph long-term memory model: semantic / episodic / procedural
- LangMem SDK: `langmem` — `create_memory_store_manager`, `ReflectionExecutor`
- Letta sleep-time agents for background consolidation pattern
- pgvector hybrid retrieval: [Building Hybrid Search for RAG](https://dev.to/lpossamai/building-hybrid-search-for-rag-combining-pgvector-and-full-text-search-with-reciprocal-rank-fusion-6nk)
- pgvector 0.8.0 release notes: [pgvector 0.8.0 Released](https://www.postgresql.org/about/news/pgvector-080-released-2952/)
- pgvector 0.8.2 security: [CVE-2026-3172 fix](https://www.postgresql.org/about/news/pgvector-082-released-3245/)
- BGE-M3 multi-functionality: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
