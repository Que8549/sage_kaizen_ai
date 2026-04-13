# Claude_Code_Memory_Implementation_Prompt.md

Use the following prompt in Claude Code inside the Sage Kaizen repository.

---

You are implementing a production-ready local **Memory Service** for the Sage Kaizen project.

Read and follow `/docs/04-Memory_Service.md` exactly.

Your task is to implement the memory system in the existing Sage Kaizen codebase using the project's current architecture and invariants.

## Non-negotiable project requirements

1. Target **Python 3.14.3**
2. Keep everything **fully local**
3. Use **PostgreSQL + pgvector >= 0.8.2** (CVE-2026-3172 security fix)
4. Do **not** introduce beta or deprecated dependencies
5. Do **not** add asyncpg — use **psycopg3** (`psycopg[binary]`) which is already installed
6. Preserve Sage Kaizen modularity
7. Preserve the current dual-brain design
8. Do not break current router, RAG, or server orchestration behavior
9. Use Windows-safe commands and paths
10. Prefer typed, maintainable, production-grade code
11. Add tests and logging with the first implementation pass

## Existing project context you must respect

- Sage Kaizen uses a dual-brain architecture:
  - Fast Brain (port 8011) for low-latency tasks
  - Architect Brain (port 8012) for deep reasoning and consolidation
- PostgreSQL + pgvector is already part of the project stack
- Existing RAG flow must continue working
- The new Memory Service is at `memory/` at the project root (NOT `sage/memory/`)
- SQL schema files go in `scripts/` (NOT `sql/`) — follow the `scripts/setup_langgraph_schema.sql` pattern
- Existing project conventions prefer modular replacement-friendly components
- The router should retrieve memory before final prompt assembly
- The memory embedder MUST reuse `rag_v1/embed/embed_client.py` (BGE-M3 port 8020, 1024-dim)
  Do NOT introduce a second embedding client or model
- pgvector 0.8.x: enable `hnsw.iterative_scan = relaxed_order` in all filtered vector queries
  to prevent silent under-retrieval when user_id filters are selective
- Use psycopg3 ConnectionPool for the sync hot path — not asyncpg, not raw connections

## Per-brain token budget (CRITICAL)

Memory bundles MUST be sized for the receiving brain:

| Brain | Max bundle tokens |
|-------|-------------------|
| FAST (Qwen2.5-Omni-7B, 16K ctx) | 600 tokens |
| ARCHITECT (Qwen3.5-27B, 64K ctx) | 1,500 tokens |

The bundle_builder.py must enforce these caps. Drop lowest-scored episodes first.

## Selective episode write policy

Do NOT write an episode for every turn. Write only when:
- user corrected or rejected a recommendation (`was_user_correction=True`)
- user stated an explicit preference (`was_explicit_preference=True`)
- event_type is in the ALWAYS_WRITE_EVENTS set (see policy.py)
- turn length > 200 tokens AND estimated importance > 0.4

See `memory/policy.py:should_write_episode()`.

## Files already implemented (do not recreate)

The following files were created by a prior implementation pass:

- `scripts/memory_schema.sql` — full DDL, all tables in `memory` schema
- `memory/__init__.py`
- `memory/models.py` — Pydantic DTOs
- `memory/schemas.py` — DB row dataclasses
- `memory/embedder.py` — wraps rag_v1/embed/embed_client.py
- `memory/repository.py` — psycopg3 CRUD with iterative scan
- `memory/ranker.py` — RRF fusion + multi-signal scoring
- `memory/retriever.py` — hybrid FTS + HNSW retrieval
- `memory/bundle_builder.py` — token-budgeted bundle assembly + prompt format
- `memory/writer.py` — write paths A–D
- `memory/policy.py` — promotion thresholds, selective write policy, decay
- `memory/audit.py` — audit log writes
- `memory/consolidator.py` — Architect-brain reflection + rule promotion
- `memory/service.py` — MemoryService facade
- `memory/langmem_bridge.py` — Phase 1 LangMem + LangGraph shortcut

## Remaining deliverables

### 1. Apply SQL schema

```bash
psql -U <user> -d <db> -f scripts/memory_schema.sql
```

Verify with:
```sql
\dn memory
\dt memory.*
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','pg_trgm');
```

pgvector should show >= 0.8.2.

### 2. Router integration

Patch `router.py` to retrieve memory between route selection and final prompt assembly.

Add the MemoryService singleton:
```python
from memory.service import MemoryService
from memory.models import MemoryContextRequest
from memory.bundle_builder import format_bundle_prompt

_MEMORY: Optional[MemoryService] = None

def _get_memory() -> MemoryService:
    global _MEMORY
    if _MEMORY is None:
        _MEMORY = MemoryService()
    return _MEMORY
```

### 3. Chat service integration

In `chat_service.py`, after route is decided and before prompt assembly:

```python
from memory.models import MemoryContextRequest
from memory.bundle_builder import format_bundle_prompt

# In prepare_messages() or equivalent:
mem_req = MemoryContextRequest(
    user_id="alquin",
    project_id="sage_kaizen",
    session_id=session_id,
    query_text=user_text,
    intent_label=intent_label,
    route_target=decision.brain.lower(),
)
bundle = _get_memory().get_memory_bundle(mem_req)
memory_segment = format_bundle_prompt(bundle)

# Prepend memory_segment to the system prompt (or inject as a user message)
# ONLY if non-empty — don't add empty blocks
```

### 4. Post-turn episode write

After the model response is complete, write an episode asynchronously:

```python
from memory.models import EpisodeWriteRequest
from memory.writer import write_episode
import threading

def _write_episode_bg(user_text: str, assistant_text: str, session_id: str) -> None:
    req = EpisodeWriteRequest(
        user_id="alquin",
        session_id=session_id,
        event_type="general",   # classifier can set this; start with "general"
        summary_text=user_text[:300],
        raw_excerpt=assistant_text[:200],
        importance=0.5,
    )
    write_episode(req)

# Fire-and-forget from the turn handler:
threading.Thread(target=_write_episode_bg, args=(user_text, assistant_text, session_id), daemon=True).start()
```

### 5. Phase 1 LangMem bridge (optional evaluation path)

To evaluate LangMem as a Phase 1 shortcut alongside the custom service:

```python
from memory.langmem_bridge import get_langmem_bridge

bridge = get_langmem_bridge()
bridge.start()   # once at app startup

# Per-turn (non-blocking, background extraction):
bridge.submit_turn(messages=conversation_messages, user_id="alquin")

# Before prompt assembly (synchronous retrieval):
segment = bridge.get_memory_prompt(query=user_text, user_id="alquin")
```

First install langmem:
```bash
pip install "langmem>=0.0.30"
```

### 6. Maintenance runner

Add a CLI entry point for reflection and pruning:

```bash
python -m memory.consolidator  # or a dedicated scripts/run_memory_reflection.py
```

The consolidator.run_reflection() can be called:
- End of session (mode="lightweight", lookback_hours=2)
- Nightly batch (mode="deep", lookback_hours=24)

### 7. Tests

Create:
- `tests/test_memory_repository.py` — profile upsert, episode insert, rule insert
- `tests/test_memory_retriever.py` — lexical retrieval, vector retrieval, RRF fusion
- `tests/test_memory_bundle_builder.py` — token cap enforcement per brain
- `tests/test_memory_policy.py` — selective write policy, promotion thresholds, decay

Test pattern (use repo's existing psycopg3 + pytest style):

```python
import pytest
from memory.policy import should_write_episode

def test_should_not_write_greeting():
    assert should_write_episode("greeting", "hi", "", False, False, 0.1) is False

def test_should_write_correction():
    assert should_write_episode("general", "...", "...", True, False, 0.5) is True

def test_bundle_respects_fast_brain_cap():
    from memory.bundle_builder import build_bundle
    # build a large set of items and verify estimated_tokens <= 600
    ...
```

## Implementation strategy

Follow these steps in order.

### Step 1 — inspect the repository

Before making changes:
- Read `docs/04-Memory_Service.md` (already updated)
- Inspect `router.py` and `chat_service.py` for the exact integration points
- Read `memory/service.py` and `memory/bundle_builder.py` to understand the interface
- Run `psql -f scripts/memory_schema.sql` to apply the schema

### Step 2 — apply SQL schema

```bash
psql -U <user> -d <db> -f scripts/memory_schema.sql
```

### Step 3 — router + chat_service integration

Patch `chat_service.py`:
- Add `_get_memory()` singleton
- Add memory bundle retrieval after route decision
- Inject `format_bundle_prompt(bundle)` into the system prompt (prepend, non-empty only)
- Add post-turn episode write (background thread)

Preserve ALL existing routing logic, RAG retrieval, wiki retrieval, and logging.

### Step 4 — install langmem (for Phase 1 evaluation)

```bash
pip install "langmem>=0.0.30"
```

Verify `memory/langmem_bridge.py` loads without error:
```bash
python -c "from memory.langmem_bridge import LangMemBridge; print('OK')"
```

### Step 5 — add tests

Create tests/ files listed above. Use pytest.

### Step 6 — smoke test

Start the app:
```bash
streamlit run ui_streamlit_server.py
```

Send a test turn and verify:
1. Memory bundle latency appears in logs (`sage_kaizen.memory.service`)
2. Episode write appears in logs (`sage_kaizen.memory.writer`)
3. Bundle segment appears in prompt logs if any profile rows exist

## pgvector 0.8.x compatibility notes

The schema and retriever are written for pgvector >= 0.8.0:

1. **Iterative scan** (`hnsw.iterative_scan = relaxed_order`) — applied via
   `SET LOCAL` inside each vector query transaction in `repository.py`.
   This prevents under-retrieval when user filters are selective.

2. **HNSW parameters** — `m=16, ef_construction=128` are good defaults for
   1024-dim BGE-M3 embeddings.  Tune ef_construction upward for better recall
   if build time allows.

3. **CVE-2026-3172** — upgrade to pgvector 0.8.2+ before running parallel
   HNSW index builds.  No schema changes needed; just a binary upgrade.

4. **No schema changes between 0.8.0, 0.8.1, 0.8.2** — the DDL in
   `scripts/memory_schema.sql` is compatible with all 0.8.x versions.

## Coding constraints

- Favor clear code over clever code
- Use comments sparingly but meaningfully
- Keep modules small and focused
- Add logging around retrieval latency and bundle sizes
- Avoid giant god classes
- Avoid tight coupling to UI or model runtime internals
- Preserve replaceability of the memory backend and embedding adapter
- psycopg3 only — no asyncpg

## Acceptance criteria

You are done only when all of the following are true:

1. `scripts/memory_schema.sql` applied successfully
2. `memory/` package imports without error
3. Router / chat_service retrieves memory before prompt assembly
4. Post-turn episodic write is implemented (selective policy enforced)
5. Reflection / consolidation entry point exists and runs
6. Phase 1 LangMem bridge (`memory/langmem_bridge.py`) installs and initialises
7. Tests exist and pass
8. Existing RAG, wiki, and search behavior is preserved
9. Logs emit `memory_bundle_latency_ms` and `memory_episode_write_latency_ms`
10. Type annotations are Pylance-clean

## Final output format

At the end of the work, provide:

1. A concise summary of what you changed
2. A file-by-file list of created and modified files
3. Any assumptions you made
4. Any follow-up work recommended for Phase 2 (hybrid scoring tuning, contradiction detection, UI panel)
5. Exact commands to:
   - apply the SQL schema
   - install langmem (Phase 1)
   - run tests
   - run the app

## Important instruction

Do not stop after planning. Implement the files and patches directly.
If you encounter ambiguity, make the safest production-ready choice consistent
with `docs/04-Memory_Service.md` and the existing repository conventions.
Provide a summary of the choice you made and why.
