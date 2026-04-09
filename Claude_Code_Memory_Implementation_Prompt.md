# Claude_Code_Memory_Implementation_Prompt.md

Use the following prompt in Claude Code inside the Sage Kaizen repository.

---

You are implementing a production-ready local **Memory Service** for the Sage Kaizen project.

Read and follow `/docs/04-Memory_Service.md` exactly.

Your task is to implement the memory system in the existing Sage Kaizen codebase using the project’s current architecture and invariants.

## Non-negotiable project requirements

1. Target **Python 3.14.3**
2. Keep everything **fully local**
3. Use **PostgreSQL + pgvector**
4. Do **not** introduce beta or deprecated dependencies
5. Preserve Sage Kaizen modularity
6. Preserve the current dual-brain design
7. Do not break current router, RAG, or server orchestration behavior
8. Use Windows-safe commands and paths
9. Prefer typed, maintainable, production-grade code
10. Add tests and logging with the first implementation pass

## Existing project context you must respect

- Sage Kaizen uses a dual-brain architecture:
  - Fast Brain for low-latency tasks
  - Architect Brain for deep reasoning and consolidation
- PostgreSQL + pgvector is already part of the project stack
- Existing RAG flow must continue working
- The new Memory Service must be a separate reusable module, not tangled into model-serving code
- Existing project conventions prefer modular replacement-friendly components
- The router should retrieve memory before final prompt assembly

## Deliverables you must create

Create or update the following:

1. `docs/04-Memory_Service.md`
   - If `04-Memory_Service.md` already exists at repo root, also copy or move it into `docs/` only if that fits the repo structure cleanly.
   - Do not silently delete the original.

2. SQL schema / migration file:
   - `sql/memory_schema.sql`

3. Python package:
   - `sage/memory/__init__.py`
   - `sage/memory/models.py`
   - `sage/memory/schemas.py`
   - `sage/memory/repository.py`
   - `sage/memory/embedder.py`
   - `sage/memory/retriever.py`
   - `sage/memory/ranker.py`
   - `sage/memory/bundle_builder.py`
   - `sage/memory/writer.py`
   - `sage/memory/consolidator.py`
   - `sage/memory/policy.py`
   - `sage/memory/audit.py`
   - `sage/memory/service.py`

4. Integration updates
   - patch the router so memory retrieval happens after route selection and before final prompt assembly
   - patch post-turn flow so episodic memory is written after each completed turn
   - add a maintenance entry point for reflection / promotion / pruning

5. Tests
   - `tests/test_memory_repository.py`
   - `tests/test_memory_retriever.py`
   - `tests/test_memory_bundle_builder.py`
   - `tests/test_memory_policy.py`
   - add integration tests if the repo already has a compatible test structure

6. Optional but preferred
   - a small Streamlit panel or debug view to inspect the active memory bundle for the current turn
   - a manual forget / pin / lock mechanism if there is already a suitable UI pattern

## Implementation strategy

Follow these steps in order.

### Step 1 — inspect the repository

Before making changes:
- inspect the current router
- inspect the current RAG integration
- inspect existing database helpers
- inspect project configuration patterns
- inspect existing logging approach
- inspect current tests
- inspect any existing prompt assembly logic
- inspect whether a `docs/` and `sql/` folder already exists

Then write a short implementation plan in the terminal output explaining:
- which files you will create
- which files you will patch
- how you will avoid breaking current behavior

### Step 2 — implement the SQL schema

Create `sql/memory_schema.sql` with:
- `CREATE EXTENSION IF NOT EXISTS vector;`
- `CREATE EXTENSION IF NOT EXISTS pg_trgm;`
- tables:
  - `memory_profiles`
  - `memory_episodes`
  - `memory_rules`
  - `memory_reflections`
  - `memory_links`
  - `memory_audit_log`
- indexes for:
  - vector retrieval
  - metadata filtering
  - full-text search
- use idempotent DDL where practical

Also add a brief header comment describing:
- purpose
- required PostgreSQL extensions
- safe re-run expectations

### Step 3 — implement typed schemas and domain models

Implement Pydantic DTOs and typed repository-facing models for:
- memory context request
- profile item
- episode item
- rule item
- reflection result
- bundle
- promotion decision

Keep the code Pylance-friendly and explicit.

### Step 4 — implement repository layer

Create a repository that:
- uses the project’s existing PostgreSQL access pattern if one exists
- otherwise uses a clean local abstraction that can be swapped later
- supports:
  - upsert profile
  - insert episode
  - insert reflection
  - insert / update rule
  - fetch active profiles
  - fetch rule candidates
  - lexical episode search
  - vector episode search
  - audit writes
  - prune operations

Make SQL readable and well logged.

### Step 5 — implement retrieval and ranking

Implement:
- metadata filtering first
- full-text retrieval
- vector retrieval
- reciprocal rank fusion or similarly simple deterministic fusion
- contradiction filtering
- duplicate suppression
- top-k trimming
- token-budget-aware bundle creation

If there is already an embedding service helper, reuse it.
If not, create a small embedding adapter interface instead of hardwiring one provider.

### Step 6 — implement write paths

Implement:
- explicit profile write path
- episodic write path
- reflection write path
- procedural rule promotion path

Rules:
- explicit user preferences may write directly to profile memory
- inferred preferences must go through thresholds
- procedural rules require promotion gating
- sensitive or ambiguous memory must not auto-promote

### Step 7 — implement consolidation

Create a consolidator that can:
- read recent episodes for a session or time window
- generate profile candidates
- generate rule candidates
- detect contradictions
- generate pruning suggestions
- write a reflection record

Make the consolidator usable in two modes:
- lightweight mode for fast-brain-assisted extraction
- deep mode for architect-brain review

Do not make model selection hardcoded if the project already has routing helpers.

### Step 8 — integrate into router

Patch the router so the flow becomes:

- normalize user input / intent
- decide brain route
- request memory bundle
- request RAG bundle if applicable
- assemble final prompt
- run model
- write episodic memory after response
- optionally queue reflection

Preserve current routing logic and logging.

### Step 9 — add tests

Add tests for:
- profile upsert
- episode insert
- lexical retrieval
- vector retrieval path abstraction
- bundle assembly
- rule promotion thresholds
- contradiction filtering
- current-turn-overrides-history logic
- stale memory pruning rules

Use the repo’s existing test framework and style if available.

### Step 10 — write developer documentation

Update or create a short developer section that explains:
- how to apply the SQL schema
- required env vars
- how memory retrieval works
- how reflection runs
- how to disable memory if needed
- how to inspect bundle logs

## Coding constraints

- Favor clear code over clever code
- Use comments sparingly but meaningfully
- Keep modules small and focused
- Add logging around retrieval latency and bundle sizes
- Avoid giant god classes
- Avoid tight coupling to UI or model runtime internals
- Preserve replaceability of the memory backend and embedding adapter

## Acceptance criteria

You are done only when all of the following are true:

1. The schema file exists
2. The memory package exists
3. The router is patched to retrieve memory before final prompt assembly
4. Post-turn episodic write exists
5. Reflection / consolidation entry point exists
6. Tests exist and are runnable
7. Existing behavior is preserved
8. Logging exists
9. The code is type-friendly
10. A concise implementation summary is produced at the end

## Final output format

At the end of the work, provide:

1. A concise summary of what you changed
2. A file-by-file list of created and modified files
3. Any assumptions you made
4. Any follow-up work recommended for phase 2
5. Exact commands to:
   - apply the SQL schema
   - run tests
   - run the app or service paths touched by this change

## Important instruction

Do not stop after planning. Implement the files and patches directly.
If you encounter ambiguity, make the safest production-ready choice consistent with `04-Memory_Service.md` and the existing repository conventions. Provide a summary of the choice you made and why.
