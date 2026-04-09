# Memory_Service.md

## Purpose

This document defines a production-ready **Memory Service** for Sage Kaizen: a fully local, modular, governed long-term memory architecture that helps the system adapt to the user's norms, preferences, workflows, and project context over time without changing model weights.

The design is optimized for the current Sage Kaizen baseline:

- Windows 11 Pro
- Python 3.14.3
- PostgreSQL + pgvector
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

## Design goals

1. Persist useful user-specific knowledge across sessions.
2. Retrieve only the most relevant history for the current turn.
3. Adapt output style and persona to the user's specific norms.
4. Avoid dumping raw transcript history into every prompt.
5. Separate stable preferences from transient episodes.
6. Add a controlled “self-improving” loop via reflection and consolidation.
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
- active `memory_profiles` rows for the user and project
- locked procedural rules
- user-confirmed norms

Budget target:
- 300 to 1200 tokens after formatting

### Step 3 — episodic candidate retrieval

Use hybrid retrieval:
1. metadata filter first
2. lexical / full-text retrieval
3. vector retrieval
4. reciprocal rank fusion
5. optional rerank
6. policy trimming

Return:
- top 3 to 8 episodic items

### Step 4 — procedural memory retrieval

Retrieve:
- matching operational rules relevant to the current intent
- workspace-specific coding patterns
- project-specific architecture norms

Return:
- top 2 to 6 rules

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
- PostgreSQL full-text search
- pgvector similarity
- metadata filters
- rank fusion
- optional reranking

---

## Storage model

The design uses five primary tables and one optional audit table.

### 1. `memory_profiles`

Stores stable user and project profile facts.

Recommended contents:
- style preferences
- environment assumptions
- trusted sources
- persistent norms
- explicit preferences
- pinned project invariants

### 2. `memory_episodes`

Stores prior interaction events and lessons.

Recommended contents:
- short text summary
- event type
- normalized intent
- tags
- embedding
- importance
- confidence
- sentiment or correction markers

### 3. `memory_rules`

Stores procedural rules and operational norms.

Recommended contents:
- rule text
- rule kind
- scope
- promotion source
- confidence
- locked / review flags

### 4. `memory_reflections`

Stores session-level and batch consolidation outputs.

Recommended contents:
- reflection summary
- extracted preference candidates
- contradiction findings
- promotion candidates
- pruning suggestions

### 5. `memory_links`

Optional relation table linking memories to each other.

Recommended contents:
- parent memory id
- child memory id
- relation type
- strength

### 6. `memory_audit_log` (optional but recommended)

Tracks writes, updates, merges, promotions, demotions, and deletes.

---

## PostgreSQL schema recommendation

Use UUID keys, JSONB metadata, generated tsvector for lexical search, and vector columns for semantic retrieval.

### Extension requirements

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

### Example DDL

```sql
CREATE TABLE IF NOT EXISTS memory_profiles (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT,
    workspace_id TEXT,
    scope TEXT NOT NULL CHECK (scope IN ('user','project','workspace','global_system')),
    profile_type TEXT NOT NULL,
    key TEXT NOT NULL,
    value_text TEXT NOT NULL,
    value_json JSONB,
    confidence REAL NOT NULL DEFAULT 1.0,
    source_type TEXT NOT NULL,
    is_pinned BOOLEAN NOT NULL DEFAULT TRUE,
    is_locked BOOLEAN NOT NULL DEFAULT FALSE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_confirmed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (user_id, COALESCE(project_id, ''), COALESCE(workspace_id, ''), scope, profile_type, key)
);

CREATE TABLE IF NOT EXISTS memory_episodes (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT,
    workspace_id TEXT,
    session_id TEXT,
    scope TEXT NOT NULL CHECK (scope IN ('user','project','workspace','session')),
    event_type TEXT NOT NULL,
    intent_label TEXT,
    summary_text TEXT NOT NULL,
    raw_excerpt TEXT,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    importance REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.6,
    sentiment REAL,
    was_user_correction BOOLEAN NOT NULL DEFAULT FALSE,
    was_explicit_preference BOOLEAN NOT NULL DEFAULT FALSE,
    contradiction_group TEXT,
    embedding vector(1024),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ,
    last_retrieved_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS memory_rules (
    id UUID PRIMARY KEY,
    user_id TEXT,
    project_id TEXT,
    workspace_id TEXT,
    scope TEXT NOT NULL CHECK (scope IN ('user','project','workspace','global_system')),
    rule_kind TEXT NOT NULL,
    rule_text TEXT NOT NULL,
    rationale TEXT,
    confidence REAL NOT NULL DEFAULT 0.7,
    promotion_count INT NOT NULL DEFAULT 0,
    source_type TEXT NOT NULL,
    source_memory_id UUID,
    is_locked BOOLEAN NOT NULL DEFAULT FALSE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    review_status TEXT NOT NULL DEFAULT 'proposed',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS memory_reflections (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT,
    workspace_id TEXT,
    session_id TEXT,
    reflection_type TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    extracted_profile_candidates JSONB NOT NULL DEFAULT '[]'::jsonb,
    extracted_rule_candidates JSONB NOT NULL DEFAULT '[]'::jsonb,
    contradictions JSONB NOT NULL DEFAULT '[]'::jsonb,
    pruning_suggestions JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence REAL NOT NULL DEFAULT 0.7,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS memory_links (
    id UUID PRIMARY KEY,
    from_memory_id UUID NOT NULL,
    to_memory_id UUID NOT NULL,
    relation_type TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS memory_audit_log (
    id UUID PRIMARY KEY,
    memory_table TEXT NOT NULL,
    memory_id UUID NOT NULL,
    action_type TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    actor_id TEXT,
    old_value JSONB,
    new_value JSONB,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## Indexing recommendation

### Lexical search

Add generated `tsvector` columns or materialized search columns on text-heavy tables.

Example:

```sql
ALTER TABLE memory_episodes
ADD COLUMN IF NOT EXISTS search_tsv tsvector
GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(summary_text, '') || ' ' || coalesce(raw_excerpt, ''))
) STORED;

CREATE INDEX IF NOT EXISTS idx_memory_episodes_search_tsv
ON memory_episodes
USING GIN (search_tsv);
```

Recommended:
- similar generated columns for `memory_rules.rule_text`
- optional trigram indexes for exact-ish string matching on command and model names

### Vector search

Use HNSW on `embedding`.

```sql
CREATE INDEX IF NOT EXISTS idx_memory_episodes_embedding_hnsw
ON memory_episodes
USING hnsw (embedding vector_cosine_ops);
```

### Metadata filtering

```sql
CREATE INDEX IF NOT EXISTS idx_memory_episodes_scope_user_project
ON memory_episodes (user_id, project_id, workspace_id, scope);

CREATE INDEX IF NOT EXISTS idx_memory_rules_scope_user_project
ON memory_rules (user_id, project_id, workspace_id, scope, is_active);

CREATE INDEX IF NOT EXISTS idx_memory_profiles_scope_user_project
ON memory_profiles (user_id, project_id, workspace_id, scope, is_active);
```

---

## Write-path design

The system uses four write paths.

### Path A — explicit profile write

Trigger:
- user explicitly states a stable preference or rule

Examples:
- "Use official docs first"
- "Avoid beta components"
- "Prefer Windows commands"
- "Keep Sage Kaizen modular"

Action:
- write directly to `memory_profiles`
- if operational, also consider a locked `memory_rules` row

### Path B — episodic write

Trigger:
- every completed turn or selected turns

Action:
- summarize the interaction into an event
- assign type, tags, importance, and confidence
- embed and store in `memory_episodes`

### Path C — reflection write

Trigger:
- end of session
- idle window
- nightly batch
- explicit maintenance run

Action:
- generate consolidated findings
- propose profile updates
- propose rule promotions
- suggest contradictions and pruning
- write to `memory_reflections`

### Path D — rule promotion

Trigger:
- repeated evidence
- user confirmation
- high-confidence reflection outcome

Action:
- create or update `memory_rules`
- increment promotion count
- optionally lock if user-approved

---

## Promotion policy

Use thresholds to prevent noisy auto-learning.

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

Memory must not grow forever.

### Suggested defaults

#### Profiles
- no decay by default
- require explicit confirmation or contradiction to change

#### Episodes
- score decays over time
- stronger recency decay for low-importance events
- retain high-importance decisions longer

#### Rules
- periodic review every 30–90 days
- demote if contradicted repeatedly
- keep locked rules until manually changed

#### Reflections
- retain for audit and maintenance
- low retrieval priority

### Deletion policy

Delete or archive:
- duplicate episodes
- low-confidence stale reflections
- contradicted or superseded rules
- expired session-scoped entries

---

## Scoring model

Final retrieval score should combine multiple signals.

### Example formula

```text
final_score =
    0.35 * semantic_score
  + 0.20 * lexical_score
  + 0.15 * recency_score
  + 0.10 * importance_score
  + 0.10 * confidence_score
  + 0.10 * scope_match_score
```

### Additional penalties

Subtract penalties for:
- contradiction
- staleness
- duplication
- oversaturation from the same session
- low provenance quality

---

## API design

Create a local Python service layer, not necessarily a separate network service at first.

Suggested module path:

```text
sage/
  memory/
    __init__.py
    models.py
    schemas.py
    repository.py
    embedder.py
    retriever.py
    ranker.py
    bundle_builder.py
    writer.py
    consolidator.py
    policy.py
    audit.py
    service.py
```

### Core service interface

```python
class MemoryService:
    def get_memory_bundle(...)
    def write_explicit_profile(...)
    def write_episode(...)
    def run_reflection(...)
    def promote_rules(...)
    def prune_memories(...)
    def explain_bundle(...)
```

### Recommended DTOs

- `MemoryContextRequest`
- `MemoryBundle`
- `ProfileMemoryItem`
- `EpisodeMemoryItem`
- `RuleMemoryItem`
- `ReflectionResult`
- `PromotionDecision`

Use Pydantic models and typed repository methods.

---

## Integration points in Sage Kaizen

### 1. Router integration

Insert memory retrieval between:
- route decision
- final prompt assembly

Flow:

```text
user input
  -> router intent normalization
  -> route selection
  -> memory bundle retrieval
  -> RAG retrieval (if needed)
  -> prompt assembly
  -> model call
  -> post-turn memory write
```

### 2. UI integration

Add optional controls in Streamlit:
- show active memory profile
- show retrieved memory bundle for current turn
- allow pin / unpin / forget
- allow “why did you remember this?”
- allow review of promoted rules

### 3. Background maintenance integration

Add a scheduler or maintenance command:
- session reflection
- nightly consolidation
- pruning
- contradiction scan
- index health check

---

## Recommended role split across brains

### Fast Brain responsibilities
- classify intent
- extract explicit preferences
- create lightweight episode summaries
- generate embeddings
- do first-pass reflection candidate extraction

### Architect Brain responsibilities
- deep consolidation
- contradiction analysis
- rule promotion decisions
- merge / split profile facts
- maintenance and governance reports

This reduces latency on the live path while preserving quality for long-term learning.

---

## Prompting guidance for memory-aware turns

Do not dump raw memory into the prompt.

Use a compact normalized form.

### Recommended format

```text
You have access to structured prior memory about this user and project.

Use the following memories only as guidance if relevant to the current request.
Prefer current user instructions over past memory.
If current instructions conflict with past memory, follow the current instructions and mark the old memory for review.

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

The Memory Service should emit structured logs for:

- bundle retrieval count
- bundle token size
- query latency
- vector hit count
- lexical hit count
- promotion decisions
- pruned rows
- contradiction detections
- reflection job status

Suggested metrics:
- `memory_bundle_latency_ms`
- `memory_episode_write_latency_ms`
- `memory_reflection_job_latency_ms`
- `memory_bundle_items_count`
- `memory_profile_items_count`
- `memory_rule_items_count`
- `memory_promotion_total`
- `memory_pruned_total`

---

## Safety and failure modes

### Risk: persona drift
Mitigation:
- pinned profile memory
- promotion thresholds
- contradiction checks
- user-confirmed locks

### Risk: stale technical preferences
Mitigation:
- recency weighting
- review timestamps
- contradiction-aware demotion

### Risk: privacy over-collection
Mitigation:
- typed memory classes
- explicit scope
- opt-out delete / forget support
- avoid storing raw transcript unless needed

### Risk: retrieval overload
Mitigation:
- strict top-k budgets
- score thresholding
- duplicate suppression
- compact memory formatting

### Risk: learning the wrong lesson
Mitigation:
- architect-only promotion for important rules
- audit log
- review queue for low-confidence candidates

---

## Recommended implementation phases

### Phase 1 — foundations
- create tables and indexes
- create typed repository layer
- create explicit profile write path
- create episodic write path
- implement memory bundle retrieval
- integrate into router

### Phase 2 — hybrid ranking
- add lexical + vector fusion
- add contradiction filtering
- add scope-aware scoring
- add compact prompt formatter

### Phase 3 — reflection and promotion
- implement reflection jobs
- implement promotion policy
- implement rule demotion and pruning
- add audit log

### Phase 4 — UI and governance
- add memory review controls
- add explainability panel
- add manual forget / pin / lock
- add maintenance dashboard

---

## File and code conventions for this project

Implementation must follow Sage Kaizen conventions:

- Python 3.14.3
- Windows-safe paths and commands
- modular components
- no beta or deprecated dependencies
- official docs preferred for technical behavior
- PostgreSQL + pgvector remain the memory backend
- fully local processing by default
- architect and fast brain roles remain replaceable

---

## Minimum acceptance criteria

A production-acceptable v1 must:

1. persist explicit user preferences
2. persist episodic memories with embeddings
3. retrieve hybrid top-k memory per turn
4. inject memory bundle before model inference
5. perform post-turn episodic write
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

### Integration tests
- router + memory bundle path
- post-turn write path
- reflection job path
- forget flow
- locked rule protection

### Behavioral tests
- user preference learned and reused
- current-turn override works
- stale preference demotes correctly
- duplicate memories do not flood prompt
- memory bundle remains within token budget

---

## Recommended next deliverables

1. `memory_schema.sql`
2. `sage/memory/models.py`
3. `sage/memory/schemas.py`
4. `sage/memory/repository.py`
5. `sage/memory/retriever.py`
6. `sage/memory/bundle_builder.py`
7. `sage/memory/writer.py`
8. `sage/memory/consolidator.py`
9. `sage/memory/service.py`
10. router integration patch
11. tests
12. Streamlit memory review panel

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

---

## References used to shape this design

These references influenced the architecture and should be reviewed during implementation:

- Anthropic Claude Code memory model: `CLAUDE.md` + auto memory
- Anthropic Claude Code extension patterns, hooks, and settings
- LangGraph long-term memory model: semantic / episodic / procedural
- Letta memory blocks and archival memory
- Letta sleep-time agents
- pgvector hybrid retrieval recommendations
