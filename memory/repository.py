"""
memory/repository.py
psycopg3 CRUD layer for all memory.* tables.

Design rules:
- All DB primitives (pool, connection, uuid, now) come from memory.db — never
  redefined here and never imported by other modules as private symbols.
- Vector literals use db.vec_str() which uses format specifier '.17g' to avoid
  locale-dependent decimal separators (fix for item 3).
- upsert_profile raises on a missing RETURNING id instead of silently falling
  back to a locally generated UUID that was never persisted (fix for item 2).
- fetch_episodes_vector no longer has a dead unreachable sql variable (fix for items 1/14).
- fetch_episodes_since added for consolidator use (fix for item 9).
- pgvector 0.8.x: SET LOCAL hnsw.iterative_scan applied inside transaction for
  all filtered vector queries.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .db import dumps, get_connection, new_uuid, now_utc, vec_str
from .schemas import EpisodeRow, ProfileRow, ReflectionRow, RuleRow
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.memory.repository")


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def upsert_profile(
    user_id: str,
    project_id: Optional[str],
    workspace_id: Optional[str],
    scope: str,
    profile_type: str,
    key: str,
    value_text: str,
    value_json: Optional[Dict[str, Any]] = None,
    confidence: float = 1.0,
    source_type: str = "explicit_user",
    is_pinned: bool = True,
    is_locked: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Insert or update a profile row.

    Returns the persisted row id from RETURNING — never a locally generated
    fallback UUID.  Raises RuntimeError if the DB does not return an id.
    """
    rid = new_uuid()
    now = now_utc()
    sql = """
        INSERT INTO memory.profiles
            (id, user_id, project_id, workspace_id, scope, profile_type, key,
             value_text, value_json, confidence, source_type, is_pinned, is_locked,
             is_active, created_at, updated_at, last_confirmed_at, metadata)
        VALUES
            (%(id)s, %(user_id)s, %(project_id)s, %(workspace_id)s, %(scope)s,
             %(profile_type)s, %(key)s, %(value_text)s, %(value_json)s,
             %(confidence)s, %(source_type)s, %(is_pinned)s, %(is_locked)s,
             TRUE, %(now)s, %(now)s, %(now)s, %(metadata)s)
        ON CONFLICT (
            user_id,
            COALESCE(project_id, ''),
            COALESCE(workspace_id, ''),
            scope, profile_type, key
        )
        DO UPDATE SET
            value_text        = EXCLUDED.value_text,
            value_json        = EXCLUDED.value_json,
            confidence        = EXCLUDED.confidence,
            source_type       = EXCLUDED.source_type,
            updated_at        = EXCLUDED.updated_at,
            last_confirmed_at = EXCLUDED.last_confirmed_at,
            is_active         = TRUE
        RETURNING id
    """
    params = {
        "id": rid, "user_id": user_id, "project_id": project_id,
        "workspace_id": workspace_id, "scope": scope,
        "profile_type": profile_type, "key": key, "value_text": value_text,
        "value_json": dumps(value_json) if value_json else None,
        "confidence": confidence, "source_type": source_type,
        "is_pinned": is_pinned, "is_locked": is_locked, "now": now,
        "metadata": dumps(metadata or {}),
    }
    with get_connection() as conn:
        row = conn.execute(sql, params).fetchone()
        conn.commit()

    if row is None:
        raise RuntimeError(
            f"upsert_profile: RETURNING id returned nothing for key={key!r} user={user_id!r}. "
            "This indicates a constraint violation or trigger rejection."
        )

    result_id = str(row["id"])
    _LOG.debug("memory.profiles | upsert key=%s user=%s → %s", key, user_id, result_id)
    return result_id


def fetch_active_profiles(
    user_id: str,
    project_id: Optional[str] = None,
) -> List[ProfileRow]:
    """Fetch all active profile rows for a user (always-on bundle)."""
    sql = """
        SELECT id, user_id, project_id, workspace_id, scope, profile_type, key,
               value_text, value_json, confidence, source_type, is_pinned, is_locked,
               is_active, created_at, updated_at, last_confirmed_at, expires_at, metadata
        FROM memory.profiles
        WHERE user_id = %(user_id)s
          AND is_active = TRUE
          AND (expires_at IS NULL OR expires_at > NOW())
          AND (project_id IS NULL OR project_id = %(project_id)s OR scope = 'user')
        ORDER BY is_pinned DESC, confidence DESC, updated_at DESC
    """
    with get_connection() as conn:
        rows = conn.execute(sql, {"user_id": user_id, "project_id": project_id}).fetchall()
    return [_profile_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

def insert_episode(
    user_id: str,
    project_id: Optional[str],
    workspace_id: Optional[str],
    session_id: Optional[str],
    scope: str,
    event_type: str,
    summary_text: str,
    embedding: Optional[List[float]] = None,
    intent_label: Optional[str] = None,
    raw_excerpt: Optional[str] = None,
    tags: Optional[List[str]] = None,
    importance: float = 0.5,
    confidence: float = 0.6,
    was_user_correction: bool = False,
    was_explicit_preference: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Insert a new episode row.  Returns the new row id."""
    rid = new_uuid()
    now = now_utc()
    v = vec_str(embedding) if embedding else None   # locale-safe serialisation
    sql = """
        INSERT INTO memory.episodes
            (id, user_id, project_id, workspace_id, session_id, scope, event_type,
             intent_label, summary_text, raw_excerpt, tags, importance, confidence,
             was_user_correction, was_explicit_preference, embedding,
             created_at, metadata)
        VALUES
            (%(id)s, %(user_id)s, %(project_id)s, %(workspace_id)s, %(session_id)s,
             %(scope)s, %(event_type)s, %(intent_label)s, %(summary_text)s,
             %(raw_excerpt)s, %(tags)s::jsonb, %(importance)s, %(confidence)s,
             %(was_user_correction)s, %(was_explicit_preference)s,
             %(embedding)s::vector, %(now)s, %(metadata)s::jsonb)
        RETURNING id
    """
    params = {
        "id": rid, "user_id": user_id, "project_id": project_id,
        "workspace_id": workspace_id, "session_id": session_id, "scope": scope,
        "event_type": event_type, "intent_label": intent_label,
        "summary_text": summary_text, "raw_excerpt": raw_excerpt,
        "tags": dumps(tags or []), "importance": importance,
        "confidence": confidence, "was_user_correction": was_user_correction,
        "was_explicit_preference": was_explicit_preference,
        "embedding": v, "now": now,
        "metadata": dumps(metadata or {}),
    }
    with get_connection() as conn:
        conn.execute(sql, params)
        conn.commit()
    _LOG.debug("memory.episodes | insert event=%s user=%s id=%s", event_type, user_id, rid)
    return rid


def fetch_episodes_lexical(
    user_id: str,
    project_id: Optional[str],
    query_text: str,
    limit: int = 10,
) -> List[Tuple[EpisodeRow, float]]:
    """Full-text search on memory.episodes.  Returns (row, ts_rank) pairs."""
    sql = """
        SELECT id, user_id, project_id, workspace_id, session_id, scope, event_type,
               intent_label, summary_text, raw_excerpt, tags, importance, confidence,
               sentiment, was_user_correction, was_explicit_preference,
               contradiction_group, created_at, last_accessed_at, last_retrieved_at,
               expires_at, metadata,
               ts_rank(search_tsv, plainto_tsquery('english', %(query)s)) AS rank
        FROM memory.episodes
        WHERE user_id = %(user_id)s
          AND (project_id IS NULL OR project_id = %(project_id)s)
          AND (expires_at IS NULL OR expires_at > NOW())
          AND search_tsv @@ plainto_tsquery('english', %(query)s)
        ORDER BY rank DESC
        LIMIT %(limit)s
    """
    with get_connection() as conn:
        rows = conn.execute(
            sql, {"user_id": user_id, "project_id": project_id,
                  "query": query_text, "limit": limit}
        ).fetchall()
    return [(_episode_row(r), float(r["rank"])) for r in rows]


def fetch_episodes_vector(
    user_id: str,
    project_id: Optional[str],
    embedding: List[float],
    limit: int = 10,
) -> List[Tuple[EpisodeRow, float]]:
    """
    Vector similarity search on memory.episodes using HNSW.

    pgvector 0.8.x: SET LOCAL hnsw.iterative_scan = relaxed_order prevents
    silent under-retrieval when user_id + project_id filters are selective.
    Vector literal uses db.vec_str() (locale-safe, '.17g' format).
    """
    v = vec_str(embedding)
    query_sql = """
        SELECT id, user_id, project_id, workspace_id, session_id, scope, event_type,
               intent_label, summary_text, raw_excerpt, tags, importance, confidence,
               sentiment, was_user_correction, was_explicit_preference,
               contradiction_group, created_at, last_accessed_at, last_retrieved_at,
               expires_at, metadata,
               1.0 - (embedding <=> %(vec)s::vector) AS similarity
        FROM memory.episodes
        WHERE user_id = %(user_id)s
          AND (project_id IS NULL OR project_id = %(project_id)s)
          AND (expires_at IS NULL OR expires_at > NOW())
          AND embedding IS NOT NULL
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(limit)s
    """
    with get_connection() as conn:
        with conn.transaction():
            conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            conn.execute("SET LOCAL hnsw.max_scan_tuples = 20000")
            rows = conn.execute(
                query_sql,
                {"vec": v, "user_id": user_id,
                 "project_id": project_id, "limit": limit},
            ).fetchall()
    return [(_episode_row(r), float(r["similarity"])) for r in rows]


def fetch_episodes_since(
    user_id: str,
    project_id: Optional[str],
    cutoff: datetime,
    limit: int = 40,
) -> List[Dict[str, Any]]:
    """
    Fetch raw episode rows created after `cutoff`, ordered newest-first.
    Used by the consolidator — returns plain dicts to avoid importing EpisodeRow
    into the consolidation layer.
    """
    sql = """
        SELECT id, event_type, summary_text, importance, confidence,
               was_user_correction, was_explicit_preference, created_at
        FROM memory.episodes
        WHERE user_id = %(user_id)s
          AND (project_id IS NULL OR project_id = %(project_id)s)
          AND created_at >= %(cutoff)s
          AND (expires_at IS NULL OR expires_at > NOW())
        ORDER BY created_at DESC
        LIMIT %(limit)s
    """
    with get_connection() as conn:
        rows = conn.execute(
            sql, {"user_id": user_id, "project_id": project_id,
                  "cutoff": cutoff, "limit": limit}
        ).fetchall()
    return [dict(r) for r in rows]


def touch_episode_retrieved(episode_id: str) -> None:
    """Update last_retrieved_at timestamp. Non-critical; caller catches exceptions."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE memory.episodes SET last_retrieved_at = NOW() WHERE id = %(id)s",
            {"id": episode_id},
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def fetch_active_rules(
    user_id: Optional[str],
    project_id: Optional[str],
    query_text: Optional[str] = None,
    limit: int = 8,
) -> List[RuleRow]:
    """Fetch active rules, optionally filtered by FTS on rule_text."""
    if query_text:
        sql = """
            SELECT id, user_id, project_id, workspace_id, scope, rule_kind, rule_text,
                   rationale, confidence, promotion_count, source_type, source_memory_id,
                   is_locked, is_active, review_status, created_at, updated_at,
                   expires_at, metadata,
                   ts_rank(search_tsv, plainto_tsquery('english', %(query)s)) AS rank
            FROM memory.rules
            WHERE (user_id IS NULL OR user_id = %(user_id)s)
              AND (project_id IS NULL OR project_id = %(project_id)s)
              AND is_active = TRUE
              AND review_status IN ('approved', 'proposed')
              AND (expires_at IS NULL OR expires_at > NOW())
              AND search_tsv @@ plainto_tsquery('english', %(query)s)
            ORDER BY is_locked DESC, rank DESC, confidence DESC
            LIMIT %(limit)s
        """
        params: Dict[str, Any] = {
            "user_id": user_id, "project_id": project_id,
            "query": query_text, "limit": limit,
        }
    else:
        sql = """
            SELECT id, user_id, project_id, workspace_id, scope, rule_kind, rule_text,
                   rationale, confidence, promotion_count, source_type, source_memory_id,
                   is_locked, is_active, review_status, created_at, updated_at,
                   expires_at, metadata
            FROM memory.rules
            WHERE (user_id IS NULL OR user_id = %(user_id)s)
              AND (project_id IS NULL OR project_id = %(project_id)s)
              AND is_active = TRUE
              AND review_status IN ('approved', 'proposed')
              AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY is_locked DESC, confidence DESC
            LIMIT %(limit)s
        """
        params = {"user_id": user_id, "project_id": project_id, "limit": limit}
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_rule_row(r) for r in rows]


def insert_rule(
    user_id: Optional[str],
    project_id: Optional[str],
    scope: str,
    rule_kind: str,
    rule_text: str,
    rationale: Optional[str] = None,
    confidence: float = 0.7,
    source_type: str = "promoted_from_episode",
    source_memory_id: Optional[str] = None,
    is_locked: bool = False,
    review_status: str = "proposed",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    rid = new_uuid()
    now = now_utc()
    sql = """
        INSERT INTO memory.rules
            (id, user_id, project_id, scope, rule_kind, rule_text, rationale,
             confidence, promotion_count, source_type, source_memory_id,
             is_locked, is_active, review_status, created_at, updated_at, metadata)
        VALUES
            (%(id)s, %(user_id)s, %(project_id)s, %(scope)s, %(rule_kind)s,
             %(rule_text)s, %(rationale)s, %(confidence)s, 0, %(source_type)s,
             %(source_memory_id)s, %(is_locked)s, TRUE, %(review_status)s,
             %(now)s, %(now)s, %(metadata)s::jsonb)
        RETURNING id
    """
    params = {
        "id": rid, "user_id": user_id, "project_id": project_id, "scope": scope,
        "rule_kind": rule_kind, "rule_text": rule_text, "rationale": rationale,
        "confidence": confidence, "source_type": source_type,
        "source_memory_id": source_memory_id, "is_locked": is_locked,
        "review_status": review_status, "now": now,
        "metadata": dumps(metadata or {}),
    }
    with get_connection() as conn:
        conn.execute(sql, params)
        conn.commit()
    _LOG.debug("memory.rules | insert kind=%s user=%s id=%s", rule_kind, user_id, rid)
    return rid


# ---------------------------------------------------------------------------
# Reflections
# ---------------------------------------------------------------------------

def insert_reflection(
    user_id: str,
    project_id: Optional[str],
    session_id: Optional[str],
    reflection_type: str,
    summary_text: str,
    profile_candidates: Optional[List[Any]] = None,
    rule_candidates: Optional[List[Any]] = None,
    contradictions: Optional[List[Any]] = None,
    pruning_suggestions: Optional[List[Any]] = None,
    confidence: float = 0.7,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    rid = new_uuid()
    sql = """
        INSERT INTO memory.reflections
            (id, user_id, project_id, session_id, reflection_type, summary_text,
             extracted_profile_candidates, extracted_rule_candidates,
             contradictions, pruning_suggestions, confidence, created_at, metadata)
        VALUES
            (%(id)s, %(user_id)s, %(project_id)s, %(session_id)s, %(reflection_type)s,
             %(summary_text)s, %(profile_candidates)s::jsonb, %(rule_candidates)s::jsonb,
             %(contradictions)s::jsonb, %(pruning_suggestions)s::jsonb,
             %(confidence)s, NOW(), %(metadata)s::jsonb)
        RETURNING id
    """
    params = {
        "id": rid, "user_id": user_id, "project_id": project_id,
        "session_id": session_id, "reflection_type": reflection_type,
        "summary_text": summary_text,
        "profile_candidates": dumps(profile_candidates or []),
        "rule_candidates": dumps(rule_candidates or []),
        "contradictions": dumps(contradictions or []),
        "pruning_suggestions": dumps(pruning_suggestions or []),
        "confidence": confidence, "metadata": dumps(metadata or {}),
    }
    with get_connection() as conn:
        conn.execute(sql, params)
        conn.commit()
    _LOG.debug("memory.reflections | insert type=%s user=%s id=%s", reflection_type, user_id, rid)
    return rid


# ---------------------------------------------------------------------------
# Forget / disable
# ---------------------------------------------------------------------------

def disable_episode(episode_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE memory.episodes SET expires_at = NOW() WHERE id = %(id)s",
            {"id": episode_id},
        )
        conn.commit()
    _LOG.info("memory.episodes | disabled id=%s", episode_id)


def disable_profile(profile_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE memory.profiles SET is_active = FALSE, updated_at = NOW() WHERE id = %(id)s",
            {"id": profile_id},
        )
        conn.commit()
    _LOG.info("memory.profiles | disabled id=%s", profile_id)


def disable_rule(rule_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE memory.rules SET is_active = FALSE, updated_at = NOW() WHERE id = %(id)s",
            {"id": rule_id},
        )
        conn.commit()
    _LOG.info("memory.rules | disabled id=%s", rule_id)


# ---------------------------------------------------------------------------
# Row constructors (internal)
# ---------------------------------------------------------------------------

def _profile_row(r: Dict[str, Any]) -> ProfileRow:
    return ProfileRow(
        id=str(r["id"]), user_id=r["user_id"], project_id=r["project_id"],
        workspace_id=r.get("workspace_id"), scope=r["scope"],
        profile_type=r["profile_type"], key=r["key"], value_text=r["value_text"],
        value_json=r["value_json"], confidence=float(r["confidence"]),
        source_type=r["source_type"], is_pinned=bool(r["is_pinned"]),
        is_locked=bool(r["is_locked"]), is_active=bool(r["is_active"]),
        created_at=r["created_at"], updated_at=r["updated_at"],
        last_confirmed_at=r.get("last_confirmed_at"), expires_at=r.get("expires_at"),
        metadata=r.get("metadata") or {},
    )


def _episode_row(r: Dict[str, Any]) -> EpisodeRow:
    return EpisodeRow(
        id=str(r["id"]), user_id=r["user_id"], project_id=r["project_id"],
        workspace_id=r.get("workspace_id"), session_id=r.get("session_id"),
        scope=r["scope"], event_type=r["event_type"],
        intent_label=r.get("intent_label"), summary_text=r["summary_text"],
        raw_excerpt=r.get("raw_excerpt"),
        tags=r["tags"] if isinstance(r["tags"], list) else [],
        importance=float(r["importance"]), confidence=float(r["confidence"]),
        sentiment=r.get("sentiment"), was_user_correction=bool(r["was_user_correction"]),
        was_explicit_preference=bool(r["was_explicit_preference"]),
        contradiction_group=r.get("contradiction_group"),
        embedding=None,
        created_at=r["created_at"], last_accessed_at=r.get("last_accessed_at"),
        last_retrieved_at=r.get("last_retrieved_at"), expires_at=r.get("expires_at"),
        metadata=r.get("metadata") or {},
    )


def _rule_row(r: Dict[str, Any]) -> RuleRow:
    return RuleRow(
        id=str(r["id"]), user_id=r.get("user_id"), project_id=r.get("project_id"),
        workspace_id=r.get("workspace_id"), scope=r["scope"],
        rule_kind=r["rule_kind"], rule_text=r["rule_text"],
        rationale=r.get("rationale"), confidence=float(r["confidence"]),
        promotion_count=int(r["promotion_count"]), source_type=r["source_type"],
        source_memory_id=r.get("source_memory_id") and str(r["source_memory_id"]),
        is_locked=bool(r["is_locked"]), is_active=bool(r["is_active"]),
        review_status=r["review_status"], created_at=r["created_at"],
        updated_at=r["updated_at"], expires_at=r.get("expires_at"),
        metadata=r.get("metadata") or {},
    )
