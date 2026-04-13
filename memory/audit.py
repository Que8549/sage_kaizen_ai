"""
memory/audit.py
Audit log writes for memory governance and reversibility.

Every write, promotion, demotion, lock, and forget must be recorded here.
The audit log is the primary mechanism for reviewing and reversing memory changes.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from sk_logging import get_logger
from .db import get_connection, new_uuid

_LOG = get_logger("sage_kaizen.memory.audit")

_INSERT_SQL = """
    INSERT INTO memory.audit_log
        (id, memory_table, memory_id, action_type, actor_type, actor_id,
         old_value, new_value, reason, created_at)
    VALUES
        (%(id)s, %(memory_table)s, %(memory_id)s, %(action_type)s, %(actor_type)s,
         %(actor_id)s, %(old_value)s::jsonb, %(new_value)s::jsonb, %(reason)s, NOW())
"""


def log_action(
    memory_table: str,       # 'profiles', 'episodes', 'rules', 'reflections'
    memory_id: str,
    action_type: str,        # 'insert', 'update', 'delete', 'promote', 'demote', 'lock', 'forget'
    actor_type: str,         # 'user', 'fast_brain', 'architect_brain', 'system', 'langmem'
    actor_id: Optional[str] = None,
    old_value: Optional[Dict[str, Any]] = None,
    new_value: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
) -> None:
    """
    Write one audit log row.  Non-blocking best-effort: exceptions are logged
    but never re-raised so audit failures never break the hot path.
    """
    try:
        params = {
            "id": new_uuid(),
            "memory_table": memory_table,
            "memory_id": memory_id,
            "action_type": action_type,
            "actor_type": actor_type,
            "actor_id": actor_id,
            "old_value": json.dumps(old_value) if old_value else "null",
            "new_value": json.dumps(new_value) if new_value else "null",
            "reason": reason,
        }
        with get_connection() as conn:
            conn.execute(_INSERT_SQL, params)
            conn.commit()
        _LOG.debug(
            "audit | table=%s id=%s action=%s actor=%s",
            memory_table, memory_id, action_type, actor_type,
        )
    except Exception as exc:
        _LOG.warning("audit | write failed (non-fatal): %s", exc)
