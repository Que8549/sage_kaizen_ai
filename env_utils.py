"""
env_utils.py

Centralised environment-variable accessors for the Sage Kaizen runtime.

Why these exist rather than using os.getenv() inline
-----------------------------------------------------
Sage Kaizen has several runtime feature flags (RAG on/off, top-k, budget caps,
search enable/disable) that are read *per call* rather than once at startup.
pydantic-settings BaseSettings is the right tool for startup configuration
(see settings.py, pg_settings.py) — it reads the environment once, validates
types, and caches the result.  For flags read on every chat turn, a BaseSettings
instance would need to be re-instantiated each call to pick up changes, which
defeats its purpose.  Plain os.getenv() wrappers with type coercion are the
correct pattern here, and centralising them avoids the three-file duplication
that previously existed across searxng_client.py, search_orchestrator.py, and
context_injector.py.

Usage
-----
    from env_utils import env_bool, env_int, env_float, env_str

    enabled  = env_bool("SAGE_RAG_ENABLED", default=True)
    top_k    = env_int("SAGE_RAG_FAST_TOPK", default=4)
    timeout  = env_float("SAGE_SEARCH_TIMEOUT_S", default=8.0)
    base_url = env_str("SAGE_SEARCH_URL", default="http://localhost:8080")
"""
from __future__ import annotations

import os


def env_bool(name: str, *, default: bool = False) -> bool:
    """
    Read an environment variable as a boolean.

    Truthy values (case-insensitive): "1", "true", "yes", "y", "on"
    All other non-empty values → False.
    Missing or empty → default.
    """
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def env_int(name: str, *, default: int) -> int:
    """
    Read an environment variable as an integer.
    Returns default on missing, empty, or non-numeric values.
    """
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v.strip())
    except ValueError:
        return default


def env_float(name: str, *, default: float) -> float:
    """
    Read an environment variable as a float.
    Returns default on missing, empty, or non-numeric values.
    """
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v.strip())
    except ValueError:
        return default


def env_str(name: str, *, default: str) -> str:
    """
    Read an environment variable as a stripped string.
    Returns default when the variable is missing or empty.
    """
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip()
