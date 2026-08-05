"""
lazy.py

One thread-safe lazy singleton helper, replacing the hand-rolled
``global X; if X is None: X = ...`` blocks that had accumulated across the
codebase.

Why this exists
---------------
Sage Kaizen has ~10 process-wide singletons: the RAG injector, the wiki and
music retrievers, the search orchestrator, the BGE-M3 embed client, the LangMem
bridge, the news resolver, the market client, and the news settings. Every one
was written out by hand, and only two of them (memory.db.get_pool and
chat_service._get_memory) actually took a lock.

That mattered because most of them are reached from
rag_v1/runtime/context_injector.py's worker pool. Two threads racing an
unlocked initialiser both see ``None`` and both construct — and for
WikiRetriever that is not merely wasteful, since each instance calls
_ensure_service() and a lost race meant two cold torch/CUDA initialisations of
jina-clip-v2 on the same physical GPU at once. sage_kaizen_ai_ingest hit that
exact failure twice (its CLAUDE.md §15, 2026-05-28 and 2026-07-18) and had to
serialise service startup to stop machines freezing.

Locks were added to each site individually on 2026-08-04; this module removes
the duplication so the next singleton is correct by construction rather than by
remembering.

Usage
-----
    from lazy import lazy_singleton

    @lazy_singleton
    def get_orchestrator() -> SearchOrchestrator:
        return SearchOrchestrator()

    get_orchestrator()          # constructs once, under a lock
    get_orchestrator()          # returns the cached instance
    get_orchestrator.reset()    # drops it — for tests

Semantics
---------
* Double-checked locking: the fast path takes no lock once initialised.
* One lock per decorated function, so unrelated singletons never contend and
  cannot deadlock each other.
* A factory returning ``None`` is NOT cached — the next call retries. That
  suits the "optional dependency unavailable" pattern used here, where a
  transient failure should not be latched for the process lifetime.
* An exception propagates and nothing is cached, so a later call can retry.
* ``.reset()`` exists for tests. Production code should not call it: another
  thread may still hold a reference to the old instance.
"""
from __future__ import annotations

import functools
import threading
from typing import Callable, Generic, TypeVar

_T = TypeVar("_T")

__all__ = ["lazy_singleton", "LazySingleton"]


class LazySingleton(Generic[_T]):
    """
    Callable wrapper holding one lazily-constructed instance. See module docs.

    Deliberately not __slots__-ed: functools.update_wrapper copies __doc__ and
    friends onto the instance, and listing __doc__ in __slots__ collides with
    this very docstring. There are ~10 of these in the process, so the memory
    saving would be meaningless anyway.
    """

    def __init__(self, factory: Callable[[], _T]) -> None:
        self._factory: Callable[[], _T] = factory
        self._lock = threading.Lock()
        self._instance: _T | None = None
        functools.update_wrapper(self, factory)  # type: ignore[arg-type]

    def __call__(self) -> _T:
        # Fast path — no lock once constructed.
        instance = self._instance
        if instance is not None:
            return instance
        with self._lock:
            # Re-check: another thread may have constructed while we waited.
            if self._instance is None:
                self._instance = self._factory()
            return self._instance  # type: ignore[return-value]

    def reset(self) -> None:
        """Drop the cached instance so the next call reconstructs. Tests only."""
        with self._lock:
            self._instance = None

    @property
    def initialised(self) -> bool:
        """True when an instance is currently cached (no lock taken)."""
        return self._instance is not None


def lazy_singleton(factory: Callable[[], _T]) -> LazySingleton[_T]:
    """
    Decorate a zero-argument factory so it constructs at most once.

    See the module docstring for semantics — in particular that a ``None``
    return is not cached, which the optional-dependency accessors rely on.
    """
    return LazySingleton(factory)
