"""
tests/test_lazy.py

Unit tests for lazy.py — the thread-safe lazy singleton helper that replaced
~10 hand-rolled `global X; if X is None: X = ...` blocks.

The concurrency tests are the point of the module: most of those hand-rolled
blocks were unlocked, and the ones reached from the context injector's worker
pool could genuinely double-construct.
"""
from __future__ import annotations

import threading
import time

import pytest

from lazy import LazySingleton, lazy_singleton


class TestBasicCaching:
    def test_constructs_on_first_call(self):
        calls = []

        @lazy_singleton
        def get() -> str:
            calls.append(1)
            return "value"

        assert get() == "value"
        assert len(calls) == 1

    def test_second_call_returns_the_cached_instance(self):
        @lazy_singleton
        def get() -> list:
            return []

        assert get() is get()

    def test_factory_runs_exactly_once(self):
        calls = []

        @lazy_singleton
        def get() -> object:
            calls.append(1)
            return object()

        for _ in range(10):
            get()
        assert len(calls) == 1

    def test_initialised_flag(self):
        @lazy_singleton
        def get() -> str:
            return "x"

        assert get.initialised is False
        get()
        assert get.initialised is True


class TestReset:
    def test_reset_forces_reconstruction(self):
        calls = []

        @lazy_singleton
        def get() -> object:
            calls.append(1)
            return object()

        first = get()
        get.reset()
        second = get()
        assert first is not second
        assert len(calls) == 2

    def test_reset_clears_the_initialised_flag(self):
        @lazy_singleton
        def get() -> str:
            return "x"

        get()
        get.reset()
        assert get.initialised is False

    def test_reset_before_first_use_is_a_noop(self):
        @lazy_singleton
        def get() -> str:
            return "x"

        get.reset()
        assert get() == "x"


class TestNoneHandling:
    def test_none_is_not_cached(self):
        """
        The optional-dependency accessors rely on this: a factory that returns
        None because a dependency is unavailable must be retried, not latched
        for the process lifetime.
        """
        calls = []

        @lazy_singleton
        def get() -> str | None:
            calls.append(1)
            return None

        assert get() is None
        assert get() is None
        assert len(calls) == 2

    def test_none_then_a_value_caches_the_value(self):
        results = [None, "ready"]

        @lazy_singleton
        def get() -> str | None:
            return results.pop(0)

        assert get() is None
        assert get() == "ready"
        assert get() == "ready"      # now cached
        assert results == []

    def test_falsy_non_none_values_are_cached(self):
        calls = []

        @lazy_singleton
        def get() -> int:
            calls.append(1)
            return 0

        assert get() == 0
        assert get() == 0
        assert len(calls) == 1, "0 is not None and must be cached"


class TestExceptions:
    def test_exception_propagates(self):
        @lazy_singleton
        def get() -> str:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            get()

    def test_nothing_is_cached_after_a_failure(self):
        state = {"fail": True}

        @lazy_singleton
        def get() -> str:
            if state["fail"]:
                raise RuntimeError("not ready")
            return "ok"

        with pytest.raises(RuntimeError):
            get()
        state["fail"] = False
        assert get() == "ok"

    def test_lock_is_released_after_a_failure(self):
        """A raising factory must not leave the lock held."""
        @lazy_singleton
        def get() -> str:
            raise ValueError("x")

        for _ in range(3):
            with pytest.raises(ValueError):
                get()


class TestConcurrency:
    def test_constructs_once_under_contention(self):
        built: list[int] = []

        @lazy_singleton
        def get() -> object:
            built.append(1)
            time.sleep(0.02)      # widen the race window
            return object()

        results: list[object] = []
        threads = [
            threading.Thread(target=lambda: results.append(get()))
            for _ in range(16)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(built) == 1, f"factory ran {len(built)} times — lock is broken"
        assert len({id(r) for r in results}) == 1, "threads saw different instances"

    def test_separate_singletons_do_not_share_a_lock(self):
        """
        One lock per accessor. A shared lock would serialise unrelated
        initialisers and could deadlock if one factory called another.
        """
        order: list[str] = []
        started = threading.Event()

        @lazy_singleton
        def slow() -> str:
            started.set()
            time.sleep(0.15)
            order.append("slow")
            return "slow"

        @lazy_singleton
        def quick() -> str:
            order.append("quick")
            return "quick"

        t = threading.Thread(target=slow)
        t.start()
        started.wait(timeout=2)
        quick()          # must not block behind slow()
        t.join()

        assert order == ["quick", "slow"]

    def test_nested_singletons_do_not_deadlock(self):
        """_get_wiki_retriever() calls _ensure_rag() — that must not deadlock."""
        @lazy_singleton
        def inner() -> str:
            return "inner"

        @lazy_singleton
        def outer() -> str:
            return f"outer({inner()})"

        assert outer() == "outer(inner)"


class TestWrapperMetadata:
    def test_preserves_name_and_docstring(self):
        @lazy_singleton
        def get_thing() -> str:
            """The docstring."""
            return "x"

        assert get_thing.__name__ == "get_thing"
        assert get_thing.__doc__ == "The docstring."

    def test_returns_a_lazysingleton(self):
        @lazy_singleton
        def get() -> str:
            return "x"

        assert isinstance(get, LazySingleton)


class TestRealAccessorsUseIt:
    """The migration is only useful if the real accessors actually went through it."""

    def test_search_orchestrator(self):
        from search.search_orchestrator import get_orchestrator
        assert isinstance(get_orchestrator, LazySingleton)

    def test_news_resolver(self):
        from news.retrieval.news_resolver import get_news_resolver
        assert isinstance(get_news_resolver, LazySingleton)

    def test_market_client(self):
        from news.retrieval.market_client import get_market_client
        assert isinstance(get_market_client, LazySingleton)

    def test_news_settings(self):
        from news.news_settings import get_news_settings
        assert isinstance(get_news_settings, LazySingleton)

    def test_memory_embedder(self):
        from memory import embedder
        assert isinstance(embedder._get_client, LazySingleton)

    def test_context_injector_accessors(self):
        import rag_v1.runtime.context_injector as ci
        assert isinstance(ci._rag_pair, LazySingleton)
        assert isinstance(ci._get_wiki_retriever, LazySingleton)
        assert isinstance(ci._get_music_retriever, LazySingleton)

    def test_langmem_bridge(self):
        from memory.langmem_bridge import get_langmem_bridge
        assert isinstance(get_langmem_bridge, LazySingleton)
