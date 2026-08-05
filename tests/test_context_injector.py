"""
tests/test_context_injector.py

Unit tests for rag_v1/runtime/context_injector.py.

Focus areas:
  * Each fetch worker's early-exit conditions and failure isolation — a broken
    context source must degrade the answer, never break the turn.
  * The shared collection deadline (bugfix 2026-08-04): five sequential
    `.result(timeout=...)` calls used to make the worst case their SUM.
  * Token-budget trimming and the `*_trimmed` flags in the structured log.
  * Thread-safety of the lazy singletons.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

import rag_v1.runtime.context_injector as ci
from router import RouteDecision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    """Each test starts with cold module-level singletons."""
    for accessor in (ci._rag_pair, ci._get_wiki_retriever, ci._get_music_retriever):
        accessor.reset()
    yield
    for accessor in (ci._rag_pair, ci._get_wiki_retriever, ci._get_music_retriever):
        accessor.reset()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "SAGE_RAG_ENABLED", "SAGE_WIKI_RAG_ENABLED", "SAGE_SEARCH_ENABLED",
        "SAGE_SEARCH_SUMMARIZE", "SAGE_RAG_MIN_CHARS", "SAGE_RAG_FAST_TOPK",
        "SAGE_RAG_ARCH_TOPK", "SAGE_RAG_WIKI_FAST_MAX_CHARS",
        "SAGE_RAG_WIKI_ARCH_MAX_CHARS", "SAGE_SEARCH_FAST_MAX_CHARS",
        "SAGE_SEARCH_ARCH_MAX_CHARS",
    ):
        monkeypatch.delenv(var, raising=False)


FAST = RouteDecision(brain="FAST", reasons=[], score=0)
ARCH = RouteDecision(brain="ARCHITECT", reasons=[], score=9)
SEARCHY = RouteDecision(
    brain="FAST", reasons=[], score=0, needs_search=True, search_categories=("news",)
)
MUSICAL = RouteDecision(brain="FAST", reasons=[], score=0, needs_music=True)

LONG = "a query long enough to clear the min-chars gate"


def msgs():
    return [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# apply_rag
# ---------------------------------------------------------------------------

class TestApplyRag:
    def test_empty_text_short_circuits(self):
        m = msgs()
        assert ci.apply_rag(m, "", FAST) == (m, [])

    def test_disabled_via_argument(self):
        m = msgs()
        assert ci.apply_rag(m, LONG, FAST, rag_enabled=False) == (m, [])

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("SAGE_RAG_ENABLED", "false")
        m = msgs()
        assert ci.apply_rag(m, LONG, FAST) == (m, [])

    def test_below_min_chars_short_circuits(self):
        m = msgs()
        assert ci.apply_rag(m, "hi", FAST) == (m, [])

    def test_min_chars_is_configurable(self, monkeypatch):
        monkeypatch.setenv("SAGE_RAG_MIN_CHARS", "1")
        injector = MagicMock()
        injector.maybe_inject.return_value = (["out"], ["src"])
        with patch.object(ci, "_ensure_rag", return_value=(injector, MagicMock())):
            out, srcs = ci.apply_rag(msgs(), "hi", FAST)
        assert out == ["out"] and srcs == ["src"]

    @pytest.mark.parametrize(
        "decision,env,expected_k",
        [
            (FAST, {}, 4),
            (ARCH, {}, 10),
            (FAST, {"SAGE_RAG_FAST_TOPK": "7"}, 7),
            (ARCH, {"SAGE_RAG_ARCH_TOPK": "21"}, 21),
        ],
    )
    def test_top_k_per_brain(self, monkeypatch, decision, env, expected_k):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        injector = MagicMock()
        injector.maybe_inject.return_value = ([], [])
        with patch.object(ci, "_ensure_rag", return_value=(injector, MagicMock())):
            ci.apply_rag(msgs(), LONG, decision)
        assert injector.maybe_inject.call_args.kwargs["top_k"] == expected_k

    def test_injector_exception_degrades_to_no_rag(self):
        injector = MagicMock()
        injector.maybe_inject.side_effect = RuntimeError("db down")
        m = msgs()
        with patch.object(ci, "_ensure_rag", return_value=(injector, MagicMock())):
            assert ci.apply_rag(m, LONG, FAST) == (m, [])


# ---------------------------------------------------------------------------
# _fetch_wiki_result
# ---------------------------------------------------------------------------

class TestFetchWikiResult:
    def test_disabled_via_argument(self):
        assert ci._fetch_wiki_result(LONG, FAST, wiki_enabled=False) == ("", [])

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("SAGE_WIKI_RAG_ENABLED", "false")
        assert ci._fetch_wiki_result(LONG, FAST) == ("", [])

    def test_empty_text(self):
        assert ci._fetch_wiki_result("", FAST) == ("", [])

    def test_below_min_chars(self):
        assert ci._fetch_wiki_result("hi", FAST) == ("", [])

    def test_no_retriever_available(self):
        with patch.object(ci, "_get_wiki_retriever", return_value=None):
            assert ci._fetch_wiki_result(LONG, FAST) == ("", [])

    def test_empty_result_returns_blank(self):
        r = MagicMock()
        r.search.return_value = MagicMock(empty=True, chunks=[])
        with patch.object(ci, "_get_wiki_retriever", return_value=r):
            assert ci._fetch_wiki_result(LONG, FAST) == ("", [])

    def test_formats_chunks_with_title_and_section(self):
        chunk = MagicMock(title="Snail", section_path=["Biology", "Shell"],
                          text="Snails have shells.", score=0.912)
        r = MagicMock()
        r.search.return_value = MagicMock(empty=False, chunks=[chunk], images=["i"])
        with patch.object(ci, "_get_wiki_retriever", return_value=r):
            block, images = ci._fetch_wiki_result(LONG, FAST)
        assert "[Snail / Biology > Shell | score=0.912]" in block
        assert "Snails have shells." in block
        assert images == ["i"]

    def test_missing_section_path_labelled_introduction(self):
        chunk = MagicMock(title="X", section_path=None, text="t", score=0.5)
        r = MagicMock()
        r.search.return_value = MagicMock(empty=False, chunks=[chunk], images=[])
        with patch.object(ci, "_get_wiki_retriever", return_value=r):
            block, _ = ci._fetch_wiki_result(LONG, FAST)
        assert "/ Introduction |" in block

    def test_multiple_chunks_are_separated(self):
        c = lambda t: MagicMock(title=t, section_path=None, text="body", score=0.5)
        r = MagicMock()
        r.search.return_value = MagicMock(empty=False, chunks=[c("A"), c("B")], images=[])
        with patch.object(ci, "_get_wiki_retriever", return_value=r):
            block, _ = ci._fetch_wiki_result(LONG, FAST)
        assert "\n\n---\n\n" in block

    def test_exception_degrades_to_blank(self):
        r = MagicMock()
        r.search.side_effect = RuntimeError("gpu gone")
        with patch.object(ci, "_get_wiki_retriever", return_value=r):
            assert ci._fetch_wiki_result(LONG, FAST) == ("", [])

    def test_display_gpu_refusal_degrades_to_blank(self):
        """The new display-GPU guard must not break the turn."""
        from rag_v1.wiki.wiki_retriever import DisplayGpuRefused
        r = MagicMock()
        r.search.side_effect = DisplayGpuRefused("cuda:0")
        with patch.object(ci, "_get_wiki_retriever", return_value=r):
            assert ci._fetch_wiki_result(LONG, FAST) == ("", [])


# ---------------------------------------------------------------------------
# _fetch_search_result
# ---------------------------------------------------------------------------

class TestFetchSearchResult:
    def test_skipped_when_decision_does_not_need_search(self):
        assert ci._fetch_search_result(LONG, FAST) == ("", None)

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("SAGE_SEARCH_ENABLED", "false")
        assert ci._fetch_search_result(LONG, SEARCHY) == ("", None)

    def test_below_min_chars(self):
        assert ci._fetch_search_result("hi", SEARCHY) == ("", None)

    def test_orchestrator_failure_degrades(self):
        with patch.object(ci, "get_orchestrator", side_effect=RuntimeError("no searxng")):
            assert ci._fetch_search_result(LONG, SEARCHY) == ("", None)

    def test_empty_evidence_returns_blank(self):
        orch = MagicMock()
        orch.search.return_value = MagicMock(empty=True)
        with patch.object(ci, "get_orchestrator", return_value=orch):
            assert ci._fetch_search_result(LONG, SEARCHY) == ("", None)

    def _evidence(self):
        result = MagicMock(title="T", url="https://e.com", source_engine="ddg",
                           published_date="2026-08-01", snippet="s")
        return MagicMock(
            empty=False, results=[result], query="q",
            fetched_at="2026-08-04T12:30:00+00:00",
            categories_queried=("news",),
        )

    def test_summary_path_builds_context_block(self):
        orch = MagicMock()
        orch.search.return_value = self._evidence()
        with (
            patch.object(ci, "get_orchestrator", return_value=orch),
            patch.object(ci, "summarize_evidence", return_value="A summary."),
        ):
            block, ev = ci._fetch_search_result(
                LONG, SEARCHY, fast_base_url="http://f", fast_model_id="m",
            )
        assert block.startswith('<search_context fetched="2026-08-04 12:30 UTC" categories="news">')
        assert block.endswith("</search_context>")
        assert "A summary." in block
        assert "[1] T | ddg | 2026-08-01 — https://e.com" in block
        assert ev is not None and ev.summarized_text == "A summary."

    def test_prefers_dedicated_summarizer_over_fast_brain(self):
        orch = MagicMock()
        orch.search.return_value = self._evidence()
        with (
            patch.object(ci, "get_orchestrator", return_value=orch),
            patch.object(ci, "summarize_evidence", return_value="s") as summ,
        ):
            ci._fetch_search_result(
                LONG, SEARCHY,
                fast_base_url="http://fast", fast_model_id="fm",
                summarizer_base_url="http://sum", summarizer_model_id="sm",
            )
        assert summ.call_args.kwargs["fast_base_url"] == "http://sum"
        assert summ.call_args.kwargs["fast_model_id"] == "sm"

    def test_falls_back_to_fast_brain_when_no_summarizer(self):
        orch = MagicMock()
        orch.search.return_value = self._evidence()
        with (
            patch.object(ci, "get_orchestrator", return_value=orch),
            patch.object(ci, "summarize_evidence", return_value="s") as summ,
        ):
            ci._fetch_search_result(
                LONG, SEARCHY, fast_base_url="http://fast", fast_model_id="fm",
            )
        assert summ.call_args.kwargs["fast_base_url"] == "http://fast"

    def test_summarization_disabled_uses_raw_snippets(self, monkeypatch):
        monkeypatch.setenv("SAGE_SEARCH_SUMMARIZE", "false")
        orch = MagicMock()
        orch.search.return_value = self._evidence()
        with (
            patch.object(ci, "get_orchestrator", return_value=orch),
            patch.object(ci, "build_raw_context", return_value="RAW") as raw,
            patch.object(ci, "summarize_evidence") as summ,
        ):
            block, _ = ci._fetch_search_result(LONG, SEARCHY)
        summ.assert_not_called()
        raw.assert_called_once()
        assert "RAW" in block

    def test_summarizer_failure_falls_back_to_raw(self):
        orch = MagicMock()
        orch.search.return_value = self._evidence()
        with (
            patch.object(ci, "get_orchestrator", return_value=orch),
            patch.object(ci, "summarize_evidence", side_effect=RuntimeError("brain down")),
            patch.object(ci, "build_raw_context", return_value="RAW"),
        ):
            block, _ = ci._fetch_search_result(
                LONG, SEARCHY, fast_base_url="http://f", fast_model_id="m",
            )
        assert "RAW" in block

    def test_no_endpoint_configured_uses_raw(self):
        orch = MagicMock()
        orch.search.return_value = self._evidence()
        with (
            patch.object(ci, "get_orchestrator", return_value=orch),
            patch.object(ci, "build_raw_context", return_value="RAW"),
            patch.object(ci, "summarize_evidence") as summ,
        ):
            block, _ = ci._fetch_search_result(LONG, SEARCHY)
        summ.assert_not_called()
        assert "RAW" in block

    def test_default_categories_when_decision_has_none(self):
        d = RouteDecision(brain="FAST", reasons=[], score=0, needs_search=True)
        orch = MagicMock()
        orch.search.return_value = MagicMock(empty=True)
        with patch.object(ci, "get_orchestrator", return_value=orch):
            ci._fetch_search_result(LONG, d)
        assert orch.search.call_args.kwargs["categories"] == ["general", "news"]


# ---------------------------------------------------------------------------
# _fetch_music_result / _fetch_news_result
# ---------------------------------------------------------------------------

class TestFetchMusicResult:
    def test_skipped_when_not_needed(self):
        assert ci._fetch_music_result(LONG, FAST) == ""

    def test_empty_text(self):
        assert ci._fetch_music_result("", MUSICAL) == ""

    def test_no_retriever(self):
        with patch.object(ci, "_get_music_retriever", return_value=None):
            assert ci._fetch_music_result(LONG, MUSICAL) == ""

    def test_no_intent_detected(self):
        with (
            patch.object(ci, "_get_music_retriever", return_value=MagicMock()),
            patch.object(ci, "_detect_music_intent", return_value=None),
        ):
            assert ci._fetch_music_result(LONG, MUSICAL) == ""

    def test_formats_context(self):
        intent = MagicMock(intent="find_songs")
        r = MagicMock()
        r.dispatch.return_value = ["s1", "s2"]
        with (
            patch.object(ci, "_get_music_retriever", return_value=r),
            patch.object(ci, "_detect_music_intent", return_value=intent),
            patch.object(ci, "format_music_context", return_value="<music_context/>"),
        ):
            assert ci._fetch_music_result(LONG, MUSICAL) == "<music_context/>"

    def test_exception_degrades(self):
        r = MagicMock()
        r.dispatch.side_effect = RuntimeError("db")
        with (
            patch.object(ci, "_get_music_retriever", return_value=r),
            patch.object(ci, "_detect_music_intent", return_value=MagicMock()),
        ):
            assert ci._fetch_music_result(LONG, MUSICAL) == ""


class TestFetchNewsResult:
    def test_empty_text(self):
        assert ci._fetch_news_result("") == ""

    def test_no_context_resolved(self):
        with patch.object(ci, "resolve_news_context", return_value=None):
            assert ci._fetch_news_result(LONG) == ""

    def test_returns_xml_block(self):
        ctx = MagicMock(source="db", is_stale=False)
        ctx.to_xml_block.return_value = "<news_context/>"
        with patch.object(ci, "resolve_news_context", return_value=ctx):
            assert ci._fetch_news_result(LONG) == "<news_context/>"

    def test_exception_degrades(self):
        with patch.object(ci, "resolve_news_context", side_effect=RuntimeError("x")):
            assert ci._fetch_news_result(LONG) == ""


# ---------------------------------------------------------------------------
# _collect — the shared-deadline bugfix
# ---------------------------------------------------------------------------

class TestCollect:
    def test_returns_worker_result(self):
        fut: Future = Future()
        fut.set_result("value")
        deadline = time.monotonic() + 30
        assert ci._collect(fut, "rag", deadline, "fallback", "label") == "value"

    def test_returns_fallback_on_worker_exception(self):
        fut: Future = Future()
        fut.set_exception(RuntimeError("worker blew up"))
        deadline = time.monotonic() + 30
        assert ci._collect(fut, "rag", deadline, "fallback", "label") == "fallback"

    @staticmethod
    def _blocked_future(release: threading.Event) -> tuple[ThreadPoolExecutor, "Future"]:
        """
        A future that stays pending until `release` is set.

        Deliberately not `ex.submit(time.sleep, 10)`: ThreadPoolExecutor's
        shutdown waits for running tasks, so a sleeping worker would add its
        full duration to the test run even after _collect() has given up.
        """
        ex = ThreadPoolExecutor(max_workers=1)
        return ex, ex.submit(release.wait)

    def test_returns_fallback_on_timeout(self):
        release = threading.Event()
        ex, fut = self._blocked_future(release)
        try:
            # Deadline already passed → zero budget → immediate fallback.
            out = ci._collect(fut, "rag", time.monotonic() - 1, "fallback", "label")
            assert out == "fallback"
        finally:
            release.set()
            ex.shutdown(wait=True)

    def test_never_waits_past_the_shared_deadline(self):
        """The regression this guards: five sequential waits used to SUM."""
        release = threading.Event()
        ex, fut = self._blocked_future(release)
        try:
            start = time.monotonic()
            out = ci._collect(fut, "search", time.monotonic() + 0.2, "fallback", "l")
            elapsed = time.monotonic() - start
        finally:
            release.set()
            ex.shutdown(wait=True)
        # search's own ceiling is 30 s; the shared deadline must win.
        assert out == "fallback"
        assert elapsed < 2.0, f"waited {elapsed:.2f}s — deadline was not honoured"

    def test_budget_is_min_of_worker_cap_and_deadline(self):
        release = threading.Event()
        ex, fut = self._blocked_future(release)
        try:
            start = time.monotonic()
            # Deadline is generous; the per-worker cap must apply instead.
            with patch.dict(ci._WORKER_TIMEOUTS, {"music": 0.2}):
                ci._collect(fut, "music", time.monotonic() + 60, "fb", "l")
            elapsed = time.monotonic() - start
        finally:
            release.set()
            ex.shutdown(wait=True)
        assert elapsed < 2.0

    def test_total_budget_matches_slowest_worker(self):
        assert ci._TOTAL_CONTEXT_BUDGET_S == max(ci._WORKER_TIMEOUTS.values())

    def test_pool_is_sized_above_the_fanout(self):
        """Concurrent turns must not starve on queued tasks."""
        assert ci._POOL._max_workers >= ci._FANOUT * 2


# ---------------------------------------------------------------------------
# apply_rag_and_wiki_parallel — assembly and injection
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_workers():
    """Patch all five fetch workers; each test overrides what it cares about."""
    with (
        patch.object(ci, "apply_rag", side_effect=lambda m, *a, **k: (m, [])) as rag,
        patch.object(ci, "_fetch_wiki_result", return_value=("", [])) as wiki,
        patch.object(ci, "_fetch_search_result", return_value=("", None)) as search,
        patch.object(ci, "_fetch_music_result", return_value="") as music,
        patch.object(ci, "_fetch_news_result", return_value="") as news,
    ):
        yield {"rag": rag, "wiki": wiki, "search": search, "music": music, "news": news}


class TestApplyRagAndWikiParallel:
    def test_returns_five_tuple(self, stub_workers):
        out = ci.apply_rag_and_wiki_parallel(msgs(), LONG, FAST)
        assert len(out) == 5

    def test_all_workers_are_submitted(self, stub_workers):
        ci.apply_rag_and_wiki_parallel(msgs(), LONG, FAST)
        for name, mock in stub_workers.items():
            mock.assert_called_once()

    def test_wiki_context_injected_into_last_user_message(self, stub_workers):
        stub_workers["wiki"].return_value = ("WIKI", ["img"])
        out, _, images, _, _ = ci.apply_rag_and_wiki_parallel(msgs(), LONG, FAST)
        assert "<wiki_context>\nWIKI\n</wiki_context>" in out[-1]["content"]
        assert images == ["img"]

    def test_wiki_images_dropped_when_block_is_empty(self, stub_workers):
        stub_workers["wiki"].return_value = ("", ["img"])
        _, _, images, _, _ = ci.apply_rag_and_wiki_parallel(msgs(), LONG, FAST)
        assert images == []

    def test_search_context_injected(self, stub_workers):
        stub_workers["search"].return_value = ("<search_context/>", MagicMock())
        out, _, _, ev, _ = ci.apply_rag_and_wiki_parallel(msgs(), LONG, SEARCHY)
        assert "<search_context/>" in out[-1]["content"]
        assert ev is not None

    def test_music_context_injected_and_returned(self, stub_workers):
        stub_workers["music"].return_value = "<music_context/>"
        out, _, _, _, music = ci.apply_rag_and_wiki_parallel(msgs(), LONG, MUSICAL)
        assert "<music_context/>" in out[-1]["content"]
        assert music == "<music_context/>"

    def test_news_context_injected(self, stub_workers):
        stub_workers["news"].return_value = "<news_context/>"
        out, *_ = ci.apply_rag_and_wiki_parallel(msgs(), LONG, FAST)
        assert "<news_context/>" in out[-1]["content"]

    def test_original_messages_not_mutated(self, stub_workers):
        stub_workers["wiki"].return_value = ("WIKI", [])
        original = msgs()
        snapshot = [dict(m) for m in original]
        ci.apply_rag_and_wiki_parallel(original, LONG, FAST)
        assert original == snapshot

    def test_injection_targets_the_last_user_turn(self, stub_workers):
        stub_workers["wiki"].return_value = ("WIKI", [])
        conv = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        out, *_ = ci.apply_rag_and_wiki_parallel(conv, LONG, FAST)
        assert "WIKI" in out[2]["content"]
        assert "WIKI" not in out[0]["content"]

    def test_worker_failure_isolated_to_its_own_source(self, stub_workers):
        stub_workers["wiki"].side_effect = RuntimeError("wiki down")
        stub_workers["news"].return_value = "<news_context/>"
        out, _, images, _, _ = ci.apply_rag_and_wiki_parallel(msgs(), LONG, FAST)
        assert images == []
        assert "<news_context/>" in out[-1]["content"]

    def test_all_workers_failing_still_returns_usable_messages(self, stub_workers):
        for m in stub_workers.values():
            m.side_effect = RuntimeError("everything is down")
        out, srcs, imgs, ev, music = ci.apply_rag_and_wiki_parallel(msgs(), LONG, FAST)
        assert out == msgs()
        assert (srcs, imgs, ev, music) == ([], [], None, "")


class TestTokenBudgetTrimming:
    def test_wiki_trimmed_for_fast_brain(self, stub_workers, monkeypatch):
        monkeypatch.setenv("SAGE_RAG_WIKI_FAST_MAX_CHARS", "50")
        stub_workers["wiki"].return_value = ("w" * 500, [])
        out, *_ = ci.apply_rag_and_wiki_parallel(msgs(), LONG, FAST)
        assert "wiki context trimmed to budget" in out[-1]["content"]

    def test_architect_gets_a_larger_wiki_budget(self, stub_workers):
        stub_workers["wiki"].return_value = ("w" * 5_000, [])
        out, *_ = ci.apply_rag_and_wiki_parallel(msgs(), LONG, ARCH)
        # 5 000 < the 16 000 ARCHITECT default → untouched.
        assert "trimmed to budget" not in out[-1]["content"]

    def test_same_block_is_trimmed_for_fast(self, stub_workers):
        stub_workers["wiki"].return_value = ("w" * 5_000, [])
        out, *_ = ci.apply_rag_and_wiki_parallel(msgs(), LONG, FAST)
        # 5 000 > the 4 000 FAST default → trimmed.
        assert "wiki context trimmed to budget" in out[-1]["content"]

    def test_search_trimmed_for_fast_brain(self, stub_workers, monkeypatch):
        monkeypatch.setenv("SAGE_SEARCH_FAST_MAX_CHARS", "40")
        stub_workers["search"].return_value = ("s" * 400, MagicMock())
        out, *_ = ci.apply_rag_and_wiki_parallel(msgs(), LONG, SEARCHY)
        assert "search context trimmed to budget" in out[-1]["content"]

    def test_block_exactly_at_budget_is_not_reported_trimmed(self, stub_workers, monkeypatch):
        """Regression: the flag was re-derived as `len >= max` after truncation."""
        monkeypatch.setenv("SAGE_RAG_WIKI_FAST_MAX_CHARS", "100")
        stub_workers["wiki"].return_value = ("w" * 100, [])
        with patch.object(ci._LOG, "info") as log:
            ci.apply_rag_and_wiki_parallel(msgs(), LONG, FAST)
        payloads = [c.args[1] for c in log.call_args_list
                    if c.args and c.args[0] == "context_injection_json %s"]
        assert payloads and '"wiki_trimmed": false' in payloads[0]

    def test_trimmed_flag_true_when_actually_trimmed(self, stub_workers, monkeypatch):
        monkeypatch.setenv("SAGE_RAG_WIKI_FAST_MAX_CHARS", "100")
        stub_workers["wiki"].return_value = ("w" * 101, [])
        with patch.object(ci._LOG, "info") as log:
            ci.apply_rag_and_wiki_parallel(msgs(), LONG, FAST)
        payloads = [c.args[1] for c in log.call_args_list
                    if c.args and c.args[0] == "context_injection_json %s"]
        assert payloads and '"wiki_trimmed": true' in payloads[0]


# ---------------------------------------------------------------------------
# Lazy singletons — thread safety
# ---------------------------------------------------------------------------

class TestLazySingletons:
    def test_ensure_rag_caches(self):
        with (
            patch.object(ci, "RagSettings") as S,
            patch.object(ci, "RagInjector") as I,
        ):
            a = ci._ensure_rag()
            b = ci._ensure_rag()
        assert a is b
        S.assert_called_once()
        I.assert_called_once()

    def test_reset_forces_reconstruction(self):
        """LazySingleton.reset() is the supported test hook."""
        with (
            patch.object(ci, "RagSettings") as S,
            patch.object(ci, "RagInjector"),
        ):
            ci._ensure_rag()
            ci._rag_pair.reset()
            ci._ensure_rag()
        assert S.call_count == 2

    def test_ensure_rag_constructs_once_under_concurrency(self):
        calls: list[int] = []

        def _slow_settings(*a, **k):
            calls.append(1)
            time.sleep(0.02)      # widen the race window
            return MagicMock()

        with (
            patch.object(ci, "RagSettings", side_effect=_slow_settings),
            patch.object(ci, "RagInjector", return_value=MagicMock()),
        ):
            threads = [threading.Thread(target=ci._ensure_rag) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(calls) == 1, f"RagSettings constructed {len(calls)} times — race"

    def test_wiki_retriever_constructed_once_under_concurrency(self):
        """A lost race here meant two cold CUDA inits of jina-clip-v2."""
        built: list[int] = []

        def _slow_build(*a, **k):
            built.append(1)
            time.sleep(0.02)
            return MagicMock()

        with (
            patch.object(ci, "_WIKI_AVAILABLE", True),
            patch.object(ci, "_ensure_rag", return_value=(MagicMock(), MagicMock())),
            patch.object(ci, "WikiRetriever", side_effect=_slow_build),
        ):
            threads = [threading.Thread(target=ci._get_wiki_retriever) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(built) == 1, f"WikiRetriever constructed {len(built)} times — race"

    def test_music_retriever_constructed_once_under_concurrency(self):
        built: list[int] = []

        def _slow_build(*a, **k):
            built.append(1)
            time.sleep(0.02)
            return MagicMock()

        with (
            patch.object(ci, "_MUSIC_AVAILABLE", True),
            patch.object(ci, "_ensure_rag", return_value=(MagicMock(), MagicMock())),
            patch.object(ci, "MusicRetriever", side_effect=_slow_build),
        ):
            threads = [threading.Thread(target=ci._get_music_retriever) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(built) == 1

    def test_returns_none_when_wiki_unavailable(self):
        with patch.object(ci, "_WIKI_AVAILABLE", False):
            assert ci._get_wiki_retriever() is None

    def test_returns_none_when_music_unavailable(self):
        with patch.object(ci, "_MUSIC_AVAILABLE", False):
            assert ci._get_music_retriever() is None
