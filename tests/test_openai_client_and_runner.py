"""
tests/test_openai_client_and_runner.py

Covers the remaining gaps:
  openai_client.py         — health_check short-circuit, model discovery,
                             blocking calls, stream abort
  router.llm_route         — the LLM-assisted routing path
  review_service/runner.py — ReviewRunner's status machine
  review_service/graph.py  — graph wiring
  review_service/nodes/web_researcher.py
"""
from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest
import requests

import openai_client as oc
from openai_client import (
    HttpTimeouts,
    LlamaServerError,
    _normalize_base_url,
    _probe_get,
    abort_active_stream,
    call_brain_blocking,
    discover_model_id,
    health_check,
    stream_chat_completions,
)

TO = HttpTimeouts(connect_s=1.0, read_s=2.0)
BASE = "http://127.0.0.1:8011"


# ---------------------------------------------------------------------------
# _normalize_base_url
# ---------------------------------------------------------------------------

class TestNormalizeBaseUrl:
    @pytest.mark.parametrize(
        "raw",
        [BASE, f"{BASE}/", f"{BASE}/v1", f"{BASE}/v1/", f"  {BASE}/v1  "],
    )
    def test_all_forms_normalise_to_the_root(self, raw):
        assert _normalize_base_url(raw) == BASE

    def test_empty_input(self):
        assert _normalize_base_url("") == ""

    def test_none_input(self):
        assert _normalize_base_url(None) == ""  # type: ignore[arg-type]

    def test_avoids_the_double_v1_bug(self):
        """'/v1/v1/models' is the classic readiness-check-never-passes bug."""
        assert not _normalize_base_url(f"{BASE}/v1").endswith("/v1")


# ---------------------------------------------------------------------------
# _probe_get / health_check
# ---------------------------------------------------------------------------

def fake_session(*, status=None, exc=None):
    """A requests.Session stand-in whose .get() returns a status or raises."""
    sess = MagicMock()
    if exc is not None:
        sess.get.side_effect = exc
    else:
        sess.get.return_value = MagicMock(status_code=status)
    return sess


class TestProbeGet:
    def test_200_is_ok(self):
        with patch.object(oc, "_session", return_value=fake_session(status=200)):
            r = _probe_get(BASE, "/health", timeouts=TO)
        assert r.ok is True
        assert r.transport_failed is False
        assert r.detail == "OK (/health)"

    def test_404_is_not_a_transport_failure(self):
        """The server answered — other paths are still worth trying."""
        with patch.object(oc, "_session", return_value=fake_session(status=404)):
            r = _probe_get(BASE, "/health", timeouts=TO)
        assert r.ok is False
        assert r.transport_failed is False
        assert r.detail == "/health=404"

    def test_503_is_not_a_transport_failure(self):
        with patch.object(oc, "_session", return_value=fake_session(status=503)):
            assert _probe_get(BASE, "/health", timeouts=TO).transport_failed is False

    @pytest.mark.parametrize(
        "exc",
        [
            requests.exceptions.ConnectionError("refused"),
            requests.exceptions.ConnectTimeout("timed out"),
            requests.exceptions.ReadTimeout("no answer"),
            requests.exceptions.Timeout("generic"),
        ],
    )
    def test_transport_errors_are_flagged(self, exc):
        with patch.object(oc, "_session", return_value=fake_session(exc=exc)):
            r = _probe_get(BASE, "/health", timeouts=TO)
        assert r.ok is False
        assert r.transport_failed is True

    def test_unexpected_error_is_not_a_transport_failure(self):
        with patch.object(oc, "_session", return_value=fake_session(exc=ValueError("odd"))):
            r = _probe_get(BASE, "/health", timeouts=TO)
        assert r.transport_failed is False
        assert "ValueError" in r.detail


class TestHealthCheck:
    def test_first_probe_success_short_circuits(self):
        sess = fake_session(status=200)
        with patch.object(oc, "_session", return_value=sess):
            ok, detail = health_check(BASE, timeouts=TO)
        assert ok is True
        assert sess.get.call_count == 1
        assert detail == "OK (/health)"

    def test_falls_through_to_later_paths_on_404(self):
        sess = MagicMock()
        sess.get.side_effect = [
            MagicMock(status_code=404),   # /health
            MagicMock(status_code=404),   # /v1/health
            MagicMock(status_code=200),   # /v1/models
        ]
        with patch.object(oc, "_session", return_value=sess):
            ok, detail = health_check(BASE, timeouts=TO)
        assert ok is True
        assert detail == "OK (/v1/models)"
        assert sess.get.call_count == 3

    def test_transport_failure_stops_after_one_probe(self):
        """
        The 2026-08-04 fix. Four sequential probes each paying the full connect
        timeout cost 8.03 s against a closed port; every path is on the same
        host:port, so one transport failure settles it.
        """
        sess = fake_session(exc=requests.exceptions.ConnectTimeout("x"))
        with patch.object(oc, "_session", return_value=sess):
            ok, detail = health_check(BASE, timeouts=TO)
        assert ok is False
        assert sess.get.call_count == 1, "did not short-circuit — the 8s bug is back"
        assert "3 probe(s) skipped" in detail

    def test_all_paths_tried_when_each_returns_http_errors(self):
        sess = MagicMock()
        sess.get.return_value = MagicMock(status_code=404)
        with patch.object(oc, "_session", return_value=sess):
            ok, detail = health_check(BASE, timeouts=TO)
        assert ok is False
        assert sess.get.call_count == len(oc._HEALTH_PATHS)
        assert "skipped" not in detail

    def test_detail_lists_every_attempted_path(self):
        sess = MagicMock()
        sess.get.return_value = MagicMock(status_code=404)
        with patch.object(oc, "_session", return_value=sess):
            _, detail = health_check(BASE, timeouts=TO)
        for path in oc._HEALTH_PATHS:
            assert path in detail

    def test_probe_order_matches_the_documented_list(self):
        assert oc._HEALTH_PATHS == ("/health", "/v1/health", "/v1/models", "/props")


# ---------------------------------------------------------------------------
# discover_model_id
# ---------------------------------------------------------------------------

class TestDiscoverModelId:
    def _sess(self, payload=None, exc=None, status=200):
        sess = MagicMock()
        if exc is not None:
            sess.get.side_effect = exc
            return sess
        resp = MagicMock(status_code=status)
        resp.json.return_value = payload
        resp.raise_for_status = MagicMock()
        sess.get.return_value = resp
        return sess

    def test_returns_the_first_model_id(self):
        with patch.object(oc, "_session",
                          return_value=self._sess({"data": [{"id": "Qwen2.5-Omni"}]})):
            assert discover_model_id(BASE, timeouts=TO) == "Qwen2.5-Omni"

    def test_empty_data_returns_none(self):
        with patch.object(oc, "_session", return_value=self._sess({"data": []})):
            assert discover_model_id(BASE, timeouts=TO) is None

    def test_missing_data_key_returns_none(self):
        with patch.object(oc, "_session", return_value=self._sess({})):
            assert discover_model_id(BASE, timeouts=TO) is None

    def test_non_string_id_returns_none(self):
        with patch.object(oc, "_session",
                          return_value=self._sess({"data": [{"id": 42}]})):
            assert discover_model_id(BASE, timeouts=TO) is None

    def test_network_error_returns_none(self):
        with patch.object(oc, "_session", return_value=self._sess(exc=OSError("down"))):
            assert discover_model_id(BASE, timeouts=TO) is None


# ---------------------------------------------------------------------------
# call_brain_blocking
# ---------------------------------------------------------------------------

class TestCallBrainBlocking:
    def _sess(self, status=200, payload=None):
        sess = MagicMock()
        resp = MagicMock(status_code=status)
        resp.json.return_value = payload or {
            "choices": [{"message": {"content": "the answer"}}]
        }
        resp.text = "error body"
        sess.post.return_value = resp
        return sess

    def test_returns_the_message_content(self):
        with patch.object(oc, "_session", return_value=self._sess()):
            out = call_brain_blocking(
                BASE, model="m", messages=[], temperature=0.7, top_p=0.8,
                max_tokens=100, timeouts=TO,
            )
        assert out == "the answer"

    def test_missing_content_returns_empty_string(self):
        sess = self._sess(payload={"choices": [{"message": {}}]})
        with patch.object(oc, "_session", return_value=sess):
            out = call_brain_blocking(
                BASE, model="m", messages=[], temperature=0.7, top_p=0.8,
                max_tokens=100, timeouts=TO,
            )
        assert out == ""

    def test_http_error_raises_llamaservererror(self):
        with patch.object(oc, "_session", return_value=self._sess(status=500)):
            with pytest.raises(LlamaServerError, match="HTTP 500"):
                call_brain_blocking(
                    BASE, model="m", messages=[], temperature=0.7, top_p=0.8,
                    max_tokens=100, timeouts=TO,
                )

    def test_sends_stream_false_and_cache_prompt(self):
        sess = self._sess()
        with patch.object(oc, "_session", return_value=sess):
            call_brain_blocking(
                BASE, model="m", messages=[], temperature=0.7, top_p=0.8,
                max_tokens=100, timeouts=TO,
            )
        payload = sess.post.call_args.kwargs["json"]
        assert payload["stream"] is False
        assert payload["cache_prompt"] is True


# ---------------------------------------------------------------------------
# stream_chat_completions
# ---------------------------------------------------------------------------

def sse(*objs) -> list[str]:
    """Render objects as SSE `data:` lines, terminated by [DONE]."""
    return [f"data: {json.dumps(o)}" for o in objs] + ["data: [DONE]"]


def streaming_session(lines, status=200):
    sess = MagicMock()
    resp = MagicMock(status_code=status)
    resp.iter_lines.return_value = iter(lines)
    resp.text = "err"
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    sess.post.return_value = resp
    return sess, resp


def delta(content=None, reasoning=None):
    d = {}
    if content is not None:
        d["content"] = content
    if reasoning is not None:
        d["reasoning_content"] = reasoning
    return {"choices": [{"delta": d}]}


class TestStreamChatCompletions:
    def _run(self, lines, **over):
        sess, _ = streaming_session(lines)
        kwargs = dict(
            model="m", messages=[], temperature=0.7, top_p=0.8,
            max_tokens=100, timeouts=TO,
        )
        kwargs.update(over)
        with patch.object(oc, "_session", return_value=sess):
            return list(stream_chat_completions(BASE, **kwargs)), sess

    def test_yields_content_chunks(self):
        out, _ = self._run(sse(delta(content="Hello "), delta(content="world")))
        assert out == ["Hello ", "world"]

    def test_wraps_reasoning_in_think_tags(self):
        out, _ = self._run(sse(delta(reasoning="pondering"), delta(content="answer")))
        assert out == ["<think>", "pondering", "</think>", "answer"]

    def test_closes_think_tag_at_done_when_still_reasoning(self):
        out, _ = self._run(sse(delta(reasoning="never finished")))
        assert out[-1] == "</think>"

    def test_ignores_malformed_sse_lines(self):
        out, _ = self._run(["data: {not json}", *sse(delta(content="ok"))])
        assert out == ["ok"]

    def test_ignores_blank_lines(self):
        out, _ = self._run(["", "   ", *sse(delta(content="ok"))])
        assert out == ["ok"]

    def test_ignores_non_data_lines(self):
        out, _ = self._run([": keepalive", "event: ping", *sse(delta(content="ok"))])
        assert out == ["ok"]

    def test_sets_utf8_encoding(self):
        """requests defaults text/* to ISO-8859-1, which garbles em-dashes."""
        sess, resp = streaming_session(sse(delta(content="—")))
        with patch.object(oc, "_session", return_value=sess):
            list(stream_chat_completions(
                BASE, model="m", messages=[], temperature=0.7, top_p=0.8,
                max_tokens=100, timeouts=TO,
            ))
        assert resp.encoding == "utf-8"

    def test_http_error_raises(self):
        sess, _ = streaming_session([], status=500)
        with patch.object(oc, "_session", return_value=sess):
            with pytest.raises(LlamaServerError, match="HTTP 500"):
                list(stream_chat_completions(
                    BASE, model="m", messages=[], temperature=0.7, top_p=0.8,
                    max_tokens=100, timeouts=TO,
                ))

    def test_thinking_budget_omitted_when_default(self):
        _, sess = self._run(sse(delta(content="x")), thinking_budget=-1)
        assert "thinking_budget" not in sess.post.call_args.kwargs["json"]

    def test_thinking_budget_sent_when_set(self):
        _, sess = self._run(sse(delta(content="x")), thinking_budget=2048)
        assert sess.post.call_args.kwargs["json"]["thinking_budget"] == 2048

    def test_thinking_budget_zero_is_sent(self):
        """0 means 'no thinking' — distinct from -1 'server default'."""
        _, sess = self._run(sse(delta(content="x")), thinking_budget=0)
        assert sess.post.call_args.kwargs["json"]["thinking_budget"] == 0

    def test_cache_prompt_is_always_enabled(self):
        _, sess = self._run(sse(delta(content="x")))
        assert sess.post.call_args.kwargs["json"]["cache_prompt"] is True

    def test_clears_the_active_stream_on_completion(self):
        self._run(sse(delta(content="x")))
        assert oc._active_stream is None


class TestAbortActiveStream:
    def test_no_active_stream_is_a_noop(self):
        with patch.object(oc, "_active_stream", None):
            abort_active_stream()

    def test_closes_the_active_response(self):
        resp = MagicMock()
        with patch.object(oc, "_active_stream", resp):
            abort_active_stream()
        resp.close.assert_called_once()

    def test_clears_the_session_cache(self):
        sess = MagicMock()
        with (
            patch.object(oc, "_active_stream", None),
            patch.dict(oc._sessions, {BASE: sess}, clear=True),
        ):
            abort_active_stream()
            assert oc._sessions == {}
        sess.close.assert_called_once()

    def test_survives_a_close_failure(self):
        resp = MagicMock()
        resp.close.side_effect = OSError("already closed")
        with patch.object(oc, "_active_stream", resp):
            abort_active_stream()   # must not raise


class TestSessionCache:
    def test_returns_the_same_session_per_host(self):
        with patch.dict(oc._sessions, {}, clear=True):
            assert oc._session(BASE) is oc._session(BASE)

    def test_different_hosts_get_different_sessions(self):
        with patch.dict(oc._sessions, {}, clear=True):
            assert oc._session(BASE) is not oc._session("http://127.0.0.1:8012")


# ---------------------------------------------------------------------------
# router.llm_route
# ---------------------------------------------------------------------------

import router as rt
from router import llm_route


class TestLlmRoute:
    def _route(self, label, **over):
        with patch.object(rt, "stream_chat_completions", return_value=iter([label])):
            kwargs = dict(
                user_text="an ambiguous question", fast_base_url=BASE,
                model_id="m", timeouts=TO,
            )
            kwargs.update(over)
            return llm_route(**kwargs)

    def test_force_architect_skips_the_call(self):
        with patch.object(rt, "stream_chat_completions") as s:
            d = llm_route("q", BASE, "m", TO, force_architect=True)
        s.assert_not_called()
        assert d.brain == "ARCHITECT"
        assert d.reasons == ["force_architect"]

    def test_empty_input_skips_the_call(self):
        with patch.object(rt, "stream_chat_completions") as s:
            d = llm_route("", BASE, "m", TO)
        s.assert_not_called()
        assert d.brain == "FAST"

    @pytest.mark.parametrize(
        "label,brain,needs_search",
        [
            ("FAST", "FAST", False),
            ("ARCHITECT", "ARCHITECT", False),
            ("SEARCH", "FAST", True),
            ("ARCHITECT+SEARCH", "ARCHITECT", True),
        ],
    )
    def test_label_mapping(self, label, brain, needs_search):
        d = self._route(label)
        assert d.brain == brain
        assert d.needs_search is needs_search

    def test_label_is_case_and_whitespace_insensitive(self):
        assert self._route("  architect  ").brain == "ARCHITECT"

    def test_search_labels_get_categories(self):
        assert self._route("SEARCH").search_categories

    def test_score_encodes_the_brain(self):
        assert self._route("ARCHITECT").score == 999
        assert self._route("FAST").score == 0

    def test_reasons_mark_llm_classification(self):
        assert self._route("FAST").reasons == ["llm_classification"]

    def test_music_detection_still_runs(self):
        d = self._route("FAST", user_text="find me a song about rain")
        assert d.needs_music is True

    def test_failure_falls_back_to_the_heuristic(self):
        with patch.object(rt, "stream_chat_completions", side_effect=OSError("down")):
            d = llm_route("please refactor this python code", BASE, "m", TO)
        assert d.brain == "ARCHITECT"
        assert d.reasons != ["llm_classification"]

    def test_caps_the_text_sent_to_the_classifier(self):
        captured = {}

        def _stream(**kw):
            captured.update(kw)
            return iter(["FAST"])

        with patch.object(rt, "stream_chat_completions", _stream):
            llm_route("x" * 5000, BASE, "m", TO)
        assert len(captured["messages"][-1]["content"]) == rt.LLM_CAP_CHARS

    def test_uses_deterministic_sampling(self):
        captured = {}

        def _stream(**kw):
            captured.update(kw)
            return iter(["FAST"])

        with patch.object(rt, "stream_chat_completions", _stream):
            llm_route("q", BASE, "m", TO)
        assert captured["temperature"] == 0.0
        assert captured["max_tokens"] == 10

    def test_emits_the_structured_route_json_log(self):
        """
        The 2026-08-04 fix: this path returned without calling _log_decision(),
        so route_json was missing for exactly the ambiguous turns.
        """
        with (
            patch.object(rt, "stream_chat_completions", return_value=iter(["FAST"])),
            patch.object(rt, "_log_decision") as log,
        ):
            llm_route("q", BASE, "m", TO)
        log.assert_called_once()


# ---------------------------------------------------------------------------
# review_service.runner
# ---------------------------------------------------------------------------

import review_service.runner as rr
from review_service.runner import ReviewRunner


@pytest.fixture
def runner():
    return ReviewRunner()


class TestReviewRunnerState:
    def test_starts_idle(self, runner):
        assert runner.status == "idle"
        assert runner.thread_id is None
        assert runner.output_paths == []
        assert runner.error is None

    @pytest.mark.parametrize(
        "status,busy",
        [("idle", False), ("running", True), ("awaiting_approval", True),
         ("done", False), ("rejected", False), ("error", False)],
    )
    def test_is_busy(self, runner, status, busy):
        runner.status = status
        assert runner.is_busy() is busy

    def test_reset_clears_everything(self, runner):
        runner.status = "done"
        runner.thread_id = "t1"
        runner.output_paths = ["/a.md"]
        runner.error = "boom"
        runner.interrupt_payload = {"synthesis": "s"}
        runner.reset()
        assert runner.status == "idle"
        assert runner.thread_id is None
        assert runner.output_paths == []
        assert runner.error is None
        assert runner.interrupt_payload is None


class TestReviewRunnerStart:
    def test_thread_id_encodes_the_mode(self, runner):
        with patch.object(rr.threading, "Thread"):
            tid = runner.start("full")
        assert tid.startswith("review-")
        assert tid.endswith("-full")

    def test_sets_running_status(self, runner):
        with patch.object(rr.threading, "Thread"):
            runner.start("staged")
        assert runner.status == "running"

    def test_launches_a_daemon_thread(self, runner):
        with patch.object(rr.threading, "Thread") as T:
            runner.start("full")
        assert T.call_args.kwargs["daemon"] is True
        T.return_value.start.assert_called_once()

    def test_clears_previous_run_state(self, runner):
        runner.output_paths = ["/old.md"]
        runner.error = "old error"
        with patch.object(rr.threading, "Thread"):
            runner.start("full")
        assert runner.output_paths == []
        assert runner.error is None

    def test_target_is_passed_into_the_initial_state(self, runner):
        with patch.object(rr.threading, "Thread") as T:
            runner.start("file", "chat_service.py")
        state = T.call_args.kwargs["args"][0]
        assert state["mode"] == "file"
        assert state["target"] == "chat_service.py"


class TestReviewRunnerResume:
    def test_sets_running_and_launches_a_thread(self, runner):
        runner.thread_id = "t1"
        with patch.object(rr.threading, "Thread") as T:
            runner.resume(True)
        assert runner.status == "running"
        T.return_value.start.assert_called_once()

    def test_approved_flag_is_forwarded(self, runner):
        runner.thread_id = "t1"
        with patch.object(rr.threading, "Thread") as T:
            runner.resume(False)
        assert T.call_args.kwargs["args"] == (False,)


class TestReviewRunnerSyncWrappers:
    def test_run_sync_records_errors(self, runner):
        async def _boom(*a, **kw):
            raise RuntimeError("graph exploded")

        with patch.object(runner, "_run_async", _boom):
            runner._run_sync(rr.default_state("full"), "t1")
        assert runner.status == "error"
        assert "graph exploded" in (runner.error or "")

    def test_resume_sync_records_errors(self, runner):
        async def _boom(*a, **kw):
            raise RuntimeError("resume exploded")

        with patch.object(runner, "_resume_async", _boom):
            runner._resume_sync(True)
        assert runner.status == "error"
        assert "resume exploded" in (runner.error or "")

    def test_run_sync_succeeds_quietly(self, runner):
        async def _ok(*a, **kw):
            return None

        with patch.object(runner, "_run_async", _ok):
            runner._run_sync(rr.default_state("full"), "t1")
        assert runner.status != "error"


# ---------------------------------------------------------------------------
# review_service.graph
# ---------------------------------------------------------------------------

class TestBuildReviewGraph:
    def test_compiles_with_a_checkpointer(self):
        import review_service.graph as g
        with (
            patch.object(g, "ChatOpenAI") as LLM,
            patch.object(g, "StateGraph") as SG,
        ):
            LLM.return_value = MagicMock()
            checkpointer = MagicMock()
            g.build_review_graph(checkpointer)
        SG.return_value.compile.assert_called_once_with(checkpointer=checkpointer)

    def test_registers_every_pipeline_node(self):
        import review_service.graph as g
        with (
            patch.object(g, "ChatOpenAI", return_value=MagicMock()),
            patch.object(g, "StateGraph") as SG,
        ):
            g.build_review_graph(MagicMock())
        added = {c.args[0] for c in SG.return_value.add_node.call_args_list}
        assert added == {
            "scope_collector", "subprocess_checks", "web_researcher",
            "code_quality_reviewer", "architect_reviewer", "flags_sanity",
            "docs_drift", "synthesizer", "human_gate", "output_writer",
        }

    def test_entry_point_is_scope_collector(self):
        import review_service.graph as g
        with (
            patch.object(g, "ChatOpenAI", return_value=MagicMock()),
            patch.object(g, "StateGraph") as SG,
        ):
            g.build_review_graph(MagicMock())
        SG.return_value.set_entry_point.assert_called_once_with("scope_collector")

    def test_human_gate_uses_a_conditional_edge(self):
        """output_writer must be unreachable without approval."""
        import review_service.graph as g
        with (
            patch.object(g, "ChatOpenAI", return_value=MagicMock()),
            patch.object(g, "StateGraph") as SG,
        ):
            g.build_review_graph(MagicMock())
        call = SG.return_value.add_conditional_edges.call_args
        assert call.args[0] == "human_gate"
        router_fn = call.args[1]
        assert router_fn({"approved": True}) == "output_writer"
        assert router_fn({"approved": False}) != "output_writer"

    def test_architect_llm_targets_port_8012(self):
        import review_service.graph as g
        with (
            patch.object(g, "ChatOpenAI") as LLM,
            patch.object(g, "StateGraph"),
        ):
            g.build_review_graph(MagicMock())
        assert ":8012" in LLM.call_args.kwargs["base_url"]

    def test_architect_llm_is_non_streaming_and_low_temperature(self):
        import review_service.graph as g
        with (
            patch.object(g, "ChatOpenAI") as LLM,
            patch.object(g, "StateGraph"),
        ):
            g.build_review_graph(MagicMock())
        assert LLM.call_args.kwargs["streaming"] is False
        assert LLM.call_args.kwargs["temperature"] == 0.2


# ---------------------------------------------------------------------------
# review_service.nodes.web_researcher
# ---------------------------------------------------------------------------

import review_service.nodes.web_researcher as wr
from review_service.state import default_state


from search.models import SearchEvidence, WebResult


def _real_result(title="Blackwell perf", url="https://x", snippet="details"):
    """A genuine WebResult — not a MagicMock, so attribute typos are caught."""
    return WebResult(
        title=title, url=url, snippet=snippet, source_engine="ddg",
        category="it", score=1.0, published_date=None,
    )


def _orchestrator(results):
    orch = MagicMock()
    orch.search.return_value = SearchEvidence(
        query="q", results=tuple(results),
        fetched_at="2026-08-04T12:00:00+00:00", categories_queried=("it",),
    )
    return orch


def _state_with_scope():
    s = default_state("full")
    s["changed_files"] = ["chat_service.py"]
    s["brains_yaml"] = "fast:\n  port: 8011"
    return s


class TestWebResearcher:
    def _run(self, coro):
        import asyncio
        return asyncio.run(coro)

    def test_returns_the_web_research_key(self):
        with patch("search.search_orchestrator.get_orchestrator",
                   side_effect=RuntimeError("no searxng")):
            out = self._run(wr.web_researcher_node(_state_with_scope()))
        assert "web_research" in out

    def test_search_failure_is_not_fatal(self):
        with patch("search.search_orchestrator.get_orchestrator",
                   side_effect=RuntimeError("down")):
            out = self._run(wr.web_researcher_node(_state_with_scope()))
        assert isinstance(out["web_research"], str)

    def test_no_queries_generated_returns_empty(self):
        out = self._run(wr.web_researcher_node(default_state("full")))
        assert out["web_research"] == ""

    def test_empty_results_produce_an_empty_block(self):
        with patch("search.search_orchestrator.get_orchestrator",
                   return_value=_orchestrator([])):
            out = self._run(wr.web_researcher_node(_state_with_scope()))
        assert isinstance(out["web_research"], str)

    # ── KNOWN DEFECT ─────────────────────────────────────────────────────
    # _searxng_search() formats each hit with `r.content`, but WebResult has
    # no `content` field — it is `snippet` (verified against the dataclass).
    # The AttributeError is swallowed by the function's own
    # `except Exception: return ""`, so EVERY successful SearXNG search is
    # silently discarded and web_research is always "".
    #
    # Net effect: the web_researcher node has never contributed anything to a
    # review, and the failure is invisible because the node is designed to
    # degrade quietly when SearXNG is down — it looks like SearXNG is always
    # down. Fix is one word: r.content -> r.snippet.
    #
    # Outside the bugs-1-7 scope agreed for this pass; xfail(strict) so the
    # build flags it the moment it IS fixed.

    _DEFECT = "known defect: _searxng_search reads r.content; WebResult has .snippet"

    @pytest.mark.xfail(strict=True, reason=_DEFECT)
    def test_results_should_be_formatted_into_the_block(self):
        with patch("search.search_orchestrator.get_orchestrator",
                   return_value=_orchestrator([_real_result()])):
            out = self._run(wr.web_researcher_node(_state_with_scope()))
        assert "Blackwell perf" in out["web_research"]

    def test_results_are_currently_always_discarded(self):
        """Pins the defect so a fix must update this test too."""
        with patch("search.search_orchestrator.get_orchestrator",
                   return_value=_orchestrator([_real_result()])):
            out = self._run(wr.web_researcher_node(_state_with_scope()))
        assert out["web_research"] == "", (
            "web_researcher returned content — the r.content defect is fixed; "
            "delete this test and un-xfail test_results_should_be_formatted_into_the_block"
        )

    def test_webresult_has_snippet_not_content(self):
        """The direct statement of the mismatch."""
        r = _real_result()
        assert hasattr(r, "snippet")
        assert not hasattr(r, "content")


class TestWebResearcherQueryBuilding:
    def test_no_files_yields_no_queries(self):
        assert wr._build_queries([], "") == []

    def test_queries_are_capped(self):
        files = [f"module_{i}.py" for i in range(50)]
        assert len(wr._build_queries(files, "x" * 100)) <= wr._MAX_QUERIES

    def test_primary_module_prefers_known_names(self):
        assert wr._primary_module(["a.py", "chat_service.py"]) == "chat service"

    def test_primary_module_falls_back_to_the_first_python_file(self):
        assert wr._primary_module(["notes.md", "helper_thing.py"]) == "helper thing"

    def test_primary_module_ignores_voice_prefixed_files(self):
        assert wr._primary_module(["[voice] main.py"]) == ""

    def test_primary_module_with_no_python_files(self):
        assert wr._primary_module(["README.md"]) == ""
