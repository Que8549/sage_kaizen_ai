"""
tests/test_chat_service.py

Unit tests for chat_service.py — the single-turn lifecycle.

Focus areas:
  * decide_route()'s documented priority order (documents > media > manual >
    heuristic > LLM tie-break), which is the part most likely to regress.
  * _build_multimodal_content()'s OpenAI content-part serialisation.
  * stream_response()'s thinking-budget resolution.
  * prepare_messages()'s graceful degradation when MemoryService is missing.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import chat_service as cs
from chat_service import (
    ChatService,
    MediaAttachment,
    TurnConfig,
    _build_multimodal_content,
    last_chat_activity_ts,
    record_chat_activity,
)
from document_parser import DocumentAttachment
from openai_client import HttpTimeouts
from prompt_library import TemplateKey
from router import RouteDecision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_memory():
    """Disable the memory singleton for every test unless explicitly re-enabled."""
    with patch.object(cs, "_MEMORY_SVC", None), patch.object(cs, "_MEMORY_DISABLED", True):
        yield


@pytest.fixture
def session():
    s = MagicMock()
    s.q5_url = "http://127.0.0.1:8011"
    s.q6_url = "http://127.0.0.1:8012"
    s.q5_model_id = "FAST"
    s.q6_model_id = "ARCH"
    s.summarizer_url = ""
    s.summarizer_model_id = ""
    s.health_q5.return_value = (False, "down")
    s.url_for_brain.side_effect = lambda b: s.q6_url if b == "ARCHITECT" else s.q5_url
    s.model_id_for_brain.side_effect = lambda b: s.q6_model_id if b == "ARCHITECT" else s.q5_model_id
    return s


@pytest.fixture
def service(session):
    return ChatService(session, "SYSTEM", HttpTimeouts(connect_s=1.0, read_s=5.0))


def make_cfg(**over) -> TurnConfig:
    base = dict(
        deep_mode=False, auto_escalate=True, auto_templates=False,
        override_templates=(),
        temperature_q5=0.7, temperature_q6=0.6,
        top_p_q5=0.8, top_p_q6=0.95,
        top_k_q5=40, top_k_q6=20,
        min_p_q5=0.05, min_p_q6=0.0,
        max_tokens_q5=1024, max_tokens_q6=4096,
    )
    base.update(over)
    return TurnConfig(**base)  # type: ignore[arg-type]


def img(label="a.png") -> MediaAttachment:
    return MediaAttachment(kind="image", data_b64="Zm9v", mime_type="image/png", label=label)


def doc(name="a.py", chars=10) -> DocumentAttachment:
    return DocumentAttachment(
        filename=name, content="x" * chars, doc_type="python",
        char_count=chars, truncated=False,
    )


# ---------------------------------------------------------------------------
# Chat-activity timestamp (shared with the news summarizers' off-peak guard)
# ---------------------------------------------------------------------------

class TestChatActivity:
    def test_records_and_reads_back(self):
        record_chat_activity()
        assert last_chat_activity_ts() > 0

    def test_timestamp_advances(self):
        record_chat_activity()
        first = last_chat_activity_ts()
        record_chat_activity()
        assert last_chat_activity_ts() >= first


# ---------------------------------------------------------------------------
# decide_route — documented priority order
# ---------------------------------------------------------------------------

class TestDecideRouteEmpty:
    def test_empty_everything_is_fast(self, service):
        d = service.decide_route("", make_cfg())
        assert d.brain == "FAST"
        assert d.reasons == ["empty_input"]

    def test_empty_text_with_media_is_not_treated_as_empty(self, service):
        d = service.decide_route("", make_cfg(media_attachments=(img(),)))
        assert d.brain == "FAST"
        assert d.modality == "image"

    def test_empty_text_with_document_is_not_treated_as_empty(self, service):
        d = service.decide_route("", make_cfg(document_attachments=(doc(),)))
        assert d.brain == "ARCHITECT"


class TestDecideRouteDocuments:
    def test_documents_force_architect(self, service):
        d = service.decide_route("summarize", make_cfg(document_attachments=(doc(),)))
        assert d.brain == "ARCHITECT"
        assert d.score == 999

    def test_documents_beat_media(self, service):
        """Documents are checked first, so a mixed turn goes to ARCHITECT."""
        cfg = make_cfg(document_attachments=(doc(),), media_attachments=(img(),))
        assert service.decide_route("hi", cfg).brain == "ARCHITECT"

    def test_documents_beat_deep_mode_off(self, service):
        cfg = make_cfg(document_attachments=(doc(),), deep_mode=False, auto_escalate=False)
        assert service.decide_route("hi", cfg).brain == "ARCHITECT"

    def test_reasons_record_count_and_chars(self, service):
        cfg = make_cfg(document_attachments=(doc("a.py", 10), doc("b.py", 25)))
        d = service.decide_route("x", cfg)
        assert "document_upload:2_files" in d.reasons
        assert "doc_chars:35" in d.reasons

    def test_modality_lists_sorted_unique_types(self, service):
        a = DocumentAttachment("a.py", "x", "python", 1)
        b = DocumentAttachment("b.md", "x", "markdown", 1)
        c = DocumentAttachment("c.py", "x", "python", 1)
        d = service.decide_route("x", make_cfg(document_attachments=(a, b, c)))
        assert d.modality == "document:markdown,python"


class TestDecideRouteMultimodal:
    @pytest.mark.parametrize(
        "kinds,expected_modality",
        [
            (("image",), "image"),
            (("video_frame",), "video"),
            (("audio",), "audio"),
            (("image", "audio"), "multimodal"),
            (("video_frame", "audio"), "multimodal"),
            (("image", "video_frame"), "video"),
        ],
    )
    def test_modality_classification(self, service, kinds, expected_modality):
        atts = tuple(
            MediaAttachment(kind=k, data_b64="x", mime_type="application/octet-stream")
            for k in kinds
        )
        d = service.decide_route("look", make_cfg(media_attachments=atts))
        assert d.modality == expected_modality

    def test_all_media_routes_to_fast(self, service):
        """ARCHITECT's mmproj is disabled; every modality must land on FAST."""
        for kind in ("image", "audio", "video_frame"):
            att = MediaAttachment(kind=kind, data_b64="x", mime_type="x/y")
            d = service.decide_route("q", make_cfg(media_attachments=(att,)))
            assert d.brain == "FAST", kind
            assert "fast_mmproj" in d.reasons

    def test_media_beats_deep_mode(self, service):
        cfg = make_cfg(media_attachments=(img(),), deep_mode=True)
        assert service.decide_route("q", cfg).brain == "FAST"


class TestDecideRouteManual:
    def test_auto_escalate_off_respects_deep_mode_on(self, service):
        d = service.decide_route("q", make_cfg(auto_escalate=False, deep_mode=True))
        assert d.brain == "ARCHITECT"
        assert d.reasons == ["manual_deep_mode"]

    def test_auto_escalate_off_respects_deep_mode_off(self, service):
        d = service.decide_route("q", make_cfg(auto_escalate=False, deep_mode=False))
        assert d.brain == "FAST"
        assert d.reasons == ["manual_fast_mode"]

    def test_deep_mode_short_circuits_before_heuristic(self, service):
        with patch("chat_service.heuristic_route") as h:
            d = service.decide_route("q", make_cfg(deep_mode=True))
        h.assert_not_called()
        assert d.brain == "ARCHITECT"


class TestDecideRouteHeuristicAndLlm:
    def test_unambiguous_heuristic_skips_llm(self, service, session):
        with patch("chat_service._router.llm_route") as llm:
            d = service.decide_route("hello", make_cfg())
        llm.assert_not_called()
        session.health_q5.assert_not_called()
        assert d.brain == "FAST"

    def test_clear_architect_skips_llm(self, service):
        with patch("chat_service._router.llm_route") as llm:
            d = service.decide_route("please refactor this python code", make_cfg())
        llm.assert_not_called()
        assert d.brain == "ARCHITECT"

    def test_ambiguous_consults_llm_when_fast_brain_is_up(self, service, session):
        session.health_q5.return_value = (True, "ok")
        expected = RouteDecision(brain="ARCHITECT", reasons=["llm_classification"], score=999)
        with (
            patch("chat_service.heuristic_route",
                  return_value=RouteDecision(brain="FAST", reasons=[], score=2)),
            patch("chat_service._router.llm_route", return_value=expected) as llm,
        ):
            d = service.decide_route("some ambiguous text", make_cfg())
        llm.assert_called_once()
        assert d is expected

    def test_ambiguous_skips_llm_when_fast_brain_is_down(self, service, session):
        session.health_q5.return_value = (False, "down")
        heur = RouteDecision(brain="FAST", reasons=[], score=2)
        with (
            patch("chat_service.heuristic_route", return_value=heur),
            patch("chat_service._router.llm_route") as llm,
        ):
            d = service.decide_route("ambiguous", make_cfg())
        llm.assert_not_called()
        assert d is heur

    def test_llm_failure_falls_back_to_heuristic(self, service, session):
        session.health_q5.return_value = (True, "ok")
        heur = RouteDecision(brain="FAST", reasons=[], score=1)
        with (
            patch("chat_service.heuristic_route", return_value=heur),
            patch("chat_service._router.llm_route", side_effect=RuntimeError("boom")),
        ):
            d = service.decide_route("ambiguous", make_cfg())
        assert d is heur


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------

class TestSelectTemplates:
    def test_override_wins(self, service):
        override = (TemplateKey.TEACHING_TUTORING,)
        out = service.select_templates("anything", make_cfg(override_templates=override))
        assert out == override

    def test_auto_templates_off_returns_empty(self, service):
        assert service.select_templates("teach me", make_cfg(auto_templates=False)) == ()

    def test_auto_templates_always_include_base_pair(self, service):
        out = service.select_templates("hello", make_cfg(auto_templates=True))
        assert out[0] == TemplateKey.UNIVERSAL_DEPTH_ANCHOR
        assert out[1] == TemplateKey.AUTO_ADAPTIVE_META

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("please teach me algebra", TemplateKey.TEACHING_TUTORING),
            ("the history of rome", TemplateKey.STRUCTURED_KNOWLEDGE),
            ("a question about philosophy", TemplateKey.PHILOSOPHY_DEEP_THINKING),
        ],
    )
    def test_hint_driven_templates(self, service, text, expected):
        out = service.select_templates(text, make_cfg(auto_templates=True))
        assert expected in out

    def test_result_is_deduplicated_and_ordered(self, service):
        out = service.select_templates(
            "teach me the history of philosophy", make_cfg(auto_templates=True)
        )
        assert len(out) == len(set(out))
        assert out[0] == TemplateKey.UNIVERSAL_DEPTH_ANCHOR

    def test_none_text_is_safe(self, service):
        assert service.select_templates("", make_cfg(auto_templates=True))


# ---------------------------------------------------------------------------
# _build_multimodal_content
# ---------------------------------------------------------------------------

class TestBuildMultimodalContent:
    def test_image_becomes_data_uri(self):
        parts = _build_multimodal_content("what is this", (img(),))
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"] == "data:image/png;base64,Zm9v"

    def test_video_frame_serialises_as_image(self):
        att = MediaAttachment(kind="video_frame", data_b64="QQ==", mime_type="image/jpeg")
        parts = _build_multimodal_content("q", (att,))
        assert parts[0]["type"] == "image_url"

    def test_audio_format_from_mime(self):
        att = MediaAttachment(kind="audio", data_b64="QQ==", mime_type="audio/wav")
        parts = _build_multimodal_content("q", (att,))
        assert parts[0] == {"type": "input_audio", "input_audio": {"data": "QQ==", "format": "wav"}}

    def test_audio_mpeg_is_normalised_to_mp3(self):
        att = MediaAttachment(kind="audio", data_b64="QQ==", mime_type="audio/mpeg")
        parts = _build_multimodal_content("q", (att,))
        assert parts[0]["input_audio"]["format"] == "mp3"

    def test_unknown_kind_is_skipped_not_fatal(self):
        att = MediaAttachment(kind="hologram", data_b64="QQ==", mime_type="x/y")
        parts = _build_multimodal_content("q", (att,))
        assert len(parts) == 1 and parts[0]["type"] == "text"

    def test_text_part_is_last(self):
        parts = _build_multimodal_content("describe", (img(), img()))
        assert parts[-1] == {"type": "text", "text": "describe"}

    def test_blank_text_emits_no_text_part(self):
        parts = _build_multimodal_content("   ", (img(),))
        assert all(p["type"] != "text" for p in parts)

    def test_doc_context_is_prepended_to_text_part(self):
        parts = _build_multimodal_content("question", (img(),), doc_context="<document/>")
        assert parts[-1]["text"] == "<document/>\n\nquestion"

    def test_doc_context_alone_still_emits_text_part(self):
        parts = _build_multimodal_content("", (img(),), doc_context="<document/>")
        assert parts[-1]["text"] == "<document/>"


# ---------------------------------------------------------------------------
# prepare_messages
# ---------------------------------------------------------------------------

@pytest.fixture
def no_rag():
    """apply_rag_and_wiki_parallel is exercised in its own test module."""
    with patch("chat_service.apply_rag_and_wiki_parallel") as m:
        m.side_effect = lambda messages, *a, **kw: (messages, [], [], None, "")
        yield m


class TestPrepareMessages:
    def test_system_message_first(self, service, no_rag):
        msgs, *_ = service.prepare_messages(
            "hi", [], RouteDecision(brain="FAST", reasons=[], score=0), (),
        )
        assert msgs[0]["role"] == "system"
        assert "SYSTEM" in msgs[0]["content"]

    def test_user_message_last(self, service, no_rag):
        msgs, *_ = service.prepare_messages(
            "hello there", [], RouteDecision(brain="FAST", reasons=[], score=0), (),
        )
        assert msgs[-1] == {"role": "user", "content": "hello there"}

    def test_history_excludes_final_entry(self, service, no_rag):
        """The caller's history already ends with this turn's user message."""
        history = [
            {"role": "user", "content": "old q"},
            {"role": "assistant", "content": "old a"},
            {"role": "user", "content": "new q"},
        ]
        msgs, *_ = service.prepare_messages(
            "new q", history, RouteDecision(brain="FAST", reasons=[], score=0), (),
        )
        contents = [m["content"] for m in msgs]
        assert contents.count("new q") == 1
        assert "old q" in contents and "old a" in contents

    def test_architect_uses_architect_core_prompt(self, service, no_rag):
        with patch("chat_service.build_system_only", return_value="S") as b:
            service.prepare_messages(
                "x", [], RouteDecision(brain="ARCHITECT", reasons=[], score=9), (),
            )
        assert b.call_args.kwargs["core_prompt"] is cs.sage_architect_core

    def test_fast_uses_fast_core_prompt(self, service, no_rag):
        with patch("chat_service.build_system_only", return_value="S") as b:
            service.prepare_messages(
                "x", [], RouteDecision(brain="FAST", reasons=[], score=0), (),
            )
        assert b.call_args.kwargs["core_prompt"] is cs.sage_fast_core

    def test_documents_prepended_to_plain_text_query(self, service, no_rag):
        msgs, *_ = service.prepare_messages(
            "explain", [], RouteDecision(brain="ARCHITECT", reasons=[], score=9), (),
            document_attachments=(doc("a.py", 5),),
        )
        content = msgs[-1]["content"]
        assert content.startswith("<document ")
        assert content.endswith("explain")

    def test_media_produces_content_part_list(self, service, no_rag):
        msgs, *_ = service.prepare_messages(
            "look", [], RouteDecision(brain="FAST", reasons=[], score=0), (),
            media_attachments=(img(),),
        )
        assert isinstance(msgs[-1]["content"], list)

    def test_empty_system_content_omits_system_message(self, service, no_rag):
        with patch("chat_service.build_system_only", return_value=""):
            msgs, *_ = service.prepare_messages(
                "x", [], RouteDecision(brain="FAST", reasons=[], score=0), (),
            )
        assert all(m["role"] != "system" for m in msgs)

    def test_returns_five_tuple(self, service, no_rag):
        out = service.prepare_messages(
            "x", [], RouteDecision(brain="FAST", reasons=[], score=0), (),
        )
        assert len(out) == 5

    def test_memory_failure_is_non_fatal(self, service, no_rag):
        svc = MagicMock()
        svc.get_memory_bundle.side_effect = RuntimeError("db gone")
        with patch("chat_service._get_memory", return_value=svc):
            msgs, *_ = service.prepare_messages(
                "hi", [], RouteDecision(brain="FAST", reasons=[], score=0), (),
            )
        assert msgs[-1]["content"] == "hi"

    def test_memory_permission_error_disables_memory_permanently(self, service, no_rag):
        svc = MagicMock()
        svc.get_memory_bundle.side_effect = RuntimeError("permission denied for schema")
        with (
            patch("chat_service._get_memory", return_value=svc),
            patch.object(cs, "_MEMORY_DISABLED", False),
        ):
            service.prepare_messages(
                "hi", [], RouteDecision(brain="FAST", reasons=[], score=0), (),
            )
            assert cs._MEMORY_DISABLED is True

    def test_memory_bundle_is_prepended_to_system(self, service, no_rag):
        svc = MagicMock()
        bundle = MagicMock(total_items=2, estimated_tokens=50)
        svc.get_memory_bundle.return_value = bundle
        with (
            patch("chat_service._get_memory", return_value=svc),
            patch("chat_service.format_bundle_prompt", return_value="<memory/>"),
        ):
            msgs, *_ = service.prepare_messages(
                "hi", [], RouteDecision(brain="FAST", reasons=[], score=0), (),
            )
        assert msgs[0]["content"].startswith("<memory/>")


# ---------------------------------------------------------------------------
# stream_response — thinking budget resolution
# ---------------------------------------------------------------------------

class TestStreamResponse:
    def _run(self, service, decision, cfg):
        with patch("chat_service.stream_chat_completions", return_value=iter(["a", "b"])) as m:
            chunks = list(service.stream_response([], decision, cfg))
        return chunks, m.call_args.kwargs

    def test_yields_chunks(self, service):
        chunks, _ = self._run(
            service, RouteDecision(brain="FAST", reasons=[], score=0), make_cfg()
        )
        assert chunks == ["a", "b"]

    def test_fast_uses_q5_sampling_params(self, service):
        _, kw = self._run(
            service, RouteDecision(brain="FAST", reasons=[], score=0), make_cfg()
        )
        assert kw["temperature"] == 0.7
        assert kw["top_k"] == 40
        assert kw["max_tokens"] == 1024

    def test_architect_uses_q6_sampling_params(self, service):
        _, kw = self._run(
            service, RouteDecision(brain="ARCHITECT", reasons=[], score=9), make_cfg()
        )
        assert kw["temperature"] == 0.6
        assert kw["top_k"] == 20
        assert kw["max_tokens"] == 4096

    def test_fast_never_sends_a_thinking_budget(self, service):
        _, kw = self._run(
            service, RouteDecision(brain="FAST", reasons=["creative:write a story"], score=0),
            make_cfg(thinking_budget=512),
        )
        assert kw["thinking_budget"] == -1

    def test_architect_creative_autocaps_thinking(self, service):
        _, kw = self._run(
            service,
            RouteDecision(brain="ARCHITECT", reasons=["creative:write a poem"], score=3),
            make_cfg(thinking_budget=-1),
        )
        assert kw["thinking_budget"] == cs._CREATIVE_THINKING_CAP

    def test_explicit_budget_overrides_creative_autocap(self, service):
        _, kw = self._run(
            service,
            RouteDecision(brain="ARCHITECT", reasons=["creative:write a poem"], score=3),
            make_cfg(thinking_budget=99),
        )
        assert kw["thinking_budget"] == 99

    def test_architect_non_creative_stays_unlimited(self, service):
        _, kw = self._run(
            service, RouteDecision(brain="ARCHITECT", reasons=["code:python"], score=3),
            make_cfg(thinking_budget=-1),
        )
        assert kw["thinking_budget"] == -1

    def test_records_chat_activity(self, service):
        with patch("chat_service.record_chat_activity") as rec:
            self._run(service, RouteDecision(brain="FAST", reasons=[], score=0), make_cfg())
        rec.assert_called_once()


# ---------------------------------------------------------------------------
# write_episode_background
# ---------------------------------------------------------------------------

class TestWriteEpisodeBackground:
    def test_noop_when_memory_unavailable(self):
        with (
            patch("chat_service._get_memory", return_value=None),
            patch("chat_service.threading.Thread") as T,
        ):
            ChatService.write_episode_background(
                "u", "a", RouteDecision(brain="FAST", reasons=[], score=0)
            )
        T.assert_not_called()

    def test_spawns_daemon_thread_when_memory_available(self):
        with (
            patch("chat_service._get_memory", return_value=MagicMock()),
            patch("chat_service.threading.Thread") as T,
        ):
            ChatService.write_episode_background(
                "u", "a", RouteDecision(brain="FAST", reasons=[], score=0)
            )
        assert T.call_args.kwargs["daemon"] is True
        T.return_value.start.assert_called_once()

    @staticmethod
    def _capturing_thread(captured: dict):
        """Replacement for threading.Thread that records target instead of running it."""
        def _factory(target=None, **kw):
            captured["target"] = target
            return MagicMock()
        return _factory

    def test_background_write_swallows_errors(self):
        captured: dict = {}
        with (
            patch("chat_service._get_memory", return_value=MagicMock()),
            patch("chat_service.threading.Thread", self._capturing_thread(captured)),
            patch("chat_service.write_episode", side_effect=RuntimeError("nope")),
        ):
            ChatService.write_episode_background(
                "u", "a", RouteDecision(brain="FAST", reasons=[], score=0)
            )
        captured["target"]()   # must not raise

    def test_background_write_passes_full_texts(self):
        captured: dict = {}
        with (
            patch("chat_service._get_memory", return_value=MagicMock()),
            patch("chat_service.threading.Thread", self._capturing_thread(captured)),
            patch("chat_service.write_episode", return_value="id-1") as we,
        ):
            ChatService.write_episode_background(
                "user text", "assistant text",
                RouteDecision(brain="FAST", reasons=[], score=0),
                session_id="s1", user_id="tester",
            )
            # Run the captured thread body while write_episode is still patched.
            captured["target"]()
            req = we.call_args.args[0]
        assert req.summary_text == "user text"
        assert req.raw_excerpt == "assistant text"
        assert req.session_id == "s1"
        assert req.user_id == "tester"


# ---------------------------------------------------------------------------
# Memory singleton
# ---------------------------------------------------------------------------

class TestGetMemory:
    def test_returns_none_and_stays_disabled_after_failure(self):
        with (
            patch.object(cs, "_MEMORY_SVC", None),
            patch.object(cs, "_MEMORY_DISABLED", False),
            patch.dict("sys.modules", {"memory.service": MagicMock(
                MemoryService=MagicMock(side_effect=RuntimeError("no schema"))
            )}),
        ):
            assert cs._get_memory() is None
            assert cs._MEMORY_DISABLED is True

    def test_short_circuits_when_disabled(self):
        with patch.object(cs, "_MEMORY_DISABLED", True):
            assert cs._get_memory() is None
