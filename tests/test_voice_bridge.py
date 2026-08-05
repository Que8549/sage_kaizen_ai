"""
tests/test_voice_bridge.py

Unit tests for voice_bridge.py.

_TtsFilter is a pure token-by-token state machine and gets the bulk of the
attention here — it is the component most likely to break silently, because a
filter bug produces *spoken* garbage rather than a stack trace.

VoiceBridge itself binds three ZMQ sockets and spawns a subprocess in
__init__, so it is constructed with those patched out rather than for real.
"""
from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest
import zmq

import voice_bridge as vb
from voice_bridge import _TtsFilter, _clean_markdown, _code_announcement


# ---------------------------------------------------------------------------
# _clean_markdown
# ---------------------------------------------------------------------------

class TestCleanMarkdown:
    def test_strips_headers(self):
        assert "#" not in _clean_markdown("# Title\n## Sub\nbody")

    def test_keeps_header_text(self):
        assert "Title" in _clean_markdown("# Title")

    def test_unwraps_bold_and_italic(self):
        assert _clean_markdown("**bold** and *ital*") == "bold and ital"

    def test_unwraps_underscore_emphasis(self):
        assert _clean_markdown("__bold__ and _ital_") == "bold and ital"

    def test_link_becomes_label(self):
        assert _clean_markdown("see [the docs](https://x.com)") == "see the docs"

    def test_inline_code_is_unwrapped(self):
        assert _clean_markdown("run `pytest` now") == "run pytest now"

    def test_bullet_markers_removed(self):
        out = _clean_markdown("- one\n* two\n+ three")
        assert out == "one\ntwo\nthree"

    def test_numbered_list_markers_removed(self):
        assert _clean_markdown("1. first\n2. second") == "first\nsecond"

    def test_indented_list_markers_removed(self):
        assert _clean_markdown("   - indented") == "indented"

    def test_sources_block_is_dropped(self):
        out = _clean_markdown("The answer.\n\nSources:\n[1] https://a.com\n[2] https://b.com")
        assert "Sources" not in out and "a.com" not in out
        assert "The answer." in out

    def test_sources_match_is_case_insensitive(self):
        assert "b.com" not in _clean_markdown("Answer.\n\nSOURCES:\n[1] https://b.com")

    def test_plain_text_unchanged(self):
        assert _clean_markdown("Just a normal sentence.") == "Just a normal sentence."

    def test_empty_string(self):
        assert _clean_markdown("") == ""


# ---------------------------------------------------------------------------
# _code_announcement
# ---------------------------------------------------------------------------

class TestCodeAnnouncement:
    @pytest.mark.parametrize(
        "lang,fragment",
        [
            ("python", "Python code"),
            ("py", "Python code"),
            ("HTML", "HTML structure"),
            ("js", "JavaScript logic"),
            ("cs", "C sharp"),
            ("cpp", "C plus plus"),
            ("sql", "SQL query"),
            ("powershell", "PowerShell script"),
            ("yaml", "YAML configuration"),
        ],
    )
    def test_known_languages(self, lang, fragment):
        assert fragment in _code_announcement(lang)

    def test_case_and_whitespace_insensitive(self):
        assert _code_announcement("  PyThOn  ") == _code_announcement("python")

    def test_unknown_language_falls_back(self):
        assert "A code block is shown" in _code_announcement("brainfuck")

    def test_empty_tag_falls_back(self):
        assert "A code block is shown" in _code_announcement("")

    def test_announcement_is_space_padded(self):
        """Padding keeps the phrase from fusing with adjacent prose in TTS."""
        out = _code_announcement("python")
        assert out.startswith(" ") and out.endswith(" ")


# ---------------------------------------------------------------------------
# _TtsFilter
# ---------------------------------------------------------------------------

def feed_all(f: _TtsFilter, text: str, chunk_size: int = 1) -> str:
    """Feed `text` through the filter in fixed-size chunks, then flush."""
    out = [f.feed(text[i:i + chunk_size]) for i in range(0, len(text), chunk_size)]
    out.append(f.flush())
    return "".join(out)


class TestTtsFilterNormal:
    def test_passes_plain_text_through(self):
        assert feed_all(_TtsFilter(), "Hello world.") == "Hello world."

    def test_single_chunk_still_flushes_fully(self):
        f = _TtsFilter()
        assert f.feed("Hello world.") + f.flush() == "Hello world."

    def test_applies_markdown_cleanup(self):
        assert feed_all(_TtsFilter(), "**bold**", chunk_size=99) == "bold"

    def test_empty_input(self):
        assert feed_all(_TtsFilter(), "") == ""


# ---------------------------------------------------------------------------
# KNOWN DEFECT — _TtsFilter never leaves IN_THINK / IN_CODE (found 2026-08-04)
# ---------------------------------------------------------------------------
# Two independent flaws in the drain loop, both verified by direct state
# inspection (see the xfail-ed tests below for the exact reproductions):
#
#   A. `_drain()` does `output.extend(self._drain_normal()); break` — so a
#      feed() that *enters* a think/code block stops there and never processes
#      the rest of that same buffer. After feeding a complete
#      "Before<think>x</think>After" in one chunk the filter is left with
#      _in_think=True and _buf='x</think>After'; flush() then discards it.
#
#   B. `_drain_think()` / `_drain_code()` do `self._buf = ""` whenever the
#      terminator is not in the *current* buffer. llama-server streams
#      token-by-token, so "</think>" almost always arrives split (e.g. "</thi"
#      then "nk>") — each fragment is discarded before the next arrives, the
#      terminator can never be reassembled, and the filter stays in IN_THINK
#      for the rest of the turn.
#
# `_HOLD_BACK` exists to solve exactly this and the class docstring credits it
# with doing so ("guards against tags split across token boundaries"), but it
# is only wired into the NORMAL state.
#
# Impact: every ARCHITECT turn emits <think> tokens, so TTS goes silent at the
# start of thinking and never recovers — the visible UI text is unaffected,
# which is why this can go unnoticed.
#
# NOT patched here: fixing the drain loop is a source change outside the
# bugs-1-7 scope agreed for this pass, and it needs a decision about how much
# to restructure the state machine. Marked xfail(strict=True) so it fails the
# build the moment it IS fixed, following the same convention
# sage_kaizen_ai_ingest used for run_lyrics_ingest.py (its CLAUDE.md §17).

_TTS_DEFECT = "known defect: _TtsFilter never exits IN_THINK/IN_CODE — see comment above"


class TestTtsFilterThink:
    def test_enters_think_state_and_suppresses_content(self):
        """The part that does work: content inside the block is withheld."""
        f = _TtsFilter()
        out = f.feed("Before<think>hidden reasoning</think>After")
        assert out == "Before"
        assert "hidden" not in out

    @pytest.mark.xfail(strict=True, reason=_TTS_DEFECT)
    def test_text_after_think_block_should_be_spoken(self):
        """Defect A: one feed() only advances the state machine one transition."""
        out = feed_all(_TtsFilter(), "Before<think>hidden reasoning</think>After", 100)
        assert "hidden" not in out
        assert "Before" in out and "After" in out

    @pytest.mark.xfail(strict=True, reason=_TTS_DEFECT)
    def test_think_split_across_token_boundaries(self):
        """Defect B: the hold-back buffer is not applied in the IN_THINK state."""
        f = _TtsFilter()
        parts = ["Say ", "<thi", "nk>", "secret", "</thi", "nk>", " done"]
        out = "".join(f.feed(p) for p in parts) + f.flush()
        assert "secret" not in out
        assert "Say" in out and "done" in out

    def test_closing_tag_split_across_chunks_is_not_detected(self):
        """Pins defect B's exact current behaviour so a fix must update this."""
        f = _TtsFilter()
        for part in ["Say ", "<think>", "secret", "</thi", "nk>", " done"]:
            f.feed(part)
        assert f._in_think is True, "filter unexpectedly recovered — defect B fixed?"
        assert f.flush() == ""

    def test_unterminated_think_suppresses_everything_after(self):
        f = _TtsFilter()
        out = f.feed("visible<think>never closed") + f.flush()
        assert "never closed" not in out

    def test_flush_discards_buffer_while_inside_think(self):
        f = _TtsFilter()
        f.feed("<think>partial")
        assert f.flush() == ""

    @pytest.mark.xfail(strict=True, reason=_TTS_DEFECT)
    def test_multiple_think_blocks(self):
        out = feed_all(_TtsFilter(), "A<think>x</think>B<think>y</think>C", 100)
        assert "x" not in out and "y" not in out
        for ch in "ABC":
            assert ch in out


class TestTtsFilterCode:
    def test_announcement_emitted_when_block_is_entered(self):
        """Works as long as the fence body arrives in a later chunk than the header."""
        out = feed_all(_TtsFilter(), "Here:\n```python\nprint('hi')\n```\nDone", 1)
        assert "print" not in out
        assert "Python code is shown in the UI." in out

    def test_untagged_fence_uses_generic_announcement(self):
        out = feed_all(_TtsFilter(), "```\nsome code\n```", 1)
        assert "A code block is shown in the UI." in out
        assert "some code" not in out

    def test_language_tag_split_across_tokens(self):
        """The IN_CODE_HEADER state DOES accumulate correctly across chunks."""
        f = _TtsFilter()
        for part in ["```", "pyth", "on", "\n", "body"]:
            f.feed(part)
        assert f._in_code is True
        assert f._code_lang == "python"

    def test_announcement_emitted_once_per_block(self):
        out = feed_all(_TtsFilter(), "```python\na\nb\nc\nd\n```", 1)
        assert out.count("Python code is shown in the UI.") == 1

    @pytest.mark.xfail(strict=True, reason=_TTS_DEFECT)
    def test_text_after_code_block_should_be_spoken(self):
        """Defect B again: the closing ``` is split across tokens and lost."""
        out = feed_all(_TtsFilter(), "```python\na\n```\nDone", 1)
        assert "Done" in out

    def test_two_blocks_get_two_announcements_at_favourable_chunking(self):
        """
        Passes at chunk_size=3 because each closing ``` happens to land wholly
        inside one chunk. See TestTtsFilterMixed for why that is luck, not
        design.
        """
        out = feed_all(
            _TtsFilter(), "```python\na\n```\nmid\n```sql\nb\n```", chunk_size=3
        )
        assert "Python code is shown" in out
        assert "SQL query is shown" in out
        assert "mid" in out

    @pytest.mark.xfail(strict=True, reason=_TTS_DEFECT)
    def test_two_blocks_get_two_announcements_token_by_token(self):
        """The realistic case: llama-server streams roughly a token at a time."""
        out = feed_all(
            _TtsFilter(), "```python\na\n```\nmid\n```sql\nb\n```", chunk_size=1
        )
        assert "Python code is shown" in out
        assert "SQL query is shown" in out
        assert "mid" in out

    def test_unterminated_code_block_suppresses_body(self):
        f = _TtsFilter()
        out = f.feed("```python\nnever closed") + f.flush()
        assert "never closed" not in out

    def test_flush_discards_buffer_while_inside_code(self):
        f = _TtsFilter()
        f.feed("```python\npartial")
        assert f.flush() == ""

    def test_flush_discards_buffer_while_in_code_header(self):
        f = _TtsFilter()
        f.feed("```pyth")
        assert f.flush() == ""


class TestTtsFilterMixed:
    @pytest.mark.xfail(strict=True, reason=_TTS_DEFECT)
    def test_think_then_code_then_prose(self):
        text = "Intro<think>plan</think>Body:\n```js\nx=1\n```\nOutro"
        out = feed_all(_TtsFilter(), text, chunk_size=2)
        assert "plan" not in out and "x=1" not in out
        assert "Intro" in out and "Body" in out and "Outro" in out
        assert "JavaScript logic is shown in the UI." in out

    _MIXED = "A<think>t</think>B\n```python\nc=1\n```\nD"
    _CHUNK_SIZES = [1, 2, 3, 5, 7, 8, 13, 100]

    def test_output_currently_varies_with_chunk_size(self):
        """
        Pins the defect's headline symptom: identical input, different speech.

        Measured 2026-08-04 — the same string produces at least three distinct
        outputs depending only on how the stream happened to be tokenised,
        from 'A' (everything after the first <think> dropped) through to the
        fully correct text at chunk_size=13. A fix makes this set collapse to
        one element and this test will fail, which is the intent.
        """
        outputs = {feed_all(_TtsFilter(), self._MIXED, n) for n in self._CHUNK_SIZES}
        assert len(outputs) > 1, (
            "output no longer varies with chunking — the defect is fixed; "
            "delete this test and un-xfail test_output_should_be_independent_of_chunking"
        )

    @pytest.mark.xfail(strict=True, reason=_TTS_DEFECT)
    def test_output_should_be_independent_of_chunking(self):
        """
        The property the hold-back design exists to provide.

        Token boundaries are an artefact of the sampler; they must not change
        what the user hears.
        """
        outputs = {feed_all(_TtsFilter(), self._MIXED, n) for n in self._CHUNK_SIZES}
        assert len(outputs) == 1, f"{len(outputs)} distinct outputs: {outputs}"

    @pytest.mark.parametrize("chunk_size", _CHUNK_SIZES)
    def test_hidden_content_is_never_spoken_at_any_chunk_size(self, chunk_size):
        """
        The safety property that DOES hold everywhere: the defect drops too
        much, never too little. Bad for usability, not a leak.
        """
        text = "A<think>SECRET</think>B\n```python\nCODEBODY\n```\nD"
        out = feed_all(_TtsFilter(), text, chunk_size=chunk_size)
        assert "SECRET" not in out
        assert "CODEBODY" not in out



class TestTtsFilterReset:
    def test_reset_clears_all_state(self):
        f = _TtsFilter()
        f.feed("<think>mid-thought")
        f.reset()
        assert f._buf == ""
        assert f._in_think is False
        assert f._in_code is False
        assert f._in_code_header is False
        assert f._code_lang == ""
        assert f._code_announced is False

    def test_filter_is_reusable_after_reset(self):
        f = _TtsFilter()
        f.feed("```python\nleftover")
        f.reset()
        assert f.feed("clean text") + f.flush() == "clean text"


# ---------------------------------------------------------------------------
# VoiceBridge — constructed with ZMQ and the subprocess patched out
# ---------------------------------------------------------------------------

@pytest.fixture
def bridge():
    """A VoiceBridge with sockets, threads and the subprocess mocked."""
    pub = MagicMock()
    ctx = MagicMock()
    ctx.socket.return_value = pub
    with (
        patch.object(vb.zmq.Context, "instance", return_value=ctx),
        patch.object(vb.threading, "Thread") as T,
        patch.object(vb.VoiceBridge, "_launch_voice_app", return_value=None),
        patch.object(vb.atexit, "register"),
    ):
        T.return_value = MagicMock()
        b = vb.VoiceBridge()
    b._pub = pub
    return b


def sent_messages(bridge) -> list[dict]:
    import msgspec.json as _json
    return [_json.decode(c.args[0]) for c in bridge._pub.send.call_args_list]


class TestVoiceBridgeInit:
    def test_binds_the_token_bus(self, bridge):
        bridge._pub.bind.assert_called_once_with(vb._ADDR_TOKEN_BUS)

    def test_starts_with_empty_transcript_queue(self, bridge):
        assert bridge.transcript_queue.empty()

    def test_voice_not_ready_initially(self, bridge):
        assert bridge.voice_ready is False

    def test_barge_in_starts_clear(self, bridge):
        assert bridge.barge_in_event.is_set() is False


class TestVoiceBridgeLaunch:
    # pathlib.Path.exists is read-only on instances, so the module-level Path
    # constants are replaced wholesale rather than patched in place.
    @staticmethod
    def _paths(python_exists: bool, script_exists: bool):
        py, sc = MagicMock(), MagicMock()
        py.exists.return_value = python_exists
        sc.exists.return_value = script_exists
        return (
            patch.object(vb, "_VOICE_PYTHON", py),
            patch.object(vb, "_VOICE_SCRIPT", sc),
        )

    def test_returns_none_when_python_missing(self):
        p, s = self._paths(False, True)
        with p, s:
            assert vb.VoiceBridge._launch_voice_app(MagicMock()) is None

    def test_returns_none_when_script_missing(self):
        p, s = self._paths(True, False)
        with p, s:
            assert vb.VoiceBridge._launch_voice_app(MagicMock()) is None

    def test_does_not_spawn_when_paths_missing(self):
        p, s = self._paths(False, False)
        with p, s, patch.object(vb.subprocess, "Popen") as P:
            vb.VoiceBridge._launch_voice_app(MagicMock())
        P.assert_not_called()

    def test_spawns_in_integrated_mode(self):
        p, s = self._paths(True, True)
        with p, s, patch.object(vb.subprocess, "Popen") as P:
            P.return_value = MagicMock(pid=4242)
            proc = vb.VoiceBridge._launch_voice_app(MagicMock())
        assert proc is not None
        argv = P.call_args.args[0]
        assert "--mode" in argv and "integrated" in argv

    def test_spawn_silences_child_output(self):
        """The voice app writes its own logs; inherited pipes would fill and block."""
        import subprocess as sp
        p, s = self._paths(True, True)
        with p, s, patch.object(vb.subprocess, "Popen") as P:
            P.return_value = MagicMock(pid=1)
            vb.VoiceBridge._launch_voice_app(MagicMock())
        assert P.call_args.kwargs["stdout"] == sp.DEVNULL
        assert P.call_args.kwargs["stderr"] == sp.DEVNULL


class TestVoiceBridgeTurnLifecycle:
    def test_start_turn_sends_session_start(self, bridge):
        bridge.start_turn("s-1", "FAST")
        msg = sent_messages(bridge)[0]
        assert msg["type"] == "session_start"
        assert msg["session_id"] == "s-1"
        assert msg["lang"] == "en-us"

    def test_start_turn_maps_brain_to_voice_profile(self, bridge):
        bridge.start_turn("s", "ARCHITECT")
        msg = sent_messages(bridge)[0]
        assert msg["speed"] == vb._BRAIN_VOICE["ARCHITECT"][1]
        assert msg["persona"] == "narrator"

    def test_unknown_brain_falls_back_to_fast_profile(self, bridge):
        bridge.start_turn("s", "MYSTERY")
        assert sent_messages(bridge)[0]["persona"] == "chat"

    def test_start_turn_resets_filter_and_barge_in(self, bridge):
        bridge.barge_in_event.set()
        bridge._filter.feed("<think>stale")
        bridge.start_turn("s", "FAST")
        assert bridge.barge_in_event.is_set() is False
        assert bridge._filter._in_think is False

    def test_start_turn_survives_dead_voice_app(self, bridge):
        bridge._pub.send.side_effect = zmq.ZMQError()
        bridge.start_turn("s", "FAST")   # must not raise

    def test_publish_token_sends_filtered_text(self, bridge):
        bridge.start_turn("s", "FAST")
        bridge._pub.send.reset_mock()
        # Longer than _HOLD_BACK (8) so the filter actually emits this turn
        # rather than withholding it all as a possible split sentinel.
        bridge.publish_token("s", "**hello** there friends and neighbours")
        msg = sent_messages(bridge)[0]
        assert msg["type"] == "token"
        assert msg["session_id"] == "s"
        assert "hello" in msg["text"]
        assert "**" not in msg["text"]

    def test_publish_token_withholds_short_tail(self, bridge):
        """Short chunks are held back in case they are a split <think> tag."""
        bridge.start_turn("s", "FAST")
        bridge._pub.send.reset_mock()
        bridge.publish_token("s", "hi")
        assert bridge._pub.send.call_count == 0

    def test_publish_token_suppressed_inside_think(self, bridge):
        bridge.start_turn("s", "FAST")
        bridge._pub.send.reset_mock()
        bridge.publish_token("s", "<think>")
        bridge.publish_token("s", "secret reasoning")
        assert bridge._pub.send.call_count == 0

    def test_publish_token_survives_dead_voice_app(self, bridge):
        bridge.start_turn("s", "FAST")
        bridge._pub.send.side_effect = zmq.ZMQError()
        bridge.publish_token("s", "text that would be sent")   # must not raise

    def test_end_turn_flushes_then_signals_done(self, bridge):
        bridge.start_turn("s", "FAST")
        bridge.publish_token("s", "held back tail")
        bridge._pub.send.reset_mock()
        bridge.end_turn("s")
        types = [m["type"] for m in sent_messages(bridge)]
        assert types[-1] == "turn_done"

    def test_end_turn_sends_turn_done_even_with_nothing_buffered(self, bridge):
        bridge.start_turn("s", "FAST")
        bridge._pub.send.reset_mock()
        bridge.end_turn("s")
        assert [m["type"] for m in sent_messages(bridge)] == ["turn_done"]

    def test_end_turn_survives_dead_voice_app(self, bridge):
        bridge._pub.send.side_effect = zmq.ZMQError()
        bridge.end_turn("s")   # must not raise


class TestVoiceBridgeGreeting:
    def test_skipped_when_voice_not_ready(self, bridge):
        bridge.play_greeting("hi")
        assert bridge._pub.send.call_count == 0

    def test_sends_three_message_sequence_when_ready(self, bridge):
        bridge._voice_ready_event.set()
        bridge.play_greeting("Sage online.")
        types = [m["type"] for m in sent_messages(bridge)]
        assert types == ["session_start", "token", "turn_done"]

    def test_greeting_text_is_carried(self, bridge):
        bridge._voice_ready_event.set()
        bridge.play_greeting("Custom greeting.")
        assert sent_messages(bridge)[1]["text"] == "Custom greeting."

    def test_all_three_share_one_session_id(self, bridge):
        bridge._voice_ready_event.set()
        bridge.play_greeting()
        ids = {m["session_id"] for m in sent_messages(bridge)}
        assert len(ids) == 1

    def test_survives_dead_voice_app(self, bridge):
        bridge._voice_ready_event.set()
        bridge._pub.send.side_effect = zmq.ZMQError()
        bridge.play_greeting()   # must not raise


class TestVoiceBridgeShutdown:
    def test_sets_stop_event(self, bridge):
        bridge.shutdown()
        assert bridge._stop_event.is_set()

    def test_is_idempotent(self, bridge):
        bridge.shutdown()
        bridge._pub.close.reset_mock()
        bridge.shutdown()
        bridge._pub.close.assert_not_called()

    def test_terminates_the_subprocess(self, bridge):
        proc = MagicMock()
        bridge._proc = proc
        bridge.shutdown()
        proc.terminate.assert_called_once()
        assert bridge._proc is None

    def test_kills_when_terminate_times_out(self, bridge):
        proc = MagicMock()
        proc.wait.side_effect = Exception("timeout")
        bridge._proc = proc
        bridge.shutdown()
        proc.kill.assert_called_once()

    def test_closes_socket_with_linger_zero(self, bridge):
        bridge.shutdown()
        bridge._pub.close.assert_called_once_with(linger=0)

    def test_survives_socket_close_failure(self, bridge):
        bridge._pub.close.side_effect = RuntimeError("already closed")
        bridge.shutdown()   # must not raise
