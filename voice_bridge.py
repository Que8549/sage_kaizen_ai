"""
voice_bridge.py — Sage Kaizen Voice ZMQ Bridge
================================================
Singleton that runs for the lifetime of the Streamlit process.

Binds all three ZMQ sockets (the voice app connects to these):
  Port 5790  PULL BIND — receives transcripts + voice_ready signal from voice app
  Port 5791  PUB  BIND — publishes LLM tokens to voice app (main thread only)
  Port 5792  PULL BIND — receives barge-in interrupt signals from voice app

Also owns the voice app subprocess (Python 3.11 venv, integrated mode).

Usage — in ui_streamlit_server.py via @st.cache_resource:

    @st.cache_resource
    def _get_voice_bridge() -> VoiceBridge:
        return VoiceBridge()

    _voice_bridge = _get_voice_bridge()

Per-turn usage:

    voice_session_id = str(uuid4())
    _voice_bridge.start_turn(voice_session_id, decision.brain)
    try:
        for piece in chat_svc.stream_response(...):
            if _voice_bridge.barge_in_event.is_set():
                _voice_bridge.barge_in_event.clear()
                break
            acc.append(piece)
            live.markdown("".join(acc))
            _voice_bridge.publish_token(voice_session_id, piece)
    except ...:
        ...
    finally:
        _voice_bridge.end_turn(voice_session_id)

Thread-safety invariants:
  - ZMQ sockets are never shared across threads.
  - _pull_5790 and _pull_5792 daemon threads each own their own socket.
  - _pub is owned exclusively by the Streamlit main thread (one turn at a time).
"""
from __future__ import annotations

import msgspec.json as _json
import queue
import re
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Optional

import zmq

from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.voice_bridge")

# ── Voice app process paths ────────────────────────────────────────────────────
_VOICE_APP_ROOT = Path(r"F:\Projects\sage_kaizen_ai_voice").resolve()
_VOICE_PYTHON   = (_VOICE_APP_ROOT / r".venv\Scripts\python.exe").resolve()
_VOICE_SCRIPT   = (_VOICE_APP_ROOT / r"scripts\run_pipeline.py").resolve()

# ── Brain → TTS voice/speed/persona mapping ───────────────────────────────────
# Keyed by RouteDecision.brain ("FAST" | "ARCHITECT")
_BRAIN_VOICE: dict[str, tuple[str, float, str]] = {
    "FAST":      ("am_onyx", 1.00, "chat"),
    "ARCHITECT": ("am_onyx", 0.87, "narrator"),
}

# ── ZMQ addresses (must match voice app src/config.py ZMQ class) ──────────────
_ADDR_TRANSCRIPT = "tcp://127.0.0.1:5790"
_ADDR_TOKEN_BUS  = "tcp://127.0.0.1:5791"
_ADDR_INTERRUPT  = "tcp://127.0.0.1:5792"


# ─────────────────────────────────────────────────────────────────────────────
# Markdown cleanup (strip decorators not suitable for speech)
# ─────────────────────────────────────────────────────────────────────────────

_MD_BOLD_ITALIC = re.compile(r"\*{1,3}([^\n*]+)\*{1,3}")
_MD_UNDER_BI    = re.compile(r"_{1,3}([^\n_]+)_{1,3}")
_MD_HEADER      = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_LINK        = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_MD_LIST        = re.compile(r"^[ \t]*[-*+]\s+", re.MULTILINE)
_MD_NUMLIST     = re.compile(r"^[ \t]*\d+\.\s+", re.MULTILINE)
_MD_SOURCES     = re.compile(r"\n+Sources:\s*\n.*$", re.DOTALL | re.IGNORECASE)


def _clean_markdown(text: str) -> str:
    """Strip markdown decorators not suitable for TTS output."""
    text = _MD_SOURCES.sub("", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_BOLD_ITALIC.sub(r"\1", text)
    text = _MD_UNDER_BI.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_LIST.sub("", text)
    text = _MD_NUMLIST.sub("", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# _TtsFilter — token-by-token state machine
# ─────────────────────────────────────────────────────────────────────────────

class _TtsFilter:
    """
    Strips content inappropriate for TTS from a streaming LLM response.

    States:
      NORMAL   — emit tokens (after light markdown cleanup)
      IN_THINK — inside <think>...</think>; suppress all content
      IN_CODE  — inside ``` code fence; suppress content, emit one announcement

    A hold-back buffer guards against tags split across token boundaries
    (e.g. token N = "<thi", token N+1 = "nk>").
    """

    CODE_SUB   = " View code block in the UI. "
    _HOLD_BACK = len("</think>")   # longest sentinel we must watch for at tail

    def __init__(self) -> None:
        self._buf:            str  = ""
        self._in_think:       bool = False
        self._in_code:        bool = False
        self._code_announced: bool = False

    def reset(self) -> None:
        """Reset all state for a new turn."""
        self._buf            = ""
        self._in_think       = False
        self._in_code        = False
        self._code_announced = False

    def feed(self, chunk: str) -> str:
        """
        Process an incoming token chunk.
        Returns text that is safe to publish to TTS (may be empty string).
        """
        self._buf += chunk
        return _clean_markdown(self._drain())

    def flush(self) -> str:
        """Drain any remaining buffer at end of turn."""
        if self._in_think or self._in_code:
            self._buf = ""
            return ""
        remaining  = _clean_markdown(self._buf)
        self._buf  = ""
        return remaining

    # ── Internal drain loop ──────────────────────────────────────────────────

    def _drain(self) -> str:
        output: list[str] = []
        while self._buf:
            if not self._in_think and not self._in_code:
                output.extend(self._drain_normal())
                break                     # _drain_normal holds or consumes all
            elif self._in_think:
                if self._drain_think():
                    continue              # exited think block — re-enter normal
                break
            else:                         # _in_code
                if self._drain_code(output):
                    continue              # exited code block — re-enter normal
                break
        return "".join(output)

    def _drain_normal(self) -> list[str]:
        """NORMAL state: look for <think> or ``` openers in buffer."""
        t_pos = self._buf.find("<think>")
        c_pos = self._buf.find("```")

        first = min(
            t_pos if t_pos >= 0 else len(self._buf),
            c_pos if c_pos >= 0 else len(self._buf),
        )

        if first == len(self._buf):
            # No opener found — emit all but the hold-back tail
            safe   = max(0, len(self._buf) - self._HOLD_BACK)
            result = [self._buf[:safe]]
            self._buf = self._buf[safe:]
            return result

        # Emit clean text before the opener
        result = [self._buf[:first]] if first > 0 else []

        if t_pos >= 0 and t_pos == first:
            self._in_think = True
            self._buf      = self._buf[t_pos + len("<think>"):]
        else:
            self._in_code        = True
            self._code_announced = False
            self._buf            = self._buf[c_pos + 3:]   # skip opening ```

        return result

    def _drain_think(self) -> bool:
        """IN_THINK state: discard until </think>. Returns True when block ended."""
        end = self._buf.find("</think>")
        if end >= 0:
            self._in_think = False
            self._buf      = self._buf[end + len("</think>"):]
            return True
        self._buf = ""   # discard all — still inside think block
        return False

    def _drain_code(self, output: list[str]) -> bool:
        """IN_CODE state: discard until closing ```. Returns True when block ended."""
        if not self._code_announced:
            output.append(self.CODE_SUB)
            self._code_announced = True
        end = self._buf.find("```")
        if end >= 0:
            self._in_code        = False
            self._code_announced = False
            self._buf            = self._buf[end + 3:]    # skip closing ```
            return True
        self._buf = ""   # discard all — still inside code block
        return False


# ─────────────────────────────────────────────────────────────────────────────
# VoiceBridge
# ─────────────────────────────────────────────────────────────────────────────

class VoiceBridge:
    """
    Singleton ZMQ bridge between the main Sage Kaizen app and the voice app.

    Create exactly once per Streamlit process via @st.cache_resource.
    Starts two daemon PULL threads, binds the PUB socket, and launches
    the voice app subprocess in integrated mode.
    """

    def __init__(self) -> None:
        # ── Thread-safe state consumed by the Streamlit script ──────────────
        self.transcript_queue:   queue.Queue[str] = queue.Queue()
        self.barge_in_event:     threading.Event  = threading.Event()
        self._voice_ready_event: threading.Event  = threading.Event()

        self._filter = _TtsFilter()

        # ── PUB socket — Streamlit main thread only, never touched by threads ─
        self._ctx = zmq.Context.instance()
        self._pub: zmq.Socket = self._ctx.socket(zmq.PUB)
        self._pub.bind(_ADDR_TOKEN_BUS)
        _LOG.info("VoiceBridge: PUB bound on %s", _ADDR_TOKEN_BUS)

        # ── Background PULL threads (each owns its own socket) ──────────────
        threading.Thread(
            target=self._recv_transcripts,
            name="VoiceBridge-PULL-5790",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._recv_barge_in,
            name="VoiceBridge-PULL-5792",
            daemon=True,
        ).start()

        # ── Launch voice app subprocess ──────────────────────────────────────
        self._proc: Optional[subprocess.Popen] = self._launch_voice_app()

    # ── Subprocess ────────────────────────────────────────────────────────────

    def _launch_voice_app(self) -> Optional[subprocess.Popen]:
        if not _VOICE_PYTHON.exists():
            _LOG.warning(
                "Voice Python not found at %s — voice features disabled", _VOICE_PYTHON
            )
            return None
        if not _VOICE_SCRIPT.exists():
            _LOG.warning(
                "Voice script not found at %s — voice features disabled", _VOICE_SCRIPT
            )
            return None

        proc = subprocess.Popen(
            [str(_VOICE_PYTHON), str(_VOICE_SCRIPT), "--mode", "integrated"],
            cwd=str(_VOICE_APP_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _LOG.info("Voice app launched (pid=%d)", proc.pid)
        return proc

    @property
    def voice_ready(self) -> bool:
        """True once the voice app has loaded models and sent its ready signal."""
        return self._voice_ready_event.is_set()

    # ── Background PULL threads ───────────────────────────────────────────────

    def _recv_transcripts(self) -> None:
        """Thread A: PULL on 5790 — receives transcripts + voice_ready signal."""
        ctx  = zmq.Context.instance()
        sock = ctx.socket(zmq.PULL)
        sock.bind(_ADDR_TRANSCRIPT)
        _LOG.info("VoiceBridge: PULL bound on %s", _ADDR_TRANSCRIPT)
        while True:
            try:
                msg   = _json.decode(sock.recv())
                mtype = msg.get("type")
                if mtype == "voice_ready":
                    _LOG.info("Voice app reported ready")
                    self._voice_ready_event.set()
                elif mtype == "transcript":
                    text = msg.get("text", "").strip()
                    if text:
                        self.transcript_queue.put(text)
                        _LOG.info("Voice transcript queued: %r", text[:60])
            except Exception:
                _LOG.exception("VoiceBridge _recv_transcripts error")

    def _recv_barge_in(self) -> None:
        """Thread B: PULL on 5792 — receives barge-in interrupt signals."""
        ctx  = zmq.Context.instance()
        sock = ctx.socket(zmq.PULL)
        sock.bind(_ADDR_INTERRUPT)
        _LOG.info("VoiceBridge: PULL bound on %s", _ADDR_INTERRUPT)
        while True:
            try:
                msg = _json.decode(sock.recv())
                if msg.get("type") == "interrupt":
                    _LOG.info(
                        "Barge-in signal (session=%.8s)",
                        msg.get("session_id", ""),
                    )
                    self.barge_in_event.set()
            except Exception:
                _LOG.exception("VoiceBridge _recv_barge_in error")

    # ── Turn publishing — Streamlit main thread only ──────────────────────────

    def start_turn(self, session_id: str, brain: str) -> None:
        """
        Send session_start to the voice app.
        Call once before entering the stream_response loop.
        """
        self._filter.reset()
        self.barge_in_event.clear()

        voice, speed, persona = _BRAIN_VOICE.get(brain, _BRAIN_VOICE["FAST"])
        msg = {
            "type":       "session_start",
            "session_id": session_id,
            "voice":      voice,
            "speed":      speed,
            "lang":       "en-us",
            "persona":    persona,
        }
        try:
            self._pub.send(_json.encode(msg), zmq.NOBLOCK)
        except zmq.ZMQError:
            _LOG.warning("VoiceBridge: failed to send session_start (voice app down?)")

    def publish_token(self, session_id: str, token: str) -> None:
        """
        Filter a streaming token and publish it to the voice app.
        Called inside the stream_response loop — must be fast.
        Silently drops the token if the voice app is unreachable.
        """
        filtered = self._filter.feed(token)
        if not filtered:
            return
        try:
            self._pub.send(
                _json.encode({
                    "type":       "token",
                    "session_id": session_id,
                    "text":       filtered,
                }),
                zmq.NOBLOCK,
            )
        except zmq.ZMQError:
            pass   # voice app down — silently skip, do not interrupt generation

    def end_turn(self, session_id: str) -> None:
        """
        Flush the filter's hold-back buffer and send turn_done.
        Call in the finally block after stream_response, even on barge-in abort.
        """
        remaining = self._filter.flush()
        if remaining:
            try:
                self._pub.send(
                    _json.encode({
                        "type":       "token",
                        "session_id": session_id,
                        "text":       remaining,
                    }),
                    zmq.NOBLOCK,
                )
            except zmq.ZMQError:
                pass

        try:
            self._pub.send(
                _json.encode({
                    "type":       "turn_done",
                    "session_id": session_id,
                }),
                zmq.NOBLOCK,
            )
        except zmq.ZMQError:
            _LOG.warning("VoiceBridge: failed to send turn_done")
