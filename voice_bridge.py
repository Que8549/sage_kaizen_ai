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

import atexit
import msgspec.json as _json
import queue
import re
import subprocess
import threading
import uuid as _uuid
from pathlib import Path

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
# Language-specific code block announcements
# ─────────────────────────────────────────────────────────────────────────────

# Maps the language tag after the opening ``` to a natural spoken phrase.
# The phrase is injected once in place of the entire code block content.
_CODE_LANG_ANNOUNCEMENTS: dict[str, str] = {
    # Web
    "html":       "An HTML structure is shown in the UI.",
    "htm":        "An HTML structure is shown in the UI.",
    "css":        "CSS styling is shown in the UI.",
    "javascript": "JavaScript logic is shown in the UI.",
    "js":         "JavaScript logic is shown in the UI.",
    "typescript": "TypeScript code is shown in the UI.",
    "ts":         "TypeScript code is shown in the UI.",
    "jsx":        "JSX component code is shown in the UI.",
    "tsx":        "TSX component code is shown in the UI.",
    # Systems / backend
    "python":     "Python code is shown in the UI.",
    "py":         "Python code is shown in the UI.",
    "csharp":     "C sharp code is shown in the UI.",
    "cs":         "C sharp code is shown in the UI.",
    "java":       "Java code is shown in the UI.",
    "cpp":        "C plus plus code is shown in the UI.",
    "c":          "C code is shown in the UI.",
    "go":         "Go code is shown in the UI.",
    "rust":       "Rust code is shown in the UI.",
    "rs":         "Rust code is shown in the UI.",
    "swift":      "Swift code is shown in the UI.",
    "kotlin":     "Kotlin code is shown in the UI.",
    "kt":         "Kotlin code is shown in the UI.",
    "ruby":       "Ruby code is shown in the UI.",
    "rb":         "Ruby code is shown in the UI.",
    "php":        "PHP code is shown in the UI.",
    "lua":        "Lua code is shown in the UI.",
    "zig":        "Zig code is shown in the UI.",
    # Data / config
    "sql":        "An SQL query is shown in the UI.",
    "json":       "JSON data is shown in the UI.",
    "xml":        "XML markup is shown in the UI.",
    "yaml":       "YAML configuration is shown in the UI.",
    "yml":        "YAML configuration is shown in the UI.",
    "toml":       "TOML configuration is shown in the UI.",
    # Shell
    "bash":       "A shell script is shown in the UI.",
    "sh":         "A shell script is shown in the UI.",
    "powershell": "A PowerShell script is shown in the UI.",
    "ps1":        "A PowerShell script is shown in the UI.",
    "cmd":        "A command script is shown in the UI.",
    # Markup / docs
    "markdown":   "Markdown content is shown in the UI.",
    "md":         "Markdown content is shown in the UI.",
}

_CODE_SUB_GENERIC = " A code block is shown in the UI. "


def _code_announcement(lang: str) -> str:
    """Return the TTS announcement for a fenced code block with the given language tag."""
    phrase = _CODE_LANG_ANNOUNCEMENTS.get(lang.lower().strip())
    return f" {phrase} " if phrase else _CODE_SUB_GENERIC


# ─────────────────────────────────────────────────────────────────────────────
# _TtsFilter — token-by-token state machine
# ─────────────────────────────────────────────────────────────────────────────

class _TtsFilter:
    """
    Strips content inappropriate for TTS from a streaming LLM response.

    States:
      NORMAL          — emit tokens (after light markdown cleanup)
      IN_THINK        — inside <think>...</think>; suppress all content
      IN_CODE_HEADER  — collecting language tag after opening ```; suppress content
      IN_CODE         — inside code fence body; suppress content, emit one announcement

    A hold-back buffer guards against tags split across token boundaries
    (e.g. token N = "<thi", token N+1 = "nk>").

    Language-specific announcements
    --------------------------------
    When a fenced block opens with a recognised language tag (e.g. ```html,
    ```python, ```javascript) the announcement names the language:
      "An HTML structure is shown in the UI."
      "Python code is shown in the UI."
    Unknown or untagged blocks fall back to "A code block is shown in the UI."
    """

    _HOLD_BACK = len("</think>")   # longest sentinel we must watch for at tail

    def __init__(self) -> None:
        self._buf:              str  = ""
        self._in_think:         bool = False
        self._in_code_header:   bool = False   # collecting lang tag after ```
        self._in_code:          bool = False
        self._code_lang:        str  = ""
        self._code_announced:   bool = False

    def reset(self) -> None:
        """Reset all state for a new turn."""
        self._buf             = ""
        self._in_think        = False
        self._in_code_header  = False
        self._in_code         = False
        self._code_lang       = ""
        self._code_announced  = False

    def feed(self, chunk: str) -> str:
        """
        Process an incoming token chunk.
        Returns text that is safe to publish to TTS (may be empty string).
        """
        self._buf += chunk
        return _clean_markdown(self._drain())

    def flush(self) -> str:
        """Drain any remaining buffer at end of turn."""
        if self._in_think or self._in_code_header or self._in_code:
            self._buf = ""
            return ""
        remaining  = _clean_markdown(self._buf)
        self._buf  = ""
        return remaining

    # ── Internal drain loop ──────────────────────────────────────────────────

    def _drain(self) -> str:
        output: list[str] = []
        while self._buf:
            if self._in_think:
                if self._drain_think():
                    continue              # exited think block — re-enter normal
                break
            elif self._in_code_header:
                if self._drain_code_header():
                    continue              # collected lang tag — enter code body
                break
            elif self._in_code:
                if self._drain_code(output):
                    continue              # exited code block — re-enter normal
                break
            else:
                output.extend(self._drain_normal())
                break                     # _drain_normal holds or consumes all
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
            # Enter code-header state to collect the language tag
            self._in_code_header = True
            self._code_lang      = ""
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

    def _drain_code_header(self) -> bool:
        """
        IN_CODE_HEADER state: collect language tag until the first newline.

        The language tag is the text between the opening ``` and the first \\n,
        e.g. "python" in ```python\\n.  It arrives across one or more tokens so
        we accumulate until we see \\n, then transition to IN_CODE.

        Returns True when the newline (and therefore the full tag) has been seen.
        """
        newline_pos = self._buf.find("\n")
        if newline_pos >= 0:
            self._code_lang      = (self._code_lang + self._buf[:newline_pos]).strip()
            self._buf            = self._buf[newline_pos + 1:]   # skip the header line
            self._in_code_header = False
            self._in_code        = True
            self._code_announced = False
            return True
        # No newline yet — accumulate what we have and wait for more tokens
        self._code_lang += self._buf
        self._buf = ""
        return False

    def _drain_code(self, output: list[str]) -> bool:
        """IN_CODE state: discard until closing ```. Returns True when block ended."""
        if not self._code_announced:
            output.append(_code_announcement(self._code_lang))
            self._code_announced = True
        end = self._buf.find("```")
        if end >= 0:
            self._in_code        = False
            self._code_announced = False
            self._code_lang      = ""
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
        self._stop_event:        threading.Event  = threading.Event()

        self._filter = _TtsFilter()

        # ── ZMQ context — LINGER=0 so ctx.destroy() never blocks on exit ────
        # Without this, zmq.Context.__del__ calls ctx.term() which waits for
        # all open sockets to drain — including the PULL sockets held by daemon
        # threads — causing the process to hang after "Stopping..." on Ctrl+C.
        self._ctx = zmq.Context.instance()
        self._ctx.setsockopt(zmq.LINGER, 0)

        # ── PUB socket — Streamlit main thread only, never touched by threads ─
        self._pub: zmq.Socket = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
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
        self._proc: subprocess.Popen | None = self._launch_voice_app()

        # ── Register cleanup so Ctrl+C actually exits ────────────────────────
        atexit.register(self.shutdown)

    # ── Subprocess ────────────────────────────────────────────────────────────

    def _launch_voice_app(self) -> subprocess.Popen | None:
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
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 500)   # unblock every 500 ms to check stop event
        sock.bind(_ADDR_TRANSCRIPT)
        _LOG.info("VoiceBridge: PULL bound on %s", _ADDR_TRANSCRIPT)
        try:
            while not self._stop_event.is_set():
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
                except zmq.Again:
                    pass   # RCVTIMEO expired — loop and re-check stop event
                except zmq.ZMQError:
                    if self._stop_event.is_set():
                        break
                    _LOG.exception("VoiceBridge _recv_transcripts ZMQ error")
                except Exception:
                    _LOG.exception("VoiceBridge _recv_transcripts error")
        finally:
            sock.close(linger=0)

    def _recv_barge_in(self) -> None:
        """Thread B: PULL on 5792 — receives barge-in interrupt signals."""
        ctx  = zmq.Context.instance()
        sock = ctx.socket(zmq.PULL)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 500)   # unblock every 500 ms to check stop event
        sock.bind(_ADDR_INTERRUPT)
        _LOG.info("VoiceBridge: PULL bound on %s", _ADDR_INTERRUPT)
        try:
            while not self._stop_event.is_set():
                try:
                    msg = _json.decode(sock.recv())
                    if msg.get("type") == "interrupt":
                        _LOG.info(
                            "Barge-in signal (session=%.8s)",
                            msg.get("session_id", ""),
                        )
                        self.barge_in_event.set()
                except zmq.Again:
                    pass   # RCVTIMEO expired — loop and re-check stop event
                except zmq.ZMQError:
                    if self._stop_event.is_set():
                        break
                    _LOG.exception("VoiceBridge _recv_barge_in ZMQ error")
                except Exception:
                    _LOG.exception("VoiceBridge _recv_barge_in error")
        finally:
            sock.close(linger=0)

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

    def play_greeting(self, text: str = "Sage Kaizen online.") -> None:
        """
        Play a one-shot greeting via the TTS pipeline.

        Called by the Streamlit UI once Q5, Q6, AND voice are all confirmed
        ready — so the user hears the announcement only after every component
        is online.

        Sends session_start → token (full greeting text) → turn_done in one
        shot.  The voice app's sentence buffer flushes on turn_done, so the
        entire phrase is synthesised and played as a single audio chunk.
        Must be called from the Streamlit main thread (owns the PUB socket).
        """
        if not self.voice_ready:
            _LOG.warning("VoiceBridge.play_greeting: voice app not ready yet — skipped")
            return
        session_id = str(_uuid.uuid4())
        voice, speed, persona = _BRAIN_VOICE["FAST"]
        try:
            self._pub.send(_json.encode({
                "type":       "session_start",
                "session_id": session_id,
                "voice":      voice,
                "speed":      speed,
                "lang":       "en-us",
                "persona":    persona,
            }), zmq.NOBLOCK)
            self._pub.send(_json.encode({
                "type":       "token",
                "session_id": session_id,
                "text":       text,
            }), zmq.NOBLOCK)
            self._pub.send(_json.encode({
                "type":       "turn_done",
                "session_id": session_id,
            }), zmq.NOBLOCK)
            _LOG.info("VoiceBridge: greeting sent — %r", text)
        except zmq.ZMQError:
            _LOG.warning("VoiceBridge: failed to send greeting (voice app down?)")

    # ── Graceful shutdown ─────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """
        Graceful teardown called on process exit (via atexit) or Ctrl+C.

        1. Signals both PULL threads to stop via _stop_event (they exit their
           while-loop on the next 500 ms RCVTIMEO tick and close their sockets).
        2. Terminates (then kills if needed) the voice app subprocess.
        3. Closes the PUB socket and destroys the ZMQ context with linger=0
           so no blocking occurs even if the PULL threads haven't finished yet.
        """
        if self._stop_event.is_set():
            return   # already shut down
        _LOG.info("VoiceBridge: shutting down")
        self._stop_event.set()

        # Terminate voice app subprocess
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

        # Close PUB socket
        try:
            self._pub.close(linger=0)
        except Exception:
            pass

        # Destroy ZMQ context — linger=0 means no blocking even if PULL
        # sockets in daemon threads are still technically open at this moment.
        try:
            self._ctx.destroy(linger=0)
        except Exception:
            pass

        _LOG.info("VoiceBridge: shutdown complete")
