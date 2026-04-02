from __future__ import annotations

import concurrent.futures
import io
import logging
import queue
import re
import threading
import time
from typing import List, Optional, Tuple
from uuid import uuid4

import streamlit as st

# ── Suppress benign Tornado WebSocket error on Ctrl+C shutdown ───────────────
# When Streamlit stops (RuntimeState.STOPPING), the browser WebSocket may
# deliver one final message.  Tornado catches the resulting RuntimeStoppedError
# and logs it as "Uncaught exception" — noisy but harmless; shutdown completes
# successfully regardless.  Filter it out to keep the logs clean.
#
# NOTE: record.getMessage() contains "Uncaught exception GET /_stcore/stream"
# — it does NOT contain "RuntimeStoppedError".  The exception is in
# record.exc_info, so we must inspect exc_info to suppress the right record.
class _StopRaceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            exc_type = record.exc_info[0]
            if exc_type is not None and "RuntimeStoppedError" in exc_type.__name__:
                return False  # suppress
        return "RuntimeStoppedError" not in record.getMessage()

_stop_race_filter = _StopRaceFilter()
logging.getLogger("tornado.application").addFilter(_stop_race_filter)
logging.getLogger("tornado.general").addFilter(_stop_race_filter)
# ─────────────────────────────────────────────────────────────────────────────

from chat_service import ChatService, MediaAttachment, TurnConfig
from input_guard import InjectionDetectedError, check_user_input
from rag_v1.retrieve.citations import format_sources_markdown
from inference_session import InferenceSession
from mermaid_streamlit import DiagramHandler
from openai_client import HttpTimeouts, LlamaServerError, _normalize_base_url, health_check
from prompt_library import TemplateKey
from router import route as _heuristic_route, llm_route as _llm_route
from router import RouteDecision
from settings import CONFIG
from voice_bridge import VoiceBridge

# ── Thread pool for parallel LLM routing on voice input ─────────────────────
# Shared for the lifetime of the Streamlit process; max_workers=2 because we
# only ever have one active voice routing future at a time.
_ROUTE_EXECUTOR: concurrent.futures.ThreadPoolExecutor = (
    concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="voice_route"
    )
)


# ─────────────────────────────────────────────────────────────────────────── #
# UI helpers                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #


_THINK_RE      = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_OUTPUT_TAG_RE = re.compile(r"</?Final\s*Output>", re.IGNORECASE)

# ── Voice command: "new chat" ──────────────────────────────────────────────
# Matches short utterances whose sole intent is to start a new conversation.
# Allows optional polite prefixes ("hey Sage", "please") and punctuation.
# Intentionally strict so longer queries that happen to contain "new chat"
# are NOT silently swallowed as commands.
_NEW_CHAT_RE = re.compile(
    r"^\s*"
    r"(?:hey\s+sage[,\s]+|ok(?:ay)?\s+sage[,\s]+|please\s+)?"
    r"(?:"
    r"(?:start\s+(?:a\s+)?)?new\s+chat"
    r"|(?:start\s+(?:a\s+)?)?new\s+conversation"
    r"|start\s+over"
    r"|clear\s+(?:the\s+)?(?:chat|conversation|history)"
    r"|reset\s+(?:the\s+)?(?:chat|conversation)"
    r")"
    r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)


def _parse_response(text: str) -> tuple[str | None, str]:
    """Split model output into (thinking, clean_response)."""
    blocks = _THINK_RE.findall(text)
    thinking = "\n\n".join(b.strip() for b in blocks if b.strip()) or None
    clean = _THINK_RE.sub("", text)
    clean = _OUTPUT_TAG_RE.sub("", clean)
    return thinking, clean.strip()


# ─────────────────────────────────────────────────────────────────────────── #
# Media helpers                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

_IMAGE_MIMES = {
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "png":  "image/png",
    "webp": "image/webp",
    "gif":  "image/gif",
    "bmp":  "image/bmp",
}
_AUDIO_MIMES = {
    "wav":  "audio/wav",
    "mp3":  "audio/mpeg",
    "flac": "audio/flac",
    "ogg":  "audio/ogg",
    "m4a":  "audio/mp4",
}
_VIDEO_MIMES = {
    "mp4":  "video/mp4",
    "mov":  "video/quicktime",
    "avi":  "video/x-msvideo",
    "mkv":  "video/x-matroska",
    "webm": "video/webm",
}
_ALL_ACCEPTED = (
    list(_IMAGE_MIMES.keys())
    + list(_AUDIO_MIMES.keys())
    + list(_VIDEO_MIMES.keys())
)


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _extract_video_frames(
    raw: bytes,
    filename: str,
    fps_sample: float = 0.5,
    max_frames: int = 8,
) -> List[MediaAttachment]:
    """
    Extract up to max_frames still images from a video file.

    Uses OpenCV (cv2) if available; falls back to a single-frame PIL stub
    so the import never hard-fails.  Install cv2 with:
        pip install opencv-python-headless
    """
    try:
        import cv2
        import numpy as np

        # Write raw bytes to a temp file because cv2 needs a path
        import tempfile, os
        suffix = f".{_ext(filename)}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, int(video_fps / fps_sample))
            frames_to_grab = list(range(0, total_frames, step))[:max_frames]

            attachments: List[MediaAttachment] = []
            for fi in frames_to_grab:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    continue
                # BGR → RGB → PNG bytes
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image as _PILImage
                pil = _PILImage.fromarray(rgb)
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                attachments.append(MediaAttachment.from_bytes(
                    raw=buf.getvalue(),
                    kind="video_frame",
                    mime_type="image/png",
                    label=f"{filename} frame {fi}",
                    frame_index=fi,
                ))
            cap.release()
        finally:
            os.unlink(tmp_path)

        return attachments

    except ImportError:
        st.warning(
            "cv2 (OpenCV) not installed — video frame extraction unavailable. "
            "Run: `pip install opencv-python-headless`"
        )
        return []
    except Exception as exc:
        st.warning(f"Video frame extraction failed: {exc}")
        return []


def _collect_attachments(
    uploaded_files: list,
    frames_per_second: float,
    max_video_frames: int,
) -> List[MediaAttachment]:
    """Convert Streamlit UploadedFile objects into MediaAttachment list."""
    attachments: List[MediaAttachment] = []
    for uf in uploaded_files:
        ext = _ext(uf.name)
        raw = uf.read()

        if ext in _IMAGE_MIMES:
            attachments.append(MediaAttachment.from_bytes(
                raw=raw,
                kind="image",
                mime_type=_IMAGE_MIMES[ext],
                label=uf.name,
            ))

        elif ext in _AUDIO_MIMES:
            attachments.append(MediaAttachment.from_bytes(
                raw=raw,
                kind="audio",
                mime_type=_AUDIO_MIMES[ext],
                label=uf.name,
            ))

        elif ext in _VIDEO_MIMES:
            frames = _extract_video_frames(raw, uf.name, frames_per_second, max_video_frames)
            attachments.extend(frames)
            if frames:
                st.info(f"Extracted {len(frames)} frame(s) from {uf.name}")

    return attachments


def _render_attachments_preview(attachments: List[MediaAttachment]) -> None:
    """Show inline previews of pending attachments above the chat input."""
    if not attachments:
        return

    imgs  = [a for a in attachments if a.kind in ("image", "video_frame")]
    audio = [a for a in attachments if a.kind == "audio"]

    if imgs:
        import base64
        st.markdown(f"**Media to send ({len(imgs)} image/frame(s))**")
        cols = st.columns(min(len(imgs), 4))
        for col, att in zip(cols, imgs[:4]):
            with col:
                raw = base64.b64decode(att.data_b64)
                st.image(raw, caption=att.label, width='stretch')
        if len(imgs) > 4:
            st.caption(f"… and {len(imgs) - 4} more image(s)/frame(s)")

    for att in audio:
        import base64
        raw = base64.b64decode(att.data_b64)
        st.audio(raw, format=att.mime_type)


# ─────────────────────────────────────────────────────────────────────────── #
# Page config                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

st.set_page_config(
    page_title="Sage Kaizen (llama-server)",
    page_icon="\U0001F9E0",
    layout="wide",
)
st.title("\U0001F9E0 Sage Kaizen (Dual-Brain Chat) \U0001F9E0")


@st.cache_resource
def _get_voice_bridge() -> VoiceBridge:
    """Create the ZMQ voice bridge once per Streamlit process lifetime."""
    return VoiceBridge()


_voice_bridge = _get_voice_bridge()


@st.cache_resource
def _auto_start_servers() -> None:
    """
    Start both brains (embed → Q5, Q6) in background threads at app load.

    Uses @st.cache_resource so this runs exactly once per Streamlit process
    regardless of how many browser tabs or reruns occur.  Both servers start
    in parallel: Q5 waits for embed first (ensure_q5_running handles that),
    while Q6 starts concurrently so the heavier ARCHITECT brain is ready as
    soon as possible.
    """
    from server_manager import ManagedServers, ensure_q5_running, ensure_q6_running
    servers = ManagedServers.from_yaml()
    threading.Thread(
        target=ensure_q5_running, args=(servers,), daemon=True, name="autostart_q5"
    ).start()
    threading.Thread(
        target=ensure_q6_running, args=(servers,), daemon=True, name="autostart_q6"
    ).start()


_auto_start_servers()

timeouts        = HttpTimeouts(connect_s=CONFIG.connect_timeout_s, read_s=CONFIG.read_timeout_s)
timeouts_status = HttpTimeouts(connect_s=0.5, read_s=1.0)

# ─────────────────────────────────────────────────────────────────────────── #
# Session state                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

if "messages"            not in st.session_state: st.session_state.messages            = []
if "last_thinking_time"  not in st.session_state: st.session_state.last_thinking_time  = None
if "q5_model_id"         not in st.session_state: st.session_state.q5_model_id         = CONFIG.q5_model_id
if "q6_model_id"         not in st.session_state: st.session_state.q6_model_id         = CONFIG.q6_model_id
if "last_route"          not in st.session_state: st.session_state.last_route          = None
if "fb_rated_ids"        not in st.session_state: st.session_state.fb_rated_ids        = set()
if "fb_stats_dirty"      not in st.session_state: st.session_state.fb_stats_dirty      = True
if "pending_attachments" not in st.session_state: st.session_state.pending_attachments = []
if "tts_enabled"         not in st.session_state: st.session_state.tts_enabled         = True

# ── Feedback DB: one-time schema init ──────────────────────────────────────
# Module-level flag: ensure_schema() only needs one DB round-trip per process.
# st.session_state would re-run it for every new browser tab; a module-level
# boolean persists for the entire Streamlit process lifetime.
_fb_schema_initialized: bool = False

try:
    from feedback.db import ensure_schema, get_conn as _fb_get_conn
    from pg_settings import PgSettings as _FeedbackSettings
    _fb_cfg = _FeedbackSettings()
    if not _fb_schema_initialized:
        ensure_schema(_fb_cfg.pg_dsn)
        _fb_schema_initialized = True
    if not st.session_state.fb_rated_ids:
        with _fb_get_conn(_fb_cfg.pg_dsn) as _fb_conn:
            with _fb_conn.cursor() as _cur:
                _rows = _cur.execute("SELECT id::text FROM public.ratings").fetchall()
        st.session_state.fb_rated_ids = {r["id"] for r in _rows}
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────── #
# Voice transcript poller                                                      #
# ─────────────────────────────────────────────────────────────────────────── #
# Runs every 100ms as an independent fragment.  When a voice transcript
# arrives from the voice app it is inspected:
#   - voice commands (e.g. "new chat") → set _voice_new_chat flag
#   - all other text                   → set _voice_input for normal processing
# A full-app rerun is triggered in both cases.

@st.fragment(run_every=0.1)
def _voice_input_poller() -> None:
    try:
        text = _voice_bridge.transcript_queue.get_nowait()
        if _NEW_CHAT_RE.match(text):
            st.session_state["_voice_new_chat"] = True
        else:
            st.session_state["_voice_input"] = text
        st.rerun()
    except queue.Empty:
        pass


_voice_input_poller()


# ─────────────────────────────────────────────────────────────────────────── #
# Server-status fragment (polls every 3 s)                                     #
# ─────────────────────────────────────────────────────────────────────────── #
# Uses @st.fragment(run_every=3) so the health captions update automatically
# as servers warm up — no user interaction required.  Called inside the
# sidebar block so its elements are rendered there.

@st.fragment(run_every=3.0)
def _render_server_status() -> None:
    """Render live server health captions; reruns every 3 s independently."""
    _q5_url = _normalize_base_url(CONFIG.q5_base_url)
    _q6_url = _normalize_base_url(CONFIG.q6_base_url)
    _ok5, _d5 = health_check(_q5_url, timeouts=timeouts_status)
    _ok6, _d6 = health_check(_q6_url, timeouts=timeouts_status)
    st.caption(f"Q5 (Omni): {'✅' if _ok5 else '❌'} {_d5}")
    st.caption(f"Q6 (Architect): {'✅' if _ok6 else '❌'} {_d6}")
    _v_icon  = "🎙" if _voice_bridge.voice_ready else "🔄"
    _v_label = "ready" if _voice_bridge.voice_ready else "loading models..."
    st.caption(f"{_v_icon} Voice: {_v_label}")


# ─────────────────────────────────────────────────────────────────────────── #
# TTS toggle fragment                                                           #
# ─────────────────────────────────────────────────────────────────────────── #
# Isolated in a fragment so toggling it does NOT trigger a full app rerun.
# A full rerun while the architect brain is streaming kills the generator,
# stopping the response mid-stream.  Storing the value in session_state lets
# the turn execution code snapshot it once at turn start.

@st.fragment
def _tts_toggle() -> None:
    val = st.toggle(
        "Voice TTS",
        value=st.session_state.tts_enabled,
        help="Send LLM responses to the TTS engine. Disable to read only, no audio output.",
    )
    st.session_state.tts_enabled = val


# ─────────────────────────────────────────────────────────────────────────── #
# Sidebar                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

with st.sidebar:
    st.subheader("Servers")

    q5_url    = _normalize_base_url(st.text_input("Q5 (Omni) base URL", value=CONFIG.q5_base_url))
    q6_url    = _normalize_base_url(st.text_input("Q6 base URL",        value=CONFIG.q6_base_url))
    embed_url = _normalize_base_url(st.text_input("Embed base URL",     value=CONFIG.embed_base_url))

    session = InferenceSession.from_urls(
        q5_url=q5_url,
        q6_url=q6_url,
        embed_url=embed_url,
        q5_model_id=st.session_state.q5_model_id,
        q6_model_id=st.session_state.q6_model_id,
    )

    if st.button("Stop servers"):
        ok5, ok6, ok_e = session.stop_all()
        if ok5 and ok6 and ok_e:
            st.success("Stopped (or already stopped).")
        else:
            st.warning("Tried to stop, but one or more servers may still be running.")

    _render_server_status()
    _tts_toggle()

    if st.button("Discover model IDs"):
        mid5, mid6 = session.discover_model_ids(timeouts_status)
        if mid5: st.session_state.q5_model_id = mid5
        if mid6: st.session_state.q6_model_id = mid6
        st.success(
            f"Model IDs: Q5={st.session_state.q5_model_id} | Q6={st.session_state.q6_model_id}"
        )

    st.subheader("Routing")
    deep_mode      = st.toggle("Deep mode (force Architect this turn)", value=False)
    auto_escalate  = st.toggle("Auto-escalate when needed", value=True)

    if st.session_state.last_route is not None:
        rd = st.session_state.last_route
        st.caption(f"\U0001F9ED Last route: {rd.brain} \u2022 score={rd.score}")
        if rd.reasons:
            st.caption("Reasons: " + ", ".join(rd.reasons[:6]))

    st.subheader("Prompt Templates")
    auto_templates     = st.toggle("Auto templates", value=True)
    override_templates = st.multiselect(
        "Template overrides (optional)",
        options=sorted(list(TemplateKey), key=lambda t: t.value),
        default=[],
        format_func=lambda t: t.value,
        help="If selected, these replace auto-templates for this turn.",
    )

    st.subheader("Generation")
    st.caption("FAST = Qwen2.5-Omni-7B  |  ARCH = Qwen3.5-27B (thinking)")
    temperature_q5  = st.slider("FAST temperature",  0.0, 2.0,  0.70, 0.05,
        help="Qwen2.5-Omni-7B. Model-card default: 0.7")
    temperature_q6  = st.slider("ARCH temperature",  0.0, 2.0,  0.60, 0.05,
        help="Qwen3.5-27B thinking mode. Model-card default: 0.6")
    top_p_q5        = st.slider("FAST top_p",        0.0, 1.0,  0.80, 0.01,
        help="Nucleus sampling. Model-card default: 0.80")
    top_p_q6        = st.slider("ARCH top_p",        0.0, 1.0,  0.95, 0.01,
        help="Nucleus sampling. Model-card default: 0.95")
    top_k_q5        = st.slider("FAST top_k",        1,   200,  40,   1,
        help="Top-K sampling. llama.cpp default: 40")
    top_k_q6        = st.slider("ARCH top_k",        1,   200,  20,   1,
        help="Top-K sampling. Model-card default: 20")
    min_p_q5        = st.slider("FAST min_p",        0.0, 0.5,  0.05, 0.01,
        help="Min-P sampling. llama.cpp default: 0.05")
    min_p_q6        = st.slider("ARCH min_p",        0.0, 0.5,  0.00, 0.01,
        help="Min-P sampling. Model-card default: 0.0 (disabled)")
    max_tokens_q5   = st.selectbox("FAST max_tokens", [1024, 2048, 4096, 8192, 16384], index=2)
    max_tokens_q6   = st.selectbox("ARCH max_tokens", [4096, 8192, 16384, 32768, 65536], index=3)

    if st.session_state.last_thinking_time is not None:
        st.caption(f"\u23F1\uFE0F Last thinking time: {st.session_state.last_thinking_time:.2f}s")

    st.subheader("Feedback Dataset")
    if st.session_state.fb_stats_dirty:
        try:
            from feedback.db import fetch_stats, get_conn as _fb_get_conn
            from pg_settings import PgSettings as _FeedbackSettings
            _fb_cfg = _FeedbackSettings()
            with _fb_get_conn(_fb_cfg.pg_dsn) as _fb_conn:
                st.session_state.fb_stats = fetch_stats(_fb_conn)
            st.session_state.fb_stats_dirty = False
        except Exception:
            st.session_state.fb_stats = {}
    _s = st.session_state.get("fb_stats", {})
    if _s:
        st.caption(
            f"\U0001F44D {_s.get('thumbs_up', 0)} / \U0001F44E {_s.get('thumbs_down', 0)}"
            f" \u2022 FAST: {_s.get('fast_up', 0)}\u2191 {_s.get('fast_down', 0)}\u2193"
            f" \u2022 ARCH: {_s.get('arch_up', 0)}\u2191 {_s.get('arch_down', 0)}\u2193"
        )
    st.caption("`python -m feedback --stats` \u2022 `--out kto.jsonl`")

    st.subheader("Wikipedia")
    wiki_enabled = st.checkbox(
        "Include Wikipedia",
        value=True,
        key="wiki_enabled",
        help="Retrieve relevant Wikipedia passages and images (requires wiki embed on port 8031).",
    )

    st.subheader("Video frame extraction")
    frames_per_second = st.slider(
        "Frames per second to sample",
        min_value=0.1, max_value=2.0, value=0.5, step=0.1,
        help="How many frames to capture per second of video. 0.5 = 1 frame every 2 seconds.",
    )
    max_video_frames = st.slider(
        "Max frames per video",
        min_value=1, max_value=16, value=8,
        help="Hard cap on frames extracted per video file to limit context token usage.",
    )

    if st.button("New chat"):
        st.session_state.messages            = []
        st.session_state.last_thinking_time  = None
        st.session_state.last_route          = None
        st.session_state.pending_attachments = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────── #
# Chat history                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        DiagramHandler.render_if_present(m["content"])

        # Show any media thumbnails stored with the message
        if m.get("media_labels"):
            st.caption("\U0001F4CE " + ", ".join(m["media_labels"]))

        if m.get("thinking"):
            with st.expander("Developer Mode Reasoning", expanded=False):
                st.markdown(m["thinking"])

        if m["role"] == "assistant" and m.get("id"):
            msg_id: str = m["id"]

            if m.get("thumb") is not None or msg_id in st.session_state.fb_rated_ids:
                icon = "\U0001F44D" if m.get("thumb", 0) == 1 else "\U0001F44E"
                st.caption(f"{icon} Rated  \u2022  {m.get('model_used', '')}")
            else:
                note_val = st.text_input(
                    "Note",
                    key=f"note_{msg_id}",
                    placeholder="Wrong facts, too verbose, etc. (optional)",
                    label_visibility="collapsed",
                )
                tb_c1, tb_c2, _ = st.columns([1, 1, 10])
                with tb_c1:
                    thumb_up   = st.button("\U0001F44D", key=f"up_{msg_id}")
                with tb_c2:
                    thumb_down = st.button("\U0001F44E", key=f"dn_{msg_id}")

                if thumb_up or thumb_down:
                    _thumb  = 1 if thumb_up else -1
                    _saved  = False
                    try:
                        from feedback.db import insert_rating, get_conn as _fb_get_conn
                        from pg_settings import PgSettings as _FeedbackSettings
                        _meta   = m.get("meta", {})
                        _fb_cfg = _FeedbackSettings()
                        with _fb_get_conn(_fb_cfg.pg_dsn) as _fb_conn:
                            insert_rating(
                                _fb_conn,
                                id=msg_id,
                                brain=_meta.get("brain", "FAST"),
                                model_id=_meta.get("model_id", ""),
                                endpoint=_meta.get("endpoint", ""),
                                route_score=float(_meta.get("route_score", 0)),
                                route_reasons=list(_meta.get("route_reasons", [])),
                                templates=list(_meta.get("templates", [])),
                                prompt_messages=list(_meta.get("prompt_messages", [])),
                                assistant_text=m["content"],
                                thumb=_thumb,
                                notes=note_val or "",
                            )
                        _saved = True
                    except Exception as _e:
                        st.warning(f"Feedback not saved: {_e}")
                    if _saved:
                        m["thumb"] = _thumb
                        st.session_state.fb_rated_ids.add(msg_id)
                        st.session_state.fb_stats_dirty = True
                        st.rerun()


# ─────────────────────────────────────────────────────────────────────────── #
# Media upload (above chat input)                                              #
# ─────────────────────────────────────────────────────────────────────────── #

with st.expander(
    "\U0001F4CE Attach media (image / audio / video)",
    expanded=bool(st.session_state.pending_attachments),
):
    uploaded_files = st.file_uploader(
        "Upload files — images, audio, or video",
        type=_ALL_ACCEPTED,
        accept_multiple_files=True,
        key="media_uploader",
        help=(
            "Images: PNG, JPG, WEBP, GIF, BMP\n"
            "Audio:  WAV, MP3, FLAC, OGG, M4A  (sent to Qwen2.5-Omni audio encoder)\n"
            "Video:  MP4, MOV, AVI, MKV, WEBM  (frames extracted; requires cv2)"
        ),
    )

    if uploaded_files:
        new_attachments = _collect_attachments(
            uploaded_files, frames_per_second, max_video_frames
        )
        # Replace pending attachments each time the uploader changes
        st.session_state.pending_attachments = new_attachments

    _render_attachments_preview(st.session_state.pending_attachments)

    if st.session_state.pending_attachments:
        if st.button("Clear attachments"):
            st.session_state.pending_attachments = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────── #
# Voice "new chat" command handler                                             #
# ─────────────────────────────────────────────────────────────────────────── #
# Processed before the normal chat input so the command is never forwarded
# to the LLM.  Mirrors the "New chat" sidebar button exactly.

if st.session_state.pop("_voice_new_chat", False):
    st.session_state.messages            = []
    st.session_state.last_thinking_time  = None
    st.session_state.last_route          = None
    st.session_state.pending_attachments = []
    st.session_state.pop("_voice_input", None)  # discard any voice input queued before the reset
    st.toast("\U0001F3A4 New chat started", icon="\U0001F5D1\uFE0F")
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────── #
# Chat input + turn execution                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

_pending_voice = st.session_state.pop("_voice_input", None)
user_text = _pending_voice or st.chat_input("Ask Sage Kaizen\u2026", key="chat_input")

if user_text:
    # Hard-reject structural injection patterns before any processing
    try:
        check_user_input(user_text)
    except InjectionDetectedError as _inj:
        st.error(f"Request blocked: {_inj}")
        st.stop()

    # Snapshot attachments for this turn then clear the pending list
    turn_attachments: Tuple[MediaAttachment, ...] = tuple(
        st.session_state.pending_attachments
    )
    st.session_state.pending_attachments = []

    media_labels = [a.label for a in turn_attachments if a.label]
    if _pending_voice:
        media_labels = ["\U0001f3a4 Voice"] + media_labels

    # Store user message (text only for display; attachments shown via labels)
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "media_labels": media_labels,
    })
    with st.chat_message("user"):
        st.markdown(user_text)
        if media_labels:
            st.caption("\U0001F4CE " + ", ".join(media_labels))

    cfg = TurnConfig(
        deep_mode=deep_mode,
        auto_escalate=auto_escalate,
        auto_templates=auto_templates,
        override_templates=tuple(override_templates),
        temperature_q5=temperature_q5,
        temperature_q6=temperature_q6,
        top_p_q5=top_p_q5,
        top_p_q6=top_p_q6,
        top_k_q5=int(top_k_q5),
        top_k_q6=int(top_k_q6),
        min_p_q5=float(min_p_q5),
        min_p_q6=float(min_p_q6),
        max_tokens_q5=int(max_tokens_q5),
        max_tokens_q6=int(max_tokens_q6),
        media_attachments=turn_attachments,
    )

    session = InferenceSession.from_urls(
        q5_url=q5_url,
        q6_url=q6_url,
        embed_url=embed_url,
        q5_model_id=st.session_state.q5_model_id,
        q6_model_id=st.session_state.q6_model_id,
    )

    chat_svc = ChatService(session, CONFIG.system_prompt, timeouts)

    # ── Routing ─────────────────────────────────────────────────────────────
    # Voice path: instant keyword heuristic → spinner appears immediately,
    #             LLM routing runs in a background thread in parallel.
    # Keyboard path: existing sequential decide_route() (LLM routing included).
    _route_future: Optional[concurrent.futures.Future] = None
    if _pending_voice:
        decision = _heuristic_route(user_text, voice_mode=True)
        _route_future = _ROUTE_EXECUTOR.submit(
            _llm_route,
            user_text=user_text,
            fast_base_url=session.q5_url,
            model_id=session.q5_model_id,
            timeouts=HttpTimeouts(connect_s=1.0, read_s=2.5),
        )
    else:
        decision = chat_svc.decide_route(user_text, cfg)

    st.session_state.last_route = decision
    use_q6 = decision.brain == "ARCHITECT"
    brain_label = "Architect (Q6)" if use_q6 else "Omni (Q5)"

    with st.chat_message("assistant"):
        reasons_str = ", ".join(decision.reasons[:6])
        st.caption(
            f"\U0001F9ED Route: {decision.brain} \u2022 score={decision.score} \u2022 "
            f"reasons: {reasons_str}"
        )
        templates_caption  = st.empty()  # filled once select_templates returns
        route_update_caption = st.empty()  # shows if parallel LLM route refines brain

        live  = st.empty()
        start = time.time()

        with st.spinner(f"Thinking\u2026 ({brain_label})"):
            # ── Server health check (inside spinner — doesn't delay label) ──
            with st.status("Checking servers\u2026", expanded=False) as s:
                ok, msg = session.ensure_q5_ready()
                if not ok:
                    s.update(label="Q5 not ready \u274C", state="error")
                    st.error(msg)
                    st.stop()

                if use_q6:
                    s.write("Starting **Architect Brain** (Q6)\u2026")
                    ok, msg = session.ensure_q6_ready()
                    if not ok:
                        s.update(label="Q6 not ready \u274C", state="error")
                        st.error(msg)
                        st.stop()
                    s.write("\u2705 **Architect Brain** ready.")

                s.update(label="Servers ready \u2705", state="complete")

            # ── Resolve parallel LLM routing result (voice path only) ───────
            # By the time ensure_q5_ready() returns, the background thread has
            # had ~20–50 ms head-start; most calls will already be done.
            if _route_future is not None:
                try:
                    llm_decision: RouteDecision = _route_future.result(timeout=2.0)
                    if llm_decision.brain != decision.brain:
                        decision  = llm_decision
                        use_q6    = decision.brain == "ARCHITECT"
                        st.session_state.last_route = decision
                        route_update_caption.caption(
                            f"\U0001F504 Route refined: {decision.brain} \u2022 "
                            f"score={decision.score}"
                        )
                        if use_q6:
                            with st.status(
                                "Starting Architect Brain\u2026", expanded=False
                            ) as sq6:
                                ok6, msg6 = session.ensure_q6_ready()
                                if not ok6:
                                    sq6.update(label="Q6 not ready \u274C", state="error")
                                    st.error(msg6)
                                    st.stop()
                                sq6.update(
                                    label="\u2705 Architect Brain ready.", state="complete"
                                )
                except Exception:
                    pass  # LLM route timed out or failed — keep heuristic result
            templates = chat_svc.select_templates(user_text, cfg)
            templates_str = ", ".join(t.value for t in templates) if templates else "(none)"
            templates_caption.caption(f"\U0001F9E9 Templates: {templates_str}")

            history: List[dict] = st.session_state.messages[-CONFIG.max_history_messages:]
            messages, rag_sources, wiki_images, search_evidence, music_context = chat_svc.prepare_messages(
                user_text, history, decision, templates,
                wiki_enabled=st.session_state.get("wiki_enabled", True),
                media_attachments=turn_attachments,
            )

            # Snapshot tts_enabled once so the generator is not sensitive to
            # mid-turn widget changes (toggling the sidebar fragment would not
            # trigger a full rerun, but a snapshot is still safer).
            _tts_on = bool(st.session_state.tts_enabled)

            voice_session_id = str(uuid4())
            if _tts_on:
                _voice_bridge.start_turn(voice_session_id, decision.brain)
            full_streamed = ""
            try:
                def _voice_stream():
                    for piece in chat_svc.stream_response(messages, decision, cfg):
                        if _tts_on and _voice_bridge.barge_in_event.is_set():
                            _voice_bridge.barge_in_event.clear()
                            return
                        if _tts_on:
                            _voice_bridge.publish_token(voice_session_id, piece)
                        yield piece
                full_streamed = live.write_stream(_voice_stream()) or ""
            except LlamaServerError as e:
                live.error(str(e))
            except Exception as e:
                live.error(f"Unexpected error: {type(e).__name__}: {e}")
            finally:
                if _tts_on:
                    _voice_bridge.end_turn(voice_session_id)

        elapsed  = time.time() - start
        st.session_state.last_thinking_time = elapsed
        final    = full_streamed.strip()
        _thinking, _clean = _parse_response(final)
        _sources_md = format_sources_markdown(rag_sources)
        if _sources_md:
            _clean = _clean + "\n\n" + _sources_md

        # Live web search citations
        if search_evidence is not None and not search_evidence.empty:
            try:
                from search.citations import format_search_sources_markdown as _fmt_search
                _search_md = _fmt_search(search_evidence)
                if _search_md:
                    _clean = _clean + "\n\n" + _search_md
            except Exception:
                pass

        if final:
            live.markdown(_clean)
            DiagramHandler.render_if_present(_clean)

            # Wikipedia images
            if wiki_images:
                import os as _os
                _valid_imgs = [img for img in wiki_images if _os.path.isfile(img.absolute_path)]
                if _valid_imgs:
                    st.markdown("**Wikipedia images**")
                    _cols = st.columns(min(len(_valid_imgs), 3))
                    for _col, _img in zip(_cols, _valid_imgs):
                        with _col:
                            st.image(_img.absolute_path, caption=_img.caption_text, width='stretch')

            # Music retrieval results
            if music_context:
                with st.expander("Music results", expanded=True):
                    st.code(music_context, language=None)

            if _thinking:
                with st.expander("Developer Mode Reasoning", expanded=False):
                    st.markdown(_thinking)

            _endpoint    = session.url_for_brain(decision.brain)
            _model_id    = session.model_id_for_brain(decision.brain)
            _new_msg_id  = str(uuid4())
            st.caption(
                f"\u23F1\uFE0F {elapsed:.2f}s \u2022 Brain: {brain_label} \u2022 "
                f"Endpoint: {_endpoint}"
            )
            if turn_attachments:
                st.caption(
                    f"\U0001F4CE {len(turn_attachments)} attachment(s): "
                    + ", ".join(a.kind for a in turn_attachments)
                )

            st.session_state.messages.append({
                "id":         _new_msg_id,
                "role":       "assistant",
                "content":    _clean,
                "thinking":   _thinking or "",
                "model_used": f"{_model_id} ({decision.brain})",
                "meta": {
                    "brain":           decision.brain,
                    "endpoint":        _endpoint,
                    "route_score":     decision.score,
                    "route_reasons":   list(decision.reasons),
                    "templates":       [t.value for t in templates],
                    "prompt_messages": messages,
                    "model_id":        _model_id,
                    "ts":              time.time(),
                    "has_media":       bool(turn_attachments),
                },
                "thumb": None,
            })

            # Inline feedback thumbs
            _note_live = st.text_input(
                "Note",
                key=f"note_{_new_msg_id}",
                placeholder="Wrong facts, too verbose, etc. (optional)",
                label_visibility="collapsed",
            )
            _tb1, _tb2, _ = st.columns([1, 1, 10])
            with _tb1:
                _up_live = st.button("\U0001F44D", key=f"up_{_new_msg_id}")
            with _tb2:
                _dn_live = st.button("\U0001F44E", key=f"dn_{_new_msg_id}")
            if _up_live or _dn_live:
                _thumb_live  = 1 if _up_live else -1
                _saved_live  = False
                try:
                    from feedback.db import insert_rating, get_conn as _fb_get_conn
                    from pg_settings import PgSettings as _FeedbackSettings
                    _fb_cfg = _FeedbackSettings()
                    with _fb_get_conn(_fb_cfg.pg_dsn) as _fb_conn:
                        insert_rating(
                            _fb_conn,
                            id=_new_msg_id,
                            brain=decision.brain,
                            model_id=_model_id,
                            endpoint=_endpoint,
                            route_score=float(decision.score),
                            route_reasons=list(decision.reasons),
                            templates=[t.value for t in templates],
                            prompt_messages=list(messages),
                            assistant_text=_clean,
                            thumb=_thumb_live,
                            notes=_note_live or "",
                        )
                    _saved_live = True
                except Exception as _e:
                    st.warning(f"Feedback not saved: {_e}")
                if _saved_live:
                    st.session_state.messages[-1]["thumb"] = _thumb_live
                    st.session_state.fb_rated_ids.add(_new_msg_id)
                    st.session_state.fb_stats_dirty = True
                    st.rerun()
