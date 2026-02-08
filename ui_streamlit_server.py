
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from settings import CONFIG
from openai_client import (
    HttpTimeouts,
    LlamaServerError,
    discover_model_id,
    health_check,
    stream_chat_completions,
)
from router import should_escalate_to_q6
from server_manager import ManagedServers, ensure_q5_running, ensure_q6_running, stop_server_on_port


def _load_system_prompt(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return (
        "You are Sage Kaizen. Prefer correctness over speed. "
        "Be structured. If unsure, say so and ask for missing info."
    )


def _base_url_to_port(url: str) -> Optional[int]:
    # http://127.0.0.1:8011 -> 8011
    try:
        if ":" in url:
            return int(url.rstrip("/").split(":")[-1])
    except Exception:
        return None
    return None


def _clamp_history_by_chars(history: List[Dict[str, str]], max_chars: int) -> List[Dict[str, str]]:
    """
    Keeps the newest messages while total content stays under max_chars.
    This is a crude but effective guard when server ctx is 4096.
    """
    kept: List[Dict[str, str]] = []
    total = 0
    for msg in reversed(history):
        c = msg.get("content", "")
        add = len(c)
        if kept and total + add > max_chars:
            break
        if not kept and add > max_chars:
            # keep at least the newest message, truncated
            kept.append({"role": msg.get("role", "user"), "content": c[-max_chars:]})
            break
        kept.append(msg)
        total += add
    kept.reverse()
    return kept


def _approx_tokens_from_text(s: str) -> int:
    # Quick heuristic for UI metrics (English-ish): ~4 chars per token
    return max(1, len(s) // 4)


st.set_page_config(page_title="Sage Kaizen (llama-server)", page_icon="🧠", layout="wide")
st.title("🧠 Sage Kaizen (Dual-Brain Chat) 🧠")

system_prompt = _load_system_prompt(CONFIG.system_prompt_path)
timeouts = HttpTimeouts(connect_s=CONFIG.connect_timeout_s, read_s=CONFIG.read_timeout_s)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict[str,str]]
if "last_thinking_time" not in st.session_state:
    st.session_state.last_thinking_time = None
if "q5_model_id" not in st.session_state:
    st.session_state.q5_model_id = CONFIG.q5_model_id
if "q6_model_id" not in st.session_state:
    st.session_state.q6_model_id = CONFIG.q6_model_id

# Optional: cache discovered model IDs to avoid repeated /v1/models calls
if "model_ids_discovered" not in st.session_state:
    st.session_state.model_ids_discovered = False

# Settings for history clamp (fallback if CONFIG doesn't define)
HISTORY_CHAR_LIMIT = getattr(CONFIG, "history_char_limit", 24000)

with st.sidebar:
    st.subheader("Servers")

    q5_url = st.text_input("Q5 base URL", value=CONFIG.q5_base_url)
    q6_url = st.text_input("Q6 base URL", value=CONFIG.q6_base_url)

    q5_port = _base_url_to_port(q5_url) or 8011
    q6_port = _base_url_to_port(q6_url) or 8012

    servers = ManagedServers(
        host="127.0.0.1",
        q5_port=q5_port,
        q6_port=q6_port,
        start_q5_bat=Path("start_q5_server.bat"),
        start_q6_bat=Path("start_q6_server.bat"),
        q5_log=Path("logs/q5_server.log"),
        q6_log=Path("logs/q6_server.log"),
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("Start Q5"):
            with st.status("Starting Q5…", expanded=True) as s:
                ok, msg = ensure_q5_running(servers)
                if ok:
                    s.update(label="Q5 ready ✅", state="complete")
                    st.success(msg)
                else:
                    s.update(label="Q5 failed ❌", state="error")
                    st.error(msg)

        if st.button("Start Q6"):
            with st.status("Starting Q6…", expanded=True) as s:
                ok, msg = ensure_q6_running(servers)
                if ok:
                    s.update(label="Q6 ready ✅", state="complete")
                    st.success(msg)
                else:
                    s.update(label="Q6 failed ❌", state="error")
                    st.error(msg)

    with colB:
        if st.button("Stop servers"):
            ok1 = stop_server_on_port(q5_port)
            ok2 = stop_server_on_port(q6_port)
            if ok1 and ok2:
                st.success("Stopped (or already stopped).")
            else:
                st.warning("Tried to stop, but one or more servers may still be running.")

    # Live status (lightweight)
    ok5, d5 = health_check(q5_url, timeouts=timeouts)
    ok6, d6 = health_check(q6_url, timeouts=timeouts)
    st.caption(f"Q5: {'✅' if ok5 else '❌'} {d5}")
    st.caption(f"Q6: {'✅' if ok6 else '❌'} {d6}")

    if st.button("Discover model IDs"):
        mid5 = discover_model_id(q5_url, timeouts=timeouts)
        mid6 = discover_model_id(q6_url, timeouts=timeouts)
        if mid5:
            st.session_state.q5_model_id = mid5
        if mid6:
            st.session_state.q6_model_id = mid6
        st.session_state.model_ids_discovered = True
        st.success(f"Model IDs: Q5={st.session_state.q5_model_id} | Q6={st.session_state.q6_model_id}")

    st.subheader("Routing")
    deep_mode = st.toggle("Deep mode (force Q6 this turn)", value=False)
    auto_escalate = st.toggle("Auto-escalate when needed", value=True)

    st.subheader("Generation")
    temperature_q5 = st.slider("Q5 temperature", 0.0, 1.2, 0.4, 0.05)
    temperature_q6 = st.slider("Q6 temperature", 0.0, 1.2, 0.6, 0.05)
    top_p_q5 = st.slider("Q5 top_p", 0.1, 1.0, 0.92, 0.01)
    top_p_q6 = st.slider("Q6 top_p", 0.1, 1.0, 0.95, 0.01)

    # Safer defaults for MoE IQ1 models
    max_tokens_q5 = st.selectbox("Q5 max_tokens", [512, 1024, 2048, 4096, 8192], index=2)  # 2048
    max_tokens_q6 = st.selectbox("Q6 max_tokens", [1024, 2048, 4096, 8192, 16384], index=2)  # 4096

    if st.session_state.last_thinking_time is not None:
        st.caption(f"⏱️ Last thinking time: {st.session_state.last_thinking_time:.2f}s")

    if st.button("New chat"):
        st.session_state.messages = []
        st.session_state.last_thinking_time = None
        st.rerun()


# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Ask Sage Kaizen…")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Decide brain for this turn BEFORE starting servers
    use_q6 = should_escalate_to_q6(user_text, deep_mode) if auto_escalate else deep_mode
    brain_name = "Q6" if use_q6 else "Q5"
    base_url = q6_url if use_q6 else q5_url
    model_id = st.session_state.q6_model_id if use_q6 else st.session_state.q5_model_id
    temperature = float(temperature_q6 if use_q6 else temperature_q5)
    top_p = float(top_p_q6 if use_q6 else top_p_q5)
    max_tokens = int(max_tokens_q6 if use_q6 else max_tokens_q5)

    # Ensure Q5 is ready always; Q6 only if needed
    with st.status("Checking servers…", expanded=False) as s:
        ok, msg = ensure_q5_running(servers)
        if not ok:
            s.update(label="Q5 not ready ❌", state="error")
            st.error(msg)
            st.stop()

        if use_q6:
            ok, msg = ensure_q6_running(servers)
            if not ok:
                s.update(label="Q6 not ready ❌", state="error")
                st.error(msg)
                st.stop()

        s.update(label="Servers ready ✅", state="complete")

    # Discover model IDs once automatically (optional but nice in production)
    if not st.session_state.model_ids_discovered:
        mid5 = discover_model_id(q5_url, timeouts=timeouts)
        mid6 = discover_model_id(q6_url, timeouts=timeouts)
        if mid5:
            st.session_state.q5_model_id = mid5
        if mid6:
            st.session_state.q6_model_id = mid6
        st.session_state.model_ids_discovered = True
        # refresh this turn's model_id if it changed
        model_id = st.session_state.q6_model_id if use_q6 else st.session_state.q5_model_id

    # Build OpenAI-compatible messages
    history = st.session_state.messages[-CONFIG.max_history_messages:]
    history = _clamp_history_by_chars(history, HISTORY_CHAR_LIMIT)

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    with st.chat_message("assistant"):
        live = st.empty()
        acc: List[str] = []
        start = time.time()
        first_token_t: Optional[float] = None

        with st.spinner(f"Thinking… ({brain_name})"):
            try:
                for piece in stream_chat_completions(
                    base_url=base_url,
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeouts=timeouts,
                ):
                    if first_token_t is None:
                        first_token_t = time.time()
                    acc.append(piece)
                    live.markdown("".join(acc))
            except LlamaServerError as e:
                live.error(str(e))
            except Exception as e:
                live.error(f"Unexpected error: {type(e).__name__}: {e}")

        end = time.time()
        elapsed = end - start
        st.session_state.last_thinking_time = elapsed
        final = "".join(acc).strip()

        if final:
            live.markdown(final)
            ttft = (first_token_t - start) if first_token_t else None
            tok_est = _approx_tokens_from_text(final)
            decode_s = (end - first_token_t) if first_token_t else None
            tps = (tok_est / decode_s) if (decode_s and decode_s > 0) else None

            metrics = [f"⏱️ Thinking: {elapsed:.2f}s", f"Brain: {brain_name}", f"Endpoint: {base_url}"]
            if ttft is not None:
                metrics.insert(1, f"TTFT: {ttft:.2f}s")
            if tps is not None:
                metrics.insert(2, f"≈tok/s: {tps:.2f}")

            st.caption(" • ".join(metrics))
            st.session_state.messages.append({"role": "assistant", "content": final})
