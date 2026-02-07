
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
from server_manager import ManagedServers, stop_server_on_port


# ----------------------------
# Helpers
# ----------------------------
def _load_system_prompt(path: Path) -> str:
    try:
        if path.exists():
            txt = path.read_text(encoding="utf-8").strip()
            if txt:
                return txt
    except Exception:
        pass
    return (
        "You are Sage Kaizen. Prefer correctness over speed. "
        "Be structured. If unsure, say so and ask for missing info."
    )


def _base_url_to_port(url: str) -> Optional[int]:
    try:
        if ":" in url:
            return int(url.rstrip("/").split(":")[-1])
    except Exception:
        return None
    return None


def _clamp_history_by_chars(history: List[Dict[str, str]], max_chars: int) -> List[Dict[str, str]]:
    """
    Keep the most recent messages up to an approximate character budget.
    This avoids overrunning the server ctx-size (we can't perfectly count tokens here).
    """
    kept: List[Dict[str, str]] = []
    total = 0
    for m in reversed(history):
        c = m.get("content", "")
        # roughly account for role + separators
        cost = len(c) + 32
        if kept and total + cost > max_chars:
            break
        kept.append(m)
        total += cost
        if total >= max_chars:
            break
    kept.reverse()
    return kept


def _start_bat_detached(bat_path: Path) -> Tuple[bool, str]:
    """
    Starts a .bat using cmd.exe. The .bat is expected to detach the server via `start "" /B ...`.
    We do NOT block here; readiness is checked separately via health endpoints.
    """
    if not bat_path.exists():
        return False, f"Missing: {bat_path}"
    try:
        import subprocess

        proc = subprocess.run(["cmd.exe", "/c", str(bat_path)], capture_output=True, text=True, shell=False)
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or f"bat exited {proc.returncode}").strip()
            return False, msg
        return True, "Started"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _wait_until_ready(base_url: str, timeouts: HttpTimeouts, timeout_s: float) -> Tuple[bool, str]:
    """
    Prefer readiness checks over port LISTENING:
    - /health if present
    - else /v1/models
    """
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        ok, detail = health_check(base_url, timeouts=timeouts)
        if ok:
            return True, detail
        last = detail
        time.sleep(0.75)
    return False, last or "Timed out"


def _ensure_one_server(
    base_url: str,
    bat_path: Path,
    timeouts: HttpTimeouts,
    ready_timeout_s: float,
) -> Tuple[bool, str]:
    """
    Ensures a specific llama-server is reachable. If not, attempts to start via its .bat then waits for readiness.
    """
    ok, detail = health_check(base_url, timeouts=timeouts)
    if ok:
        return True, detail

    started, msg = _start_bat_detached(bat_path)
    if not started:
        return False, f"Start failed: {msg}"

    ok2, detail2 = _wait_until_ready(base_url, timeouts=timeouts, timeout_s=ready_timeout_s)
    if ok2:
        return True, f"{msg} • {detail2}"
    return False, f"Started but not ready: {detail2}"


def _approx_tokens(text: str) -> float:
    """
    Rough token estimate (~4 chars/token). Good enough to compare runs on the same setup.
    """
    if not text:
        return 0.0
    return max(1.0, len(text) / 4.0)


# ----------------------------
# Streamlit app
# ----------------------------
st.set_page_config(page_title="Sage Kaizen (llama-server)", page_icon="🧠", layout="wide")
st.title("🧠 Sage Kaizen (llama-server dual brain)")

system_prompt = _load_system_prompt(CONFIG.system_prompt_path)
timeouts = HttpTimeouts(connect_s=CONFIG.connect_timeout_s, read_s=CONFIG.read_timeout_s)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None  # dict
if "q5_model_id" not in st.session_state:
    st.session_state.q5_model_id = CONFIG.q5_model_id
if "q6_model_id" not in st.session_state:
    st.session_state.q6_model_id = CONFIG.q6_model_id


with st.sidebar:
    st.subheader("Servers")

    q5_url = st.text_input("Q5 base URL", value=CONFIG.q5_base_url)
    q6_url = st.text_input("Q6 base URL", value=CONFIG.q6_base_url)

    q5_port = _base_url_to_port(q5_url) or 8011
    q6_port = _base_url_to_port(q6_url) or 8012

    servers = ManagedServers(
        q5_port=q5_port,
        q6_port=q6_port,
        start_q5_bat=Path("start_q5_server.bat"),
        start_q6_bat=Path("start_q6_server.bat"),
    )

    # Readiness timeouts (DeepSeek can take minutes to load)
    q5_ready_timeout_s = st.number_input("Q5 start timeout (s)", min_value=30, max_value=21600, value=1800, step=30)
    q6_ready_timeout_s = st.number_input("Q6 start timeout (s)", min_value=30, max_value=21600, value=2400, step=30)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Start Q5"):
            with st.status("Starting Q5 server…", expanded=True) as s:
                ok, msg = _ensure_one_server(q5_url, servers.start_q5_bat, timeouts, float(q5_ready_timeout_s))
                if ok:
                    s.update(label="Q5 ready ✅", state="complete")
                    st.success(msg)
                else:
                    s.update(label="Q5 failed ❌", state="error")
                    st.error(msg)

        if st.button("Start Q6"):
            with st.status("Starting Q6 server…", expanded=True) as s:
                ok, msg = _ensure_one_server(q6_url, servers.start_q6_bat, timeouts, float(q6_ready_timeout_s))
                if ok:
                    s.update(label="Q6 ready ✅", state="complete")
                    st.success(msg)
                else:
                    s.update(label="Q6 failed ❌", state="error")
                    st.error(msg)

    with colB:
        if st.button("Stop Q5"):
            st.success("Stopped." if stop_server_on_port(q5_port) else "Stop attempted (check netstat).")
        if st.button("Stop Q6"):
            st.success("Stopped." if stop_server_on_port(q6_port) else "Stop attempted (check netstat).")

    # Live status
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
        st.success(f"Model IDs: Q5={st.session_state.q5_model_id} | Q6={st.session_state.q6_model_id}")

    st.subheader("Routing")
    deep_mode = st.toggle("Deep mode (force Q6 this turn)", value=False)
    auto_escalate = st.toggle("Auto-escalate when needed", value=True)
    start_q6_on_demand = st.toggle("Start Q6 only when needed", value=True)

    st.subheader("Generation (accuracy-first)")
    # Conservative defaults for IQ1 models
    temperature_q5 = st.slider("Q5 temperature", 0.0, 1.2, 0.4, 0.05)
    temperature_q6 = st.slider("Q6 temperature", 0.0, 1.2, 0.6, 0.05)
    top_p_q5 = st.slider("Q5 top_p", 0.1, 1.0, 0.92, 0.01)
    top_p_q6 = st.slider("Q6 top_p", 0.1, 1.0, 0.95, 0.01)

    # Safer default limits to avoid "forever thinking"
    max_tokens_q5 = st.selectbox("Q5 max_tokens", [512, 1024, 2048, 4096, 8192], index=2)
    max_tokens_q6 = st.selectbox("Q6 max_tokens", [1024, 2048, 4096, 8192, 16384], index=2)

    # Soft time limit (because Streamlit can't truly cancel mid-run)
    abort_after_s = st.number_input("Abort generation after (s)", min_value=10, max_value=7200, value=1800, step=10)

    # History clamp
    max_history_msgs = st.number_input("Max history messages", min_value=4, max_value=64, value=int(CONFIG.max_history_messages), step=1)
    max_history_chars = st.number_input("Max history chars (approx)", min_value=2000, max_value=200000, value=30000, step=1000)

    if st.session_state.last_metrics:
        m = st.session_state.last_metrics
        st.caption(
            f"Last: brain={m.get('brain')} • TTFT={m.get('ttft_s'):.2f}s • "
            f"tok/s≈{m.get('tok_s'):.2f} • total={m.get('total_s'):.2f}s"
        )

    if st.button("New chat"):
        st.session_state.messages = []
        st.session_state.last_metrics = None
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

    # Decide brain for this turn
    use_q6 = should_escalate_to_q6(user_text, deep_mode) if auto_escalate else deep_mode
    brain_name = "Q6" if use_q6 else "Q5"
    base_url = q6_url if use_q6 else q5_url
    model_id = st.session_state.q6_model_id if use_q6 else st.session_state.q5_model_id
    temperature = float(temperature_q6 if use_q6 else temperature_q5)
    top_p = float(top_p_q6 if use_q6 else top_p_q5)
    max_tokens = int(max_tokens_q6 if use_q6 else max_tokens_q5)

    # Ensure Q5 is ready; ensure Q6 only if needed (recommended for IQ1_M)
    with st.status("Ensuring server readiness…", expanded=False) as s:
        ok, msg = _ensure_one_server(q5_url, servers.start_q5_bat, timeouts, float(q5_ready_timeout_s))
        if not ok:
            s.update(label="Q5 not ready ❌", state="error")
            st.error(msg)
            st.stop()
        if use_q6:
            if start_q6_on_demand:
                ok2, msg2 = _ensure_one_server(q6_url, servers.start_q6_bat, timeouts, float(q6_ready_timeout_s))
                if not ok2:
                    s.update(label="Q6 not ready ❌", state="error")
                    st.error(msg2)
                    st.stop()
            else:
                ok2, detail2 = health_check(q6_url, timeouts=timeouts)
                if not ok2:
                    s.update(label="Q6 not reachable ❌", state="error")
                    st.error(detail2)
                    st.stop()

        s.update(label="Servers ready ✅", state="complete")

    # Discover model id if still default placeholder and server supports /v1/models
    if model_id in ("Q5", "Q6", ""):
        discovered = discover_model_id(base_url, timeouts=timeouts)
        if discovered:
            model_id = discovered
            if use_q6:
                st.session_state.q6_model_id = discovered
            else:
                st.session_state.q5_model_id = discovered

    # Build messages with clamps
    history = st.session_state.messages[-int(max_history_msgs):]
    history = _clamp_history_by_chars(history, int(max_history_chars))
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    with st.chat_message("assistant"):
        live = st.empty()
        acc: List[str] = []

        start = time.time()
        first_token_time: Optional[float] = None

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
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                    acc.append(piece)
                    live.markdown("".join(acc))

                    if (now - start) > float(abort_after_s):
                        acc.append("\n\n*(Stopped: exceeded time limit.)*")
                        break

            except LlamaServerError as e:
                live.error(str(e))
            except Exception as e:
                live.error(f"Unexpected error: {type(e).__name__}: {e}")

        end = time.time()
        final = "".join(acc).strip()
        if final:
            live.markdown(final)
            st.session_state.messages.append({"role": "assistant", "content": final})

        total_s = end - start
        ttft_s = (first_token_time - start) if first_token_time else total_s
        decode_s = max(0.001, end - (first_token_time or start))
        tok_est = _approx_tokens(final)
        tok_s = tok_est / decode_s

        st.session_state.last_metrics = {
            "brain": brain_name,
            "total_s": float(total_s),
            "ttft_s": float(ttft_s),
            "tok_s": float(tok_s),
            "tok_est": float(tok_est),
        }

        st.caption(
            f"⏱️ Total: {total_s:.2f}s • TTFT: {ttft_s:.2f}s • tok/s≈{tok_s:.2f} (≈{tok_est:.0f} tok) "
            f"• Brain: {brain_name} • Endpoint: {base_url}"
        )
