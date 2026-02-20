from __future__ import annotations

import time
from typing import List

import streamlit as st

from chat_service import ChatService, TurnConfig
from inference_session import InferenceSession
from mermaid_streamlit import DiagramHandler
from openai_client import HttpTimeouts, LlamaServerError
from prompt_library import TemplateKey
from settings import CONFIG


# ─────────────────────────────────────────────────────────────────────────── #
# UI helpers                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def _normalize_base_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[:-3].rstrip("/")
    return u


# ─────────────────────────────────────────────────────────────────────────── #
# Page config                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

st.set_page_config(page_title="Sage Kaizen (llama-server)", page_icon="\U0001F9E0", layout="wide")
st.title("\U0001F9E0 Sage Kaizen (Dual-Brain Chat) \U0001F9E0")

timeouts = HttpTimeouts(connect_s=CONFIG.connect_timeout_s, read_s=CONFIG.read_timeout_s)
timeouts_status = HttpTimeouts(connect_s=2.0, read_s=2.0)

# ─────────────────────────────────────────────────────────────────────────── #
# Session state                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_thinking_time" not in st.session_state:
    st.session_state.last_thinking_time = None
if "q5_model_id" not in st.session_state:
    st.session_state.q5_model_id = CONFIG.q5_model_id
if "q6_model_id" not in st.session_state:
    st.session_state.q6_model_id = CONFIG.q6_model_id
if "last_route" not in st.session_state:
    st.session_state.last_route = None

# ─────────────────────────────────────────────────────────────────────────── #
# Sidebar                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

with st.sidebar:
    st.subheader("Servers")

    q5_url = _normalize_base_url(st.text_input("Q5 base URL", value=CONFIG.q5_base_url))
    q6_url = _normalize_base_url(st.text_input("Q6 base URL", value=CONFIG.q6_base_url))
    embed_url = _normalize_base_url(st.text_input("Embed base URL", value=CONFIG.embed_base_url))

    session = InferenceSession.from_urls(
        q5_url=q5_url,
        q6_url=q6_url,
        embed_url=embed_url,
        q5_model_id=st.session_state.q5_model_id,
        q6_model_id=st.session_state.q6_model_id,
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("Start servers"):
            with st.status("Starting Embed \u2192 Q5 (Fast Brain)\u2026", expanded=True) as s:
                s.write("Waiting for **Fast Brain** (watch logs/q5_server.log)\u2026")
                ok, msg = session.ensure_q5_ready()
                if ok:
                    s.write("\u2705 **Fast Brain** loaded.")
                    s.update(label="Q5 ready \u2705", state="complete")
                    st.success(msg)
                    st.info("Q6 starts automatically when a turn escalates (or enable Deep mode).")
                else:
                    s.update(label="Failed to start Q5 \u274C", state="error")
                    st.error(msg)

    with colB:
        if st.button("Stop servers"):
            ok5, ok6, ok_e = session.stop_all()
            if ok5 and ok6 and ok_e:
                st.success("Stopped (or already stopped).")
            else:
                st.warning("Tried to stop, but one or more servers may still be running.")

    ok5, d5 = session.health_q5(timeouts_status)
    ok6, d6 = session.health_q6(timeouts_status)
    ok5_icon = "\u2705" if ok5 else "\u274C"
    ok6_icon = "\u2705" if ok6 else "\u274C"
    st.caption(f"Q5: {ok5_icon} {d5}")
    st.caption(f"Q6: {ok6_icon} {d6}")

    if st.button("Discover model IDs"):
        mid5, mid6 = session.discover_model_ids(timeouts_status)
        if mid5:
            st.session_state.q5_model_id = mid5
        if mid6:
            st.session_state.q6_model_id = mid6
        st.success(
            f"Model IDs: Q5={st.session_state.q5_model_id} | Q6={st.session_state.q6_model_id}"
        )

    st.subheader("Routing")
    deep_mode = st.toggle("Deep mode (force Architect this turn)", value=False)
    auto_escalate = st.toggle("Auto-escalate when needed", value=True)

    if st.session_state.last_route is not None:
        rd = st.session_state.last_route
        st.caption(f"\U0001F9ED Last route: {rd.brain} \u2022 score={rd.score}")
        if rd.reasons:
            st.caption("Reasons: " + ", ".join(rd.reasons[:6]))

    st.subheader("Prompt Templates")
    auto_templates = st.toggle("Auto templates", value=True)
    override_templates = st.multiselect(
        "Template overrides (optional)",
        options=list(TemplateKey),
        default=[],
        format_func=lambda t: t.value,
        help="If selected, these replace auto-templates for this turn.",
    )

    st.subheader("Generation")
    temperature_q5 = st.slider("Q5 temperature", 0.0, 1.2, 0.4, 0.05)
    temperature_q6 = st.slider("Q6 temperature", 0.0, 1.2, 0.6, 0.05)
    top_p_q5 = st.slider("Q5 top_p", 0.1, 1.0, 0.92, 0.01)
    top_p_q6 = st.slider("Q6 top_p", 0.1, 1.0, 0.95, 0.01)
    max_tokens_q5 = st.selectbox("Q5 max_tokens", [1024, 2048, 4096, 8192], index=2)
    max_tokens_q6 = st.selectbox("Q6 max_tokens", [2048, 4096, 8192, 16384], index=2)

    if st.session_state.last_thinking_time is not None:
        st.caption(f"\u23F1\uFE0F Last thinking time: {st.session_state.last_thinking_time:.2f}s")

    if st.button("New chat"):
        st.session_state.messages = []
        st.session_state.last_thinking_time = None
        st.session_state.last_route = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────── #
# Chat                                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Ask Sage Kaizen\u2026")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    cfg = TurnConfig(
        deep_mode=deep_mode,
        auto_escalate=auto_escalate,
        auto_templates=auto_templates,
        override_templates=tuple(override_templates),
        temperature_q5=temperature_q5,
        temperature_q6=temperature_q6,
        top_p_q5=top_p_q5,
        top_p_q6=top_p_q6,
        max_tokens_q5=int(max_tokens_q5),
        max_tokens_q6=int(max_tokens_q6),
    )

    session = InferenceSession.from_urls(
        q5_url=q5_url,
        q6_url=q6_url,
        embed_url=embed_url,
        q5_model_id=st.session_state.q5_model_id,
        q6_model_id=st.session_state.q6_model_id,
    )

    chat_svc = ChatService(session, CONFIG.system_prompt, timeouts)

    decision = chat_svc.decide_route(user_text, cfg)
    st.session_state.last_route = decision
    use_q6 = decision.brain == "ARCHITECT"

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

    templates = chat_svc.select_templates(user_text, cfg)
    history: List[dict] = st.session_state.messages[-CONFIG.max_history_messages:]
    messages = chat_svc.prepare_messages(user_text, history, decision, templates)

    brain_label = "Architect (Q6)" if use_q6 else "Fast (Q5)"
    with st.chat_message("assistant"):
        reasons_str = ", ".join(decision.reasons[:6])
        templates_str = ", ".join(t.value for t in templates) if templates else "(none)"
        st.caption(
            f"\U0001F9ED Route: {decision.brain} \u2022 score={decision.score} \u2022 "
            f"reasons: {reasons_str}"
        )
        st.caption(f"\U0001F9E9 Templates: {templates_str}")

        live = st.empty()
        acc: List[str] = []
        start = time.time()

        with st.spinner(f"Thinking\u2026 ({brain_label})"):
            try:
                for piece in chat_svc.stream_response(messages, decision, cfg):
                    acc.append(piece)
                    live.markdown("".join(acc))
            except LlamaServerError as e:
                live.error(str(e))
            except Exception as e:
                live.error(f"Unexpected error: {type(e).__name__}: {e}")

        elapsed = time.time() - start
        st.session_state.last_thinking_time = elapsed
        final = "".join(acc).strip()

        if final:
            live.markdown(final)
            DiagramHandler.render_if_present(final)
            endpoint = session.url_for_brain(decision.brain)
            st.caption(
                f"\u23F1\uFE0F {elapsed:.2f}s \u2022 Brain: {brain_label} \u2022 "
                f"Endpoint: {endpoint}"
            )
            st.session_state.messages.append({"role": "assistant", "content": final})
