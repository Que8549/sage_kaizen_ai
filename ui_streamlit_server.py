from __future__ import annotations

import time
import router 
import streamlit as st
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional
from router import RouteDecision, route
from server_manager import ManagedServers, ensure_q5_running, ensure_q6_running, stop_server_on_port
from settings import CONFIG

from openai_client import (
    HttpTimeouts,
    LlamaServerError,
    discover_model_id,
    health_check,
    stream_chat_completions,
)

from prompt_library import (
    TemplateKey,
    build_messages,
    sage_architect_core,
    sage_fast_core,
)

from mermaid_streamlit import (
    build_sage_kaizen_mermaid,
    probe_llama_server,
    render_mermaid_with_exports,
)


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
    try:
        if ":" in url:
            return int(url.rstrip("/").split(":")[-1])
    except Exception:
        return None
    return None


def _normalize_base_url_ui(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[:-3].rstrip("/")
    return u


def _manual_decision(deep_mode: bool) -> RouteDecision:
    # If user disables auto-escalate, we only respect deep_mode.
    if deep_mode:
        return RouteDecision(brain="ARCHITECT", reasons=["manual_deep_mode"], score=999)
    return RouteDecision(brain="FAST", reasons=["manual_fast_mode"], score=0)

def _auto_templates(user_text: str) -> tuple[TemplateKey, ...]:
    """
    Lightweight heuristic for automatic template selection.
    (Per-turn; user can override from sidebar.)
    """
    txt = (user_text or "").lower()
    templates: List[TemplateKey] = [
        TemplateKey.UNIVERSAL_DEPTH_ANCHOR,
        TemplateKey.AUTO_ADAPTIVE_META,
    ]

    if any(k in txt for k in ("teach", "tutor", "explain like", "for a 3rd grader", "for a 10th grader")):
        templates.append(TemplateKey.TEACHING_TUTORING)

    if any(k in txt for k in ("history", "civilization", "ancient", "timeline", "religion", "religious")):
        templates.append(TemplateKey.STRUCTURED_KNOWLEDGE)

    if any(k in txt for k in ("philosophy", "theology", "ethics", "meaning", "debate")):
        templates.append(TemplateKey.PHILOSOPHY_DEEP_THINKING)

    # Preserve order, remove duplicates
    return tuple(dict.fromkeys(templates))


st.set_page_config(page_title="Sage Kaizen (llama-server)", page_icon="🧠", layout="wide")
st.title("🧠 Sage Kaizen (Dual-Brain Chat) 🧠")

system_prompt = _load_system_prompt(CONFIG.system_prompt_path)
timeouts = HttpTimeouts(connect_s=CONFIG.connect_timeout_s, read_s=CONFIG.read_timeout_s)
timeouts_status = HttpTimeouts(connect_s=2.0, read_s=2.0)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_thinking_time" not in st.session_state:
    st.session_state.last_thinking_time = None
if "q5_model_id" not in st.session_state:
    st.session_state.q5_model_id = CONFIG.q5_model_id
if "q6_model_id" not in st.session_state:
    st.session_state.q6_model_id = CONFIG.q6_model_id
if "last_route" not in st.session_state:
    st.session_state.last_route = None  # type: ignore[assignment]

with st.sidebar:
    st.subheader("Servers")

    q5_url = _normalize_base_url_ui(st.text_input("Q5 base URL", value=CONFIG.q5_base_url))
    q6_url = _normalize_base_url_ui(st.text_input("Q6 base URL", value=CONFIG.q6_base_url))
    embed_url = _normalize_base_url_ui(st.text_input("Embed base URL", value=CONFIG.embed_base_url))

    q5_port = _base_url_to_port(q5_url) or 8011
    q6_port = _base_url_to_port(q6_url) or 8012
    embed_port = _base_url_to_port(embed_url) or 8020

    servers = ManagedServers(
        q5_port=q5_port,
        q6_port=q6_port,
        embed_port=embed_port,
        start_q5_bat=Path("start_q5_server.bat"),
        start_q6_bat=Path("start_q6_server.bat"),
        start_embed_bat=Path("start_embedding_point.bat"),
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("Start servers"):
            with st.status("Starting Embeddings → Q5 (Fast Brain)…", expanded=True) as s:
                s.write("Waiting for **Fast Brain** to finish loading (watch logs/q5_server.log)…")
                ok, msg = ensure_q5_running(servers)
                if ok:
                    s.write("✅ **Fast Brain** loaded (Q5 ready).")
                    s.update(label="Q5 ready ✅", state="complete")
                    st.success(msg)
                    st.info("Q6 (Architect Brain) will start automatically only when a turn escalates (or enable Deep mode).")
                else:
                    s.update(label="Failed to start Q5 ❌", state="error")
                    st.error(msg)

    with colB:
        if st.button("Stop servers"):
            ok1 = stop_server_on_port(q5_port)
            ok2 = stop_server_on_port(q6_port)
            ok3 = stop_server_on_port(embed_port)
            if ok1 and ok2 and ok3:
                st.success("Stopped (or already stopped).")
            else:
                st.warning("Tried to stop, but one or more servers may still be running.")

    ok5, d5 = health_check(q5_url, timeouts=timeouts_status)
    ok6, d6 = health_check(q6_url, timeouts=timeouts_status)
    st.caption(f"Q5: {'✅' if ok5 else '❌'} {d5}")
    st.caption(f"Q6: {'✅' if ok6 else '❌'} {d6}")

    if st.button("Discover model IDs"):
        mid5 = discover_model_id(q5_url, timeouts=timeouts_status)
        mid6 = discover_model_id(q6_url, timeouts=timeouts_status)
        if mid5:
            st.session_state.q5_model_id = mid5
        if mid6:
            st.session_state.q6_model_id = mid6
        st.success(f"Model IDs: Q5={st.session_state.q5_model_id} | Q6={st.session_state.q6_model_id}")

    st.subheader("Routing")
    deep_mode = st.toggle("Deep mode (force Architect this turn)", value=False)
    auto_escalate = st.toggle("Auto-escalate when needed", value=True)

    st.subheader("Prompt Templates")
    auto_templates = st.toggle("Auto templates", value=True)

    override_templates = st.multiselect(
        "Template overrides (optional)",
        options=list(TemplateKey),
        default=[],
        format_func=lambda t: t.value,
        help="If you select templates here, they are used for this turn and auto-templates are ignored.",
    )


    # Display last routing decision (helpful for tuning)
    if st.session_state.last_route is not None:
        rd: RouteDecision = st.session_state.last_route  # type: ignore[assignment]
        st.caption(f"🧭 Last route: {rd.brain} • score={rd.score}")
        if rd.reasons:
            st.caption("Reasons: " + ", ".join(rd.reasons[:6]))
    
    st.subheader("Generation")
    temperature_q5 = st.slider("Q5 temperature", 0.0, 1.2, 0.4, 0.05)
    temperature_q6 = st.slider("Q6 temperature", 0.0, 1.2, 0.6, 0.05)
    top_p_q5 = st.slider("Q5 top_p", 0.1, 1.0, 0.92, 0.01)
    top_p_q6 = st.slider("Q6 top_p", 0.1, 1.0, 0.95, 0.01)
    max_tokens_q5 = st.selectbox("Q5 max_tokens", [1024, 2048, 4096, 8192], index=2)
    max_tokens_q6 = st.selectbox("Q6 max_tokens", [2048, 4096, 8192, 16384], index=2)

    if st.session_state.last_thinking_time is not None:
        st.caption(f"⏱️ Last thinking time: {st.session_state.last_thinking_time:.2f}s")

    if st.button("New chat"):
        st.session_state.messages = []
        st.session_state.last_thinking_time = None
        st.session_state.last_route = None
        st.rerun()


# ----------------------------
# Mermaid Diagram (Architecture)
# ----------------------------

def _arch_sig(info) -> str:
    """Stable signature of probe fields; changes only when server-reported config changes."""
    if info is None:
        payload = {"ok": False}
    else:
        payload = {
            "ok": bool(getattr(info, "ok", False)),
            "model_id": getattr(info, "model_id", None),
            "alias": getattr(info, "alias", None),
            "ctx_size": getattr(info, "ctx_size", None),
            "n_gpu_layers": getattr(info, "n_gpu_layers", None),
            "device": getattr(info, "device", None),
            "tensor_split": getattr(info, "tensor_split", None),
            "split_mode": getattr(info, "split_mode", None),
            "main_gpu": getattr(info, "main_gpu", None),
            "source": getattr(info, "source", None),
            # keep raw out of signature to avoid noisy changes
        }
    b = json.dumps(payload, sort_keys=True, default=str).encode("utf-8", errors="replace")
    return hashlib.sha256(b).hexdigest()


if "arch_mermaid_src" not in st.session_state:
    st.session_state.arch_mermaid_src = ""
if "arch_sig_q5" not in st.session_state:
    st.session_state.arch_sig_q5 = ""
if "arch_sig_q6" not in st.session_state:
    st.session_state.arch_sig_q6 = ""

# Manual refresh (forces regeneration even if signature unchanged)
col_m1, col_m2 = st.columns([1, 6])
with col_m1:
    refresh_arch = st.button("Refresh diagram now", help="Re-probe Q5/Q6 and rebuild the Mermaid diagram.")
with col_m2:
    st.markdown("### Mermaid Diagram")

# Probe (cached inside mermaid_streamlit.py with a short TTL)
q5_info = probe_llama_server(q5_url)
q6_info = probe_llama_server(q6_url)

sig5 = _arch_sig(q5_info)
sig6 = _arch_sig(q6_info)

changed = refresh_arch or (sig5 != st.session_state.arch_sig_q5) or (sig6 != st.session_state.arch_sig_q6)
if changed:
    st.session_state.arch_sig_q5 = sig5
    st.session_state.arch_sig_q6 = sig6
    st.session_state.arch_mermaid_src = build_sage_kaizen_mermaid(q5_info, q6_info)

with st.expander("View architecture diagram (Mermaid)", expanded=True):
    st.caption(
        f"Q5 probe: {'OK' if q5_info.ok else 'DOWN'} ({q5_info.source}) • "
        f"Q6 probe: {'OK' if q6_info.ok else 'DOWN'} ({q6_info.source})"
    )
    render_mermaid_with_exports(st.session_state.arch_mermaid_src, height=700, theme="default")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Ask Sage Kaizen…")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # New router integration:
    # - If auto_escalate is enabled: route() decides FAST vs ARCHITECT, deep_mode can force ARCHITECT.
    # - If auto_escalate is disabled: only deep_mode decides.
    if auto_escalate:
        decision = route(user_text, force_architect=deep_mode)
    else:
        decision = _manual_decision(deep_mode)

    st.session_state.last_route = decision

    use_q6 = decision.brain == "ARCHITECT"

    with st.status("Checking servers…", expanded=False) as s:
        ok, msg = ensure_q5_running(servers)
        if not ok:
            s.update(label="Q5 not ready ❌", state="error")
            st.error(msg)
            st.stop()

        if use_q6:
            s.write("Starting **Architect Brain** (Q6) load…")
            ok, msg = ensure_q6_running(servers)
            if not ok:
                s.update(label="Q6 not ready ❌", state="error")
                st.error(msg)
                st.stop()
            s.write("✅ **Architect Brain** loaded (Q6 ready).")

        s.update(label="Servers ready ✅", state="complete")

    brain_name = "Architect (Q6)" if use_q6 else "Fast (Q5)"
    base_url = q6_url if use_q6 else q5_url
    model_id = st.session_state.q6_model_id if use_q6 else st.session_state.q5_model_id
    temperature = float(temperature_q6 if use_q6 else temperature_q5)
    top_p = float(top_p_q6 if use_q6 else top_p_q5)
    max_tokens = int(max_tokens_q6 if use_q6 else max_tokens_q5)

    # ----- Prompt library integration -----
    core_prompt = sage_architect_core if use_q6 else sage_fast_core

    if override_templates:
        templates = tuple(override_templates)
    else:
        templates = _auto_templates(user_text) if auto_templates else ()

    # Build base messages (system + core + templates + current user)
    messages = build_messages(
        user_text=user_text,
        system_prompt=system_prompt,
        core_prompt=core_prompt,
        templates=templates,
    )

    # Append prior conversation history (excluding the current user turn to avoid duplication)
    history = st.session_state.messages[-CONFIG.max_history_messages:]
    prior = history[:-1] if history else []
    messages.extend(prior)

    messages = router.apply_rag(messages, user_text, decision)

    with st.chat_message("assistant"):
        # Show routing meta (tiny, useful)
        st.caption(f"🧭 Route: {decision.brain} • score={decision.score} • reasons: {', '.join(decision.reasons[:6])}")
        st.caption(f"🧩 Templates: {', '.join(t.value for t in templates) if templates else '(none)'}")
        
        live = st.empty()
        acc: List[str] = []
        start = time.time()

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
            st.caption(f"⏱️ Thinking time: {elapsed:.2f}s • Brain: {brain_name} • Endpoint: {base_url}")
            st.session_state.messages.append({"role": "assistant", "content": final})
