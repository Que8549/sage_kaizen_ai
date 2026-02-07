# ui_streamlit.py
from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, TypedDict, Literal, cast, Iterator

import streamlit as st
from llama_cpp import Llama

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage

# Optional: GPU/RAM telemetry (must never crash the app)
try:
    from test_using_gpus import debug_gpu_memory_banner
except Exception:
    debug_gpu_memory_banner = None


# ----------------------------
# Config (edit paths as needed)
# ----------------------------
Q5_MODEL_PATH = r"E:/DeepSeek-V3.2-GGUF/Q5_K_M/DeepSeek-V3.2-Q5_K_M-00001-of-00010.gguf"
Q6_MODEL_PATH = r"E:/DeepSeek-V3.2-GGUF/Q6_K/DeepSeek-V3.2-Q6_K-00001-of-00012.gguf"

SYSTEM_PROMPT_PATH = r"./sage_kaizen_system_prompt.txt"


# ----------------------------
# Types
# ----------------------------
class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


# ----------------------------
# Helpers
# ----------------------------
def load_system_prompt(path: str) -> str:
    if not os.path.exists(path):
        return (
            "You are Sage Kaizen. Be accurate, structured, and helpful. "
            "Prefer correctness over speed. If unsure, say so and ask for missing info."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def safe_debug(tag: str) -> None:
    if debug_gpu_memory_banner is None:
        return
    try:
        debug_gpu_memory_banner(tag)
    except Exception:
        pass


DEPTH_HINTS = (
    "explain",
    "analyze",
    "compare",
    "why",
    "how",
    "history",
    "philosophy",
    "theology",
    "deep",
    "in depth",
    "detailed",
    "step-by-step",
    "teach",
    "tutor",
    "architecture",
    "design",
    "tradeoff",
    "pros and cons",
    "evaluate",
)
CODE_HINTS = ("code", "python", "c#", "typescript", "debug", "stack trace", "error")


def should_escalate_to_q6(user_text: str, deep_mode_toggle: bool) -> bool:
    if deep_mode_toggle:
        return True
    txt = user_text.lower()
    if any(k in txt for k in DEPTH_HINTS):
        return True
    if any(k in txt for k in CODE_HINTS):
        return True
    if " and " in txt or " also " in txt or " vs " in txt or "compare" in txt:
        return True
    return False


def stream_chat_completion_to_placeholder(
    llm: Llama,
    *,
    messages_for_llama: Any,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    max_tokens: int,
    placeholder: Any,
) -> Tuple[str, float]:
    """
    Streams llama-cpp-python chat completion chunks into a Streamlit placeholder.

    Returns: (final_text, elapsed_seconds)

    Pylance-safe:
      - Treat streaming response as Iterator[Any] (chunk dicts)
      - Do not index the stream itself
    """
    start = time.time()
    acc: list[str] = []

    stream = llm.create_chat_completion(
        messages=messages_for_llama,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in cast(Iterator[Any], stream):
        # Typical chunk:
        # {"choices": [{"delta": {"content": "..."}, ...}], ...}
        try:
            choice0 = chunk["choices"][0]
            delta = choice0.get("delta", {})
            piece = delta.get("content")
            if piece:
                acc.append(piece)
                placeholder.markdown("".join(acc))
        except Exception:
            # Be resilient to minor shape differences across builds
            pass

    elapsed = time.time() - start
    return "".join(acc).strip(), elapsed


# ----------------------------
# Brain config
# ----------------------------
@dataclass
class BrainConfig:
    # Context + batching
    n_ctx: int = 8192
    # Conservative default to avoid RAM runaway; increase once stable.
    n_batch: int = 512
    n_ubatch: int = 256

    # GPU offload
    n_gpu_layers: int = 128
    tensor_split: Tuple[float, float] = (0.55, 0.45)  # push more onto 5080 than VRAM-proportional
    split_mode: int = 1
    main_gpu: int = 0

    flash_attn: bool = True
    offload_kqv: bool = True

    use_mmap: bool = True
    use_mlock: bool = False

    # CPU-side work (sampling/logits) can bottleneck; threads can help throughput
    n_threads: int = max(1, (os.cpu_count() or 1) - 2)
    n_threads_batch: int = max(1, (os.cpu_count() or 1) - 2)

    verbose: bool = False


def _llama_kwargs(cfg: BrainConfig) -> Dict[str, Any]:
    return dict(
        n_ctx=cfg.n_ctx,
        n_batch=cfg.n_batch,
        n_ubatch=cfg.n_ubatch,
        n_gpu_layers=cfg.n_gpu_layers,
        tensor_split=[float(cfg.tensor_split[0]), float(cfg.tensor_split[1])],
        split_mode=cfg.split_mode,
        main_gpu=cfg.main_gpu,
        flash_attn=cfg.flash_attn,
        offload_kqv=cfg.offload_kqv,
        use_mmap=cfg.use_mmap,
        use_mlock=cfg.use_mlock,
        n_threads=cfg.n_threads,
        n_threads_batch=cfg.n_threads_batch,
        verbose=cfg.verbose,
    )


# ----------------------------
# Streamlit app
# ----------------------------
st.set_page_config(page_title="Sage Kaizen", page_icon="🧠", layout="wide")
st.title("🧠 Sage Kaizen (Dual-Brain Chat) 🧠")

system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH)

# --- State: UI must render instantly; loading happens only after button click ---
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False  # user intent: clicked Load models
if "models_ready" not in st.session_state:
    st.session_state.models_ready = False   # actual: load succeeded
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_used_brain" not in st.session_state:
    st.session_state.last_used_brain = None

with st.sidebar:
    st.subheader("Routing")
    deep_mode = st.toggle("Deep mode (force Q6 this turn)", value=False)
    auto_escalate = st.toggle("Auto-escalate when needed", value=True)

    if "last_thinking_time" in st.session_state:
        st.caption(f"⏱️ Last thinking time: {st.session_state.last_thinking_time:.2f}s")

    st.subheader("Model / Perf")
    st.caption("Tip: Higher GPU layers uses more VRAM and can reduce CPU/RAM pressure.")

    n_ctx = st.selectbox("Context (n_ctx)", [4096, 8192, 12288, 16384], index=1)

    # Conservative default (RAM stability); increase if stable
    n_batch = st.selectbox("Batch (n_batch)", [256, 512, 1024, 1536, 2048], index=1)
    n_ubatch = st.selectbox("Micro-batch (n_ubatch)", [128, 256, 512], index=1)

    # Offload defaults: higher to push compute to GPU
    q5_gpu_layers = st.selectbox("Q5 n_gpu_layers", [64, 80, 96, 112, 128, 160, 192, 224], index=5)  # 160 default
    q6_gpu_layers = st.selectbox("Q6 n_gpu_layers", [32, 48, 64, 80, 96, 112, 128, 160], index=5)   # 112 default

    # Push more weight to GPU1 so 5080 does more than ~3%
    split = st.selectbox(
        "tensor_split (5090, 5080)",
        [(0.67, 0.33), (0.60, 0.40), (0.55, 0.45), (0.50, 0.50)],
        index=2,
    )

    preload_q6 = st.toggle(
        "Preload Q6 (uses more RAM)",
        value=True,
        help="If RAM is pegged at 100%, disable this to load Q6 only when needed.",
    )

    if st.button("New chat"):
        st.session_state.messages = []
        st.session_state.last_used_brain = None
        st.rerun()

    if st.button("Load models"):
        st.session_state.models_loaded = True
        st.session_state.models_ready = False
        st.rerun()

    if st.button("Unload models"):
        st.cache_resource.clear()
        st.session_state.models_loaded = False
        st.session_state.models_ready = False
        st.rerun()


@st.cache_resource
def get_brains(
    n_ctx: int,
    n_batch: int,
    n_ubatch: int,
    q5_layers: int,
    q6_layers: int,
    tensor_split: Tuple[float, float],
    preload_q6: bool,
) -> Tuple[Llama, Optional[Llama]]:
    cfg_common = BrainConfig(n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch, tensor_split=tensor_split)

    cfg_q5 = BrainConfig(**{**cfg_common.__dict__, "n_gpu_layers": q5_layers})
    cfg_q6 = BrainConfig(**{**cfg_common.__dict__, "n_gpu_layers": q6_layers})

    safe_debug("before Q5 load")
    q5 = Llama(model_path=Q5_MODEL_PATH, **_llama_kwargs(cfg_q5))
    safe_debug("after Q5 load")

    q6: Optional[Llama] = None
    if preload_q6:
        safe_debug("before Q6 load")
        q6 = Llama(model_path=Q6_MODEL_PATH, **_llama_kwargs(cfg_q6))
        safe_debug("after Q6 load")

    return q5, q6


# ---- Load models only after click; show progress via st.status ----
q5_llm: Optional[Llama] = None
q6_llm: Optional[Llama] = None

if st.session_state.models_loaded:
    with st.status("Loading models…", expanded=True) as status:
        try:
            status.write("Loading Q5…")
            q5_llm, q6_llm = get_brains(
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_ubatch=n_ubatch,
                q5_layers=q5_gpu_layers,
                q6_layers=q6_gpu_layers,
                tensor_split=split,
                preload_q6=preload_q6,
            )
            if preload_q6:
                status.write("Q6 loaded ✅")
            else:
                status.write("Q6 preload disabled (will load on demand) ✅")

            st.session_state.models_ready = True
            status.update(label="Models ready ✅", state="complete")
        except Exception as e:
            st.session_state.models_ready = False
            status.update(label="Model load failed ❌", state="error")
            st.error(f"Model load failed: {e}")
            st.code(traceback.format_exc())
else:
    st.info("Models are not loaded yet. Click **Load models** in the sidebar to start.")


# Render chat history immediately (even before load)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# Disable input until models are ready
user_text = st.chat_input(
    "Ask Sage Kaizen…",
    disabled=(not st.session_state.models_ready) or (q5_llm is None),
)

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    use_q6 = should_escalate_to_q6(user_text, deep_mode) if auto_escalate else deep_mode
    brain_name = "Q6" if use_q6 else "Q5"

    # Ensure Q6 exists if required (load on demand if preload disabled)
    if use_q6 and q6_llm is None:
        with st.status("Loading Q6 on demand…", expanded=True) as status:
            try:
                status.write("Loading Q6…")
                cfg_common = BrainConfig(n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch, tensor_split=split)
                cfg_q6 = BrainConfig(**{**cfg_common.__dict__, "n_gpu_layers": q6_gpu_layers})
                q6_llm = Llama(model_path=Q6_MODEL_PATH, **_llama_kwargs(cfg_q6))
                status.update(label="Q6 loaded ✅", state="complete")
            except Exception as e:
                status.update(label="Q6 load failed ❌", state="error")
                st.error(f"Q6 load failed: {e}")
                st.code(traceback.format_exc())
                use_q6 = False
                brain_name = "Q5"

    llm = (q6_llm if use_q6 else q5_llm)
    assert llm is not None
    st.session_state.last_used_brain = brain_name

    # Build message list (limit history)
    messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]
    history = st.session_state.messages[-20:]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Cast to satisfy llama-cpp-python typing (Pylance-safe)
    messages_for_llama = cast("list[ChatCompletionRequestMessage]", cast(Any, messages))

    # Accuracy-first sampling defaults (explicit kwargs, no **params)
    temperature = 0.4 if not use_q6 else 0.6
    top_p = 0.92 if not use_q6 else 0.95
    top_k = 40 if not use_q6 else 50
    min_p = 0.02 if not use_q6 else 0.03
    max_tokens = 4096 if not use_q6 else 8192

    with st.chat_message("assistant"):
        live = st.empty()

        with st.spinner(f"Thinking… ({brain_name})"):
            safe_debug("before chat_completion (stream)")

            content, elapsed = stream_chat_completion_to_placeholder(
                llm,
                messages_for_llama=messages_for_llama,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_tokens,
                placeholder=live,
            )

            safe_debug("after chat_completion (stream)")
            st.session_state.last_thinking_time = elapsed

        # Ensure final content is rendered
        live.markdown(content)
        st.caption(f"⏱️ Thinking time: {elapsed:.2f}s • Brain: {brain_name}")

    st.session_state.messages.append({"role": "assistant", "content": content})
