# ui_streamlit.py
from __future__ import annotations

import os
import time
import traceback
import streamlit as st
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypedDict, Literal, cast
from llama_cpp import Llama

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage

# Optional: your existing GPU/RAM banner. Make it non-fatal.
try:
    from test_using_gpus import debug_gpu_memory_banner
except Exception:
    debug_gpu_memory_banner = None


# For Pylance-friendly typing (available in newer llama-cpp-python)
try:
    from llama_cpp import ChatCompletionRequestMessage
except Exception:  # pragma: no cover
    ChatCompletionRequestMessage = Dict[str, str]  # type: ignore[misc,assignment]

class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

# ----------------------------
# Config (edit paths as needed)
# ----------------------------
Q5_MODEL_PATH = r"E:/DeepSeek-V3.2-GGUF/Q5_K_M/DeepSeek-V3.2-Q5_K_M-00001-of-00010.gguf"
Q6_MODEL_PATH = r"E:/DeepSeek-V3.2-GGUF/Q6_K/DeepSeek-V3.2-Q6_K-00001-of-00012.gguf"

SYSTEM_PROMPT_PATH = r"./sage_kaizen_system_prompt.txt"


# ----------------------------
# Helpers
# ----------------------------
def load_system_prompt(path: str) -> str:
    if not os.path.exists(path):
        return (
            "You are Sage Kaizen. Be accurate, structured, and helpful. "
            "If you are unsure, say so and ask for the missing info."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def safe_debug(tag: str) -> None:
    if debug_gpu_memory_banner is None:
        return
    try:
        debug_gpu_memory_banner(tag)
    except Exception:
        # Never let telemetry affect UI or outputs.
        pass


# Heuristics for escalation (simple + effective)
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


@dataclass
class BrainConfig:
    n_ctx: int = 8192
    n_batch: int = 1024
    n_ubatch: int = 256

    n_gpu_layers: int = 96
    tensor_split: Tuple[float, float] = (0.60, 0.40)  # push some work onto 5080
    split_mode: int = 1
    main_gpu: int = 0

    flash_attn: bool = True
    offload_kqv: bool = True

    use_mmap: bool = True
    use_mlock: bool = False

    verbose: bool = False


def _llama_kwargs(cfg: BrainConfig) -> Dict[str, Any]:
    # Keep args minimal + stable across builds.
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
        verbose=cfg.verbose,
    )


# ----------------------------
# Streamlit app
# ----------------------------
st.set_page_config(page_title="Sage Kaizen", page_icon="🧠", layout="wide")
st.title("🧠 Sage Kaizen (Dual-Brain Chat) 🧠 ")

system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH)

# --- Minimal-patch state to allow instant UI render before model load ---
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False  # user intent: clicked Load models
if "models_ready" not in st.session_state:
    st.session_state.models_ready = False   # actual: models successfully loaded
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role": "user"|"assistant", "content": str}]
if "last_used_brain" not in st.session_state:
    st.session_state.last_used_brain = None

with st.sidebar:
    st.subheader("Routing")
    deep_mode = st.toggle(
        "Deep mode (force Q6 this turn)",
        value=False,
        help="Escalate the next assistant response to Q6.",
    )
    auto_escalate = st.toggle("Auto-escalate when needed", value=True)

    st.subheader("Model / Perf")
    st.caption("Changing these will reload models after you click **Load models** again.")

    n_ctx = st.selectbox("Context (n_ctx)", [4096, 8192, 12288, 16384], index=1)
    n_batch = st.selectbox("Batch (n_batch)", [256, 512, 1024, 1536, 2048], index=2)
    n_ubatch = st.selectbox("Micro-batch (n_ubatch)", [128, 256, 512], index=1)

    q5_gpu_layers = st.selectbox("Q5 n_gpu_layers", [32, 48, 64, 80, 96, 112, 128, 160], index=4)
    q6_gpu_layers = st.selectbox("Q6 n_gpu_layers", [16, 24, 32, 48, 64, 80, 96, 112], index=6)

    split = st.selectbox(
        "tensor_split (5090, 5080)",
        [(0.67, 0.33), (0.60, 0.40), (0.55, 0.45), (0.50, 0.50)],
        index=1,
    )

    if st.button("New chat"):
        st.session_state.messages = []
        st.session_state.last_used_brain = None
        st.rerun()

    # --- Minimal-patch controls: load behind a button ---
    if st.button("Load models"):
        st.session_state.models_loaded = True
        st.session_state.models_ready = False
        st.rerun()

    if st.button("Unload models"):
        st.cache_resource.clear()
        st.session_state.models_loaded = False
        st.session_state.models_ready = False
        st.rerun()


# Cache both brains (models stay loaded)
@st.cache_resource
def get_brains(
    n_ctx: int,
    n_batch: int,
    n_ubatch: int,
    q5_layers: int,
    q6_layers: int,
    tensor_split: Tuple[float, float],
) -> Tuple[Llama, Llama]:
    cfg_common = BrainConfig(n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch, tensor_split=tensor_split)

    cfg_q5 = BrainConfig(**{**cfg_common.__dict__, "n_gpu_layers": q5_layers})
    cfg_q6 = BrainConfig(**{**cfg_common.__dict__, "n_gpu_layers": q6_layers})

    safe_debug("before Q5 load")
    q5 = Llama(model_path=Q5_MODEL_PATH, **_llama_kwargs(cfg_q5))
    safe_debug("after Q5 load")

    safe_debug("before Q6 load")
    q6 = Llama(model_path=Q6_MODEL_PATH, **_llama_kwargs(cfg_q6))
    safe_debug("after Q6 load")

    return q5, q6


# --- Minimal-patch: do not load models until the button is clicked ---
q5_llm: Optional[Llama] = None
q6_llm: Optional[Llama] = None

if st.session_state.models_loaded:
    # Show progress/status while models load (UI already rendered)
    with st.status("Loading models (Q5 then Q6)…", expanded=True) as status:
        try:
            status.write("Starting model load…")
            q5_llm, q6_llm = get_brains(
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_ubatch=n_ubatch,
                q5_layers=q5_gpu_layers,
                q6_layers=q6_gpu_layers,
                tensor_split=split,
            )
            st.session_state.models_ready = True
            status.update(label="Models loaded ✅", state="complete")
        except Exception as e:
            st.session_state.models_ready = False
            status.update(label="Model load failed ❌", state="error")
            st.error(f"Model load failed: {e}")
            st.code(traceback.format_exc())
else:
    st.info("Models are not loaded yet. Click **Load models** in the sidebar to start.")


# Render history (always visible, even before models load)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# User input (disabled until models are ready)
user_text = st.chat_input(
    "Ask Sage Kaizen…",
    disabled=(not st.session_state.models_ready) or (q5_llm is None) or (q6_llm is None),
)

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Decide brain for THIS turn
    use_q6 = should_escalate_to_q6(user_text, deep_mode) if auto_escalate else deep_mode
    brain_name = "Q6" if use_q6 else "Q5"
    llm = q6_llm if use_q6 else q5_llm
    st.session_state.last_used_brain = brain_name

    assert llm is not None  # for type checkers

    # Build message list for true chat completion (last N turns to control ctx growth)
    messages: list[ChatMessage] = [
        {"role": "system", "content": system_prompt}
    ]
    history = st.session_state.messages[-20:]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages_for_llama = cast("list[ChatCompletionRequestMessage]", cast(Any, messages))

    # Accuracy-focused sampling defaults (explicit kwargs = Pylance friendly)
    temperature = 0.4 if not use_q6 else 0.6
    top_p = 0.92 if not use_q6 else 0.95
    top_k = 40 if not use_q6 else 50
    min_p = 0.02 if not use_q6 else 0.03
    max_tokens = 4096 if not use_q6 else 8192

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking… ({brain_name})"):
            start = time.time()
            safe_debug("before chat_completion")

            resp = llm.create_chat_completion(
                messages=messages_for_llama,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_tokens,
                stream=False,  # IMPORTANT: avoid iterator return type
            )

            safe_debug("after chat_completion")
            elapsed = time.time() - start

            resp_dict = cast(Dict[str, Any], resp)
            content = resp_dict["choices"][0]["message"]["content"].strip()

            st.markdown(content)
            st.caption(f"Brain: {brain_name} • {elapsed:.2f}s")

    st.session_state.messages.append({"role": "assistant", "content": content})
