# sage_kaizen_prompt_lib.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple
from time import sleep
from test_using_gpus import debug_gpu_memory_banner

# NOTE: The biggest compatibility variable for Python 3.13 on Windows is whether
# llama_cpp has a wheel for your Python version/arch. The code below is 3.13-safe,
# but llama_cpp installation must also be 3.13-compatible.
try:
    os.environ["LLAMA_CPP_LIB"] = r"F:\Projects\sage_kaizen_ai\llama.cpp\build\bin\Release\llama.dll"
    from llama_cpp import Llama # requires your CUDA 13.1-enabled build
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import llama_cpp.Llama.\n"
        "This is usually an installation / wheel-availability issue on Python 3.13.\n"
        "Verify llama-cpp-python supports Python 3.13 on Windows for your environment.\n"
        f"Original error: {e}"
    ) from e


# -----------------------------
# 1) Model paths (edit as needed)
# -----------------------------
Q5_MODEL_PATH = r"E:/DeepSeek-V3.2-GGUF/Q5_K_M/DeepSeek-V3.2-Q5_K_M-00001-of-00010.gguf"
Q6_MODEL_PATH = r"E:/DeepSeek-V3.2-GGUF/Q6_K/DeepSeek-V3.2-Q6_K-00001-of-00012.gguf"

SYSTEM_PROMPT_PATH = r"./sage_kaizen_system_prompt.txt"


# -----------------------------
# 2) Quant + profile definitions
# -----------------------------
class Quant(str, Enum):
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"


@dataclass(frozen=True)
class SamplingProfile:
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    max_tokens: int = 12288
    # stop sequences can be None or a sequence of strings
    stop: Optional[Sequence[str]] = None


Q5_PRECISION = SamplingProfile(
    temperature=0.4,
    top_k=40,
    top_p=0.92,
    min_p=0.02,
)

Q6_DEPTH = SamplingProfile(
    temperature=0.6,
    top_k=50,
    top_p=0.95,
    min_p=0.03,
)


# -----------------------------
# 3) The 7 drop-in templates
# -----------------------------
class TemplateKey(str, Enum):
    UNIVERSAL_DEPTH_ANCHOR = "universal_depth_anchor"           # (1)
    STRUCTURED_KNOWLEDGE = "structured_knowledge"               # (2)
    TEACHING_TUTORING = "teaching_tutoring"                     # (3)
    PHILOSOPHY_DEEP_THINKING = "philosophy_deep_thinking"       # (4)
    ANTI_EARLY_STOP = "anti_early_stop"                         # (5)
    VOICE_LOOP_FRIENDLY = "voice_loop_friendly"                 # (6)
    AUTO_ADAPTIVE_META = "auto_adaptive_meta"                   # (7)


TEMPLATES: Dict[TemplateKey, str] = {
    # (1) Universal Depth Anchor (use everywhere)
    TemplateKey.UNIVERSAL_DEPTH_ANCHOR: (
        "You are Sage Kaizen.\n"
        "Respond with depth, clarity, and structure.\n"
        "Assume the user values historical accuracy, nuance, and synthesis.\n"
        "Avoid shallow summaries.\n"
        "Explain reasoning explicitly.\n"
    ),

    # (2) Structured Knowledge Template (history/science/religion)
    TemplateKey.STRUCTURED_KNOWLEDGE: (
        "Answer using the following structure:\n\n"
        "1. Historical context and timeframe\n"
        "2. Primary civilizations or schools involved\n"
        "3. Methods, tools, or systems used\n"
        "4. Religious, philosophical, or symbolic meaning\n"
        "5. Long-term influence on later traditions\n\n"
        "Use clear sections and concrete examples.\n"
    ),

    # (3) Teaching / Tutoring Mode (grades 6–12)
    TemplateKey.TEACHING_TUTORING: (
        "Explain this topic as if teaching an intelligent student.\n"
        "Start with the big picture, then zoom into details.\n"
        "Define key terms.\n"
        "Use examples.\n"
        "End with a concise summary.\n"
    ),

    # (4) Philosophy / Deep Thinking Mode
    TemplateKey.PHILOSOPHY_DEEP_THINKING: (
        "Analyze this topic from multiple perspectives.\n"
        "Include historical, philosophical, and practical viewpoints.\n"
        "Highlight tensions, debates, or uncertainties.\n"
        "Do not rush to conclusions.\n"
    ),

    # (5) Anti-Early-Stop Continuation Trigger
    TemplateKey.ANTI_EARLY_STOP: (
        "Do not stop early; continue until the topic is thoroughly explored.\n"
    ),

    # (6) Voice Loop Friendly Template
    TemplateKey.VOICE_LOOP_FRIENDLY: (
        "Give a thoughtful, well-structured explanation.\n"
        "Speak clearly.\n"
        "Expand ideas fully but naturally.\n"
    ),

    # (7) Auto-Adaptive Meta Prompt (advanced)
    TemplateKey.AUTO_ADAPTIVE_META: (
        "If this topic benefits from detail, expand fully.\n"
        "If it is simple, be concise.\n"
        "Err on the side of depth.\n"
    ),
}


# -----------------------------
# 4) System prompt loader
# -----------------------------
def load_system_prompt(path: str = SYSTEM_PROMPT_PATH) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"System prompt file not found: {path}\n"
            "Create it (see sage_kaizen_system_prompt.txt in the instructions)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# -----------------------------
# 5) Prompt formatting (create_completion-friendly)
# -----------------------------
def build_prompt(
    user_text: str,
    *,
    system_prompt: str,
    templates: Tuple[TemplateKey, ...],
    include_system_prompt: bool = True,
) -> str:
    parts = []

    if include_system_prompt and system_prompt:
        parts.append("### System\n" + system_prompt.strip())

    if templates:
        templ_text = "\n".join(TEMPLATES[t].rstrip() for t in templates)
        parts.append("### Instructions\n" + templ_text.strip())

    parts.append("### User\n" + user_text.strip())
    parts.append("### Assistant\n")
    return "\n\n".join(parts)


# -----------------------------
# 6) Quant auto-router heuristics
# -----------------------------
DEPTH_HINTS = (
    "explain", "analyze", "compare", "why", "how", "history", "religion", "philosophy",
    "deep", "in depth", "detailed", "step-by-step", "teach", "tutor", "lesson",
    "architecture", "design", "tradeoff", "pros and cons", "evaluate",
)
SHORT_HINTS = (
    "quick", "brief", "tl;dr", "one sentence", "short answer", "summarize",
)
CODE_HINTS = (
    "code", "python", "c#", "typescript", "bash", "powershell", "implement",
    "write a script", "debug", "error", "stack trace",
)


def choose_quant(user_text: str, templates: Tuple[TemplateKey, ...]) -> Quant:
    txt = user_text.lower()

    # Explicit brevity request
    if any(k in txt for k in SHORT_HINTS):
        return Quant.Q5_K_M

    # Template-driven depth
    depth_templates = {
        TemplateKey.STRUCTURED_KNOWLEDGE,
        TemplateKey.TEACHING_TUTORING,
        TemplateKey.PHILOSOPHY_DEEP_THINKING,
        TemplateKey.ANTI_EARLY_STOP,
        TemplateKey.AUTO_ADAPTIVE_META,
    }
    if any(t in depth_templates for t in templates):
        return Quant.Q6_K

    # Multi-part signals
    multi_part = (" and " in txt) or (" also " in txt) or (" vs " in txt) or ("compare" in txt)
    has_depth = any(k in txt for k in DEPTH_HINTS)
    has_code = any(k in txt for k in CODE_HINTS)

    if multi_part or has_depth or has_code:
        return Quant.Q6_K

    return Quant.Q5_K_M


def sampling_for_quant(q: Quant) -> SamplingProfile:
    return Q6_DEPTH if q == Quant.Q6_K else Q5_PRECISION


# -----------------------------
# 7) Model manager (lazy-load each quant)
# -----------------------------
class SageKaizenLLM:
    def __init__(
        self,
        *,
        # Context & batching
        n_ctx: int = 8192,
        n_batch: int = 2048,
        n_ubatch: int = 512,

        # GPU offload
        n_gpu_layers: int = -1, # prefer "offload as many as possible" 61 fine for testing,
        tensor_split: Tuple[float, float] = (2.0, 1.0),  # 5090:5080 = 32GB:16GB
        split_mode: int = 1,     # layer-wise
        main_gpu: int = 0,       # 5090 as primary

        # Performance toggles
        flash_attn: bool = True,
        offload_kqv: bool = True,

        # CPU parallelism (9950X3D has lots of cores; tune to your preference)
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,

        verbose: bool = True,
    ):
        self._llm_q5: Optional[Llama] = None
        self._llm_q6: Optional[Llama] = None

        init_kwargs: Dict[str, Any] = dict(
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_ubatch=n_ubatch,

            n_gpu_layers=n_gpu_layers,
            tensor_split=[float(tensor_split[0]), float(tensor_split[1])],
            split_mode=split_mode,
            main_gpu=main_gpu,

            flash_attn=flash_attn,      # supported by llama-cpp-python :contentReference[oaicite:10]{index=10}
            offload_kqv=offload_kqv,    # supported by llama-cpp-python :contentReference[oaicite:11]{index=11}

            verbose=verbose,
        )

        # Only set thread params if caller provided them
        if n_threads is not None:
            init_kwargs["n_threads"] = int(n_threads)
        if n_threads_batch is not None:
            init_kwargs["n_threads_batch"] = int(n_threads_batch)

        self._init_kwargs = init_kwargs

    def _load(self, quant: Quant) -> Llama:
        if quant == Quant.Q5_K_M:
            if self._llm_q5 is None:
                self._llm_q5 = Llama(model_path=Q5_MODEL_PATH, **self._init_kwargs)
            return self._llm_q5

        if self._llm_q6 is None:
            self._llm_q6 = Llama(model_path=Q6_MODEL_PATH, **self._init_kwargs)
        return self._llm_q6


    def generate(
        self,
        user_text: str,
        *,
        templates: Tuple[TemplateKey, ...] = (TemplateKey.UNIVERSAL_DEPTH_ANCHOR,),
        system_prompt_path: str = SYSTEM_PROMPT_PATH,
        force_quant: Optional[Quant] = None,
        log_path: str = "output.txt",
    ) -> Dict[str, Any]:
        """
        Returns:
          chosen_quant (str)
          elapsed_time (float)
          prompt (str)
          text (str)
          raw (dict)
        """
        system_prompt = load_system_prompt(system_prompt_path)

        chosen = force_quant or choose_quant(user_text, templates)
        profile = sampling_for_quant(chosen)

        prompt = build_prompt(
            user_text,
            system_prompt=system_prompt,
            templates=templates,
            include_system_prompt=True,
        )
        

        llm = self._load(chosen)

        debug_gpu_memory_banner("after model load")

        start = time.time()

        # Mirror your working create_completion usage :contentReference[oaicite:1]{index=1}
        raw = llm.create_completion(
            prompt=prompt,
            temperature=profile.temperature,
            min_p=profile.min_p,
            top_k=profile.top_k,
            top_p=profile.top_p,
            max_tokens=profile.max_tokens,
            stop=profile.stop,  # None or sequence[str]
        )

        debug_gpu_memory_banner("after create_completion (inference)")

        elapsed = time.time() - start
        text = raw["choices"][0]["text"]

        # Logging should never break inference
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n\n==============================\n")
                f.write(f"Chosen Quant: {chosen.value}\n")
                f.write(f"Elapsed: {elapsed}\n")
                f.write(f"User: {user_text}\n")
                f.write(f"Templates: {[t.value for t in templates]}\n")
                f.write(f"Prompt:\n{prompt}\n")
                f.write(f"Text:\n{text}\n")
        except Exception:
            pass

        return {
            "chosen_quant": chosen.value,
            "elapsed_time": elapsed,
            "prompt": prompt,
            "text": text,
            "raw": raw,
        }
