# sage_kaizen_prompt_lib.py
from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
from dotenv import load_dotenv

from test_using_gpus import debug_gpu_memory_banner

# NOTE: The biggest compatibility variable for Python 3.13/3.14 on Windows is whether
# llama_cpp has a wheel for your Python version/arch. The code below is 3.13/3.14-safe,
# but llama_cpp installation must also be compatible.
try:
    os.environ["LLAMA_CPP_LIB"] = r"F:\Projects\sage_kaizen_ai\llama.cpp\build\bin\Release\llama.dll"
    from llama_cpp import Llama  # requires your CUDA 13.1-enabled build
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import llama_cpp.Llama.\n"
        "This is usually an installation / wheel-availability issue on Windows for your Python version.\n"
        "Verify llama-cpp-python supports your Python version on Windows for your environment.\n"
        f"Original error: {e}"
    ) from e


# -----------------------------
# 1) Model paths (edit as needed)
# -----------------------------
Q5_MODEL_PATH = r"E:/DeepSeek-V3.2-GGUF/UD-IQ1_S/DeepSeek-V3.2-UD-IQ1_S-00001-of-00004.gguf"
Q6_MODEL_PATH = r"E:/DeepSeek-V3.2-GGUF/UD-IQ1_M/DeepSeek-V3.2-UD-IQ1_M-00001-of-00005.gguf"

SYSTEM_PROMPT_PATH = r"./sage_kaizen_system_prompt.txt"


# -----------------------------
# 2) Quant + profile definitions
# -----------------------------
class Quant(str, Enum):
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"

load_dotenv()

@dataclass(frozen=True)
class SamplingProfile:
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    # Keep your original default to preserve behavior.
    # (We change *batching/offload* for stability; generation policy stays consistent.)
    max_tokens: int = 12288
    stop: Optional[Sequence[str]] = None # stop sequences can be None or a sequence of strings


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
    # (2) Structured Knowledge Template (history/science/religion, etc.)
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
# Wrap GPU debug calls so they can’t stop generation
# -----------------------------
def _safe_debug(tag: str) -> None:
    try:
        debug_gpu_memory_banner(tag)
    except Exception:
        pass

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

    if any(k in txt for k in SHORT_HINTS):
        return Quant.Q5_K_M

    depth_templates = {
        TemplateKey.STRUCTURED_KNOWLEDGE,
        TemplateKey.TEACHING_TUTORING,
        TemplateKey.PHILOSOPHY_DEEP_THINKING,
        TemplateKey.ANTI_EARLY_STOP,
        TemplateKey.AUTO_ADAPTIVE_META,
    }
    if any(t in depth_templates for t in templates):
        return Quant.Q6_K

    multi_part = (" and " in txt) or (" also " in txt) or (" vs " in txt) or ("compare" in txt)
    has_depth = any(k in txt for k in DEPTH_HINTS)
    has_code = any(k in txt for k in CODE_HINTS)

    if multi_part or has_depth or has_code:
        return Quant.Q6_K

    return Quant.Q5_K_M


def sampling_for_quant(q: Quant) -> SamplingProfile:
    return Q6_DEPTH if q == Quant.Q6_K else Q5_PRECISION


# -----------------------------
# 7) Helpers: safe kwargs + persistent config
# -----------------------------
def _filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    llama-cpp-python and llama.cpp flags differ across builds.
    Filter kwargs to those accepted by the target callable, preventing crashes.
    """
    try:
        sig = inspect.signature(fn)
    except Exception:
        return kwargs  # best-effort fallback

    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def _default_cache_path() -> Path:
    # Keep it local to the repo by default, but allow override.
    env = os.getenv("SAGE_KAIZEN_CONFIG_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return Path("./sage_kaizen_llm_config.json").resolve()


def _load_cached_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cached_config(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


# -----------------------------
# 8) Model manager (lazy-load each quant)
# -----------------------------
class SageKaizenLLM:
    """
    High-impact changes:
      - Safer defaults for batching to avoid RAM runaway
      - Force a real n_gpu_layers baseline (avoid -1 "auto" CPU fallback)
      - Cache per-quant GPU layer choices to avoid repeated load attempts
      - Pass cache_prompt/use_mmap/etc only when supported by your build
    """

    def __init__(
        self,
        *,
        # Context & batching (RAM stability)
        n_ctx: int = 8192,

        # n_batch=1024, n_ubatch=256
        # n_batch=1024, n_ubatch=512
        # n_batch=1536, n_ubatch=512
        # If stable, try n_batch=1536 Only then consider n_batch=2048 again (that was your RAM runaway risk)

        n_batch: int = 1024,     # was 2048; 512 is far more stable on large models
        n_ubatch: int = 256,    # was 512

        # GPU offload (VRAM utilization)
        # Avoid -1 here; it can silently "helpfully" fall back to CPU on some builds.
        n_gpu_layers: int = 80,
        tensor_split: Tuple[float, float] = (0.67, 0.33),  # 5090:5080 ≈ 32GB:16GB
        split_mode: int = 1,
        main_gpu: int = 0,

        # Performance toggles
        flash_attn: bool = True,
        offload_kqv: bool = True,

        # llama.cpp load behavior (reduce load time / memory pressure)
        use_mmap: bool = True,
        use_mlock: bool = False,

        # CPU parallelism
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,

        verbose: bool = True,

        # config caching
        config_path: Optional[str] = None,
    ):
        self._llm_q5: Optional[Llama] = None
        self._llm_q6: Optional[Llama] = None

        self._config_path = Path(config_path).expanduser().resolve() if config_path else _default_cache_path()
        self._config = _load_cached_config(self._config_path)

        # Base init kwargs (shared)
        base_kwargs: Dict[str, Any] = dict(
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_ubatch=n_ubatch,

            tensor_split=[float(tensor_split[0]), float(tensor_split[1])],
            split_mode=split_mode,
            main_gpu=main_gpu,

            flash_attn=flash_attn,
            offload_kqv=offload_kqv,

            use_mmap=use_mmap,
            use_mlock=use_mlock,

            verbose=verbose,
        )

        if n_threads is not None:
            base_kwargs["n_threads"] = int(n_threads)
        if n_threads_batch is not None:
            base_kwargs["n_threads_batch"] = int(n_threads_batch)

        # Store defaults; per-quant can override via cache/env.
        self._base_kwargs = base_kwargs
        self._default_n_gpu_layers = int(n_gpu_layers)

    def _get_cached_gpu_layers(self, quant: Quant) -> Optional[int]:
        # Env vars override everything (fast iteration, no code changes)
        env_key = "SAGE_KAIZEN_GPU_LAYERS_Q6" if quant == Quant.Q6_K else "SAGE_KAIZEN_GPU_LAYERS_Q5"
        if os.getenv(env_key):
            try:
                return int(os.environ[env_key])
            except Exception:
                pass

        key = f"n_gpu_layers.{quant.value}"
        val = self._config.get(key)
        if isinstance(val, int):
            return val
        return None

    def _set_cached_gpu_layers(self, quant: Quant, layers: int) -> None:
        key = f"n_gpu_layers.{quant.value}"
        self._config[key] = int(layers)
        _save_cached_config(self._config_path, self._config)

    def _init_kwargs_for_quant(self, quant: Quant, n_gpu_layers: int) -> Dict[str, Any]:
        kwargs = dict(self._base_kwargs)
        kwargs["n_gpu_layers"] = int(n_gpu_layers)
        return kwargs

    def _load_with_fallback_gpu_layers(self, model_path: str, quant: Quant) -> Llama:
        """
        Attempt a single load using cached/desired GPU layers.
        If that fails (OOM/alloc), fall back down a ladder once, then cache success.
        """
        cached_layers = self._get_cached_gpu_layers(quant)
        target = cached_layers if cached_layers is not None else self._default_n_gpu_layers

        # One-pass fallback ladder: start at target, then step down.
        # (Keeps load attempts bounded to avoid long repeated loads.)
        candidates = [target]
        for c in (96, 80, 64, 48, 32, 24, 16, 8, 0):
            if c not in candidates:
                candidates.append(c)

        last_err: Optional[BaseException] = None
        for layers in candidates:
            try:
                init_kwargs = self._init_kwargs_for_quant(quant, layers)
                # Filter to ctor signature (varies by build)
                safe_kwargs = _filter_kwargs_for_callable(Llama, init_kwargs)
                llm = Llama(model_path=model_path, **safe_kwargs)
                self._set_cached_gpu_layers(quant, layers)
                return llm
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(
            f"Failed to load model for {quant.value} with all GPU layer candidates.\n"
            f"Last error: {last_err}"
        ) from last_err

    def _load(self, quant: Quant) -> Llama:
        if quant == Quant.Q5_K_M:
            if self._llm_q5 is None:
                self._llm_q5 = self._load_with_fallback_gpu_layers(Q5_MODEL_PATH, quant)
            return self._llm_q5

        if self._llm_q6 is None:
            self._llm_q6 = self._load_with_fallback_gpu_layers(Q6_MODEL_PATH, quant)
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

        # Load model (lazy). Log right after load for trustworthy VRAM signal.
        llm = self._load(chosen)
        _safe_debug("after model load")

        # Build create_completion kwargs; filter to what's supported.
        completion_kwargs: Dict[str, Any] = dict(
            prompt=prompt,
            temperature=profile.temperature,
            min_p=profile.min_p,
            top_k=profile.top_k,
            top_p=profile.top_p,
            max_tokens=profile.max_tokens,
            stop=profile.stop,
            # Improves repeated calls with similar prompts; ignored if unsupported.
            cache_prompt=True, # This is the most likely “silent behavior change” knob across builds.
        )
        completion_kwargs = _filter_kwargs_for_callable(llm.create_completion, completion_kwargs)

        start = time.time()
        raw = llm.create_completion(**completion_kwargs)
        elapsed = time.time() - start

        _safe_debug("after create_completion (inference)")

        text = raw["choices"][0]["text"]

        # Logging should never break inference
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n\n==============================\n")
                f.write(f"Chosen Quant: {chosen.value}\n")
                f.write(f"Elapsed: {elapsed:,d}\n")
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
