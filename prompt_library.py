# prompt_library.py
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, List
import os

class TemplateKey(str, Enum):
    UNIVERSAL_DEPTH_ANCHOR = "universal_depth_anchor"
    STRUCTURED_KNOWLEDGE = "structured_knowledge"
    TEACHING_TUTORING = "teaching_tutoring"
    PHILOSOPHY_DEEP_THINKING = "philosophy_deep_thinking"
    ANTI_EARLY_STOP = "anti_early_stop"
    VOICE_LOOP_FRIENDLY = "voice_loop_friendly"
    AUTO_ADAPTIVE_META = "auto_adaptive_meta"


TEMPLATES: Dict[TemplateKey, str] = {
    TemplateKey.UNIVERSAL_DEPTH_ANCHOR: (
        "You are Sage Kaizen.\n"
        "Respond with depth, clarity, and structure.\n"
        "Assume the user values historical accuracy, nuance, and synthesis.\n"
        "Avoid shallow summaries.\n"
    ),
    TemplateKey.STRUCTURED_KNOWLEDGE: (
        "Answer using the following structure:\n\n"
        "1. Historical context and timeframe\n"
        "2. Primary civilizations or schools involved\n"
        "3. Methods, tools, or systems used\n"
        "4. Religious, philosophical, or symbolic meaning\n"
        "5. Long-term influence on later traditions\n\n"
        "Use clear sections and concrete examples.\n"
    ),
    TemplateKey.TEACHING_TUTORING: (
        "Explain this topic as if teaching an intelligent student.\n"
        "Start with the big picture, then zoom into details.\n"
        "Define key terms.\n"
        "Use examples.\n"
        "End with a concise summary.\n"
    ),
    TemplateKey.PHILOSOPHY_DEEP_THINKING: (
        "Analyze this topic from multiple perspectives.\n"
        "Include historical, philosophical, and practical viewpoints.\n"
        "Highlight tensions, debates, or uncertainties.\n"
        "Do not rush to conclusions.\n"
    ),
    TemplateKey.ANTI_EARLY_STOP: "Do not stop early; continue until the topic is thoroughly explored.\n",
    TemplateKey.VOICE_LOOP_FRIENDLY: (
        "Speak clearly and naturally.\n"
        "Use short paragraphs and clean phrasing.\n"
        "Avoid overly long lists unless asked.\n"
    ),
    TemplateKey.AUTO_ADAPTIVE_META: (
        "If this topic benefits from detail, expand fully.\n"
        "If it is simple, be concise.\n"
        "Err on the side of depth.\n"
    ),
}


# ---- System prompts (your named “cores”) ----

SYSTEM_PROMPT_PATH = os.environ.get("SAGE_KAIZEN_SYSTEM_PROMPT_PATH", "sage_kaizen_system_prompt.txt")

def load_system_prompt(path: str = SYSTEM_PROMPT_PATH) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"System prompt file not found: {p.resolve()}\n"
            "Create it (e.g. sage_kaizen_system_prompt.txt)."
        )
    return p.read_text(encoding="utf-8").strip()


# Model-specific “core roles” you requested
sage_fast_core = (
    "You are Sage Kaizen – the Fast Brain.\n"
    "Prioritize speed, clarity, and concise structured answers.\n"
    "If deep architecture or high-stakes correctness is needed, escalate.\n"
)

sage_architect_core = (
    "You are Sage Kaizen – the Architect Brain.\n"
    "Prioritize correctness over speed.\n"
    "Be structured; separate assumptions vs conclusions.\n"
    "When helpful, propose production-ready architecture and tuning.\n"
)


baseline_benchmark_prompts: Dict[str, str] = {
    "stars_religion_baseline": (
        "What is the earliest known civilization to systematically study the stars, "
        "and how did their astronomical observations influence their religious practices "
        "and political authority? Provide historical examples and explain causal relationships."
    ),
    "dual_gpu_architecture": (
        "Design a scalable dual-GPU local AI architecture using llama.cpp where GPU0 "
        "prioritizes deep reasoning and GPU1 prioritizes low-latency responses. Include routing, "
        "KV/cache strategy, failure fallback behavior, telemetry, and a Mermaid diagram."
    ),
    "tutoring_seasons_multi_level": (
        "Explain why seasons occur on Earth for: (1) a 3rd grader, (2) a 10th grader, "
        "(3) a college-level physics student."
    ),
}


def build_messages(
    user_text: str,
    *,
    system_prompt: str,
    core_prompt: str = "",
    templates: Tuple[TemplateKey, ...] = (),
) -> List[dict]:
    """
    Returns OpenAI-style chat messages list for llama-server.
    """
    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    if core_prompt:
        parts.append(core_prompt.strip())
    if templates:
        parts.append("\n".join(TEMPLATES[t].rstrip() for t in templates).strip())

    system = "\n\n".join(p for p in parts if p)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text.strip()},
    ]
