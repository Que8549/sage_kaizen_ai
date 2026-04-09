"""
review_service/nodes/flags_sanity.py — brains.yaml flag sanity check.

Focused ARCHITECT call that validates the llama-server configuration
for flag conflicts, deprecated settings, VRAM budget math, and
context window consistency.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sk_logging import get_logger
from ..state import ReviewState

_LOG = get_logger("sage_kaizen.review_service.flags_sanity")

_SYSTEM_PROMPT = """\
You are validating a llama-server configuration file (brains.yaml) for Sage Kaizen.

Hardware context:
  - GPU0: RTX 5090, 32 GB VRAM (CUDA0) — ARCHITECT brain + BGE-M3 embed
  - GPU1: RTX 5080, 16 GB VRAM (CUDA1) — FAST brain (audio+vision mmproj)
  - CPU: AMD Ryzen 9 9950X3D, 16 physical cores / 32 threads
  - RAM: 192 GB DDR5

Check for these specific issues and report each as [CRITICAL], [HIGH], [MEDIUM], or [LOW]:

1. Flag conflicts:
   - no_kv_unified=true with cont_batching=true (known crash)
   - flash_attn=true without mmproj (fine) vs flash_attn=true with mmproj disabled (fine)
   - split_mode=none with n_gpu_layers=all on a single-GPU brain (correct; confirm)
   - parallel > 1 with no_cont_batching=true (defeats parallelism)

2. VRAM budget math:
   - Compute: model_weights + mmproj + kv_cache + compute_buffer vs GPU VRAM
   - KV cache formula: ctx_size * n_layers * n_kv_heads * head_dim * 2 * cache_type_bytes
   - For Qwen2.5-Omni-7B: 28 layers, 4 KV heads, head_dim=128
   - For Qwen3.5-27B: 64 layers (only 16 use KV), 8 KV heads, head_dim=128
   - q8_0 = 1 byte/element, q4_0 = 0.5 byte/element, f16 = 2 bytes/element

3. Context window:
   - FAST: ctx_size=16384 (model max: 32K; fine for audio+vision use)
   - ARCHITECT: ctx_size=131072 (model requires >=128K for thinking mode; verified)
   - batch_size should not exceed ctx_size

4. Thread counts:
   - threads + threads_batch should not exceed physical core count (16)
   - threads_http is separate (can use more since it is IO-bound)

5. Deprecated or renamed flags:
   - Any flag that changed name in llama.cpp between b8639 (current build) and latest
   - cache_reuse, ngram-map-k, swa_full are ARCHITECT-specific — verify consistency

6. Performance recommendations:
   - Is ubatch_size appropriately sized relative to batch_size?
   - Is cache_ram sufficient for the current system prompt + RAG injection size?
   - Is slot_prompt_similarity tuned for the actual cache hit rate?

Output a markdown bullet list. Group by brain (fast:, architect:, embed:).
If nothing is wrong, say "No issues found for <brain>."
"""


def make_flags_sanity_node(llm: ChatOpenAI):
    async def flags_sanity_node(state: ReviewState) -> dict:
        context = _build_context(state)
        _LOG.info("review.architect_call | node=flags_sanity | context_chars=%d", len(context))
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
        try:
            response = await llm.ainvoke(messages)
            findings = response.content
        except Exception as exc:
            _LOG.exception("flags_sanity failed: %s", exc)
            findings = f"[flags_sanity ERROR: {exc}]"

        _LOG.info("review.architect_call | node=flags_sanity | response_chars=%d", len(findings))
        return {"flags_findings": findings}

    return flags_sanity_node


def _build_context(state: ReviewState) -> str:
    parts: list[str] = []
    if state.get("brains_yaml"):
        parts.append(f"<brains_yaml>\n{state['brains_yaml']}\n</brains_yaml>")
    if state.get("architect_findings"):
        # Give flags_sanity the prior findings so it doesn't duplicate GPU concerns
        parts.append(
            f"<prior_findings_summary>\n"
            f"The architect reviewer already noted the following GPU/performance issues:\n"
            f"{state['architect_findings'][:2000]}\n"
            f"Focus on flag-level issues not already covered above.\n"
            f"</prior_findings_summary>"
        )
    return "\n\n".join(parts)
