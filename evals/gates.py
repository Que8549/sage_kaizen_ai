"""
evals/gates.py — Layer 1: hard compatibility gates.

Binary pass/fail. A candidate model that fails a blocking gate is not worth
benchmarking, let alone judging: a leaderboard win that costs the audio encoder
is a downgrade for this system.

This is CLAUDE.md §11's "Functionality Checklist" turned from a list you read
into a check you run.

Two kinds of gate:

  * **static** — read GGUF metadata and config. No GPU, no server, ~1 second.
  * **live**   — probe a running llama-server. Needs the brain up.

Static gates are the ones worth running before downloading 20 GB of anything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from evals.gguf_meta import GgufError, GgufMetadata, read_gguf_metadata

__all__ = [
    "GateResult",
    "GateReport",
    "static_gates_fast",
    "static_gates_architect",
    "run_static_gates",
    "check_router_label_discipline",
    "check_think_contract",
    "ROUTER_LABELS",
    "DEVICE_CO_TENANTS",
]

# The four labels router.llm_route() parses. A FAST candidate that cannot emit
# exactly one of these breaks two-tier routing — see router._CLASSIFY_SYSTEM.
ROUTER_LABELS: frozenset[str] = frozenset(
    {"FAST", "ARCHITECT", "SEARCH", "ARCHITECT+SEARCH"}
)

# What else *can* live on each GPU, as approximate resident sizes in GiB.
#
# This is a PLANNING table, not an automatic deduction. `free_vram_gib()` reports
# memory that is actually free right now, so anything already loaded — the
# desktop, a running embed server — is implicitly accounted for. Subtracting
# these on top would double-count, which is exactly the false FAIL this table
# produced on the incumbent ARCHITECT config when it was first written.
#
# Pass one of these explicitly to model "what if I also start X", or measure
# free VRAM with the co-tenants already running, which is more accurate.
# Sizes below are MEASURED, not estimated (2026-08-06, nvidia-smi deltas):
# BGE-M3 1152 MiB, jina-clip-v2 3234 MiB. The earlier 2.0 GiB figure for
# jina-clip-v2 understated it by ~60%, which is the wrong direction for a gate
# whose job is to refuse a model that will not fit.
DEVICE_CO_TENANTS: dict[str, dict[str, float]] = {
    # CUDA0 hosts ARCHITECT alone as of 2026-08-06. BGE-M3 was moved off it —
    # the display GPU was down to 106 MiB free with both resident.
    "CUDA0": {},
    # CUDA1 is the crowded one: FAST shares it with BGE-M3, the wiki embed
    # service and CLAP. Wiki embed B only runs during ingest, which never runs
    # concurrently with the app — so it is listed but is not a chat-time tenant.
    "CUDA1": {
        "BGE-M3 embed (8020)": 1.13,
        "wiki embed A (8031, jina-clip-v2)": 3.16,
        "wiki embed B (8032, jina-clip-v2, ingest only)": 3.16,
        "CLAP embed (8040)": 1.5,
    },
    "CUDA2": {},
}

_GIB = 1024 ** 3


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict."""

    name: str
    passed: bool
    detail: str
    blocking: bool = True

    @property
    def symbol(self) -> str:
        if self.passed:
            return "PASS"
        return "FAIL" if self.blocking else "WARN"

    def __str__(self) -> str:
        return f"[{self.symbol}] {self.name}: {self.detail}"


@dataclass(frozen=True)
class GateReport:
    """All gate verdicts for one candidate."""

    brain: str
    model_path: str
    results: list[GateResult] = field(default_factory=list)

    @property
    def blocking_failures(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed and r.blocking]

    @property
    def warnings(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed and not r.blocking]

    @property
    def passed(self) -> bool:
        """True only when every blocking gate passed. Warnings do not block."""
        return not self.blocking_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "brain": self.brain,
            "model_path": self.model_path,
            "passed": self.passed,
            "results": [
                {"name": r.name, "passed": r.passed, "detail": r.detail,
                 "blocking": r.blocking}
                for r in self.results
            ],
        }

    def render(self) -> str:
        lines = [f"Gate report — {self.brain}  ({self.model_path})", "-" * 72]
        lines += [str(r) for r in self.results]
        lines.append("-" * 72)
        if self.passed:
            note = f"PASSED ({len(self.warnings)} warning(s))" if self.warnings else "PASSED"
        else:
            note = f"BLOCKED by {len(self.blocking_failures)} gate(s)"
        lines.append(note)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Static gates
# ---------------------------------------------------------------------------

def _load_meta(path: str | Path) -> tuple[GgufMetadata | None, GateResult]:
    """Read GGUF metadata, returning the failure gate if it cannot be read."""
    try:
        meta = read_gguf_metadata(path)
    except FileNotFoundError:
        return None, GateResult("model_file", False, f"not found: {path}")
    except GgufError as exc:
        return None, GateResult("model_file", False, f"unreadable GGUF: {exc}")
    return meta, GateResult(
        "model_file", True,
        f"{meta.architecture or '?'} arch, {meta.tensor_count} tensors",
    )


def gate_combined_mmproj(mmproj_path: str | Path) -> list[GateResult]:
    """
    FAST requires ONE mmproj carrying both audio and vision.

    Qwen2.5-Omni is currently the only model in the llama.cpp ecosystem that
    does this (projector type "qwen2.5o"). Splitting the modalities across two
    files, or dropping either encoder, breaks `chat_service` media routing:
    every modality is routed to FAST because ARCHITECT's mmproj is disabled to
    preserve speculative decoding.
    """
    meta, load_result = _load_meta(mmproj_path)
    results = [GateResult("mmproj_file", load_result.passed, load_result.detail)]
    if meta is None:
        return results

    results.append(GateResult(
        "mmproj_vision_encoder", meta.has_vision_encoder,
        "present" if meta.has_vision_encoder else "MISSING — image/video upload breaks",
    ))
    results.append(GateResult(
        "mmproj_audio_encoder", meta.has_audio_encoder,
        "present" if meta.has_audio_encoder else "MISSING — .wav/.mp3 upload breaks",
    ))
    both = meta.has_vision_encoder and meta.has_audio_encoder
    results.append(GateResult(
        "mmproj_combined", both,
        f"projector={meta.projector_type or '?'} carries both modalities" if both
        else "audio and vision must live in ONE mmproj",
    ))
    return results


def gate_mtp_head(meta: GgufMetadata, expected_heads: int = 1) -> GateResult:
    """
    ARCHITECT's throughput depends on built-in MTP prediction heads.

    block_count exceeds the transformer layer count by the number of MTP heads;
    brains.yaml's `spec_draft_n_max` must equal that difference. A candidate
    without MTP is not disqualified — but it forfeits the measured ~1.8x
    speedup, so this is a warning the operator must weigh, not a hard veto.
    """
    blocks = meta.block_count
    if blocks is None:
        return GateResult(
            "mtp_head", False, "block_count not declared — cannot verify",
            blocking=False,
        )
    layer_key = f"{meta.architecture}.block_count"
    detail = f"{layer_key}={blocks}"
    if blocks and expected_heads:
        return GateResult(
            "mtp_head", True,
            f"{detail} (expect {expected_heads} MTP head(s); set spec_draft_n_max to match)",
            blocking=False,
        )
    return GateResult("mtp_head", False, f"{detail} — no MTP head", blocking=False)


def gate_vram_budget(
    model_path: str | Path,
    device: str,
    *,
    free_gib: float,
    overhead_gib: float = 0.0,
    co_tenants: dict[str, float] | None = None,
    floor_gib: float = 2.0,
) -> GateResult:
    """
    Does the model fit on `device` alongside everything else that lives there?

    Deliberately conservative and deliberately crude: it compares the model file
    size plus a caller-supplied overhead (mmproj + KV cache + compute buffer)
    against free VRAM. Per-architecture KV maths is a rabbit hole — Qwen3.6 uses
    KV on only 16 of 64 layers — and brains.yaml already carries the
    hand-computed budget for the incumbent.

    `free_gib` is expected to come from `free_vram_gib()`, i.e. memory actually
    free at this moment. Anything already resident is therefore already counted;
    `co_tenants` defaults to **nothing** so it is not double-counted. Pass a
    DEVICE_CO_TENANTS entry explicitly to model "what if I also start X".

    `floor_gib` mirrors brains.yaml's `fit_target`: headroom that must remain
    free after loading.
    """
    p = Path(model_path)
    if not p.is_file():
        return GateResult("vram_budget", False, f"model not found: {p}")

    model_gib = p.stat().st_size / _GIB
    tenants = co_tenants or {}
    tenant_gib = sum(tenants.values())
    needed = model_gib + overhead_gib
    available = free_gib - tenant_gib
    remaining = available - needed

    detail = (
        f"model {model_gib:.1f} + overhead {overhead_gib:.1f} = {needed:.1f} GiB; "
        f"{device} free {free_gib:.1f} GiB"
    )
    if tenants:
        note = ", ".join(f"{k} {v:.1f}" for k, v in tenants.items())
        detail += f" - planned co-tenants {tenant_gib:.1f} ({note}) = {available:.1f} GiB"
    detail += f"; headroom {remaining:.1f} GiB (floor {floor_gib:.1f})"
    return GateResult("vram_budget", remaining >= floor_gib, detail)


def gate_context_size(meta: GgufMetadata, required: int) -> GateResult:
    """
    Does the model's trained context reach what brains.yaml asks for?

    ARCHITECT needs >= 128K: the Qwen3.6 model card requires it for thinking
    mode, and the review service routinely sends 70K-char scopes.
    """
    trained = None
    for key, value in meta.kv.items():
        if key.endswith(".context_length"):
            try:
                trained = int(value)
            except (TypeError, ValueError):
                trained = None
            break
    if trained is None:
        return GateResult(
            "context_size", False, "context_length not declared", blocking=False
        )
    return GateResult(
        "context_size", trained >= required,
        f"trained {trained:,} vs required {required:,}",
    )


def static_gates_fast(
    model_path: str | Path,
    mmproj_path: str | Path,
    *,
    free_gib: float | None = None,
    overhead_gib: float = 3.5,
    required_ctx: int = 32768,
    plan_co_tenants: bool = False,
) -> GateReport:
    """
    Static gates for a FAST-brain candidate (CUDA1, multimodal, TTFT-bound).

    `plan_co_tenants=True` additionally reserves room for both wiki embed
    services and CLAP — use it when sizing a candidate that must coexist with a
    wiki ingest session.
    """
    results: list[GateResult] = []
    meta, load_result = _load_meta(model_path)
    results.append(load_result)

    if meta is not None:
        results.append(gate_context_size(meta, required_ctx))

    results.extend(gate_combined_mmproj(mmproj_path))

    if free_gib is not None:
        results.append(gate_vram_budget(
            model_path, "CUDA1", free_gib=free_gib, overhead_gib=overhead_gib,
            co_tenants=DEVICE_CO_TENANTS["CUDA1"] if plan_co_tenants else None,
        ))
    return GateReport("fast", str(model_path), results)


def static_gates_architect(
    model_path: str | Path,
    *,
    free_gib: float | None = None,
    overhead_gib: float = 5.5,
    required_ctx: int = 131072,
    plan_co_tenants: bool = False,
) -> GateReport:
    """
    Static gates for an ARCHITECT candidate (CUDA0, reasoning, throughput-bound).

    Default overhead 5.5 GiB matches brains.yaml's hand-computed budget:
    KV q8_0 @128K ~4.0 + MTP draft KV ~0.05 + compute buffer ~1.5.
    """
    results: list[GateResult] = []
    meta, load_result = _load_meta(model_path)
    results.append(load_result)

    if meta is not None:
        results.append(gate_context_size(meta, required_ctx))
        results.append(gate_mtp_head(meta))

    if free_gib is not None:
        results.append(gate_vram_budget(
            model_path, "CUDA0", free_gib=free_gib, overhead_gib=overhead_gib,
            co_tenants=DEVICE_CO_TENANTS["CUDA0"] if plan_co_tenants else None,
        ))
    return GateReport("architect", str(model_path), results)


def run_static_gates(brain: str, **kwargs: Any) -> GateReport:
    """Dispatch to the right static gate set. `brain` is "fast" or "architect"."""
    dispatch: dict[str, Callable[..., GateReport]] = {
        "fast": static_gates_fast,
        "architect": static_gates_architect,
    }
    try:
        fn = dispatch[brain.lower()]
    except KeyError:
        raise ValueError(
            f"unknown brain {brain!r} — expected one of {sorted(dispatch)}"
        ) from None
    return fn(**kwargs)


# ---------------------------------------------------------------------------
# Live gates — require a running llama-server
# ---------------------------------------------------------------------------

def check_router_label_discipline(
    completions: Sequence[str], *, min_rate: float = 0.98
) -> GateResult:
    """
    Did the candidate emit clean router labels?

    `router.llm_route` maps the reply onto a brain with substring matching, so a
    chatty model ("Sure! This looks like a FAST query.") still routes — but a
    model that editorialises burns tokens on the latency-critical path and can
    match both ARCHITECT and SEARCH by accident. The gate demands the reply BE a
    label, not merely contain one.

    Pass `completions` from 50-ish representative prompts.
    """
    if not completions:
        return GateResult("router_label_discipline", False, "no completions supplied")

    clean = sum(1 for c in completions if c.strip().upper() in ROUTER_LABELS)
    rate = clean / len(completions)
    offenders = [c.strip()[:40] for c in completions
                 if c.strip().upper() not in ROUTER_LABELS][:3]
    detail = f"{clean}/{len(completions)} exact labels ({rate:.0%})"
    if offenders:
        detail += f"; e.g. {offenders}"
    return GateResult("router_label_discipline", rate >= min_rate, detail)


def check_think_contract(response_payload: dict[str, Any]) -> list[GateResult]:
    """
    Does the candidate honour the thinking contract ARCHITECT depends on?

    brains.yaml sets `reasoning_format: deepseek`, which routes thinking tokens
    to `reasoning_content` and leaves `content` clean. Two things break if that
    changes:

      * the Streamlit UI renders raw <think> in the answer body
      * voice_bridge._TtsFilter relies on <think>/</think> arriving as discrete
        markers to suppress reasoning from TTS — thinking leaked into `content`
        would be spoken aloud

    Pass one non-streaming /v1/chat/completions response body.
    """
    try:
        message = response_payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return [GateResult("think_contract", False, "malformed completion payload")]

    content = message.get("content") or ""
    reasoning = message.get("reasoning_content")

    results = [GateResult(
        "think_reasoning_field", reasoning is not None,
        "reasoning_content present" if reasoning is not None
        else "reasoning_content MISSING — reasoning_format=deepseek not honoured",
    )]
    leaked = "<think>" in content or "</think>" in content
    results.append(GateResult(
        "think_no_leak_into_content", not leaked,
        "content is clean" if not leaked
        else "<think> leaked into content — UI and TTS filter both break",
    ))
    return results


def parse_props_context(props_payload: dict[str, Any]) -> GateResult:
    """Check a live server's /props reports the context size brains.yaml asked for."""
    n_ctx = None
    for key in ("n_ctx", "default_generation_settings"):
        value = props_payload.get(key)
        if isinstance(value, int):
            n_ctx = value
            break
        if isinstance(value, dict) and isinstance(value.get("n_ctx"), int):
            n_ctx = value["n_ctx"]
            break
    if n_ctx is None:
        return GateResult("live_context_size", False, "/props did not report n_ctx",
                          blocking=False)
    return GateResult("live_context_size", True, f"server reports n_ctx={n_ctx:,}",
                      blocking=False)


def free_vram_gib(device_index: int) -> float | None:
    """
    Free VRAM on a device, in GiB, via NVML. None when NVML is unavailable.

    Kept optional so the static gates run on a machine with no NVIDIA driver.
    """
    try:
        import pynvml
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(info.free) / _GIB
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
