"""
evals/bench.py — Layer 2: performance measurement and run comparison.

Wraps `llama-bench` and turns its output into comparable, stored records.

What llama-bench does and does not measure
------------------------------------------
It feeds **randomly generated tokens**, which do not exercise speculative
decoding meaningfully, and it exposes no `--spec-*` flags. So its `tg` figure is
the model's decode ceiling *without* MTP:

    base tg       llama-bench                  decode without MTP
    effective tg  llama-server + real prompts  decode with MTP
    MTP speedup   effective / base

Measuring only `llama-bench` understates an MTP model and overstates a
non-MTP one. Both numbers are needed before comparing candidates.

Operational note: `llama-bench` loads its own copy of the model. Running it
while `llama-server` holds the same weights double-allocates VRAM
(2 x 21.3 GiB > 32 GiB on CUDA0), so callers must stop the server first —
`scripts/run_bench.py` refuses to start if the port is listening.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "BenchRow",
    "BenchRun",
    "parse_llama_bench_json",
    "load_run",
    "save_run",
    "compare_runs",
    "RunComparison",
    "build_llama_bench_argv",
]

# llama-bench writes progress lines ("llama-bench: benchmark 1/4: ...") and CUDA
# banners to the same stream as its JSON when both are captured together, so the
# payload has to be recovered rather than json.loads()'d directly.
# The optional inner group matches an empty "[]" too, so a run that produced no
# measurements reports that specifically rather than "no JSON array found".
_JSON_BLOCK = re.compile(r"\[\s*(?:\{.*})?\s*]", re.DOTALL)
_NOISE_PREFIXES = (
    "llama-bench:", "ggml_", "load_", "build:", "llama_", "print_info",
    "init_", "  Device ", "register_", "load:",
)


@dataclass(frozen=True)
class BenchRow:
    """One llama-bench measurement."""

    test: str            # "pp512", "tg128"
    n_prompt: int
    n_gen: int
    n_depth: int
    avg_ts: float        # tokens/second
    stddev_ts: float

    @property
    def is_prefill(self) -> bool:
        return self.n_gen == 0

    @property
    def kind(self) -> str:
        return "prefill" if self.is_prefill else "decode"


@dataclass
class BenchRun:
    """A labelled set of measurements for one model configuration."""

    label: str
    model_filename: str
    model_type: str = ""
    model_size_bytes: int = 0
    build_commit: str = ""
    build_number: int = 0
    gpu_info: str = ""
    flash_attn: int = 0
    type_k: str = ""
    type_v: str = ""
    n_ubatch: int = 0
    devices: str = ""
    measured_at: str = ""
    rows: list[BenchRow] = field(default_factory=list)

    @property
    def model_size_gib(self) -> float:
        return self.model_size_bytes / (1024 ** 3)

    def row(self, test: str, depth: int = 0) -> BenchRow | None:
        for r in self.rows:
            if r.test == test and r.n_depth == depth:
                return r
        return None

    def decode_rows(self) -> list[BenchRow]:
        return [r for r in self.rows if not r.is_prefill]

    def prefill_rows(self) -> list[BenchRow]:
        return [r for r in self.rows if r.is_prefill]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rows"] = [asdict(r) for r in self.rows]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchRun":
        rows = [BenchRow(**r) for r in data.get("rows", [])]
        payload = {k: v for k, v in data.items() if k != "rows"}
        return cls(rows=rows, **payload)

    def render(self) -> str:
        lines = [
            f"{self.label}  —  {self.model_type or '?'}  ({self.model_size_gib:.2f} GiB)",
            f"  build {self.build_commit or '?'} (b{self.build_number or 0}) | "
            f"{self.devices or '?'} | fa={self.flash_attn} | "
            f"KV {self.type_k or '?'}/{self.type_v or '?'} | ubatch={self.n_ubatch}",
            f"  {'test':<12}{'depth':>8}{'t/s':>12}{'stddev':>10}",
        ]
        for r in self.rows:
            lines.append(f"  {r.test:<12}{r.n_depth:>8}{r.avg_ts:>12.2f}{r.stddev_ts:>10.2f}")
        return "\n".join(lines)


def _strip_noise(raw: str) -> str:
    """Drop llama-bench's progress/banner lines so the JSON payload can be found."""
    return "\n".join(
        line for line in raw.splitlines()
        if not any(line.lstrip().startswith(p) for p in _NOISE_PREFIXES)
    )


def parse_llama_bench_json(raw: str, *, label: str = "") -> BenchRun:
    """
    Parse `llama-bench -o json` output, tolerating interleaved progress lines.

    Raises ValueError when no JSON array can be recovered.
    """
    match = _JSON_BLOCK.search(_strip_noise(raw))
    if match is None:
        raise ValueError("no JSON array found in llama-bench output")
    try:
        records = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"llama-bench JSON is malformed: {exc}") from exc
    if not records:
        raise ValueError("llama-bench returned no measurements")

    first = records[0]
    rows: list[BenchRow] = []
    for rec in records:
        n_prompt = int(rec.get("n_prompt", 0))
        n_gen = int(rec.get("n_gen", 0))
        test = f"pp{n_prompt}" if n_gen == 0 else f"tg{n_gen}"
        rows.append(BenchRow(
            test=test,
            n_prompt=n_prompt,
            n_gen=n_gen,
            n_depth=int(rec.get("n_depth", 0)),
            avg_ts=float(rec.get("avg_ts", 0.0)),
            stddev_ts=float(rec.get("stddev_ts", 0.0)),
        ))

    return BenchRun(
        label=label or first.get("model_type", "unlabelled"),
        model_filename=str(first.get("model_filename", "")),
        model_type=str(first.get("model_type", "")),
        model_size_bytes=int(first.get("model_size", 0)),
        build_commit=str(first.get("build_commit", "")),
        build_number=int(first.get("build_number", 0)),
        gpu_info=str(first.get("gpu_info", "")),
        flash_attn=int(first.get("flash_attn", 0)),
        type_k=str(first.get("type_k", "")),
        type_v=str(first.get("type_v", "")),
        n_ubatch=int(first.get("n_ubatch", 0)),
        devices=str(first.get("devices", "")),
        measured_at=str(first.get("test_time", "")) or datetime.now(timezone.utc).isoformat(),
        rows=rows,
    )


def save_run(run: BenchRun, path: str | Path) -> Path:
    """Write a run as JSON so any two runs can be diffed later."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    return p


def load_run(path: str | Path) -> BenchRun:
    """Load a previously saved run, or parse raw llama-bench output."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Raw llama-bench capture rather than a saved run.
        return parse_llama_bench_json(raw, label=p.stem)
    if isinstance(data, dict) and "rows" in data:
        return BenchRun.from_dict(data)
    return parse_llama_bench_json(raw, label=p.stem)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RowDelta:
    test: str
    n_depth: int
    baseline_ts: float
    candidate_ts: float

    @property
    def pct_change(self) -> float:
        if self.baseline_ts == 0:
            return 0.0
        return (self.candidate_ts - self.baseline_ts) / self.baseline_ts * 100.0

    @property
    def regressed(self) -> bool:
        return self.pct_change < 0


@dataclass(frozen=True)
class RunComparison:
    """Candidate vs baseline, per measurement."""

    baseline_label: str
    candidate_label: str
    deltas: list[RowDelta]
    tolerance_pct: float = 10.0
    missing: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[RowDelta]:
        """Measurements worse than the tolerance allows."""
        return [d for d in self.deltas if d.pct_change < -self.tolerance_pct]

    @property
    def passed(self) -> bool:
        return not self.regressions

    def render(self) -> str:
        lines = [
            f"{self.candidate_label} vs {self.baseline_label} "
            f"(tolerance {self.tolerance_pct:.0f}%)",
            f"  {'test':<12}{'depth':>8}{'baseline':>12}{'candidate':>12}{'change':>10}",
        ]
        for d in self.deltas:
            flag = "  <-- REGRESSION" if d in self.regressions else ""
            lines.append(
                f"  {d.test:<12}{d.n_depth:>8}{d.baseline_ts:>12.2f}"
                f"{d.candidate_ts:>12.2f}{d.pct_change:>9.1f}%{flag}"
            )
        for name in self.missing:
            lines.append(f"  {name:<12}{'—':>8}{'not measured in both runs':>34}")
        lines.append("  PASS" if self.passed else
                     f"  FAIL — {len(self.regressions)} regression(s)")
        return "\n".join(lines)


def compare_runs(
    baseline: BenchRun, candidate: BenchRun, *, tolerance_pct: float = 10.0
) -> RunComparison:
    """
    Compare two runs measurement-by-measurement.

    Only measurements present in BOTH runs are compared; the rest are reported
    as missing rather than silently dropped, because a candidate benchmarked
    with a different matrix is not actually comparable.
    """
    base_index = {(r.test, r.n_depth): r for r in baseline.rows}
    deltas: list[RowDelta] = []
    missing: list[str] = []

    for r in candidate.rows:
        key = (r.test, r.n_depth)
        b = base_index.get(key)
        if b is None:
            missing.append(f"{r.test}@{r.n_depth}")
            continue
        deltas.append(RowDelta(
            test=r.test, n_depth=r.n_depth,
            baseline_ts=b.avg_ts, candidate_ts=r.avg_ts,
        ))

    cand_keys = {(r.test, r.n_depth) for r in candidate.rows}
    missing += [f"{t}@{d}" for (t, d) in base_index if (t, d) not in cand_keys]

    return RunComparison(
        baseline_label=baseline.label,
        candidate_label=candidate.label,
        deltas=deltas,
        tolerance_pct=tolerance_pct,
        missing=sorted(set(missing)),
    )


def build_llama_bench_argv(
    exe: str | Path,
    model: str | Path,
    *,
    device: str,
    n_prompt: Sequence[int] = (512,),
    n_gen: Sequence[int] = (128,),
    depths: Sequence[int] = (0, 8192),
    batch_size: int = 2048,
    ubatch_size: int = 512,
    threads: int = 16,
    cache_type_k: str = "q8_0",
    cache_type_v: str = "q8_0",
    flash_attn: bool = True,
    repetitions: int = 3,
) -> list[str]:
    """
    Build a llama-bench command line mirroring brains.yaml's runtime settings.

    Matching the production flags matters: measuring at ubatch 1024 a model that
    serves at 512 produces a number that does not describe the deployed system.
    """
    return [
        str(exe),
        "-m", str(model),
        "-dev", device,
        "-ngl", "999",
        "-sm", "none",
        "-fa", "on" if flash_attn else "off",
        "-ctk", cache_type_k,
        "-ctv", cache_type_v,
        "-b", str(batch_size),
        "-ub", str(ubatch_size),
        "-t", str(threads),
        "-p", ",".join(str(v) for v in n_prompt),
        "-n", ",".join(str(v) for v in n_gen),
        "-d", ",".join(str(v) for v in depths),
        "-r", str(repetitions),
        "-o", "json",
        "--progress",
    ]
