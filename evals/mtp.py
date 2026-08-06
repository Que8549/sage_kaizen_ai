"""
evals/mtp.py — Layer 2b: the *effective* decode rate, measured against a live
llama-server.

`evals/bench.py` measures the other half. Keeping them apart is deliberate:

    base tg       llama-bench, random tokens   decode ceiling WITHOUT MTP
    effective tg  llama-server, real prompts   decode WITH MTP
    MTP speedup   effective / base

llama-bench exposes no `--spec-*` flags (re-verified against build b10298), so
it cannot produce the second number no matter how it is invoked. This module
drives real chat completions against a running server and reads the speculative
decoding counters that llama.cpp added to `/metrics` in b10298 (upstream
#26389):

    llamacpp:spec_decode_num_draft_tokens_total     tokens the MTP head drafted
    llamacpp:spec_decode_num_accepted_tokens_total  drafts the target accepted
    llamacpp:spec_decode_num_drafts_total           verification steps

Before those counters existed, acceptance rate had to be scraped out of the
server log. Now it is a two-line HTTP read, which is why this measurement is
worth doing today and was not worth doing in June.

The server must be started with `--metrics` (`metrics: true` in brains.yaml) —
the endpoint is disabled by default and returns 501 otherwise.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

__all__ = [
    "SpecCounters",
    "TurnResult",
    "MtpRun",
    "parse_prometheus",
    "spec_counters",
    "fetch_counters",
    "run_turn",
    "save_mtp_run",
    "load_mtp_run",
    "DEFAULT_PROMPTS",
]

# Prometheus exposition: `name value` or `name{label="x"} value`. Comments and
# blank lines are skipped. Values may be integers, floats or `nan`.
_METRIC_LINE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?P<labels>\{[^}]*})?"
    r"\s+(?P<value>[-+0-9.eE]+|[Nn]a[Nn])$"
)

_DRAFTED = "llamacpp:spec_decode_num_draft_tokens_total"
_ACCEPTED = "llamacpp:spec_decode_num_accepted_tokens_total"
_DRAFTS = "llamacpp:spec_decode_num_drafts_total"


class _HttpClient(Protocol):
    """The slice of httpx.Client this module uses, so tests can pass a fake."""

    def get(self, url: str, **kwargs: Any) -> Any: ...
    def post(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class SpecCounters:
    """A snapshot of the server's speculative-decoding counters."""

    drafted: int = 0
    accepted: int = 0
    drafts: int = 0

    def __sub__(self, other: "SpecCounters") -> "SpecCounters":
        """Delta between two snapshots — the work done by one turn.

        Clamped at zero so a server restart mid-run (counters reset to 0)
        reports nothing rather than a negative rate that would silently
        corrupt an average.
        """
        return SpecCounters(
            drafted=max(0, self.drafted - other.drafted),
            accepted=max(0, self.accepted - other.accepted),
            drafts=max(0, self.drafts - other.drafts),
        )

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens the target model kept. 0.0 when idle."""
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_step(self) -> float:
        """
        Committed tokens per verification step.

        With one MTP head this is bounded by 2.0 (the main token plus one
        accepted draft), and equals 1 + acceptance_rate. It is the ceiling on
        what speculative decoding can buy for this configuration.
        """
        return (self.drafts + self.accepted) / self.drafts if self.drafts else 0.0

    @property
    def is_active(self) -> bool:
        """False when the server is not speculating at all (counters flat)."""
        return self.drafts > 0


def parse_prometheus(text: str) -> dict[str, float]:
    """
    Parse a Prometheus exposition body into {metric_name: value}.

    Labelled series are folded by summing across labels: the only labelled
    metric here is accepted-tokens-per-position, whose sum over positions is
    the meaningful total.
    """
    out: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        name = match.group("name")
        out[name] = out.get(name, 0.0) + value if match.group("labels") else value
    return out


def spec_counters(metrics: dict[str, float]) -> SpecCounters:
    """Extract the three speculative-decoding counters from parsed metrics."""
    return SpecCounters(
        drafted=int(metrics.get(_DRAFTED, 0)),
        accepted=int(metrics.get(_ACCEPTED, 0)),
        drafts=int(metrics.get(_DRAFTS, 0)),
    )


def fetch_counters(client: _HttpClient, base_url: str) -> SpecCounters:
    """
    Read `/metrics` and return the speculative-decoding counters.

    Raises RuntimeError when the endpoint is disabled, which is a setup error
    worth failing loudly on — silently reporting zeros would look exactly like
    a model that never speculates.
    """
    response = client.get(f"{base_url.rstrip('/')}/metrics", timeout=10.0)
    if response.status_code != 200:
        raise RuntimeError(
            f"/metrics returned {response.status_code}; start llama-server with "
            f"--metrics (metrics: true in brains.yaml)"
        )
    return spec_counters(parse_prometheus(response.text))


@dataclass(frozen=True)
class TurnResult:
    """One completion measured end to end, with the spec counters it moved."""

    label: str
    prompt_tokens: int = 0
    predicted_tokens: int = 0
    prompt_ms: float = 0.0
    predicted_ms: float = 0.0
    predicted_per_second: float = 0.0
    reasoning_tokens: int = 0
    counters: SpecCounters = field(default_factory=SpecCounters)

    @property
    def acceptance_rate(self) -> float:
        return self.counters.acceptance_rate

    def speedup_over(self, base_tg: float) -> float:
        """Effective rate divided by a llama-bench base decode rate."""
        return self.predicted_per_second / base_tg if base_tg else 0.0


# Prompts chosen to exercise what ARCHITECT actually serves: long-form reasoning
# with thinking enabled. Speculative acceptance is content-dependent — code and
# structured text draft far better than prose — so a single prompt would give a
# number that does not generalise. Frozen here rather than sampled from the log
# so successive runs stay comparable (the golden set, when it exists, is a
# different instrument with a different purpose).
DEFAULT_PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "reasoning",
        "A train leaves station A at 3pm travelling 60 km/h toward station B, "
        "240 km away. Another leaves B at 4pm travelling 80 km/h toward A. "
        "Work out where and when they meet, then explain which assumptions in "
        "the problem are unrealistic and how each would shift the answer.",
    ),
    (
        "code",
        "Write a Python function that merges overlapping intervals, then walk "
        "through its complexity, its behaviour on empty and single-element "
        "input, and how you would test it.",
    ),
    (
        "creative",
        "Write a 400-word story about a lighthouse keeper who receives a letter "
        "with no return address.",
    ),
    (
        "architecture",
        "Compare write-through and write-back caching for a read-heavy service "
        "with occasional bursts of writes. Cover consistency, failure modes, "
        "and what you would measure before choosing.",
    ),
)


def run_turn(
    client: _HttpClient,
    base_url: str,
    prompt: str,
    *,
    label: str = "",
    max_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    timeout: float = 600.0,
) -> TurnResult:
    """
    Send one non-streaming completion and pair the server's own timings with
    the spec-counter delta it produced.

    Non-streaming on purpose: `timings.predicted_per_second` is measured by the
    server itself, so it excludes the HTTP read loop that would otherwise be
    folded into a client-side stopwatch. This measures decode, not the network.

    Sampling defaults follow the Qwen3.6 model card for thinking mode
    (temp 0.6 / top_p 0.95 / top_k 20) — acceptance rate is sensitive to
    sampling, so measuring at other settings would not describe production.
    """
    url = base_url.rstrip("/")
    before = fetch_counters(client, url)

    response = client.post(
        f"{url}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": False,
            "timings_per_token": True,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    after = fetch_counters(client, url)
    timings = payload.get("timings") or {}
    message = (payload.get("choices") or [{}])[0].get("message") or {}
    reasoning = message.get("reasoning_content") or ""

    return TurnResult(
        label=label or prompt[:32],
        prompt_tokens=int(timings.get("prompt_n", 0)),
        predicted_tokens=int(timings.get("predicted_n", 0)),
        prompt_ms=float(timings.get("prompt_ms", 0.0)),
        predicted_ms=float(timings.get("predicted_ms", 0.0)),
        predicted_per_second=float(timings.get("predicted_per_second", 0.0)),
        # Character count, not a token count — the server does not report
        # reasoning tokens separately. Useful only to confirm thinking ran.
        reasoning_tokens=len(reasoning),
        counters=after - before,
    )


@dataclass
class MtpRun:
    """A labelled set of live-server measurements."""

    label: str
    model: str = ""
    base_tg: float = 0.0          # from the paired llama-bench run, if known
    base_label: str = ""
    measured_at: str = ""
    turns: list[TurnResult] = field(default_factory=list)

    @property
    def totals(self) -> SpecCounters:
        total = SpecCounters()
        for turn in self.turns:
            # Sum of deltas: add via the inverse of __sub__'s clamping by
            # constructing directly, so totals are exact.
            total = SpecCounters(
                drafted=total.drafted + turn.counters.drafted,
                accepted=total.accepted + turn.counters.accepted,
                drafts=total.drafts + turn.counters.drafts,
            )
        return total

    @property
    def weighted_tg(self) -> float:
        """
        Tokens per second across the whole run, weighted by tokens generated.

        A plain mean over turns would let a 40-token turn count as much as a
        512-token one.
        """
        tokens = sum(t.predicted_tokens for t in self.turns)
        seconds = sum(t.predicted_ms for t in self.turns) / 1000.0
        return tokens / seconds if seconds else 0.0

    @property
    def speedup(self) -> float:
        return self.weighted_tg / self.base_tg if self.base_tg else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["turns"] = [asdict(t) for t in self.turns]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MtpRun":
        turns = [
            TurnResult(**{**t, "counters": SpecCounters(**t.get("counters", {}))})
            for t in data.get("turns", [])
        ]
        payload = {k: v for k, v in data.items() if k != "turns"}
        return cls(turns=turns, **payload)

    def render(self) -> str:
        totals = self.totals
        lines = [
            f"{self.label}  —  effective decode with speculative decoding",
            f"  model {self.model or '?'}",
            f"  {'turn':<14}{'pred':>7}{'ms':>10}{'t/s':>9}"
            f"{'drafted':>9}{'accept':>8}{'rate':>8}",
        ]
        for t in self.turns:
            lines.append(
                f"  {t.label:<14}{t.predicted_tokens:>7}{t.predicted_ms:>10.0f}"
                f"{t.predicted_per_second:>9.2f}{t.counters.drafted:>9}"
                f"{t.counters.accepted:>8}{t.acceptance_rate * 100:>7.1f}%"
            )
        lines.append(
            f"  {'TOTAL':<14}{sum(t.predicted_tokens for t in self.turns):>7}"
            f"{sum(t.predicted_ms for t in self.turns):>10.0f}"
            f"{self.weighted_tg:>9.2f}{totals.drafted:>9}"
            f"{totals.accepted:>8}{totals.acceptance_rate * 100:>7.1f}%"
        )
        if not totals.is_active:
            lines.append(
                "  WARNING: no drafts recorded — speculative decoding is not "
                "running (check spec_type in brains.yaml and the startup log)"
            )
        if self.base_tg:
            lines.append(
                f"  base tg {self.base_tg:.2f} t/s ({self.base_label or 'llama-bench'}) "
                f"-> speedup {self.speedup:.2f}x"
            )
            lines.append(
                f"  ceiling at this acceptance rate: "
                f"{totals.tokens_per_step:.2f}x (1 + acceptance, one MTP head)"
            )
        return "\n".join(lines)


def save_mtp_run(run: MtpRun, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not run.measured_at:
        run.measured_at = datetime.now(timezone.utc).isoformat()
    p.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    return p


def load_mtp_run(path: str | Path) -> MtpRun:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MtpRun.from_dict(data)


def build_prompt_set(only: Sequence[str] | None = None) -> list[tuple[str, str]]:
    """DEFAULT_PROMPTS, optionally filtered to named slices."""
    if not only:
        return list(DEFAULT_PROMPTS)
    wanted = {name.lower() for name in only}
    return [(label, text) for label, text in DEFAULT_PROMPTS if label in wanted]
