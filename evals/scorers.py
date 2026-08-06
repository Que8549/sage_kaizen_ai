"""
evals/scorers.py — Layer 3, tier 1: deterministic quality scorers.

Free, fast, and objective. No LLM judge, no human. These run on every candidate
and catch this system's *actual* failure modes rather than generic ones.

Each scorer answers one question about one response. `score_response()` runs all
of them; `compare_reports()` turns two runs into a regression verdict.

Why these five
--------------
* **CJK ratio** — Qwen2.5-Omni-7B code-switches to Chinese mid-response on
  long-form creative tasks. That was discovered anecdotally and worked around in
  `router.CREATIVE_HINTS` (+3 score routes creative writing to ARCHITECT) without
  ever being measured. This scorer makes it a number.
* **think leakage** — brains.yaml sets `reasoning_format: deepseek`, so thinking
  belongs in `reasoning_content`, never in `content`. Leakage breaks the UI and
  would be read aloud by TTS.
* **citations** — the RAG_GROUNDED_RESPONSE template asks for grounded answers;
  a model that ignores injected context silently defeats the whole RAG stack.
* **refusal** — the system prompt is deliberately unrestricted. A candidate that
  is more censored is a regression *for this system*, however it scores publicly.
* **length** — catches both early stopping (ANTI_EARLY_STOP exists as a template
  for a reason) and runaway generation.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "cjk_ratio",
    "contains_cjk",
    "has_think_leakage",
    "count_citations",
    "looks_like_refusal",
    "ScoreCard",
    "score_response",
    "ScoreReport",
    "score_responses",
    "compare_reports",
]

# CJK Unified Ideographs, plus the Extension A block and Japanese kana. The
# incumbent's failure mode is Mandarin, but kana would be equally wrong here.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x30FF),   # hiragana + katakana
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
)

# Citation shapes the prompt library asks for, and that the retrievers emit:
#   [1]  [Source Name]  [Title / Section | score=0.912]
_CITATION_RE = re.compile(r"\[(?:\d+|[^\]\n]{2,80})\]")

# Deliberately narrow. A response that merely *discusses* refusal must not match;
# only an actual refusal opening should.
_REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"^\s*(?:i'm sorry,? but|i am sorry,? but)\b",
        r"^\s*i (?:can(?:'|no)t|am unable to|won't) (?:help|assist|comply|provide)\b",
        r"^\s*(?:sorry,? )?i (?:can(?:'|no)t|am not able to) (?:help|assist) with (?:that|this)\b",
        r"\bas an ai (?:language )?model,? i (?:can(?:'|no)t|am unable)\b",
        r"^\s*i must decline\b",
    )
)

_THINK_MARKERS = ("<think>", "</think>")


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------

def cjk_ratio(text: str) -> float:
    """
    Fraction of characters in CJK/kana ranges, 0.0–1.0.

    Whitespace is excluded from the denominator so short replies are not
    flattened by indentation. Returns 0.0 for empty input.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    hits = sum(
        1 for c in chars
        if any(lo <= ord(c) <= hi for lo, hi in _CJK_RANGES)
    )
    return hits / len(chars)


def contains_cjk(text: str, *, threshold: float = 0.005) -> bool:
    """
    True when CJK content exceeds `threshold`.

    Not zero: a legitimate English answer may quote a single ideograph when the
    question is *about* Chinese. The failure being detected is paragraphs of it.
    """
    return cjk_ratio(text) > threshold


def has_think_leakage(text: str) -> bool:
    """True when thinking markers appear in what should be clean content."""
    return any(marker in text for marker in _THINK_MARKERS)


def count_citations(text: str) -> int:
    """Count bracketed citation markers."""
    return len(_CITATION_RE.findall(text))


def looks_like_refusal(text: str) -> bool:
    """
    True when the response opens with a refusal.

    Deliberately conservative — false negatives are preferable to flagging every
    response that happens to contain the word "sorry".
    """
    head = text.strip()[:400]
    return any(p.search(head) for p in _REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# Per-response scorecard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreCard:
    """Deterministic scores for one response."""

    prompt_id: str
    slice_name: str
    char_count: int
    cjk_ratio: float
    cjk_flag: bool
    think_leak: bool
    citations: int
    citations_expected: bool
    missing_citations: bool
    refusal: bool

    @property
    def hard_failures(self) -> list[str]:
        """Failures that veto an upgrade regardless of any judge's opinion."""
        out: list[str] = []
        if self.cjk_flag:
            out.append(f"cjk_ratio={self.cjk_ratio:.3f}")
        if self.think_leak:
            out.append("think_leak")
        if self.missing_citations:
            out.append("missing_citations")
        return out

    @property
    def clean(self) -> bool:
        return not self.hard_failures

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_response(
    text: str,
    *,
    prompt_id: str = "",
    slice_name: str = "",
    citations_expected: bool = False,
    cjk_threshold: float = 0.005,
) -> ScoreCard:
    """
    Score one response.

    `citations_expected` should be True when RAG/wiki/search context was injected
    into the turn — `context_injection_json` records exactly that per turn, so a
    golden set mined from the logs carries it for free.
    """
    ratio = cjk_ratio(text)
    citations = count_citations(text)
    return ScoreCard(
        prompt_id=prompt_id,
        slice_name=slice_name,
        char_count=len(text),
        cjk_ratio=ratio,
        cjk_flag=ratio > cjk_threshold,
        think_leak=has_think_leakage(text),
        citations=citations,
        citations_expected=citations_expected,
        missing_citations=citations_expected and citations == 0,
        refusal=looks_like_refusal(text),
    )


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass
class ScoreReport:
    """Aggregate deterministic scores over a response set."""

    label: str
    cards: list[ScoreCard] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.cards)

    @property
    def cjk_failures(self) -> list[ScoreCard]:
        return [c for c in self.cards if c.cjk_flag]

    @property
    def think_leaks(self) -> list[ScoreCard]:
        return [c for c in self.cards if c.think_leak]

    @property
    def missing_citations(self) -> list[ScoreCard]:
        return [c for c in self.cards if c.missing_citations]

    @property
    def refusals(self) -> list[ScoreCard]:
        return [c for c in self.cards if c.refusal]

    @property
    def refusal_rate(self) -> float:
        return len(self.refusals) / self.n if self.n else 0.0

    @property
    def mean_chars(self) -> float:
        return statistics.fmean(c.char_count for c in self.cards) if self.cards else 0.0

    @property
    def stdev_chars(self) -> float:
        if len(self.cards) < 2:
            return 0.0
        return statistics.stdev([c.char_count for c in self.cards])

    @property
    def clean_rate(self) -> float:
        if not self.n:
            return 0.0
        return sum(1 for c in self.cards if c.clean) / self.n

    def by_slice(self) -> dict[str, list[ScoreCard]]:
        out: dict[str, list[ScoreCard]] = {}
        for c in self.cards:
            out.setdefault(c.slice_name or "(unsliced)", []).append(c)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n": self.n,
            "clean_rate": self.clean_rate,
            "cjk_failures": len(self.cjk_failures),
            "think_leaks": len(self.think_leaks),
            "missing_citations": len(self.missing_citations),
            "refusal_rate": self.refusal_rate,
            "mean_chars": self.mean_chars,
            "stdev_chars": self.stdev_chars,
            "cards": [c.to_dict() for c in self.cards],
        }

    def render(self) -> str:
        lines = [
            f"Deterministic scores — {self.label}  (n={self.n})",
            "-" * 60,
            f"  clean responses     {self.clean_rate:>8.1%}",
            f"  CJK failures        {len(self.cjk_failures):>8}",
            f"  <think> leaks       {len(self.think_leaks):>8}",
            f"  missing citations   {len(self.missing_citations):>8}",
            f"  refusal rate        {self.refusal_rate:>8.1%}",
            f"  mean length         {self.mean_chars:>8.0f} chars "
            f"(sd {self.stdev_chars:.0f})",
        ]
        per_slice = self.by_slice()
        if len(per_slice) > 1:
            lines.append("  by slice:")
            for name, cards in sorted(per_slice.items()):
                bad = sum(1 for c in cards if not c.clean)
                lines.append(f"    {name:<24}{len(cards):>4} responses, {bad} unclean")
        return "\n".join(lines)


def score_responses(
    responses: Iterable[dict[str, Any]], *, label: str = "", cjk_threshold: float = 0.005
) -> ScoreReport:
    """
    Score a set of responses.

    Each item needs `text`; `prompt_id`, `slice`, and `citations_expected` are
    optional and default to empty/False.
    """
    cards = [
        score_response(
            item.get("text", ""),
            prompt_id=str(item.get("prompt_id", "")),
            slice_name=str(item.get("slice", "")),
            citations_expected=bool(item.get("citations_expected", False)),
            cjk_threshold=cjk_threshold,
        )
        for item in responses
    ]
    return ScoreReport(label=label, cards=cards)


# ---------------------------------------------------------------------------
# Regression verdict
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreComparison:
    """Candidate vs baseline on the deterministic scorers."""

    baseline_label: str
    candidate_label: str
    regressions: list[str]
    improvements: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.regressions

    def render(self) -> str:
        lines = [f"{self.candidate_label} vs {self.baseline_label}"]
        for r in self.regressions:
            lines.append(f"  REGRESSION  {r}")
        for i in self.improvements:
            lines.append(f"  improvement {i}")
        for n in self.notes:
            lines.append(f"  note        {n}")
        lines.append("  PASS" if self.passed else "  FAIL")
        return "\n".join(lines)


def compare_reports(
    baseline: ScoreReport,
    candidate: ScoreReport,
    *,
    refusal_tolerance: float = 0.05,
    length_tolerance_sd: float = 2.0,
) -> ScoreComparison:
    """
    Turn two reports into an upgrade verdict on the deterministic axes.

    The three hard-failure counts are *ratchets*: a candidate may improve on
    them but must never get worse. Refusal rate and mean length are tolerance
    bands, since small movement there is normal between models.
    """
    regressions: list[str] = []
    improvements: list[str] = []
    notes: list[str] = []

    for name, base_n, cand_n in (
        ("CJK failures", len(baseline.cjk_failures), len(candidate.cjk_failures)),
        ("<think> leaks", len(baseline.think_leaks), len(candidate.think_leaks)),
        ("missing citations", len(baseline.missing_citations),
         len(candidate.missing_citations)),
    ):
        if cand_n > base_n:
            regressions.append(f"{name}: {base_n} -> {cand_n}")
        elif cand_n < base_n:
            improvements.append(f"{name}: {base_n} -> {cand_n}")

    delta_refusal = candidate.refusal_rate - baseline.refusal_rate
    if delta_refusal > refusal_tolerance:
        regressions.append(
            f"refusal rate: {baseline.refusal_rate:.1%} -> {candidate.refusal_rate:.1%} "
            f"(system prompt is deliberately unrestricted)"
        )
    elif delta_refusal < -refusal_tolerance:
        improvements.append(
            f"refusal rate: {baseline.refusal_rate:.1%} -> {candidate.refusal_rate:.1%}"
        )

    if baseline.stdev_chars > 0:
        z = abs(candidate.mean_chars - baseline.mean_chars) / baseline.stdev_chars
        if z > length_tolerance_sd:
            direction = "shorter" if candidate.mean_chars < baseline.mean_chars else "longer"
            regressions.append(
                f"mean length {direction}: {baseline.mean_chars:.0f} -> "
                f"{candidate.mean_chars:.0f} chars ({z:.1f} sd)"
            )
    else:
        notes.append("baseline length stdev is 0 — length check skipped")

    if baseline.n != candidate.n:
        notes.append(
            f"response counts differ ({baseline.n} vs {candidate.n}) — "
            "runs may not be comparable"
        )

    return ScoreComparison(
        baseline_label=baseline.label,
        candidate_label=candidate.label,
        regressions=regressions,
        improvements=improvements,
        notes=notes,
    )
