"""
evals/golden.py — Layer 3: the frozen golden set.

Mines real turns out of `log.sage_kaizen` and stratifies them into the slices
that matter for this system. Nothing invented: the prompts a model is judged on
should be the prompts it will actually receive.

Two structured records make this possible; both are already written on every
turn and neither has been used until now:

  * `route_json`             — brain, score, modality, reasons, search_used,
                               input_chars, processing_time_ms  (router.py)
  * `context_injection_json` — rag_chunks, wiki/search/music/news chars,
                               trimmed flags                    (context_injector.py)

Status
------
**The set is deliberately NOT frozen yet.** The log is currently dominated by
wiki-ingest work rather than representative chat traffic, so a set mined today
would over-weight whatever was being tested and under-weight normal use. Mine it
once ordinary usage has resumed.

Freezing rules, once you do:

  * write it to `benchmarks/golden/<name>.jsonl` and commit it
  * never regenerate in place — a set that changes cannot detect regression
  * new slices go in a NEW versioned file, so old comparisons stay valid
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "GoldenItem",
    "GoldenSet",
    "SLICE_TARGETS",
    "classify_slice",
    "parse_route_json",
    "extract_json_payload",
    "stratified_sample",
    "load_golden_set",
    "save_golden_set",
    "SELECT_TURNS_SQL",
]

# Target composition, mirroring Benchmarking_Kaizen_Models.md §3 Layer 3.
# Sized so the whole set fits one evening of judged comparison.
SLICE_TARGETS: dict[str, int] = {
    "creative": 20,      # the code-switching detector
    "code": 20,
    "tutoring": 20,
    "knowledge": 15,
    "rag_grounded": 25,
    "search_grounded": 15,
    "voice_short": 15,
    "philosophy": 10,
    "multimodal": 10,
}

# Pull candidate turns with both structured records for the same run_id.
# Deliberately a plain string rather than a query builder: it is run by hand
# during a mining session, and being able to paste it into psql matters more
# than composability.
SELECT_TURNS_SQL = """
SELECT
    log_date,
    run_id,
    description
FROM log.sage_kaizen
WHERE log_name = 'sage_kaizen.router'
  AND description LIKE 'route_json %%'
  AND log_date >= %s
ORDER BY log_date DESC
LIMIT %s
"""

_PREFIX_RE = re.compile(r"^\s*(?:route_json|context_injection_json)\s+")


@dataclass(frozen=True)
class GoldenItem:
    """One frozen evaluation prompt, with the context it was answered in."""

    prompt_id: str
    slice_name: str
    prompt: str
    brain: str                      # brain that served it originally
    modality: str = "text"
    input_chars: int = 0
    citations_expected: bool = False
    rag_chunks: int = 0
    search_used: bool = False
    source_run_id: str = ""
    logged_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldenItem":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class GoldenSet:
    """A frozen, versioned set of evaluation prompts."""

    name: str
    items: list[GoldenItem] = field(default_factory=list)
    created: str = ""
    note: str = ""

    def __len__(self) -> int:
        return len(self.items)

    def by_slice(self) -> dict[str, list[GoldenItem]]:
        out: dict[str, list[GoldenItem]] = {}
        for item in self.items:
            out.setdefault(item.slice_name, []).append(item)
        return out

    def composition(self) -> dict[str, int]:
        return {k: len(v) for k, v in sorted(self.by_slice().items())}

    def render(self) -> str:
        lines = [f"Golden set '{self.name}' — {len(self)} items"]
        if self.note:
            lines.append(f"  {self.note}")
        for name, count in self.composition().items():
            target = SLICE_TARGETS.get(name)
            suffix = f" (target {target})" if target else ""
            lines.append(f"  {name:<20}{count:>4}{suffix}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def extract_json_payload(description: str) -> dict[str, Any] | None:
    """
    Pull the JSON object out of a `route_json {...}` log line.

    Returns None when the line is not one of the structured records or its
    payload does not parse — log lines are not a trusted format.
    """
    if not description:
        return None
    stripped = _PREFIX_RE.sub("", description, count=1)
    start = stripped.find("{")
    if start < 0:
        return None
    try:
        payload = json.loads(stripped[start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_route_json(description: str) -> dict[str, Any] | None:
    """Parse a `route_json` line, or None if it is not one."""
    if "route_json" not in description:
        return None
    return extract_json_payload(description)


def classify_slice(
    prompt: str,
    *,
    route: dict[str, Any] | None = None,
    injection: dict[str, Any] | None = None,
) -> str:
    """
    Assign a turn to an evaluation slice.

    Uses the app's OWN hint tuples as the source of truth rather than a second,
    drifting copy — if `router.CREATIVE_HINTS` changes, this follows.

    Precedence is deliberate: modality and grounding describe what the *system*
    did and are checked first; topic hints describe what the user asked and are
    the fallback.
    """
    route = route or {}
    injection = injection or {}
    text = (prompt or "").lower()

    modality = str(route.get("modality", "text"))
    if modality and modality != "text":
        return "multimodal"

    if int(injection.get("rag_chunks", 0) or 0) > 0 or int(
        injection.get("wiki_chars", 0) or 0
    ) > 0:
        return "rag_grounded"

    if bool(route.get("search_used", False)):
        return "search_grounded"

    # Import lazily so this module stays importable without the whole app.
    from chat_service import _KNOWLEDGE_HINTS, _PHILOSOPHY_HINTS, _TEACH_HINTS
    from router import CODE_HINTS, CREATIVE_HINTS

    if any(h in text for h in CREATIVE_HINTS):
        return "creative"
    if any(h in text for h in CODE_HINTS):
        return "code"
    if any(h in text for h in _TEACH_HINTS):
        return "tutoring"
    if any(h in text for h in _PHILOSOPHY_HINTS):
        return "philosophy"
    if any(h in text for h in _KNOWLEDGE_HINTS):
        return "knowledge"

    if int(route.get("input_chars", 0) or 0) < 150:
        return "voice_short"
    return "general"


def stratified_sample(
    items: Sequence[GoldenItem],
    targets: dict[str, int] | None = None,
    *,
    seed: int = 20260806,
) -> list[GoldenItem]:
    """
    Take up to `targets[slice]` items from each slice.

    Seeded so a mining run is reproducible — re-running with the same input and
    seed yields the same set, which matters when the set is about to be frozen.
    Under-filled slices are returned in full rather than padded from elsewhere;
    a thin slice should be visible, not disguised.
    """
    targets = SLICE_TARGETS if targets is None else targets
    rng = random.Random(seed)

    grouped: dict[str, list[GoldenItem]] = {}
    for item in items:
        grouped.setdefault(item.slice_name, []).append(item)

    out: list[GoldenItem] = []
    for slice_name, wanted in targets.items():
        pool = grouped.get(slice_name, [])
        if len(pool) <= wanted:
            out.extend(pool)
        else:
            out.extend(rng.sample(pool, wanted))
    return out


# ---------------------------------------------------------------------------
# Persistence — JSONL, one item per line, diff-friendly under git
# ---------------------------------------------------------------------------

def save_golden_set(golden: GoldenSet, path: str | Path) -> Path:
    """Write a golden set as JSONL. The first line is the header record."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({
        "_header": True,
        "name": golden.name,
        "created": golden.created,
        "note": golden.note,
        "count": len(golden),
        "composition": golden.composition(),
    })]
    lines += [json.dumps(item.to_dict()) for item in golden.items]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def load_golden_set(path: str | Path) -> GoldenSet:
    """Load a golden set from JSONL."""
    p = Path(path)
    name, created, note = p.stem, "", ""
    items: list[GoldenItem] = []

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("_header"):
            name = record.get("name", name)
            created = record.get("created", "")
            note = record.get("note", "")
            continue
        items.append(GoldenItem.from_dict(record))

    return GoldenSet(name=name, items=items, created=created, note=note)
