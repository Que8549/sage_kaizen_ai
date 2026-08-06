"""
scripts/run_scorers.py — Layer 3 tier 1: deterministic quality scorers.

    # score one response set
    python scripts/run_scorers.py --responses benchmarks/results/cand.jsonl

    # compare a candidate against a baseline
    python scripts/run_scorers.py --responses cand.jsonl --baseline base.jsonl

Input is JSONL, one response per line:

    {"prompt_id": "g-001", "slice": "creative",
     "citations_expected": false, "text": "..."}

`citations_expected` should be true when RAG/wiki/search context was injected —
a golden set mined from `context_injection_json` carries that for free.

Exit code is 0 when nothing regressed, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evals.scorers import compare_reports, score_responses  # noqa: E402


def _load_jsonl(path: str | Path) -> list[dict]:
    items: list[dict] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{n}: invalid JSON — {exc}")
        if record.get("_header"):
            continue
        items.append(record)
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Layer 3 tier 1: deterministic quality scorers."
    )
    parser.add_argument("--responses", required=True, help="candidate responses (JSONL)")
    parser.add_argument("--baseline", help="baseline responses (JSONL) to compare against")
    parser.add_argument("--cjk-threshold", type=float, default=0.005)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    candidate = score_responses(
        _load_jsonl(args.responses),
        label=Path(args.responses).stem,
        cjk_threshold=args.cjk_threshold,
    )

    if args.json:
        print(json.dumps(candidate.to_dict(), indent=2))
    else:
        print(candidate.render())

    if not args.baseline:
        # Without a baseline this is descriptive, not a verdict — but hard
        # failures are absolute, so still fail on them.
        unclean = [c for c in candidate.cards if not c.clean]
        if unclean:
            print(f"\n{len(unclean)} response(s) have hard failures:")
            for card in unclean[:10]:
                print(f"  {card.prompt_id or '(no id)':<12} {', '.join(card.hard_failures)}")
        return 1 if unclean else 0

    baseline = score_responses(
        _load_jsonl(args.baseline),
        label=Path(args.baseline).stem,
        cjk_threshold=args.cjk_threshold,
    )
    comparison = compare_reports(baseline, candidate)
    print()
    print(comparison.render())
    return 0 if comparison.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
