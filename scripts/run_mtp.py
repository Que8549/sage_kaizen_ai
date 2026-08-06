"""
scripts/run_mtp.py — Layer 2b: measure the EFFECTIVE decode rate with
speculative decoding, against a live llama-server.

    # measure ARCHITECT, pairing against a saved llama-bench baseline
    python scripts/run_mtp.py --brain architect --label mtp-b10298 \
                              --base-run baseline-b10298

    # a single slice, shorter generations
    python scripts/run_mtp.py --brain architect --label smoke \
                              --only code --max-tokens 256

This is the counterpart to `run_bench.py`, not a replacement:

    run_bench.py   llama-bench, random tokens, server STOPPED  -> base tg
    run_mtp.py     llama-server, real prompts, server RUNNING  -> effective tg

Requires `metrics: true` in the brain's brains.yaml block — llama.cpp's
`/metrics` endpoint is disabled by default and the speculative-decoding
counters live there.

Results land in benchmarks/results/<label>.mtp.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx  # noqa: E402

from evals.bench import load_run  # noqa: E402
from evals.mtp import (  # noqa: E402
    MtpRun, build_prompt_set, fetch_counters, run_turn, save_mtp_run,
)
from server_manager import ManagedServers, _ensure_brain_running  # noqa: E402

_RESULTS = _ROOT / "benchmarks" / "results"


def _base_tg_from(label: str) -> tuple[float, str]:
    """
    Pull the depth-0 tg128 figure out of a saved llama-bench run.

    Depth 0 rather than 8192 because the measured turns start from an empty
    context; pairing against the 8192 figure would compare decode at two
    different context depths and flatter the speedup.
    """
    run = load_run(_RESULTS / f"{label}.run.json")
    row = run.row("tg128", depth=0)
    if row is None:
        raise ValueError(f"run '{label}' has no tg128@0 measurement to pair against")
    return row.avg_ts, label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure effective decode rate with speculative decoding."
    )
    parser.add_argument("--brain", choices=["fast", "architect"], default="architect")
    parser.add_argument("--label", required=True, help="name for this run")
    parser.add_argument("--base-run",
                        help="llama-bench run label to compute the MTP speedup against")
    parser.add_argument("--only", nargs="*",
                        help="prompt slices to run (default: all)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--warmup", action="store_true",
                        help="run one discarded turn first to warm caches")
    parser.add_argument("--no-start", action="store_true",
                        help="fail instead of starting the server if it is down")
    args = parser.parse_args(argv)

    servers = ManagedServers.from_yaml()
    cfg = servers.fast if args.brain == "fast" else servers.architect

    if not bool(cfg.server.get("metrics", False)):
        print(
            f"error: {args.brain} has no `metrics: true` in brains.yaml. The "
            f"speculative-decoding counters live on /metrics, which llama.cpp "
            f"leaves disabled by default.",
            file=sys.stderr,
        )
        return 2

    if args.no_start:
        ok, msg = True, "not checked (--no-start)"
    else:
        ok, msg = _ensure_brain_running(cfg, args.brain.upper())
    print(f"{args.brain}: {msg}")
    if not ok:
        return 3

    base_tg, base_label = 0.0, ""
    if args.base_run:
        try:
            base_tg, base_label = _base_tg_from(args.base_run)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    prompts = build_prompt_set(args.only)
    if not prompts:
        print(f"error: no prompts matched {args.only}", file=sys.stderr)
        return 2

    run = MtpRun(
        label=args.label,
        model=str(cfg.model),
        base_tg=base_tg,
        base_label=base_label,
    )

    with httpx.Client() as client:
        try:
            start = fetch_counters(client, cfg.base_url)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        print(f"spec counters at start: drafted={start.drafted} "
              f"accepted={start.accepted} drafts={start.drafts}")

        if args.warmup:
            print("warmup turn (discarded) ...")
            run_turn(client, cfg.base_url, prompts[0][1],
                     label="warmup", max_tokens=64)

        for label, prompt in prompts:
            print(f"  {label} ...", flush=True)
            run.turns.append(
                run_turn(client, cfg.base_url, prompt,
                         label=label, max_tokens=args.max_tokens)
            )

    print()
    print(run.render())
    out = save_mtp_run(run, _RESULTS / f"{args.label}.mtp.json")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
