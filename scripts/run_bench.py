"""
scripts/run_bench.py — Layer 2 performance measurement.

    # measure the incumbent
    python scripts/run_bench.py --brain architect --label baseline

    # measure a candidate
    python scripts/run_bench.py --brain architect --label cand \
                                --model E:/candidate.gguf

    # compare two saved runs
    python scripts/run_bench.py --compare baseline cand

Results land in benchmarks/results/<label>.run.json.

IMPORTANT: llama-bench loads its own copy of the model. Running it while
llama-server holds the same weights double-allocates VRAM (2 x 21.3 GiB > 32 GiB
on CUDA0), so this refuses to start when the brain's port is listening.

Remember what this measures: llama-bench feeds RANDOM tokens and exposes no
speculative-decoding flags, so `tg` here is the decode ceiling WITHOUT MTP.
The effective rate — with MTP, real prompts and HTTP — must be measured against
a running llama-server. Comparing a candidate's llama-bench number against a
production number is comparing two different things.
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evals.bench import (  # noqa: E402
    build_llama_bench_argv, compare_runs, load_run, parse_llama_bench_json, save_run,
)
from server_manager import ManagedServers  # noqa: E402

_RESULTS = _ROOT / "benchmarks" / "results"
_DEVICE = {"fast": "CUDA1", "architect": "CUDA0"}


def _bench_exe(server_exe: Path) -> Path:
    """
    llama-bench.exe sits beside llama-server.exe in the same build output dir.

    brains.yaml only names the server binary — it is the launch config, not a
    tools index — so the bench binary is derived from it. Deriving rather than
    adding a second YAML key keeps both tools pinned to the same build, which
    is the whole point of comparing numbers across runs.
    """
    return server_exe.with_name("llama-bench" + server_exe.suffix)


def _port_is_listening(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def _run_path(label: str) -> Path:
    return _RESULTS / f"{label}.run.json"


def _do_compare(baseline_label: str, candidate_label: str, tolerance: float) -> int:
    try:
        baseline = load_run(_run_path(baseline_label))
        candidate = load_run(_run_path(candidate_label))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    comparison = compare_runs(baseline, candidate, tolerance_pct=tolerance)
    print(comparison.render())
    return 0 if comparison.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Layer 2 performance measurement via llama-bench."
    )
    parser.add_argument("--brain", choices=["fast", "architect"])
    parser.add_argument("--model", help="GGUF to measure (default: brains.yaml incumbent)")
    parser.add_argument("--label", help="name for this run")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--tolerance", type=float, default=10.0,
                        help="percent regression allowed before failing (default: 10)")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--depths", default="0,8192",
                        help="context depths to measure (default: 0,8192)")
    parser.add_argument("--force", action="store_true",
                        help="benchmark even if the brain's port is listening (unsafe)")
    args = parser.parse_args(argv)

    if args.compare:
        return _do_compare(*args.compare, tolerance=args.tolerance)

    if not args.brain or not args.label:
        parser.error("--brain and --label are required unless --compare is used")

    servers = ManagedServers.from_yaml()
    cfg = servers.fast if args.brain == "fast" else servers.architect

    if _port_is_listening(cfg.port) and not args.force:
        print(
            f"error: {args.brain} server is listening on {cfg.port}. llama-bench "
            f"loads its own copy of the model — running both would double-allocate "
            f"VRAM. Stop the server first, or pass --force if you know the model "
            f"fits twice.",
            file=sys.stderr,
        )
        return 2

    bench_exe = _bench_exe(cfg.exe)
    if not bench_exe.exists():
        print(f"error: llama-bench not found at {bench_exe}", file=sys.stderr)
        return 2

    model = args.model or str(cfg.model)
    argv_bench = build_llama_bench_argv(
        bench_exe, model,
        device=_DEVICE[args.brain],
        batch_size=int(cfg.server.get("batch_size", 2048)),
        ubatch_size=int(cfg.server.get("ubatch_size", 512)),
        threads=int(cfg.server.get("threads", 16)),
        cache_type_k=str(cfg.server.get("cache_type_k", "q8_0")),
        cache_type_v=str(cfg.server.get("cache_type_v", "q8_0")),
        flash_attn=bool(cfg.server.get("flash_attn", True)),
        depths=[int(d) for d in args.depths.split(",") if d.strip()],
        repetitions=args.repetitions,
    )

    print(f"running llama-bench on {model}")
    print(f"  {' '.join(argv_bench)}\n")
    completed = subprocess.run(
        argv_bench, capture_output=True, text=True,
        encoding="utf-8", errors="replace", cwd=str(_ROOT),
    )
    if completed.returncode != 0:
        print(completed.stdout[-2000:], file=sys.stderr)
        print(f"error: llama-bench exited {completed.returncode}", file=sys.stderr)
        return completed.returncode

    try:
        run = parse_llama_bench_json(completed.stdout, label=args.label)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    out = save_run(run, _run_path(args.label))
    print(run.render())
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
