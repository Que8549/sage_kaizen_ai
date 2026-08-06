"""
scripts/run_model_gate.py — Layer 1 hard gates for a model candidate.

Run this BEFORE downloading 20 GB of anything you can already rule out, and
before benchmarking anything you did download.

    python scripts/run_model_gate.py --brain fast
    python scripts/run_model_gate.py --brain architect --json
    python scripts/run_model_gate.py --brain fast --model E:/candidate.gguf \
                                     --mmproj E:/candidate-mmproj.gguf
    python scripts/run_model_gate.py --brain fast --plan-co-tenants

With no --model, the incumbent from brains.yaml is checked — useful as a
regression check after a config change.

Exit code is 0 when every blocking gate passed, 1 otherwise, so this can gate a
larger script.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evals.gates import free_vram_gib, static_gates_architect, static_gates_fast  # noqa: E402
from server_manager import ManagedServers  # noqa: E402

_DEVICE_INDEX = {"fast": 1, "architect": 0}   # CUDA1 / CUDA0 per brains.yaml


def _incumbent_paths(brain: str) -> tuple[str, str | None]:
    """Resolve the model (and mmproj) currently configured in brains.yaml."""
    servers = ManagedServers.from_yaml()
    cfg = servers.fast if brain == "fast" else servers.architect
    return str(cfg.model), cfg.server.get("mmproj")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Layer 1 hard gates for a model candidate."
    )
    parser.add_argument("--brain", required=True, choices=["fast", "architect"])
    parser.add_argument("--model", help="candidate GGUF (default: brains.yaml incumbent)")
    parser.add_argument("--mmproj", help="candidate mmproj GGUF (FAST only)")
    parser.add_argument(
        "--plan-co-tenants", action="store_true",
        help="also reserve VRAM for the other services that share the GPU",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    model, mmproj = args.model, args.mmproj
    if model is None:
        model, incumbent_mmproj = _incumbent_paths(args.brain)
        mmproj = mmproj or incumbent_mmproj

    free = free_vram_gib(_DEVICE_INDEX[args.brain])

    if args.brain == "fast":
        if not mmproj:
            print("error: --mmproj is required for the FAST brain "
                  "(it must carry both audio and vision)", file=sys.stderr)
            return 2
        report = static_gates_fast(
            model, mmproj, free_gib=free, plan_co_tenants=args.plan_co_tenants
        )
    else:
        report = static_gates_architect(
            model, free_gib=free, plan_co_tenants=args.plan_co_tenants
        )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())
        if free is None:
            print("\nnote: NVML unavailable — VRAM gate skipped")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
