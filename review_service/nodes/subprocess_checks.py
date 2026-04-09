"""
review_service/nodes/subprocess_checks.py — Static analysis node (no LLM).

Runs pyright, ruff, and pytest --collect-only against changed Python files
in parallel via asyncio.gather + asyncio.create_subprocess_exec.

Results are trimmed and fed to architect_reviewer as context.
If any tool is not installed or times out, the node stores a
"not installed / timed out" message and continues — these are optional checks.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from sk_logging import get_logger
from ..state import ReviewState

_LOG = get_logger("sage_kaizen.review_service.subprocess_checks")

_MAIN_ROOT  = Path("F:/Projects/sage_kaizen_ai")
_VOICE_ROOT = Path("F:/Projects/sage_kaizen_ai_voice")
_TIMEOUT    = 60      # seconds per subprocess
_MAX_OUT    = 3_000   # max chars per tool output (keeps ARCHITECT prompt lean)
_MAX_QUALITY_OUT = 4_000  # max chars for vulture / extended-ruff outputs

# Directories inside project roots that are external repos — excluded from
# vulture and extended-ruff scans to avoid noise from vendored code.
_EXCLUDE_DIRS = "llama.cpp,flash-attention,__pycache__,.venv,node_modules"


async def subprocess_checks_node(state: ReviewState) -> dict:
    changed_py = [
        f for f in state.get("changed_files", [])
        if f.endswith(".py") and not f.startswith("[voice]")
    ]
    full_paths = [str(_MAIN_ROOT / f) for f in changed_py if (_MAIN_ROOT / f).exists()]

    _LOG.info("review.subprocess_checks | py_files=%d", len(full_paths))

    # Run pyright and ruff in parallel; pytest after (depends on file list)
    pyright_task = _run("pyright",  ["pyright"] + (full_paths or ["."]), _TIMEOUT)
    ruff_task    = _run("ruff",     ["ruff", "check", "--output-format=concise"] + (full_paths or ["."]), _TIMEOUT)

    pyright_raw, ruff_raw = await asyncio.gather(pyright_task, ruff_task)

    # pytest --collect-only: discover tests affected by changed modules
    pytest_raw = await _run(
        "pytest",
        ["python", "-m", "pytest", "--collect-only", "-q", "--tb=no"]
        + (full_paths or ["."]),
        _TIMEOUT,
    )

    # ── Full-mode only: whole-tree dead-code + extended quality rules ──
    vulture_raw      = ""
    ruff_quality_raw = ""
    if state["mode"] == "full":
        roots = [p for p in [str(_MAIN_ROOT), str(_VOICE_ROOT)] if Path(p).exists()]
        vulture_task = _run(
            "vulture",
            ["vulture"] + roots + ["--min-confidence", "80", "--exclude", _EXCLUDE_DIRS],
            _TIMEOUT,
        )
        ruff_quality_task = _run(
            "ruff-quality",
            [
                "ruff", "check",
                "--select=C90,B,SIM,UP,PERF,RUF,PIE",
                "--output-format=concise",
                f"--exclude={_EXCLUDE_DIRS}",
            ] + roots,
            _TIMEOUT,
        )
        vulture_raw, ruff_quality_raw = await asyncio.gather(vulture_task, ruff_quality_task)
        _LOG.info(
            "review.subprocess_checks | vulture_chars=%d ruff_quality_chars=%d",
            len(vulture_raw), len(ruff_quality_raw),
        )

    return {
        "pyright_output":    _trim(pyright_raw,      _MAX_OUT),
        "ruff_output":       _trim(ruff_raw,         _MAX_OUT),
        "pytest_collect":    _trim(pytest_raw,       _MAX_OUT),
        "vulture_output":    _trim(vulture_raw,      _MAX_QUALITY_OUT),
        "ruff_quality_output": _trim(ruff_quality_raw, _MAX_QUALITY_OUT),
    }


async def _run(tool: str, cmd: list[str], timeout_s: int) -> str:
    """Run a subprocess and return combined stdout+stderr, trimmed."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(_MAIN_ROOT),
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            return stdout.decode("utf-8", errors="replace")
        except asyncio.TimeoutError:
            proc.kill()
            return f"[{tool}: timed out after {timeout_s}s]"
    except FileNotFoundError:
        return f"[{tool}: not installed or not on PATH]"
    except Exception as exc:
        _LOG.warning("subprocess_checks.%s failed: %s", tool, exc)
        return f"[{tool}: error — {exc}]"


def _trim(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [{len(text) - max_chars} chars truncated]"
