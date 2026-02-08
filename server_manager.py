from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------
# Production notes:
# - llama-server may not bind until after model load on some builds.
# - but in your case it often EXITs during load (CUDA OOM), so you must
#   fail-fast by reading logs when connection is refused.
# - readiness is best validated via HTTP (/health or /v1/models),
#   not only netstat LISTENING.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ManagedServers:
    host: str = "127.0.0.1"
    q5_port: int = 8011
    q6_port: int = 8012
    start_q5_bat: Path = Path("start_q5_server.bat")
    start_q6_bat: Path = Path("start_q6_server.bat")

    # Optional logs for diagnostics
    q5_log: Optional[Path] = Path("logs/q5_server.log")
    q6_log: Optional[Path] = Path("logs/q6_server.log")

    # Startup timeouts (DeepSeek can take minutes if it actually loads)
    q5_start_timeout_s: float = 1800.0  # 30 min (but we fail fast on fatal log errors)
    q6_start_timeout_s: float = 2700.0  # 45 min

    # HTTP probe behavior
    http_probe_timeout_s: float = 2.0
    poll_interval_s: float = 0.5


# ----------------------------- helpers --------------------------------


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, shell=False)


def is_port_listening(port: int) -> bool:
    """Returns True if something is LISTENING on the port (netstat -ano)."""
    proc = _run(["cmd.exe", "/c", "netstat -ano"])
    if proc.returncode != 0:
        return False
    needle = f":{port} "
    for line in proc.stdout.splitlines():
        if needle in line and "LISTENING" in line:
            return True
    return False


def find_pid_by_port(port: int) -> Optional[int]:
    """Returns PID of process LISTENING on the port (first match), else None."""
    proc = _run(["cmd.exe", "/c", "netstat -ano"])
    if proc.returncode != 0:
        return None
    needle = f":{port} "
    for line in proc.stdout.splitlines():
        if needle in line and "LISTENING" in line:
            parts = line.split()
            if len(parts) >= 5 and parts[-1].isdigit():
                return int(parts[-1])
    return None


def kill_by_pid(pid: int) -> bool:
    proc = _run(["cmd.exe", "/c", f"taskkill /PID {pid} /F"])
    return proc.returncode == 0


def stop_server_on_port(port: int) -> bool:
    pid = find_pid_by_port(port)
    if pid is None:
        return True
    return kill_by_pid(pid)


def start_server_from_bat(bat_path: Path) -> Tuple[bool, str]:
    """
    Starts a .bat in a separate process.
    The .bat should detach the server (e.g., `start "" /B ...`) so this returns quickly.
    """
    if not bat_path.exists():
        return False, f"Missing bat: {bat_path}"

    proc = _run(["cmd.exe", "/c", str(bat_path)])
    if proc.returncode != 0:
        return False, (proc.stderr.strip() or proc.stdout.strip() or f"bat exited {proc.returncode}")
    return True, "Started"


def _tail(path: Optional[Path], n_lines: int = 120) -> str:
    try:
        if not path or not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return ""


_FATAL_MARKERS = [
    # CLI / startup issues
    "error while handling argument",
    "unknown value for --flash-attn",
    "to show complete usage",

    # Model load failures
    "failed to load model",
    "error loading model",
    "llama_model_load: error loading model",
    "llama_model_load_from_file_impl: failed to load model",

    # CUDA OOM patterns you are seeing
    "cudaMalloc failed",
    "failed to allocate CUDA0 buffer",
    "unable to allocate CUDA0 buffer",
    "ggml_backend_cuda_buffer_type_alloc_buffer",
    "alloc_tensor_range: failed to allocate CUDA0 buffer",
    "out of memory",
]


def _log_has_fatal_error(log_path: Optional[Path]) -> Optional[str]:
    if not log_path or not log_path.exists():
        return None
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    for marker in _FATAL_MARKERS:
        if marker in txt:
            return marker
    return None


def _http_ready(host: str, port: int, timeout_s: float) -> Tuple[bool, str]:
    """
    Returns (ready, detail) by probing /health then /v1/models.
    Uses stdlib urllib so we don't depend on requests here.
    """
    base = f"http://{host}:{port}"

    # 1) /health (if supported)
    try:
        with urlopen(f"{base}/health", timeout=timeout_s) as resp:
            if 200 <= resp.status < 300:
                return True, "OK (/health)"
    except HTTPError as e:
        # /health exists but returns error -> treat as not ready, but include code
        return False, f"HTTP {e.code} (/health)"
    except URLError as e:
        return False, f"ConnectionError: {e.reason}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    # 2) /v1/models (OpenAI-compatible)
    try:
        with urlopen(f"{base}/v1/models", timeout=timeout_s) as resp:
            if 200 <= resp.status < 300:
                return True, "OK (/v1/models)"
            return False, f"HTTP {resp.status} (/v1/models)"
    except HTTPError as e:
        return False, f"HTTP {e.code} (/v1/models)"
    except URLError as e:
        return False, f"ConnectionError: {e.reason}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _wait_for_ready(
    *,
    host: str,
    port: int,
    timeout_s: float,
    log_path: Optional[Path],
    http_probe_timeout_s: float,
    poll_interval_s: float,
) -> Tuple[bool, str]:
    """
    Waits until HTTP readiness succeeds, OR fails fast if log shows fatal error.
    Returns (ok, detail).
    """
    end = time.time() + timeout_s
    last_detail = "Not ready yet"

    while time.time() < end:
        # Fail-fast if process already hit a fatal error
        fatal = _log_has_fatal_error(log_path)
        if fatal:
            tail = _tail(log_path)
            detail = (
                f"Fatal server error detected in log: '{fatal}'.\n\n"
                f"--- log tail ---\n{tail}"
            )
            return False, detail

        ready, detail = _http_ready(host, port, http_probe_timeout_s)
        last_detail = detail
        if ready:
            return True, detail

        # If connection refused and port isn't listening, it's probably crashed or still loading.
        # We keep waiting, but the fatal-log check above will cut this off quickly.
        time.sleep(poll_interval_s)

    # Timeout: include last HTTP status + log tail
    tail = _tail(log_path)
    detail = f"Timeout waiting for server readiness. Last probe: {last_detail}"
    if tail:
        detail += f"\n\n--- log tail ---\n{tail}"
    return False, detail


# ----------------------------- public API ------------------------------


def ensure_q5_running(servers: ManagedServers) -> Tuple[bool, str]:
    """Ensure Q5 is ready (HTTP-ready). Starts it if not ready."""
    # If already ready, return quickly
    ready, detail = _http_ready(servers.host, servers.q5_port, servers.http_probe_timeout_s)
    if ready:
        return True, "Q5 already ready"

    ok, msg = start_server_from_bat(servers.start_q5_bat)
    if not ok:
        return False, f"Q5 start failed: {msg}"

    ok2, detail2 = _wait_for_ready(
        host=servers.host,
        port=servers.q5_port,
        timeout_s=servers.q5_start_timeout_s,
        log_path=servers.q5_log,
        http_probe_timeout_s=servers.http_probe_timeout_s,
        poll_interval_s=servers.poll_interval_s,
    )
    if ok2:
        return True, "Q5 ready"
    return False, f"Q5 started but not ready:\n{detail2}"


def ensure_q6_running(servers: ManagedServers) -> Tuple[bool, str]:
    """Ensure Q6 is ready (HTTP-ready). Starts it if not ready."""
    ready, detail = _http_ready(servers.host, servers.q6_port, servers.http_probe_timeout_s)
    if ready:
        return True, "Q6 already ready"

    ok, msg = start_server_from_bat(servers.start_q6_bat)
    if not ok:
        return False, f"Q6 start failed: {msg}"

    ok2, detail2 = _wait_for_ready(
        host=servers.host,
        port=servers.q6_port,
        timeout_s=servers.q6_start_timeout_s,
        log_path=servers.q6_log,
        http_probe_timeout_s=servers.http_probe_timeout_s,
        poll_interval_s=servers.poll_interval_s,
    )
    if ok2:
        return True, "Q6 ready"
    return False, f"Q6 started but not ready:\n{detail2}"


def ensure_running(servers: ManagedServers) -> Tuple[bool, str]:
    """
    Backwards-compatible: ensures BOTH are ready.
    (Your updated UI should usually call ensure_q5_running first and only ensure_q6_running on escalation.)
    """
    ok5, msg5 = ensure_q5_running(servers)
    if not ok5:
        return False, msg5
    ok6, msg6 = ensure_q6_running(servers)
    if not ok6:
        return False, msg6
    return True, "Q5+Q6 ready"
