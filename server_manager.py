from __future__ import annotations

import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import requests

# Project root = folder containing this file (works regardless of Streamlit CWD)
_PROJECT_ROOT = Path(__file__).resolve().parent
_LOGS_DIR = _PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class ManagedServers:
    host: str = "127.0.0.1"
    q5_port: int = 8011
    q6_port: int = 8012

    start_q5_bat: Path = _PROJECT_ROOT / "start_q5_server.bat"
    start_q6_bat: Path = _PROJECT_ROOT / "start_q6_server.bat"

    q5_log: Path = _LOGS_DIR / "q5_server.log"
    q6_log: Path = _LOGS_DIR / "q6_server.log"

    q5_start_timeout_s: float = 1800.0   # 30 min
    q6_start_timeout_s: float = 2700.0   # 45 min


_FATAL_MARKERS = (
    "error while handling argument",
    "failed to load model",
    "error loading model",
    "exiting due to model loading error",
    "cudaMalloc failed",
    "unable to allocate CUDA0 buffer",
    "failed to allocate CUDA0 buffer",
    "ggml_backend_cuda_buffer_type_alloc_buffer",
)


def _tail(path: Optional[Path], n_lines: int = 120) -> str:
    try:
        if not path or not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return ""


def _log_has_fatal_error(log_path: Optional[Path]) -> Optional[str]:
    if not log_path or not log_path.exists():
        return None
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    for m in _FATAL_MARKERS:
        if m in txt:
            return m
    return None


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, shell=False)


def find_pid_by_port(port: int) -> Optional[int]:
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


def stop_server_on_port(port: int) -> bool:
    pid = find_pid_by_port(port)
    if pid is None:
        return True
    proc = _run(["cmd.exe", "/c", f"taskkill /PID {pid} /F"])
    return proc.returncode == 0


def _creation_flags_no_window() -> int:
    flags = 0
    flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return flags


def start_server_from_bat(*, bat_path: Path) -> Tuple[bool, str]:
    """
    Start BAT asynchronously, detached, with no console window.
    Logging is handled inside the BAT (>> logs\q5_server.log / q6_server.log).
    """
    if not bat_path.exists():
        return False, f"Missing bat: {bat_path}"

    DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0)
    CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    try:
        subprocess.Popen(
            ["cmd.exe", "/c", str(bat_path)],
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
            close_fds=True,
        )
        return True, "Spawned"
    except Exception as e:
        return False, f"Failed to spawn {bat_path.name}: {type(e).__name__}: {e}"


def _tcp_connect_ok(host: str, port: int, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _http_ready(base_url: str, timeout_s: float = 2.0) -> Tuple[bool, str]:
    """
    Readiness probe (llama-server):

    Prefer:
      - /health
      - /v1/health
    Fallback:
      - /v1/models
      - /props

    Any 200 from health endpoints => READY.
    """
    base = base_url.rstrip("/")

    for path in ("/health", "/v1/health"):
        try:
            r = requests.get(f"{base}{path}", timeout=(timeout_s, timeout_s))
            if r.status_code == 200:
                return True, f"OK ({path})"
        except Exception:
            pass

    try:
        r = requests.get(f"{base}/v1/models", timeout=(timeout_s, timeout_s))
        if r.status_code == 200:
            return True, "OK (/v1/models)"
    except Exception:
        pass

    try:
        r = requests.get(f"{base}/props", timeout=(timeout_s, timeout_s))
        if r.status_code == 200:
            return True, "OK (/props)"
        return False, f"HTTP {r.status_code} (/props)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _wait_for_ready(
    *,
    host: str,
    port: int,
    base_url: str,
    timeout_s: float,
    log_path: Path,
    poll_s: float = 0.25,
) -> Tuple[bool, str]:
    end = time.time() + timeout_s
    last = "not ready"
    while time.time() < end:
        fatal = _log_has_fatal_error(log_path)
        if fatal:
            return False, f"Fatal error in log: '{fatal}'\n\n--- log tail ---\n{_tail(log_path)}"

        if _tcp_connect_ok(host, port, timeout_s=0.25):
            ok, detail = _http_ready(base_url, timeout_s=1.5)
            last = detail
            if ok:
                return True, detail

        time.sleep(poll_s)

    return False, f"Timeout waiting for readiness. Last: {last}\n\n--- log tail ---\n{_tail(log_path)}"


def ensure_q5_running(servers: ManagedServers) -> Tuple[bool, str]:
    base_url = f"http://{servers.host}:{servers.q5_port}"

    if find_pid_by_port(servers.q5_port) is not None:
        ok, _ = _http_ready(base_url, timeout_s=1.5)
        if ok:
            return True, "Q5 already ready"

    stop_server_on_port(servers.q5_port)

    ok, msg = start_server_from_bat(bat_path=servers.start_q5_bat)
    if not ok:
        return False, f"Q5 start failed: {msg}"

    return _wait_for_ready(
        host=servers.host,
        port=servers.q5_port,
        base_url=base_url,
        timeout_s=servers.q5_start_timeout_s,
        log_path=servers.q5_log,
    )


def ensure_q6_running(servers: ManagedServers) -> Tuple[bool, str]:
    base_url = f"http://{servers.host}:{servers.q6_port}"

    if find_pid_by_port(servers.q6_port) is not None:
        ok, _ = _http_ready(base_url, timeout_s=1.5)
        if ok:
            return True, "Q6 already ready"

    stop_server_on_port(servers.q6_port)

    ok, msg = start_server_from_bat(bat_path=servers.start_q5_bat)
    if not ok:
        return False, f"Q6 start failed: {msg}"

    return _wait_for_ready(
        host=servers.host,
        port=servers.q6_port,
        base_url=base_url,
        timeout_s=servers.q6_start_timeout_s,
        log_path=servers.q6_log,
    )


# Backwards-compatible helpers
def is_port_listening(port: int) -> bool:
    return find_pid_by_port(port) is not None




