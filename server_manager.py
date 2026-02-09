from __future__ import annotations

import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import requests


_PROJECT_ROOT = Path(__file__).resolve().parent  # bulletproof regardless of CWD


@dataclass(frozen=True)
class ManagedServers:
    host: str = "127.0.0.1"
    q5_port: int = 8011
    q6_port: int = 8012
    start_q5_bat: Path = _PROJECT_ROOT / "start_q5_server.bat"
    start_q6_bat: Path = _PROJECT_ROOT / "start_q6_server.bat"
    q5_log: Optional[Path] = _PROJECT_ROOT / "logs" / "q5_server.log"
    q6_log: Optional[Path] = _PROJECT_ROOT / "logs" / "q6_server.log"
    q5_start_timeout_s: float = 1800.0  # 30 min
    q6_start_timeout_s: float = 2700.0  # 45 min


_FATAL_MARKERS = (
    "error while handling argument",
    "unknown value for --flash-attn",
    "failed to load model",
    "error loading model",
    "exiting due to model loading error",
    "cudaMalloc failed",
    "unable to allocate CUDA0 buffer",
    "failed to allocate CUDA0 buffer",
    "ggml_backend_cuda_buffer_type_alloc_buffer",
)

_READY_LOG_MARKERS = (
    "main: model loaded",
    "main: server is listening on http://",
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
    try:
        txt = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    for m in _FATAL_MARKERS:
        if m in txt:
            return m
    return None


def _log_looks_ready(log_path: Optional[Path]) -> bool:
    if not log_path or not log_path.exists():
        return False
    try:
        txt = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return all(m in txt for m in _READY_LOG_MARKERS)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, shell=False)


def start_server_from_bat(bat_path: Path) -> Tuple[bool, str]:
    """
    Bulletproof: do NOT block Streamlit waiting for the .bat to exit.
    Always spawn the bat detached so this returns immediately.
    """
    if not bat_path.exists():
        return False, f"Missing bat: {bat_path}"

    DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)

    try:
        subprocess.Popen(
            ["cmd.exe", "/c", str(bat_path)],
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
        return True, "Started (detached)"
    except Exception as e:
        return False, f"Failed to start bat: {type(e).__name__}: {e}"


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


def _tcp_connect_ok(host: str, port: int, timeout_s: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _http_ready(base_url: str, timeout_s: float = 1.5) -> Tuple[bool, str]:
    base = base_url.rstrip("/")

    def _get(path: str) -> Tuple[bool, str]:
        try:
            r = requests.get(f"{base}{path}", timeout=(timeout_s, timeout_s))
            if r.status_code == 200:
                return True, f"OK ({path})"
            return False, f"{path}={r.status_code}"
        except Exception as e:
            return False, f"{path}={type(e).__name__}"

    for p in ("/health", "/v1/health", "/v1/models", "/props"):
        ok, d = _get(p)
        if ok:
            return True, d
    return False, "not ready"


def _wait_for_ready(
    *,
    host: str,
    port: int,
    base_url: str,
    timeout_s: float,
    log_path: Optional[Path],
    poll_s: float = 0.05,  # near-instant responsiveness
) -> Tuple[bool, str]:
    end = time.time() + timeout_s
    last = "not ready"

    while time.time() < end:
        fatal = _log_has_fatal_error(log_path)
        if fatal:
            return False, f"Fatal error in log: '{fatal}'\n\n--- log tail ---\n{_tail(log_path)}"

        # Fast success: TCP + HTTP
        if _tcp_connect_ok(host, port):
            ok, detail = _http_ready(base_url)
            last = detail
            if ok:
                return True, detail

        # Fallback success: log markers (your log proves ready)
        if _log_looks_ready(log_path):
            return True, "OK (log markers: model loaded + listening)"

        time.sleep(poll_s)

    if _log_looks_ready(log_path):
        return True, "OK (log markers: model loaded + listening)"

    return False, f"Timeout waiting for readiness. Last: {last}\n\n--- log tail ---\n{_tail(log_path)}"


def ensure_q5_running(servers: ManagedServers) -> Tuple[bool, str]:
    base_url = f"http://{servers.host}:{servers.q5_port}"

    # If already ready, return immediately (no delay)
    ok, detail = _http_ready(base_url, timeout_s=1.0)
    if ok:
        return True, f"Q5 ready: {detail}"

    # Avoid duplicate spawns
    if find_pid_by_port(servers.q5_port) is None:
        ok, msg = start_server_from_bat(servers.start_q5_bat)
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

    ok, detail = _http_ready(base_url, timeout_s=1.0)
    if ok:
        return True, f"Q6 ready: {detail}"

    if find_pid_by_port(servers.q6_port) is None:
        ok, msg = start_server_from_bat(servers.start_q6_bat)
        if not ok:
            return False, f"Q6 start failed: {msg}"

    return _wait_for_ready(
        host=servers.host,
        port=servers.q6_port,
        base_url=base_url,
        timeout_s=servers.q6_start_timeout_s,
        log_path=servers.q6_log,
    )


def is_port_listening(port: int) -> bool:
    return find_pid_by_port(port) is not None


def ensure_running(servers: ManagedServers) -> Tuple[bool, str]:
    ok5, msg5 = ensure_q5_running(servers)
    if not ok5:
        return False, msg5
    ok6, msg6 = ensure_q6_running(servers)
    if not ok6:
        return False, msg6
    return True, "Q5+Q6 ready"
