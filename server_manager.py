\
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# NOTE:
# - llama-server may not bind the listening port until after model load on some builds.
# - DeepSeek V3.x can take several minutes to load.
# Therefore we use generous, configurable wait timeouts and (optionally) tail logs on failure.


@dataclass(frozen=True)
class ManagedServers:
    q5_port: int = 8011
    q6_port: int = 8012
    start_q5_bat: Path = Path("start_q5_server.bat")
    start_q6_bat: Path = Path("start_q6_server.bat")
    # Optional log locations (used only for diagnostics on timeout)
    q5_log: Optional[Path] = Path("logs/q5_server.log")
    q6_log: Optional[Path] = Path("logs/q6_server.log")
    # How long to wait for first bind after starting (seconds)
    q5_start_timeout_s: float = 900.0  # 15 minutes
    q6_start_timeout_s: float = 1200.0  # 20 minutes


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, shell=False)


def is_port_listening(port: int) -> bool:
    """
    Returns True if something is LISTENING on the port.
    Uses: netstat -ano
    """
    proc = _run(["cmd.exe", "/c", "netstat -ano"])
    if proc.returncode != 0:
        return False
    needle = f":{port} "
    for line in proc.stdout.splitlines():
        # Handles 0.0.0.0:PORT and [::]:PORT
        if needle in line and "LISTENING" in line:
            return True
    return False


def find_pid_by_port(port: int) -> Optional[int]:
    """
    Returns PID of process LISTENING on the port (first match), else None.
    """
    proc = _run(["cmd.exe", "/c", "netstat -ano"])
    if proc.returncode != 0:
        return None
    needle = f":{port} "
    for line in proc.stdout.splitlines():
        if needle in line and "LISTENING" in line:
            parts = line.split()
            # Typical: TCP  0.0.0.0:8011  0.0.0.0:0  LISTENING  12345
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


def _tail(path: Path, n_lines: int = 60) -> str:
    try:
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = lines[-n_lines:]
        return "\n".join(tail)
    except Exception:
        return ""


def wait_for_port(port: int, timeout_s: float) -> bool:
    """
    Waits until netstat reports LISTENING on the port.
    """
    end = time.time() + timeout_s
    while time.time() < end:
        if is_port_listening(port):
            return True
        time.sleep(0.5)
    return False


def ensure_running(servers: ManagedServers) -> Tuple[bool, str]:
    """
    Ensures both Q5 and Q6 ports are listening by launching bat files if needed.
    Returns (ok, message).
    """
    msgs: list[str] = []

    if not is_port_listening(servers.q5_port):
        ok, msg = start_server_from_bat(servers.start_q5_bat)
        msgs.append(f"Q5: {msg}")
        if ok and not wait_for_port(servers.q5_port, servers.q5_start_timeout_s):
            detail = "Q5 failed to start (timeout waiting for LISTENING). " + " ".join(msgs)
            if servers.q5_log:
                tail = _tail(servers.q5_log)
                if tail:
                    detail += "\n\n--- q5 log tail ---\n" + tail
            return False, detail

    if not is_port_listening(servers.q6_port):
        ok, msg = start_server_from_bat(servers.start_q6_bat)
        msgs.append(f"Q6: {msg}")
        if ok and not wait_for_port(servers.q6_port, servers.q6_start_timeout_s):
            detail = "Q6 failed to start (timeout waiting for LISTENING). " + " ".join(msgs)
            if servers.q6_log:
                tail = _tail(servers.q6_log)
                if tail:
                    detail += "\n\n--- q6 log tail ---\n" + tail
            return False, detail

    return True, " | ".join(msgs) if msgs else "Already running"
