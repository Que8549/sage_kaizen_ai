from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


# ---------------------------
# Project paths (pinned)
# ---------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
_LOGS_DIR = _PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

_FATAL_MARKERS = (
    "error while handling argument",
    "failed to load model",
    "error loading model",
    "exiting due to model loading error",
    "cudaMalloc failed",
    "unable to allocate CUDA0 buffer",
    "failed to allocate CUDA0 buffer",
    "ggml_backend_cuda_buffer_type_alloc_buffer",
    "The filename, directory name, or volume label syntax is incorrect",
    "EXE not found",
    "MODEL not found",
)


@dataclass(frozen=True)
class ManagedServers:
    host: str = "127.0.0.1"

    # Embedding server (aligned with BAT)
    embed_port: int = 8020
    start_embed_bat: Path = _PROJECT_ROOT / "start_embedding_point.bat"
    embed_log: Path = _PROJECT_ROOT / "logs" / "embed_server.log"
    embed_start_timeout_s: float = 1800.0  # 30 minutes

    # Chat brains
    q5_port: int = 8011
    q6_port: int = 8012

    start_q5_bat: Path = _PROJECT_ROOT / "start_q5_server.bat"
    start_q6_bat: Path = _PROJECT_ROOT / "start_q6_server.bat"

    q5_log: Path = _LOGS_DIR / "q5_server.log"
    q6_log: Path = _LOGS_DIR / "q6_server.log"

    q5_start_timeout_s: float = 1800.0  # 30 min
    q6_start_timeout_s: float = 2700.0  # 45 min


# ---------------------------
# Windows helpers
# ---------------------------

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
        cwd=str(_PROJECT_ROOT),
    )


def find_pid_by_port(port: int) -> Optional[int]:
    """
    Finds PID that is LISTENING on 127.0.0.1:<port> or 0.0.0.0:<port>.
    """
    cp = _run(["cmd.exe", "/c", "netstat -ano -p tcp"])
    if cp.returncode != 0:
        return None

    patt = re.compile(rf"^\s*TCP\s+(\S+):{port}\s+\S+\s+LISTENING\s+(\d+)\s*$", re.I)
    for line in cp.stdout.splitlines():
        m = patt.match(line)
        if m:
            return int(m.group(2))
    return None


def stop_server_on_port(port: int) -> bool:
    """
    Stops server listening on given port. Returns True if nothing is running or kill succeeded.
    """
    pid = find_pid_by_port(port)
    if pid is None:
        return True
    cp = _run(["taskkill", "/PID", str(pid), "/F"])
    return cp.returncode == 0


# ---------------------------
# HTTP readiness
# ---------------------------

def _http_get_status(url: str, timeout_s: float) -> int:
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return int(getattr(resp, "status", 0) or 0)
    except HTTPError as e:
        return int(getattr(e, "code", 0) or 0)
    except URLError:
        return 0
    except Exception:
        return 0


def _http_ready(base_url: str, timeout_s: float = 1.0) -> Tuple[bool, str]:
    """
    Probes /health then /v1/health first, then falls back to /v1/models and /props.
    """
    endpoints = ["/health", "/v1/health", "/v1/models", "/props"]
    for ep in endpoints:
        st = _http_get_status(base_url + ep, timeout_s=timeout_s)
        if st == 200:
            return True, f"ready via {ep}"
    return False, "not ready"


# ---------------------------
# Log tail + ready wait
# ---------------------------

def _tail(path: Optional[Path], n_lines: int = 160) -> str:
    try:
        if not path or not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return ""


def _log_has_fatal_error(tail_text: str) -> Optional[str]:
    low = tail_text.lower()
    for marker in _FATAL_MARKERS:
        if marker.lower() in low:
            return marker
    return None


def _wait_for_ready(host: str, port: int, base_url: str, timeout_s: float, log_path: Path) -> Tuple[bool, str]:
    start = time.monotonic()
    last_tail = ""

    while (time.monotonic() - start) < timeout_s:
        ok, how = _http_ready(base_url, timeout_s=1.0)
        if ok:
            return True, f"ready ({how})"

        tail_text = _tail(log_path)
        if tail_text and tail_text != last_tail:
            last_tail = tail_text
            fatal = _log_has_fatal_error(tail_text)
            if fatal:
                return False, f"fatal in log: {fatal}\n--- tail ---\n{tail_text}"

        time.sleep(0.15)

    return False, f"timeout waiting for {host}:{port}\n--- tail ---\n{_tail(log_path)}"


# ---------------------------
# BAT parsing + direct spawn
# ---------------------------

_SET_QUOTED = re.compile(r'^\s*set\s+"([^=]+)=(.*)"\s*$', re.I)
_SET_PLAIN  = re.compile(r'^\s*set\s+([^=]+)=(.*)\s*$', re.I)
_VAR_REF    = re.compile(r"%([^%]+)%")


def _parse_set_vars(bat_text: str) -> Dict[str, str]:
    vars: Dict[str, str] = {}
    for raw in bat_text.splitlines():
        line = raw.strip()
        m = _SET_QUOTED.match(line)
        if m:
            vars[m.group(1).strip()] = m.group(2)
            continue
        m = _SET_PLAIN.match(line)
        if m:
            k = m.group(1).strip().strip('"')
            v = m.group(2).strip().strip('"')
            vars[k] = v
    return vars


def _expand_vars(s: str, vars: Dict[str, str]) -> str:
    for _ in range(10):
        new = _VAR_REF.sub(lambda m: vars.get(m.group(1), m.group(0)), s)
        if new == s:
            return s
        s = new
    return s


def _extract_llama_command_lines(bat_text: str) -> list[str]:
    lines = bat_text.splitlines()
    cmd_lines: list[str] = []
    started = False
    for raw in lines:
        stripped = raw.strip()
        if not started and stripped.startswith('"%EXE%"'):
            started = True
        if started:
            cmd_lines.append(raw.rstrip())
            if not raw.rstrip().endswith("^"):
                break
    return cmd_lines


def _build_llama_tokens_from_bat(bat_path: Path) -> Tuple[list[str], Dict[str, str]]:
    bat_text = bat_path.read_text(encoding="utf-8", errors="ignore")

    vars = _parse_set_vars(bat_text)
    vars = {k: _expand_vars(v, vars) for k, v in vars.items()}

    cmd_lines = _extract_llama_command_lines(bat_text)
    if not cmd_lines:
        raise RuntimeError('Could not find llama-server command block (line starting with "%EXE%")')

    cmd = " ".join([ln.rstrip().rstrip("^").strip() for ln in cmd_lines])
    cmd = _expand_vars(cmd, vars)

    cmd = re.split(r"\s1>>", cmd, maxsplit=1)[0].strip()

    tokens = shlex.split(cmd, posix=False)

    def unquote(t: str) -> str:
        t = t.strip()
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
            return t[1:-1]
        return t

    tokens = [unquote(t) for t in tokens]
    return tokens, vars


def start_server_from_bat(bat_path: Path, log_path: Path) -> Tuple[bool, str]:
    if not bat_path.exists():
        return False, f"BAT not found: {bat_path}"

    try:
        tokens, _vars = _build_llama_tokens_from_bat(bat_path)
    except Exception as e:
        return False, f"Failed to parse BAT: {e}"

    if not tokens:
        return False, "Empty command tokens parsed from BAT"

    exe = tokens[0]
    if not Path(exe).exists():
        return False, f"EXE not found (from bat): {exe}"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%a %m/%d/%Y %H:%M:%S", time.localtime())
    header = f"\n==== PY LAUNCH {bat_path.name} {ts} ====\n".encode("utf-8", errors="ignore")

    try:
        with open(log_path, "ab", buffering=0) as fh:
            fh.write(header)
            fh.flush()

            creationflags = 0
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

            subprocess.Popen(
                tokens,
                cwd=str(_PROJECT_ROOT),
                stdout=fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=False,
            )
    except Exception as e:
        return False, f"Failed to spawn llama-server: {e}"

    return True, "spawned"


# ---------------------------
# Public API used by Streamlit
# ---------------------------

def ensure_embed_running(servers: ManagedServers) -> Tuple[bool, str]:
    base_url = f"http://{servers.host}:{servers.embed_port}"

    pid = find_pid_by_port(servers.embed_port)
    if pid is not None:
        ok, how = _http_ready(base_url, timeout_s=1.0)
        if ok:
            return True, f"EMBED already ready ({how})"

    stop_server_on_port(servers.embed_port)

    ok, msg = start_server_from_bat(bat_path=servers.start_embed_bat, log_path=servers.embed_log)
    if not ok:
        return False, f"EMBED start failed: {msg}"

    return _wait_for_ready(
        host=servers.host,
        port=servers.embed_port,
        base_url=base_url,
        timeout_s=servers.embed_start_timeout_s,
        log_path=servers.embed_log,
    )


def ensure_q5_running(servers: ManagedServers) -> Tuple[bool, str]:
    # NEW: embedding must be ready first
    ok, msg = ensure_embed_running(servers)
    if not ok:
        return False, f"Embeddings not ready: {msg}"

    base_url = f"http://{servers.host}:{servers.q5_port}"

    pid = find_pid_by_port(servers.q5_port)
    if pid is not None:
        ok, how = _http_ready(base_url, timeout_s=1.0)
        if ok:
            return True, f"Q5 already ready ({how})"

    stop_server_on_port(servers.q5_port)

    ok, msg = start_server_from_bat(bat_path=servers.start_q5_bat, log_path=servers.q5_log)
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

    pid = find_pid_by_port(servers.q6_port)
    if pid is not None:
        ok, how = _http_ready(base_url, timeout_s=1.0)
        if ok:
            return True, f"Q6 already ready ({how})"

    stop_server_on_port(servers.q6_port)

    ok, msg = start_server_from_bat(bat_path=servers.start_q6_bat, log_path=servers.q6_log)
    if not ok:
        return False, f"Q6 start failed: {msg}"

    return _wait_for_ready(
        host=servers.host,
        port=servers.q6_port,
        base_url=base_url,
        timeout_s=servers.q6_start_timeout_s,
        log_path=servers.q6_log,
    )
