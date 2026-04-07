"""
server_manager.py

Manages the lifecycle of all three llama-server instances.
Configuration is read from config/brains/brains.yaml — no .bat parsing.

Public API (unchanged from the bat-based version):
    ManagedServers          – holds loaded BrainConfig for all three servers
    ManagedServers.from_yaml(path) – primary constructor
    ensure_embed_running(servers)  – start embed server if not running
    ensure_q5_running(servers)     – start fast brain (and embed) if not running
    ensure_q6_running(servers)     – start architect brain if not running
    find_pid_by_port(port)         – find PID listening on a port
    stop_server_on_port(port)      – kill server on a port
"""
from __future__ import annotations

import atexit
import functools
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml

from openai_client import HttpTimeouts, health_check


# ---------------------------
# Project paths (pinned)
# ---------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
_LOGS_DIR = _PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

_BRAINS_YAML = _PROJECT_ROOT / "config" / "brains" / "brains.yaml"

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

# ──────────────────────────────────────────────────────────────────────────── #
# Spawned-process registry + atexit cleanup                                     #
# ──────────────────────────────────────────────────────────────────────────── #
# llama-server.exe is launched with CREATE_NEW_PROCESS_GROUP so the hidden
# console window is never shown and Ctrl+C from the parent console is not
# forwarded to the child (which is in its own process group).  As a result the
# servers must be explicitly terminated when the Streamlit process exits.
#
# Strategy: track every Popen object returned by start_server_from_config() in
# _spawned_procs and register a single atexit handler to terminate them all.
# proc.poll() != None means the server already stopped (normal stop or restart),
# so we skip it safely.

_spawned_procs: list[subprocess.Popen] = []


def _kill_spawned_servers() -> None:
    """atexit handler — terminate all llama-server processes started this session."""
    for proc in _spawned_procs:
        try:
            if proc.poll() is None:   # still running
                proc.terminate()
        except Exception:
            pass


atexit.register(_kill_spawned_servers)


# ──────────────────────────────────────────────────────────────────────────── #
# Flag-type tables for _build_argv()                                            #
# ──────────────────────────────────────────────────────────────────────────── #

# Boolean flags that take an explicit "on"/"off" value (present even when False).
# All other boolean YAML keys are treated as presence-only (--flag when True, omitted when False).
_BOOL_ONOFF: frozenset = frozenset({"flash-attn", "log-colors", "fit"})


# ──────────────────────────────────────────────────────────────────────────── #
# BrainConfig                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class BrainConfig:
    """
    Configuration for a single llama-server instance, loaded from brains.yaml.

    The `server` dict holds every key under the server: section of the YAML.
    _build_argv() converts it to a list of CLI arguments.
    """
    name: str                       # "fast", "architect", or "embed"
    exe: Path                       # absolute path to llama-server.exe
    model: Path                     # absolute path to GGUF file
    log: Path                       # absolute path to log file
    startup_timeout_s: float        # max seconds to wait for /health readiness
    server: Dict[str, Any]          # server: section from YAML (host, port, flags…)

    @property
    def host(self) -> str:
        return str(self.server.get("host", "127.0.0.1"))

    @property
    def port(self) -> int:
        return int(self.server.get("port", 8080))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


# ──────────────────────────────────────────────────────────────────────────── #
# YAML loading                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

# Default timeouts used when brains.yaml does not specify startup_timeout_s
_DEFAULT_TIMEOUTS: Dict[str, float] = {
    "fast": 1800.0,
    "architect": 2700.0,
    "embed": 300.0,
}


def _load_brain_config(yaml_path: Path, name: str) -> BrainConfig:
    """
    Parse one brain entry from brains.yaml into a BrainConfig.

    Raises KeyError if the named brain is missing from the file.
    """
    data: Dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    raw = data[name]
    return BrainConfig(
        name=name,
        exe=Path(raw["exe"]),
        model=Path(raw["model"]),
        log=Path(raw["log"]),
        startup_timeout_s=float(
            raw.get("startup_timeout_s", _DEFAULT_TIMEOUTS.get(name, 600.0))
        ),
        server=dict(raw.get("server", {})),
    )


# ──────────────────────────────────────────────────────────────────────────── #
# ManagedServers                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class ManagedServers:
    """
    Holds loaded BrainConfig for all llama-server instances.

    Ports, log paths, and startup timeouts are all sourced from brains.yaml.
    brains.yaml is the single authoritative config source.

    Construct via the class method:
        servers = ManagedServers.from_yaml()           # uses default path
        servers = ManagedServers.from_yaml(my_path)    # custom path

    Optional fields (None when not configured in brains.yaml):
        summarizer — lightweight CPU brain for search summarization (port 8013).
                     Activate by uncommenting the summarizer: section in brains.yaml.
    """
    embed: BrainConfig
    fast: BrainConfig
    architect: BrainConfig
    summarizer: Optional[BrainConfig] = field(default=None)

    @classmethod
    def from_yaml(cls, path: Path = _BRAINS_YAML) -> "ManagedServers":
        """Load all three brain configs from YAML; result is cached for the process lifetime."""
        return _load_managed_servers(path)

    # ── Convenience properties (keep call sites in ensure_* functions clean) ──

    @property
    def host(self) -> str:
        return self.fast.host

    @property
    def embed_port(self) -> int:
        return self.embed.port

    @property
    def q5_port(self) -> int:
        return self.fast.port

    @property
    def q6_port(self) -> int:
        return self.architect.port

    @property
    def embed_log(self) -> Path:
        return self.embed.log

    @property
    def q5_log(self) -> Path:
        return self.fast.log

    @property
    def q6_log(self) -> Path:
        return self.architect.log

    @property
    def embed_start_timeout_s(self) -> float:
        return self.embed.startup_timeout_s

    @property
    def q5_start_timeout_s(self) -> float:
        return self.fast.startup_timeout_s

    @property
    def q6_start_timeout_s(self) -> float:
        return self.architect.startup_timeout_s


@functools.lru_cache(maxsize=None)
def _load_managed_servers(path: Path) -> ManagedServers:
    """Parse brains.yaml once and cache the result for the process lifetime.

    Called exclusively by ManagedServers.from_yaml().  Using lru_cache here
    means repeated calls (e.g. on every Streamlit rerun) skip the file read
    and YAML parse entirely after the first call.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"brains.yaml not found: {path}\n"
            "Expected at config/brains/brains.yaml relative to project root."
        )
    # Try to load the optional summarizer brain (commented out by default).
    # KeyError means the section is absent/commented — not an error.
    _summarizer: Optional[BrainConfig] = None
    try:
        _summarizer = _load_brain_config(path, "summarizer")
    except KeyError:
        pass
    return ManagedServers(
        embed=_load_brain_config(path, "embed"),
        fast=_load_brain_config(path, "fast"),
        architect=_load_brain_config(path, "architect"),
        summarizer=_summarizer,
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Windows process helpers                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

def _run(cmd: list) -> subprocess.CompletedProcess:
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
    """Find the PID that is LISTENING on 127.0.0.1:<port> or 0.0.0.0:<port>."""
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
    """Kill server on given port. Returns True if nothing running or kill succeeded."""
    pid = find_pid_by_port(port)
    if pid is None:
        return True
    cp = _run(["taskkill", "/PID", str(pid), "/F"])
    return cp.returncode == 0


# ──────────────────────────────────────────────────────────────────────────── #
# HTTP readiness                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

def _http_ready(base_url: str, timeout_s: float = 1.0) -> Tuple[bool, str]:
    """Probe /health, /v1/health, /v1/models, /props — delegates to openai_client.health_check()."""
    return health_check(base_url, timeouts=HttpTimeouts(connect_s=timeout_s, read_s=timeout_s))


# ──────────────────────────────────────────────────────────────────────────── #
# Log tail + fatal error detection                                               #
# ──────────────────────────────────────────────────────────────────────────── #

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


def _wait_for_ready(
    host: str,
    port: int,
    base_url: str,
    timeout_s: float,
    log_path: Path,
) -> Tuple[bool, str]:
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


# ──────────────────────────────────────────────────────────────────────────── #
# Command building from YAML config                                             #
# ──────────────────────────────────────────────────────────────────────────── #

def _build_argv(brain: BrainConfig) -> list:
    """
    Build the full subprocess argument list from a BrainConfig.

    Argument order:
      <exe>  --model <path>  <server flags in YAML order>  --log-file <path>

    Boolean flag handling:
      - Keys in _BOOL_ONOFF    → --flag on  /  --flag off
      - All other bool keys    → --flag (only when True; omitted when False)
        This includes negation flags such as --no-kv-unified and
        --no-cont-batching: set the YAML key to true to emit the flag,
        false (or omit) to leave the positive default in effect.
      - All other keys         → --flag <value>

    --log-file is always appended last (project invariant: never rely on
    stdout/stderr redirection for long-running servers).
    """
    argv: list = [str(brain.exe), "--model", str(brain.model)]

    for yaml_key, value in brain.server.items():
        cli_name = yaml_key.replace("_", "-")
        flag = f"--{cli_name}"

        if isinstance(value, bool):
            if cli_name in _BOOL_ONOFF:
                # e.g. flash_attn: true  → --flash-attn on
                #      log_colors: false → --log-colors off
                argv.extend([flag, "on" if value else "off"])
            else:
                # Presence-only (known or unknown): include only when True
                if value:
                    argv.append(flag)
        else:
            # Regular key=value: --ctx-size 4096, --alias "Qwen…", etc.
            argv.extend([flag, str(value)])

    # Project invariant: always --log-file; the server writes its own logs
    argv.extend(["--log-file", str(brain.log)])
    return argv


# ──────────────────────────────────────────────────────────────────────────── #
# Server launch                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

def start_server_from_config(brain: BrainConfig) -> Tuple[bool, str]:
    """
    Spawn a llama-server process using config loaded from brains.yaml.

    Steps:
      1. Validate exe and model paths exist.
      2. Build the argv from BrainConfig via _build_argv().
      3. Write a startup header to the log file.
      4. Spawn the process (no cmd.exe, no shell=True, no stdout redirect —
         the server writes its own log via --log-file per project invariant).

    Returns (True, "spawned") on success, (False, reason) on failure.
    """
    if not brain.exe.exists():
        return False, f"EXE not found: {brain.exe}"
    if not brain.model.exists():
        return False, f"MODEL not found: {brain.model}"

    argv = _build_argv(brain)

    brain.log.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%a %m/%d/%Y %H:%M:%S", time.localtime())
    header = (
        f"\n==== {brain.name.upper()} START (yaml) {ts} ====\n"
        f"EXE={brain.exe}\n"
        f"MODEL={brain.model}\n"
    ).encode("utf-8", errors="ignore")

    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )

        # Open the log once: write the startup header, then keep it open so
        # the child inherits the fd for stderr.  Early CUDA initialisation
        # messages (ggml_cuda_init, cudaMalloc failures) are written to stderr
        # before llama-server's own --log-file logger is ready; capturing them
        # here makes startup failures diagnosable without --verbose flags.
        log_fh = open(brain.log, "ab", buffering=0)
        log_fh.write(header)
        log_fh.flush()
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(_PROJECT_ROOT),
                stdout=subprocess.DEVNULL,  # server writes chat via --log-file
                stderr=log_fh,              # capture early CUDA / arg errors
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=False,
            )
        finally:
            log_fh.close()  # parent closes its copy; child keeps its own fd

        _spawned_procs.append(proc)
    except Exception as e:
        return False, f"Failed to spawn llama-server: {e}"

    return True, "spawned"


# ──────────────────────────────────────────────────────────────────────────── #
# Public API: ensure_* functions (called by InferenceSession)                   #
# ──────────────────────────────────────────────────────────────────────────── #

def ensure_embed_running(servers: ManagedServers) -> Tuple[bool, str]:
    base_url = servers.embed.base_url

    if find_pid_by_port(servers.embed_port) is not None:
        ok, how = _http_ready(base_url, timeout_s=1.0)
        if ok:
            return True, f"EMBED already ready ({how})"

    stop_server_on_port(servers.embed_port)

    ok, msg = start_server_from_config(servers.embed)
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
    # Embedding server must be ready before the chat brain starts
    ok, msg = ensure_embed_running(servers)
    if not ok:
        return False, f"Embeddings not ready: {msg}"

    base_url = servers.fast.base_url

    if find_pid_by_port(servers.q5_port) is not None:
        ok, how = _http_ready(base_url, timeout_s=1.0)
        if ok:
            return True, f"Q5 already ready ({how})"

    stop_server_on_port(servers.q5_port)

    ok, msg = start_server_from_config(servers.fast)
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
    base_url = servers.architect.base_url

    if find_pid_by_port(servers.q6_port) is not None:
        ok, how = _http_ready(base_url, timeout_s=1.0)
        if ok:
            return True, f"Q6 already ready ({how})"

    stop_server_on_port(servers.q6_port)

    ok, msg = start_server_from_config(servers.architect)
    if not ok:
        return False, f"Q6 start failed: {msg}"

    return _wait_for_ready(
        host=servers.host,
        port=servers.q6_port,
        base_url=base_url,
        timeout_s=servers.q6_start_timeout_s,
        log_path=servers.q6_log,
    )


def ensure_summarizer_running(servers: ManagedServers) -> Tuple[bool, str]:
    """
    Start the optional CPU summarizer brain (port 8013) if configured and not running.

    Returns (False, reason) immediately when no summarizer: section exists in
    brains.yaml — the caller must treat this as "not available, fall back to
    FAST brain" rather than an error.
    """
    if servers.summarizer is None:
        return False, "summarizer: section not configured in brains.yaml"

    s = servers.summarizer
    base_url = s.base_url

    if find_pid_by_port(s.port) is not None:
        ok, how = _http_ready(base_url, timeout_s=1.0)
        if ok:
            return True, f"Summarizer already ready ({how})"

    stop_server_on_port(s.port)

    ok, msg = start_server_from_config(s)
    if not ok:
        return False, f"Summarizer start failed: {msg}"

    return _wait_for_ready(
        host=s.host,
        port=s.port,
        base_url=base_url,
        timeout_s=s.startup_timeout_s,
        log_path=s.log,
    )
