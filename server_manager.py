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
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

from openai_client import HttpTimeouts, health_check
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.server_manager")


# ---------------------------
# Project paths (pinned)
# ---------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
_LOGS_DIR = _PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

_BRAINS_YAML = _PROJECT_ROOT / "config" / "brains" / "brains.yaml"

_FATAL_MARKERS = (
    "error: invalid argument",       # unknown/removed CLI flag (fast-fail, no timeout)
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
_spawned_procs_lock = threading.Lock()


def _register_spawned(proc: subprocess.Popen) -> None:
    """
    Track a spawned server, dropping any that have already exited.

    Without the prune this list only ever grew: every restart cycle appended a
    new Popen and kept the dead one forever, holding its OS process handle
    open. A long Streamlit session that restarts brains repeatedly accumulated
    them for the life of the process.
    """
    with _spawned_procs_lock:
        _spawned_procs[:] = [p for p in _spawned_procs if p.poll() is None]
        _spawned_procs.append(proc)


def _kill_spawned_servers() -> None:
    """atexit handler — terminate all llama-server processes started this session."""
    with _spawned_procs_lock:
        procs = list(_spawned_procs)
    for proc in procs:
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
    server: dict[str, Any]          # server: section from YAML (host, port, flags…)

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
_DEFAULT_TIMEOUTS: dict[str, float] = {
    "fast": 1800.0,
    "architect": 2700.0,
    "embed": 300.0,
}


def _load_brain_config(yaml_path: Path, name: str) -> BrainConfig:
    """
    Parse one brain entry from brains.yaml into a BrainConfig.

    Raises KeyError if the named brain is missing from the file.
    """
    data: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
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
    summarizer: BrainConfig | None = field(default=None)

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
    _summarizer: BrainConfig | None = None
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


def find_pid_by_port(port: int) -> int | None:
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

def _http_ready(base_url: str, timeout_s: float = 1.0) -> tuple[bool, str]:
    """Probe /health, /v1/health, /v1/models, /props — delegates to openai_client.health_check()."""
    return health_check(base_url, timeouts=HttpTimeouts(connect_s=timeout_s, read_s=timeout_s))


# ──────────────────────────────────────────────────────────────────────────── #
# Log tail + fatal error detection                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _tail(path: Path | None, n_lines: int = 160) -> str:
    try:
        if not path or not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return ""


def _log_has_fatal_error(tail_text: str) -> str | None:
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
) -> tuple[bool, str]:
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

_CUDA_DEVICE_RE = re.compile(r"^CUDA(\d+)$", re.IGNORECASE)


def _plan_cuda_isolation(device: Any) -> tuple[str, str] | None:
    """
    Translate a physical device spec into (CUDA_VISIBLE_DEVICES, renumbered spec).

    `--device CUDA1` restricts where llama.cpp *places* tensors, but not which
    devices it *initialises*: every llama-server still builds a CUDA context on
    every visible GPU. Measured 2026-08-06 with three servers running — each
    held a context on all three cards, ~231 MiB apiece, including 694 MiB on the
    RTX 5080 that this app never uses at all.

    `CUDA_VISIBLE_DEVICES` is the documented way to actually hide them
    (upstream: "if you set it, llama.cpp only sees the specified GPUs"). The
    catch is that hiding renumbers what is left: with CUDA_VISIBLE_DEVICES=1 the
    surviving GPU is called CUDA0 inside the process, so a literal
    `--device CUDA1` would fail. Translating here keeps brains.yaml written in
    *physical* device numbers — which is what makes it readable, and what every
    comment and invariant in it assumes — while the process still sees one GPU.

    Returns None for anything not recognisably CUDA (a CPU-only brain like the
    summarizer, or a non-CUDA backend), leaving those spawns untouched.

    Note this is NOT the approach sage_kaizen_ai_ingest uses. It tried
    CUDA_VISIBLE_DEVICES and reverted: it broke jina-clip-v2's
    trust_remote_code paths, which passed /health and then returned 500 on real
    inference (its wiki_ingest.py documents this). That is a PyTorch failure
    mode; llama-server is C++ and was verified here before this was written —
    a BGE-M3 server started under CUDA_VISIBLE_DEVICES=1 with --device CUDA0
    served real embeddings and held a context on exactly one GPU.
    """
    if not isinstance(device, str):
        return None
    parts = [p.strip() for p in device.split(",") if p.strip()]
    indices: list[str] = []
    for part in parts:
        match = _CUDA_DEVICE_RE.match(part)
        if match is None:
            return None
        indices.append(match.group(1))
    if not indices:
        return None
    # After hiding, the survivors are renumbered 0..n-1 in the order listed.
    return ",".join(indices), ",".join(f"CUDA{i}" for i in range(len(indices)))


def _child_env(brain: BrainConfig) -> dict[str, str]:
    """
    Environment for a spawned llama-server: the parent's, plus GPU isolation.

    CUDA_DEVICE_ORDER is pinned to PCI_BUS_ID rather than inherited. The index
    in CUDA_VISIBLE_DEVICES only means the card brains.yaml names if the CUDA
    runtime enumerates in PCI order; the default is FASTEST_FIRST, under which
    the mapping is a guess. It happens to be set as a user-level variable on
    this machine, but relying on that would make a correct GPU assignment
    depend on an environment variable nothing in this repo controls — and the
    failure mode is ARCHITECT silently landing on the display GPU.
    """
    env = dict(os.environ)
    plan = _plan_cuda_isolation(brain.server.get("device"))
    if plan is not None:
        visible, _ = plan
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = visible
    return env


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

    The `device` key is rewritten to its post-isolation name — see
    _plan_cuda_isolation. brains.yaml keeps saying CUDA1; the process is handed
    CUDA0 and shown only that card.

    --log-file is always appended last (project invariant: never rely on
    stdout/stderr redirection for long-running servers).
    """
    argv: list = [str(brain.exe), "--model", str(brain.model)]
    isolation = _plan_cuda_isolation(brain.server.get("device"))

    for yaml_key, value in brain.server.items():
        cli_name = yaml_key.replace("_", "-")
        flag = f"--{cli_name}"

        if yaml_key == "device" and isolation is not None:
            argv.extend([flag, isolation[1]])
            continue

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

def start_server_from_config(brain: BrainConfig) -> tuple[bool, str]:
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
    isolation = _plan_cuda_isolation(brain.server.get("device"))

    # Record the physical->visible GPU mapping through sk_logging, NOT into the
    # header below. llama-server opens --log-file in truncating mode, so
    # anything written to that path before the spawn is erased the moment the
    # child starts — verified 2026-08-06: not one "START (yaml)" header has ever
    # survived in logs/*.log. The header is still written because it is the only
    # thing that captures early CUDA stderr on a spawn that dies before the
    # child's logger exists; it just cannot be relied on afterwards.
    #
    # This matters more than it used to: after isolation the server's own log
    # only ever says "CUDA0", so this record is the only thing tying a run to
    # the physical card it used.
    if isolation is not None:
        _LOG.info(
            "%s launch | device=%s | CUDA_VISIBLE_DEVICES=%s | --device %s",
            brain.name, brain.server.get("device"), isolation[0], isolation[1],
        )
    else:
        _LOG.info("%s launch | no CUDA isolation (device=%s)",
                  brain.name, brain.server.get("device"))

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
                env=_child_env(brain),
            )
        finally:
            log_fh.close()  # parent closes its copy; child keeps its own fd

        _register_spawned(proc)
    except Exception as e:
        return False, f"Failed to spawn llama-server: {e}"

    return True, "spawned"


# ──────────────────────────────────────────────────────────────────────────── #
# Public API: ensure_* functions (called by InferenceSession)                   #
# ──────────────────────────────────────────────────────────────────────────── #

def _ensure_brain_running(brain: BrainConfig, label: str) -> tuple[bool, str]:
    """
    Bring one llama-server up, or confirm it already is.

    Steps, in order:
      1. If something is listening on the port AND answers a readiness probe,
         it is already up — return without touching it.
      2. Otherwise clear the port. Something listening but not answering is a
         stale or wedged process; respawning on an occupied port would fail.
      3. Spawn from the BrainConfig (which came from brains.yaml).
      4. Block until /health answers or the configured timeout expires,
         aborting early on a fatal marker in the server's own log.

    `label` appears in the returned status strings, which the Streamlit status
    panel renders verbatim.

    Extracted 2026-08-05: ensure_embed_running / ensure_q5_running /
    ensure_q6_running / ensure_summarizer_running were four copies of this
    body differing only in which BrainConfig they read and what they called
    themselves.
    """
    base_url = brain.base_url

    if find_pid_by_port(brain.port) is not None:
        ok, how = _http_ready(base_url, timeout_s=1.0)
        if ok:
            return True, f"{label} already ready ({how})"

    stop_server_on_port(brain.port)

    ok, msg = start_server_from_config(brain)
    if not ok:
        return False, f"{label} start failed: {msg}"

    return _wait_for_ready(
        host=brain.host,
        port=brain.port,
        base_url=base_url,
        timeout_s=brain.startup_timeout_s,
        log_path=brain.log,
    )


def ensure_embed_running(servers: ManagedServers) -> tuple[bool, str]:
    """Start the BGE-M3 embedding server (port 8020) if not already running."""
    return _ensure_brain_running(servers.embed, "EMBED")


def ensure_q5_running(servers: ManagedServers) -> tuple[bool, str]:
    """
    Start the FAST brain (port 8011) if not already running.

    The embedding server is brought up first: RAG retrieval runs on every turn,
    so a chat brain without embeddings would answer without context.
    """
    ok, msg = ensure_embed_running(servers)
    if not ok:
        return False, f"Embeddings not ready: {msg}"
    return _ensure_brain_running(servers.fast, "Q5")


def ensure_q6_running(servers: ManagedServers) -> tuple[bool, str]:
    """Start the ARCHITECT brain (port 8012) if not already running."""
    return _ensure_brain_running(servers.architect, "Q6")


def ensure_summarizer_running(servers: ManagedServers) -> tuple[bool, str]:
    """
    Start the optional CPU summarizer brain (port 8013) if configured and not running.

    Returns (False, reason) immediately when no summarizer: section exists in
    brains.yaml — the caller must treat this as "not available, fall back to
    FAST brain" rather than an error.
    """
    if servers.summarizer is None:
        return False, "summarizer: section not configured in brains.yaml"
    return _ensure_brain_running(servers.summarizer, "Summarizer")
