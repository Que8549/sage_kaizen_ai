"""
tests/test_server_manager.py

Unit tests for server_manager.py.

The invariants from CLAUDE.md §5 get explicit assertions here, because they are
exactly the kind of thing a well-meaning refactor breaks:
  * no cmd.exe in the llama-server launch path
  * no stdout/stderr redirection for the server's own log — always --log-file
  * brains.yaml is the sole config source (no .bat parsing)
  * paths fully expanded before use
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

import server_manager as sm
from server_manager import (
    BrainConfig,
    ManagedServers,
    _build_argv,
    _child_env,
    _plan_cuda_isolation,
    _load_brain_config,
    _log_has_fatal_error,
    _tail,
    find_pid_by_port,
    start_server_from_config,
    stop_server_on_port,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def brain(tmp_path) -> BrainConfig:
    exe = tmp_path / "llama-server.exe"
    model = tmp_path / "model.gguf"
    exe.write_bytes(b"MZ")
    model.write_bytes(b"GGUF")
    return BrainConfig(
        name="fast",
        exe=exe,
        model=model,
        log=tmp_path / "logs" / "fast.log",
        startup_timeout_s=30.0,
        server={"host": "127.0.0.1", "port": 8011, "ctx-size": 32768},
    )


@pytest.fixture
def yaml_file(tmp_path) -> Path:
    data = {
        "embed": {
            "exe": str(tmp_path / "s.exe"), "model": str(tmp_path / "e.gguf"),
            "log": str(tmp_path / "e.log"),
            "server": {"host": "127.0.0.1", "port": 8020},
        },
        "fast": {
            "exe": str(tmp_path / "s.exe"), "model": str(tmp_path / "f.gguf"),
            "log": str(tmp_path / "f.log"), "startup_timeout_s": 1800,
            "server": {"host": "127.0.0.1", "port": 8011},
        },
        "architect": {
            "exe": str(tmp_path / "s.exe"), "model": str(tmp_path / "a.gguf"),
            "log": str(tmp_path / "a.log"),
            "server": {"host": "127.0.0.1", "port": 8012},
        },
    }
    p = tmp_path / "brains.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# CUDA isolation — _plan_cuda_isolation / _child_env
#
# These decide which physical GPU a brain lands on. The failure mode if the
# translation is wrong is ARCHITECT silently loading onto the display GPU, so
# the mapping is asserted rather than assumed.
# ---------------------------------------------------------------------------

def _brain_on(tmp_path, device) -> BrainConfig:
    return BrainConfig(
        name="t", exe=tmp_path / "e", model=tmp_path / "m", log=tmp_path / "l",
        startup_timeout_s=1.0,
        server={"port": 8012, "device": device, "ctx_size": 4096},
    )


class TestPlanCudaIsolation:
    def test_single_device_is_hidden_and_renumbered(self):
        # CUDA1 physical -> the process sees one GPU, called CUDA0.
        assert _plan_cuda_isolation("CUDA1") == ("1", "CUDA0")

    def test_cuda0_maps_to_itself(self):
        assert _plan_cuda_isolation("CUDA0") == ("0", "CUDA0")

    def test_high_index_device(self):
        assert _plan_cuda_isolation("CUDA2") == ("2", "CUDA0")

    def test_multiple_devices_renumber_in_listed_order(self):
        # Order matters: CUDA_VISIBLE_DEVICES=2,0 makes physical 2 become CUDA0.
        assert _plan_cuda_isolation("CUDA2,CUDA0") == ("2,0", "CUDA0,CUDA1")

    def test_whitespace_between_devices_is_tolerated(self):
        assert _plan_cuda_isolation("CUDA0, CUDA1") == ("0,1", "CUDA0,CUDA1")

    def test_case_insensitive(self):
        assert _plan_cuda_isolation("cuda1") == ("1", "CUDA0")

    def test_non_cuda_backend_is_left_alone(self):
        # Hiding CUDA devices for a Vulkan/CPU spawn would be meaningless at
        # best and wrong at worst.
        assert _plan_cuda_isolation("Vulkan0") is None

    def test_mixed_backends_are_left_alone(self):
        assert _plan_cuda_isolation("CUDA0,Vulkan0") is None

    def test_missing_device_is_left_alone(self):
        # The CPU-only summarizer has no device key at all.
        assert _plan_cuda_isolation(None) is None

    def test_non_string_device_is_left_alone(self):
        assert _plan_cuda_isolation(0) is None

    def test_empty_string_is_left_alone(self):
        assert _plan_cuda_isolation("") is None

    def test_bare_cuda_without_index_is_left_alone(self):
        assert _plan_cuda_isolation("CUDA") is None


class TestChildEnv:
    def test_sets_visible_devices_to_the_physical_index(self, tmp_path):
        env = _child_env(_brain_on(tmp_path, "CUDA1"))
        assert env["CUDA_VISIBLE_DEVICES"] == "1"

    def test_pins_pci_bus_order(self, tmp_path):
        # Without PCI_BUS_ID the runtime default is FASTEST_FIRST, under which
        # "1" is not reliably the card brains.yaml means.
        env = _child_env(_brain_on(tmp_path, "CUDA1"))
        assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    def test_overrides_an_inherited_device_order(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
        env = _child_env(_brain_on(tmp_path, "CUDA1"))
        assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    def test_inherits_the_parent_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SAGE_KAIZEN_TEST_MARKER", "kept")
        env = _child_env(_brain_on(tmp_path, "CUDA1"))
        assert env["SAGE_KAIZEN_TEST_MARKER"] == "kept"

    def test_cpu_only_brain_gets_no_isolation(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        b = BrainConfig("s", tmp_path / "e", tmp_path / "m", tmp_path / "l",
                        1.0, {"port": 8013, "n_gpu_layers": 0})
        assert "CUDA_VISIBLE_DEVICES" not in _child_env(b)

    def test_does_not_mutate_os_environ(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        _child_env(_brain_on(tmp_path, "CUDA1"))
        assert "CUDA_VISIBLE_DEVICES" not in os.environ


class TestBuildArgvDeviceTranslation:
    def test_device_flag_is_renumbered(self, tmp_path):
        # brains.yaml says CUDA1; the process is shown only that card, where it
        # is called CUDA0. Passing CUDA1 through verbatim would fail to start.
        argv = _build_argv(_brain_on(tmp_path, "CUDA1"))
        assert argv[argv.index("--device") + 1] == "CUDA0"

    def test_device_and_env_agree(self, tmp_path):
        b = _brain_on(tmp_path, "CUDA2")
        argv = _build_argv(b)
        assert argv[argv.index("--device") + 1] == "CUDA0"
        assert _child_env(b)["CUDA_VISIBLE_DEVICES"] == "2"

    def test_untranslatable_device_passes_through_verbatim(self, tmp_path):
        argv = _build_argv(_brain_on(tmp_path, "Vulkan0"))
        assert argv[argv.index("--device") + 1] == "Vulkan0"

    def test_other_flags_are_unaffected(self, tmp_path):
        argv = _build_argv(_brain_on(tmp_path, "CUDA1"))
        assert argv[argv.index("--ctx-size") + 1] == "4096"


# ---------------------------------------------------------------------------
# BrainConfig
# ---------------------------------------------------------------------------

class TestBrainConfig:
    def test_host_from_server_dict(self, brain):
        assert brain.host == "127.0.0.1"

    def test_port_from_server_dict(self, brain):
        assert brain.port == 8011

    def test_base_url_composition(self, brain):
        assert brain.base_url == "http://127.0.0.1:8011"

    def test_host_default(self, tmp_path):
        b = BrainConfig("x", tmp_path / "e", tmp_path / "m", tmp_path / "l", 1.0, {})
        assert b.host == "127.0.0.1"

    def test_port_default(self, tmp_path):
        b = BrainConfig("x", tmp_path / "e", tmp_path / "m", tmp_path / "l", 1.0, {})
        assert b.port == 8080

    def test_is_frozen(self, brain):
        with pytest.raises(Exception):
            brain.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _build_argv — flag-type tables
# ---------------------------------------------------------------------------

class TestBuildArgv:
    def test_exe_first_then_model(self, brain):
        argv = _build_argv(brain)
        assert argv[0] == str(brain.exe)
        assert argv[1] == "--model"
        assert argv[2] == str(brain.model)

    def test_log_file_is_always_last(self, brain):
        argv = _build_argv(brain)
        assert argv[-2] == "--log-file"
        assert argv[-1] == str(brain.log)

    def test_underscores_become_hyphens(self, tmp_path):
        b = BrainConfig("x", tmp_path / "e", tmp_path / "m", tmp_path / "l", 1.0,
                        {"ctx_size": 4096, "n_gpu_layers": 99})
        argv = _build_argv(b)
        assert "--ctx-size" in argv and "--n-gpu-layers" in argv
        assert "--ctx_size" not in argv

    def test_key_value_pairs(self, brain):
        argv = _build_argv(brain)
        assert argv[argv.index("--ctx-size") + 1] == "32768"

    @pytest.mark.parametrize("flag", ["flash-attn", "log-colors", "fit"])
    def test_on_off_flags_true(self, tmp_path, flag):
        b = BrainConfig("x", tmp_path / "e", tmp_path / "m", tmp_path / "l", 1.0,
                        {flag.replace("-", "_"): True})
        argv = _build_argv(b)
        assert argv[argv.index(f"--{flag}") + 1] == "on"

    @pytest.mark.parametrize("flag", ["flash-attn", "log-colors", "fit"])
    def test_on_off_flags_false_still_emitted(self, tmp_path, flag):
        """These three are explicit-value flags, present even when False."""
        b = BrainConfig("x", tmp_path / "e", tmp_path / "m", tmp_path / "l", 1.0,
                        {flag.replace("-", "_"): False})
        argv = _build_argv(b)
        assert argv[argv.index(f"--{flag}") + 1] == "off"

    def test_presence_only_bool_true_emits_bare_flag(self, tmp_path):
        b = BrainConfig("x", tmp_path / "e", tmp_path / "m", tmp_path / "l", 1.0,
                        {"no_kv_unified": True})
        argv = _build_argv(b)
        assert "--no-kv-unified" in argv
        assert argv[argv.index("--no-kv-unified") + 1] == "--log-file"

    def test_presence_only_bool_false_is_omitted(self, tmp_path):
        b = BrainConfig("x", tmp_path / "e", tmp_path / "m", tmp_path / "l", 1.0,
                        {"no_kv_unified": False})
        assert "--no-kv-unified" not in _build_argv(b)

    def test_yaml_key_order_is_preserved(self, tmp_path):
        b = BrainConfig("x", tmp_path / "e", tmp_path / "m", tmp_path / "l", 1.0,
                        {"alpha": 1, "beta": 2, "gamma": 3})
        argv = _build_argv(b)
        assert argv.index("--alpha") < argv.index("--beta") < argv.index("--gamma")

    def test_every_argv_element_is_a_string(self, brain):
        assert all(isinstance(a, str) for a in _build_argv(brain))

    # ── CLAUDE.md §5 invariants ──────────────────────────────────────────────

    def test_invariant_no_cmd_exe_in_argv(self, brain):
        """Invariant 2: never launch llama-server via cmd.exe."""
        argv = _build_argv(brain)
        assert not any("cmd.exe" in a.lower() or a == "/c" for a in argv)

    def test_invariant_no_shell_redirection_operators(self, brain):
        """Invariant 3: always --log-file, never > or >> redirection."""
        argv = _build_argv(brain)
        assert not any(a in (">", ">>", "2>", "2>&1") for a in argv)

    def test_invariant_log_file_always_present(self, brain):
        assert "--log-file" in _build_argv(brain)

    def test_invariant_paths_are_fully_expanded(self, brain):
        """Invariant 4: no %ROOT% or $VAR left in any path argument."""
        argv = _build_argv(brain)
        assert not any("%" in a or "$" in a for a in argv)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

class TestLoadBrainConfig:
    def test_loads_named_brain(self, yaml_file):
        cfg = _load_brain_config(yaml_file, "fast")
        assert cfg.name == "fast"
        assert cfg.port == 8011

    def test_explicit_startup_timeout_honoured(self, yaml_file):
        assert _load_brain_config(yaml_file, "fast").startup_timeout_s == 1800.0

    def test_default_timeout_per_brain_name(self, yaml_file):
        assert _load_brain_config(yaml_file, "architect").startup_timeout_s == 2700.0
        assert _load_brain_config(yaml_file, "embed").startup_timeout_s == 300.0

    def test_missing_brain_raises_keyerror(self, yaml_file):
        with pytest.raises(KeyError):
            _load_brain_config(yaml_file, "nonexistent")

    def test_paths_become_path_objects(self, yaml_file):
        cfg = _load_brain_config(yaml_file, "fast")
        assert isinstance(cfg.exe, Path)
        assert isinstance(cfg.model, Path)
        assert isinstance(cfg.log, Path)


class TestManagedServers:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        sm._load_managed_servers.cache_clear()
        yield
        sm._load_managed_servers.cache_clear()

    def test_from_yaml_loads_three_brains(self, yaml_file):
        s = ManagedServers.from_yaml(yaml_file)
        assert s.fast.port == 8011
        assert s.architect.port == 8012
        assert s.embed.port == 8020

    def test_summarizer_absent_is_not_an_error(self, yaml_file):
        assert ManagedServers.from_yaml(yaml_file).summarizer is None

    def test_summarizer_loaded_when_present(self, tmp_path, yaml_file):
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        data["summarizer"] = {
            "exe": str(tmp_path / "s.exe"), "model": str(tmp_path / "sum.gguf"),
            "log": str(tmp_path / "sum.log"),
            "server": {"host": "127.0.0.1", "port": 8013, "alias": "Qwen3-4B"},
        }
        yaml_file.write_text(yaml.safe_dump(data), encoding="utf-8")
        s = ManagedServers.from_yaml(yaml_file)
        assert s.summarizer is not None
        assert s.summarizer.port == 8013

    def test_missing_file_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="brains.yaml not found"):
            ManagedServers.from_yaml(tmp_path / "nope.yaml")

    def test_result_is_cached(self, yaml_file):
        a = ManagedServers.from_yaml(yaml_file)
        b = ManagedServers.from_yaml(yaml_file)
        assert a is b

    def test_convenience_properties(self, yaml_file):
        s = ManagedServers.from_yaml(yaml_file)
        assert s.host == "127.0.0.1"
        assert (s.q5_port, s.q6_port, s.embed_port) == (8011, 8012, 8020)
        assert s.q5_log == s.fast.log
        assert s.q6_log == s.architect.log
        assert s.embed_log == s.embed.log
        assert s.q5_start_timeout_s == s.fast.startup_timeout_s
        assert s.q6_start_timeout_s == s.architect.startup_timeout_s
        assert s.embed_start_timeout_s == s.embed.startup_timeout_s


# ---------------------------------------------------------------------------
# Log tail + fatal-marker detection
# ---------------------------------------------------------------------------

class TestTail:
    def test_missing_file_returns_empty(self, tmp_path):
        assert _tail(tmp_path / "nope.log") == ""

    def test_none_path_returns_empty(self):
        assert _tail(None) == ""

    def test_returns_last_n_lines(self, tmp_path):
        p = tmp_path / "a.log"
        p.write_text("\n".join(str(i) for i in range(500)), encoding="utf-8")
        assert _tail(p, n_lines=10).splitlines() == [str(i) for i in range(490, 500)]

    def test_tolerates_invalid_encoding(self, tmp_path):
        p = tmp_path / "a.log"
        p.write_bytes(b"ok\n\xff\xfe bad bytes\n")
        assert "ok" in _tail(p)


class TestLogHasFatalError:
    @pytest.mark.parametrize(
        "text",
        [
            "error: invalid argument --bogus",
            "failed to load model",
            "ERROR LOADING MODEL",
            "cudaMalloc failed",
            "unable to allocate CUDA0 buffer",
            "EXE not found",
        ],
    )
    def test_detects_markers_case_insensitively(self, text):
        assert _log_has_fatal_error(text) is not None

    def test_returns_none_for_healthy_log(self):
        assert _log_has_fatal_error("loading model...\nserver listening on 8011") is None

    def test_empty_text(self):
        assert _log_has_fatal_error("") is None

    def test_returns_the_matched_marker(self):
        assert _log_has_fatal_error("xx failed to load model xx") == "failed to load model"


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

class TestFindPidByPort:
    def test_parses_listening_pid(self):
        out = (
            "  Proto  Local Address          Foreign Address        State           PID\n"
            "  TCP    127.0.0.1:8011         0.0.0.0:0              LISTENING       4242\n"
        )
        with patch.object(sm, "_run", return_value=MagicMock(returncode=0, stdout=out)):
            assert find_pid_by_port(8011) == 4242

    def test_returns_none_when_port_absent(self):
        out = "  TCP    127.0.0.1:9999         0.0.0.0:0              LISTENING       1\n"
        with patch.object(sm, "_run", return_value=MagicMock(returncode=0, stdout=out)):
            assert find_pid_by_port(8011) is None

    def test_ignores_non_listening_states(self):
        out = "  TCP    127.0.0.1:8011         1.2.3.4:5              ESTABLISHED     77\n"
        with patch.object(sm, "_run", return_value=MagicMock(returncode=0, stdout=out)):
            assert find_pid_by_port(8011) is None

    def test_returns_none_on_command_failure(self):
        with patch.object(sm, "_run", return_value=MagicMock(returncode=1, stdout="")):
            assert find_pid_by_port(8011) is None

    def test_does_not_match_a_port_that_is_a_suffix(self):
        """Port 011 must not match a line for port 8011."""
        out = "  TCP    127.0.0.1:8011  0.0.0.0:0  LISTENING  4242\n"
        with patch.object(sm, "_run", return_value=MagicMock(returncode=0, stdout=out)):
            assert find_pid_by_port(11) is None


class TestStopServerOnPort:
    def test_returns_true_when_nothing_listening(self):
        with patch.object(sm, "find_pid_by_port", return_value=None):
            assert stop_server_on_port(8011) is True

    def test_kills_and_reports_success(self):
        with (
            patch.object(sm, "find_pid_by_port", return_value=999),
            patch.object(sm, "_run", return_value=MagicMock(returncode=0)) as run,
        ):
            assert stop_server_on_port(8011) is True
        assert run.call_args.args[0] == ["taskkill", "/PID", "999", "/F"]

    def test_reports_failure(self):
        with (
            patch.object(sm, "find_pid_by_port", return_value=999),
            patch.object(sm, "_run", return_value=MagicMock(returncode=1)),
        ):
            assert stop_server_on_port(8011) is False


# ---------------------------------------------------------------------------
# start_server_from_config
# ---------------------------------------------------------------------------

class TestStartServerFromConfig:
    def test_missing_exe(self, tmp_path):
        b = BrainConfig("x", tmp_path / "nope.exe", tmp_path / "m.gguf",
                        tmp_path / "l.log", 1.0, {})
        ok, msg = start_server_from_config(b)
        assert ok is False and "EXE not found" in msg

    def test_missing_model(self, tmp_path, brain):
        b = BrainConfig("x", brain.exe, tmp_path / "nope.gguf", tmp_path / "l.log", 1.0, {})
        ok, msg = start_server_from_config(b)
        assert ok is False and "MODEL not found" in msg

    def test_spawns_and_reports_success(self, brain):
        with patch.object(sm.subprocess, "Popen") as P:
            P.return_value = MagicMock(poll=MagicMock(return_value=None))
            ok, msg = start_server_from_config(brain)
        assert (ok, msg) == (True, "spawned")

    def test_creates_log_directory(self, brain):
        with patch.object(sm.subprocess, "Popen") as P:
            P.return_value = MagicMock(poll=MagicMock(return_value=None))
            start_server_from_config(brain)
        assert brain.log.parent.is_dir()

    def test_writes_startup_header_to_log(self, brain):
        with patch.object(sm.subprocess, "Popen") as P:
            P.return_value = MagicMock(poll=MagicMock(return_value=None))
            start_server_from_config(brain)
        text = brain.log.read_text(encoding="utf-8", errors="ignore")
        assert "FAST START (yaml)" in text
        assert str(brain.exe) in text

    def test_spawn_failure_is_reported_not_raised(self, brain):
        with patch.object(sm.subprocess, "Popen", side_effect=OSError("denied")):
            ok, msg = start_server_from_config(brain)
        assert ok is False and "Failed to spawn" in msg

    # ── CLAUDE.md §5 invariants at the spawn site ────────────────────────────

    def test_invariant_argv_is_a_list_not_a_shell_string(self, brain):
        with patch.object(sm.subprocess, "Popen") as P:
            P.return_value = MagicMock(poll=MagicMock(return_value=None))
            start_server_from_config(brain)
        assert isinstance(P.call_args.args[0], list)

    def test_invariant_no_shell_true(self, brain):
        with patch.object(sm.subprocess, "Popen") as P:
            P.return_value = MagicMock(poll=MagicMock(return_value=None))
            start_server_from_config(brain)
        assert P.call_args.kwargs.get("shell", False) is False

    def test_invariant_exe_is_invoked_directly(self, brain):
        with patch.object(sm.subprocess, "Popen") as P:
            P.return_value = MagicMock(poll=MagicMock(return_value=None))
            start_server_from_config(brain)
        assert P.call_args.args[0][0] == str(brain.exe)

    def test_invariant_stdout_is_devnull_not_the_log(self, brain):
        """The server writes its own log via --log-file; stdout must not be it."""
        with patch.object(sm.subprocess, "Popen") as P:
            P.return_value = MagicMock(poll=MagicMock(return_value=None))
            start_server_from_config(brain)
        assert P.call_args.kwargs["stdout"] == subprocess.DEVNULL


class TestSpawnedProcessRegistry:
    def test_dead_processes_are_pruned(self):
        dead = MagicMock(poll=MagicMock(return_value=0))
        alive = MagicMock(poll=MagicMock(return_value=None))
        with patch.object(sm, "_spawned_procs", [dead]):
            sm._register_spawned(alive)
            assert sm._spawned_procs == [alive]

    def test_live_processes_are_kept(self):
        a = MagicMock(poll=MagicMock(return_value=None))
        b = MagicMock(poll=MagicMock(return_value=None))
        with patch.object(sm, "_spawned_procs", [a]):
            sm._register_spawned(b)
            assert sm._spawned_procs == [a, b]

    def test_atexit_terminates_only_running_processes(self):
        dead = MagicMock(poll=MagicMock(return_value=0))
        alive = MagicMock(poll=MagicMock(return_value=None))
        with patch.object(sm, "_spawned_procs", [dead, alive]):
            sm._kill_spawned_servers()
        dead.terminate.assert_not_called()
        alive.terminate.assert_called_once()

    def test_atexit_survives_terminate_failure(self):
        bad = MagicMock(poll=MagicMock(return_value=None))
        bad.terminate.side_effect = OSError("gone")
        with patch.object(sm, "_spawned_procs", [bad]):
            sm._kill_spawned_servers()   # must not raise


# ---------------------------------------------------------------------------
# ensure_* orchestration
# ---------------------------------------------------------------------------

@pytest.fixture
def servers(yaml_file):
    sm._load_managed_servers.cache_clear()
    s = ManagedServers.from_yaml(yaml_file)
    yield s
    sm._load_managed_servers.cache_clear()


class TestEnsureRunning:
    def test_embed_already_ready_short_circuits(self, servers):
        with (
            patch.object(sm, "find_pid_by_port", return_value=1),
            patch.object(sm, "_http_ready", return_value=(True, "OK (/health)")),
            patch.object(sm, "start_server_from_config") as start,
        ):
            ok, msg = sm.ensure_embed_running(servers)
        assert ok is True and "already ready" in msg
        start.assert_not_called()

    def test_embed_starts_when_not_listening(self, servers):
        with (
            patch.object(sm, "find_pid_by_port", return_value=None),
            patch.object(sm, "stop_server_on_port", return_value=True),
            patch.object(sm, "start_server_from_config", return_value=(True, "spawned")) as start,
            patch.object(sm, "_wait_for_ready", return_value=(True, "ready")),
        ):
            assert sm.ensure_embed_running(servers) == (True, "ready")
        start.assert_called_once_with(servers.embed)

    def test_embed_start_failure_is_reported(self, servers):
        with (
            patch.object(sm, "find_pid_by_port", return_value=None),
            patch.object(sm, "stop_server_on_port", return_value=True),
            patch.object(sm, "start_server_from_config", return_value=(False, "EXE not found")),
        ):
            ok, msg = sm.ensure_embed_running(servers)
        assert ok is False and "EMBED start failed" in msg

    def test_q5_requires_embed_first(self, servers):
        """The embedding server must be ready before the chat brain starts."""
        with patch.object(sm, "ensure_embed_running", return_value=(False, "boom")) as e:
            ok, msg = sm.ensure_q5_running(servers)
        e.assert_called_once()
        assert ok is False and "Embeddings not ready" in msg

    def test_q5_starts_after_embed_is_ready(self, servers):
        with (
            patch.object(sm, "ensure_embed_running", return_value=(True, "ok")),
            patch.object(sm, "find_pid_by_port", return_value=None),
            patch.object(sm, "stop_server_on_port", return_value=True),
            patch.object(sm, "start_server_from_config", return_value=(True, "spawned")) as start,
            patch.object(sm, "_wait_for_ready", return_value=(True, "ready")),
        ):
            assert sm.ensure_q5_running(servers) == (True, "ready")
        start.assert_called_once_with(servers.fast)

    def test_q6_does_not_require_embed(self, servers):
        with (
            patch.object(sm, "ensure_embed_running") as e,
            patch.object(sm, "find_pid_by_port", return_value=None),
            patch.object(sm, "stop_server_on_port", return_value=True),
            patch.object(sm, "start_server_from_config", return_value=(True, "spawned")) as start,
            patch.object(sm, "_wait_for_ready", return_value=(True, "ready")),
        ):
            sm.ensure_q6_running(servers)
        e.assert_not_called()
        start.assert_called_once_with(servers.architect)

    def test_summarizer_absent_is_reported_not_raised(self, servers):
        ok, msg = sm.ensure_summarizer_running(servers)
        assert ok is False
        assert "not configured" in msg

    def test_stale_port_is_cleared_before_restart(self, servers):
        """A listening-but-unhealthy port gets killed before respawning."""
        with (
            patch.object(sm, "find_pid_by_port", return_value=123),
            patch.object(sm, "_http_ready", return_value=(False, "not ready")),
            patch.object(sm, "stop_server_on_port", return_value=True) as stop,
            patch.object(sm, "start_server_from_config", return_value=(True, "spawned")),
            patch.object(sm, "_wait_for_ready", return_value=(True, "ready")),
        ):
            sm.ensure_embed_running(servers)
        stop.assert_called_once_with(servers.embed_port)


class TestWaitForReady:
    def test_returns_immediately_when_ready(self, tmp_path):
        with patch.object(sm, "_http_ready", return_value=(True, "OK (/health)")):
            ok, msg = sm._wait_for_ready("h", 1, "http://h:1", 5.0, tmp_path / "l.log")
        assert ok is True and "ready" in msg

    def test_aborts_early_on_fatal_log_marker(self, tmp_path):
        log = tmp_path / "l.log"
        log.write_text("error: invalid argument --bogus", encoding="utf-8")
        with (
            patch.object(sm, "_http_ready", return_value=(False, "no")),
            patch.object(sm.time, "sleep"),
        ):
            ok, msg = sm._wait_for_ready("h", 1, "http://h:1", 30.0, log)
        assert ok is False and "fatal in log" in msg

    def test_times_out_with_log_tail(self, tmp_path):
        log = tmp_path / "l.log"
        log.write_text("loading model, please wait", encoding="utf-8")
        with (
            patch.object(sm, "_http_ready", return_value=(False, "no")),
            patch.object(sm.time, "sleep"),
        ):
            ok, msg = sm._wait_for_ready("h", 1, "http://h:1", 0.0, log)
        assert ok is False and "timeout" in msg
