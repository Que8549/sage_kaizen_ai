"""
tests/test_prompt_and_settings.py

Unit tests for prompt_library.py, settings.py, pg_settings.py and
inference_session.py — the startup-configuration layer.

prompt_library.py is the single source of truth for all prompts (CLAUDE.md
invariant 5), so these tests assert its structural contracts rather than exact
wording, which is expected to be edited freely.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prompt_library import (
    TEMPLATES,
    TemplateKey,
    build_system_only,
    sage_architect_core,
    sage_fast_core,
    sage_kaizen_system_prompt,
)


# ---------------------------------------------------------------------------
# prompt_library
# ---------------------------------------------------------------------------

class TestTemplateKey:
    def test_is_a_str_enum(self):
        assert issubclass(TemplateKey, str)

    def test_every_key_has_a_template(self):
        missing = [k for k in TemplateKey if k not in TEMPLATES]
        assert not missing, f"TemplateKey(s) with no TEMPLATES entry: {missing}"

    def test_no_orphan_templates(self):
        orphans = [k for k in TEMPLATES if k not in set(TemplateKey)]
        assert not orphans

    def test_all_templates_are_non_empty(self):
        blank = [k for k, v in TEMPLATES.items() if not v.strip()]
        assert not blank


class TestCorePrompts:
    @pytest.mark.parametrize(
        "prompt", [sage_kaizen_system_prompt, sage_fast_core, sage_architect_core]
    )
    def test_non_empty(self, prompt):
        assert isinstance(prompt, str) and prompt.strip()

    def test_fast_and_architect_cores_differ(self):
        assert sage_fast_core != sage_architect_core

    def test_fast_core_carries_the_english_only_instruction(self):
        """Mitigation for Qwen2.5-Omni-7B's mid-response Chinese code-switching."""
        assert "english" in sage_fast_core.lower()

    @pytest.mark.parametrize(
        "prompt", [sage_kaizen_system_prompt, sage_fast_core, sage_architect_core]
    )
    def test_prompts_are_valid_unicode_text(self, prompt):
        """
        Guards the encoding gotcha, not typography.

        Smart quotes in prompt *prose* are fine and are present deliberately.
        What would break things is a prompt that can't round-trip through UTF-8
        (the encoding used for every log file, the DB, and the HTTP payload).
        """
        assert prompt.encode("utf-8").decode("utf-8") == prompt


class TestBuildSystemOnly:
    def test_system_prompt_alone(self):
        out = build_system_only(system_prompt="BASE", core_prompt="", templates=())
        assert "BASE" in out

    def test_core_prompt_included(self):
        out = build_system_only(system_prompt="BASE", core_prompt="CORE", templates=())
        assert "BASE" in out and "CORE" in out

    def test_templates_appended(self):
        out = build_system_only(
            system_prompt="BASE", core_prompt="CORE",
            templates=(TemplateKey.TEACHING_TUTORING,),
        )
        # build_system_only strips the assembled result, so compare on strip().
        assert TEMPLATES[TemplateKey.TEACHING_TUTORING].strip() in out

    def test_multiple_templates_all_present(self):
        keys = (TemplateKey.UNIVERSAL_DEPTH_ANCHOR, TemplateKey.AUTO_ADAPTIVE_META)
        out = build_system_only(system_prompt="B", core_prompt="C", templates=keys)
        for k in keys:
            assert TEMPLATES[k].strip() in out

    def test_sections_are_blank_line_separated(self):
        out = build_system_only(
            system_prompt="BASE", core_prompt="CORE",
            templates=(TemplateKey.TEACHING_TUTORING,),
        )
        assert out.startswith("BASE\n\nCORE\n\n")

    def test_all_empty_inputs_do_not_crash(self):
        assert isinstance(build_system_only(system_prompt="", core_prompt="", templates=()), str)

    def test_result_is_a_string(self):
        out = build_system_only(
            system_prompt="B", core_prompt="C", templates=tuple(TemplateKey)
        )
        assert isinstance(out, str) and out


# ---------------------------------------------------------------------------
# settings.ServerConfig
# ---------------------------------------------------------------------------

class TestServerConfig:
    @pytest.fixture
    def cfg(self):
        from settings import ServerConfig
        # _env_file=None isolates the test from the developer's real .env.
        return ServerConfig(_env_file=None)  # type: ignore[call-arg]

    def test_default_endpoints(self, cfg):
        assert cfg.sage_q5_base_url == "http://127.0.0.1:8011"
        assert cfg.sage_q6_base_url == "http://127.0.0.1:8012"
        assert cfg.sage_embed_base_url == "http://127.0.0.1:8020"

    def test_ports_match_the_documented_service_inventory(self, cfg):
        """FAST 8011 / ARCHITECT 8012 / embed 8020 per CLAUDE.md §2."""
        assert cfg.q5_base_url.endswith(":8011")
        assert cfg.q6_base_url.endswith(":8012")
        assert cfg.embed_base_url.endswith(":8020")

    def test_alias_properties_mirror_fields(self, cfg):
        assert cfg.q5_base_url == cfg.sage_q5_base_url
        assert cfg.q6_base_url == cfg.sage_q6_base_url
        assert cfg.embed_base_url == cfg.sage_embed_base_url
        assert cfg.q5_model_id == cfg.sage_q5_model_id
        assert cfg.q6_model_id == cfg.sage_q6_model_id
        assert cfg.embedded_model_id == cfg.sage_embed_model_id
        assert cfg.max_history_messages == cfg.sage_max_history_messages
        assert cfg.connect_timeout_s == cfg.sage_connect_timeout_s
        assert cfg.read_timeout_s == cfg.sage_read_timeout_s
        assert cfg.stream_keepalive_s == cfg.sage_stream_keepalive_s

    def test_system_prompt_comes_from_prompt_library(self, cfg):
        assert cfg.system_prompt == sage_kaizen_system_prompt

    def test_read_timeout_allows_long_architect_turns(self, cfg):
        """ARCHITECT runs at ~7.5 t/s; a short read timeout would truncate."""
        assert cfg.read_timeout_s >= 600

    def test_env_override(self, monkeypatch):
        from settings import ServerConfig
        monkeypatch.setenv("SAGE_Q5_BASE_URL", "http://otherhost:9999")
        assert ServerConfig(_env_file=None).q5_base_url == "http://otherhost:9999"  # type: ignore[call-arg]

    def test_env_is_case_insensitive(self, monkeypatch):
        from settings import ServerConfig
        monkeypatch.setenv("sage_max_history_messages", "5")
        assert ServerConfig(_env_file=None).max_history_messages == 5  # type: ignore[call-arg]

    def test_unknown_env_vars_are_ignored(self, monkeypatch):
        from settings import ServerConfig
        monkeypatch.setenv("SAGE_TOTALLY_UNKNOWN_KEY", "x")
        ServerConfig(_env_file=None)  # type: ignore[call-arg]  # extra="ignore"

    def test_module_level_config_singleton_exists(self):
        from settings import CONFIG
        assert CONFIG.system_prompt == sage_kaizen_system_prompt


# ---------------------------------------------------------------------------
# pg_settings
# ---------------------------------------------------------------------------

class TestPgSettings:
    def test_builds_a_dsn(self, monkeypatch):
        from pg_settings import PgSettings
        s = PgSettings()
        assert isinstance(s.pg_dsn, str)
        assert s.pg_dsn.startswith("postgresql://")

    def test_dsn_is_stable_across_instances(self):
        from pg_settings import PgSettings
        assert PgSettings().pg_dsn == PgSettings().pg_dsn


# ---------------------------------------------------------------------------
# inference_session
# ---------------------------------------------------------------------------

@pytest.fixture
def servers():
    s = MagicMock()
    s.q5_port, s.q6_port, s.embed_port = 8011, 8012, 8020
    s.summarizer = None
    return s


@pytest.fixture
def session(servers):
    from inference_session import InferenceSession
    return InferenceSession(
        q5_url="http://127.0.0.1:8011",
        q6_url="http://127.0.0.1:8012",
        embed_url="http://127.0.0.1:8020",
        q5_model_id="FAST-ID",
        q6_model_id="ARCH-ID",
        servers=servers,
    )


class TestInferenceSessionBrainMapping:
    def test_url_for_architect(self, session):
        assert session.url_for_brain("ARCHITECT") == "http://127.0.0.1:8012"

    def test_url_for_fast(self, session):
        assert session.url_for_brain("FAST") == "http://127.0.0.1:8011"

    def test_unknown_brain_falls_back_to_fast(self, session):
        assert session.url_for_brain("MYSTERY") == "http://127.0.0.1:8011"

    def test_model_id_for_architect(self, session):
        assert session.model_id_for_brain("ARCHITECT") == "ARCH-ID"

    def test_model_id_for_fast(self, session):
        assert session.model_id_for_brain("FAST") == "FAST-ID"


class TestInferenceSessionHealth:
    def test_health_q5_delegates(self, session):
        from openai_client import HttpTimeouts
        t = HttpTimeouts(connect_s=1, read_s=1)
        with patch("inference_session.health_check", return_value=(True, "ok")) as hc:
            assert session.health_q5(t) == (True, "ok")
        assert hc.call_args.args[0] == "http://127.0.0.1:8011"

    def test_health_q6_delegates(self, session):
        from openai_client import HttpTimeouts
        t = HttpTimeouts(connect_s=1, read_s=1)
        with patch("inference_session.health_check", return_value=(False, "down")) as hc:
            assert session.health_q6(t) == (False, "down")
        assert hc.call_args.args[0] == "http://127.0.0.1:8012"

    def test_discover_model_ids_probes_both(self, session):
        from openai_client import HttpTimeouts
        t = HttpTimeouts(connect_s=1, read_s=1)
        with patch("inference_session.discover_model_id", side_effect=["a", "b"]) as d:
            assert session.discover_model_ids(t) == ("a", "b")
        assert d.call_count == 2


class TestInferenceSessionLifecycle:
    def test_ensure_q5_delegates(self, session, servers):
        with patch("inference_session.ensure_q5_running", return_value=(True, "up")) as f:
            assert session.ensure_q5_ready() == (True, "up")
        f.assert_called_once_with(servers)

    def test_ensure_q6_delegates(self, session, servers):
        with patch("inference_session.ensure_q6_running", return_value=(True, "up")) as f:
            assert session.ensure_q6_ready() == (True, "up")
        f.assert_called_once_with(servers)

    def test_ensure_summarizer_delegates(self, session, servers):
        with patch("inference_session.ensure_summarizer_running",
                   return_value=(False, "not configured")) as f:
            assert session.ensure_summarizer_ready() == (False, "not configured")
        f.assert_called_once_with(servers)

    def test_stop_all_stops_three_ports(self, session):
        with patch("inference_session.stop_server_on_port", return_value=True) as stop:
            assert session.stop_all() == (True, True, True)
        assert [c.args[0] for c in stop.call_args_list] == [8011, 8012, 8020]


class TestInferenceSessionFactory:
    def test_from_urls_without_summarizer(self):
        from inference_session import InferenceSession
        managed = MagicMock()
        managed.summarizer = None
        with patch("inference_session.ManagedServers") as MS:
            MS.from_yaml.return_value = managed
            s = InferenceSession.from_urls("a", "b", "c", "d", "e")
        assert s.summarizer_url == ""
        assert s.summarizer_model_id == ""

    def test_from_urls_with_summarizer(self):
        from inference_session import InferenceSession
        managed = MagicMock()
        managed.summarizer.base_url = "http://127.0.0.1:8013"
        managed.summarizer.server = {"alias": "Qwen3-4B"}
        with patch("inference_session.ManagedServers") as MS:
            MS.from_yaml.return_value = managed
            s = InferenceSession.from_urls("a", "b", "c", "d", "e")
        assert s.summarizer_url == "http://127.0.0.1:8013"
        assert s.summarizer_model_id == "Qwen3-4B"

    def test_from_urls_preserves_supplied_urls(self):
        from inference_session import InferenceSession
        managed = MagicMock()
        managed.summarizer = None
        with patch("inference_session.ManagedServers") as MS:
            MS.from_yaml.return_value = managed
            s = InferenceSession.from_urls("u5", "u6", "ue", "m5", "m6")
        assert (s.q5_url, s.q6_url, s.embed_url) == ("u5", "u6", "ue")
        assert (s.q5_model_id, s.q6_model_id) == ("m5", "m6")
