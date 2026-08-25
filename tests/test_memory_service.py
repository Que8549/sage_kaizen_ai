"""
tests/test_memory_service.py

Unit tests for memory/service.py — MemoryService facade.

Key behaviors under test:
1. get_memory_bundle uses the module-level _POOL (not a per-call executor).
2. timeout=2.0 on each .result() call: a stalled retrieve function returns [] gracefully.
3. Partial failures: one retrieve failing returns [] for that source but not others.
4. The returned MemoryBundle reflects what was retrieved.
"""
from __future__ import annotations

import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from memory.models import MemoryContextRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(**kwargs) -> MemoryContextRequest:
    defaults = dict(
        user_id="alquin",
        project_id="sage_kaizen",
        query_text="hello world",
        route_target="fast",
        max_bundle_tokens=None,
    )
    defaults.update(kwargs)
    return MemoryContextRequest(**defaults)


# ---------------------------------------------------------------------------
# Module-level pool
# ---------------------------------------------------------------------------

class TestModuleLevelPool:
    def test_pool_is_reused_across_calls(self):
        """_POOL should be the same object across multiple get_memory_bundle calls."""
        import memory.service as svc_module
        pool_ids: set[int] = set()

        original_submit = svc_module._POOL.submit

        def tracking_submit(fn, *args, **kwargs):
            pool_ids.add(id(svc_module._POOL))
            f: Future = Future()
            f.set_result([])
            return f

        with patch.object(svc_module._POOL, "submit", side_effect=tracking_submit):
            from memory.service import MemoryService
            svc = MemoryService()
            with patch("memory.service.build_bundle", return_value=MagicMock(
                profiles=[], rules=[], episodes=[],
                estimated_tokens=0, was_truncated=False, total_items=0,
            )):
                svc.get_memory_bundle(_make_request())
                svc.get_memory_bundle(_make_request())

        # Only one pool id should have appeared — the same module-level _POOL
        assert len(pool_ids) == 1


# ---------------------------------------------------------------------------
# Timeout fallback — stalled retrieve functions return []
# ---------------------------------------------------------------------------

class TestTimeoutFallback:
    def _run_with_stalled(self, stall_which: str):
        """
        Replace one of the retrieve_* functions with one that sleeps long enough
        to trigger the 2-second timeout, then verify get_memory_bundle returns
        gracefully with an empty list for that source.
        """
        import memory.service as svc_module

        sentinel_profiles = [MagicMock()]
        sentinel_rules    = [MagicMock()]
        sentinel_episodes = [MagicMock()]

        def slow_fn(*args, **kwargs):
            time.sleep(5)  # Far exceeds the 2.0 s timeout
            return []

        def fast_profiles(*args, **kwargs):
            return sentinel_profiles

        def fast_rules(*args, **kwargs):
            return sentinel_rules

        def fast_episodes(*args, **kwargs):
            return sentinel_episodes

        patches = {
            "retrieve_profiles": fast_profiles,
            "retrieve_rules":    fast_rules,
            "retrieve_episodes": fast_episodes,
        }
        patches[stall_which] = slow_fn

        captured: dict = {}

        def fake_build_bundle(profiles, rules, episodes, max_tokens):
            captured["profiles"] = profiles
            captured["rules"]    = rules
            captured["episodes"] = episodes
            return MagicMock(
                profiles=profiles, rules=rules, episodes=episodes,
                estimated_tokens=0, was_truncated=False, total_items=0,
            )

        with (
            patch("memory.service.retrieve_profiles", patches["retrieve_profiles"]),
            patch("memory.service.retrieve_rules",    patches["retrieve_rules"]),
            patch("memory.service.retrieve_episodes", patches["retrieve_episodes"]),
            patch("memory.service.build_bundle",      fake_build_bundle),
        ):
            from memory.service import MemoryService
            svc = MemoryService()
            svc.get_memory_bundle(_make_request())

        return captured

    @pytest.mark.slow
    def test_stalled_profiles_returns_empty_list(self):
        captured = self._run_with_stalled("retrieve_profiles")
        assert captured["profiles"] == []
        # Other sources still returned their values
        assert len(captured["rules"]) > 0
        assert len(captured["episodes"]) > 0

    @pytest.mark.slow
    def test_stalled_rules_returns_empty_list(self):
        captured = self._run_with_stalled("retrieve_rules")
        assert captured["rules"] == []
        assert len(captured["profiles"]) > 0
        assert len(captured["episodes"]) > 0

    @pytest.mark.slow
    def test_stalled_episodes_returns_empty_list(self):
        captured = self._run_with_stalled("retrieve_episodes")
        assert captured["episodes"] == []
        assert len(captured["profiles"]) > 0
        assert len(captured["rules"]) > 0


# ---------------------------------------------------------------------------
# Normal path — all three sources return data
# ---------------------------------------------------------------------------

class TestNormalPath:
    def test_bundle_contains_all_sources(self):
        profiles_data = [MagicMock()]
        rules_data    = [MagicMock(), MagicMock()]
        episodes_data = [MagicMock()]

        with (
            patch("memory.service.retrieve_profiles", return_value=profiles_data),
            patch("memory.service.retrieve_rules",    return_value=rules_data),
            patch("memory.service.retrieve_episodes", return_value=episodes_data),
            patch("memory.service.build_bundle") as mock_build,
        ):
            mock_build.return_value = MagicMock(
                profiles=profiles_data, rules=rules_data, episodes=episodes_data,
                estimated_tokens=100, was_truncated=False, total_items=4,
            )
            from memory.service import MemoryService
            svc = MemoryService()
            bundle = svc.get_memory_bundle(_make_request())

        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args
        assert call_kwargs.kwargs["profiles"] is profiles_data
        assert call_kwargs.kwargs["rules"]    is rules_data
        assert call_kwargs.kwargs["episodes"] is episodes_data

    def test_architect_route_uses_higher_token_budget(self):
        with (
            patch("memory.service.retrieve_profiles", return_value=[]),
            patch("memory.service.retrieve_rules",    return_value=[]),
            patch("memory.service.retrieve_episodes", return_value=[]),
            patch("memory.service.build_bundle") as mock_build,
        ):
            mock_build.return_value = MagicMock(
                profiles=[], rules=[], episodes=[],
                estimated_tokens=0, was_truncated=False, total_items=0,
            )
            from memory.service import MemoryService, _ARCHITECT_MAX_TOKENS, _FAST_MAX_TOKENS
            svc = MemoryService()
            svc.get_memory_bundle(_make_request(route_target="architect"))
            architect_tokens = mock_build.call_args.kwargs["max_tokens"]
            mock_build.reset_mock()

            svc.get_memory_bundle(_make_request(route_target="fast"))
            fast_tokens = mock_build.call_args.kwargs["max_tokens"]

        assert architect_tokens == _ARCHITECT_MAX_TOKENS
        assert fast_tokens == _FAST_MAX_TOKENS
        assert architect_tokens > fast_tokens

    def test_custom_max_tokens_overrides_brain_default(self):
        with (
            patch("memory.service.retrieve_profiles", return_value=[]),
            patch("memory.service.retrieve_rules",    return_value=[]),
            patch("memory.service.retrieve_episodes", return_value=[]),
            patch("memory.service.build_bundle") as mock_build,
        ):
            mock_build.return_value = MagicMock(
                profiles=[], rules=[], episodes=[],
                estimated_tokens=0, was_truncated=False, total_items=0,
            )
            from memory.service import MemoryService
            svc = MemoryService()
            svc.get_memory_bundle(_make_request(max_bundle_tokens=999))
            assert mock_build.call_args.kwargs["max_tokens"] == 999


# ---------------------------------------------------------------------------
# Exception in retrieve function (not timeout) — also returns []
# ---------------------------------------------------------------------------

class TestExceptionFallback:
    def test_exception_in_profiles_returns_empty(self):
        def boom(*a, **k):
            raise RuntimeError("DB down")

        with (
            patch("memory.service.retrieve_profiles", boom),
            patch("memory.service.retrieve_rules",    return_value=[]),
            patch("memory.service.retrieve_episodes", return_value=[]),
            patch("memory.service.build_bundle") as mock_build,
        ):
            mock_build.return_value = MagicMock(
                profiles=[], rules=[], episodes=[],
                estimated_tokens=0, was_truncated=False, total_items=0,
            )
            from memory.service import MemoryService
            svc = MemoryService()
            # Must not raise — degrades gracefully
            bundle = svc.get_memory_bundle(_make_request())

        assert mock_build.call_args.kwargs["profiles"] == []
