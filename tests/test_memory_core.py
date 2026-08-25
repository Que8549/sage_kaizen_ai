"""
tests/test_memory_core.py

Unit tests for the pure-logic core of the memory package:
  policy.py         — selective write, promotion thresholds, decay/pruning
  ranker.py         — RRF fusion, multi-signal scoring, contradictions, dedupe
  bundle_builder.py — per-brain token budget and prompt formatting
  embedder.py       — BGE-M3 client singleton
  db.py             — pool helpers and pgvector literal formatting

These modules have no I/O of their own, so they are tested directly rather
than through mocks.
"""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from memory import bundle_builder as bb
from memory import db as mdb
from memory import policy, ranker
from memory.models import EpisodeMemoryItem, ProfileMemoryItem, RuleMemoryItem
from memory.schemas import EpisodeRow


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def episode_row(
    id="e1", summary="a summary", importance=0.5, confidence=0.6,
    correction=False, preference=False, contradiction_group=None,
    scope="project", age_days=0,
) -> EpisodeRow:
    return EpisodeRow(
        id=id, user_id="u", project_id="p", workspace_id=None, session_id=None,
        scope=scope, event_type="general", intent_label=None,
        summary_text=summary, raw_excerpt=None, tags=[],
        importance=importance, confidence=confidence, sentiment=None,
        was_user_correction=correction, was_explicit_preference=preference,
        contradiction_group=contradiction_group, embedding=None,
        created_at=datetime.now(tz=timezone.utc) - timedelta(days=age_days),
        last_accessed_at=None, last_retrieved_at=None, expires_at=None,
    )


def profile_item(key="tone", value="concise") -> ProfileMemoryItem:
    return ProfileMemoryItem(
        id="p1", profile_type="preference", key=key, value_text=value,
        confidence=0.9, scope="user", is_pinned=False, source_type="explicit",
    )


def rule_item(text="always use --log-file", locked=False) -> RuleMemoryItem:
    return RuleMemoryItem(
        id="r1", rule_kind="invariant", rule_text=text, confidence=0.95,
        is_locked=locked, review_status="approved",
    )


def episode_item(summary="did a thing", correction=False, score=0.5) -> EpisodeMemoryItem:
    return EpisodeMemoryItem(
        id="e1", event_type="general", summary_text=summary, tags=[],
        importance=0.5, confidence=0.6, was_user_correction=correction,
        was_explicit_preference=False, created_at=datetime.now(tz=timezone.utc),
        retrieval_score=score,
    )


# ---------------------------------------------------------------------------
# policy.should_write_episode
# ---------------------------------------------------------------------------

class TestShouldWriteEpisode:
    @pytest.mark.parametrize(
        "event", ["greeting", "acknowledgement", "trivial", "clarification"]
    )
    def test_never_write_events_are_dropped(self, event):
        assert policy.should_write_episode(event, "x" * 5000, "y" * 5000) is False

    @pytest.mark.parametrize(
        "event",
        ["correction", "preference", "decision", "approval",
         "architecture_choice", "model_selection", "code_change", "bug_report"],
    )
    def test_always_write_events_are_kept(self, event):
        assert policy.should_write_episode(event, "hi", "ok") is True

    def test_user_correction_always_wins(self):
        assert policy.should_write_episode(
            "general", "no", "sorry", was_user_correction=True
        ) is True

    def test_explicit_preference_always_wins(self):
        assert policy.should_write_episode(
            "general", "I prefer X", "noted", was_explicit_preference=True
        ) is True

    def test_never_write_beats_correction_flag(self):
        """The never-list is checked first — order matters."""
        assert policy.should_write_episode(
            "greeting", "hello", "hi", was_user_correction=True
        ) is False

    def test_short_turn_is_dropped(self):
        assert policy.should_write_episode("general", "hi", "hello") is False

    def test_long_important_turn_is_written(self):
        text = "x" * 400
        assert policy.should_write_episode(
            "general", text, text, estimated_importance=0.5
        ) is True

    def test_long_unimportant_turn_is_dropped(self):
        text = "x" * 400
        assert policy.should_write_episode(
            "general", text, text, estimated_importance=0.1
        ) is False

    def test_token_estimate_is_the_combined_length(self):
        # 200 tokens * 3.5 chars = 700 chars combined
        assert policy.should_write_episode("general", "x" * 350, "y" * 350) is True
        assert policy.should_write_episode("general", "x" * 100, "y" * 100) is False


# ---------------------------------------------------------------------------
# policy.check_rule_promotion
# ---------------------------------------------------------------------------

class TestCheckRulePromotion:
    def test_below_confidence_threshold_is_rejected(self):
        assert policy.check_rule_promotion("r", "invariant", 0.5, "because") is None

    def test_at_threshold_is_accepted(self):
        d = policy.check_rule_promotion(
            "r", "invariant", policy.RULE_PROMOTE_CONFIDENCE, "because"
        )
        assert d is not None

    def test_blank_rule_text_is_rejected(self):
        assert policy.check_rule_promotion("   ", "invariant", 0.99, "because") is None

    def test_decision_is_not_pre_approved(self):
        d = policy.check_rule_promotion("r", "invariant", 0.99, "because")
        assert d is not None and d.approved is False

    def test_fields_are_carried_through(self):
        d = policy.check_rule_promotion(
            "always X", "invariant", 0.95, "rationale here",
            source_memory_id="m1", scope="user",
        )
        assert d is not None
        assert d.rule_text == "always X"
        assert d.rule_kind == "invariant"
        assert d.source_memory_id == "m1"
        assert d.scope == "user"


# ---------------------------------------------------------------------------
# policy decay + pruning
# ---------------------------------------------------------------------------

class TestComputeDecayWeight:
    def test_zero_age_returns_importance(self):
        assert policy.compute_decay_weight(0, 0.7) == pytest.approx(0.7)

    def test_decays_monotonically_with_age(self):
        vals = [policy.compute_decay_weight(d, 0.5) for d in (0, 30, 60, 120)]
        assert vals == sorted(vals, reverse=True)

    @pytest.mark.parametrize(
        "importance,half_life", [(0.9, 180.0), (0.7, 90.0), (0.5, 60.0), (0.2, 30.0)]
    )
    def test_half_life_bands(self, importance, half_life):
        """At the band's half-life the weight should be ~half the importance."""
        got = policy.compute_decay_weight(int(half_life), importance)
        assert got == pytest.approx(importance / 2, rel=0.01)

    def test_high_importance_decays_slower_than_low(self):
        # Compare at equal age, normalised by starting importance.
        hi = policy.compute_decay_weight(90, 0.9) / 0.9
        lo = policy.compute_decay_weight(90, 0.2) / 0.2
        assert hi > lo


class TestShouldPruneEpisode:
    def test_corrections_are_never_pruned(self):
        assert policy.should_prune_episode(
            episode_row(correction=True, importance=0.01), age_days=10_000
        ) is False

    def test_preferences_are_never_pruned(self):
        assert policy.should_prune_episode(
            episode_row(preference=True, importance=0.01), age_days=10_000
        ) is False

    def test_high_importance_is_never_pruned(self):
        assert policy.should_prune_episode(
            episode_row(importance=0.8), age_days=10_000
        ) is False

    def test_fresh_ordinary_episode_is_kept(self):
        assert policy.should_prune_episode(episode_row(importance=0.5), age_days=1) is False

    def test_fully_decayed_episode_is_pruned(self):
        assert policy.should_prune_episode(episode_row(importance=0.2), age_days=365) is True

    def test_very_old_low_importance_is_pruned(self):
        assert policy.should_prune_episode(episode_row(importance=0.25), age_days=400) is True


# ---------------------------------------------------------------------------
# ranker.rrf_fuse
# ---------------------------------------------------------------------------

class TestRrfFuse:
    def test_empty_inputs(self):
        assert ranker.rrf_fuse([], []) == []

    def test_single_list_passthrough(self):
        rows = [(episode_row(id="a"), 0.9), (episode_row(id="b"), 0.8)]
        out = ranker.rrf_fuse(rows, [])
        assert [r.id for r, _ in out] == ["a", "b"]

    def test_documents_in_both_lists_rank_highest(self):
        a, b, c = episode_row(id="a"), episode_row(id="b"), episode_row(id="c")
        lexical = [(a, 1.0), (b, 0.9)]
        vector = [(c, 1.0), (a, 0.9)]
        out = ranker.rrf_fuse(lexical, vector)
        assert out[0][0].id == "a"   # appears in both

    def test_result_is_sorted_descending(self):
        a, b = episode_row(id="a"), episode_row(id="b")
        out = ranker.rrf_fuse([(a, 1.0), (b, 0.5)], [(a, 1.0)])
        scores = [s for _, s in out]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_ids_in_output(self):
        a = episode_row(id="a")
        out = ranker.rrf_fuse([(a, 1.0)], [(a, 1.0)])
        assert len(out) == 1

    def test_uses_the_documented_rrf_constant(self):
        a = episode_row(id="a")
        out = ranker.rrf_fuse([(a, 1.0)], [])
        assert out[0][1] == pytest.approx(1.0 / (ranker._RRF_K + 1))


# ---------------------------------------------------------------------------
# ranker.score_episodes
# ---------------------------------------------------------------------------

class TestScoreEpisodes:
    def test_empty_input(self):
        assert ranker.score_episodes([]) == []

    def test_scores_are_bounded(self):
        rows = [(episode_row(id=str(i), importance=1.0, confidence=1.0), 1.0)
                for i in range(3)]
        assert all(0.0 <= s <= 1.0 for _, s in ranker.score_episodes(rows))

    def test_recent_beats_old_all_else_equal(self):
        new = episode_row(id="new", age_days=0)
        old = episode_row(id="old", age_days=170)
        out = ranker.score_episodes([(new, 1.0), (old, 1.0)])
        assert out[0][0].id == "new"

    def test_important_beats_unimportant(self):
        hi = episode_row(id="hi", importance=1.0)
        lo = episode_row(id="lo", importance=0.0)
        out = ranker.score_episodes([(hi, 1.0), (lo, 1.0)])
        assert out[0][0].id == "hi"

    def test_correction_gets_a_bonus(self):
        plain = episode_row(id="plain")
        corr = episode_row(id="corr", correction=True)
        out = dict((r.id, s) for r, s in ranker.score_episodes([(plain, 1.0), (corr, 1.0)]))
        assert out["corr"] > out["plain"]

    def test_preference_gets_a_smaller_bonus_than_correction(self):
        plain = episode_row(id="plain")
        pref = episode_row(id="pref", preference=True)
        corr = episode_row(id="corr", correction=True)
        out = dict((r.id, s) for r, s in
                   ranker.score_episodes([(plain, 1.0), (pref, 1.0), (corr, 1.0)]))
        assert out["plain"] < out["pref"] < out["corr"]

    def test_scope_match_is_rewarded(self):
        match = episode_row(id="m", scope="project")
        other = episode_row(id="o", scope="user")
        out = dict((r.id, s) for r, s in
                   ranker.score_episodes([(match, 1.0), (other, 1.0)], scope_filter="project"))
        assert out["m"] > out["o"]

    def test_no_scope_filter_treats_all_as_matching(self):
        a = episode_row(id="a", scope="project")
        b = episode_row(id="b", scope="user")
        out = dict((r.id, s) for r, s in ranker.score_episodes([(a, 1.0), (b, 1.0)]))
        assert out["a"] == pytest.approx(out["b"])

    def test_output_is_sorted(self):
        rows = [(episode_row(id="a", importance=0.1), 1.0),
                (episode_row(id="b", importance=0.9), 1.0)]
        scores = [s for _, s in ranker.score_episodes(rows)]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# ranker.filter_contradictions / deduplicate
# ---------------------------------------------------------------------------

class TestFilterContradictions:
    def test_no_groups_passes_everything(self):
        rows = [(episode_row(id="a"), 0.9), (episode_row(id="b"), 0.8)]
        assert len(ranker.filter_contradictions(rows)) == 2

    def test_highest_scored_in_a_group_wins(self):
        a = episode_row(id="a", contradiction_group="g1")
        b = episode_row(id="b", contradiction_group="g1")
        out = ranker.filter_contradictions([(a, 0.9), (b, 0.8)])
        assert [r.id for r, _ in out] == ["a"]

    def test_different_groups_both_survive(self):
        a = episode_row(id="a", contradiction_group="g1")
        b = episode_row(id="b", contradiction_group="g2")
        assert len(ranker.filter_contradictions([(a, 0.9), (b, 0.8)])) == 2

    def test_ungrouped_items_are_never_suppressed(self):
        a = episode_row(id="a", contradiction_group="g1")
        b = episode_row(id="b", contradiction_group="g1")
        c = episode_row(id="c")
        out = ranker.filter_contradictions([(a, 0.9), (b, 0.8), (c, 0.7)])
        assert [r.id for r, _ in out] == ["a", "c"]

    def test_empty_input(self):
        assert ranker.filter_contradictions([]) == []


class TestDeduplicate:
    def test_identical_summaries_are_collapsed(self):
        a = episode_row(id="a", summary="the user prefers concise answers")
        b = episode_row(id="b", summary="the user prefers concise answers")
        out = ranker.deduplicate([(a, 0.9), (b, 0.8)])
        assert [r.id for r, _ in out] == ["a"]

    def test_distinct_summaries_both_survive(self):
        a = episode_row(id="a", summary="alpha beta gamma delta")
        b = episode_row(id="b", summary="completely different words entirely")
        assert len(ranker.deduplicate([(a, 0.9), (b, 0.8)])) == 2

    def test_threshold_is_configurable(self):
        a = episode_row(id="a", summary="one two three four")
        b = episode_row(id="b", summary="one two three five")
        assert len(ranker.deduplicate([(a, 0.9), (b, 0.8)], similarity_threshold=0.5)) == 1
        assert len(ranker.deduplicate([(a, 0.9), (b, 0.8)], similarity_threshold=0.99)) == 2

    def test_empty_summaries_do_not_crash(self):
        a = episode_row(id="a", summary="")
        b = episode_row(id="b", summary="")
        ranker.deduplicate([(a, 0.9), (b, 0.8)])

    def test_is_case_insensitive(self):
        a = episode_row(id="a", summary="Alpha Beta Gamma")
        b = episode_row(id="b", summary="alpha beta gamma")
        assert len(ranker.deduplicate([(a, 0.9), (b, 0.8)])) == 1

    def test_empty_input(self):
        assert ranker.deduplicate([]) == []


# ---------------------------------------------------------------------------
# bundle_builder
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_never_returns_zero(self):
        assert bb._estimate_tokens("") == 1

    def test_scales_with_length(self):
        assert bb._estimate_tokens("x" * 350) == 100

    def test_uses_the_documented_ratio(self):
        assert bb._CHARS_PER_TOKEN == 3.5


class TestBuildBundle:
    def test_empty_inputs_produce_an_empty_bundle(self):
        b = bb.build_bundle([], [], [], max_tokens=600)
        assert b.total_items == 0
        assert b.was_truncated is False

    def test_everything_fits_under_a_generous_cap(self):
        b = bb.build_bundle([profile_item()], [rule_item()], [episode_item()], 1500)
        assert b.total_items == 3
        assert b.was_truncated is False

    def test_profiles_get_first_claim_on_the_budget(self):
        """Profiles are considered before rules, which come before episodes."""
        profiles = [profile_item(key=f"k{i}", value="v" * 200) for i in range(5)]
        b = bb.build_bundle(profiles, [rule_item()], [episode_item()], max_tokens=200)
        assert len(b.profiles) >= 1
        assert b.was_truncated is True

    def test_fill_is_skip_and_continue_not_stop_at_first_overflow(self):
        """
        A large item that doesn't fit is skipped, and smaller later items can
        still be packed into the remaining budget.

        Worth pinning explicitly: the docstring's "truncates episodes first,
        then rules, then profiles" reads as strict priority truncation, but the
        implementation keeps going after a miss. Packing more in is the better
        behaviour; the wording is just looser than the code.
        """
        huge_profile = profile_item(key="big", value="v" * 5_000)
        tiny_episode = episode_item(summary="ok")
        b = bb.build_bundle([huge_profile], [], [tiny_episode], max_tokens=200)
        assert b.profiles == []          # didn't fit
        assert len(b.episodes) == 1      # still packed
        assert b.was_truncated is True

    def test_truncation_is_flagged(self):
        episodes = [episode_item(summary="s" * 500) for _ in range(20)]
        b = bb.build_bundle([], [], episodes, max_tokens=100)
        assert b.was_truncated is True

    def test_header_overhead_is_accounted_for(self):
        b = bb.build_bundle([], [], [], max_tokens=600)
        assert b.estimated_tokens == bb._HEADER_TOKENS

    def test_estimated_tokens_never_exceeds_the_cap(self):
        items = [episode_item(summary="word " * 50) for _ in range(50)]
        b = bb.build_bundle([], [], items, max_tokens=300)
        assert b.estimated_tokens <= 300

    def test_fast_and_architect_budgets_differ_in_effect(self):
        items = [episode_item(summary="word " * 30) for _ in range(40)]
        fast = bb.build_bundle([], [], items, max_tokens=600)
        arch = bb.build_bundle([], [], items, max_tokens=1500)
        assert len(arch.episodes) > len(fast.episodes)

    def test_total_items_matches_the_kept_lists(self):
        b = bb.build_bundle([profile_item()], [rule_item()], [episode_item()], 1500)
        assert b.total_items == len(b.profiles) + len(b.rules) + len(b.episodes)


class TestFormatBundlePrompt:
    def test_empty_bundle_yields_empty_string(self):
        assert bb.format_bundle_prompt(bb.build_bundle([], [], [], 600)) == ""

    def test_profile_section(self):
        out = bb.format_bundle_prompt(bb.build_bundle([profile_item("tone", "concise")], [], [], 600))
        assert "[USER PROFILE]" in out
        assert "- tone: concise" in out

    def test_rules_section(self):
        out = bb.format_bundle_prompt(bb.build_bundle([], [rule_item("always X")], [], 600))
        assert "[PROJECT NORMS]" in out
        assert "- always X" in out

    def test_locked_rule_is_marked(self):
        out = bb.format_bundle_prompt(bb.build_bundle([], [rule_item("X", locked=True)], [], 600))
        assert "[LOCKED]" in out

    def test_episodes_section(self):
        out = bb.format_bundle_prompt(bb.build_bundle([], [], [episode_item("did X")], 600))
        assert "[RELEVANT PRIOR EPISODES]" in out
        assert "did X" in out

    def test_correction_is_marked(self):
        out = bb.format_bundle_prompt(
            bb.build_bundle([], [], [episode_item("no, use Y", correction=True)], 600)
        )
        assert "[USER CORRECTION]" in out

    def test_header_tells_the_model_to_prefer_current_instructions(self):
        """Prompt-injection-adjacent: prior memory must not override the user."""
        out = bb.format_bundle_prompt(bb.build_bundle([profile_item()], [], [], 600))
        assert "Prefer current user instructions over past memory" in out

    def test_absent_sections_are_omitted(self):
        out = bb.format_bundle_prompt(bb.build_bundle([profile_item()], [], [], 600))
        assert "[PROJECT NORMS]" not in out
        assert "[RELEVANT PRIOR EPISODES]" not in out


# ---------------------------------------------------------------------------
# memory.db helpers
# ---------------------------------------------------------------------------

class TestVecStr:
    def test_formats_a_pgvector_literal(self):
        assert mdb.vec_str([1.0, 2.0, 3.0]).startswith("[")
        assert mdb.vec_str([1.0, 2.0, 3.0]).endswith("]")

    def test_uses_dots_not_locale_separators(self):
        """'.17g' avoids comma decimal separators on e.g. LC_NUMERIC=de_DE."""
        assert "," not in mdb.vec_str([1.5]).replace("[", "").replace("]", "")

    def test_preserves_precision(self):
        v = 0.12345678901234567
        assert str(v)[:10] in mdb.vec_str([v])

    def test_empty_vector(self):
        assert mdb.vec_str([]) == "[]"

    def test_element_count_is_preserved(self):
        assert mdb.vec_str([0.1] * 1024).count(",") == 1023


class TestDbMisc:
    def test_new_uuid_is_unique(self):
        assert mdb.new_uuid() != mdb.new_uuid()

    def test_new_uuid_is_a_valid_uuid_string(self):
        import uuid
        uuid.UUID(mdb.new_uuid())

    def test_now_utc_is_timezone_aware(self):
        assert mdb.now_utc().tzinfo is not None

    def test_dumps_produces_json(self):
        import json
        assert json.loads(mdb.dumps({"a": 1})) == {"a": 1}


class TestGetPool:
    def test_pool_is_created_once(self):
        with (
            patch.object(mdb, "_pool", None),
            patch.object(mdb, "ConnectionPool") as CP,
            patch.object(mdb, "PgSettings") as PS,
        ):
            PS.return_value.pg_dsn = "postgresql://x"
            a = mdb.get_pool()
            b = mdb.get_pool()
        assert a is b
        CP.assert_called_once()

    def test_pool_is_created_once_under_concurrency(self):
        built: list[int] = []

        def _slow(*a, **kw):
            built.append(1)
            time.sleep(0.02)
            return MagicMock()

        with (
            patch.object(mdb, "_pool", None),
            patch.object(mdb, "ConnectionPool", side_effect=_slow),
            patch.object(mdb, "PgSettings") as PS,
        ):
            PS.return_value.pg_dsn = "postgresql://x"
            threads = [threading.Thread(target=mdb.get_pool) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(built) == 1

    def test_get_connection_yields_from_the_pool(self):
        pool = MagicMock()
        conn = MagicMock()
        pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
        pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(mdb, "get_pool", return_value=pool):
            with mdb.get_connection() as c:
                assert c is conn


# ---------------------------------------------------------------------------
# memory.embedder
# ---------------------------------------------------------------------------

from memory import embedder as emb


class TestEmbedder:
    @pytest.fixture(autouse=True)
    def _reset(self):
        emb._get_client.reset()
        yield
        emb._get_client.reset()

    def test_empty_input_short_circuits(self):
        with patch.object(emb, "_get_client") as gc:
            assert emb.embed_texts([]) == []
        gc.assert_not_called()

    def test_delegates_to_the_shared_client(self):
        client = MagicMock()
        client.embed.return_value = [[0.1] * 1024]
        with patch.object(emb, "EmbedClient", return_value=client):
            assert emb.embed_texts(["hello"]) == [[0.1] * 1024]

    def test_embed_one_unwraps_the_batch(self):
        client = MagicMock()
        client.embed.return_value = [[0.2] * 1024]
        with patch.object(emb, "EmbedClient", return_value=client):
            assert emb.embed_one("hello") == [0.2] * 1024

    def test_client_is_a_singleton(self):
        with patch.object(emb, "EmbedClient", return_value=MagicMock()) as EC:
            emb._get_client()
            emb._get_client()
        EC.assert_called_once()

    def test_client_is_built_once_under_concurrency(self):
        built: list[int] = []

        def _slow(*a, **kw):
            built.append(1)
            time.sleep(0.02)
            return MagicMock()

        with patch.object(emb, "EmbedClient", side_effect=_slow):
            threads = [threading.Thread(target=emb._get_client) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(built) == 1

    def test_points_at_the_bge_m3_service_port(self):
        """BGE-M3 lives on 8020 per the CLAUDE.md service inventory."""
        assert emb._BGE_BASE_URL.endswith(":8020")
        assert emb._DIMS == 1024
