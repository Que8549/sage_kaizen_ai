"""
tests/test_router.py

Unit tests for router.py heuristic routing logic.

Covers:
- Empty/force_architect overrides
- Brain selection for FAST vs ARCHITECT inputs
- Score accumulation from depth/code/creative hints
- FAST_HINTS counterweight
- Voice-mode bias
- Search detection (_detect_search)
- Music detection (_detect_music)
- News depth routing
"""
from __future__ import annotations

import pytest
from router import (
    ARCHITECT_THRESHOLD,
    RouteDecision,
    _detect_music,
    _detect_search,
    _is_news_query,
    route,
)


# ---------------------------------------------------------------------------
# Basic routing overrides
# ---------------------------------------------------------------------------

class TestRouteOverrides:
    def test_empty_input_routes_to_fast(self):
        d = route("")
        assert d.brain == "FAST"
        assert d.score == 0

    def test_force_architect_overrides_everything(self):
        d = route("quick summary", force_architect=True)
        assert d.brain == "ARCHITECT"
        assert d.score == 999

    def test_whitespace_only_routes_to_fast(self):
        # Non-empty string, no hints → FAST
        d = route("   ")
        assert d.brain == "FAST"


# ---------------------------------------------------------------------------
# FAST brain — short, simple queries
# ---------------------------------------------------------------------------

class TestFastRouting:
    def test_simple_greeting(self):
        d = route("Hello!")
        assert d.brain == "FAST"

    def test_simple_question(self):
        d = route("What is the capital of France?")
        assert d.brain == "FAST"

    def test_summarize_hint_biases_fast(self):
        # FAST_HINTS contains "summarize" which adds -2 to score
        d = route("Can you summarize this paragraph?")
        assert d.brain == "FAST"

    def test_tl_dr_biases_fast(self):
        d = route("give me a tl;dr")
        assert d.brain == "FAST"

    def test_short_answer_biases_fast(self):
        d = route("short answer only please")
        assert d.brain == "FAST"


# ---------------------------------------------------------------------------
# ARCHITECT brain — deep/complex queries
# ---------------------------------------------------------------------------

class TestArchitectRouting:
    def test_code_hint_routes_architect(self):
        d = route("write me a Python class for rate limiting")
        assert d.brain == "ARCHITECT"
        assert any("code:" in r for r in d.reasons)

    def test_debug_hint_routes_architect(self):
        d = route("help me debug this stack trace")
        assert d.brain == "ARCHITECT"

    def test_very_long_input_routes_architect(self):
        # VERY_LONG_INPUT = 2000 chars → score +4
        long_text = "word " * 500  # ~2500 chars
        d = route(long_text)
        assert d.brain == "ARCHITECT"
        assert "very_long_input" in d.reasons

    def test_depth_hint_routes_architect_when_stacked(self):
        # Single depth hint +2 and code hint +3 → score 5 ≥ threshold
        d = route("explain the architecture and write a Python implementation")
        assert d.brain == "ARCHITECT"

    def test_creative_writing_routes_architect(self):
        d = route("write a short story about a dragon")
        assert d.brain == "ARCHITECT"
        assert any("creative:" in r for r in d.reasons)

    def test_comparison_and_depth_routes_architect(self):
        d = route("compare and analyze the pros and cons of REST vs GraphQL")
        assert d.brain == "ARCHITECT"

    def test_sql_hint_routes_architect(self):
        d = route("write a PostgreSQL query to find duplicates")
        assert d.brain == "ARCHITECT"

    def test_unit_test_hint_routes_architect(self):
        d = route("write a unit test for this function")
        assert d.brain == "ARCHITECT"


# ---------------------------------------------------------------------------
# Score accumulation
# ---------------------------------------------------------------------------

class TestScoreAccumulation:
    def test_long_input_adds_two(self):
        # LONG_INPUT = 800 chars, VERY_LONG_INPUT = 2000 chars
        text = "word " * 170  # ~850 chars — between LONG and VERY_LONG
        d = route(text)
        assert "long_input" in d.reasons

    def test_fast_hint_counterweights_depth_hint(self):
        # depth +2, fast -2 → net 0 → FAST
        d = route("briefly explain what recursion is")
        assert d.brain == "FAST"
        # The score might be 0 or negative
        assert d.score < ARCHITECT_THRESHOLD

    def test_comparison_marker_adds_one(self):
        d = route("cats vs dogs")
        assert "comparison_marker" in d.reasons

    def test_versus_marker_adds_one(self):
        d = route("Python versus JavaScript for backend")
        assert "comparison_marker" in d.reasons

    def test_multipart_marker_adds_one(self):
        d = route("what is the syntax and also give examples")
        assert "multi_part_marker" in d.reasons


# ---------------------------------------------------------------------------
# Voice mode
# ---------------------------------------------------------------------------

class TestVoiceMode:
    def test_voice_mode_short_query_biases_fast(self):
        # "explain" would normally add +2, but voice_mode + short query adds -1
        d = route("explain recursion", voice_mode=True)
        # score: depth=+2, voice_short=-1 → net 1 < 3 → FAST
        assert d.brain == "FAST"
        assert "voice_short_query" in d.reasons

    def test_voice_mode_long_query_still_uses_architect(self):
        # Queries >= VOICE_BIAS_THRESHOLD (150) chars bypass voice bias entirely
        long_query = (
            "write a Python class for managing database connections with "
            "connection pooling, retry logic, and exponential backoff for "
            "a distributed microservices architecture"
        )
        assert len(long_query) >= 150, "query must be >= VOICE_BIAS_THRESHOLD"
        d = route(long_query, voice_mode=True)
        assert d.brain == "ARCHITECT"

    def test_voice_mode_false_does_not_add_bias(self):
        d = route("explain recursion", voice_mode=False)
        assert "voice_short_query" not in d.reasons


# ---------------------------------------------------------------------------
# Search detection
# ---------------------------------------------------------------------------

class TestDetectSearch:
    @pytest.mark.parametrize("query", [
        "what's today's news",
        "search for the latest Python releases",
        "what is the weather right now",
        "current weather in Seattle",
        "latest version of Django",
        "what happened yesterday in Ukraine",
        "bitcoin price today",
        "search the web for AI news",
    ])
    def test_search_queries_detected(self, query):
        needs_search, cats = _detect_search(query.lower())
        assert needs_search is True
        assert len(cats) > 0

    @pytest.mark.parametrize("query", [
        "explain how recursion works",
        "write a Python class",
        "define the concept of entropy",
        "how do I center a div in CSS",
        # Note: queries sharing many tokens with search phrases (e.g. "what is the X")
        # may fuzzy-match "what is the latest" above the 76% threshold — avoid those.
    ])
    def test_non_search_queries_not_detected(self, query):
        needs_search, cats = _detect_search(query.lower())
        assert needs_search is False

    def test_weather_routes_to_general_category(self):
        _, cats = _detect_search("what is the weather today")
        assert "general" in cats

    def test_news_routes_to_news_category(self):
        _, cats = _detect_search("today's top news headlines")
        assert "news" in cats

    def test_route_sets_needs_search_flag(self):
        d = route("what's the weather right now")
        assert d.needs_search is True

    def test_route_non_search_has_false_flag(self):
        d = route("explain recursion clearly")
        assert d.needs_search is False


# ---------------------------------------------------------------------------
# Music detection
# ---------------------------------------------------------------------------

class TestDetectMusic:
    @pytest.mark.parametrize("query", [
        "find songs about heartbreak",
        "play me a song",
        "make a playlist for a road trip",
        "find more like this track",
        "song that says California",
        "instrumental songs only",
        "songs with bpm over 120",
        "find songs in the key of C",
    ])
    def test_music_queries_detected(self, query):
        assert _detect_music(query.lower()) is True

    @pytest.mark.parametrize("query", [
        "explain how music theory works",
        "what is the history of jazz",
        "who invented the piano",
    ])
    def test_non_music_queries_not_detected(self, query):
        assert _detect_music(query.lower()) is False

    def test_route_sets_needs_music_flag(self):
        d = route("find songs about the ocean")
        assert d.needs_music is True

    def test_route_non_music_has_false_flag(self):
        d = route("who invented the piano?")
        assert d.needs_music is False


# ---------------------------------------------------------------------------
# News depth routing
# ---------------------------------------------------------------------------

class TestNewsDepthRouting:
    def test_news_with_depth_hint_routes_architect(self):
        # "detailed" (depth hint) + "news" (news signal) → +3 news boost
        d = route("give me a detailed summary of this week's news")
        assert d.brain == "ARCHITECT"
        assert "news:depth_query" in d.reasons

    def test_simple_news_without_depth_routes_fast(self):
        # "today's top news" — news signal but no _NEWS_DEPTH_HINTS → no +3 boost
        d = route("what's today's top news?")
        assert d.brain == "FAST"
        assert "news:depth_query" not in d.reasons

    def test_is_news_query_true_for_news_terms(self):
        assert _is_news_query("latest news about the election") is True
        assert _is_news_query("today's headlines") is True

    def test_is_news_query_false_for_non_news(self):
        assert _is_news_query("how do I sort a list in python") is False


# ---------------------------------------------------------------------------
# RouteDecision dataclass
# ---------------------------------------------------------------------------

class TestRouteDecision:
    def test_defaults(self):
        d = RouteDecision(brain="FAST", reasons=[], score=0)
        assert d.needs_search is False
        assert d.needs_music is False
        assert d.search_categories == ()
        assert d.modality == "text"

    def test_frozen(self):
        d = RouteDecision(brain="FAST", reasons=[], score=0)
        with pytest.raises((AttributeError, TypeError)):
            d.brain = "ARCHITECT"  # type: ignore[misc]
