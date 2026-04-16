"""
news/tests/test_news_resolver.py

Unit tests for NewsResolver.

Tests DB-first resolution (fresh brief → returns NewsContext with source="db_brief"),
stale fallback (no fresh brief → returns stale brief with is_stale=True),
and None return when no brief exists.

DB calls are mocked so this test runs without a live PostgreSQL connection.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from news.retrieval.news_resolver import NewsContext, NewsResolver


def _make_brief_row(source: str = "db_brief") -> dict:
    return {
        "brief_id":        "00000000-0000-0000-0000-000000000001",
        "brief_kind":      "daily",
        "headline_summary": "Top story today",
        "summary_short":   "Short summary.",
        "summary_long":    "Long summary goes here.",
        "freshness_at":    datetime.now(timezone.utc),
        "profile_name":    "general_brief",
    }


@pytest.fixture
def resolver():
    with patch("news.retrieval.news_resolver.get_news_settings") as mock_cfg:
        mock_cfg.return_value = MagicMock(
            pg_dsn="postgresql://test/test",
            brief_freshness_hours=4,
        )
        r = NewsResolver()
        r._dsn = "postgresql://test/test"
        yield r


# ---------------------------------------------------------------------------
# Fresh brief path
# ---------------------------------------------------------------------------

def test_fresh_brief_returns_db_context(resolver):
    fresh_row = _make_brief_row()
    with patch("news.retrieval.news_resolver.conn_ctx") as mock_ctx:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = fresh_row
        mock_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = resolver._resolve_news_brief("daily")

    assert result is not None
    assert result.source == "db_brief"
    assert result.is_stale is False
    assert "Top story today" in result.content


def test_fresh_brief_xml_block_format(resolver):
    fresh_row = _make_brief_row()
    with patch("news.retrieval.news_resolver.conn_ctx") as mock_ctx:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = fresh_row
        mock_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = resolver._resolve_news_brief("daily")

    xml = result.to_xml_block()
    assert xml.startswith('<news_context source="db_brief"')
    assert "</news_context>" in xml


# ---------------------------------------------------------------------------
# Stale fallback path
# ---------------------------------------------------------------------------

def test_stale_fallback_when_no_fresh(resolver):
    stale_row = _make_brief_row()
    call_count = 0

    def fake_fetchone():
        nonlocal call_count
        call_count += 1
        # First call (fresh brief check) → None
        # Second call (running check) → None
        # Third call (any brief) → stale_row
        if call_count in (1, 2):
            return None
        return stale_row

    with patch("news.retrieval.news_resolver.conn_ctx") as mock_ctx:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.side_effect = fake_fetchone
        mock_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = resolver._resolve_news_brief("daily")

    assert result is not None
    assert result.is_stale is True
    assert result.source == "stale"


def test_none_when_no_brief_at_all(resolver):
    with patch("news.retrieval.news_resolver.conn_ctx") as mock_ctx:
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        mock_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = resolver._resolve_news_brief("daily")

    assert result is None


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def test_news_intent_detection(resolver):
    positives = [
        "what's in the news today",
        "give me the latest news",
        "morning briefing please",
        "top stories",
        "news summary",
    ]
    for q in positives:
        assert resolver._is_news_query(q.lower()), f"Should detect news intent: {q!r}"


def test_non_news_queries_not_detected(resolver):
    negatives = [
        "how do I bake bread",
        "tell me a joke",
        "what time is it",
        "write me a poem",
    ]
    for q in negatives:
        assert not resolver._is_news_query(q.lower()), f"Should NOT detect news intent: {q!r}"


def test_market_intent_detection(resolver):
    positives = [
        "what is the stock price of nvidia",
        "bitcoin price right now",
        "how has AAPL performed this week",
    ]
    for q in positives:
        assert resolver._is_market_query(q.lower()), f"Should detect market intent: {q!r}"


# ---------------------------------------------------------------------------
# Non-news query → None
# ---------------------------------------------------------------------------

def test_resolve_returns_none_for_non_news(resolver):
    result = resolver.resolve("can you help me debug this Python function?")
    assert result is None
