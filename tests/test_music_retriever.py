"""
tests/test_music_retriever.py

Unit tests for rag_v1/media/music_retriever.py.

Key behaviors under test:
1. conn.close() is NOT called after any query (conn_ctx manages lifecycle).
2. search_by_mood uses conn_ctx and returns MusicResult list.
3. find_similar uses conn_ctx; CLAP fallback on no title/path match.
4. filter_by_attributes uses conn_ctx and returns results.
5. get_similar_cluster uses conn_ctx and returns results.
6. detect_intent classifies query intent correctly.
7. All public methods return [] gracefully on DB failure.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest
from rag_v1.media.music_retriever import (
    MusicIntent,
    MusicRetriever,
    detect_intent,
    format_music_context,
)


DSN = "postgresql://localhost/test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_clap():
    clap = MagicMock()
    clap.embed_text.return_value = [[0.1] * 512]
    return clap


@pytest.fixture
def retriever(mock_clap):
    r = MusicRetriever(pg_dsn=DSN)
    r._clap = mock_clap
    return r


def _make_db_row(**kwargs) -> dict:
    defaults = dict(
        media_id="00000000-0000-0000-0000-000000000001",
        file_path="/music/track.mp3",
        title="Test Track",
        artist="Test Artist",
        bpm=120.0,
        key="C major",
        has_vocals=True,
        is_explicit=False,
        cluster_id=None,
        score=0.85,
    )
    defaults.update(kwargs)
    return defaults


@pytest.fixture
def mock_conn_ctx():
    """Patch conn_ctx to return a mock connection."""
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.execute.return_value = cursor
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None

    conn = MagicMock()
    conn.closed = False
    conn.cursor.return_value = cursor

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)

    with patch("rag_v1.media.music_retriever.conn_ctx", return_value=ctx) as patched:
        yield patched, conn, cursor


# ---------------------------------------------------------------------------
# conn.close() is not called
# ---------------------------------------------------------------------------

class TestNoConnClose:
    def test_search_by_mood_does_not_close_conn(self, retriever, mock_conn_ctx):
        _, conn, _ = mock_conn_ctx
        retriever.search_by_mood("energetic dance music")
        conn.close.assert_not_called()

    def test_find_similar_does_not_close_conn(self, retriever, mock_conn_ctx):
        _, conn, _ = mock_conn_ctx
        retriever.find_similar("California Love")
        conn.close.assert_not_called()

    def test_filter_by_attributes_does_not_close_conn(self, retriever, mock_conn_ctx):
        _, conn, _ = mock_conn_ctx
        retriever.filter_by_attributes(has_vocals=True)
        conn.close.assert_not_called()

    def test_get_similar_cluster_does_not_close_conn(self, retriever, mock_conn_ctx):
        _, conn, _ = mock_conn_ctx
        retriever.get_similar_cluster("California Love")
        conn.close.assert_not_called()


# ---------------------------------------------------------------------------
# Graceful degradation on DB failure
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_search_by_mood_returns_empty_on_db_error(self, retriever):
        with patch("rag_v1.media.music_retriever.conn_ctx", side_effect=RuntimeError("DB down")):
            result = retriever.search_by_mood("energetic dance")
        assert result == []

    def test_find_similar_returns_empty_on_db_error(self, retriever):
        with patch("rag_v1.media.music_retriever.conn_ctx", side_effect=RuntimeError("DB down")):
            result = retriever.find_similar("test song")
        assert result == []

    def test_filter_by_attributes_returns_empty_on_db_error(self, retriever):
        with patch("rag_v1.media.music_retriever.conn_ctx", side_effect=RuntimeError("DB down")):
            result = retriever.filter_by_attributes()
        assert result == []

    def test_get_similar_cluster_returns_empty_on_db_error(self, retriever):
        with patch("rag_v1.media.music_retriever.conn_ctx", side_effect=RuntimeError("DB down")):
            result = retriever.get_similar_cluster("test")
        assert result == []

    def test_search_by_mood_returns_empty_on_clap_failure(self, retriever):
        retriever._clap.embed_text.side_effect = RuntimeError("CLAP down")
        result = retriever.search_by_mood("energetic dance")
        assert result == []


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

class TestResultParsing:
    def test_search_by_mood_parses_rows(self, retriever, mock_conn_ctx):
        _, conn, cursor = mock_conn_ctx
        cursor.fetchall.return_value = [_make_db_row()]

        results = retriever.search_by_mood("dance music")
        assert len(results) == 1
        assert results[0].title == "Test Track"
        assert results[0].artist == "Test Artist"
        assert results[0].score == pytest.approx(0.85)

    def test_result_display_name_title_and_artist(self, retriever, mock_conn_ctx):
        _, conn, cursor = mock_conn_ctx
        cursor.fetchall.return_value = [_make_db_row(title="Song", artist="Band")]

        results = retriever.search_by_mood("rock music")
        assert results[0].display_name == "Song — Band"

    def test_filter_by_attributes_score_defaults_to_one(self, retriever, mock_conn_ctx):
        _, conn, cursor = mock_conn_ctx
        row = _make_db_row()
        del row["score"]  # attribute filter rows have no score column
        cursor.fetchall.return_value = [row]

        results = retriever.filter_by_attributes()
        assert results[0].score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# detect_intent
# ---------------------------------------------------------------------------

class TestDetectIntent:
    @pytest.mark.parametrize("query,expected_intent", [
        ("find songs about heartbreak", "mood"),
        ("song that says California dreamin", "lyrics"),
        ("find more like Dear Mama", "similar"),
        ("make a playlist for a road trip", "playlist"),
        ("playlist for working out", "playlist"),
        ("instrumental songs only", "attribute"),
        ("songs with bpm over 120", "attribute"),
        ("group my music by sound", "cluster"),
        ("songs that sound similar to each other", "cluster"),
    ])
    def test_intent_detection(self, query, expected_intent):
        intent = detect_intent(query)
        assert intent is not None
        assert intent.intent == expected_intent

    def test_returns_none_for_non_music_query(self):
        assert detect_intent("explain how neural networks work") is None
        assert detect_intent("what is the capital of France?") is None

    def test_playlist_extracts_bpm_params(self):
        # BPM regex matches "bpm<space>NNN" or "bpm>NNN"; "over NNN" (with space)
        # is NOT supported by the current regex (no \s* between "over" and \d+).
        intent = detect_intent("make a playlist for the gym, bpm > 140")
        assert intent is not None
        assert intent.intent == "playlist"
        assert intent.params.get("min_bpm") == 140

    def test_attribute_extracts_has_vocals_false(self):
        intent = detect_intent("show me instrumental songs")
        assert intent is not None
        assert intent.params.get("has_vocals") is False

    def test_attribute_extracts_explicit_flag(self):
        intent = detect_intent("show explicit songs")
        assert intent is not None
        assert intent.params.get("is_explicit") is True


# ---------------------------------------------------------------------------
# format_music_context
# ---------------------------------------------------------------------------

class TestFormatMusicContext:
    def test_returns_empty_for_no_results(self):
        intent = MusicIntent(intent="mood", query="dance music")
        assert format_music_context(intent, []) == ""

    def test_contains_intent_and_query(self):
        from rag_v1.media.music_retriever import MusicResult
        intent = MusicIntent(intent="mood", query="upbeat songs")
        result = MusicResult(
            media_id="1", file_path="/m/t.mp3", title="Song", artist="Band",
            score=0.9,
        )
        output = format_music_context(intent, [result])
        assert "mood" in output
        assert "upbeat songs" in output
        assert "Song" in output

    def test_contains_music_context_tags(self):
        from rag_v1.media.music_retriever import MusicResult
        intent = MusicIntent(intent="lyrics", query="California")
        result = MusicResult(
            media_id="1", file_path="/m/t.mp3", title="Track", artist="A",
            score=0.8,
        )
        output = format_music_context(intent, [result])
        assert "<music_context" in output
        assert "</music_context>" in output
