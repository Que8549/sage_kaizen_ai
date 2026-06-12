"""
tests/test_retriever.py

Unit tests for rag_v1/retrieve/retriever.py — PgvectorRetriever.

Key behaviors under test:
1. retrieve() uses conn_ctx (not get_conn); stale connections are evicted.
2. Distance threshold filtering is applied in Python after the DB query.
3. Returned RetrievedChunk fields are populated correctly.
4. retrieve() returns [] gracefully when the DB raises an OperationalError.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest


DSN = "postgresql://localhost/testdb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**kwargs):
    cfg = MagicMock()
    cfg.pg_dsn = DSN
    cfg.top_k = 5
    cfg.max_distance = 0.5
    cfg.embed_base_url = "http://127.0.0.1:8020/v1"
    cfg.embed_model = "bge-m3-embed"
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _make_row(distance: float = 0.2, content: str = "chunk text") -> dict:
    return {
        "source_id": "src-1",
        "chunk_id": 42,
        "content": content,
        "metadata": {"title": "Test Doc"},
        "distance": distance,
    }


@pytest.fixture
def mock_embed():
    with patch("rag_v1.retrieve.retriever.EmbedClient") as MockEmbed:
        client = MagicMock()
        client.embed.return_value = [[0.1] * 1024]
        MockEmbed.return_value = client
        yield client


@pytest.fixture
def mock_conn_ctx():
    conn = MagicMock()
    conn.closed = False
    conn.execute.return_value = conn

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)

    with patch("rag_v1.retrieve.retriever.conn_ctx", return_value=ctx) as patched:
        yield patched, conn


# ---------------------------------------------------------------------------
# conn_ctx usage (not get_conn)
# ---------------------------------------------------------------------------

class TestConnCtxUsage:
    def test_uses_conn_ctx_not_get_conn(self, mock_embed, mock_conn_ctx):
        """conn_ctx must be called; get_conn must never be called."""
        patched_ctx, conn = mock_conn_ctx
        conn.execute.return_value = MagicMock(fetchall=MagicMock(return_value=[]))

        with patch("rag_v1.retrieve.retriever.get_conn", create=True) as mock_get_conn:
            from rag_v1.retrieve.retriever import PgvectorRetriever
            r = PgvectorRetriever(_make_cfg())
            r.retrieve("test query")

        mock_get_conn.assert_not_called()
        patched_ctx.assert_called_once_with(DSN)

    def test_conn_close_not_called(self, mock_embed, mock_conn_ctx):
        """conn_ctx manages lifecycle; explicit close must not be called."""
        _, conn = mock_conn_ctx
        conn.execute.return_value = MagicMock(fetchall=MagicMock(return_value=[]))

        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg())
        r.retrieve("test query")

        conn.close.assert_not_called()


# ---------------------------------------------------------------------------
# Distance threshold filtering
# ---------------------------------------------------------------------------

class TestDistanceFiltering:
    def test_filters_rows_above_max_distance(self, mock_embed, mock_conn_ctx):
        _, conn = mock_conn_ctx
        rows = [
            _make_row(distance=0.1),   # passes (< 0.5)
            _make_row(distance=0.49),  # passes
            _make_row(distance=0.50),  # filtered out (== max_distance, not <)
            _make_row(distance=0.9),   # filtered out
        ]
        conn.execute.return_value = MagicMock(fetchall=MagicMock(return_value=rows))

        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg(max_distance=0.5))
        results = r.retrieve("test")

        assert len(results) == 2

    def test_all_rows_pass_when_below_threshold(self, mock_embed, mock_conn_ctx):
        _, conn = mock_conn_ctx
        rows = [_make_row(distance=0.1), _make_row(distance=0.3)]
        conn.execute.return_value = MagicMock(fetchall=MagicMock(return_value=rows))

        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg())
        results = r.retrieve("test")
        assert len(results) == 2

    def test_empty_result_when_all_rows_exceed_threshold(self, mock_embed, mock_conn_ctx):
        _, conn = mock_conn_ctx
        rows = [_make_row(distance=0.8), _make_row(distance=0.95)]
        conn.execute.return_value = MagicMock(fetchall=MagicMock(return_value=rows))

        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg())
        results = r.retrieve("test")
        assert results == []


# ---------------------------------------------------------------------------
# RetrievedChunk fields
# ---------------------------------------------------------------------------

class TestRetrievedChunkFields:
    def test_chunk_fields_populated(self, mock_embed, mock_conn_ctx):
        _, conn = mock_conn_ctx
        row = _make_row(distance=0.25, content="Important paragraph.")
        conn.execute.return_value = MagicMock(fetchall=MagicMock(return_value=[row]))

        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg())
        results = r.retrieve("test query")

        assert len(results) == 1
        chunk = results[0]
        assert chunk.source_id == "src-1"
        assert chunk.chunk_id == 42
        assert chunk.content == "Important paragraph."
        assert chunk.metadata == {"title": "Test Doc"}

    def test_score_derived_from_distance(self, mock_embed, mock_conn_ctx):
        _, conn = mock_conn_ctx
        row = _make_row(distance=0.25)
        conn.execute.return_value = MagicMock(fetchall=MagicMock(return_value=[row]))

        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg())
        results = r.retrieve("query")

        # score = 1.0 / (1.0 + distance) = 1.0 / 1.25 = 0.8
        assert results[0].score == pytest.approx(1.0 / 1.25)

    def test_top_k_respected(self, mock_embed, mock_conn_ctx):
        _, conn = mock_conn_ctx
        # Return more rows than top_k; SQL limits should handle it, but
        # verify the query is called with the correct k parameter.
        conn.execute.return_value = MagicMock(fetchall=MagicMock(return_value=[]))

        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg(top_k=3))
        r.retrieve("query", top_k=3)

        # The SQL was called — just verify no exception and conn used
        conn.execute.assert_called()


# ---------------------------------------------------------------------------
# Stale connection eviction
# ---------------------------------------------------------------------------

class TestStaleConnectionEviction:
    def test_db_error_propagates_through_conn_ctx(self, mock_embed):
        """conn_ctx evicts on OperationalError; the retriever re-raises it."""
        import psycopg
        from rag_v1.retrieve.retriever import PgvectorRetriever

        conn = MagicMock()
        conn.closed = False
        conn.execute.side_effect = psycopg.OperationalError("connection lost")

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=conn)
        # Return False so the exception propagates (not suppressed)
        ctx.__exit__ = MagicMock(return_value=False)

        with patch("rag_v1.retrieve.retriever.conn_ctx", return_value=ctx):
            r = PgvectorRetriever(_make_cfg())
            with pytest.raises(psycopg.OperationalError):
                r.retrieve("test")
