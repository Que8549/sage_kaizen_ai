"""
tests/test_retriever.py

Unit tests for rag_v1/retrieve/retriever.py — PgvectorRetriever.

Key behaviors under test:
1. retrieve() delegates to rag_v1.db.vector_index.vector_search, which owns
   connection handling and ANN tuning for every pgvector caller.
2. Distance threshold filtering is applied in Python after the DB query.
3. Returned RetrievedChunk fields are populated correctly.
4. retrieve() propagates OperationalError rather than masking it.

These used to patch `retriever.conn_ctx` and assert the connection was never
closed by hand.  That contract still holds, but it moved: pooling and the
`with` that returns the connection now live in vector_search, and pg.py's own
tests cover them.  Asserting it here would only re-test a mock.
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
def mock_search():
    """Patch vector_search; `.rows` is what the fake query returns."""
    with patch("rag_v1.retrieve.retriever.vector_search") as patched:
        patched.return_value = []
        yield patched


# ---------------------------------------------------------------------------
# conn_ctx usage (not get_conn)
# ---------------------------------------------------------------------------

class TestVectorSearchDelegation:
    def test_delegates_to_vector_search_with_dsn(self, mock_embed, mock_search):
        """The DSN and the bound parameters must reach vector_search intact."""
        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg(top_k=7))
        r.retrieve("test query")

        mock_search.assert_called_once()
        dsn, sql_text, params = mock_search.call_args.args
        assert dsn == DSN
        assert "rag_chunks" in sql_text
        # (query_vector, query_vector, k) — the vector is bound twice because
        # it appears in both the SELECT distance and the ORDER BY.
        assert params[2] == 7
        assert params[0] == params[1]

    def test_top_k_argument_overrides_config(self, mock_embed, mock_search):
        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg(top_k=5))
        r.retrieve("test query", top_k=2)

        _, _, params = mock_search.call_args.args
        assert params[2] == 2


# ---------------------------------------------------------------------------
# Distance threshold filtering
# ---------------------------------------------------------------------------

class TestDistanceFiltering:
    def test_filters_rows_above_max_distance(self, mock_embed, mock_search):
        rows = [
            _make_row(distance=0.1),   # passes (< 0.5)
            _make_row(distance=0.49),  # passes
            _make_row(distance=0.50),  # filtered out (== max_distance, not <)
            _make_row(distance=0.9),   # filtered out
        ]
        mock_search.return_value = rows

        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg(max_distance=0.5))
        results = r.retrieve("test")

        assert len(results) == 2

    def test_all_rows_pass_when_below_threshold(self, mock_embed, mock_search):
        rows = [_make_row(distance=0.1), _make_row(distance=0.3)]
        mock_search.return_value = rows

        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg())
        results = r.retrieve("test")
        assert len(results) == 2

    def test_empty_result_when_all_rows_exceed_threshold(self, mock_embed, mock_search):
        rows = [_make_row(distance=0.8), _make_row(distance=0.95)]
        mock_search.return_value = rows

        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg())
        results = r.retrieve("test")
        assert results == []


# ---------------------------------------------------------------------------
# RetrievedChunk fields
# ---------------------------------------------------------------------------

class TestRetrievedChunkFields:
    def test_chunk_fields_populated(self, mock_embed, mock_search):
        row = _make_row(distance=0.25, content="Important paragraph.")
        mock_search.return_value = [row]

        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg())
        results = r.retrieve("test query")

        assert len(results) == 1
        chunk = results[0]
        assert chunk.source_id == "src-1"
        assert chunk.chunk_id == 42
        assert chunk.content == "Important paragraph."
        assert chunk.metadata == {"title": "Test Doc"}

    def test_score_derived_from_distance(self, mock_embed, mock_search):
        row = _make_row(distance=0.25)
        mock_search.return_value = [row]

        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg())
        results = r.retrieve("query")

        # score = 1.0 / (1.0 + distance) = 1.0 / 1.25 = 0.8
        assert results[0].score == pytest.approx(1.0 / 1.25)

    def test_top_k_respected(self, mock_embed, mock_search):
        # SQL LIMIT does the truncation; assert k reaches the query.

        from rag_v1.retrieve.retriever import PgvectorRetriever
        r = PgvectorRetriever(_make_cfg(top_k=3))
        r.retrieve("query", top_k=3)

        assert mock_search.call_args.args[2][2] == 3


# ---------------------------------------------------------------------------
# Stale connection eviction
# ---------------------------------------------------------------------------

class TestStaleConnectionEviction:
    def test_db_error_propagates(self, mock_embed):
        """
        The pool evicts a dead connection; the retriever must not swallow the
        error, because doc-RAG returning [] silently is the failure mode that
        makes a dead database look like an empty corpus.
        """
        import psycopg
        from rag_v1.retrieve.retriever import PgvectorRetriever

        with patch("rag_v1.retrieve.retriever.vector_search",
                   side_effect=psycopg.OperationalError("connection lost")):
            r = PgvectorRetriever(_make_cfg())
            with pytest.raises(psycopg.OperationalError):
                r.retrieve("test")
