"""
tests/test_vector_index.py

Unit tests for rag_v1/db/vector_index.py.

Two things are worth guarding here, and neither is obvious from reading the
module:

1.  **The right knob for the right access method.**  ivfflat and hnsw have
    different recall parameters and setting the wrong one is completely
    silent — an ivfflat scan left at the default probes = 1 examines one list
    out of 4000 and returns almost nothing, which is indistinguishable from
    "no matches".  So these tests assert on the SQL actually emitted.

2.  **The constants have one home.**  wiki_retriever and the partition
    migration used to carry their own copies, which drifted.  The aliasing
    tests fail if someone re-hardcodes a value locally.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_v1.db.vector_index import (
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_QUERY_TIMEOUT_MS,
    IVFFLAT_DEFAULT_PROBES,
    VECTOR_INDEX_AMS,
    WIKI_EMBED_DIMS,
    WIKI_IVFFLAT_LISTS,
    WIKI_IVFFLAT_PROBES,
    apply_vector_tuning,
    vector_search,
)

DSN = "postgresql://localhost/testdb"


def _executed(conn: MagicMock) -> list[str]:
    """Render every statement passed to conn.execute() as plain SQL text."""
    out: list[str] = []
    for c in conn.execute.call_args_list:
        stmt = c.args[0]
        out.append(stmt if isinstance(stmt, str) else stmt.as_string(None))
    return out


# ---------------------------------------------------------------------------
# apply_vector_tuning — knob selection
# ---------------------------------------------------------------------------

class TestKnobSelection:
    def test_ivfflat_sets_probes_and_not_ef_search(self):
        conn = MagicMock()
        apply_vector_tuning(conn, index_am="ivfflat")
        sqls = " ".join(_executed(conn))
        assert f"SET ivfflat.probes = {WIKI_IVFFLAT_PROBES}" in sqls
        assert "hnsw.ef_search" not in sqls

    def test_hnsw_sets_ef_search_and_not_probes(self):
        conn = MagicMock()
        apply_vector_tuning(conn, index_am="hnsw")
        sqls = " ".join(_executed(conn))
        assert f"SET hnsw.ef_search = {DEFAULT_HNSW_EF_SEARCH}" in sqls
        assert "ivfflat.probes" not in sqls

    def test_unknown_am_falls_back_to_hnsw(self):
        """
        An unrecognised access method must still tune something.  Falling
        through with no knob set would leave pgvector on its defaults, and
        ivfflat's default of 1 probe is the silent-empty-result case.
        """
        conn = MagicMock()
        apply_vector_tuning(conn, index_am="brin-nonsense")
        assert "hnsw.ef_search" in " ".join(_executed(conn))

    def test_timeout_always_applied(self):
        for am in ("ivfflat", "hnsw"):
            conn = MagicMock()
            apply_vector_tuning(conn, index_am=am)
            assert f"SET statement_timeout = {DEFAULT_QUERY_TIMEOUT_MS}" in \
                " ".join(_executed(conn))

    def test_overrides_are_honoured(self):
        conn = MagicMock()
        apply_vector_tuning(conn, index_am="ivfflat", probes=17, timeout_ms=1234)
        sqls = " ".join(_executed(conn))
        assert "SET ivfflat.probes = 17" in sqls
        assert "SET statement_timeout = 1234" in sqls

    def test_values_are_literals_not_placeholders(self):
        """
        These are GUCs, and Postgres does not accept a bound parameter in SET.
        Rendering through sql.Literal is what makes that safe; a %s here would
        fail at runtime only, on the wiki path, under load.
        """
        conn = MagicMock()
        apply_vector_tuning(conn, index_am="ivfflat")
        assert "%s" not in " ".join(_executed(conn))


# ---------------------------------------------------------------------------
# vector_search — delegation
# ---------------------------------------------------------------------------

class TestVectorSearch:
    def _patched_conn(self):
        conn = MagicMock()
        cur = MagicMock()
        cur.execute.return_value = cur
        cur.fetchall.return_value = [{"id": 1}]
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=conn)
        ctx.__exit__ = MagicMock(return_value=False)
        return conn, cur, ctx

    def test_returns_rows_and_tunes_first(self):
        conn, cur, ctx = self._patched_conn()
        with patch("rag_v1.db.vector_index.conn_ctx", return_value=ctx) as cc:
            rows = vector_search(DSN, "SELECT 1", (), index_am="ivfflat")

        assert rows == [{"id": 1}]
        cc.assert_called_once_with(DSN)
        # Tuning must precede the query, or the first query of every checkout
        # runs on the wrong setting.
        assert "ivfflat.probes" in " ".join(_executed(conn))
        cur.execute.assert_called_once_with("SELECT 1", ())

    def test_params_passed_through_untouched(self):
        conn, cur, ctx = self._patched_conn()
        params = ([0.1, 0.2], 0.5, 10)
        with patch("rag_v1.db.vector_index.conn_ctx", return_value=ctx):
            vector_search(DSN, "SELECT %s, %s, %s", params)
        assert cur.execute.call_args.args[1] == params

    def test_connection_released_on_exception(self):
        """The pool only reclaims the connection if the `with` exits."""
        conn, cur, ctx = self._patched_conn()
        cur.execute.side_effect = RuntimeError("boom")
        with patch("rag_v1.db.vector_index.conn_ctx", return_value=ctx):
            with pytest.raises(RuntimeError):
                vector_search(DSN, "SELECT 1", ())
        ctx.__exit__.assert_called_once()


# ---------------------------------------------------------------------------
# Constants — single source of truth
# ---------------------------------------------------------------------------

class TestConstants:
    def test_ivfflat_preferred_over_hnsw(self):
        """
        Order is meaningful: wiki_chunks carries ivfflat since the 2026-08-24
        migration, and _query_index_kind picks the first match.
        """
        assert VECTOR_INDEX_AMS[0] == "ivfflat"
        assert set(VECTOR_INDEX_AMS) == {"ivfflat", "hnsw"}

    def test_probes_not_sqrt_of_lists(self):
        """
        Guards the bug two earlier tests actively enforced.  sqrt(lists) = 63
        is pgvector's single-index guidance; across 32 partitions it produced
        66 s queries that blew the statement timeout and returned nothing.
        """
        sqrt_lists = int(WIKI_IVFFLAT_LISTS ** 0.5)
        assert WIKI_IVFFLAT_PROBES < sqrt_lists
        assert WIKI_IVFFLAT_PROBES > IVFFLAT_DEFAULT_PROBES

    def test_wiki_retriever_aliases_shared_constants(self):
        """wiki_retriever must not re-hardcode these; they drifted before."""
        import rag_v1.wiki.wiki_retriever as wr
        assert wr._IVFFLAT_PROBES is WIKI_IVFFLAT_PROBES
        assert wr._WIKI_EMBED_DIMS is WIKI_EMBED_DIMS
        assert wr._WIKI_VECTOR_INDEX_AM is VECTOR_INDEX_AMS

    def test_halfvec_sql_matches_declared_dims(self):
        """
        pgvector only uses the halfvec index when the query casts to exactly
        the indexed type and width.  A mismatch here builds a correct index
        that is then silently never used.
        """
        import rag_v1.wiki.wiki_retriever as wr
        assert f"halfvec({WIKI_EMBED_DIMS})" in wr._SQL_TOP_CHUNKS_HALFVEC
