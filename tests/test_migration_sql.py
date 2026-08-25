"""
tests/test_migration_sql.py

Asserts the DDL that scripts/migrate_wiki_chunks_partitioned.py generates.

Why this file exists: the index phase runs only after a multi-day copy, so a
mistake in its statement would not surface for days, on a machine that reboots
under load. The halfvec cast is the sharpest edge — pgvector only uses a halfvec
index when the query casts identically, so a mismatch here builds a correct
index that WikiRetriever then silently never uses, leaving a 3.5 TB sequential
scan behind an index that looks present.

`Composed.as_string(None)` renders without a connection, so this needs no
database.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load_migration_module():
    """Import the script by path — scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(
        "_migration_under_test",
        _ROOT / "scripts" / "migrate_wiki_chunks_partitioned.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mig = _load_migration_module()


def _rendered(part: str = "wiki_chunks_part_p007", tablespace: str | None = None) -> str:
    return mig._index_stmt(part, tablespace).as_string(None)


class TestIndexStatement:
    def test_uses_ivfflat(self):
        # Switched from hnsw 2026-08-17: HNSW projected ~44 days, bottlenecked
        # on reading source vectors off the HDD, not on graph construction.
        assert "USING ivfflat" in _rendered()
        assert "USING hnsw" not in _rendered()

    def test_casts_to_halfvec_at_the_model_dimensionality(self):
        # Must match _WIKI_EMBED_DIMS / the cast in wiki_retriever's halfvec SQL.
        assert f"embedding::halfvec({mig.EMBED_DIMS})" in _rendered()

    def test_uses_the_halfvec_cosine_opclass(self):
        # vector_cosine_ops here would not match a halfvec expression at all.
        assert "halfvec_cosine_ops" in _rendered()
        assert "vector_cosine_ops" not in _rendered()

    def test_dimension_renders_bare_not_quoted(self):
        # A type modifier must be an integer literal: halfvec('1024') is a
        # syntax error, and Literal() quoting strings makes that easy to hit.
        assert "halfvec(1024)" in _rendered()
        assert "halfvec('1024')" not in _rendered()

    def test_carries_the_lists_parameter(self):
        # lists ~= sqrt(rows per partition) = sqrt(16.0M) ~= 4000.
        assert f"lists = {mig.IVFFLAT_LISTS}" in _rendered()

    def test_probes_is_deliberately_below_sqrt_of_lists(self):
        """probes is NOT sqrt(lists) here, and that is the whole point.

        sqrt(lists) = 63 assumes ONE index. wiki_chunks is 32 partitions and a
        nearest-neighbour query cannot prune them, so 63 means 2016 lists and
        ~8M vectors per query. Measured 2026-08-24: 66 s/query at 63 versus a
        25 s statement_timeout -- every wiki-RAG query would have returned
        nothing. probes=10 gives p90 10.0 s at identical recall.

        This test previously asserted the sqrt rule and therefore enforced the
        bug. If someone "restores" it, this is why they should not.
        """
        assert mig.IVFFLAT_PROBES < mig.IVFFLAT_LISTS ** 0.5
        assert mig.IVFFLAT_PROBES == 5

    def test_retriever_probes_matches_the_migration(self):
        # Two constants in two files describing one setting; drift means the
        # query silently stops matching what the index was tuned for.
        from rag_v1.wiki.wiki_retriever import _IVFFLAT_PROBES
        assert _IVFFLAT_PROBES == mig.IVFFLAT_PROBES

    def test_index_name_is_derived_from_the_partition(self):
        assert '"wiki_chunks_part_p007_hv_ivf"' in _rendered()

    def test_targets_the_named_partition(self):
        assert 'ON "wiki_chunks_part_p007"' in _rendered()

    def test_identifiers_are_quoted(self):
        # Composed via sql.Identifier rather than f-strings, so names are quoted
        # rather than assumed not to need quoting.
        assert '"wiki_chunks_part_p007"' in _rendered()


class TestIndexTablespace:
    def test_tablespace_is_appended_when_given(self):
        assert 'TABLESPACE "sage_nvme"' in _rendered(tablespace="sage_nvme")

    def test_tablespace_is_omitted_when_none(self):
        assert "TABLESPACE" not in _rendered(tablespace=None)

    def test_empty_tablespace_is_treated_as_none(self):
        # --index-tablespace '' is the documented way to use the default.
        assert "TABLESPACE" not in _rendered(tablespace="")

    def test_tablespace_follows_the_with_clause(self):
        rendered = _rendered(tablespace="sage_nvme")
        assert rendered.index("WITH (") < rendered.index("TABLESPACE")


class TestPartitionNaming:
    def test_zero_padded_to_three_digits(self):
        assert mig._partition_name(0) == "wiki_chunks_part_p000"
        assert mig._partition_name(7) == "wiki_chunks_part_p007"

    def test_handles_two_and_three_digit_indexes(self):
        assert mig._partition_name(31) == "wiki_chunks_part_p031"
        assert mig._partition_name(127) == "wiki_chunks_part_p127"

    def test_names_sort_in_partition_order(self):
        # Zero padding matters: without it p10 sorts before p2, and --status
        # would report partitions out of order.
        names = [mig._partition_name(i) for i in range(12)]
        assert names == sorted(names)

    def test_names_are_unique(self):
        names = {mig._partition_name(i) for i in range(mig.DEFAULT_PARTITIONS)}
        assert len(names) == mig.DEFAULT_PARTITIONS


class TestFetchOneGuard:
    def test_returns_the_row(self):
        class _Cur:
            def fetchone(self):
                return {"n": 5}

        assert mig._one(_Cur())["n"] == 5

    def test_raises_instead_of_returning_none(self):
        # Every call site subscripts the result immediately; None would surface
        # as a TypeError several frames away from the query that produced it.
        class _Cur:
            def fetchone(self):
                return None

        with pytest.raises(RuntimeError, match="exactly one row"):
            mig._one(_Cur())


class TestMigrationConstants:
    def test_partition_count_matches_the_memory_budget_rationale(self):
        # 508M rows / 32 partitions ~= 15.9M rows; at ~2.3 KB per halfvec HNSW
        # element that is ~36 GB, which fits a 48 GB maintenance_work_mem.
        # Raising this without re-checking that arithmetic silently returns the
        # build to the on-disk path it was designed to avoid.
        assert mig.DEFAULT_PARTITIONS == 32
        assert mig.DEFAULT_MWM == "48GB"

    def test_embedding_dimensionality_matches_jina_clip_v2(self):
        assert mig.EMBED_DIMS == 1024

    def test_source_and_target_are_distinct(self):
        assert mig.SOURCE != mig.TARGET


class TestGrantReplication:
    """
    The swap renames a table; it does not carry access control across.

    The new table was built under the owner DSN and inherited that role's
    default ACL (owner-only), so the moment the rename completed both
    applications got `permission denied for table wiki_chunks` — right after
    a four-day index build, and in direct violation of the contract that main
    can read every ingest-owned table at all times.  It was fixed by hand in
    the database; these tests exist so the fix cannot be lost from the code
    again, and so applying the same migration to wiki_images does not repeat
    it.
    """

    class _FakeCur:
        def __init__(self, row):
            self._row = row
            self.executed = []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, q, params=None):
            self.executed.append((q, params))
            return self

        def fetchone(self):
            return self._row

    class _FakeConn:
        def __init__(self, row):
            self.row = row
            self.statements = []

        def cursor(self, **kw):
            return TestGrantReplication._FakeCur(self.row)

        def execute(self, q, params=None):
            self.statements.append(
                q if isinstance(q, str) else q.as_string(None)
            )
            return self

    def _run(self, acl, partitions=2):
        conn = self._FakeConn({"owner": "postgres", "acl": acl})
        mig._replicate_grants(conn, "wiki_chunks", "wiki_chunks_part", partitions)
        return conn.statements

    def test_owner_applied_to_parent_and_every_partition(self):
        stmts = self._run(None, partitions=3)
        owners = [s for s in stmts if "OWNER TO" in s]
        # parent + 3 partitions
        assert len(owners) == 4
        assert all('"postgres"' in s for s in owners)
        assert any('"wiki_chunks_part"' in s for s in owners)
        assert any('"wiki_chunks_part_p002"' in s for s in owners)

    def test_null_acl_still_sets_owner_but_grants_nothing(self):
        # relacl is NULL when nobody has been GRANTed on the table.  There is
        # nothing to replicate, and inventing a grant would be wrong.
        stmts = self._run(None, partitions=1)
        assert any("OWNER TO" in s for s in stmts)
        assert not any("GRANT" in s for s in stmts)

    def test_grants_replicated_to_partitions_not_just_parent(self):
        # Privileges are not inherited at query time: a SELECT that touches a
        # partition is checked against that partition's own ACL.
        stmts = self._run(["sage=arwdDxt/postgres"], partitions=2)
        grants = [s for s in stmts if "GRANT ALL" in s]
        assert len(grants) == 3            # parent + 2 partitions
        assert all('"sage"' in s for s in grants)

    def test_public_grantee_rendered_as_keyword_not_identifier(self):
        # An empty grantee in relacl means PUBLIC.  Quoting it as an
        # identifier would create a role literally named "PUBLIC".
        stmts = self._run(["=r/postgres"], partitions=1)
        grants = [s for s in stmts if "GRANT ALL" in s]
        assert grants and all("TO PUBLIC" in s for s in grants)
        assert not any('"PUBLIC"' in s for s in grants)

    def test_multiple_grantees_each_replicated(self):
        stmts = self._run(["sage=arwdDxt/postgres", "alquin=r/postgres"],
                          partitions=1)
        grants = [s for s in stmts if "GRANT ALL" in s]
        assert len(grants) == 4           # 2 grantees x (parent + 1 partition)
        assert any('"sage"' in s for s in grants)
        assert any('"alquin"' in s for s in grants)
