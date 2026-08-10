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
    def test_uses_hnsw(self):
        assert "USING hnsw" in _rendered()

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

    def test_carries_the_tuning_parameters(self):
        rendered = _rendered()
        assert f"m = {mig.HNSW_M}" in rendered
        assert f"ef_construction = {mig.HNSW_EF_CONSTRUCTION}" in rendered

    def test_index_name_is_derived_from_the_partition(self):
        assert '"wiki_chunks_part_p007_hv_hnsw"' in _rendered()

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
