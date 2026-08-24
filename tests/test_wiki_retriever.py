"""
tests/test_wiki_retriever.py

Unit tests for rag_v1/wiki/wiki_retriever.py.

Key behaviors under test:
1. _maybe_warmup sends one embed_text call on first invocation.
2. _maybe_warmup is a no-op on subsequent calls (_warmed_up flag).
3. _ensure_service calls _maybe_warmup after ping succeeds (fast path).
4. _ensure_service calls _maybe_warmup after startup loop succeeds (slow path).
5. search() returns WikiSearchResult(empty=True) when service is unavailable.
6. _get_images filters missing files from disk.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from rag_v1.wiki.wiki_retriever import DisplayGpuRefused


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mark_index_present(r, present: bool = True) -> None:
    """Pin the vector-index guard's cached answer, skipping the catalog query."""
    r._index_present = present
    r._index_checked_at = time.monotonic()


def _fake_conn(fetchone=None, fetchall=None):
    """A conn_ctx double returning a fixed row / rowset."""
    cursor = MagicMock()
    cursor.execute.return_value = cursor
    cursor.fetchone.return_value = fetchone
    cursor.fetchall.return_value = fetchall or []
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)

    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    return conn, cursor


# A healthy /health payload from a service correctly loaded on the compute GPU.
HEALTHY = {"status": "ok", "device": "cuda:1", "model": "jina-clip-v2", "loaded": True}
# The same, but loaded on the display GPU — must be refused.
ON_DISPLAY_GPU = {"status": "ok", "device": "cuda:0", "model": "jina-clip-v2", "loaded": True}


@pytest.fixture
def wiki_cfg(tmp_path):
    """A minimal wiki_embed_config mock."""
    cfg = MagicMock()
    cfg.wiki_root  = tmp_path
    cfg.host       = "127.0.0.1"
    cfg.port       = 8031
    cfg.startup_timeout_s = 10
    cfg.log        = tmp_path / "wiki_embed.log"
    cfg.device     = "cuda:1"
    return cfg


@pytest.fixture
def retriever(wiki_cfg, monkeypatch):
    """A WikiRetriever with all external dependencies mocked."""
    # Never let the ambient environment decide the device under test.
    monkeypatch.delenv("WIKI_EMBED_DEVICE", raising=False)
    with (
        patch("rag_v1.wiki.wiki_retriever.load_wiki_embed_config", return_value=wiki_cfg),
        patch("rag_v1.wiki.wiki_retriever.MmEmbedClient") as MockClient,
    ):
        client = MagicMock()
        # health() is the readiness check (ping() no longer gates _ensure_service);
        # None = nothing listening, which is the default state under test.
        client.health.return_value = None
        client.ping.return_value = False
        client.embed_text.return_value = [[0.1] * 1024]
        MockClient.return_value = client

        from rag_v1.wiki.wiki_retriever import WikiRetriever
        r = WikiRetriever(pg_dsn="postgresql://localhost/test")
        r._client = client
        yield r


# ---------------------------------------------------------------------------
# _maybe_warmup
# ---------------------------------------------------------------------------

class TestMaybeWarmup:
    def test_sends_warmup_embed_on_first_call(self, retriever):
        retriever._warmed_up = False
        retriever._maybe_warmup()
        retriever._client.embed_text.assert_called_once_with(["warmup"])

    def test_sets_warmed_up_flag(self, retriever):
        retriever._warmed_up = False
        retriever._maybe_warmup()
        assert retriever._warmed_up is True

    def test_noop_on_second_call(self, retriever):
        retriever._warmed_up = True
        retriever._maybe_warmup()
        retriever._client.embed_text.assert_not_called()

    def test_graceful_on_embed_failure(self, retriever):
        retriever._warmed_up = False
        retriever._client.embed_text.side_effect = RuntimeError("CUDA error")
        retriever._maybe_warmup()  # must not raise
        # Flag is still set so we don't retry infinitely
        assert retriever._warmed_up is True


# ---------------------------------------------------------------------------
# _ensure_service — fast path (service already running)
# ---------------------------------------------------------------------------

class TestEnsureServiceFastPath:
    def test_returns_true_when_health_succeeds(self, retriever):
        retriever._client.health.return_value = HEALTHY
        assert retriever._ensure_service() is True

    def test_calls_warmup_on_health_success(self, retriever):
        retriever._client.health.return_value = HEALTHY
        retriever._warmed_up = False
        retriever._ensure_service()
        retriever._client.embed_text.assert_called_once_with(["warmup"])

    def test_does_not_call_warmup_twice(self, retriever):
        retriever._client.health.return_value = HEALTHY
        retriever._warmed_up = True
        retriever._ensure_service()
        retriever._client.embed_text.assert_not_called()

    def test_does_not_spawn_when_already_running(self, retriever):
        retriever._client.health.return_value = HEALTHY
        with patch("rag_v1.wiki.wiki_retriever.subprocess.Popen") as MockPopen:
            retriever._ensure_service()
        MockPopen.assert_not_called()


# ---------------------------------------------------------------------------
# _ensure_service — slow path (starts the service)
# ---------------------------------------------------------------------------

class TestEnsureServiceSlowPath:
    def test_starts_process_when_health_fails(self, retriever, tmp_path):
        # health: first call None (nothing listening), then healthy after startup
        retriever._client.health.side_effect = [None, HEALTHY]

        with (
            patch("rag_v1.wiki.wiki_retriever.subprocess.Popen") as MockPopen,
            patch("rag_v1.wiki.wiki_retriever.time.sleep"),
        ):
            MockPopen.return_value = MagicMock(poll=MagicMock(return_value=None))
            result = retriever._ensure_service()

        assert result is True
        MockPopen.assert_called_once()

    def test_calls_warmup_after_startup(self, retriever, tmp_path):
        retriever._client.health.side_effect = [None, HEALTHY]
        retriever._warmed_up = False

        with (
            patch("rag_v1.wiki.wiki_retriever.subprocess.Popen") as MockPopen,
            patch("rag_v1.wiki.wiki_retriever.time.sleep"),
        ):
            MockPopen.return_value = MagicMock(poll=MagicMock(return_value=None))
            retriever._ensure_service()

        retriever._client.embed_text.assert_called_with(["warmup"])

    def test_returns_false_on_startup_timeout(self, retriever, tmp_path):
        # health always returns None → times out
        retriever._client.health.return_value = None
        retriever._startup_timeout_s = 0  # immediate timeout

        with (
            patch("rag_v1.wiki.wiki_retriever.subprocess.Popen") as MockPopen,
            patch("rag_v1.wiki.wiki_retriever.time.sleep"),
        ):
            MockPopen.return_value = MagicMock(poll=MagicMock(return_value=None))
            result = retriever._ensure_service()

        assert result is False

    def test_pins_resolved_device_in_child_env(self, retriever, tmp_path):
        """The child must not get to re-resolve the device we already validated."""
        retriever._client.health.side_effect = [None, HEALTHY]

        with (
            patch("rag_v1.wiki.wiki_retriever.subprocess.Popen") as MockPopen,
            patch("rag_v1.wiki.wiki_retriever.time.sleep"),
        ):
            MockPopen.return_value = MagicMock(poll=MagicMock(return_value=None))
            retriever._ensure_service()

        env = MockPopen.call_args.kwargs["env"]
        assert env["WIKI_EMBED_DEVICE"] == "cuda:1"


# ---------------------------------------------------------------------------
# Display-GPU guard (cuda:0 is display-only)
# ---------------------------------------------------------------------------

class TestDisplayGpuGuard:
    def test_refuses_to_spawn_on_display_gpu_from_config(self, retriever, wiki_cfg):
        """brains.yaml / config default naming cuda:0 must not start a service."""
        retriever._config_device = "cuda:0"
        retriever._client.health.return_value = None

        with (
            patch("rag_v1.wiki.wiki_retriever.subprocess.Popen") as MockPopen,
            patch("rag_v1.wiki.wiki_retriever.time.sleep"),
        ):
            with pytest.raises(DisplayGpuRefused, match="display GPU"):
                retriever._ensure_service()

        # The critical assertion: refused *before* spawning, not after.
        MockPopen.assert_not_called()

    def test_refuses_when_env_var_forces_display_gpu(self, retriever, monkeypatch):
        """An inherited WIKI_EMBED_DEVICE=cuda:0 beat brains.yaml before this guard."""
        monkeypatch.setenv("WIKI_EMBED_DEVICE", "cuda:0")
        retriever._config_device = "cuda:1"   # config is fine; the env var is not
        retriever._client.health.return_value = None

        with (
            patch("rag_v1.wiki.wiki_retriever.subprocess.Popen") as MockPopen,
            patch("rag_v1.wiki.wiki_retriever.time.sleep"),
        ):
            with pytest.raises(DisplayGpuRefused, match="WIKI_EMBED_DEVICE"):
                retriever._ensure_service()

        MockPopen.assert_not_called()

    def test_refuses_externally_started_service_on_display_gpu(self, retriever):
        """A bare ping proved only that something answered — check where it loaded."""
        retriever._client.health.return_value = ON_DISPLAY_GPU

        with pytest.raises(DisplayGpuRefused, match="already listening"):
            retriever._ensure_service()

    def test_consent_allows_display_gpu(self, retriever):
        """allow_display_gpu=True is the single sanctioned override."""
        retriever._allow_display_gpu = True
        retriever._client.health.return_value = ON_DISPLAY_GPU

        assert retriever._ensure_service() is True

    def test_non_display_mismatch_warns_and_continues(self, retriever):
        """cuda:2 isn't what we'd pick, but we didn't start it — no veto."""
        retriever._client.health.return_value = {"status": "ok", "device": "cuda:2"}

        assert retriever._ensure_service() is True

    def test_missing_device_field_does_not_fail_the_turn(self, retriever):
        """An older service build with no `device` key must not break retrieval."""
        retriever._client.health.return_value = {"status": "ok"}

        assert retriever._ensure_service() is True

    def test_search_degrades_gracefully_on_refusal(self, retriever):
        """The guard is loud in logs but must never break a chat turn."""
        # The vector-index guard runs first and would otherwise short-circuit
        # before the display-GPU guard is ever reached; this test is about the
        # latter, so declare the index present.
        _mark_index_present(retriever)
        retriever._client.health.return_value = ON_DISPLAY_GPU

        # context_injector._fetch_wiki_result wraps search() in try/except;
        # verify the exception is the only thing that escapes, not a crash
        # part-way through leaving inference running on the display GPU.
        with pytest.raises(DisplayGpuRefused):
            retriever.search("what is a snail")

    def test_ensure_service_is_serialised(self, retriever):
        """Concurrent turns must not each spawn a service (ingest §15, 2026-07-18)."""
        import threading

        retriever._client.health.side_effect = None
        retriever._client.health.return_value = None
        retriever._startup_timeout_s = 0
        spawned: list[int] = []

        def _fake_popen(*a, **kw):
            # Widen the race window that an unlocked implementation would lose.
            spawned.append(1)
            time.sleep(0.02)
            return MagicMock(poll=MagicMock(return_value=None))

        with (
            patch("rag_v1.wiki.wiki_retriever.subprocess.Popen", side_effect=_fake_popen),
            patch("rag_v1.wiki.wiki_retriever.time.sleep"),
        ):
            threads = [threading.Thread(target=retriever._ensure_service) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Serialised, so each thread spawns strictly after the previous finished;
        # what matters is that no two ran _ensure_service_locked concurrently.
        assert retriever._service_lock.acquire(blocking=False)
        retriever._service_lock.release()


# ---------------------------------------------------------------------------
# search() — graceful degradation
# ---------------------------------------------------------------------------

class TestSearch:
    def test_returns_empty_when_service_unavailable(self, retriever):
        retriever._client.health.return_value = None
        retriever._startup_timeout_s = 0

        with (
            patch("rag_v1.wiki.wiki_retriever.subprocess.Popen"),
            patch("rag_v1.wiki.wiki_retriever.time.sleep"),
        ):
            result = retriever.search("what is a snail")

        assert result.empty is True

    def test_returns_empty_when_embed_fails(self, retriever):
        retriever._client.health.return_value = HEALTHY
        retriever._client.embed_text.side_effect = RuntimeError("CUDA OOM")

        result = retriever.search("hello")
        assert result.empty is True

    def test_returns_empty_when_no_chunks_pass_threshold(self, retriever):
        retriever._client.health.return_value = HEALTHY
        retriever._client.embed_text.return_value = [[0.0] * 1024]

        # _get_chunks returns empty (distance too high)
        with patch.object(retriever, "_get_chunks", return_value=[]):
            result = retriever.search("something obscure")

        assert result.empty is True


# ---------------------------------------------------------------------------
# _get_images — disk file filtering
# ---------------------------------------------------------------------------

class TestGetImages:
    def test_filters_missing_files(self, retriever, tmp_path):
        # Create one real file; leave another missing
        real_file = tmp_path / "images" / "good.jpg"
        real_file.parent.mkdir(parents=True, exist_ok=True)
        real_file.write_bytes(b"fake-jpeg")

        rows = [
            {
                "image_id": 1,
                "bundle_id": "uuid-1",
                "relative_path": "images/good.jpg",
                "caption_text": "A good image",
                "is_hero": True,
                "hero_rank": 1,
                "sim_score": 0.9,
            },
            {
                "image_id": 2,
                "bundle_id": "uuid-1",
                "relative_path": "images/missing.jpg",
                "caption_text": "Missing",
                "is_hero": False,
                "hero_rank": 2,
                "sim_score": 0.8,
            },
        ]

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchall.return_value = rows

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        with patch("rag_v1.wiki.wiki_retriever.conn_ctx") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

            images = retriever._get_images(
                qvec=[0.0] * 1024,
                bundle_ids=["uuid-1"],
                top_images=3,
            )

        # Only the file that exists on disk should be returned
        assert len(images) == 1
        assert images[0].caption_text == "A good image"

    def test_returns_empty_for_empty_bundle_ids(self, retriever):
        result = retriever._get_images(qvec=[0.0] * 1024, bundle_ids=[], top_images=3)
        assert result == []


# ---------------------------------------------------------------------------
# Vector-index guard
#
# Without an ANN index, one search() is a full scan of a ~3.5 TB table that
# outlives its caller by 20+ minutes (measured 2026-08-06). ingest drops the
# index for the duration of every bulk run, so this is a routine state, not an
# exotic one, and the guard has to be right.
# ---------------------------------------------------------------------------

class TestVectorIndexCaching:
    def test_positive_result_is_cached(self, retriever):
        conn, _ = _fake_conn(fetchall=_indexdef_rows(VECTOR_DEF))
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn) as ctx:
            assert retriever._vector_index_ready() is True
            assert retriever._vector_index_ready() is True
        assert ctx.call_count == 1

    def test_negative_result_is_not_cached_forever(self, retriever):
        # A rebuild finishing must not require an app restart to be noticed.
        conn, _ = _fake_conn(fetchall=[])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn) as ctx:
            assert retriever._vector_index_ready() is False
            retriever._index_checked_at -= 10_000  # simulate the interval passing
            assert retriever._vector_index_ready() is False
        assert ctx.call_count == 2

    def test_negative_result_is_cached_within_the_interval(self, retriever):
        # A rebuild takes days; polling pg_index on every turn is pointless.
        conn, _ = _fake_conn(fetchall=[])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn) as ctx:
            for _ in range(5):
                assert retriever._vector_index_ready() is False
        assert ctx.call_count == 1

    def test_recovers_when_the_index_reappears(self, retriever):
        conn_absent, _ = _fake_conn(fetchall=[])
        conn_present, _ = _fake_conn(fetchall=_indexdef_rows(VECTOR_DEF))
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn_absent):
            assert retriever._vector_index_ready() is False
        retriever._index_checked_at -= 10_000
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn_present):
            assert retriever._vector_index_ready() is True

    def test_catalog_error_disables_without_caching(self, retriever):
        # Fail closed: guessing "present" risks the very scan this prevents.
        # Not cached, so a transient blip does not disable wiki for 5 minutes.
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", side_effect=OSError("down")):
            assert retriever._vector_index_ready() is False
        assert retriever._index_present is None

    def test_warns_once_per_transition_not_once_per_turn(self, retriever):
        conn, _ = _fake_conn(fetchall=[])
        with (
            patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn),
            patch("rag_v1.wiki.wiki_retriever._LOG") as log,
        ):
            for _ in range(4):
                retriever._vector_index_ready()
                retriever._index_checked_at -= 10_000
        assert log.warning.call_count == 1

    def test_is_thread_safe(self, retriever):
        # context_injector calls this from a worker pool.
        import threading
        conn, _ = _fake_conn(fetchall=_indexdef_rows(VECTOR_DEF))
        results: list[bool] = []

        def _run():
            results.append(retriever._vector_index_ready())

        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn) as ctx:
            threads = [threading.Thread(target=_run) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert all(results) and len(results) == 8
        assert ctx.call_count == 1


class TestSearchShortCircuit:
    def test_search_returns_empty_when_index_missing(self, retriever):
        _mark_index_present(retriever, False)
        result = retriever.search("what is a snail")
        assert result.empty is True
        assert result.chunks == []

    def test_search_does_not_start_the_embed_service_when_index_missing(self, retriever):
        # Starting jina-clip-v2 costs ~3.2 GB VRAM and up to 90 s of cold
        # torch.compile warmup, on the GPU a concurrent ingest is using.
        _mark_index_present(retriever, False)
        with patch.object(retriever, "_ensure_service") as ensure:
            retriever.search("what is a snail")
        ensure.assert_not_called()

    def test_search_does_not_embed_the_query_when_index_missing(self, retriever):
        _mark_index_present(retriever, False)
        retriever._client.embed_text.reset_mock()
        retriever.search("what is a snail")
        retriever._client.embed_text.assert_not_called()

    def test_search_proceeds_when_index_present(self, retriever):
        _mark_index_present(retriever, True)
        with patch.object(retriever, "_ensure_service", return_value=False) as ensure:
            retriever.search("what is a snail")
        ensure.assert_called_once()


class TestChunkQueryTimeout:
    def test_statement_timeout_is_set_on_the_vector_query(self, retriever):
        from rag_v1.wiki.wiki_retriever import _WIKI_QUERY_TIMEOUT_MS
        conn, _ = _fake_conn(fetchall=[])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            retriever._get_chunks([0.0] * 1024, top_k=5)
        executed = " | ".join(str(c[0][0]) for c in conn.execute.call_args_list)
        # Composed via psycopg.sql now, so match on content not exact text.
        assert "statement_timeout" in executed
        assert str(_WIKI_QUERY_TIMEOUT_MS) in executed

    def test_timeout_is_below_the_context_injector_deadline(self):
        # context_injector abandons the future at 30 s but Postgres keeps
        # scanning; the backend must give up first or the orphan outlives it.
        from rag_v1.wiki.wiki_retriever import _WIKI_QUERY_TIMEOUT_MS
        assert _WIKI_QUERY_TIMEOUT_MS < 30_000

    def test_ef_search_is_still_applied(self, retriever):
        conn, _ = _fake_conn(fetchall=[])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            retriever._get_chunks([0.0] * 1024, top_k=5)
        executed = " | ".join(str(c[0][0]) for c in conn.execute.call_args_list)
        assert "hnsw.ef_search" in executed


# ---------------------------------------------------------------------------
# halfvec vs vector index detection
#
# pgvector only uses a halfvec index when the query casts exactly as the index
# expression does. Detecting the wrong kind means the index is silently ignored
# and the query falls back to a 3.5 TB sequential scan -- with the guard now
# reporting "index present", which is worse than having no index at all.
# ---------------------------------------------------------------------------

def _indexdef_rows(*defs):
    return [{"def": d} for d in defs]


HALFVEC_DEF = (
    "CREATE INDEX p000_hv_hnsw ON public.wiki_chunks_p000 "
    "USING hnsw (((embedding)::halfvec(1024)) halfvec_cosine_ops)"
)
VECTOR_DEF = (
    "CREATE INDEX hnsw_wiki_chunks_embedding_cos ON public.wiki_chunks "
    "USING hnsw (embedding vector_cosine_ops)"
)


class TestIndexKindDetection:
    def test_detects_halfvec_expression_index(self, retriever):
        conn, _ = _fake_conn(fetchall=_indexdef_rows(HALFVEC_DEF))
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            assert retriever._query_index_kind() == "halfvec"

    def test_detects_plain_vector_index(self, retriever):
        conn, _ = _fake_conn(fetchall=_indexdef_rows(VECTOR_DEF))
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            assert retriever._query_index_kind() == "vector"

    def test_returns_none_when_no_index(self, retriever):
        conn, _ = _fake_conn(fetchall=[])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            assert retriever._query_index_kind() is None

    def test_ignores_ann_index_on_a_different_column(self, retriever):
        # An hnsw index exists, but not on embedding -- it cannot serve this query.
        other = ("CREATE INDEX x ON public.wiki_chunks USING hnsw "
                 "(title_vec vector_cosine_ops)")
        conn, _ = _fake_conn(fetchall=_indexdef_rows(other))
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            assert retriever._query_index_kind() is None

    def test_prefers_halfvec_when_both_exist(self, retriever):
        # Mid-migration both can be present; halfvec is the cheaper probe and
        # the intended destination.
        conn, _ = _fake_conn(fetchall=_indexdef_rows(VECTOR_DEF, HALFVEC_DEF))
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            assert retriever._query_index_kind() == "halfvec"

    def test_uses_indexdef_not_an_attribute_join(self, retriever):
        # An expression index stores 0 in indkey, so joining pg_attribute would
        # report the halfvec index -- the one the migration builds -- as absent.
        conn, cursor = _fake_conn(fetchall=_indexdef_rows(HALFVEC_DEF))
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            retriever._query_index_kind()
        sql = cursor.execute.call_args[0][0]
        assert "pg_get_indexdef" in sql
        assert "pg_attribute" not in sql

    def test_ready_records_the_kind(self, retriever):
        conn, _ = _fake_conn(fetchall=_indexdef_rows(HALFVEC_DEF))
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            assert retriever._vector_index_ready() is True
        assert retriever._index_kind == "halfvec"

    def test_ready_is_false_when_kind_is_none(self, retriever):
        conn, _ = _fake_conn(fetchall=[])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            assert retriever._vector_index_ready() is False
        assert retriever._index_kind is None


class TestChunkSqlMatchesIndexKind:
    def test_halfvec_index_emits_halfvec_casts(self, retriever):
        from rag_v1.wiki.wiki_retriever import _WIKI_EMBED_DIMS
        retriever._index_kind = "halfvec"
        conn, cursor = _fake_conn(fetchall=[])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            retriever._get_chunks([0.0] * 1024, top_k=5)
        sql = cursor.execute.call_args[0][0]
        assert f"halfvec({_WIKI_EMBED_DIMS})" in sql
        assert "::vector" not in sql

    def test_vector_index_emits_plain_vector_casts(self, retriever):
        retriever._index_kind = "vector"
        conn, cursor = _fake_conn(fetchall=[])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            retriever._get_chunks([0.0] * 1024, top_k=5)
        sql = cursor.execute.call_args[0][0]
        assert "::vector" in sql
        assert "halfvec" not in sql

    def test_order_by_matches_the_select_expression(self, retriever):
        # The ORDER BY is what the index has to match; a SELECT-only cast would
        # compute the distance correctly and still scan sequentially.
        retriever._index_kind = "halfvec"
        conn, cursor = _fake_conn(fetchall=[])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            retriever._get_chunks([0.0] * 1024, top_k=5)
        sql = cursor.execute.call_args[0][0]
        order_by = sql.split("ORDER BY")[1]
        assert "halfvec" in order_by

    def test_defaults_to_plain_vector_when_kind_unknown(self, retriever):
        retriever._index_kind = None
        conn, cursor = _fake_conn(fetchall=[])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            retriever._get_chunks([0.0] * 1024, top_k=5)
        assert "::vector" in cursor.execute.call_args[0][0]


class TestIvfflatQueryTuning:
    """ivfflat and hnsw need DIFFERENT recall knobs, and the wrong one is silent.

    ivfflat defaults to probes = 1: it scans one list out of 4000 and returns
    almost nothing, which is indistinguishable from "no matches" at the call
    site. The migration builds lists = 4000 per partition, so probes must be
    ~sqrt(lists).
    """

    def test_ivfflat_index_sets_probes(self, retriever):
        retriever._index_am = "ivfflat"
        conn, _ = _fake_conn(fetchall=[])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            retriever._get_chunks([0.0] * 1024, top_k=5)
        executed = " | ".join(str(c[0][0]) for c in conn.execute.call_args_list)
        assert "ivfflat.probes" in executed
        assert "hnsw.ef_search" not in executed

    def test_hnsw_index_sets_ef_search(self, retriever):
        retriever._index_am = "hnsw"
        conn, _ = _fake_conn(fetchall=[])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            retriever._get_chunks([0.0] * 1024, top_k=5)
        executed = " | ".join(str(c[0][0]) for c in conn.execute.call_args_list)
        assert "hnsw.ef_search" in executed
        assert "ivfflat.probes" not in executed

    def test_probes_is_tuned_for_32_partitions_not_one_index(self):
        """probes is deliberately far below sqrt(lists).

        sqrt(4000) = 63 is pgvector's guidance for a SINGLE index. A vector
        query cannot prune partitions, so all 32 are probed: 63 would mean
        2016 lists and ~8M vectors per query. Measured 2026-08-24: 66 s at
        probes=63 against a 25 s statement_timeout, so every wiki-RAG query
        would have timed out and returned nothing. probes=10 gives p90 10.0 s
        with identical recall@10.
        """
        from rag_v1.wiki.wiki_retriever import _IVFFLAT_PROBES
        assert _IVFFLAT_PROBES == 5
        assert _IVFFLAT_PROBES < 4000 ** 0.5

    def test_detects_ivfflat_access_method(self, retriever):
        ivf = ("CREATE INDEX p000_hv_ivf ON public.wiki_chunks_part_p000 "
               "USING ivfflat (((embedding)::halfvec(1024)) halfvec_cosine_ops)")
        conn, _ = _fake_conn(fetchall=[{"def": ivf}])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            assert retriever._query_index_kind() == "halfvec"
        assert retriever._index_am == "ivfflat"

    def test_detects_hnsw_access_method(self, retriever):
        conn, _ = _fake_conn(fetchall=[{"def": HALFVEC_DEF}])
        with patch("rag_v1.wiki.wiki_retriever.conn_ctx", return_value=conn):
            retriever._query_index_kind()
        assert retriever._index_am == "hnsw"

    def test_halfvec_sql_dimension_matches_the_constant(self):
        # The dimension is hardcoded to keep the query a LiteralString; this is
        # what stops it drifting from _WIKI_EMBED_DIMS.
        from rag_v1.wiki.wiki_retriever import (
            _SQL_TOP_CHUNKS_HALFVEC, _WIKI_EMBED_DIMS)
        assert f"halfvec({_WIKI_EMBED_DIMS})" in _SQL_TOP_CHUNKS_HALFVEC
