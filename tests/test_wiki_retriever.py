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

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wiki_cfg(tmp_path):
    """A minimal wiki_embed_config mock."""
    cfg = MagicMock()
    cfg.wiki_root  = tmp_path
    cfg.host       = "127.0.0.1"
    cfg.port       = 8031
    cfg.startup_timeout_s = 10
    cfg.log        = tmp_path / "wiki_embed.log"
    return cfg


@pytest.fixture
def retriever(wiki_cfg):
    """A WikiRetriever with all external dependencies mocked."""
    with (
        patch("rag_v1.wiki.wiki_retriever.load_wiki_embed_config", return_value=wiki_cfg),
        patch("rag_v1.wiki.wiki_retriever.MmEmbedClient") as MockClient,
    ):
        client = MagicMock()
        client.ping.return_value = False          # service not running by default
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
    def test_returns_true_when_ping_succeeds(self, retriever):
        retriever._client.ping.return_value = True
        assert retriever._ensure_service() is True

    def test_calls_warmup_on_ping_success(self, retriever):
        retriever._client.ping.return_value = True
        retriever._warmed_up = False
        retriever._ensure_service()
        retriever._client.embed_text.assert_called_once_with(["warmup"])

    def test_does_not_call_warmup_twice(self, retriever):
        retriever._client.ping.return_value = True
        retriever._warmed_up = True
        retriever._ensure_service()
        retriever._client.embed_text.assert_not_called()


# ---------------------------------------------------------------------------
# _ensure_service — slow path (starts the service)
# ---------------------------------------------------------------------------

class TestEnsureServiceSlowPath:
    def test_starts_process_when_ping_fails(self, retriever, tmp_path):
        # ping: first call False, then True (after startup)
        retriever._client.ping.side_effect = [False, True]

        with (
            patch("rag_v1.wiki.wiki_retriever.subprocess.Popen") as MockPopen,
            patch("rag_v1.wiki.wiki_retriever.time.sleep"),
        ):
            MockPopen.return_value = MagicMock(poll=MagicMock(return_value=None))
            result = retriever._ensure_service()

        assert result is True
        MockPopen.assert_called_once()

    def test_calls_warmup_after_startup_ping(self, retriever, tmp_path):
        retriever._client.ping.side_effect = [False, True]
        retriever._warmed_up = False

        with (
            patch("rag_v1.wiki.wiki_retriever.subprocess.Popen") as MockPopen,
            patch("rag_v1.wiki.wiki_retriever.time.sleep"),
        ):
            MockPopen.return_value = MagicMock(poll=MagicMock(return_value=None))
            retriever._ensure_service()

        retriever._client.embed_text.assert_called_with(["warmup"])

    def test_returns_false_on_startup_timeout(self, retriever, tmp_path):
        # ping always returns False → times out
        retriever._client.ping.return_value = False
        retriever._startup_timeout_s = 0  # immediate timeout

        with (
            patch("rag_v1.wiki.wiki_retriever.subprocess.Popen") as MockPopen,
            patch("rag_v1.wiki.wiki_retriever.time.sleep"),
        ):
            MockPopen.return_value = MagicMock(poll=MagicMock(return_value=None))
            result = retriever._ensure_service()

        assert result is False


# ---------------------------------------------------------------------------
# search() — graceful degradation
# ---------------------------------------------------------------------------

class TestSearch:
    def test_returns_empty_when_service_unavailable(self, retriever):
        retriever._client.ping.return_value = False
        retriever._startup_timeout_s = 0

        with (
            patch("rag_v1.wiki.wiki_retriever.subprocess.Popen"),
            patch("rag_v1.wiki.wiki_retriever.time.sleep"),
        ):
            result = retriever.search("what is a snail")

        assert result.empty is True

    def test_returns_empty_when_embed_fails(self, retriever):
        retriever._client.ping.return_value = True
        retriever._client.embed_text.side_effect = RuntimeError("CUDA OOM")

        result = retriever.search("hello")
        assert result.empty is True

    def test_returns_empty_when_no_chunks_pass_threshold(self, retriever):
        retriever._client.ping.return_value = True
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
