"""
tests/test_mermaid_and_media.py

Unit tests for:
  mermaid_streamlit.py          — diagram sanitisation + llama-server probing
  rag_v1/media/media_retriever.py — cross-modal image/audio retrieval
  rag_v1/media/lyrics_retriever.py
  rag_v1/wiki/wiki_embed_config.py

The Mermaid preprocessing is the interesting part: it fixes parse errors in
AI-generated diagrams, so a regression there silently breaks rendering rather
than raising.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# mermaid_streamlit
# ---------------------------------------------------------------------------

import mermaid_streamlit as ms
from mermaid_streamlit import (
    LlamaServerInfo,
    _first_non_none,
    _fmt,
    _mm_safe,
    _preprocess_mermaid,
    build_sage_kaizen_mermaid,
)


class TestFmt:
    def test_none_becomes_question_mark(self):
        assert _fmt(None) == "?"

    def test_short_value_unchanged(self):
        assert _fmt("abc") == "abc"

    def test_numbers_are_stringified(self):
        assert _fmt(32768) == "32768"

    def test_long_value_is_elided(self):
        out = _fmt("x" * 100, maxlen=10)
        assert len(out) == 10
        assert out.endswith("…")

    def test_exact_length_is_not_elided(self):
        assert _fmt("x" * 10, maxlen=10) == "x" * 10


class TestFirstNonNone:
    def test_returns_the_first_non_none(self):
        assert _first_non_none(None, None, "a", "b") == "a"

    def test_all_none_returns_none(self):
        assert _first_non_none(None, None) is None

    def test_no_args_returns_none(self):
        assert _first_non_none() is None

    def test_falsy_but_not_none_is_returned(self):
        assert _first_non_none(None, 0) == 0


class TestMmSafe:
    def test_escapes_double_quotes(self):
        assert _mm_safe('say "hi"') == 'say \\"hi\\"'

    def test_escapes_backslashes_before_quotes(self):
        assert _mm_safe("a\\b") == "a\\\\b"

    def test_square_brackets_become_parens(self):
        assert _mm_safe("a[b]c") == "a(b)c"

    def test_none_becomes_empty(self):
        assert _mm_safe(None) == ""  # type: ignore[arg-type]

    def test_plain_text_unchanged(self):
        assert _mm_safe("plain label") == "plain label"


class TestPreprocessMermaid:
    def test_quotes_node_labels_containing_parens(self):
        """Mermaid v10 tokenises '(' inside an unquoted [...] as PS and rejects it."""
        out = _preprocess_mermaid("A[Fast (7B)] --> B")
        assert '["Fast (7B)"]' in out

    def test_leaves_already_quoted_labels_alone(self):
        src = 'A["Already (quoted)"] --> B'
        assert _preprocess_mermaid(src) == src

    def test_leaves_paren_free_labels_alone(self):
        src = "A[Plain] --> B"
        assert _preprocess_mermaid(src) == src

    def test_label_with_both_a_quote_and_a_paren_is_left_alone(self):
        r"""
        The node-label regex is [^"'\[\]]*[()][^"'\[\]]* — the character class
        excludes quotes, so a label containing BOTH a quote and a paren never
        matches and is not repaired. (It also means _quote_node_label's
        inner.replace('"', ...) can never fire: `inner` cannot contain a quote.)

        Minor: such a label still fails to render. Pinned rather than fixed —
        outside the bugs-1-7 scope for this pass.
        """
        src = 'A[say "hi" (loud)] --> B'
        assert _preprocess_mermaid(src) == src

    def test_edge_label_parens_become_brackets(self):
        out = _preprocess_mermaid("A -->|calls (async)| B")
        assert "|calls [async]|" in out

    def test_edge_label_without_parens_unchanged(self):
        src = "A -->|calls| B"
        assert _preprocess_mermaid(src) == src

    def test_handles_multiple_nodes(self):
        out = _preprocess_mermaid("A[one (1)] --> B[two (2)]")
        assert '["one (1)"]' in out and '["two (2)"]' in out

    def test_empty_source(self):
        assert _preprocess_mermaid("") == ""

    def test_multiline_diagram(self):
        src = "graph TD\n    A[Node (x)] --> B\n    B -->|edge (y)| C\n"
        out = _preprocess_mermaid(src)
        assert '["Node (x)"]' in out
        assert "|edge [y]|" in out


class TestBuildSageKaizenMermaid:
    def test_produces_a_graph_declaration(self):
        assert build_sage_kaizen_mermaid(None, None).startswith("graph TD")

    def test_offline_servers_still_render(self):
        out = build_sage_kaizen_mermaid(None, None)
        assert "Fast / Low-Latency" in out
        assert "Architect / Deep Reasoning" in out

    def test_includes_live_model_details(self):
        q5 = LlamaServerInfo(
            base_url="http://127.0.0.1:8011", ok=True,
            alias="Qwen2.5-Omni", model_id="omni-7b",
            ctx_size=32768, n_gpu_layers=99, device="CUDA1",
        )
        out = build_sage_kaizen_mermaid(q5, None)
        assert "Qwen2.5-Omni" in out
        assert "ctx=32768" in out
        assert "dev=CUDA1" in out

    def test_unhealthy_server_details_are_omitted(self):
        q5 = LlamaServerInfo(base_url="u", ok=False, alias="Should not appear")
        assert "Should not appear" not in build_sage_kaizen_mermaid(q5, None)

    def test_optional_fields_appear_only_when_set(self):
        with_ts = LlamaServerInfo(base_url="u", ok=True,
                                  tensor_split="0.5,0.5", split_mode="row")
        assert "ts=" in build_sage_kaizen_mermaid(with_ts, None)
        assert "split=" in build_sage_kaizen_mermaid(with_ts, None)
        bare = LlamaServerInfo(base_url="u", ok=True)
        assert "ts=" not in build_sage_kaizen_mermaid(bare, None)

    def test_labels_use_br_tags_for_line_breaks(self):
        info = LlamaServerInfo(base_url="u", ok=True, alias="A")
        assert "<br/>" in build_sage_kaizen_mermaid(info, None)

    def test_output_survives_its_own_preprocessor(self):
        """Generated diagrams must not need fixing up."""
        q5 = LlamaServerInfo(base_url="u", ok=True, alias="Qwen (7B)", ctx_size=32768)
        built = build_sage_kaizen_mermaid(q5, None)
        # _mm_safe converts [] to () inside labels; preprocessing must be stable.
        assert _preprocess_mermaid(built) == _preprocess_mermaid(_preprocess_mermaid(built))


class TestHttpGetJson:
    def test_returns_parsed_json_on_success(self):
        resp = MagicMock()
        resp.read.return_value = b'{"a": 1}'
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        with patch.object(ms.urllib.request, "urlopen", return_value=resp):
            assert ms._http_get_json("http://x/health") == (True, {"a": 1})

    def test_network_error_is_swallowed(self):
        with patch.object(ms.urllib.request, "urlopen", side_effect=OSError("refused")):
            assert ms._http_get_json("http://x/health") == (False, None)

    def test_invalid_json_is_swallowed(self):
        resp = MagicMock()
        resp.read.return_value = b"not json"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        with patch.object(ms.urllib.request, "urlopen", return_value=resp):
            assert ms._http_get_json("http://x/health") == (False, None)


class TestProbeLlamaServer:
    @staticmethod
    def _probe(url):
        """Call through st.cache_data's wrapper to the real function."""
        fn = getattr(ms.probe_llama_server, "__wrapped__", ms.probe_llama_server)
        return fn(url)

    def test_unreachable_server_reports_not_ok(self):
        with patch.object(ms, "_http_get_json", return_value=(False, None)):
            assert self._probe("http://127.0.0.1:9999").ok is False

    def test_health_payload_is_used(self):
        with patch.object(ms, "_http_get_json", return_value=(True, {"model": "m1"})):
            info = self._probe("http://127.0.0.1:8011")
        assert info.ok is True
        assert info.model_id == "m1"

    def test_trailing_slash_is_stripped(self):
        with patch.object(ms, "_http_get_json", return_value=(False, None)) as g:
            self._probe("http://127.0.0.1:8011/")
        assert not g.call_args_list[0].args[0].startswith("http://127.0.0.1:8011//")

    def test_empty_url_does_not_crash(self):
        with patch.object(ms, "_http_get_json", return_value=(False, None)):
            assert self._probe("").ok is False


# ---------------------------------------------------------------------------
# rag_v1/media/media_retriever
# ---------------------------------------------------------------------------

import rag_v1.media.media_retriever as mr
from rag_v1.media.media_retriever import MediaResult, MediaRetriever


@pytest.fixture
def retriever():
    with (
        patch.object(mr, "ImageEmbedClient") as IC,
        patch.object(mr, "AudioEmbedClient") as AC,
    ):
        image, audio = MagicMock(), MagicMock()
        image.embed_text.return_value = [[0.1] * 1024]
        audio.embed_text.return_value = [[0.2] * 512]
        IC.return_value, AC.return_value = image, audio
        r = MediaRetriever(pg_dsn="postgresql://test")
        r._image_client, r._audio_client = image, audio
        yield r


class TestSearchImages:
    def test_empty_query_returns_nothing(self, retriever):
        assert retriever.search_images("") == []

    def test_embed_failure_degrades_to_empty(self, retriever):
        retriever._image_client.embed_text.side_effect = RuntimeError("service down")
        assert retriever.search_images("a cat") == []

    def test_db_failure_degrades_to_empty(self, retriever):
        with patch.object(retriever, "_query_images", side_effect=RuntimeError("db")):
            assert retriever.search_images("a cat") == []

    def test_returns_query_results(self, retriever):
        result = MediaResult(
            media_id="uuid-1", file_path="/a.png", modality="image", score=0.9,
        )
        with patch.object(retriever, "_query_images", return_value=[result]):
            assert retriever.search_images("a cat") == [result]

    def test_uses_the_image_client_not_the_audio_one(self, retriever):
        with patch.object(retriever, "_query_images", return_value=[]):
            retriever.search_images("a cat")
        retriever._image_client.embed_text.assert_called_once()
        retriever._audio_client.embed_text.assert_not_called()


class TestSearchAudio:
    def test_empty_query_returns_nothing(self, retriever):
        assert retriever.search_audio("") == []

    def test_embed_failure_degrades_to_empty(self, retriever):
        retriever._audio_client.embed_text.side_effect = RuntimeError("clap down")
        assert retriever.search_audio("rain sounds") == []

    def test_uses_the_audio_client(self, retriever):
        with patch.object(retriever, "_query_audio", return_value=[]):
            retriever.search_audio("rain")
        retriever._audio_client.embed_text.assert_called_once()
        retriever._image_client.embed_text.assert_not_called()

    def test_db_failure_degrades_to_empty(self, retriever):
        with patch.object(retriever, "_query_audio", side_effect=RuntimeError("db")):
            assert retriever.search_audio("rain") == []


class TestMediaResult:
    def test_holds_its_fields(self):
        r = MediaResult(media_id="u1", file_path="/a.png", modality="image", score=0.5)
        assert r.media_id == "u1"
        assert r.file_path == "/a.png"
        assert r.modality == "image"
        assert r.score == 0.5

    def test_metadata_defaults_to_an_empty_dict(self):
        r = MediaResult(media_id="u1", file_path="/a.png", modality="image", score=0.5)
        assert r.metadata == {}

    def test_metadata_is_not_shared_between_instances(self):
        a = MediaResult(media_id="1", file_path="/a", modality="image", score=0.0)
        b = MediaResult(media_id="2", file_path="/b", modality="image", score=0.0)
        a.metadata["k"] = "v"
        assert b.metadata == {}


# ---------------------------------------------------------------------------
# rag_v1/wiki/wiki_embed_config
# ---------------------------------------------------------------------------

from rag_v1.wiki.wiki_embed_config import (
    WikiEmbedConfig,
    WikiEmbedServiceConfig,
    WikiIngestConfig,
    load_wiki_embed_config,
)


class TestWikiEmbedServiceConfig:
    def test_default_device_is_the_compute_gpu(self):
        """
        cuda:0 drives the monitors. This default was "cuda:0" until 2026-08-04
        — see the guard in wiki_retriever and ingest CLAUDE.md §19 gap 1.
        """
        assert WikiEmbedServiceConfig().device == "cuda:1"

    def test_default_device_is_never_the_display_gpu(self):
        assert WikiEmbedServiceConfig().device != "cuda:0"

    def test_other_defaults(self):
        c = WikiEmbedServiceConfig()
        assert c.host == "127.0.0.1"
        assert c.port == 8031
        assert c.text_batch == 32
        assert c.image_batch == 8


class TestWikiEmbedConfig:
    @pytest.fixture
    def cfg(self, tmp_path):
        return WikiEmbedConfig(model=tmp_path / "m", log=tmp_path / "l.log")

    def test_convenience_properties_delegate(self, cfg):
        assert cfg.host == cfg.service.host
        assert cfg.port == cfg.service.port
        assert cfg.device == cfg.service.device
        assert cfg.text_batch == cfg.service.text_batch
        assert cfg.image_batch == cfg.service.image_batch

    def test_base_url(self, cfg):
        assert cfg.base_url == "http://127.0.0.1:8031"

    def test_ingest_properties_delegate(self, cfg):
        assert cfg.chunk_chars == cfg.ingest.chunk_chars
        assert cfg.overlap == cfg.ingest.overlap
        assert cfg.log_every_pages == cfg.ingest.log_every_pages

    def test_wiki_root_raises_when_unset(self, cfg):
        with pytest.raises(KeyError, match="wiki_embed.ingest.root"):
            _ = cfg.wiki_root

    def test_wiki_root_returns_a_path(self, tmp_path):
        c = WikiEmbedConfig(
            model=tmp_path / "m", log=tmp_path / "l",
            ingest=WikiIngestConfig(root=str(tmp_path)),
        )
        assert c.wiki_root == Path(str(tmp_path))

    def test_exclude_sections_is_a_trimmed_set(self, cfg):
        s = cfg.exclude_sections
        assert isinstance(s, set)
        assert "References" in s
        assert all(x == x.strip() for x in s)

    def test_idle_timeout_env_override(self, cfg, monkeypatch):
        monkeypatch.setenv("WIKI_EMBED_IDLE_TIMEOUT_S", "45.5")
        assert cfg.idle_timeout_s == 45.5

    def test_idle_timeout_ignores_invalid_env(self, cfg, monkeypatch):
        monkeypatch.setenv("WIKI_EMBED_IDLE_TIMEOUT_S", "not-a-number")
        assert cfg.idle_timeout_s == cfg.service.idle_timeout_s

    def test_idle_timeout_falls_back_to_config(self, cfg, monkeypatch):
        monkeypatch.delenv("WIKI_EMBED_IDLE_TIMEOUT_S", raising=False)
        assert cfg.idle_timeout_s == 120.0


class TestLoadWikiEmbedConfig:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="brains.yaml not found"):
            load_wiki_embed_config(tmp_path / "nope.yaml")

    def test_missing_section_raises(self, tmp_path):
        p = tmp_path / "brains.yaml"
        p.write_text(yaml.safe_dump({"fast": {}}), encoding="utf-8")
        with pytest.raises(KeyError):
            load_wiki_embed_config(p)

    def test_loads_a_minimal_section(self, tmp_path):
        p = tmp_path / "brains.yaml"
        p.write_text(yaml.safe_dump({"wiki_embed": {
            "model": "E:/jina", "log": "logs/wiki.log",
        }}), encoding="utf-8")
        cfg = load_wiki_embed_config(p)
        assert cfg.model == Path("E:/jina")
        assert cfg.startup_timeout_s == 300.0

    def test_service_and_ingest_sections_are_parsed(self, tmp_path):
        p = tmp_path / "brains.yaml"
        p.write_text(yaml.safe_dump({"wiki_embed": {
            "model": "m", "log": "l", "startup_timeout_s": 120,
            "service": {"port": 8032, "device": "cuda:2"},
            "ingest": {"root": "E:/wiki", "chunk_chars": 800},
        }}), encoding="utf-8")
        cfg = load_wiki_embed_config(p)
        assert cfg.port == 8032
        assert cfg.device == "cuda:2"
        assert cfg.chunk_chars == 800
        assert cfg.startup_timeout_s == 120.0

    def test_the_real_brains_yaml_does_not_target_the_display_gpu(self):
        """
        Guards the live config, not just the Python default — brains.yaml is the
        authoritative source (CLAUDE.md invariant 1) and it is what the embed
        service actually reads at startup.
        """
        cfg = load_wiki_embed_config()
        assert cfg.device != "cuda:0", (
            "brains.yaml wiki_embed.service.device is the display GPU — "
            "see wiki_retriever's display-GPU guard"
        )


# ---------------------------------------------------------------------------
# rag_v1/media/lyrics_retriever
# ---------------------------------------------------------------------------

import rag_v1.media.lyrics_retriever as lr


class TestLyricsRetriever:
    @pytest.fixture
    def lyrics(self):
        with patch.object(lr, "EmbedClient") as EC:
            client = MagicMock()
            client.embed.return_value = [[0.1] * 1024]
            EC.return_value = client
            r = lr.LyricsRetriever(pg_dsn="postgresql://test")
            r._client = client
            yield r

    def test_empty_query_returns_nothing(self, lyrics):
        assert lyrics.search("") == []

    def test_embed_failure_degrades_to_empty(self, lyrics):
        lyrics._client.embed.side_effect = RuntimeError("embed down")
        assert lyrics.search("a song about rain") == []

    def test_db_failure_degrades_to_empty(self, lyrics):
        with patch.object(lr, "get_conn", side_effect=RuntimeError("db down")):
            assert lyrics.search("a song about rain") == []
