"""
tests/test_search_models.py

Unit tests for search/models.py and search/citations.py.

Pure data-model tests — no DB, no network.
"""
from __future__ import annotations

import pytest
from search.models import SearchEvidence, WebResult
from search.citations import format_search_sources_markdown


# ---------------------------------------------------------------------------
# WebResult
# ---------------------------------------------------------------------------

class TestWebResult:
    def test_construction(self):
        r = WebResult(
            title="Test",
            url="https://example.com",
            snippet="A snippet",
            source_engine="brave",
            category="general",
            score=0.9,
        )
        assert r.title == "Test"
        assert r.url == "https://example.com"
        assert r.published_date is None

    def test_frozen(self):
        r = WebResult(
            title="T", url="u", snippet="s",
            source_engine="brave", category="general", score=1.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            r.title = "new"  # type: ignore[misc]

    def test_with_published_date(self):
        r = WebResult(
            title="T", url="u", snippet="s",
            source_engine="ddg", category="news", score=0.5,
            published_date="2026-01-15",
        )
        assert r.published_date == "2026-01-15"


# ---------------------------------------------------------------------------
# SearchEvidence
# ---------------------------------------------------------------------------

class TestSearchEvidence:
    def _make_result(self, url="https://example.com") -> WebResult:
        return WebResult(
            title="Title", url=url, snippet="Snippet",
            source_engine="brave", category="general", score=0.8,
        )

    def test_empty_when_no_results(self):
        ev = SearchEvidence(
            query="test",
            results=(),
            fetched_at="2026-01-01T00:00:00Z",
            categories_queried=("general",),
        )
        assert ev.empty is True

    def test_not_empty_with_results(self):
        ev = SearchEvidence(
            query="test",
            results=(self._make_result(),),
            fetched_at="2026-01-01T00:00:00Z",
            categories_queried=("general",),
        )
        assert ev.empty is False

    def test_summarized_text_defaults_to_empty(self):
        ev = SearchEvidence(
            query="q", results=(), fetched_at="2026-01-01T00:00:00Z",
            categories_queried=(),
        )
        assert ev.summarized_text == ""

    def test_frozen(self):
        ev = SearchEvidence(
            query="q", results=(), fetched_at="2026-01-01T00:00:00Z",
            categories_queried=(),
        )
        with pytest.raises((AttributeError, TypeError)):
            ev.query = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# format_search_sources_markdown
# ---------------------------------------------------------------------------

class TestFormatSearchSourcesMarkdown:
    def _make_evidence(self, results=(), **kwargs) -> SearchEvidence:
        return SearchEvidence(
            query="test query",
            results=results,
            fetched_at="2026-06-12T14:30:00Z",
            categories_queried=("general", "news"),
            **kwargs,
        )

    def test_returns_empty_for_no_results(self):
        ev = self._make_evidence()
        assert format_search_sources_markdown(ev) == ""

    def test_contains_live_web_header(self):
        r = WebResult(title="Example", url="https://ex.com", snippet="s",
                      source_engine="brave", category="general", score=0.9)
        ev = self._make_evidence(results=(r,))
        md = format_search_sources_markdown(ev)
        assert "**Live Web**" in md

    def test_contains_categories(self):
        r = WebResult(title="T", url="u", snippet="s",
                      source_engine="brave", category="general", score=1.0)
        ev = self._make_evidence(results=(r,))
        md = format_search_sources_markdown(ev)
        assert "general" in md
        assert "news" in md

    def test_contains_result_title_and_url(self):
        r = WebResult(title="My Article", url="https://mysite.com/article",
                      snippet="s", source_engine="ddg", category="news", score=0.7)
        ev = self._make_evidence(results=(r,))
        md = format_search_sources_markdown(ev)
        assert "My Article" in md
        assert "https://mysite.com/article" in md

    def test_contains_engine_name(self):
        r = WebResult(title="T", url="u", snippet="s",
                      source_engine="startpage", category="general", score=0.5)
        ev = self._make_evidence(results=(r,))
        md = format_search_sources_markdown(ev)
        assert "startpage" in md

    def test_published_date_included_when_present(self):
        r = WebResult(title="T", url="u", snippet="s", source_engine="brave",
                      category="news", score=0.9, published_date="2026-06-10")
        ev = self._make_evidence(results=(r,))
        md = format_search_sources_markdown(ev)
        assert "2026-06-10" in md

    def test_fetched_at_timestamp_formatted(self):
        r = WebResult(title="T", url="u", snippet="s",
                      source_engine="brave", category="general", score=1.0)
        ev = self._make_evidence(results=(r,))
        md = format_search_sources_markdown(ev)
        # ISO T is replaced with space; seconds stripped
        assert "2026-06-12 14:30 UTC" in md

    def test_multiple_results_each_appear(self):
        results = tuple(
            WebResult(title=f"Article {i}", url=f"https://site{i}.com",
                      snippet="s", source_engine="brave", category="general", score=0.9)
            for i in range(3)
        )
        ev = self._make_evidence(results=results)
        md = format_search_sources_markdown(ev)
        for i in range(3):
            assert f"Article {i}" in md
