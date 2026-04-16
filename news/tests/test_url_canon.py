"""
news/tests/test_url_canon.py

Unit tests for URL canonicalization in SearXNGNewsClient.

Tests that tracking parameters are stripped, HTTP is upgraded to HTTPS,
hostname is lowercased, trailing slashes are removed, and the SHA-256
url_hash is stable across equivalent forms.
"""
import hashlib
import pytest

from news.collectors.searxng_news_client import SearXNGNewsClient


@pytest.fixture
def client():
    return SearXNGNewsClient(base_url="http://localhost:8080")


def test_strips_utm_params(client):
    url = "https://example.com/article?utm_source=twitter&utm_medium=social&q=news"
    result = client._canonicalize_url(url)
    assert "utm_source" not in result
    assert "utm_medium" not in result
    assert "q=news" in result


def test_strips_fbclid(client):
    url = "https://example.com/story?fbclid=abc123"
    result = client._canonicalize_url(url)
    assert "fbclid" not in result


def test_http_upgraded_to_https(client):
    url = "http://example.com/article"
    result = client._canonicalize_url(url)
    assert result.startswith("https://")


def test_https_unchanged(client):
    url = "https://example.com/article"
    result = client._canonicalize_url(url)
    assert result.startswith("https://")


def test_hostname_lowercased(client):
    url = "https://Example.COM/article"
    result = client._canonicalize_url(url)
    assert "example.com" in result
    assert "Example.COM" not in result


def test_trailing_slash_stripped(client):
    url = "https://example.com/article/"
    result = client._canonicalize_url(url)
    # Trailing slash on non-root paths removed
    assert not result.endswith("/") or result == "https://example.com/"


def test_url_hash_stable_for_equivalent_urls(client):
    url_a = "http://Example.COM/article?utm_source=tw&real=1"
    url_b = "https://example.com/article?real=1"
    canon_a = client._canonicalize_url(url_a)
    canon_b = client._canonicalize_url(url_b)
    assert canon_a == canon_b

    hash_a = hashlib.sha256(canon_a.encode()).digest()
    hash_b = hashlib.sha256(canon_b.encode()).digest()
    assert hash_a == hash_b


def test_multiple_tracking_params_all_stripped(client):
    url = (
        "https://news.site/story?id=42"
        "&utm_source=x&utm_medium=y&utm_campaign=z"
        "&utm_content=a&utm_term=b&gclid=c&msclkid=d"
        "&ref=e&source=f"
    )
    result = client._canonicalize_url(url)
    for param in ("utm_source", "utm_medium", "utm_campaign",
                  "utm_content", "utm_term", "gclid", "msclkid"):
        assert param not in result
    assert "id=42" in result
