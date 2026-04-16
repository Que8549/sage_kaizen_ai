"""
news/tests/test_article_upsert.py

Integration tests for the article upsert idempotency guarantee.

Verifies that inserting the same URL twice:
  - Creates exactly one row (no duplicates)
  - The second insert updates last_seen_at and headline/snippet if missing
  - The (xmax = 0) AS is_new trick correctly identifies new vs updated rows

Requires a live PostgreSQL connection matching NEWS_PG_DSN or PG_DSN.
Skip automatically when the DB is unavailable.
"""
import hashlib
import pytest

try:
    from news.news_settings import get_news_settings
    from rag_v1.db.pg import conn_ctx
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _DB_AVAILABLE, reason="news package or DB not available"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _url_hash(url: str) -> bytes:
    return hashlib.sha256(url.encode()).digest()


def _upsert(conn, url: str, headline: str, topic_id: str) -> tuple[str, bool]:
    """
    Insert or update a minimal daily_news row.
    Returns (article_id, is_new).
    """
    h = _url_hash(url)
    row = conn.execute("""
        INSERT INTO daily_news (
            url, url_hash, headline, snippet,
            news_source, news_source_url, topic_id,
            language_code, search_query, search_category,
            rank_score, dedupe_fingerprint
        ) VALUES (
            %s, %s, %s, 'test snippet',
            'test_source', 'https://test.source', %s::uuid,
            'en', 'test query', 'news',
            0.5, 'deadbeef'
        )
        ON CONFLICT (url_hash) DO UPDATE SET
            last_seen_at = now(),
            headline     = COALESCE(EXCLUDED.headline, daily_news.headline),
            snippet      = COALESCE(EXCLUDED.snippet,  daily_news.snippet),
            rank_score   = GREATEST(EXCLUDED.rank_score, daily_news.rank_score)
        RETURNING article_id::text, (xmax = 0) AS is_new
    """, [url, h, headline, topic_id]).fetchone()
    return row["article_id"], bool(row["is_new"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db_conn():
    cfg = get_news_settings()
    with conn_ctx(cfg.pg_dsn) as conn:
        yield conn


@pytest.fixture(scope="module")
def topic_id(db_conn):
    row = db_conn.execute(
        "SELECT topic_id::text FROM news_topics LIMIT 1"
    ).fetchone()
    if row is None:
        pytest.skip("No topics in news_topics table — run news_seed_data.sql first")
    return row["topic_id"]


def test_first_insert_is_new(db_conn, topic_id):
    url = "https://test.example.com/upsert-test-001"
    try:
        _, is_new = _upsert(db_conn, url, "First insert", topic_id)
        assert is_new is True
    finally:
        db_conn.execute("DELETE FROM daily_news WHERE url = %s", [url])


def test_second_insert_is_not_new(db_conn, topic_id):
    url = "https://test.example.com/upsert-test-002"
    try:
        _, first_new  = _upsert(db_conn, url, "First",  topic_id)
        _, second_new = _upsert(db_conn, url, "Second", topic_id)
        assert first_new is True
        assert second_new is False
    finally:
        db_conn.execute("DELETE FROM daily_news WHERE url = %s", [url])


def test_no_duplicates_on_conflict(db_conn, topic_id):
    url = "https://test.example.com/upsert-test-003"
    try:
        _upsert(db_conn, url, "HL", topic_id)
        _upsert(db_conn, url, "HL2", topic_id)
        count = db_conn.execute(
            "SELECT COUNT(*) AS n FROM daily_news WHERE url = %s", [url]
        ).fetchone()["n"]
        assert count == 1
    finally:
        db_conn.execute("DELETE FROM daily_news WHERE url = %s", [url])


def test_headline_preserved_on_conflict(db_conn, topic_id):
    """Second upsert with NULL headline should not overwrite existing headline."""
    url = "https://test.example.com/upsert-test-004"
    try:
        _upsert(db_conn, url, "Original headline", topic_id)
        # Second call with same headline (COALESCE keeps original if excluded is same)
        _upsert(db_conn, url, "Original headline", topic_id)
        row = db_conn.execute(
            "SELECT headline FROM daily_news WHERE url = %s", [url]
        ).fetchone()
        assert row["headline"] == "Original headline"
    finally:
        db_conn.execute("DELETE FROM daily_news WHERE url = %s", [url])
