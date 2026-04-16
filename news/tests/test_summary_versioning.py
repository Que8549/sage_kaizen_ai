"""
news/tests/test_summary_versioning.py

Integration tests for article summary versioning.

Verifies that when a new summary is inserted for an article:
  - The old summary row is set is_active=false
  - The new summary row has is_active=true
  - There is exactly one active summary per (article_id, summary_kind)

Requires a live PostgreSQL connection.
"""
import uuid
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

_DEACTIVATE_SQL = """
UPDATE news_article_summaries
SET is_active = false
WHERE article_id = %s::uuid AND is_active = true
"""

_INSERT_SQL = """
INSERT INTO news_article_summaries (
    article_id, run_id, summary_kind,
    summary_short, summary_medium,
    key_points_json, entities_json,
    model_name, prompt_version, is_active
) VALUES (
    %s::uuid, %s::uuid, 'article_short',
    %s, '',
    '[]'::jsonb, '[]'::jsonb,
    'test-model', 'v1', true
)
"""


@pytest.fixture(scope="module")
def cfg():
    return get_news_settings()


@pytest.fixture
def article_id(cfg):
    """
    Create a minimal daily_news row and return its article_id.
    Cleaned up after the test.
    """
    tid_row = None
    with conn_ctx(cfg.pg_dsn) as conn:
        tid_row = conn.execute(
            "SELECT topic_id::text FROM news_topics LIMIT 1"
        ).fetchone()

    if tid_row is None:
        pytest.skip("No topics in DB — run news_seed_data.sql first")

    import hashlib
    test_url = f"https://test.example.com/summary-version-{uuid.uuid4()}"
    url_hash = hashlib.sha256(test_url.encode()).digest()

    with conn_ctx(cfg.pg_dsn) as conn:
        row = conn.execute("""
            INSERT INTO daily_news (
                url, url_hash, headline, snippet,
                news_source, news_source_url, topic_id,
                language_code, search_query, search_category,
                rank_score, dedupe_fingerprint
            ) VALUES (
                %s, %s, 'Test headline', 'Test snippet',
                'test', 'https://test.example.com', %s::uuid,
                'en', 'test', 'news', 0.5, 'aabbccdd'
            )
            RETURNING article_id::text
        """, [test_url, url_hash, tid_row["topic_id"]]).fetchone()
        aid = row["article_id"]

    yield aid

    with conn_ctx(cfg.pg_dsn) as conn:
        conn.execute("DELETE FROM news_article_summaries WHERE article_id = %s::uuid", [aid])
        conn.execute("DELETE FROM daily_news WHERE article_id = %s::uuid", [aid])


def test_first_summary_is_active(cfg, article_id):
    run_id = str(uuid.uuid4())
    with conn_ctx(cfg.pg_dsn) as conn:
        conn.execute(_INSERT_SQL, [article_id, run_id, "First summary"])

    with conn_ctx(cfg.pg_dsn) as conn:
        rows = conn.execute("""
            SELECT is_active FROM news_article_summaries
            WHERE article_id = %s::uuid AND summary_kind = 'article_short'
        """, [article_id]).fetchall()

    assert len(rows) == 1
    assert rows[0]["is_active"] is True


def test_second_summary_deactivates_first(cfg, article_id):
    run_id1 = str(uuid.uuid4())
    run_id2 = str(uuid.uuid4())
    with conn_ctx(cfg.pg_dsn) as conn:
        conn.execute(_INSERT_SQL, [article_id, run_id1, "First"])

    with conn_ctx(cfg.pg_dsn) as conn:
        conn.execute(_DEACTIVATE_SQL, [article_id])
        conn.execute(_INSERT_SQL, [article_id, run_id2, "Second"])

    with conn_ctx(cfg.pg_dsn) as conn:
        rows = conn.execute("""
            SELECT is_active, summary_short FROM news_article_summaries
            WHERE article_id = %s::uuid AND summary_kind = 'article_short'
            ORDER BY created_at
        """, [article_id]).fetchall()

    assert len(rows) == 2
    assert rows[0]["is_active"] is False
    assert rows[0]["summary_short"] == "First"
    assert rows[1]["is_active"] is True
    assert rows[1]["summary_short"] == "Second"


def test_exactly_one_active_summary(cfg, article_id):
    """After N versioning cycles, exactly one row is active."""
    for i in range(3):
        run_id = str(uuid.uuid4())
        with conn_ctx(cfg.pg_dsn) as conn:
            conn.execute(_DEACTIVATE_SQL, [article_id])
            conn.execute(_INSERT_SQL, [article_id, run_id, f"Version {i}"])

    with conn_ctx(cfg.pg_dsn) as conn:
        active_count = conn.execute("""
            SELECT COUNT(*) AS n FROM news_article_summaries
            WHERE article_id = %s::uuid AND is_active = true
        """, [article_id]).fetchone()["n"]

    assert active_count == 1
