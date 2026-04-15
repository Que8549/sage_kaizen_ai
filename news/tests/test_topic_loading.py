"""
news/tests/test_topic_loading.py

Integration tests for topic and query loading in TopicCollector.

Verifies that:
  - Only enabled topics are loaded
  - Disabled topics are excluded
  - Each loaded topic has at least one associated query template
  - Topic slugs match expected seed values

Requires a live PostgreSQL connection.
"""
import pytest

try:
    from news.news_settings import get_news_settings
    from news.collectors.topic_collector import TopicCollector
    from rag_v1.db.pg import conn_ctx
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _DB_AVAILABLE, reason="news package or DB not available"
)

# Expected seed topics from news_seed_data.sql
_EXPECTED_SLUGS = {
    "ai", "technology", "science", "world", "business",
    "cybersecurity", "united_kingdom", "atlanta_georgia",
    "csharp", "python", "computer_programming",
}


@pytest.fixture(scope="module")
def cfg():
    return get_news_settings()


@pytest.fixture(scope="module")
def collector(cfg):
    return TopicCollector()


def test_loads_only_enabled_topics(collector):
    topics = collector._load_topics()
    loaded_slugs = {t["topic_slug"] for t in topics}
    # All loaded topics must be enabled
    assert len(loaded_slugs) > 0, "No topics loaded — run news_seed_data.sql"
    # Every expected slug should be present (all seeded as enabled)
    for slug in _EXPECTED_SLUGS:
        assert slug in loaded_slugs, f"Expected topic '{slug}' not found"


def test_all_topics_have_queries(collector, cfg):
    topics = collector._load_topics()
    with conn_ctx(cfg.pg_dsn) as conn:
        for topic in topics:
            rows = conn.execute("""
                SELECT COUNT(*) AS n FROM news_topic_queries
                WHERE topic_id = %s::uuid AND is_enabled = true
            """, [topic["topic_id"]]).fetchone()
            assert rows["n"] > 0, (
                f"Topic '{topic['topic_slug']}' has no enabled queries"
            )


def test_no_disabled_topics_loaded(collector, cfg):
    """If any topics are disabled, they must not appear in the loaded set."""
    with conn_ctx(cfg.pg_dsn) as conn:
        disabled = conn.execute("""
            SELECT topic_slug FROM news_topics WHERE is_enabled = false
        """).fetchall()

    topics = collector._load_topics()
    loaded_slugs = {t["topic_slug"] for t in topics}

    for row in disabled:
        assert row["topic_slug"] not in loaded_slugs, (
            f"Disabled topic '{row['topic_slug']}' was loaded"
        )


def test_topic_count_matches_seed(collector):
    topics = collector._load_topics()
    assert len(topics) == len(_EXPECTED_SLUGS), (
        f"Expected {len(_EXPECTED_SLUGS)} topics, loaded {len(topics)}"
    )
