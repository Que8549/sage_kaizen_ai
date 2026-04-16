"""
news/tests/test_image_pipeline.py

Integration tests for the news image pipeline.

Verifies that:
  - An image row in news_article_images references a valid article
  - Re-embedding an existing image deactivates the old embedding and inserts a new one
  - Exactly one active embedding exists per image after re-embed

Requires a live PostgreSQL connection.
The pipeline itself (download + jina-clip-v2) is NOT invoked — we test only
the DB state machine using direct SQL.
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


@pytest.fixture(scope="module")
def cfg():
    return get_news_settings()


@pytest.fixture
def article_and_image(cfg):
    """Create a minimal article + image row. Yields (article_id, image_id)."""
    import hashlib

    tid_row = None
    with conn_ctx(cfg.pg_dsn) as conn:
        tid_row = conn.execute(
            "SELECT topic_id::text FROM news_topics LIMIT 1"
        ).fetchone()
    if tid_row is None:
        pytest.skip("No topics in DB — run news_seed_data.sql first")

    test_url = f"https://test.example.com/img-pipeline-{uuid.uuid4()}"
    url_hash = hashlib.sha256(test_url.encode()).digest()

    with conn_ctx(cfg.pg_dsn) as conn:
        a_row = conn.execute("""
            INSERT INTO daily_news (
                url, url_hash, headline, snippet,
                news_source, news_source_url, topic_id,
                language_code, search_query, search_category,
                rank_score, dedupe_fingerprint
            ) VALUES (
                %s, %s, 'Img test headline', 'Img snippet',
                'test_img', 'https://test.example.com', %s::uuid,
                'en', 'test', 'news', 0.5, 'img0001'
            )
            RETURNING article_id::text
        """, [test_url, url_hash, tid_row["topic_id"]]).fetchone()
        article_id = a_row["article_id"]

        img_row = conn.execute("""
            INSERT INTO news_article_images (
                article_id, image_url, image_url_hash,
                source_page_url, image_kind, fetch_status
            ) VALUES (
                %s::uuid,
                'https://img.example.com/photo.jpg',
                %s,
                %s,
                'og_image',
                'pending'
            )
            RETURNING image_id::text
        """, [
            article_id,
            hashlib.sha256(b"https://img.example.com/photo.jpg").digest(),
            test_url,
        ]).fetchone()
        image_id = img_row["image_id"]

    yield article_id, image_id

    with conn_ctx(cfg.pg_dsn) as conn:
        conn.execute("DELETE FROM news_image_embeddings WHERE image_id = %s::uuid", [image_id])
        conn.execute("DELETE FROM news_article_images WHERE image_id = %s::uuid", [image_id])
        conn.execute("DELETE FROM daily_news WHERE article_id = %s::uuid", [article_id])


def _insert_fake_embedding(conn, image_id: str, article_id: str, topic_id: str,
                            vec_val: float = 0.1):
    """Insert a 1024-dim embedding with a single repeated value."""
    vec = "[" + ",".join([str(vec_val)] * 1024) + "]"
    conn.execute("""
        INSERT INTO news_image_embeddings (
            image_id, article_id, topic_id,
            embedding, model_name, embed_version, is_active
        ) VALUES (
            %s::uuid, %s::uuid, %s::uuid,
            %s::vector, 'test-model', 'v1', true
        )
    """, [image_id, article_id, topic_id, vec])


def test_article_image_fk_exists(cfg, article_and_image):
    article_id, image_id = article_and_image
    with conn_ctx(cfg.pg_dsn) as conn:
        row = conn.execute("""
            SELECT article_id::text FROM news_article_images
            WHERE image_id = %s::uuid
        """, [image_id]).fetchone()
    assert row is not None
    assert row["article_id"] == article_id


def test_re_embed_deactivates_old(cfg, article_and_image):
    article_id, image_id = article_and_image

    with conn_ctx(cfg.pg_dsn) as conn:
        tid_row = conn.execute(
            "SELECT topic_id::text FROM news_topics LIMIT 1"
        ).fetchone()
        topic_id = tid_row["topic_id"]

        # Insert first embedding
        _insert_fake_embedding(conn, image_id, article_id, topic_id, 0.1)

    with conn_ctx(cfg.pg_dsn) as conn:
        # Deactivate + re-embed (simulating re-embed cycle)
        conn.execute("""
            UPDATE news_image_embeddings SET is_active = false
            WHERE image_id = %s::uuid AND is_active = true
        """, [image_id])
        _insert_fake_embedding(conn, image_id, article_id, topic_id, 0.2)

    with conn_ctx(cfg.pg_dsn) as conn:
        rows = conn.execute("""
            SELECT is_active FROM news_image_embeddings
            WHERE image_id = %s::uuid
            ORDER BY created_at
        """, [image_id]).fetchall()

    assert len(rows) == 2
    assert rows[0]["is_active"] is False
    assert rows[1]["is_active"] is True


def test_exactly_one_active_embedding(cfg, article_and_image):
    article_id, image_id = article_and_image

    with conn_ctx(cfg.pg_dsn) as conn:
        tid_row = conn.execute(
            "SELECT topic_id::text FROM news_topics LIMIT 1"
        ).fetchone()
        topic_id = tid_row["topic_id"]

        for i in range(3):
            conn.execute("""
                UPDATE news_image_embeddings SET is_active = false
                WHERE image_id = %s::uuid AND is_active = true
            """, [image_id])
            _insert_fake_embedding(conn, image_id, article_id, topic_id, float(i) / 10)

    with conn_ctx(cfg.pg_dsn) as conn:
        n = conn.execute("""
            SELECT COUNT(*) AS n FROM news_image_embeddings
            WHERE image_id = %s::uuid AND is_active = true
        """, [image_id]).fetchone()["n"]

    assert n == 1
