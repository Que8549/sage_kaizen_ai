-- =============================================================================
-- news/db/news_seed_data.sql
--
-- Initial topics, profiles, and query templates for Sage Kaizen News Runtime.
--
-- Prerequisites:
--   news/db/news_schema.sql must be applied first.
--
-- Run once:
--   psql -U sage -d sage_kaizen -f news/db/news_seed_data.sql
--
-- All inserts use ON CONFLICT DO NOTHING so this file is safe to re-run.
-- To update a query template: UPDATE news_topic_queries SET ... WHERE ...;
-- =============================================================================


-- =============================================================================
-- SECTION 1: Topics
-- =============================================================================

INSERT INTO news_topics (topic_slug, display_name, description, priority_weight, default_category, default_time_range)
VALUES
    ('ai',                   'Artificial Intelligence', 'AI research, LLMs, robotics AI, safety, and industry news',                  2.0,  'technology', 'day'),
    ('technology',           'Technology',              'General tech news, software releases, hardware, and product launches',         1.5,  'technology', 'day'),
    ('science',              'Science',                 'Scientific discoveries, research papers, space, biology, and physics',         1.5,  'science',    'day'),
    ('world',                'World News',              'International politics, conflicts, diplomacy, and global events',              1.5,  'news',       'day'),
    ('business',             'Business & Economy',      'Markets, corporate news, earnings, economic indicators, and finance',         1.5,  'news',       'day'),
    ('cybersecurity',        'Cybersecurity',           'Breaches, vulnerabilities, ransomware, infosec research, and threat intel',   2.0,  'technology', 'day'),
    ('united_kingdom',       'United Kingdom',          'UK domestic news: politics, economy, culture, and current events',            1.0,  'news',       'day'),
    ('atlanta_georgia',      'Atlanta, Georgia',        'Local Atlanta and Georgia news, politics, weather, and community events',      1.0,  'news',       'day'),
    ('csharp',               'C# / .NET',               'C# language updates, .NET runtime, ASP.NET, and Microsoft dev ecosystem',     1.5,  'technology', 'week'),
    ('python',               'Python',                  'Python language releases, PyPI, ecosystem libraries, and community news',     1.5,  'technology', 'week'),
    ('computer_programming', 'Computer Programming',    'Developer tools, open-source projects, software engineering practices',       1.0,  'technology', 'day')
ON CONFLICT (topic_slug) DO NOTHING;


-- =============================================================================
-- SECTION 2: Default profile — general_brief covers all topics
-- =============================================================================

INSERT INTO news_profiles (profile_name, description, summary_window_default, top_n_default, include_market_data)
VALUES
    ('general_brief', 'Daily briefing covering all tracked topics',           1,  10, false),
    ('ai_brief',      'Focused briefing for AI and technology topics only',   1,   8, false),
    ('dev_brief',     'Developer-focused: C#, Python, programming, tech',     1,   8, false),
    ('local_brief',   'UK and Atlanta local news only',                       1,   6, false),
    ('markets_brief', 'Business, economy, and optional market data',          1,   8, true)
ON CONFLICT (profile_name) DO NOTHING;


-- =============================================================================
-- SECTION 3: Profile → topic mappings
-- =============================================================================

-- general_brief: all topics
INSERT INTO news_profile_topics (profile_id, topic_id, sort_order)
SELECT p.profile_id, nt.topic_id, t.sort_order
FROM   news_profiles p
CROSS JOIN LATERAL (
    VALUES
        ('ai',                   1),
        ('technology',           2),
        ('science',              3),
        ('world',                4),
        ('business',             5),
        ('cybersecurity',        6),
        ('united_kingdom',       7),
        ('atlanta_georgia',      8),
        ('csharp',               9),
        ('python',               10),
        ('computer_programming', 11)
) AS t(topic_slug, sort_order)
JOIN news_topics nt ON nt.topic_slug = t.topic_slug
WHERE p.profile_name = 'general_brief'
ON CONFLICT (profile_id, topic_id) DO NOTHING;

-- ai_brief: ai, technology, cybersecurity
INSERT INTO news_profile_topics (profile_id, topic_id, sort_order)
SELECT p.profile_id, nt.topic_id, t.sort_order
FROM   news_profiles p
CROSS JOIN LATERAL (
    VALUES ('ai', 1), ('technology', 2), ('cybersecurity', 3)
) AS t(topic_slug, sort_order)
JOIN news_topics nt ON nt.topic_slug = t.topic_slug
WHERE p.profile_name = 'ai_brief'
ON CONFLICT (profile_id, topic_id) DO NOTHING;

-- dev_brief: csharp, python, computer_programming, technology
INSERT INTO news_profile_topics (profile_id, topic_id, sort_order)
SELECT p.profile_id, nt.topic_id, t.sort_order
FROM   news_profiles p
CROSS JOIN LATERAL (
    VALUES ('csharp', 1), ('python', 2), ('computer_programming', 3), ('technology', 4)
) AS t(topic_slug, sort_order)
JOIN news_topics nt ON nt.topic_slug = t.topic_slug
WHERE p.profile_name = 'dev_brief'
ON CONFLICT (profile_id, topic_id) DO NOTHING;

-- local_brief: united_kingdom, atlanta_georgia
INSERT INTO news_profile_topics (profile_id, topic_id, sort_order)
SELECT p.profile_id, nt.topic_id, t.sort_order
FROM   news_profiles p
CROSS JOIN LATERAL (
    VALUES ('united_kingdom', 1), ('atlanta_georgia', 2)
) AS t(topic_slug, sort_order)
JOIN news_topics nt ON nt.topic_slug = t.topic_slug
WHERE p.profile_name = 'local_brief'
ON CONFLICT (profile_id, topic_id) DO NOTHING;

-- markets_brief: business, technology, ai
INSERT INTO news_profile_topics (profile_id, topic_id, sort_order)
SELECT p.profile_id, nt.topic_id, t.sort_order
FROM   news_profiles p
CROSS JOIN LATERAL (
    VALUES ('business', 1), ('technology', 2), ('ai', 3)
) AS t(topic_slug, sort_order)
JOIN news_topics nt ON nt.topic_slug = t.topic_slug
WHERE p.profile_name = 'markets_brief'
ON CONFLICT (profile_id, topic_id) DO NOTHING;


-- =============================================================================
-- SECTION 4: Query templates
-- Two to three queries per topic; distinct formulations improve recall.
-- All use time_range='day' for daily runs; reconciliation jobs use 'week'.
-- =============================================================================

-- ai
INSERT INTO news_topic_queries (topic_id, query_text, searxng_categories, time_range, rank_weight, max_results)
SELECT topic_id, q.query_text, q.categories::text[], q.time_range, q.rank_weight, q.max_results
FROM   news_topics
CROSS JOIN LATERAL (
    VALUES
        ('artificial intelligence news today',    '{technology,news}',   'day', 1.0, 20),
        ('large language model news',             '{technology}',         'day', 1.0, 15),
        ('AI research breakthroughs 2025',        '{technology,science}', 'day', 0.8, 15)
) AS q(query_text, categories, time_range, rank_weight, max_results)
WHERE  topic_slug = 'ai'
ON CONFLICT DO NOTHING;

-- Simpler insert style for remaining topics (avoids the category-unnest join complexity)
DO $$
DECLARE
    v_topic_id uuid;
BEGIN

    -- ── technology ───────────────────────────────────────────────────────────
    SELECT topic_id INTO v_topic_id FROM news_topics WHERE topic_slug = 'technology';
    INSERT INTO news_topic_queries (topic_id, query_text, searxng_categories, time_range, rank_weight, max_results)
    VALUES
        (v_topic_id, 'technology news today',             '{technology,news}', 'day',  1.0, 20),
        (v_topic_id, 'software and hardware releases',    '{technology}',      'day',  0.9, 15),
        (v_topic_id, 'tech industry latest news',         '{news}',            'day',  0.8, 15)
    ON CONFLICT DO NOTHING;

    -- ── science ──────────────────────────────────────────────────────────────
    SELECT topic_id INTO v_topic_id FROM news_topics WHERE topic_slug = 'science';
    INSERT INTO news_topic_queries (topic_id, query_text, searxng_categories, time_range, rank_weight, max_results)
    VALUES
        (v_topic_id, 'science news today',                '{science,news}',    'day',  1.0, 20),
        (v_topic_id, 'scientific discoveries this week',  '{science}',         'day',  0.9, 15),
        (v_topic_id, 'space astronomy news',              '{science,news}',    'day',  0.8, 10)
    ON CONFLICT DO NOTHING;

    -- ── world ─────────────────────────────────────────────────────────────────
    SELECT topic_id INTO v_topic_id FROM news_topics WHERE topic_slug = 'world';
    INSERT INTO news_topic_queries (topic_id, query_text, searxng_categories, time_range, rank_weight, max_results)
    VALUES
        (v_topic_id, 'world news today',                  '{news}',            'day',  1.0, 20),
        (v_topic_id, 'international headlines today',     '{news}',            'day',  0.9, 20),
        (v_topic_id, 'global events breaking news',       '{news}',            'day',  0.8, 15)
    ON CONFLICT DO NOTHING;

    -- ── business ─────────────────────────────────────────────────────────────
    SELECT topic_id INTO v_topic_id FROM news_topics WHERE topic_slug = 'business';
    INSERT INTO news_topic_queries (topic_id, query_text, searxng_categories, time_range, rank_weight, max_results)
    VALUES
        (v_topic_id, 'business and economy news today',   '{news}',            'day',  1.0, 20),
        (v_topic_id, 'stock market news today',           '{news}',            'day',  0.9, 15),
        (v_topic_id, 'corporate earnings financial news', '{news}',            'day',  0.8, 15)
    ON CONFLICT DO NOTHING;

    -- ── cybersecurity ─────────────────────────────────────────────────────────
    SELECT topic_id INTO v_topic_id FROM news_topics WHERE topic_slug = 'cybersecurity';
    INSERT INTO news_topic_queries (topic_id, query_text, searxng_categories, time_range, rank_weight, max_results)
    VALUES
        (v_topic_id, 'cybersecurity news today',          '{technology,news}', 'day',  1.0, 20),
        (v_topic_id, 'data breach ransomware 2025',       '{technology,news}', 'day',  1.0, 15),
        (v_topic_id, 'vulnerability CVE security patch',  '{technology}',      'day',  0.9, 15)
    ON CONFLICT DO NOTHING;

    -- ── united_kingdom ────────────────────────────────────────────────────────
    SELECT topic_id INTO v_topic_id FROM news_topics WHERE topic_slug = 'united_kingdom';
    INSERT INTO news_topic_queries (topic_id, query_text, searxng_categories, time_range, rank_weight, max_results)
    VALUES
        (v_topic_id, 'United Kingdom news today',         '{news}',            'day',  1.0, 20),
        (v_topic_id, 'UK politics economy BBC today',     '{news}',            'day',  0.9, 15),
        (v_topic_id, 'Britain England Scotland Wales news','{news}',           'day',  0.8, 15)
    ON CONFLICT DO NOTHING;

    -- ── atlanta_georgia ───────────────────────────────────────────────────────
    SELECT topic_id INTO v_topic_id FROM news_topics WHERE topic_slug = 'atlanta_georgia';
    INSERT INTO news_topic_queries (topic_id, query_text, searxng_categories, time_range, rank_weight, max_results)
    VALUES
        (v_topic_id, 'Atlanta Georgia news today',        '{news}',            'day',  1.0, 20),
        (v_topic_id, 'Atlanta local news AJC WSB',        '{news}',            'day',  0.9, 15),
        (v_topic_id, 'Georgia state news politics',       '{news}',            'day',  0.8, 15)
    ON CONFLICT DO NOTHING;

    -- ── csharp ────────────────────────────────────────────────────────────────
    SELECT topic_id INTO v_topic_id FROM news_topics WHERE topic_slug = 'csharp';
    INSERT INTO news_topic_queries (topic_id, query_text, searxng_categories, time_range, rank_weight, max_results)
    VALUES
        (v_topic_id, 'C# .NET news release 2025',         '{technology}',      'week', 1.0, 15),
        (v_topic_id, 'ASP.NET Blazor MAUI latest',        '{technology}',      'week', 0.9, 10),
        (v_topic_id, 'Microsoft dotnet announcement',     '{technology,news}', 'week', 0.8, 10)
    ON CONFLICT DO NOTHING;

    -- ── python ────────────────────────────────────────────────────────────────
    SELECT topic_id INTO v_topic_id FROM news_topics WHERE topic_slug = 'python';
    INSERT INTO news_topic_queries (topic_id, query_text, searxng_categories, time_range, rank_weight, max_results)
    VALUES
        (v_topic_id, 'Python programming news 2025',      '{technology}',      'week', 1.0, 15),
        (v_topic_id, 'Python release PyPI package news',  '{technology}',      'week', 0.9, 10),
        (v_topic_id, 'CPython PEP new Python features',   '{technology}',      'week', 0.8, 10)
    ON CONFLICT DO NOTHING;

    -- ── computer_programming ─────────────────────────────────────────────────
    SELECT topic_id INTO v_topic_id FROM news_topics WHERE topic_slug = 'computer_programming';
    INSERT INTO news_topic_queries (topic_id, query_text, searxng_categories, time_range, rank_weight, max_results)
    VALUES
        (v_topic_id, 'programming developer news today',  '{technology,news}', 'day',  1.0, 20),
        (v_topic_id, 'open source software release 2025', '{technology}',      'day',  0.9, 15),
        (v_topic_id, 'software engineering tools IDE',    '{technology}',      'day',  0.8, 10)
    ON CONFLICT DO NOTHING;

END;
$$;


-- =============================================================================
-- Verification query — run after seeding to confirm counts
-- =============================================================================
-- SELECT t.display_name, count(q.topic_query_id) AS query_count
-- FROM   news_topics t
-- LEFT JOIN news_topic_queries q ON q.topic_id = nt.topic_id
-- GROUP BY t.display_name
-- ORDER BY t.display_name;
