"""
news/retrieval/news_resolver.py

DB-first / live / hybrid news query resolution.

Called as an optional fifth parallel worker inside
rag_v1/runtime/context_injector.apply_rag_and_wiki_parallel().

Decision tree:
  1. Is this a news query?  → keyword detection
     No  → return None (zero overhead on normal chat)
  2. Is it a market point-lookup?
     Yes → yfinance live lookup → return market context
  3. Do we have a fresh daily brief in news_briefs?
     (freshness_at > now() - cfg.brief_freshness_hours)
     Yes → DB-only: return brief content
  4. Is a collection run currently in progress?
     Yes → return stale brief with a freshness warning
  5. Hybrid: return whatever briefs exist + flag as potentially stale

Returns a NewsContext dataclass or None.

Context is injected as a <news_context> block in the user message,
matching the existing <search_context> pattern.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from rapidfuzz import fuzz as _fuzz

from lazy import lazy_singleton
from news.news_settings import get_news_settings
from rag_v1.db.pg import conn_ctx
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.resolver", file_name="news_agent.log")

# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

# Phrases that strongly indicate a live news request.
_NEWS_INTENT_PHRASES: tuple[str, ...] = (
    "what's in the news", "what is in the news",
    "today's news", "today's top stories", "top stories today",
    "top stories", "what happened today", "what happened yesterday",
    "latest news", "breaking news", "news today", "news this week",
    "summarize the news", "summarize today", "summarize the top",
    "what are the headlines", "headlines today", "morning briefing",
    "news briefing", "daily brief", "weekly brief",
    "news about", "any news on", "latest on",
    "news of the war", "news since",
    "last 7 days", "past week news", "this week's news",
    "news summary",
    # Natural variants not covered by the phrases above
    "top news",           # "what's today's top news?", "top news stories"
    "today's top",        # "today's top news", "today's top stories" (redundant but safe)
    "what's the news",    # "what's the news today?", "what's the news this week?"
    "what is the news",   # same, uncontracted
    "what's happening today", "what's going on today",
    "give me the news", "tell me the news",
    "catch me up", "catch up on the news",
)

# Phrases that indicate a market / price lookup.
#
# Split into two tiers on 2026-08-05.  Previously this was one flat tuple whose
# comment claimed it kept "strict matching to avoid false positives on queries
# like 'how much is this worth'" — while "how much is" was itself in the list,
# so that exact example matched.  The tiers make the comment true:
#
#   STRONG — unambiguously financial on their own.
#   WEAK   — ordinary English that only means "market" when an instrument is
#            actually named, so they additionally require _extract_ticker() to
#            find one.  "how much is nvidia" → market; "how much is this
#            worth" → not.
_MARKET_PHRASES_STRONG: tuple[str, ...] = (
    "stock price", "share price", "trading at", "market price",
    "bitcoin price", "crypto price", "btc price", "eth price",
    "stock today", "stock yesterday", "market today",
    "closing price", "open price", "52-week",
    "performed this week", "performed this month",
    "nasdaq", "s&p", "dow jones", "oil price", "gold price",
)

_MARKET_PHRASES_WEAK: tuple[str, ...] = (
    "what is the price", "what's the price", "how much is", "how has",
)

# Retained as the union for callers/tests that just want "is this phrase
# market-ish at all" without the instrument requirement.
_MARKET_PHRASES: tuple[str, ...] = _MARKET_PHRASES_STRONG + _MARKET_PHRASES_WEAK

# Common ticker patterns in user messages (e.g. "NVDA", "BTC-USD").
_TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5}(?:-[A-Z]{2,3})?)\b")

# Compiled word-boundary matchers for the company-name aliases, built once.
# `\b` does not work directly against keys containing "&" or "." (e.g.
# "s&p 500"), so the boundary is asserted with lookarounds on word characters
# instead — equivalent for our keys and safe for punctuation-bearing ones.
_NAME_BOUNDARY_RE: dict[str, re.Pattern[str]] = {}


def _name_matches(name: str, txt_lower: str) -> bool:
    """
    True when `name` occurs in `txt_lower` as a whole word, not a substring.

    The boundary class includes the hyphen, so hyphenated compounds do not
    match: "meta-analysis" must not resolve to META. No key in
    _NAME_TO_TICKER contains a hyphen, so this costs nothing (the hyphenated
    forms like "BTC-USD" are the mapping's *values*).
    """
    pattern = _NAME_BOUNDARY_RE.get(name)
    if pattern is None:
        pattern = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")
        _NAME_BOUNDARY_RE[name] = pattern
    return pattern.search(txt_lower) is not None

# Fuzzy match threshold (same calibration as router.py — see comments there).
_FUZZY_THRESHOLD = 76

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NewsContext:
    """Resolved news context ready for injection into the user message."""
    source: str               # "db_brief" | "market" | "hybrid" | "stale"
    content: str              # The context text to inject
    freshness_at: datetime | None = None
    is_stale: bool = False

    def to_xml_block(self) -> str:
        freshness_str = ""
        if self.freshness_at:
            freshness_str = f' freshness="{self.freshness_at.isoformat()}"'
        stale_str = ' stale="true"' if self.is_stale else ""
        return (
            f'<news_context source="{self.source}"{freshness_str}{stale_str}>\n'
            f'{self.content}\n'
            f'</news_context>'
        )

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_FETCH_FRESH_BRIEF_SQL = """
SELECT
    b.brief_id::text,
    b.brief_kind,
    b.headline_summary,
    b.summary_short,
    b.summary_long,
    b.freshness_at,
    p.profile_name
FROM news_briefs b
JOIN news_profiles p ON p.profile_id = b.profile_id
WHERE b.is_final    = true
  AND b.brief_kind  = %s
  AND b.brief_date  = CURRENT_DATE
  AND b.freshness_at > now() - (%s || ' hours')::interval
ORDER BY b.freshness_at DESC
LIMIT 1
"""

_FETCH_ANY_BRIEF_SQL = """
SELECT
    b.brief_id::text,
    b.brief_kind,
    b.headline_summary,
    b.summary_short,
    b.summary_long,
    b.freshness_at,
    p.profile_name
FROM news_briefs b
JOIN news_profiles p ON p.profile_id = b.profile_id
WHERE b.is_final   = true
  AND b.brief_kind = %s
ORDER BY b.brief_date DESC, b.freshness_at DESC
LIMIT 1
"""

_CHECK_COLLECTION_RUNNING_SQL = """
SELECT run_id FROM news_runs
WHERE run_type  = 'collection'
  AND status    = 'running'
  AND started_at > now() - INTERVAL '15 minutes'
LIMIT 1
"""


# ---------------------------------------------------------------------------
# NewsResolver
# ---------------------------------------------------------------------------

class NewsResolver:
    """
    Resolves news-intent queries against stored briefs or live sources.

    One instance is typically shared for the process lifetime (lazy singleton).
    All DB calls use thread-local connections; the resolver is thread-safe.
    """

    def __init__(self) -> None:
        self._cfg = get_news_settings()
        self._dsn = self._cfg.pg_dsn

    def resolve(self, user_text: str) -> NewsContext | None:
        """
        Resolve a user message against the news pipeline.

        Returns a NewsContext if the query is news-related, or None otherwise.
        The caller injects the .to_xml_block() into the user message.
        """
        txt = user_text.lower()

        # ── 1. Market lookup ──────────────────────────────────────────────────
        if self._is_market_query(txt):
            return self._resolve_market(user_text)

        # ── 2. News intent gate ───────────────────────────────────────────────
        if not self._is_news_query(txt):
            return None

        # ── 3. Determine the right brief kind ─────────────────────────────────
        if any(p in txt for p in ("7 day", "7-day", "week", "past week", "this week")):
            kind = "rolling_7_day"
        else:
            kind = "daily"

        return self._resolve_news_brief(kind)

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _is_news_query(self, txt: str) -> bool:
        # Stage 1: exact substring (zero cost, handles all enumerated phrases).
        for p in _NEWS_INTENT_PHRASES:
            if p in txt:
                return True
        # Stage 2: fuzzy — catches paraphrases not in the list.
        # Only applied to multi-word phrases; single words are too short to
        # fuzzy-match reliably without false positives.
        for p in _NEWS_INTENT_PHRASES:
            if len(p.split()) >= 2 and _fuzz.token_set_ratio(txt, p) >= _FUZZY_THRESHOLD:
                return True
        return False

    def _is_market_query(self, txt: str) -> bool:
        """
        True when the query is a market/price lookup.

        Strong phrases ("stock price", "trading at") are financial on their
        own.  Weak ones ("how much is", "how has") are ordinary English and
        additionally require a nameable instrument, so "how much is this worth"
        is correctly rejected while "how much is nvidia" is not.
        """
        if any(p in txt for p in _MARKET_PHRASES_STRONG):
            return True
        if any(p in txt for p in _MARKET_PHRASES_WEAK):
            return self._extract_ticker(txt) is not None
        return False

    # ------------------------------------------------------------------
    # Resolution paths
    # ------------------------------------------------------------------

    def _resolve_news_brief(self, kind: str) -> NewsContext | None:
        """Try DB-first, fall back to stale brief, then to None."""

        # Fresh brief?
        with conn_ctx(self._dsn) as conn:
            row = conn.execute(
                _FETCH_FRESH_BRIEF_SQL,
                [kind, self._cfg.brief_freshness_hours],
            ).fetchone()

        if row:
            content = self._format_brief(row)
            _LOG.debug("news_resolver | db-fresh brief | kind=%s", kind)
            return NewsContext(
                source="db_brief",
                content=content,
                freshness_at=row["freshness_at"],
                is_stale=False,
            )

        # Check if collection is running (data will arrive soon).
        with conn_ctx(self._dsn) as conn:
            running = conn.execute(_CHECK_COLLECTION_RUNNING_SQL).fetchone()

        # Fetch the most recent brief regardless of date.
        with conn_ctx(self._dsn) as conn:
            stale_row = conn.execute(_FETCH_ANY_BRIEF_SQL, [kind]).fetchone()

        if stale_row:
            content = self._format_brief(stale_row)
            if running:
                content += (
                    "\n\n[Note: A fresh collection is in progress; "
                    "this summary may be updated shortly.]"
                )
            else:
                content += (
                    "\n\n[Note: This summary may not reflect the very latest news. "
                    "The next scheduled collection will refresh it.]"
                )
            _LOG.debug("news_resolver | stale brief | kind=%s", kind)
            return NewsContext(
                source="stale",
                content=content,
                freshness_at=stale_row["freshness_at"],
                is_stale=True,
            )

        # No brief at all — signal live search should handle it.
        _LOG.debug("news_resolver | no brief found | kind=%s", kind)
        return None

    def _resolve_market(self, user_text: str) -> NewsContext | None:
        """Extract ticker from user text and call yfinance."""
        try:
            from news.retrieval.market_client import get_market_client, normalize_ticker
            client = get_market_client()

            # Try to find a recognisable ticker or company name in the text.
            ticker = self._extract_ticker(user_text)
            if not ticker:
                return None  # Can't identify the instrument; let live search handle it.

            # Check for date reference ("yesterday", "last week", specific date).
            yesterday_match = re.search(r"\byesterday\b", user_text, re.IGNORECASE)
            week_match = re.search(r"\b(this week|past week|last week|7.day)\b",
                                   user_text, re.IGNORECASE)

            if yesterday_match:
                from datetime import date, timedelta
                target = date.today() - timedelta(days=1)
                data = client.get_price_on_date(ticker, target)
            elif week_match:
                data = client.get_recent_history(ticker, days=7)
            else:
                data = client.get_current_price(ticker)

            content = client.format_for_context(data)
            _LOG.debug("news_resolver | market lookup | ticker=%s", ticker)
            return NewsContext(source="market", content=content)

        except Exception as exc:
            _LOG.warning("news_resolver | market error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Ticker extraction
    # ------------------------------------------------------------------

    def _extract_ticker(self, text: str) -> str | None:
        from news.retrieval.market_client import normalize_ticker, _NAME_TO_TICKER
        txt_lower = text.lower()

        # Check known names first (longest match wins).
        #
        # Word-boundary matching, not bare `name in txt_lower`. _NAME_TO_TICKER
        # keys include 3-4 letter aliases ("eth", "btc", "amd", "dow", "oil",
        # "gold", "meta") which, matched as raw substrings, fire inside ordinary
        # words: "something" contains "eth", "goldfish" contains "gold". That
        # injected a live gold-futures price into any turn mentioning a
        # goldfish. Fixed 2026-08-05.
        #
        # Longest-first still matters so "dow jones" beats "dow", and \b works
        # for multi-word keys too ("s&p 500" is bounded at each end).
        names_sorted = sorted(_NAME_TO_TICKER.keys(), key=len, reverse=True)
        for name in names_sorted:
            if _name_matches(name, txt_lower):
                return _NAME_TO_TICKER[name]

        # Fall back to uppercase ticker pattern.
        matches = _TICKER_PATTERN.findall(text)
        for m in matches:
            if len(m) >= 2 and m not in ("I", "A", "AT", "BE", "BY", "DO", "IN",
                                          "IS", "IT", "ME", "MY", "NO", "OF", "ON",
                                          "OR", "SO", "TO", "UP", "US", "WE"):
                return m
        return None

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_brief(row: dict) -> str:
        parts = []
        if row.get("headline_summary"):
            parts.append(f"**Headline:** {row['headline_summary']}")
        if row.get("summary_short"):
            parts.append(row["summary_short"])
        if row.get("summary_long"):
            parts.append(row["summary_long"])
        return "\n\n".join(parts) if parts else "(No brief content available)"


# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------
@lazy_singleton
def get_news_resolver() -> NewsResolver:
    """
    Process-wide resolver.

    Locked: resolve_news_context() runs on the context injector's news worker
    thread, concurrently with the other four fetches.
    """
    return NewsResolver()


def resolve_news_context(user_text: str) -> NewsContext | None:
    """Convenience function for context_injector integration."""
    try:
        return get_news_resolver().resolve(user_text)
    except Exception as exc:
        _LOG.warning("news_resolver | unhandled error: %s", exc)
        return None
