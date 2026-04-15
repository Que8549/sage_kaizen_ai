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
from typing import Optional

from news.news_settings import get_news_settings
from rag_v1.db.pg import conn_ctx
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.resolver")

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
)

# Phrases that indicate a market / price lookup.
_MARKET_PHRASES: tuple[str, ...] = (
    "stock price", "share price", "trading at", "market price",
    "what is the price", "what's the price", "how much is",
    "bitcoin price", "crypto price", "btc price", "eth price",
    "stock today", "stock yesterday", "market today",
    "closing price", "open price", "52-week",
    "how has", "performed this week", "performed this month",
    "nasdaq", "s&p", "dow jones", "oil price", "gold price",
)

# Common ticker patterns in user messages (e.g. "NVDA", "BTC-USD").
_TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5}(?:-[A-Z]{2,3})?)\b")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NewsContext:
    """Resolved news context ready for injection into the user message."""
    source: str               # "db_brief" | "market" | "hybrid" | "stale"
    content: str              # The context text to inject
    freshness_at: Optional[datetime] = None
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

    def resolve(self, user_text: str) -> Optional[NewsContext]:
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
        return any(p in txt for p in _NEWS_INTENT_PHRASES)

    def _is_market_query(self, txt: str) -> bool:
        return any(p in txt for p in _MARKET_PHRASES)

    # ------------------------------------------------------------------
    # Resolution paths
    # ------------------------------------------------------------------

    def _resolve_news_brief(self, kind: str) -> Optional[NewsContext]:
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

    def _resolve_market(self, user_text: str) -> Optional[NewsContext]:
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

    def _extract_ticker(self, text: str) -> Optional[str]:
        from news.retrieval.market_client import normalize_ticker, _NAME_TO_TICKER
        txt_lower = text.lower()

        # Check known names first (longest match wins).
        names_sorted = sorted(_NAME_TO_TICKER.keys(), key=len, reverse=True)
        for name in names_sorted:
            if name in txt_lower:
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
_resolver: Optional[NewsResolver] = None


def get_news_resolver() -> NewsResolver:
    global _resolver
    if _resolver is None:
        _resolver = NewsResolver()
    return _resolver


def resolve_news_context(user_text: str) -> Optional[NewsContext]:
    """Convenience function for context_injector integration."""
    try:
        return get_news_resolver().resolve(user_text)
    except Exception as exc:
        _LOG.warning("news_resolver | unhandled error: %s", exc)
        return None
