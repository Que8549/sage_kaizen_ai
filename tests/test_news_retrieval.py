"""
tests/test_news_retrieval.py

Unit tests for news/retrieval/ — the query-time half of the news pipeline
(collection lives in the sage_kaizen_ai_ingest project).

  market_client.py — ticker normalisation + yfinance wrappers
  news_resolver.py — news-intent gate, DB-first brief lookup with stale
                     fallback, and market routing

yfinance and psycopg are both mocked; nothing here touches the network or a
database.
"""
from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import news.retrieval.market_client as mc
import news.retrieval.news_resolver as nr
from news.retrieval.market_client import MarketClient, normalize_ticker
from news.retrieval.news_resolver import NewsContext, NewsResolver


# ---------------------------------------------------------------------------
# market_client.normalize_ticker
# ---------------------------------------------------------------------------

class TestNormalizeTicker:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("nvidia", "NVDA"),
            ("NVIDIA", "NVDA"),
            ("  nvidia  ", "NVDA"),
            ("nvda", "NVDA"),
            ("apple", "AAPL"),
            ("google", "GOOGL"),
            ("alphabet", "GOOGL"),
            ("facebook", "META"),
            ("bitcoin", "BTC-USD"),
            ("btc", "BTC-USD"),
            ("ethereum", "ETH-USD"),
            ("s&p 500", "^GSPC"),
            ("nasdaq", "^IXIC"),
            ("dow jones", "^DJI"),
            ("crude oil", "CL=F"),
            ("gold", "GC=F"),
        ],
    )
    def test_known_names_and_aliases(self, raw, expected):
        assert normalize_ticker(raw) == expected

    def test_crypto_gets_the_usd_pair_suffix(self):
        """yfinance needs BTC-USD, not BTC."""
        assert normalize_ticker("btc").endswith("-USD")

    def test_indices_keep_the_caret_prefix(self):
        assert normalize_ticker("sp500").startswith("^")

    def test_unknown_input_is_passed_through(self):
        assert normalize_ticker("SOMETICKER") not in ("", None)

    def test_map_values_are_all_uppercase_symbols(self):
        for sym in mc._NAME_TO_TICKER.values():
            assert sym == sym.upper()


# ---------------------------------------------------------------------------
# MarketClient
# ---------------------------------------------------------------------------

@pytest.fixture
def yf():
    """A stand-in yfinance module injected into sys.modules."""
    fake = MagicMock()
    with patch.dict("sys.modules", {"yfinance": fake}):
        yield fake


class TestGetCurrentPrice:
    def test_returns_normalised_payload(self, yf):
        yf.Ticker.return_value.fast_info = MagicMock(last_price=123.456789, currency="USD")
        out = MarketClient().get_current_price("nvidia")
        assert out["ticker"] == "NVDA"
        assert out["price"] == 123.4568        # rounded to 4dp
        assert out["currency"] == "USD"
        assert out["source"] == "yfinance"

    def test_timestamp_is_iso_utc(self, yf):
        yf.Ticker.return_value.fast_info = MagicMock(last_price=1.0, currency="USD")
        assert "+00:00" in MarketClient().get_current_price("AAPL")["timestamp"]

    def test_missing_price_returns_an_error_dict(self, yf):
        info = MagicMock()
        info.last_price = None
        info.regularMarketPrice = None
        yf.Ticker.return_value.fast_info = info
        out = MarketClient().get_current_price("AAPL")
        assert "error" in out
        assert "No price data" in out["error"]

    def test_exception_returns_an_error_dict(self, yf):
        yf.Ticker.side_effect = RuntimeError("yfinance exploded")
        out = MarketClient().get_current_price("AAPL")
        assert out["error"] == "yfinance exploded"
        assert out["ticker"] == "AAPL"


class TestGetPriceOnDate:
    def _hist(self, close=99.5):
        hist = MagicMock()
        hist.empty = False
        hist.__getitem__.return_value.iloc.__getitem__.return_value = close
        return hist

    def test_returns_the_close(self, yf):
        yf.download.return_value = self._hist(99.5)
        out = MarketClient().get_price_on_date("nvidia", date(2026, 8, 1))
        assert out["ticker"] == "NVDA"
        assert out["close"] == 99.5
        assert out["date"] == "2026-08-01"

    def test_none_history_is_handled(self, yf):
        """yfinance's download() is typed DataFrame | None."""
        yf.download.return_value = None
        assert "error" in MarketClient().get_price_on_date("AAPL", date(2026, 8, 1))

    def test_empty_history_is_handled(self, yf):
        hist = MagicMock()
        hist.empty = True
        yf.download.return_value = hist
        assert "error" in MarketClient().get_price_on_date("AAPL", date(2026, 8, 1))

    def test_requests_a_single_day_window(self, yf):
        yf.download.return_value = self._hist()
        target = date(2026, 8, 1)
        MarketClient().get_price_on_date("AAPL", target)
        kwargs = yf.download.call_args.kwargs
        assert kwargs["start"] == target
        assert kwargs["end"] == target + timedelta(days=1)

    def test_exception_returns_an_error_dict(self, yf):
        yf.download.side_effect = OSError("network down")
        assert "error" in MarketClient().get_price_on_date("AAPL", date(2026, 8, 1))


class TestGetRecentHistory:
    def _hist(self, rows):
        hist = MagicMock()
        hist.empty = False
        hist.iterrows.return_value = iter(rows)
        return hist

    def test_builds_ohlc_rows(self, yf):
        row = {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5}
        yf.download.return_value = self._hist([(datetime(2026, 8, 1), row)])
        out = MarketClient().get_recent_history("nvidia", days=7)
        assert out["ticker"] == "NVDA"
        assert out["days"] == 7
        assert out["history"] == [
            {"date": "2026-08-01", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5}
        ]

    def test_index_is_stringified_to_ten_chars(self, yf):
        """Index type from iterrows() is Hashable — str(idx)[:10], not .date()."""
        row = {"Open": 1, "High": 1, "Low": 1, "Close": 1}
        yf.download.return_value = self._hist([("2026-08-01 00:00:00+00:00", row)])
        assert MarketClient().get_recent_history("AAPL")["history"][0]["date"] == "2026-08-01"

    def test_none_history_is_handled(self, yf):
        yf.download.return_value = None
        assert "error" in MarketClient().get_recent_history("AAPL")

    def test_empty_history_is_handled(self, yf):
        hist = MagicMock()
        hist.empty = True
        yf.download.return_value = hist
        assert "error" in MarketClient().get_recent_history("AAPL")

    def test_exception_returns_an_error_dict(self, yf):
        yf.download.side_effect = ValueError("bad ticker")
        assert "error" in MarketClient().get_recent_history("AAPL")


class TestFormatForContext:
    def test_error_dict(self):
        out = MarketClient().format_for_context({"error": "no data"})
        assert out == "Market data unavailable: no data"

    def test_current_price(self):
        out = MarketClient().format_for_context({
            "ticker": "NVDA", "price": 123.45, "currency": "USD", "timestamp": "T",
        })
        assert "NVDA" in out and "123.45" in out and "USD" in out

    def test_price_on_date(self):
        out = MarketClient().format_for_context({
            "ticker": "NVDA", "date": "2026-08-01", "close": 99.5, "source": "yfinance",
        })
        assert "2026-08-01" in out and "99.5" in out

    def test_history_shows_the_last_five_rows(self):
        history = [{"date": f"2026-08-{i:02d}", "close": float(i)} for i in range(1, 11)]
        out = MarketClient().format_for_context({
            "ticker": "NVDA", "days": 10, "history": history,
        })
        assert out.count("close=") == 5
        assert "2026-08-10" in out
        assert "2026-08-01" not in out

    def test_unrecognised_shape_falls_back_to_repr(self):
        assert "surprise" in MarketClient().format_for_context({"surprise": 1})


class TestGetMarketClientSingleton:
    def test_is_a_singleton(self):
        mc.get_market_client.reset()
        assert mc.get_market_client() is mc.get_market_client()
        mc.get_market_client.reset()

    def test_built_once_under_concurrency(self):
        built: list[int] = []

        def _slow(*a, **kw):
            built.append(1)
            time.sleep(0.02)
            return MagicMock()

        mc.get_market_client.reset()
        with patch.object(mc, "MarketClient", side_effect=_slow):
            threads = [threading.Thread(target=mc.get_market_client) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        mc.get_market_client.reset()

        assert len(built) == 1


# ---------------------------------------------------------------------------
# NewsContext
# ---------------------------------------------------------------------------

class TestNewsContext:
    def test_minimal_xml_block(self):
        block = NewsContext(source="db_brief", content="body").to_xml_block()
        assert block.startswith('<news_context source="db_brief">')
        assert block.endswith("</news_context>")
        assert "\nbody\n" in block

    def test_freshness_attribute(self):
        ts = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
        block = NewsContext(source="db_brief", content="b", freshness_at=ts).to_xml_block()
        assert 'freshness="2026-08-04T12:00:00+00:00"' in block

    def test_stale_attribute(self):
        block = NewsContext(source="stale", content="b", is_stale=True).to_xml_block()
        assert 'stale="true"' in block

    def test_no_stale_attribute_when_fresh(self):
        assert "stale=" not in NewsContext(source="db_brief", content="b").to_xml_block()


# ---------------------------------------------------------------------------
# NewsResolver
# ---------------------------------------------------------------------------

@pytest.fixture
def resolver():
    cfg = MagicMock()
    cfg.pg_dsn = "postgresql://test"
    cfg.brief_freshness_hours = 6
    with patch.object(nr, "get_news_settings", return_value=cfg):
        yield NewsResolver()


class TestNewsIntentGate:
    @pytest.mark.parametrize(
        "text",
        ["today's news", "what's the latest news", "top headlines", "news briefing"],
    )
    def test_recognises_news_queries(self, resolver, text):
        assert resolver._is_news_query(text) is True

    def test_current_events_is_not_a_news_intent_phrase_here(self, resolver):
        """
        Worth pinning: router.py's _NEWS_CONTENT_SIGNALS DOES list "current
        events", but news_resolver's own _NEWS_INTENT_PHRASES does not, and the
        fuzzy stage doesn't bridge the gap either. So the router can decide a
        turn is news-shaped while the resolver declines to fetch a brief for
        it. Two lists, two answers — not wrong, but not obviously intended.
        """
        assert resolver._is_news_query("current events in the middle east") is False

    @pytest.mark.parametrize(
        "text", ["write me a poem", "how do I sort a list in python", "hello"]
    )
    def test_rejects_non_news_queries(self, resolver, text):
        assert resolver._is_news_query(text) is False

    @pytest.mark.parametrize(
        "text", ["nvidia stock price", "what is bitcoin trading at", "stock market today"]
    )
    def test_recognises_market_queries(self, resolver, text):
        assert resolver._is_market_query(text) is True

    def test_market_gate_rejects_vague_value_questions(self, resolver):
        """
        Fixed 2026-08-05. _MARKET_PHRASES used to be one flat tuple whose
        comment claimed it kept "strict matching to avoid false positives on
        queries like 'how much is this worth'" — while "how much is" was
        itself in the list, so that exact example matched.

        The phrases are now tiered: weak ones like "how much is" additionally
        require a nameable instrument.
        """
        assert resolver._is_market_query("how much is this worth") is False

    def test_weak_phrase_with_an_instrument_is_a_market_query(self, resolver):
        assert resolver._is_market_query("how much is nvidia") is True

    def test_strong_phrases_need_no_instrument(self, resolver):
        assert resolver._is_market_query("what is the stock price") is True

    @pytest.mark.parametrize("phrase", nr._MARKET_PHRASES_WEAK)
    def test_every_weak_phrase_alone_is_rejected(self, resolver, phrase):
        assert resolver._is_market_query(phrase) is False

    @pytest.mark.parametrize("phrase", nr._MARKET_PHRASES_STRONG)
    def test_every_strong_phrase_alone_is_accepted(self, resolver, phrase):
        assert resolver._is_market_query(phrase) is True

    def test_tiers_partition_the_legacy_phrase_tuple(self):
        assert set(nr._MARKET_PHRASES) == (
            set(nr._MARKET_PHRASES_STRONG) | set(nr._MARKET_PHRASES_WEAK)
        )
        assert not (set(nr._MARKET_PHRASES_STRONG) & set(nr._MARKET_PHRASES_WEAK))


class TestResolveRouting:
    def test_market_queries_route_to_market(self, resolver):
        with patch.object(resolver, "_resolve_market", return_value="MARKET") as m:
            assert resolver.resolve("nvidia stock price") == "MARKET"
        m.assert_called_once()

    def test_non_news_returns_none(self, resolver):
        assert resolver.resolve("write me a haiku about snails") is None

    def test_daily_brief_is_the_default_kind(self, resolver):
        with patch.object(resolver, "_resolve_news_brief", return_value=None) as m:
            resolver.resolve("today's news")
        assert m.call_args.args[0] == "daily"

    @pytest.mark.parametrize(
        "text",
        ["news from this week", "7 day news summary", "7-day news",
         "news from the past week"],
    )
    def test_week_phrasing_selects_the_rolling_brief(self, resolver, text):
        with patch.object(resolver, "_resolve_news_brief", return_value=None) as m:
            resolver.resolve(text)
        assert m.call_args.args[0] == "rolling_7_day"


class TestResolveNewsBrief:
    def _conn(self, rows):
        """conn_ctx yields a connection whose execute().fetchone() walks `rows`."""
        conn = MagicMock()
        conn.execute.return_value.fetchone.side_effect = list(rows)
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=conn)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    def test_fresh_brief_is_returned(self, resolver):
        ts = datetime(2026, 8, 4, tzinfo=timezone.utc)
        row = {"headline_summary": "Big news", "summary_short": "short",
               "summary_long": None, "freshness_at": ts}
        with patch.object(nr, "conn_ctx", return_value=self._conn([row])):
            out = resolver._resolve_news_brief("daily")
        assert out is not None
        assert out.source == "db_brief"
        assert out.is_stale is False
        assert "Big news" in out.content

    def test_stale_brief_when_no_fresh_one(self, resolver):
        ts = datetime(2026, 8, 1, tzinfo=timezone.utc)
        stale = {"headline_summary": "Old news", "summary_short": None,
                 "summary_long": None, "freshness_at": ts}
        # fresh → None, running → None, stale → row
        with patch.object(nr, "conn_ctx", return_value=self._conn([None, None, stale])):
            out = resolver._resolve_news_brief("daily")
        assert out is not None
        assert out.source == "stale"
        assert out.is_stale is True
        assert "may not reflect the very latest news" in out.content

    def test_stale_brief_notes_a_running_collection(self, resolver):
        stale = {"headline_summary": "Old", "summary_short": None,
                 "summary_long": None, "freshness_at": None}
        with patch.object(nr, "conn_ctx",
                          return_value=self._conn([None, {"run_id": "r1"}, stale])):
            out = resolver._resolve_news_brief("daily")
        assert out is not None
        assert "fresh collection is in progress" in out.content

    def test_no_brief_at_all_returns_none(self, resolver):
        with patch.object(nr, "conn_ctx", return_value=self._conn([None, None, None])):
            assert resolver._resolve_news_brief("daily") is None


class TestFormatBrief:
    def test_headline_is_bolded(self):
        out = NewsResolver._format_brief({"headline_summary": "H"})
        assert out == "**Headline:** H"

    def test_all_sections_joined(self):
        out = NewsResolver._format_brief({
            "headline_summary": "H", "summary_short": "S", "summary_long": "L",
        })
        assert out == "**Headline:** H\n\nS\n\nL"

    def test_empty_row_gets_a_placeholder(self):
        assert NewsResolver._format_brief({}) == "(No brief content available)"

    def test_none_values_are_skipped(self):
        out = NewsResolver._format_brief({"headline_summary": None, "summary_short": "S"})
        assert out == "S"


class TestExtractTicker:
    def test_finds_a_company_name(self, resolver):
        assert resolver._extract_ticker("what is nvidia trading at") == "NVDA"

    def test_longest_name_wins(self, resolver):
        """'dow jones' must beat the shorter 'dow'."""
        assert resolver._extract_ticker("how is the dow jones doing") == "^DJI"

    def test_falls_back_to_an_uppercase_symbol(self, resolver):
        assert resolver._extract_ticker("price of TSM today") == "TSM"

    def test_common_short_words_are_not_tickers(self, resolver):
        assert resolver._extract_ticker("IT IS ON US") is None

    def test_no_match_returns_none(self, resolver):
        assert resolver._extract_ticker("how are you today") is None

    @pytest.mark.parametrize(
        "text",
        [
            "what is the price of something",   # "som-ETH-ing"
            "I like goldfish",                  # "GOLD-fish"
            "an amderivative product",          # "AMD-erivative"
            "a meta-analysis of the data",      # hyphenated compound
            "the bitcoinery of it all",
        ],
    )
    def test_short_aliases_no_longer_match_inside_words(self, resolver, text):
        """
        Fixed 2026-08-05. _extract_ticker used bare `name in txt_lower` over
        _NAME_TO_TICKER, whose keys include 3-4 letter aliases ("eth", "btc",
        "amd", "dow", "oil", "gold", "meta"). Those matched inside ordinary
        words — "something" contains "eth", "goldfish" contains "gold" — so a
        turn mentioning a goldfish got a live gold-futures price injected.

        Matching is now word-bounded.
        """
        assert resolver._extract_ticker(text) is None

    @pytest.mark.parametrize(
        "text,ticker",
        [
            ("how is gold doing", "GC=F"),
            ("the price of oil", "CL=F"),
            ("what about eth", "ETH-USD"),
            ("meta earnings", "META"),
            ("amd stock", "AMD"),
            ("how is the dow", "^DJI"),
        ],
    )
    def test_short_aliases_still_match_as_whole_words(self, resolver, text, ticker):
        """The fix must not break legitimate short-alias lookups."""
        assert resolver._extract_ticker(text) == ticker

    def test_punctuation_bearing_alias_still_matches(self, resolver):
        r"""Plain `` misbehaves around "&"; the matcher uses [\w-] lookarounds."""
        assert resolver._extract_ticker("how is the s&p 500 doing") == "^GSPC"

    def test_longest_alias_wins(self, resolver):
        """Names are tried longest-first, so "intel" beats the shorter "amd"."""
        assert resolver._extract_ticker("amd vs intel") == "INTC"

    def test_standalone_common_words_still_match(self, resolver):
        """
        "gold"/"oil" as whole words do resolve. That is intended: word-boundary
        matching cannot tell the commodity from the idiom ("the gold standard"),
        and it does not need to — _extract_ticker is only reached once
        _is_market_query has already fired, which those phrases never do.
        """
        assert resolver._extract_ticker("the gold standard") == "GC=F"
        assert resolver._is_market_query("the gold standard") is False


class TestResolveMarket:
    @pytest.fixture
    def client(self):
        c = MagicMock()
        c.format_for_context.return_value = "FORMATTED"
        with patch.object(mc, "get_market_client", return_value=c):
            yield c

    def test_unidentifiable_instrument_returns_none(self, resolver, client):
        # Deliberately avoids any string containing a short alias like "eth"
        # or "gold" — see test_short_aliases_match_inside_unrelated_words.
        assert resolver._resolve_market("what is the current valuation") is None

    def test_current_price_is_the_default_path(self, resolver, client):
        out = resolver._resolve_market("what is nvidia trading at")
        client.get_current_price.assert_called_once()
        assert out is not None and out.source == "market"
        assert out.content == "FORMATTED"

    def test_yesterday_uses_the_dated_lookup(self, resolver, client):
        resolver._resolve_market("what was nvidia's price yesterday")
        client.get_price_on_date.assert_called_once()
        assert client.get_price_on_date.call_args.args[1] == date.today() - timedelta(days=1)

    @pytest.mark.parametrize("phrase", ["this week", "past week", "last week", "7-day"])
    def test_week_phrasing_uses_history(self, resolver, client, phrase):
        resolver._resolve_market(f"how has nvidia done {phrase}")
        client.get_recent_history.assert_called_once()

    def test_client_failure_returns_none(self, resolver, client):
        client.get_current_price.side_effect = RuntimeError("yfinance down")
        assert resolver._resolve_market("nvidia price") is None


class TestResolveNewsContextEntryPoint:
    def test_delegates_to_the_singleton(self):
        r = MagicMock()
        r.resolve.return_value = "CTX"
        with patch.object(nr, "get_news_resolver", return_value=r):
            assert nr.resolve_news_context("today's news") == "CTX"

    def test_swallows_unhandled_errors(self):
        with patch.object(nr, "get_news_resolver", side_effect=RuntimeError("db gone")):
            assert nr.resolve_news_context("today's news") is None

    def test_resolver_failure_is_not_fatal(self):
        r = MagicMock()
        r.resolve.side_effect = RuntimeError("boom")
        with patch.object(nr, "get_news_resolver", return_value=r):
            assert nr.resolve_news_context("today's news") is None


class TestGetNewsResolverSingleton:
    def test_is_a_singleton(self):
        nr.get_news_resolver.reset()
        with patch.object(nr, "NewsResolver", return_value=MagicMock()):
            assert nr.get_news_resolver() is nr.get_news_resolver()
        nr.get_news_resolver.reset()

    def test_built_once_under_concurrency(self):
        built: list[int] = []

        def _slow(*a, **kw):
            built.append(1)
            time.sleep(0.02)
            return MagicMock()

        nr.get_news_resolver.reset()
        with patch.object(nr, "NewsResolver", side_effect=_slow):
            threads = [threading.Thread(target=nr.get_news_resolver) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        nr.get_news_resolver.reset()

        assert len(built) == 1
