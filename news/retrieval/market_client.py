"""
news/retrieval/market_client.py

Lightweight market data wrapper using yfinance.

Handles point-in-time price lookups and short history windows for questions
such as:
  "What was Nvidia's stock price yesterday?"
  "What is Bitcoin trading at right now?"
  "How has AAPL performed this week?"

yfinance is local-first, no API key required, works on Windows.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.news.market_client")

# ---------------------------------------------------------------------------
# Ticker normalization map — common names → ticker symbols
# ---------------------------------------------------------------------------
_NAME_TO_TICKER: dict[str, str] = {
    "nvidia":        "NVDA",
    "nvda":          "NVDA",
    "apple":         "AAPL",
    "aapl":          "AAPL",
    "microsoft":     "MSFT",
    "msft":          "MSFT",
    "google":        "GOOGL",
    "alphabet":      "GOOGL",
    "googl":         "GOOGL",
    "amazon":        "AMZN",
    "amzn":          "AMZN",
    "tesla":         "TSLA",
    "tsla":          "TSLA",
    "meta":          "META",
    "facebook":      "META",
    "amd":           "AMD",
    "intel":         "INTC",
    "intc":          "INTC",
    "qualcomm":      "QCOM",
    "qcom":          "QCOM",
    "bitcoin":       "BTC-USD",
    "btc":           "BTC-USD",
    "ethereum":      "ETH-USD",
    "eth":           "ETH-USD",
    "sp500":         "^GSPC",
    "s&p 500":       "^GSPC",
    "s&p500":        "^GSPC",
    "nasdaq":        "^IXIC",
    "dow":           "^DJI",
    "dow jones":     "^DJI",
    "oil":           "CL=F",
    "crude oil":     "CL=F",
    "gold":          "GC=F",
}


def normalize_ticker(raw: str) -> str:
    """
    Convert a human-readable name or raw ticker to a yfinance-compatible symbol.

    Examples:
        "nvidia"  → "NVDA"
        "BTC"     → "BTC-USD"
        "AAPL"    → "AAPL"   (unchanged)
    """
    key = raw.strip().lower()
    if key in _NAME_TO_TICKER:
        return _NAME_TO_TICKER[key]
    return raw.strip().upper()


# ---------------------------------------------------------------------------
# MarketClient
# ---------------------------------------------------------------------------

class MarketClient:
    """
    Wraps yfinance for point-in-time and short-history price lookups.

    All methods return plain dicts suitable for JSON serialization and
    injection into the news context block.
    """

    def get_current_price(self, ticker_raw: str) -> dict:
        """
        Return the most recent available price for a ticker.

        Returns a dict with keys: ticker, price, currency, timestamp, source.
        Returns an error dict on failure.
        """
        ticker = normalize_ticker(ticker_raw)
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            currency = getattr(info, "currency", "USD")
            if price is None:
                return {"error": f"No price data available for {ticker}"}
            return {
                "ticker": ticker,
                "price": round(float(price), 4),
                "currency": currency,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "yfinance",
            }
        except Exception as exc:
            _LOG.warning("market_client | get_current_price | ticker=%s | %s", ticker, exc)
            return {"error": str(exc), "ticker": ticker}

    def get_price_on_date(self, ticker_raw: str, target_date: date) -> dict:
        """
        Return the closing price for a ticker on a specific date.

        Uses the day's OHLCV data; returns the adjusted close.
        """
        ticker = normalize_ticker(ticker_raw)
        try:
            import yfinance as yf
            start = target_date
            end   = target_date + timedelta(days=1)
            hist  = yf.download(ticker, start=start, end=end,
                                progress=False, auto_adjust=True)
            if hist.empty:
                return {"error": f"No data for {ticker} on {target_date}"}
            close = float(hist["Close"].iloc[-1])
            return {
                "ticker": ticker,
                "date": str(target_date),
                "close": round(close, 4),
                "source": "yfinance",
            }
        except Exception as exc:
            _LOG.warning("market_client | get_price_on_date | ticker=%s | %s", ticker, exc)
            return {"error": str(exc), "ticker": ticker}

    def get_recent_history(self, ticker_raw: str, days: int = 7) -> dict:
        """
        Return a short price history (OHLCV) for the last N calendar days.
        """
        ticker = normalize_ticker(ticker_raw)
        try:
            import yfinance as yf
            end   = date.today()
            start = end - timedelta(days=days)
            hist  = yf.download(ticker, start=start, end=end,
                                progress=False, auto_adjust=True)
            if hist.empty:
                return {"error": f"No history for {ticker}"}
            rows = []
            for idx_date, row in hist.iterrows():
                rows.append({
                    "date":  str(idx_date.date()),
                    "open":  round(float(row["Open"]),  4),
                    "high":  round(float(row["High"]),  4),
                    "low":   round(float(row["Low"]),   4),
                    "close": round(float(row["Close"]), 4),
                })
            return {
                "ticker":  ticker,
                "days":    days,
                "history": rows,
                "source":  "yfinance",
            }
        except Exception as exc:
            _LOG.warning("market_client | get_recent_history | ticker=%s | %s", ticker, exc)
            return {"error": str(exc), "ticker": ticker}

    def format_for_context(self, data: dict) -> str:
        """Convert a market data dict to a short readable context string."""
        if "error" in data:
            return f"Market data unavailable: {data['error']}"
        if "history" in data:
            lines = [f"{r['date']}: close={r['close']}" for r in data["history"][-5:]]
            return (
                f"{data['ticker']} ({data['days']}-day history)\n"
                + "\n".join(lines)
            )
        if "close" in data:
            return f"{data['ticker']} on {data['date']}: close={data['close']} ({data.get('source', '')})"
        if "price" in data:
            return (
                f"{data['ticker']}: {data['price']} {data.get('currency', 'USD')} "
                f"(as of {data.get('timestamp', 'now')})"
            )
        return str(data)


# Module-level lazy singleton.
_client: Optional[MarketClient] = None


def get_market_client() -> MarketClient:
    global _client
    if _client is None:
        _client = MarketClient()
    return _client
