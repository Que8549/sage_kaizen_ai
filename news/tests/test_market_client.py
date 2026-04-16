"""
news/tests/test_market_client.py

Unit tests for MarketClient.

Tests that:
  - get_current_price returns a dict with expected keys
  - get_price_on_date returns a dict with a close price
  - get_recent_history returns a dict with a history list
  - format_for_context produces a readable string
  - Error dicts are handled gracefully by format_for_context
  - normalize_ticker maps common names to symbols

yfinance is mocked so these tests run offline.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from news.retrieval.market_client import (
    MarketClient,
    normalize_ticker,
    get_market_client,
)


@pytest.fixture
def client():
    return MarketClient()


# ---------------------------------------------------------------------------
# normalize_ticker
# ---------------------------------------------------------------------------

def test_normalize_nvidia(client):
    assert normalize_ticker("nvidia") == "NVDA"

def test_normalize_bitcoin(client):
    assert normalize_ticker("bitcoin") == "BTC-USD"

def test_normalize_sp500(client):
    assert normalize_ticker("s&p 500") == "^GSPC"

def test_normalize_unknown_uppercases(client):
    assert normalize_ticker("xyzq") == "XYZQ"

def test_normalize_aapl_unchanged(client):
    assert normalize_ticker("AAPL") == "AAPL"


# ---------------------------------------------------------------------------
# get_current_price
# ---------------------------------------------------------------------------

def test_get_current_price_returns_expected_keys(client):
    mock_info = MagicMock()
    mock_info.last_price = 135.42
    mock_info.currency = "USD"

    mock_ticker = MagicMock()
    mock_ticker.fast_info = mock_info

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = client.get_current_price("AAPL")

    assert "ticker" in result
    assert "price" in result
    assert "currency" in result
    assert result["ticker"] == "AAPL"
    assert result["price"] == 135.42


def test_get_current_price_handles_exception(client):
    with patch("yfinance.Ticker", side_effect=RuntimeError("network error")):
        result = client.get_current_price("AAPL")

    assert "error" in result
    assert result["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# get_price_on_date
# ---------------------------------------------------------------------------

def test_get_price_on_date_returns_close(client):
    import pandas as pd
    mock_hist = pd.DataFrame(
        {"Close": [130.0]},
        index=pd.to_datetime(["2025-04-10"]),
    )
    mock_hist.index.name = "Date"

    with patch("yfinance.download", return_value=mock_hist):
        result = client.get_price_on_date("AAPL", date(2025, 4, 10))

    assert result["close"] == 130.0
    assert result["ticker"] == "AAPL"
    assert result["date"] == "2025-04-10"


def test_get_price_on_date_empty_returns_error(client):
    import pandas as pd
    with patch("yfinance.download", return_value=pd.DataFrame()):
        result = client.get_price_on_date("AAPL", date(2025, 4, 10))

    assert "error" in result


# ---------------------------------------------------------------------------
# get_recent_history
# ---------------------------------------------------------------------------

def test_get_recent_history_returns_rows(client):
    import pandas as pd
    import numpy as np
    dates = pd.to_datetime(["2025-04-07", "2025-04-08", "2025-04-09"])
    mock_hist = pd.DataFrame({
        "Open":  [100.0, 101.0, 102.0],
        "High":  [105.0, 106.0, 107.0],
        "Low":   [98.0,  99.0,  100.0],
        "Close": [103.0, 104.0, 105.0],
    }, index=dates)

    with patch("yfinance.download", return_value=mock_hist):
        result = client.get_recent_history("NVDA", days=7)

    assert "history" in result
    assert len(result["history"]) == 3
    assert result["history"][0]["close"] == 103.0


# ---------------------------------------------------------------------------
# format_for_context
# ---------------------------------------------------------------------------

def test_format_for_context_current_price(client):
    data = {"ticker": "NVDA", "price": 900.5, "currency": "USD", "timestamp": "2025-04-10T12:00:00Z"}
    text = client.format_for_context(data)
    assert "NVDA" in text
    assert "900.5" in text


def test_format_for_context_error(client):
    data = {"error": "No data available", "ticker": "XYZ"}
    text = client.format_for_context(data)
    assert "unavailable" in text.lower()


def test_format_for_context_history(client):
    data = {
        "ticker": "BTC-USD",
        "days": 7,
        "history": [
            {"date": "2025-04-07", "close": 80000.0},
            {"date": "2025-04-08", "close": 81000.0},
        ],
    }
    text = client.format_for_context(data)
    assert "BTC-USD" in text
    assert "80000" in text


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def test_get_market_client_returns_singleton():
    c1 = get_market_client()
    c2 = get_market_client()
    assert c1 is c2
