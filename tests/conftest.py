"""
tests/conftest.py

Shared pytest fixtures for the Sage Kaizen test suite.

Fixtures here are available to every test file without an explicit import.
"""
from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# psycopg3 connection / cursor mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cursor():
    """A mock psycopg3 cursor that supports the context manager protocol."""
    cur = MagicMock()
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    cur.execute.return_value = cur
    cur.fetchall.return_value = []
    cur.fetchone.return_value = None
    return cur


@pytest.fixture
def mock_conn(mock_cursor):
    """
    A mock psycopg3 Connection.

    .cursor() returns a context-manager that yields mock_cursor.
    .execute() is a no-op by default.
    .closed is False.
    """
    conn = MagicMock()
    conn.closed = False
    conn.broken = False
    conn.cursor.return_value = mock_cursor
    conn.execute.return_value = mock_cursor
    return conn


@pytest.fixture
def patch_psycopg_connect(mock_conn):
    """Patch psycopg.connect globally so no real DB connection is opened."""
    with patch("psycopg.connect", return_value=mock_conn) as mock_connect:
        yield mock_connect, mock_conn


# ---------------------------------------------------------------------------
# requests.Session mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_requests_session():
    """A mock requests.Session with a configurable .post() response."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.encoding = "utf-8"
    response.iter_lines.return_value = iter([])
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    session.post.return_value = response
    session.get.return_value = response
    return session, response


# ---------------------------------------------------------------------------
# Minimal RouteDecision helper
# ---------------------------------------------------------------------------

@pytest.fixture
def fast_decision():
    """A RouteDecision routing to FAST brain."""
    from router import RouteDecision
    return RouteDecision(brain="FAST", reasons=["test"], score=0)


@pytest.fixture
def architect_decision():
    """A RouteDecision routing to ARCHITECT brain."""
    from router import RouteDecision
    return RouteDecision(brain="ARCHITECT", reasons=["test"], score=5)
