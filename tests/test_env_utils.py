"""
tests/test_env_utils.py

Unit tests for env_utils.py — per-call environment-variable accessors.

These are pure-function tests: no DB, no network, no mocking beyond
os.environ manipulation via monkeypatch.
"""
from __future__ import annotations

import pytest
from env_utils import env_bool, env_float, env_int, env_str


# ---------------------------------------------------------------------------
# env_bool
# ---------------------------------------------------------------------------

class TestEnvBool:
    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "YES", "y", "Y", "on", "ON"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("TEST_BOOL", value)
        assert env_bool("TEST_BOOL") is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "NO", "n", "N", "off", "OFF", "anything"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("TEST_BOOL", value)
        assert env_bool("TEST_BOOL") is False

    def test_missing_returns_default_false(self, monkeypatch):
        monkeypatch.delenv("TEST_BOOL", raising=False)
        assert env_bool("TEST_BOOL") is False

    def test_missing_returns_custom_default(self, monkeypatch):
        monkeypatch.delenv("TEST_BOOL", raising=False)
        assert env_bool("TEST_BOOL", default=True) is True

    def test_empty_string_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "")
        assert env_bool("TEST_BOOL", default=True) is True

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "  true  ")
        assert env_bool("TEST_BOOL") is True


# ---------------------------------------------------------------------------
# env_int
# ---------------------------------------------------------------------------

class TestEnvInt:
    def test_valid_integer(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "42")
        assert env_int("TEST_INT", default=0) == 42

    def test_negative_integer(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "-7")
        assert env_int("TEST_INT", default=0) == -7

    def test_missing_returns_default(self, monkeypatch):
        monkeypatch.delenv("TEST_INT", raising=False)
        assert env_int("TEST_INT", default=99) == 99

    def test_empty_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "")
        assert env_int("TEST_INT", default=5) == 5

    def test_non_numeric_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "abc")
        assert env_int("TEST_INT", default=3) == 3

    def test_float_string_returns_default(self, monkeypatch):
        # int("3.14") raises ValueError → default
        monkeypatch.setenv("TEST_INT", "3.14")
        assert env_int("TEST_INT", default=0) == 0

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "  10  ")
        assert env_int("TEST_INT", default=0) == 10


# ---------------------------------------------------------------------------
# env_float
# ---------------------------------------------------------------------------

class TestEnvFloat:
    def test_valid_float(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        assert env_float("TEST_FLOAT", default=0.0) == pytest.approx(3.14)

    def test_integer_string_converts_to_float(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "5")
        assert env_float("TEST_FLOAT", default=0.0) == pytest.approx(5.0)

    def test_missing_returns_default(self, monkeypatch):
        monkeypatch.delenv("TEST_FLOAT", raising=False)
        assert env_float("TEST_FLOAT", default=1.5) == pytest.approx(1.5)

    def test_empty_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "")
        assert env_float("TEST_FLOAT", default=2.0) == pytest.approx(2.0)

    def test_non_numeric_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "not_a_number")
        assert env_float("TEST_FLOAT", default=9.9) == pytest.approx(9.9)

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "  2.5  ")
        assert env_float("TEST_FLOAT", default=0.0) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# env_str
# ---------------------------------------------------------------------------

class TestEnvStr:
    def test_returns_value(self, monkeypatch):
        monkeypatch.setenv("TEST_STR", "hello")
        assert env_str("TEST_STR", default="") == "hello"

    def test_missing_returns_default(self, monkeypatch):
        monkeypatch.delenv("TEST_STR", raising=False)
        assert env_str("TEST_STR", default="fallback") == "fallback"

    def test_empty_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_STR", "")
        assert env_str("TEST_STR", default="fallback") == "fallback"

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("TEST_STR", "  trimmed  ")
        assert env_str("TEST_STR", default="") == "trimmed"

    def test_url_preserved(self, monkeypatch):
        url = "http://localhost:8080"
        monkeypatch.setenv("TEST_STR", url)
        assert env_str("TEST_STR", default="") == url
