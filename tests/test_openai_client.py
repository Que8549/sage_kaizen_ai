"""
tests/test_openai_client.py

Unit tests for openai_client.py.

Covers:
- _session: returns and caches a Session per base URL
- _session: thread-safety under concurrent access (no duplicate sessions)
- _session: uses _sessions_lock (read-modify-write protected)
- abort_active_stream: closes active stream and clears sessions under lock
- stream_chat_completions: registers _active_stream before status check
- stream_chat_completions: deregisters _active_stream in finally block
- _normalize_base_url: strips /v1 suffix
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


@pytest.fixture(autouse=True)
def reset_openai_client_state():
    """Clear module-level singletons before each test."""
    import openai_client as oc
    oc._sessions.clear()
    oc._active_stream = None
    yield
    oc._sessions.clear()
    oc._active_stream = None


# ---------------------------------------------------------------------------
# _normalize_base_url
# ---------------------------------------------------------------------------

class TestNormalizeBaseUrl:
    def test_strips_v1_suffix(self):
        from openai_client import _normalize_base_url
        assert _normalize_base_url("http://127.0.0.1:8011/v1") == "http://127.0.0.1:8011"

    def test_strips_trailing_slash(self):
        from openai_client import _normalize_base_url
        assert _normalize_base_url("http://127.0.0.1:8011/") == "http://127.0.0.1:8011"

    def test_passthrough_clean_url(self):
        from openai_client import _normalize_base_url
        url = "http://127.0.0.1:8011"
        assert _normalize_base_url(url) == url

    def test_strips_v1_with_trailing_slash(self):
        from openai_client import _normalize_base_url
        assert _normalize_base_url("http://127.0.0.1:8011/v1/") == "http://127.0.0.1:8011"


# ---------------------------------------------------------------------------
# _session — caching and thread-safety
# ---------------------------------------------------------------------------

class TestSession:
    def test_returns_session_for_url(self):
        from openai_client import _session
        import requests
        with patch("openai_client.requests.Session") as MockSession:
            MockSession.return_value = MagicMock()
            s = _session("http://127.0.0.1:8011")
            assert s is MockSession.return_value

    def test_caches_session_for_same_url(self):
        from openai_client import _session
        with patch("openai_client.requests.Session") as MockSession:
            MockSession.return_value = MagicMock()
            s1 = _session("http://127.0.0.1:8011")
            s2 = _session("http://127.0.0.1:8011")
            assert s1 is s2
            assert MockSession.call_count == 1

    def test_different_urls_get_different_sessions(self):
        from openai_client import _session
        sess_a = MagicMock()
        sess_b = MagicMock()
        with patch("openai_client.requests.Session", side_effect=[sess_a, sess_b]):
            s1 = _session("http://127.0.0.1:8011")
            s2 = _session("http://127.0.0.1:8012")
            assert s1 is sess_a
            assert s2 is sess_b

    def test_concurrent_calls_create_exactly_one_session(self):
        """No duplicate Session objects for the same URL under parallel access."""
        from openai_client import _session, _sessions
        sessions_created = []

        original_session_cls = None

        import requests as _requests

        class TrackingSession(_requests.Session):
            def __init__(self):
                super().__init__()
                sessions_created.append(self)

        with patch("openai_client.requests.Session", TrackingSession):
            barrier = threading.Barrier(10)

            def call_session():
                barrier.wait()  # All threads start simultaneously
                return _session("http://127.0.0.1:8011")

            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = [pool.submit(call_session) for _ in range(10)]
                results = [f.result() for f in futures]

        # All threads should have gotten the same session object
        assert len(set(id(r) for r in results)) == 1
        # Only one Session should have been constructed
        assert len(sessions_created) == 1


# ---------------------------------------------------------------------------
# abort_active_stream
# ---------------------------------------------------------------------------

class TestAbortActiveStream:
    def test_closes_active_stream_response(self):
        import openai_client as oc
        mock_resp = MagicMock()
        oc._active_stream = mock_resp

        oc.abort_active_stream()

        mock_resp.close.assert_called_once()
        assert oc._active_stream is None

    def test_noop_when_no_active_stream(self):
        import openai_client as oc
        assert oc._active_stream is None
        oc.abort_active_stream()  # must not raise

    def test_clears_all_sessions(self):
        import openai_client as oc
        sess = MagicMock()
        oc._sessions["http://127.0.0.1:8011"] = sess

        oc.abort_active_stream()

        sess.close.assert_called_once()
        assert len(oc._sessions) == 0

    def test_thread_safe_clears_sessions_under_lock(self):
        """abort_active_stream and _session must not race on _sessions dict."""
        import openai_client as oc
        errors: list[Exception] = []

        def aborter():
            for _ in range(50):
                oc.abort_active_stream()

        def session_getter():
            import requests as _r
            with patch("openai_client.requests.Session", _r.Session):
                for _ in range(50):
                    try:
                        oc._session("http://127.0.0.1:8011")
                    except Exception as e:
                        errors.append(e)

        t1 = threading.Thread(target=aborter)
        t2 = threading.Thread(target=session_getter)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == [], f"Race condition errors: {errors}"


# ---------------------------------------------------------------------------
# stream_chat_completions — _active_stream lifecycle
# ---------------------------------------------------------------------------

class TestStreamChatCompletions:
    def _make_sse_response(self, tokens: list[str]) -> MagicMock:
        """Build a mock SSE response yielding the given content tokens."""
        lines = []
        for token in tokens:
            obj = {"choices": [{"delta": {"content": token}}]}
            lines.append(f'data: {json.dumps(obj)}')
        lines.append("data: [DONE]")

        resp = MagicMock()
        resp.status_code = 200
        resp.encoding = "utf-8"
        resp.iter_lines.return_value = iter(lines)
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_active_stream_registered_during_iteration(self):
        """_active_stream must be set before the first SSE line is consumed."""
        import openai_client as oc

        sse_lines = [
            'data: {"choices": [{"delta": {"content": "hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            "data: [DONE]",
        ]

        registered_during: list = []

        resp = MagicMock()
        resp.status_code = 200
        resp.encoding = "utf-8"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        def capturing_iter(decode_unicode=True):
            # Capture _active_stream state at the moment iteration begins
            registered_during.append(oc._active_stream)
            return iter(sse_lines)

        resp.iter_lines = capturing_iter

        with patch("openai_client._session") as mock_sess_fn:
            sess = MagicMock()
            sess.post.return_value = resp
            mock_sess_fn.return_value = sess

            from openai_client import HttpTimeouts, stream_chat_completions
            tokens = list(stream_chat_completions(
                "http://127.0.0.1:8011",
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.7, top_p=0.95, max_tokens=100,
                timeouts=HttpTimeouts(connect_s=5, read_s=30),
            ))

        assert tokens == ["hello", " world"]
        # _active_stream must have been set (to resp) before iter_lines was called
        assert len(registered_during) == 1
        assert registered_during[0] is resp

    def test_active_stream_cleared_after_completion(self):
        import openai_client as oc
        resp = self._make_sse_response(["hi"])

        with patch("openai_client._session") as mock_sess_fn:
            sess = MagicMock()
            sess.post.return_value = resp
            mock_sess_fn.return_value = sess

            from openai_client import HttpTimeouts, stream_chat_completions
            list(stream_chat_completions(
                "http://127.0.0.1:8011",
                model="test-model",
                messages=[],
                temperature=0.7, top_p=0.95, max_tokens=100,
                timeouts=HttpTimeouts(connect_s=5, read_s=30),
            ))

        assert oc._active_stream is None

    def test_active_stream_cleared_on_error_response(self):
        """HTTP 500 → LlamaServerError; _active_stream must be cleared."""
        import openai_client as oc
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Internal Server Error"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        with patch("openai_client._session") as mock_sess_fn:
            sess = MagicMock()
            sess.post.return_value = resp
            mock_sess_fn.return_value = sess

            from openai_client import HttpTimeouts, LlamaServerError, stream_chat_completions
            with pytest.raises(LlamaServerError):
                list(stream_chat_completions(
                    "http://127.0.0.1:8011",
                    model="test-model",
                    messages=[],
                    temperature=0.7, top_p=0.95, max_tokens=100,
                    timeouts=HttpTimeouts(connect_s=5, read_s=30),
                ))

        assert oc._active_stream is None

    def test_yields_content_tokens(self):
        resp = self._make_sse_response(["The ", "answer ", "is 42"])

        with patch("openai_client._session") as mock_sess_fn:
            sess = MagicMock()
            sess.post.return_value = resp
            mock_sess_fn.return_value = sess

            from openai_client import HttpTimeouts, stream_chat_completions
            tokens = list(stream_chat_completions(
                "http://127.0.0.1:8011",
                model="test-model",
                messages=[{"role": "user", "content": "what is 6*7?"}],
                temperature=0.0, top_p=1.0, max_tokens=50,
                timeouts=HttpTimeouts(connect_s=5, read_s=30),
            ))

        assert tokens == ["The ", "answer ", "is 42"]

    def test_thinking_tokens_wrapped_in_think_tags(self):
        """reasoning_content → <think>…</think> wrapper."""
        lines = [
            'data: {"choices": [{"delta": {"reasoning_content": "hmm"}}]}',
            'data: {"choices": [{"delta": {"content": "answer"}}]}',
            "data: [DONE]",
        ]
        resp = MagicMock()
        resp.status_code = 200
        resp.encoding = "utf-8"
        resp.iter_lines.return_value = iter(lines)
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        with patch("openai_client._session") as mock_sess_fn:
            sess = MagicMock()
            sess.post.return_value = resp
            mock_sess_fn.return_value = sess

            from openai_client import HttpTimeouts, stream_chat_completions
            tokens = list(stream_chat_completions(
                "http://127.0.0.1:8012",
                model="architect",
                messages=[],
                temperature=0.6, top_p=0.95, max_tokens=200,
                timeouts=HttpTimeouts(connect_s=5, read_s=120),
            ))

        assert tokens[0] == "<think>"
        assert "hmm" in tokens
        assert "</think>" in tokens
        assert "answer" in tokens
