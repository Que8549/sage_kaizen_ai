from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Iterator, NamedTuple

import requests


class LlamaServerError(RuntimeError):
    pass


# Per-host requests.Session cache — reuses TCP connections across turns.
# Plain requests.post()/get() open a new socket per call; a Session keeps the
# connection alive so the TCP handshake cost (~0.1–1 ms on loopback) is paid
# once per server, not once per turn.
_sessions: dict[str, requests.Session] = {}
_sessions_lock = threading.Lock()  # guards _sessions read-modify-write

# Reference to the currently-active streaming response so it can be closed
# from outside (e.g. shutdown monitor) to interrupt a long LLM stream.
_active_stream: requests.Response | None = None
_active_stream_lock = threading.Lock()


def abort_active_stream() -> None:
    """
    Close the active streaming HTTP response, if any.

    Causes the blocking recv() inside iter_lines() to raise a ConnectionError,
    which propagates up through stream_chat_completions() and write_stream(),
    allowing the Streamlit script to finish and the process to shut down.

    Safe to call from any thread.
    """
    global _active_stream
    with _active_stream_lock:
        resp = _active_stream
        _active_stream = None
    if resp is not None:
        try:
            resp.close()
        except Exception:
            pass
    # Also close all sessions so no new streams can start.
    with _sessions_lock:
        sessions_to_close = list(_sessions.values())
        _sessions.clear()
    for s in sessions_to_close:
        try:
            s.close()
        except Exception:
            pass


def _session(base_url: str) -> requests.Session:
    """Return (or create) a persistent Session for this base URL."""
    with _sessions_lock:
        sess = _sessions.get(base_url)
        if sess is None:
            sess = requests.Session()
            # Keep up to 4 connections per host alive in the pool
            adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=4)
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)
            _sessions[base_url] = sess
        return sess


@dataclass(frozen=True)
class HttpTimeouts:
    connect_s: float
    read_s: float


def _timeout_tuple(t: HttpTimeouts) -> tuple[float, float]:
    return (float(t.connect_s), float(t.read_s))


def _normalize_base_url(base_url: str) -> str:
    """
    Accept either:
      - http://127.0.0.1:8011
      - http://127.0.0.1:8011/
      - http://127.0.0.1:8011/v1
      - http://127.0.0.1:8011/v1/

    And normalize to the server ROOT (no trailing /v1).
    This avoids the common '/v1/v1/models' bug that makes readiness checks fail forever.
    """
    b = (base_url or "").strip().rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3].rstrip("/")
    return b


# Readiness probe order (llama.cpp documented endpoints).  Different
# llama-server builds/roles expose different subsets, hence the fallbacks.
_HEALTH_PATHS: tuple[str, ...] = ("/health", "/v1/health", "/v1/models", "/props")


class _ProbeResult(NamedTuple):
    ok: bool
    detail: str
    # True when the request failed below the HTTP layer — nothing accepted the
    # connection, or it accepted and never answered.  See health_check().
    transport_failed: bool


def _probe_get(base: str, path: str, *, timeouts: HttpTimeouts) -> _ProbeResult:
    """
    Probe one readiness path.

    Returns (ok, detail, transport_failed).
      ok               — True on HTTP 200.
      detail           — "OK (<path>)", "<path>=<status>", or "<path>=<ExcName>".
      transport_failed — True for connect refusal/timeout or read timeout, i.e.
                         a failure that says something about the *endpoint*
                         rather than about this particular path.
    """
    url = f"{base}{path}"
    try:
        r = _session(base).get(url, timeout=_timeout_tuple(timeouts))
        if r.status_code == 200:
            return _ProbeResult(True, f"OK ({path})", False)
        # A real HTTP response — the server is listening and answering, this
        # path just isn't the right one (404) or isn't ready yet (503).
        return _ProbeResult(False, f"{path}={r.status_code}", False)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        # ConnectionError covers refusal and DNS/socket failures; Timeout covers
        # ConnectTimeout and ReadTimeout.  requests' ConnectTimeout subclasses
        # both, so ordering these two in one except clause is safe.
        return _ProbeResult(False, f"{path}={type(e).__name__}", True)
    except Exception as e:
        return _ProbeResult(False, f"{path}={type(e).__name__}", False)


def health_check(base_url: str, *, timeouts: HttpTimeouts) -> tuple[bool, str]:
    """
    Returns (ok, detail).

    Probes readiness over _HEALTH_PATHS in order and returns True on the first
    HTTP 200, otherwise False with a combined detail string.

    Short-circuit on transport failure
    ----------------------------------
    Every path is on the same host:port.  If a probe cannot reach that endpoint
    at all — connection refused, connect timeout, or read timeout — the
    remaining paths cannot possibly succeed, so we stop instead of paying the
    timeout again for each one.

    This is not a micro-optimisation.  Before this change a down server cost
    4 x connect_s to diagnose: measured 8.03 s against a closed port with the
    HttpTimeouts(connect_s=2.0, read_s=5.0) that ChatService.decide_route()
    uses on every ambiguous-score turn, and twice that for the UI's two-brain
    status panel on every rerun.  It is now bounded by a single probe.

    A non-200 HTTP *response* is not a transport failure: the server is up and
    answering, so the remaining paths are still worth trying (an older build may
    404 on /health but serve /v1/models).
    """
    base = _normalize_base_url(base_url)
    details: list[str] = []

    for path in _HEALTH_PATHS:
        result = _probe_get(base, path, timeouts=timeouts)
        if result.ok:
            return True, result.detail
        details.append(result.detail)
        if result.transport_failed:
            skipped = len(_HEALTH_PATHS) - len(details)
            suffix = f"; {skipped} probe(s) skipped — endpoint unreachable" if skipped else ""
            return False, f"not ready ({'; '.join(details)}{suffix})"

    return False, f"not ready ({'; '.join(details)})"


def discover_model_id(base_url: str, *, timeouts: HttpTimeouts) -> str | None:
    """
    Returns the first model id from /v1/models, if available.
    Falls back to None if the endpoint is unavailable or payload is unexpected.
    """
    base = _normalize_base_url(base_url)
    try:
        r = _session(base).get(f"{base}/v1/models", timeout=_timeout_tuple(timeouts))
        r.raise_for_status()
        data = r.json()
        items = data.get("data", [])
        if isinstance(items, list) and items:
            mid = items[0].get("id")
            if isinstance(mid, str) and mid:
                return mid
    except Exception:
        return None
    return None


def _iter_sse_data_lines(resp: requests.Response) -> Iterator[str]:
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line:
            continue
        if line.startswith("data:"):
            yield line[len("data:"):].strip()


def call_brain_blocking(
    base_url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeouts: HttpTimeouts,
    top_k: int = -1,
    min_p: float = 0.0,
) -> str:
    """Non-streaming brain call — for background batch jobs that don't need live SSE.

    Returns the full response content string with thinking tokens already stripped
    by the server (stream=False never emits reasoning_content separately).
    """
    base = _normalize_base_url(base_url)
    url = f"{base}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "top_k": int(top_k),
        "min_p": float(min_p),
        "cache_prompt": True,
    }
    r = _session(base).post(url, json=payload, timeout=_timeout_tuple(timeouts))
    if r.status_code // 100 != 2:
        try:
            body = r.text
        except Exception:
            body = "<unreadable>"
        raise LlamaServerError(f"{url} returned HTTP {r.status_code}: {body[:500]}")
    data = r.json()
    return data["choices"][0]["message"].get("content", "") or ""


def stream_chat_completions(
    base_url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],  # Any: content may be str or multimodal list
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeouts: HttpTimeouts,
    top_k: int = -1,
    min_p: float = 0.0,
    thinking_budget: int = -1,
) -> Iterator[str]:
    base = _normalize_base_url(base_url)
    url = f"{base}/v1/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "top_k": int(top_k),
        "min_p": float(min_p),
        # Activate llama-server's prompt prefix cache per request.
        # The server matches this slot's KV prefix against prior turns and
        # reuses cached tokens, dramatically reducing TTFT for repeated
        # system-prompt / RAG prefixes (93% reduction with --cram enabled).
        "cache_prompt": True,
    }

    # thinking_budget controls the ARCHITECT reasoning token budget per-request.
    # -1 = unlimited (default for code/architecture/analysis turns)
    #  0 = no thinking (instant output, no CoT)
    #  N = cap at N thinking tokens then emit answer
    # Only inject when != -1 to avoid overriding the server default on FAST turns.
    if thinking_budget != -1:
        payload["thinking_budget"] = int(thinking_budget)

    global _active_stream
    with _session(base).post(url, json=payload, stream=True, timeout=_timeout_tuple(timeouts)) as r:
        # Register the response immediately so abort_active_stream() can close
        # it even if the status check or abort_active_stream() races with us.
        with _active_stream_lock:
            _active_stream = r
        try:
            if r.status_code // 100 != 2:
                try:
                    body = r.text
                except Exception:
                    body = "<unreadable>"
                raise LlamaServerError(f"{url} returned HTTP {r.status_code}: {body[:500]}")

            # llama-server sends text/event-stream without a charset declaration;
            # requests defaults to ISO-8859-1 for text/* types per the HTTP spec,
            # which garbles multi-byte UTF-8 characters (e.g. box-drawing, em-dash).
            r.encoding = "utf-8"
        except LlamaServerError:
            with _active_stream_lock:
                if _active_stream is r:
                    _active_stream = None
            raise

        try:
            _in_reasoning = False
            for data in _iter_sse_data_lines(r):
                if data == "[DONE]":
                    if _in_reasoning:
                        yield "</think>"
                    return
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0].get("delta", {})
                    reasoning = delta.get("reasoning_content")
                    content = delta.get("content")
                    if isinstance(reasoning, str) and reasoning:
                        if not _in_reasoning:
                            yield "<think>"
                            _in_reasoning = True
                        yield reasoning
                    if isinstance(content, str) and content:
                        if _in_reasoning:
                            yield "</think>"
                            _in_reasoning = False
                        yield content
                except Exception:
                    continue
        finally:
            with _active_stream_lock:
                if _active_stream is r:
                    _active_stream = None
