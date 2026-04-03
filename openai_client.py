from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests


class LlamaServerError(RuntimeError):
    pass


# Per-host requests.Session cache — reuses TCP connections across turns.
# Plain requests.post()/get() open a new socket per call; a Session keeps the
# connection alive so the TCP handshake cost (~0.1–1 ms on loopback) is paid
# once per server, not once per turn.
_sessions: Dict[str, requests.Session] = {}


def _session(base_url: str) -> requests.Session:
    """Return (or create) a persistent Session for this base URL."""
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


def _timeout_tuple(t: HttpTimeouts) -> Tuple[float, float]:
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


def _probe_get(base: str, path: str, *, timeouts: HttpTimeouts) -> Tuple[bool, str]:
    """
    Returns (ok, detail). ok is True if HTTP 200.
    detail includes status code or exception type.
    """
    url = f"{base}{path}"
    try:
        r = _session(base).get(url, timeout=_timeout_tuple(timeouts))
        if r.status_code == 200:
            return True, f"OK ({path})"
        return False, f"{path}={r.status_code}"
    except Exception as e:
        return False, f"{path}={type(e).__name__}"


def health_check(base_url: str, *, timeouts: HttpTimeouts) -> Tuple[bool, str]:
    """
    Returns (ok, detail).

    Probes readiness in this order (llama.cpp documented endpoints):
      1) GET /health
      2) GET /v1/health
      3) GET /v1/models
      4) GET /props

    We return True on the first success (HTTP 200), otherwise False with a combined detail string.
    """
    base = _normalize_base_url(base_url)

    ok, d1 = _probe_get(base, "/health", timeouts=timeouts)
    if ok:
        return True, d1

    ok, d2 = _probe_get(base, "/v1/health", timeouts=timeouts)
    if ok:
        return True, d2

    ok, d3 = _probe_get(base, "/v1/models", timeouts=timeouts)
    if ok:
        return True, d3

    ok, d4 = _probe_get(base, "/props", timeouts=timeouts)
    if ok:
        return True, d4

    return False, f"not ready ({d1}; {d2}; {d3}; {d4})"


def discover_model_id(base_url: str, *, timeouts: HttpTimeouts) -> Optional[str]:
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


def stream_chat_completions(
    base_url: str,
    *,
    model: str,
    messages: List[Dict[str, Any]],  # Any: content may be str or multimodal list
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeouts: HttpTimeouts,
    top_k: int = -1,
    min_p: float = 0.0,
) -> Iterator[str]:
    base = _normalize_base_url(base_url)
    url = f"{base}/v1/chat/completions"

    payload: Dict[str, Any] = {
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

    with _session(base).post(url, json=payload, stream=True, timeout=_timeout_tuple(timeouts)) as r:
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
