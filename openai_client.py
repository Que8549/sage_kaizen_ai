\
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests


class LlamaServerError(RuntimeError):
    pass


@dataclass(frozen=True)
class HttpTimeouts:
    connect_s: float
    read_s: float


def _timeout_tuple(t: HttpTimeouts) -> Tuple[float, float]:
    return (t.connect_s, t.read_s)


def health_check(base_url: str, *, timeouts: HttpTimeouts) -> Tuple[bool, str]:
    """
    Returns (ok, detail). Tries /health then /v1/models.
    """
    base = base_url.rstrip("/")
    # 1) /health (newer server versions)
    try:
        r = requests.get(f"{base}/health", timeout=_timeout_tuple(timeouts))
        if r.status_code == 200:
            return True, "OK (/health)"
    except Exception:
        pass

    # 2) /v1/models (OpenAI-compatible)
    try:
        r = requests.get(f"{base}/v1/models", timeout=_timeout_tuple(timeouts))
        if r.status_code == 200:
            return True, "OK (/v1/models)"
        return False, f"HTTP {r.status_code} at /v1/models"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def discover_model_id(base_url: str, *, timeouts: HttpTimeouts) -> Optional[str]:
    """
    Tries to read model id from /v1/models. Returns first model id if found.
    """
    base = base_url.rstrip("/")
    try:
        r = requests.get(f"{base}/v1/models", timeout=_timeout_tuple(timeouts))
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
    """
    Iterates over SSE lines that look like: 'data: {...}' or 'data: [DONE]'.
    """
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
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeouts: HttpTimeouts,
) -> Iterator[str]:
    """
    Calls POST {base_url}/v1/chat/completions with stream=True and yields token pieces (delta.content).
    Raises LlamaServerError on non-2xx responses.
    """
    base = base_url.rstrip("/")
    url = f"{base}/v1/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }

    with requests.post(url, json=payload, stream=True, timeout=_timeout_tuple(timeouts)) as r:
        if r.status_code // 100 != 2:
            # Try to include response body for debugging
            try:
                body = r.text
            except Exception:
                body = "<unreadable>"
            raise LlamaServerError(f"{url} returned HTTP {r.status_code}: {body[:500]}")

        for data in _iter_sse_data_lines(r):
            if data == "[DONE]":
                return
            try:
                obj = json.loads(data)
                delta = obj["choices"][0].get("delta", {})
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    yield piece
            except Exception:
                # Ignore malformed/keepalive chunks
                continue
