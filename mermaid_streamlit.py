from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components


# ----------------------------
# Live llama-server probing
# ----------------------------

@dataclass(frozen=True)
class LlamaServerInfo:
    base_url: str
    ok: bool
    model_id: Optional[str] = None
    alias: Optional[str] = None
    ctx_size: Optional[int] = None
    n_gpu_layers: Optional[Any] = None
    device: Optional[Any] = None
    tensor_split: Optional[Any] = None
    split_mode: Optional[Any] = None
    main_gpu: Optional[Any] = None
    raw: Optional[Dict[str, Any]] = None
    source: str = "unknown"  # which endpoint succeeded


def _http_get_json(url: str, timeout_s: float = 0.8) -> Tuple[bool, Optional[Dict[str, Any]]]:
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
        return True, json.loads(data.decode("utf-8", errors="replace"))
    except Exception:
        return False, None


def _first_non_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


# Cache probes briefly to avoid hammering endpoints during normal Streamlit reruns
@st.cache_data(ttl=10.0, show_spinner=False)
def probe_llama_server(base_url: str) -> LlamaServerInfo:
    """Best-effort live config discovery from a running llama-server."""
    base_url = (base_url or "").rstrip("/")

    # 1) /health (common)
    ok, payload = _http_get_json(f"{base_url}/health")
    if ok and isinstance(payload, dict):
        model_id = payload.get("model") or payload.get("model_id") or payload.get("id")
        return LlamaServerInfo(
            base_url=base_url,
            ok=True,
            model_id=model_id if isinstance(model_id, str) else None,
            raw=payload,
            source="/health",
        )

    # 2) /props (supported when --props is enabled)
    ok, payload = _http_get_json(f"{base_url}/props")
    if ok and isinstance(payload, dict):
        model_id = _first_non_none(
            payload.get("model"),
            payload.get("model_id"),
            payload.get("id"),
            (payload.get("model_info") or {}).get("model"),
            (payload.get("model_info") or {}).get("model_id"),
        )
        alias = _first_non_none(payload.get("alias"), payload.get("name"))
        ctx_size = _first_non_none(payload.get("ctx_size"), payload.get("n_ctx"), payload.get("context_length"))
        n_gpu_layers = _first_non_none(payload.get("n_gpu_layers"), payload.get("gpu_layers"))
        device = _first_non_none(payload.get("device"), payload.get("devices"))
        tensor_split = payload.get("tensor_split")
        split_mode = payload.get("split_mode")
        main_gpu = payload.get("main_gpu")

        # ctx_size can be str/int; normalize if possible
        norm_ctx: Optional[int] = None
        try:
            if ctx_size is not None:
                norm_ctx = int(ctx_size)
        except Exception:
            norm_ctx = None

        return LlamaServerInfo(
            base_url=base_url,
            ok=True,
            model_id=model_id if isinstance(model_id, str) else None,
            alias=alias if isinstance(alias, str) else None,
            ctx_size=norm_ctx,
            n_gpu_layers=n_gpu_layers,
            device=device,
            tensor_split=tensor_split,
            split_mode=split_mode,
            main_gpu=main_gpu,
            raw=payload,
            source="/props",
        )

    # 3) OpenAI-compatible /v1/models
    ok, payload = _http_get_json(f"{base_url}/v1/models")
    if ok and isinstance(payload, dict):
        data = payload.get("data")
        model_id = None
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                model_id = first.get("id")
        return LlamaServerInfo(
            base_url=base_url,
            ok=True,
            model_id=model_id if isinstance(model_id, str) else None,
            raw=payload,
            source="/v1/models",
        )

    return LlamaServerInfo(base_url=base_url, ok=False, raw=None, source="none")


def _fmt(v: Any, maxlen: int = 28) -> str:
    if v is None:
        return "?"
    s = str(v)
    return (s[: maxlen - 1] + "…") if len(s) > maxlen else s


def _mm_safe(s: str) -> str:
    """Make a string safe for Mermaid node labels."""
    # Mermaid labels get confused by unescaped quotes/brackets and some punctuation.
    # We'll render labels inside quoted node syntax: C["..."]
    # So we must escape double quotes and backslashes.
    s = (s or "")
    s = s.replace("\\", "\\\\").replace('"', "\\\"")
    # Avoid square brackets inside labels (can confuse bracketed node syntax).
    s = s.replace("[", "(").replace("]", ")")
    return s


def _preprocess_mermaid(src: str) -> str:
    """Sanitize Mermaid source to fix parse errors common in AI-generated diagrams.

    Mermaid v10 tokenizes '(' inside an unquoted '[...]' label as PS
    (parenthesis-start), which the parser rejects.  Converting those labels
    to quoted form — ["text (with parens)"] — is the canonical fix.

    Only targets labels that are *not* already quoted and contain '(' or ')'.
    Leaves quoted labels, sub-graph syntax, and shape variants untouched.
    """
    def _quote_if_needed(m: re.Match) -> str:
        inner = m.group(1)
        escaped = inner.replace('"', '\\"')
        return f'["{escaped}"]'

    # Match [text] where text:
    #   - does not start with " or ' (already quoted)
    #   - does not contain [ or ] (avoids [[subroutine]] and nested shapes)
    #   - contains at least one ( or )
    return re.sub(r'\[([^"\'\[\]]*[()][^"\'\[\]]*)\]', _quote_if_needed, src)


def build_sage_kaizen_mermaid(q5: Optional[LlamaServerInfo], q6: Optional[LlamaServerInfo]) -> str:
    """Build Mermaid diagram; uses quoted node labels + <br/> for line breaks."""

    def node_label(title: str, info: Optional[LlamaServerInfo]) -> str:
        lines = [title]
        if info and info.ok:
            if info.alias:
                lines.append(_fmt(info.alias))
            if info.model_id:
                # NO nested [] — use plain text
                lines.append(f"model={_fmt(info.model_id)}")
            lines.append(f"ctx={_fmt(info.ctx_size)}")
            lines.append(f"ngl={_fmt(info.n_gpu_layers)}")
            lines.append(f"dev={_fmt(info.device)}")
            if info.tensor_split is not None:
                lines.append(f"ts={_fmt(info.tensor_split)}")
            if info.split_mode is not None:
                lines.append(f"split={_fmt(info.split_mode)}")
        return _mm_safe("<br/>".join(lines))

    q5_label = node_label("GPU0 - Deep Reasoning", q5)
    q6_label = node_label("GPU1 - Low-Latency Responses", q6)

    # Use quoted node labels to tolerate spaces, punctuation, and <br/>
    return f"""graph TD
    A[\"User Request\"] --> B{{Routing Strategy}}
    B --> C[\"{q5_label}\"]
    B --> D[\"{q6_label}\"]
    C --> E{{KV Cache Allocation}}
    D --> F{{KV Cache Allocation}}
    E --> G[\"Large KV Cache\"]
    F --> H[\"Small KV Cache\"]
    G --> I[\"Process Request\"]
    H --> I
    I --> J{{Telemetry Collection}}
    J --> K[\"Request Latency\"]
    J --> L[\"Cache Utilization\"]
    J --> M[\"Error Logs\"]
    J --> N[\"Throughput\"]
    C --> O[\"Speculative Decoding\"]
    D --> P[\"Speculative Decoding\"]
"""


# ----------------------------
# Mermaid renderer + export
# ----------------------------


def render_mermaid_with_exports(
    diagram: str,
    *,
    height: int = 620,
    theme: str = "default",  # default | dark | forest | neutral
) -> None:
    """Render Mermaid in Streamlit and provide SVG/PNG download buttons."""

    diagram_text = _preprocess_mermaid((diagram or "").strip())

    # IMPORTANT: do NOT html.escape() the Mermaid source; it changes syntax.
    # Instead, embed it into JS safely via JSON string literal.
    diagram_js = json.dumps(diagram_text)

    mermaid_html = f"""<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <style>
      body {{
        margin: 0;
        background: transparent;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      }}
      .toolbar {{
        display: flex;
        gap: 8px;
        padding: 8px 10px;
        align-items: center;
        border-bottom: 1px solid rgba(128,128,128,0.25);
      }}
      button {{
        padding: 6px 10px;
        border-radius: 8px;
        border: 1px solid rgba(128,128,128,0.35);
        background: rgba(255,255,255,0.06);
        color: inherit;
        cursor: pointer;
      }}
      button:hover {{
        border-color: rgba(128,128,128,0.6);
      }}
      .status {{
        opacity: 0.75;
        font-size: 12px;
        margin-left: auto;
      }}
      #out {{ padding: 10px; }}
      #out svg {{ max-width: 100% !important; height: auto !important; }}
    </style>

    <script type=\"module\">
      import mermaid from \"https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs\";

      const graphDef = {diagram_js};

      mermaid.initialize({{
        startOnLoad: false,
        securityLevel: \"loose\",
        theme: {json.dumps(theme)}
      }});

      function downloadBlob(blob, filename) {{
        const url = URL.createObjectURL(blob);
        const a = document.createElement(\"a\");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }}

      function svgToPng(svgText, scale=2) {{
        return new Promise((resolve, reject) => {{
          const svgBlob = new Blob([svgText], {{type: \"image/svg+xml;charset=utf-8\"}});
          const url = URL.createObjectURL(svgBlob);
          const img = new Image();
          img.onload = () => {{
            try {{
              const w = img.width || 1200;
              const h = img.height || 800;
              const canvas = document.createElement(\"canvas\");
              canvas.width = Math.ceil(w * scale);
              canvas.height = Math.ceil(h * scale);
              const ctx = canvas.getContext(\"2d\");
              ctx.setTransform(scale, 0, 0, scale, 0, 0);
              ctx.drawImage(img, 0, 0);
              canvas.toBlob((b) => {{
                if (!b) reject(new Error(\"PNG conversion failed\"));
                else resolve(b);
              }}, \"image/png\");
            }} catch (e) {{
              reject(e);
            }} finally {{
              URL.revokeObjectURL(url);
            }}
          }};
          img.onerror = reject;
          img.src = url;
        }});
      }}

      async function render() {{
        const statusEl = document.getElementById(\"status\");
        try {{
          statusEl.textContent = \"Rendering…\";
          const id = \"m\" + Math.random().toString(16).slice(2);
          const result = await mermaid.render(id, graphDef);
          const svg = result.svg;
          window.__LAST_SVG__ = svg;
          document.getElementById(\"out\").innerHTML = svg;
          statusEl.textContent = \"Ready\";
        }} catch (e) {{
          statusEl.textContent = \"Render error\";
          document.getElementById(\"out\").innerHTML =
            \"<pre style='white-space:pre-wrap;color:#b00'>\" + String(e) + \"</pre>\";
        }}
      }}

      window.addEventListener(\"DOMContentLoaded\", () => {{
        document.getElementById(\"btnSvg\").addEventListener(\"click\", () => {{
          const svg = window.__LAST_SVG__;
          if (!svg) return;
          downloadBlob(new Blob([svg], {{type:\"image/svg+xml;charset=utf-8\"}}), \"sage_kaizen_arch.svg\");
        }});

        document.getElementById(\"btnPng\").addEventListener(\"click\", async () => {{
          const svg = window.__LAST_SVG__;
          if (!svg) return;
          try {{
            const pngBlob = await svgToPng(svg, 2);
            downloadBlob(pngBlob, \"sage_kaizen_arch.png\");
          }} catch (e) {{
            alert(\"PNG export failed: \" + e);
          }}
        }});

        document.getElementById(\"btnRerender\").addEventListener(\"click\", () => {{
          render();
        }});

        render();
      }});
    </script>
  </head>

  <body>
    <div class=\"toolbar\">
      <button id=\"btnRerender\" title=\"Re-render diagram\">Re-render</button>
      <button id=\"btnSvg\" title=\"Download as SVG\">Download SVG</button>
      <button id=\"btnPng\" title=\"Download as PNG\">Download PNG</button>
      <div class=\"status\" id=\"status\">Starting…</div>
    </div>
    <div id=\"out\"></div>
  </body>
</html>"""

    components.html(mermaid_html, height=height, scrolling=True)


# ----------------------------
# On-demand diagram handler
# ----------------------------

class DiagramHandler:
    """Detects Mermaid code blocks in AI responses and renders them inline.

    Usage (inside a st.chat_message context):
        DiagramHandler.render_if_present(response_text)
    """

    _MERMAID_BLOCK_RE = re.compile(
        r"```mermaid\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )

    # Phrases in user input or AI output that signal diagram intent.
    _TRIGGER_RE = re.compile(
        r"(?:create|generate|draw|show|view|display|make)\s+"
        r"(?:a\s+|an\s+|the\s+)?(?:mermaid\s+)?diagram"
        r"|mermaid\s+diagram"
        r"|architecture\s+diagram"
        r"|view\s+(?:the\s+)?architecture",
        re.IGNORECASE,
    )

    @classmethod
    def is_triggered(cls, text: str) -> bool:
        """Return True if *text* contains a diagram trigger phrase."""
        return bool(cls._TRIGGER_RE.search(text or ""))

    @classmethod
    def extract_mermaid(cls, text: str) -> Optional[str]:
        """Return the first ```mermaid ... ``` block found in *text*, or None."""
        m = cls._MERMAID_BLOCK_RE.search(text or "")
        return m.group(1).strip() if m else None

    @classmethod
    def render_if_present(
        cls,
        response_text: str,
        *,
        height: int = 500,
        theme: str = "default",
    ) -> bool:
        """If *response_text* contains a Mermaid block, render it inline.

        Call this inside a ``st.chat_message`` context so the diagram appears
        as part of the assistant message.  Returns True when a diagram was
        rendered.
        """
        src = cls.extract_mermaid(response_text)
        if src is None:
            return False
        render_mermaid_with_exports(src, height=height, theme=theme)
        return True
