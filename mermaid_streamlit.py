# mermaid_streamlit.py
from __future__ import annotations

import html
import json
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


@st.cache_data(ttl=2.0, show_spinner=False)
def probe_llama_server(base_url: str) -> LlamaServerInfo:
    """
    Attempts to read live config from a running llama-server.
    Tries (in order): /health, /props, /v1/models
    Returns best-effort normalized fields + raw payload.
    """
    base_url = base_url.rstrip("/")

    # 1) /health (common)
    ok, payload = _http_get_json(f"{base_url}/health")
    if ok and isinstance(payload, dict):
        # Some builds return model info here (or status only).
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
        # Keys vary a bit by build/version; we keep this defensive.
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
        return LlamaServerInfo(
            base_url=base_url,
            ok=True,
            model_id=model_id if isinstance(model_id, str) else None,
            alias=alias if isinstance(alias, str) else None,
            ctx_size=int(ctx_size) if isinstance(ctx_size, (int, float, str)) and str(ctx_size).isdigit() else None,
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


def build_sage_kaizen_mermaid(
    q5: Optional[LlamaServerInfo],
    q6: Optional[LlamaServerInfo],
) -> str:
    """
    Builds a Mermaid diagram with labels filled from live server config.
    """
    q5_label = "GPU0 - Deep Reasoning"
    q6_label = "GPU1 - Low-Latency Responses"

    if q5 and q5.ok:
        q5_label += f"\\n{_fmt(q5.alias) if q5.alias else ''}{('' if not q5.model_id else f'[{_fmt(q5.model_id)}]')}"
        q5_label += f"\\nctx={_fmt(q5.ctx_size)}"
        q5_label += f"\\nngl={_fmt(q5.n_gpu_layers)}"
        q5_label += f"\\ndev={_fmt(q5.device)}"

    if q6 and q6.ok:
        q6_label += f"\\n{_fmt(q6.alias) if q6.alias else ''}{('' if not q6.model_id else f'[{_fmt(q6.model_id)}]')}"
        q6_label += f"\\nctx={_fmt(q6.ctx_size)}"
        q6_label += f"\\nngl={_fmt(q6.n_gpu_layers)}"
        q6_label += f"\\ndev={_fmt(q6.device)}"
        if q6.tensor_split is not None:
            q6_label += f"\\nts={_fmt(q6.tensor_split)}"
        if q6.split_mode is not None:
            q6_label += f"\\nsplit={_fmt(q6.split_mode)}"

    return f"""\
graph TD
    A[User Request] --> B{{Routing Strategy}}
    B --> C[{q5_label}]
    B --> D[{q6_label}]
    C --> E{{KV Cache Allocation}}
    D --> F{{KV Cache Allocation}}
    E --> G[Large KV Cache]
    F --> H[Small KV Cache]
    G --> I[Process Request]
    H --> I[Process Request]
    I --> J{{Telemetry Collection}}
    J --> K[Request Latency]
    J --> L[Cache Utilization]
    J --> M[Error Logs]
    J --> N[Throughput]
    C --> O[Speculative Decoding]
    D --> P[Speculative Decoding]
"""


# ----------------------------
# Mermaid renderer + export
# ----------------------------

def render_mermaid_with_exports(
    diagram: str,
    *,
    height: int = 620,
    theme: str = "default",   # default | dark | forest | neutral
) -> None:
    """
    Renders Mermaid and provides:
    - Download SVG
    - Download PNG (canvas)
    All handled inside the iframe (no Python roundtrip needed).
    """
    safe_diagram = html.escape((diagram or "").strip())
    safe_theme = html.escape(theme)

    # Uses mermaid.render() as recommended by Mermaid usage docs
    # and then injects resulting SVG into the DOM. :contentReference[oaicite:2]{index=2}
    mermaid_html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
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
      #out {{
        padding: 10px;
      }}
      #out svg {{
        max-width: 100% !important;
        height: auto !important;
      }}
      pre {{
        display:none;
      }}
    </style>

    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";

      const graphDef = `{safe_diagram}`;

      mermaid.initialize({{
        startOnLoad: false,
        securityLevel: "loose",
        theme: "{safe_theme}"
      }});

      function downloadBlob(blob, filename) {{
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }}

      function svgToPng(svgText, scale=2) {{
        return new Promise((resolve, reject) => {{
          const svgBlob = new Blob([svgText], {{type: "image/svg+xml;charset=utf-8"}});
          const url = URL.createObjectURL(svgBlob);
          const img = new Image();
          img.onload = () => {{
            try {{
              const w = img.width || 1200;
              const h = img.height || 800;
              const canvas = document.createElement("canvas");
              canvas.width = Math.ceil(w * scale);
              canvas.height = Math.ceil(h * scale);
              const ctx = canvas.getContext("2d");
              ctx.setTransform(scale, 0, 0, scale, 0, 0);
              ctx.drawImage(img, 0, 0);
              canvas.toBlob((b) => {{
                if (!b) reject(new Error("PNG conversion failed"));
                else resolve(b);
              }}, "image/png");
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
        const statusEl = document.getElementById("status");
        try {{
          statusEl.textContent = "Rendering…";
          const id = "m" + Math.random().toString(16).slice(2);
          const {{ svg }} = await mermaid.render(id, graphDef);
          window.__LAST_SVG__ = svg;
          const out = document.getElementById("out");
          out.innerHTML = svg;
          statusEl.textContent = "Ready";
        }} catch (e) {{
          statusEl.textContent = "Render error";
          const out = document.getElementById("out");
          out.innerHTML = "<pre style='white-space:pre-wrap;color:#b00'>" + String(e) + "</pre>";
        }}
      }}

      window.addEventListener("DOMContentLoaded", () => {{
        document.getElementById("btnSvg").addEventListener("click", () => {{
          const svg = window.__LAST_SVG__;
          if (!svg) return;
          downloadBlob(new Blob([svg], {{type:"image/svg+xml;charset=utf-8"}}), "sage_kaizen_arch.svg");
        }});

        document.getElementById("btnPng").addEventListener("click", async () => {{
          const svg = window.__LAST_SVG__;
          if (!svg) return;
          try {{
            const pngBlob = await svgToPng(svg, 2);
            downloadBlob(pngBlob, "sage_kaizen_arch.png");
          }} catch (e) {{
            alert("PNG export failed: " + e);
          }}
        }});

        document.getElementById("btnRerender").addEventListener("click", () => {{
          render();
        }});

        render();
      }});
    </script>
  </head>

  <body>
    <div class="toolbar">
      <button id="btnRerender" title="Re-render diagram">Re-render</button>
      <button id="btnSvg" title="Download as SVG">Download SVG</button>
      <button id="btnPng" title="Download as PNG">Download PNG</button>
      <div class="status" id="status">Starting…</div>
    </div>
    <div id="out"></div>
    <pre class="mermaid">{safe_diagram}</pre>
  </body>
</html>
"""
    # No `key=` here (Streamlit docs signature doesn't include it). :contentReference[oaicite:3]{index=3}
    components.html(mermaid_html, height=height, scrolling=True)
