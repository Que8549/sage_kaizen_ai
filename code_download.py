"""
code_download.py

Detects fenced code blocks in AI responses and renders a Streamlit download
button for each block whose language tag maps to a known file type.

Usage (inside a st.chat_message context):
    CodeFileHandler.render_downloads(response_text)
    CodeFileHandler.render_downloads(response_text, key_prefix="hist_42")

This mirrors the DiagramHandler pattern in mermaid_streamlit.py.
No inline preview is rendered — download only.
"""
from __future__ import annotations

import re

import streamlit as st

# ---------------------------------------------------------------------------
# Language tag → (file extension, display label, MIME type)
# ---------------------------------------------------------------------------

_LANG_META: dict[str, tuple[str, str, str]] = {
    # Web
    "html":       (".html", "HTML",        "text/html"),
    "htm":        (".html", "HTML",        "text/html"),
    "css":        (".css",  "CSS",         "text/css"),
    "javascript": (".js",   "JavaScript",  "application/javascript"),
    "js":         (".js",   "JavaScript",  "application/javascript"),
    "typescript": (".ts",   "TypeScript",  "text/x-typescript"),
    "ts":         (".ts",   "TypeScript",  "text/x-typescript"),
    "jsx":        (".jsx",  "JSX",         "text/jsx"),
    "tsx":        (".tsx",  "TSX",         "text/tsx"),
    # Systems / backend
    "python":     (".py",   "Python",      "text/x-python"),
    "py":         (".py",   "Python",      "text/x-python"),
    "csharp":     (".cs",   "C#",          "text/x-csharp"),
    "cs":         (".cs",   "C#",          "text/x-csharp"),
    "java":       (".java", "Java",        "text/x-java-source"),
    "cpp":        (".cpp",  "C++",         "text/x-c++src"),
    "c":          (".c",    "C",           "text/x-csrc"),
    "go":         (".go",   "Go",          "text/x-go"),
    "rust":       (".rs",   "Rust",        "text/x-rustsrc"),
    "rs":         (".rs",   "Rust",        "text/x-rustsrc"),
    "swift":      (".swift","Swift",       "text/x-swift"),
    "kotlin":     (".kt",   "Kotlin",      "text/x-kotlin"),
    "kt":         (".kt",   "Kotlin",      "text/x-kotlin"),
    "ruby":       (".rb",   "Ruby",        "text/x-ruby"),
    "rb":         (".rb",   "Ruby",        "text/x-ruby"),
    "php":        (".php",  "PHP",         "text/x-php"),
    "lua":        (".lua",  "Lua",         "text/x-lua"),
    "zig":        (".zig",  "Zig",         "text/x-zig"),
    "r":          (".r",    "R",           "text/x-r"),
    # Data / config
    "sql":        (".sql",  "SQL",         "application/sql"),
    "json":       (".json", "JSON",        "application/json"),
    "xml":        (".xml",  "XML",         "application/xml"),
    "yaml":       (".yaml", "YAML",        "application/x-yaml"),
    "yml":        (".yml",  "YAML",        "application/x-yaml"),
    "toml":       (".toml", "TOML",        "application/toml"),
    # Shell / scripting
    "bash":       (".sh",   "Bash",        "application/x-sh"),
    "sh":         (".sh",   "Shell",       "application/x-sh"),
    "powershell": (".ps1",  "PowerShell",  "application/x-powershell"),
    "ps1":        (".ps1",  "PowerShell",  "application/x-powershell"),
    # Docs
    "markdown":   (".md",   "Markdown",    "text/markdown"),
    "md":         (".md",   "Markdown",    "text/markdown"),
}

# Default filename stem when the language doesn't suggest a better one.
_DEFAULT_STEM: dict[str, str] = {
    ".html": "output",
    ".css":  "styles",
    ".js":   "script",
    ".ts":   "script",
    ".jsx":  "component",
    ".tsx":  "component",
    ".py":   "script",
    ".cs":   "Program",
    ".java": "Main",
    ".cpp":  "main",
    ".c":    "main",
    ".go":   "main",
    ".rs":   "main",
    ".sh":   "script",
    ".ps1":  "script",
    ".sql":  "query",
    ".json": "data",
    ".xml":  "data",
    ".yaml": "config",
    ".yml":  "config",
    ".toml": "config",
    ".md":   "README",
    ".r":    "script",
}

# Matches ```<lang>\n<content>``` — non-greedy so multiple blocks are found.
_CODE_BLOCK_RE = re.compile(r"```(\w+)\s*\n(.*?)```", re.DOTALL)


def _assemble_html(base_html: str, css_blocks: list[str], js_blocks: list[str]) -> str:
    """
    Stitch separate CSS and JS blocks into a base HTML document.

    Uses index-based insertion (case-insensitive tag search) so mixed-case
    HTML is handled correctly.

    CSS → <style> tags injected before </head>.
         Falls back to after <head>, then prepend.
    JS  → <script> tags injected before </body>.
         Falls back to append.
    """
    if css_blocks:
        style_tag = "\n".join(f"<style>\n{css}\n</style>" for css in css_blocks)
        lower = base_html.lower()
        if "</head>" in lower:
            i = lower.index("</head>")
            base_html = base_html[:i] + style_tag + "\n" + base_html[i:]
        elif "<head>" in lower:
            i = lower.index("<head>") + len("<head>")
            base_html = base_html[:i] + "\n" + style_tag + base_html[i:]
        else:
            base_html = style_tag + "\n" + base_html

    if js_blocks:
        script_tag = "\n".join(f"<script>\n{js}\n</script>" for js in js_blocks)
        lower = base_html.lower()
        if "</body>" in lower:
            i = lower.index("</body>")
            base_html = base_html[:i] + script_tag + "\n" + base_html[i:]
        else:
            base_html = base_html + "\n" + script_tag

    return base_html


class CodeFileHandler:
    """
    Scans an AI response for fenced code blocks with a recognised language tag
    and renders a Streamlit download button for each one.

    Only languages in _LANG_META get a button; unknown or untagged blocks
    (plain ```) are ignored.

    key_prefix must be unique per render site so Streamlit widget keys don't
    collide across the history loop and the live response panel.
    """

    @classmethod
    def render_downloads(
        cls,
        response_text: str,
        *,
        key_prefix: str = "live",
    ) -> bool:
        """
        Render download buttons for every downloadable code block found.
        Also renders an assembled single-file HTML download when html + css/js
        blocks appear together in the same response.
        Returns True if at least one button was rendered.
        """
        blocks = _CODE_BLOCK_RE.findall(response_text or "")
        rendered = False

        # Deduplicate: track (lang, content) pairs already shown this render.
        seen: set[tuple[str, str]] = set()

        # Accumulated for potential single-file assembly.
        html_blocks: list[str] = []
        css_blocks:  list[str] = []
        js_blocks:   list[str] = []

        for idx, (raw_lang, content) in enumerate(blocks):
            lang = raw_lang.lower().strip()
            meta = _LANG_META.get(lang)
            if meta is None:
                continue

            ext, label, mime = meta
            pair = (lang, content)
            if pair in seen:
                continue
            seen.add(pair)

            # Collect for assembly.
            if lang in ("html", "htm"):
                html_blocks.append(content.strip())
            elif lang == "css":
                css_blocks.append(content.strip())
            elif lang in ("javascript", "js"):
                js_blocks.append(content.strip())

            stem     = _DEFAULT_STEM.get(ext, "output")
            filename = f"{stem}{ext}"
            key      = f"code_dl_{key_prefix}_{lang}_{idx}"

            st.download_button(
                label     = f"⬇️ Download {label} file",
                data      = content.strip().encode("utf-8"),
                file_name = filename,
                mime      = mime,
                key       = key,
            )
            rendered = True

        # Assembled single-file HTML button — only when html + css or js are present.
        if html_blocks and (css_blocks or js_blocks):
            assembled = _assemble_html(html_blocks[0], css_blocks, js_blocks)
            st.download_button(
                label     = "⬇️ Download Assembled HTML (single file)",
                data      = assembled.encode("utf-8"),
                file_name = "assembled.html",
                mime      = "text/html",
                key       = f"code_dl_{key_prefix}_assembled",
            )
            rendered = True

        return rendered
