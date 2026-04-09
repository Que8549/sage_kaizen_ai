"""
review_service/output/patch_writer.py — Write .patch files for suggested fixes.

Extracts code blocks from the synthesized report and writes each one as a
unified diff .patch file in reviews/patches/.

These are suggestions only — never applied automatically.
Output: reviews/patches/YYYY-MM-DD-HHMM-{slug}.patch
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path

from ..state import ReviewState

_PATCHES_DIR = Path("F:/Projects/sage_kaizen_ai/reviews/patches")

# Matches: ### Patch: <title> ... ```diff ... ``` blocks in synthesis
_PATCH_BLOCK_RE = re.compile(
    r"###\s+Patch:\s+(?P<title>[^\n]+)\n"
    r".*?"                                        # optional metadata lines
    r"```diff\n(?P<diff>.*?)```",
    re.DOTALL,
)

# Also matches standalone ```diff blocks with a preceding **File:** line
_STANDALONE_DIFF_RE = re.compile(
    r"\*\*File\*\*:\s*(?P<file>\S+).*?```diff\n(?P<diff>.*?)```",
    re.DOTALL,
)


def write_patch_files(state: ReviewState) -> list[str]:
    """
    Extract patch blocks from synthesis and write .patch files.
    Returns list of written file paths.
    """
    synthesis = state.get("synthesis", "")
    if not synthesis:
        return []

    _PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    now       = datetime.now()
    datestamp = now.strftime("%Y-%m-%d-%H%M")
    paths: list[str] = []

    # Extract named patch blocks
    for m in _PATCH_BLOCK_RE.finditer(synthesis):
        title   = m.group("title").strip()
        diff    = m.group("diff").strip()
        if not diff or "APPROXIMATE" in diff and len(diff) < 20:
            continue   # skip approximate placeholders with no real content
        slug    = _slugify(title)
        path    = _PATCHES_DIR / f"{datestamp}-{slug}.patch"
        header  = f"# Sage Kaizen Suggested Patch\n# Title: {title}\n# Date: {now.isoformat()}\n# NOTE: Suggestion only — not applied automatically.\n\n"
        path.write_text(header + diff + "\n", encoding="utf-8")
        paths.append(str(path))

    # Extract standalone diff blocks if no named patches found
    if not paths:
        for i, m in enumerate(_STANDALONE_DIFF_RE.finditer(synthesis), start=1):
            file_ref = m.group("file").strip()
            diff     = m.group("diff").strip()
            if not diff:
                continue
            slug  = _slugify(file_ref) or f"patch-{i}"
            path  = _PATCHES_DIR / f"{datestamp}-{slug}.patch"
            header = f"# Sage Kaizen Suggested Patch\n# File: {file_ref}\n# Date: {now.isoformat()}\n# NOTE: Suggestion only — not applied automatically.\n\n"
            path.write_text(header + diff + "\n", encoding="utf-8")
            paths.append(str(path))

    return paths


def _slugify(text: str) -> str:
    """Convert a title to a filename-safe slug."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:60]
