from __future__ import annotations

"""ZIM Dump — rag_v1/ingest/zim_dump.py

Extracts .zim file articles and images to a structured local directory tree.
Designed for large Wikipedia / Wikisource ZIM files (millions of articles).

Folder hierarchy
----------------
<out_dir> / <FIRST_LETTER> / <first3chars> / <article_slug> /
    <article_slug>_<YYYY-MM>.md
    image1.jpg
    image2.png
    ...

Example:
    I:\\llm_data\\wikipedia_maxi_2025-08\\
        A\\
            alb\\
                Albert_Einstein\\
                    Albert_Einstein_2025-08.md
                    Einstein_1921_by_F_Schmutzer.jpg

Two-level bucketing prevents Windows Explorer from choking on 7M+ items:
- Level 1: first letter (A-Z) or '#' for non-alpha
- Level 2: first 3 characters of the slug (lower), e.g. "alb"
- Level 3: full article slug

Resume-safe: skips any article folder that already exists on disk.

Usage
-----
    # Dump all configured ZIM jobs:
    python -m rag_v1.ingest.zim_dump

    # Dump a single ZIM file:
    python -m rag_v1.ingest.zim_dump --zim <path.zim> --out <output_dir>
"""

import argparse
import os
import re
import sys
import time
import unicodedata
import datetime

_DATE_RE = re.compile(r"^\d{4}-\d{2}")
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote

try:
    import html2text as _html2text  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "html2text is required: pip install html2text"
    ) from exc

try:
    import lxml.html  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError("lxml is required: pip install lxml") from exc

from libzim.reader import Archive  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Job configuration — edit here to add / change ZIM mappings
# ---------------------------------------------------------------------------

@dataclass
class ZimJob:
    zim_path: Path
    out_dir: Path


ZIM_JOBS: list[ZimJob] = [
    ZimJob(
        zim_path=Path(
            r"C:\Users\Alquin\Desktop\wiki_download"
            r"\wikisource_en_all_maxi_2026-02.zim"
        ),
        out_dir=Path(r"H:\llm_wiki_data"),
    ),
    # ZimJob(
    #     zim_path=Path(
    #         r"C:\Users\Alquin\AppData\Roaming\kiwix-desktop"
    #         r"\wikipedia_en_all_nopic_2025-12.zim"
    #     ),
    #     out_dir=Path(r"I:\llm_data\wikipedia_nopic_2025_12"),
    # ),
    # ZimJob(
    #     zim_path=Path(
    #         r"C:\Users\Alquin\AppData\Roaming\kiwix-desktop"
    #         r"\wikisource_en_all_maxi_2026-02.zim"
    #     ),
    #     out_dir=Path(r"I:\llm_data\wikisource_maxi_2026_02"),
    # ),
    # ZimJob(
    #     zim_path=Path(
    #         r"C:\Users\Alquin\AppData\Roaming\kiwix-desktop"
    #         r"\wikisource_en_all_nopic_2026-02.zim"
    #     ),
    #     out_dir=Path(r"I:\llm_data\wikisource_nopic_2026_02"),
    # ),
]


# ---------------------------------------------------------------------------
# Path / slug helpers
# ---------------------------------------------------------------------------

_INVALID_WIN = re.compile(r'[\\/:*?"<>|]')
_MULTI_UNDERSCORE = re.compile(r"_+")
_MAX_SLUG_LEN = 180  # slug length; long paths handled via \\?\ prefix (see _win_abs)


def slugify(title: str) -> str:
    """Return a Windows-safe filename slug for *title*.

    - Normalises Unicode (NFC)
    - Replaces spaces and Windows-illegal chars with '_'
    - Collapses repeated underscores
    - Strips leading/trailing dots and underscores
    - Falls back to '_empty' if nothing remains
    """
    s = unicodedata.normalize("NFC", title)
    s = s.replace(" ", "_")
    s = _INVALID_WIN.sub("_", s)
    s = _MULTI_UNDERSCORE.sub("_", s)
    s = s.strip("._")
    return s[:_MAX_SLUG_LEN] or "_empty"


def article_folder(out_dir: Path, title: str) -> Path:
    """Compute the two-level bucketed output folder for *title*.

    Structure: <out_dir> / <LETTER> / <first3lower> / <slug>
    """
    slug = slugify(title)
    first = slug[0].upper() if slug and slug[0].isalpha() else "#"
    second = slug[:3].lower()
    return out_dir / first / second / slug


# ---------------------------------------------------------------------------
# Windows long-path helpers
# ---------------------------------------------------------------------------
# Windows MAX_PATH is 260 chars.  With _MAX_SLUG_LEN=180, the output path
# (out_dir + \A\ + \alb\ + \slug\ + \slug_date.md) can reach ~413 chars,
# which silently causes OSError on write even though mkdir succeeds.
# The \\?\ extended-path prefix raises the Windows limit to ~32 767 chars.

def _win_abs(p: Path) -> str:
    """Return an absolute path string with the \\?\\ extended prefix on Windows."""
    s = os.path.abspath(str(p))
    return "\\\\?\\" + s if sys.platform == "win32" and not s.startswith("\\\\") else s


def _mkdir_safe(p: Path) -> None:
    """Create directory tree using extended-length paths on Windows."""
    os.makedirs(_win_abs(p), exist_ok=True)


def _exists_safe(p: Path) -> bool:
    """os.path.exists using extended-length paths on Windows."""
    return os.path.exists(_win_abs(p))


def _write_text_safe(p: Path, text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *p* using extended-length paths on Windows."""
    with open(_win_abs(p), "w", encoding=encoding) as fh:
        fh.write(text)


def _write_bytes_safe(p: Path, data: bytes) -> None:
    """Write *data* to *p* using extended-length paths on Windows."""
    with open(_win_abs(p), "wb") as fh:
        fh.write(data)


# ---------------------------------------------------------------------------
# ZIM archive helpers
# ---------------------------------------------------------------------------

# Metadata path candidates across old and new ZIM formats
_DATE_PATHS = ("M/Date", "Date")

# MIME types we save as image files
_IMAGE_MIME = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/svg+xml",
    "image/bmp",
}


def zim_date(archive: Archive) -> str:
    """Return 'YYYY-MM' from ZIM metadata, or 'unknown' if unavailable.

    Tries the modern get_metadata() API first (new-namespace ZIM files like
    wikipedia_en_all_maxi_2025-08.zim), then falls back to get_entry_by_path()
    for older ZIM formats.
    """
    # Modern API: archive.get_metadata("Date") returns bytes directly.
    try:
        raw = archive.get_metadata("Date")
        date_str = raw.decode("utf-8", errors="ignore").strip()
        if _DATE_RE.match(date_str):
            return date_str[:7]
    except Exception:
        pass

    # Legacy fallback: old-namespace ZIM files store metadata as entries.
    for candidate in _DATE_PATHS:
        try:
            raw = bytes(archive.get_entry_by_path(candidate).get_item().content)
            date_str = raw.decode("utf-8", errors="ignore").strip()
            if _DATE_RE.match(date_str):
                return date_str[:7]  # 'YYYY-MM' from 'YYYY-MM-DD'
        except Exception:
            continue
    return "unknown"


def iter_article_entries(archive: Archive) -> Iterator[object]:
    """Yield every non-redirect entry whose MIME type is HTML.

    Uses archive._get_entry_by_id — the private but universally used
    iteration API for libzim 3.x Python bindings.
    """
    # all_entry_count includes redirects; entry_count is user-facing only.
    # We iterate all_entry_count to cover every stored entry.
    count = getattr(archive, "all_entry_count", archive.entry_count)
    for i in range(count):
        try:
            entry = archive._get_entry_by_id(i)
        except Exception:
            continue
        if entry.is_redirect:
            continue
        try:
            mime = entry.get_item().mimetype
        except Exception:
            continue
        if mime.startswith("text/html"):
            yield entry


# ---------------------------------------------------------------------------
# HTML processing
# ---------------------------------------------------------------------------

# XPath selectors tried in order to find the main article body.
# Covers MediaWiki (Wikipedia / Wikisource) both old and new skins.
_CONTENT_XPATHS = (
    '//div[@id="mw-content-text"]',
    '//div[contains(@class,"mw-content-ltr")]',
    '//div[contains(@class,"mw-content-rtl")]',
    '//div[@id="bodyContent"]',
    '//article',
)


def _extract_main_content(html_bytes: bytes) -> bytes:
    """Return the serialised HTML of the main article body.

    Strips site chrome (nav menus, sidebars, footers) by selecting the
    MediaWiki content div.  Falls back to the original *html_bytes* if no
    recognised selector matches so non-MediaWiki ZIM files still work.
    """
    try:
        tree = lxml.html.fromstring(html_bytes)
        for xpath in _CONTENT_XPATHS:
            elems = tree.xpath(xpath)
            if elems:
                raw = lxml.html.tostring(elems[0])
                return raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    except Exception:
        pass
    return html_bytes


def _make_converter() -> _html2text.HTML2Text:
    h = _html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True   # images saved as separate files
    h.ignore_tables = False
    h.body_width = 0          # no forced line wrapping
    h.unicode_snob = True
    return h


_converter = _make_converter()


def html_to_markdown(html_bytes: bytes) -> str:
    """Convert article HTML bytes to a Markdown string."""
    try:
        html = html_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""
    try:
        return _converter.handle(html)
    except Exception:
        # Converter may have raised RecursionError on deeply-nested Wikipedia
        # HTML (infoboxes, nested tables).  Retry with a fresh instance so
        # state corruption on the module-level singleton doesn't cascade.
        try:
            return _make_converter().handle(html)
        except Exception:
            return ""


def extract_image_zim_paths(html_bytes: bytes, article_zim_path: str) -> list[str]:
    """Return ZIM entry paths for all images referenced in *html_bytes*.

    Resolves relative src values against *article_zim_path*, handles:
      - ../I/filename.jpg  (old ZIM namespace: article in A/, image in I/)
      - I/filename.jpg     (path-relative reference)
      - //upload.wikimedia.org/...  (skipped — external)
      - data:...           (skipped — inline)
    """
    try:
        html = html_bytes.decode("utf-8", errors="replace")
        tree = lxml.html.fromstring(html)
    except Exception:
        return []

    # Base directory for resolving relative paths (parent of article entry)
    parts = article_zim_path.rsplit("/", 1)
    base_dir = (parts[0] + "/") if len(parts) > 1 else ""

    paths: list[str] = []
    for img in tree.iter("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src:
            continue
        src = unquote(src)

        # Skip inline data URIs and external URLs
        if src.startswith("data:") or src.startswith("//") or "://" in src:
            continue

        if src.startswith("../"):
            resolved = src[3:]           # strip one directory level
        elif src.startswith("./"):
            resolved = base_dir + src[2:]
        elif src.startswith("/"):
            resolved = src[1:]           # strip leading slash
        else:
            resolved = base_dir + src

        if resolved:
            paths.append(resolved)

    return paths


# ---------------------------------------------------------------------------
# Single-article extraction
# ---------------------------------------------------------------------------

def extract_article(
    archive: Archive,
    entry: object,
    out_dir: Path,
    date_tag: str,
) -> bool:
    """Extract one HTML entry (text + images) to *out_dir*.

    Returns True when files were written, False when skipped or on error.
    """
    title: str = getattr(entry, "title", None) or entry.path.split("/")[-1]  # type: ignore[union-attr]
    folder = article_folder(out_dir, title)

    # Resume: if the folder already exists, skip entirely
    if _exists_safe(folder):
        return False

    try:
        item = entry.get_item()  # type: ignore[union-attr]
        html_bytes = bytes(item.content)
    except Exception:
        return False

    if not html_bytes:
        return False

    main_html = _extract_main_content(html_bytes)
    markdown = html_to_markdown(main_html)
    if not markdown.strip():
        return False

    _mkdir_safe(folder)

    slug = slugify(title)
    md_filename = f"{slug}_{date_tag}.md"
    _write_text_safe(folder / md_filename, markdown)

    # Save images referenced in the article HTML
    article_path: str = getattr(entry, "path", "")  # type: ignore[assignment]
    for img_zim_path in extract_image_zim_paths(html_bytes, article_path):
        try:
            img_entry = archive.get_entry_by_path(img_zim_path)
            # Follow redirect if the image entry is a redirect
            if img_entry.is_redirect:
                img_entry = img_entry.get_redirect_entry()
            img_item = img_entry.get_item()
            if img_item.mimetype not in _IMAGE_MIME:
                continue
            img_bytes = bytes(img_item.content)
            if not img_bytes:
                continue
            img_filename = Path(img_zim_path).name
            _write_bytes_safe(folder / img_filename, img_bytes)
        except Exception:
            pass  # missing / inaccessible image — skip silently

    return True


# ---------------------------------------------------------------------------
# Per-job driver
# ---------------------------------------------------------------------------

def dump_zim(job: ZimJob) -> None:
    print(f"\n{'=' * 60}")
    print(f"ZIM : {job.zim_path}")
    print(f"OUT : {job.out_dir}")

    if not job.zim_path.exists():
        print(f"  SKIP — ZIM file not found: {job.zim_path}")
        return

    job.out_dir.mkdir(parents=True, exist_ok=True)

    archive = Archive(job.zim_path)
    date_tag = zim_date(archive)
    total = getattr(archive, "all_entry_count", archive.entry_count)
    print(f"Total entries : {total:,}  |  Date tag : {date_tag}")
    print("Starting extraction — resume-safe (existing folders skipped).")

    written = skipped = errors = 0
    t0 = time.monotonic()
    error_log_path = job.out_dir / "_dump_errors.log"

    for i, entry in enumerate(iter_article_entries(archive)):
        try:
            ok = extract_article(archive, entry, job.out_dir, date_tag)
            if ok:
                written += 1
            else:
                skipped += 1
        except Exception as exc:
            errors += 1
            label = getattr(entry, "path", str(i))
            if errors <= 100:
                print(f"  ERROR [{label}]: {type(exc).__name__}: {exc}")
            with open(_win_abs(error_log_path), "a", encoding="utf-8") as ef:
                ef.write(f"{label}\t{type(exc).__name__}\t{exc}\n")

        if (i + 1) % 10_000 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / max(elapsed, 0.001)
            eta_s = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  {i + 1:>9,}  written={written:,}  skipped={skipped:,}"
                f"  errors={errors}  rate={rate:,.0f}/s  ETA={eta_s / 60:.1f}min"
            )

    elapsed = time.monotonic() - t0
    time_total = datetime.timedelta(seconds=elapsed)

    print(
        f"\nDone — written={written:,}  skipped={skipped:,}  errors={errors:,}"
        f"  elapsed=days: {time_total.days} hrs: {time_total.seconds // 3600} mins: {(time_total.seconds % 3600) // 60} secs: {time_total.seconds % 60}"   # {elapsed / 60:.1f}min
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump ZIM articles + images to a local directory tree."
    )
    parser.add_argument(
        "--zim",
        type=Path,
        metavar="PATH",
        help="Path to a single .zim file (overrides built-in ZIM_JOBS list)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="Output directory (required when --zim is used)",
    )
    args = parser.parse_args()

    if args.zim:
        if not args.out:
            parser.error("--out is required when --zim is used")
        jobs: list[ZimJob] = [ZimJob(zim_path=args.zim, out_dir=args.out)]
    else:
        jobs = ZIM_JOBS

    for job in jobs:
        dump_zim(job)


if __name__ == "__main__":
    main()
