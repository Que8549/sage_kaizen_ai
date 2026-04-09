"""
review_service/nodes/scope_collector.py — Scope Collection Node

Collects all context needed for the review without calling any LLM.
Multi-project aware: main app, voice app, and SearXNG config.

Chunking strategy for full repo mode
-------------------------------------
Full repo mode cannot send complete diffs (potentially 500K+ chars vs 128K
ARCHITECT context). Strategy:

  Phase 1 — Inventory:
    git diff --stat (file names + change counts; no content)
    Arch docs + brains.yaml always included (constant ~18K chars)

  Phase 2 — Priority Scoring:
    1. Python files with >20 changed lines
    2. Python files with 5–20 changed lines
    3. Config files (*.yaml, *.json, *.toml)
    4. Documentation files (*.md)
    5. Other

  Phase 3 — Budget Fill (70,000 char budget for diff material):
    Each prioritized file: up to 3,000 chars of unified diff content.
    Architecture docs: up to 15,000 chars (always).
    brains.yaml: always (~3,000 chars).
    Remaining budget filled by priority-ranked files.

  Phase 4 — Overflow:
    Files not included → stored in state["overflow_files"]
    Synthesizer notes the count in the final report.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import git

from sk_logging import get_logger
from ..state import ReviewState

_LOG = get_logger("sage_kaizen.review_service.scope_collector")

# ── Project roots ──────────────────────────────────────────────────────────
MAIN_ROOT    = Path("F:/Projects/sage_kaizen_ai")
VOICE_ROOT   = Path("F:/Projects/sage_kaizen_ai_voice")
SEARXNG_ROOT = Path("F:/Projects/searxng")

# ── Arch docs included in every review ────────────────────────────────────
_ARCH_DOC_PATHS = [
    MAIN_ROOT / "docs" / "01-ARCHITECTURE.md",
    MAIN_ROOT / "docs" / "02-ARCH-PATTERNS.md",
    MAIN_ROOT / "CLAUDE.md",
]

# ── Context budget (chars) ────────────────────────────────────────────────
_TOTAL_BUDGET        = 70_000    # max chars for diff material
_PER_FILE_DIFF_LIMIT = 3_000     # max diff chars per individual file
_ARCH_DOCS_LIMIT     = 15_000    # max chars for architecture docs block
_BRAINS_YAML_LIMIT   = 5_000     # max chars for brains.yaml


async def scope_collector_node(state: ReviewState) -> dict:
    """
    Collect all review scope without invoking any LLM.
    Returns partial state dict to merge into ReviewState.
    """
    mode   = state["mode"]
    target = state.get("target", "")

    _LOG.info("review.scope.start | mode=%s target=%r", mode, target)

    # ── Architecture docs (always) ─────────────────────────────────────
    arch_docs = _read_arch_docs()

    # ── brains.yaml (always) ───────────────────────────────────────────
    brains_yaml = _read_file_capped(
        MAIN_ROOT / "config" / "brains" / "brains.yaml", _BRAINS_YAML_LIMIT
    )

    # ── Mode-dependent diff ────────────────────────────────────────────
    git_diff, changed_files, overflow_files = _collect_diff(mode, target)

    # ── TODO/FIXME scan across changed Python files ────────────────────
    todo_markers = _scan_todos(changed_files)

    # ── Multi-project file trees ───────────────────────────────────────
    file_tree = _collect_file_trees()

    # ── SearXNG config (static — no git) ──────────────────────────────
    searxng_config = _read_searxng_config()
    if searxng_config:
        git_diff = git_diff + "\n\n<searxng_config>\n" + searxng_config + "\n</searxng_config>"

    scope_char_count = len(git_diff) + len(arch_docs) + len(brains_yaml)
    _LOG.info(
        "review.scope.done | changed=%d overflow=%d scope_chars=%d",
        len(changed_files), len(overflow_files), scope_char_count,
    )

    return {
        "git_diff":        git_diff,
        "changed_files":   changed_files,
        "todo_markers":    todo_markers,
        "arch_docs":       arch_docs,
        "brains_yaml":     brains_yaml,
        "file_tree":       file_tree,
        "overflow_files":  overflow_files,
        "scope_char_count": scope_char_count,
    }


# ── Diff collection by mode ────────────────────────────────────────────────

def _collect_diff(mode: str, target: str) -> tuple[str, list[str], list[str]]:
    """Return (diff_text, changed_files, overflow_files)."""
    overflow: list[str] = []

    try:
        main_repo  = git.Repo(str(MAIN_ROOT))
        voice_repo = git.Repo(str(VOICE_ROOT))
    except git.InvalidGitRepositoryError as exc:
        _LOG.warning("Git repo not found: %s", exc)
        return "", [], []

    if mode == "staged":
        diff_text    = _safe_git(main_repo, "diff", "--staged")
        changed_files = _staged_files(main_repo)
        # Prepend voice staged diff if any
        voice_staged = _safe_git(voice_repo, "diff", "--staged")
        if voice_staged.strip():
            diff_text = diff_text + "\n\n# Voice app staged:\n" + voice_staged

    elif mode == "file":
        changed_files = [target] if target else []
        diff_text     = _safe_git(main_repo, "diff", "HEAD", "--", target) if target else ""
        # Also read the file content for full context
        full_path = MAIN_ROOT / target
        if full_path.exists():
            content = full_path.read_text(encoding="utf-8", errors="replace")
            diff_text += f"\n\n<file_content path=\"{target}\">\n{content[:8000]}\n</file_content>"

    elif mode == "regression":
        base = target or "HEAD~1"
        diff_text     = _safe_git(main_repo, "diff", f"{base}..HEAD")
        changed_files = _files_between_refs(main_repo, base, "HEAD")

    else:  # full
        diff_text, changed_files, overflow = _full_mode_diff(main_repo, voice_repo)

    return diff_text, changed_files, overflow


def _full_mode_diff(
    main_repo: "git.Repo", voice_repo: "git.Repo"
) -> tuple[str, list[str], list[str]]:
    """
    Full mode chunking strategy.
    Returns (budget-capped diff text, all changed files, overflow files).
    """
    # Phase 1: stat (no content; safe size)
    main_stat  = _safe_git(main_repo, "diff", "main", "HEAD", "--stat")
    voice_stat = _safe_git(voice_repo, "diff", "main", "HEAD", "--stat")

    # Collect all changed files across both repos
    main_files  = _changed_files_from_stat(main_stat,  prefix="")
    voice_files = _changed_files_from_stat(voice_stat, prefix="[voice] ")
    all_changed  = main_files + voice_files

    # Phase 2: priority score each file
    scored = sorted(all_changed, key=_priority_score, reverse=True)

    # Phase 3: fill budget
    budget_remaining = _TOTAL_BUDGET
    diff_parts: list[str] = [
        f"# Main App — git diff --stat\n{main_stat}\n",
        f"# Voice App — git diff --stat\n{voice_stat}\n",
    ]
    included: list[str] = []
    overflow: list[str] = []

    for file_path in scored:
        if budget_remaining <= 0:
            overflow.append(file_path)
            continue
        # Strip [voice] prefix for actual git diff
        clean_path = file_path.removeprefix("[voice] ")
        repo = voice_repo if file_path.startswith("[voice]") else main_repo
        file_diff = _safe_git(repo, "diff", "main", "HEAD", "--", clean_path)
        file_diff = file_diff[:_PER_FILE_DIFF_LIMIT]
        if file_diff.strip():
            diff_parts.append(f"\n## {file_path}\n{file_diff}")
            budget_remaining -= len(file_diff)
        included.append(file_path)

    return "\n".join(diff_parts), all_changed, overflow


def _priority_score(file_path: str) -> int:
    """Higher score = higher review priority."""
    p = file_path.lower().removeprefix("[voice] ")
    if p.endswith(".py"):
        return 4
    if p.endswith((".yaml", ".json", ".toml")):
        return 3
    if p.endswith(".md"):
        return 2
    return 1


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_git(repo: "git.Repo", *args: str) -> str:
    try:
        return repo.git.execute(["git"] + list(args))
    except Exception as exc:
        _LOG.warning("git %s failed: %s", " ".join(args), exc)
        return f"[git {' '.join(args)} failed: {exc}]"


def _staged_files(repo: "git.Repo") -> list[str]:
    try:
        return [item.a_path for item in repo.index.diff("HEAD")]
    except Exception:
        return []


def _files_between_refs(repo: "git.Repo", base: str, head: str) -> list[str]:
    try:
        return [item.a_path for item in repo.commit(base).diff(head)]
    except Exception:
        return []


def _changed_files_from_stat(stat_text: str, prefix: str = "") -> list[str]:
    """Parse filenames from `git diff --stat` output."""
    files = []
    for line in stat_text.splitlines():
        m = re.match(r"^\s+(\S+.*?)\s+\|\s+\d+", line)
        if m:
            files.append(prefix + m.group(1).strip())
    return files


def _read_arch_docs() -> str:
    parts = []
    for path in _ARCH_DOC_PATHS:
        if path.exists():
            content = path.read_text(encoding="utf-8", errors="replace")
            parts.append(f"# {path.name}\n{content[:_ARCH_DOCS_LIMIT // len(_ARCH_DOC_PATHS)]}")
    return "\n\n".join(parts)


def _read_file_capped(path: Path, max_chars: int) -> str:
    if not path.exists():
        return f"[{path} not found]"
    content = path.read_text(encoding="utf-8", errors="replace")
    return content[:max_chars]


def _scan_todos(changed_files: list[str]) -> str:
    """Grep TODO/FIXME/HACK/NOQA in changed Python files."""
    py_files = [
        str(MAIN_ROOT / f) for f in changed_files
        if f.endswith(".py") and (MAIN_ROOT / f).exists()
    ]
    if not py_files:
        return ""
    try:
        result = subprocess.run(
            ["rg", "--no-heading", "-n", "TODO|FIXME|HACK|NOQA"] + py_files[:20],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout[:3000]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _collect_file_trees() -> str:
    trees = []
    for root, label in [
        (MAIN_ROOT,    "Main App"),
        (VOICE_ROOT,   "Voice App"),
        (SEARXNG_ROOT, "SearXNG"),
    ]:
        if not root.exists():
            continue
        entries = [
            p.name for p in sorted(root.iterdir())
            if not p.name.startswith(".") and p.name not in ("__pycache__", ".venv")
        ]
        trees.append(f"### {label} ({root})\n" + "\n".join(f"  {e}" for e in entries[:40]))
    return "\n\n".join(trees)


def _read_searxng_config() -> str:
    compose = SEARXNG_ROOT / "docker-compose.yml"
    if not compose.exists():
        return ""
    return compose.read_text(encoding="utf-8", errors="replace")[:2000]
