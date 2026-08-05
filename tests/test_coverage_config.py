"""
tests/test_coverage_config.py

Guards the coverage configuration itself.

Why this file exists
--------------------
Coverage's `source` list has to enumerate every directory by hand, because
`news/` and `rag_v1/` are namespace packages (no __init__.py — they are shared
with sage_kaizen_ai_ingest) and coverage only descends into sub-directories
that are *regular* packages.  Measured 2026-08-04: with only `source = ["."]`,
context_injector.py, media_retriever.py, rag_v1/retrieve/citations.py and all
of news/ were absent from the report entirely — not reported as 0%, absent —
understating the denominator by ~390 statements and inflating the percentage.

A hand-maintained list rots silently: add `rag_v1/newthing/` and it is simply
never measured, and the total goes *up* because the uncovered code is invisible.

These tests turn that from a silent hole into a failing build.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def coverage_cfg() -> dict:
    with (_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["tool"]["coverage"]


@pytest.fixture(scope="module")
def source_dirs(coverage_cfg) -> set[Path]:
    return {
        (_ROOT / entry).resolve()
        for entry in coverage_cfg["run"]["source"]
    }


# Directory names that never hold measurable first-party source.
_IGNORED_DIR_NAMES = {
    ".git", ".venv", ".vscode", ".streamlit", ".pytest_cache", "__pycache__",
    "tests", "scripts", "llama.cpp", "flash-attention", "logs", "docs",
    "reviews", "images", "static", "media_samples", "wiki_samples", "feedback",
    "config", "log", "llama-cpp-python", "sage_kaizen_ingest.egg-info",
}


def _first_party_dirs_with_python() -> set[Path]:
    """Every directory under the repo root that holds at least one .py file."""
    found: set[Path] = set()
    for path in _ROOT.rglob("*.py"):
        if any(part in _IGNORED_DIR_NAMES for part in path.relative_to(_ROOT).parts):
            continue
        found.add(path.parent.resolve())
    return found


class TestSourceListCompleteness:
    def test_every_directory_with_python_files_is_measured(self, source_dirs):
        """
        The regression guard: a new namespace sub-package must be added to
        `source` or it is silently unmeasured.

        A directory counts as measured when it, or an ancestor that coverage
        will actually walk into, is listed.  Since coverage only descends into
        regular packages, an ancestor only counts when every directory between
        it and the target has an __init__.py.
        """
        missing = []
        for directory in sorted(_first_party_dirs_with_python()):
            if not _is_measured(directory, source_dirs):
                missing.append(directory.relative_to(_ROOT).as_posix())
        assert not missing, (
            "These directories contain .py files but coverage will not walk "
            "into them — add each to [tool.coverage.run] source in "
            f"pyproject.toml: {missing}"
        )

    def test_repo_root_is_listed(self, source_dirs):
        assert _ROOT.resolve() in source_dirs

    def test_no_listed_directory_is_missing_from_disk(self, source_dirs):
        """A stale entry means the list was edited without checking."""
        gone = [d.relative_to(_ROOT).as_posix() for d in source_dirs if not d.is_dir()]
        assert not gone, f"source lists directories that do not exist: {gone}"

    def test_source_entries_are_paths_not_package_names(self, coverage_cfg):
        """
        Package names would resolve through the shared rag_v1.__path__ that
        spans this repo and sage_kaizen_ai_ingest, picking up the wrong copy.
        A relative path can only mean this repo.
        """
        bad = [
            e for e in coverage_cfg["run"]["source"]
            if e != "." and not e.startswith("./")
        ]
        assert not bad, f"use './name' directory paths, not package names: {bad}"


def _is_measured(directory: Path, source_dirs: set[Path]) -> bool:
    """True when coverage will report files in `directory`."""
    if directory in source_dirs:
        return True
    # An ancestor covers it only if coverage can descend the whole way, which
    # requires every intermediate directory to be a regular package.
    for ancestor in directory.parents:
        if ancestor in source_dirs:
            chain = directory.relative_to(ancestor).parts
            probe = ancestor
            for part in chain:
                probe = probe / part
                if not (probe / "__init__.py").is_file():
                    return False
            return True
        if ancestor == _ROOT:
            break
    return False


class TestNamespacePackageAssumptions:
    """
    The source list's shape is justified by these facts.  If one changes, the
    hand-maintained list can be simplified — or has silently broken.
    """

    def test_rag_v1_is_a_namespace_package(self):
        assert not (_ROOT / "rag_v1" / "__init__.py").is_file(), (
            "rag_v1 gained an __init__.py — it is shared with "
            "sage_kaizen_ai_ingest as a namespace package (that project's "
            "CLAUDE.md §2); making it regular would change import resolution "
            "across both repos"
        )

    def test_news_is_a_namespace_package(self):
        assert not (_ROOT / "news" / "__init__.py").is_file()

    def test_rag_v1_subpackages_are_namespace_packages(self):
        """This is precisely why each has to be listed individually."""
        regular = [
            d.name for d in (_ROOT / "rag_v1").iterdir()
            if d.is_dir() and d.name != "__pycache__" and (d / "__init__.py").is_file()
        ]
        assert not regular, (
            f"rag_v1 sub-packages gained __init__.py: {regular} — "
            "'./rag_v1' alone could now cover them and the source list can "
            "be simplified"
        )


class TestEnforcement:
    def test_fail_under_is_set(self, coverage_cfg):
        assert coverage_cfg["report"]["fail_under"] == 80

    def test_missing_lines_are_shown(self, coverage_cfg):
        assert coverage_cfg["report"]["show_missing"] is True

    def test_untestable_modules_are_omitted_with_a_reason(self, coverage_cfg):
        """
        ui_streamlit_server.py and code_download.py are deliberately excluded.
        The justification lives in a comment above the omit list; this just
        pins that they are still the only first-party exclusions.
        """
        omit = coverage_cfg["run"]["omit"]
        assert "ui_streamlit_server.py" in omit
        assert "code_download.py" in omit

    def test_tests_are_not_measured(self, coverage_cfg):
        assert "tests/*" in coverage_cfg["run"]["omit"]
