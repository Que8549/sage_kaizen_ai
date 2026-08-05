"""
tests/test_review_service.py

Unit tests for review_service/ — the LangGraph-based codebase review pipeline.

The LLM nodes are factory functions taking a ChatOpenAI, which makes them
straightforward to exercise with a stub LLM: each node's real contract is
"build a context block, call the model once, never let an exception escape".

Output writers are redirected at their module-level Path constants so nothing
is written into the real reviews/ or docs/03-DECISIONS/ trees.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from review_service.state import ReviewState, default_state
from review_service.trigger import (
    REVIEW_TRIGGER_RE,
    ReviewCommand,
    is_review_command,
    parse_review_command,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stub_llm(content: str = "FINDINGS", *, fail: bool = False) -> MagicMock:
    """A ChatOpenAI stand-in whose ainvoke() returns (or raises) as directed."""
    llm = MagicMock()

    async def _ainvoke(messages):
        if fail:
            raise RuntimeError("ARCHITECT unreachable")
        return MagicMock(content=content)

    llm.ainvoke = _ainvoke
    return llm


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# trigger.py
# ---------------------------------------------------------------------------

class TestIsReviewCommand:
    @pytest.mark.parametrize(
        "text",
        [
            "review your codebase",
            "Review the codebase",
            "REVIEW YOUR CODEBASE",
            "codebase review",
            "architect review",
            "code review",
            "run code review",
            "review staged",
            "review staged changes",
            "review the staged changes",
            "review the file chat_service.py",
            "review the module memory",
            "regression audit",
            "review mode",
        ],
    )
    def test_recognised_phrases(self, text):
        assert is_review_command(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "what is a code smell?",
            "can you review my resume",
            "tell me about the codebase",
            "",
            "   ",
            "reviewing the architecture of transformers",
        ],
    )
    def test_non_triggers(self, text):
        assert is_review_command(text) is False

    def test_matches_when_embedded_in_a_sentence(self):
        assert is_review_command("Hey Sage, please review your codebase now") is True

    def test_leading_and_trailing_whitespace_tolerated(self):
        assert is_review_command("\n  architect review  \n") is True


class TestParseReviewCommand:
    def test_default_is_full_mode(self):
        assert parse_review_command("review your codebase") == ReviewCommand(mode="full")

    def test_staged_mode(self):
        assert parse_review_command("review staged changes").mode == "staged"

    def test_file_mode_extracts_target(self):
        cmd = parse_review_command("review the file chat_service.py")
        assert cmd.mode == "file"
        assert cmd.target == "chat_service.py"

    def test_module_keyword_also_yields_file_mode(self):
        cmd = parse_review_command("review the module router.py")
        assert cmd.mode == "file"
        assert cmd.target == "router.py"

    def test_file_mode_without_the_article(self):
        assert parse_review_command("review file memory/db.py").target == "memory/db.py"

    def test_regression_mode_default_base(self):
        cmd = parse_review_command("regression audit")
        assert cmd.mode == "regression"
        assert cmd.target == "HEAD~1"

    def test_regression_mode_extracts_ref(self):
        cmd = parse_review_command("regression audit after HEAD~5")
        assert cmd.mode == "regression"
        # NOTE: lowercased. parse_review_command() matches against text.lower(),
        # so the captured ref loses its case — "HEAD~5" becomes "head~5" — while
        # the no-ref default a few lines below is the hardcoded uppercase
        # "HEAD~1". Git ref names ARE case-sensitive; "head~5" only resolves
        # because Windows' filesystem is case-insensitive, and the same command
        # would fail on Linux. Pinned here rather than fixed: it is outside the
        # bugs-1-7 scope agreed for this pass. Fix = capture from the original
        # text, not the lowercased copy.
        assert cmd.target == "head~5"

    def test_regression_ref_case_is_inconsistent_with_the_default(self):
        """Documents the asymmetry above; delete both once it is fixed."""
        assert parse_review_command("regression audit").target == "HEAD~1"
        assert parse_review_command("regression audit after HEAD~2").target == "head~2"

    def test_file_mode_takes_precedence_over_staged(self):
        cmd = parse_review_command("review the file staged_thing.py")
        assert cmd.mode == "file"

    def test_is_case_insensitive(self):
        assert parse_review_command("REVIEW STAGED").mode == "staged"


class TestTriggerRegex:
    def test_is_compiled_case_insensitive(self):
        assert REVIEW_TRIGGER_RE.search("ARCHITECT REVIEW") is not None

    def test_does_not_match_partial_words(self):
        assert REVIEW_TRIGGER_RE.search("previewing") is None


# ---------------------------------------------------------------------------
# state.py
# ---------------------------------------------------------------------------

class TestDefaultState:
    def test_carries_mode_and_target(self):
        s = default_state("file", "a.py")
        assert s["mode"] == "file" and s["target"] == "a.py"

    def test_target_defaults_to_empty(self):
        assert default_state("full")["target"] == ""

    def test_all_string_fields_start_empty(self):
        s = default_state("full")
        for key in (
            "git_diff", "todo_markers", "arch_docs", "brains_yaml", "file_tree",
            "pyright_output", "ruff_output", "pytest_collect", "vulture_output",
            "ruff_quality_output", "web_research", "architect_findings",
            "flags_findings", "docs_findings", "code_quality_findings", "synthesis",
        ):
            assert s[key] == "", key  # type: ignore[literal-required]

    def test_all_list_fields_start_empty(self):
        s = default_state("full")
        for key in ("changed_files", "overflow_files", "output_paths"):
            assert s[key] == [], key  # type: ignore[literal-required]

    def test_not_approved_by_default(self):
        assert default_state("full")["approved"] is False

    def test_no_error_by_default(self):
        assert default_state("full")["error"] is None

    def test_scope_char_count_starts_at_zero(self):
        assert default_state("full")["scope_char_count"] == 0

    def test_lists_are_not_shared_between_instances(self):
        a, b = default_state("full"), default_state("full")
        a["changed_files"].append("x")
        assert b["changed_files"] == []


# ---------------------------------------------------------------------------
# LLM nodes — flags_sanity / docs_drift / architect_reviewer /
#             code_quality_reviewer / synthesizer
# ---------------------------------------------------------------------------

from review_service.nodes.architect_reviewer import make_architect_reviewer_node
from review_service.nodes.code_quality_reviewer import make_code_quality_reviewer_node
from review_service.nodes.docs_drift import make_docs_drift_node
from review_service.nodes.flags_sanity import make_flags_sanity_node
from review_service.nodes.synthesizer import make_synthesizer_node

def _active_state() -> ReviewState:
    """
    A state with enough populated for every LLM node to actually call the model.

    code_quality_reviewer deliberately short-circuits on an empty state (full
    mode only, and only when vulture/ruff-quality produced output), so the
    shared contract tests have to supply those inputs.
    """
    s = default_state("full")
    s["vulture_output"] = "x.py:10: unused function 'foo' (90% confidence)"
    s["ruff_quality_output"] = "x.py:1:1: SIM108 use ternary"
    s["git_diff"] = "diff --git a/x.py b/x.py"
    s["brains_yaml"] = "fast:\n  port: 8011"
    s["arch_docs"] = "# CLAUDE.md"
    s["architect_findings"] = "prior findings"
    return s


_LLM_NODES = [
    (make_flags_sanity_node,          "flags_findings"),
    (make_docs_drift_node,            "docs_findings"),
    (make_architect_reviewer_node,    "architect_findings"),
    (make_code_quality_reviewer_node, "code_quality_findings"),
    (make_synthesizer_node,           "synthesis"),
]


@pytest.mark.parametrize("factory,out_key", _LLM_NODES, ids=lambda v: getattr(v, "__name__", v))
class TestLlmNodesCommonContract:
    def test_returns_only_its_own_key(self, factory, out_key):
        node = factory(stub_llm("RESULT"))
        out = run(node(_active_state()))
        assert set(out) == {out_key}

    def test_writes_the_model_response(self, factory, out_key):
        node = factory(stub_llm("RESULT"))
        assert run(node(_active_state()))[out_key] == "RESULT"

    def test_llm_failure_is_captured_not_raised(self, factory, out_key):
        node = factory(stub_llm(fail=True))
        out = run(node(_active_state()))
        assert "ERROR" in out[out_key]
        assert "ARCHITECT unreachable" in out[out_key]

    def test_empty_state_does_not_crash(self, factory, out_key):
        """An all-defaults state must be survivable, even if the node skips."""
        node = factory(stub_llm("ok"))
        out = run(node(default_state("full")))
        assert out_key in out

    def test_sends_a_system_and_a_human_message(self, factory, out_key):
        captured = {}

        llm = MagicMock()

        async def _ainvoke(messages):
            captured["messages"] = messages
            return MagicMock(content="x")

        llm.ainvoke = _ainvoke
        run(factory(llm)(_active_state()))
        roles = [type(m).__name__ for m in captured["messages"]]
        assert roles[0] == "SystemMessage"
        assert roles[-1] == "HumanMessage"


class TestCodeQualityReviewerSkips:
    """This node is full-mode-only and input-gated; the skips are intentional."""

    @pytest.mark.parametrize("mode", ["staged", "file", "regression"])
    def test_skipped_outside_full_mode(self, mode):
        s = _active_state()
        s["mode"] = mode
        node = make_code_quality_reviewer_node(stub_llm("SHOULD NOT BE CALLED"))
        assert run(node(s))["code_quality_findings"] == ""

    def test_skipped_when_no_tool_output(self):
        s = default_state("full")
        node = make_code_quality_reviewer_node(stub_llm("SHOULD NOT BE CALLED"))
        assert run(node(s))["code_quality_findings"] == ""

    def test_runs_with_only_vulture_output(self):
        s = default_state("full")
        s["vulture_output"] = "x.py:1: unused import"
        node = make_code_quality_reviewer_node(stub_llm("FOUND"))
        assert run(node(s))["code_quality_findings"] == "FOUND"

    def test_runs_with_only_ruff_quality_output(self):
        s = default_state("full")
        s["ruff_quality_output"] = "x.py:1:1: B008"
        node = make_code_quality_reviewer_node(stub_llm("FOUND"))
        assert run(node(s))["code_quality_findings"] == "FOUND"

    def test_context_notes_absent_tools_explicitly(self):
        from review_service.nodes.code_quality_reviewer import _build_context
        ctx = _build_context(default_state("full"))
        assert "vulture not installed" in ctx
        assert "no extended ruff findings" in ctx


class TestFlagsSanityContext:
    def test_includes_brains_yaml(self):
        from review_service.nodes.flags_sanity import _build_context
        s = default_state("full")
        s["brains_yaml"] = "fast:\n  port: 8011"
        ctx = _build_context(s)
        assert "<brains_yaml>" in ctx and "port: 8011" in ctx

    def test_includes_prior_architect_findings(self):
        from review_service.nodes.flags_sanity import _build_context
        s = default_state("full")
        s["architect_findings"] = "GPU risk noted"
        assert "prior_findings_summary" in _build_context(s)

    def test_prior_findings_are_capped(self):
        from review_service.nodes.flags_sanity import _build_context
        s = default_state("full")
        s["architect_findings"] = "x" * 10_000
        assert len(_build_context(s)) < 5_000

    def test_empty_state_yields_empty_context(self):
        from review_service.nodes.flags_sanity import _build_context
        assert _build_context(default_state("full")) == ""


class TestDocsDriftContext:
    def test_includes_arch_docs(self):
        from review_service.nodes.docs_drift import _build_context
        s = default_state("full")
        s["arch_docs"] = "# CLAUDE.md"
        assert "<arch_docs>" in _build_context(s)

    def test_includes_changed_files(self):
        from review_service.nodes.docs_drift import _build_context
        s = default_state("full")
        s["changed_files"] = ["a.py", "b.py"]
        ctx = _build_context(s)
        assert "a.py" in ctx and "b.py" in ctx

    def test_diff_is_condensed(self):
        from review_service.nodes.docs_drift import _build_context
        s = default_state("full")
        s["git_diff"] = "d" * 50_000
        assert len(_build_context(s)) < 20_000


# ---------------------------------------------------------------------------
# scope_collector — pure helpers
# ---------------------------------------------------------------------------

from review_service.nodes import scope_collector as sc


class TestPriorityScore:
    @pytest.mark.parametrize(
        "path,score",
        [
            ("chat_service.py", 4),
            ("config/brains.yaml", 3),
            ("data.json", 3),
            ("pyproject.toml", 3),
            ("README.md", 2),
            ("image.png", 1),
        ],
    )
    def test_scores_by_extension(self, path, score):
        assert sc._priority_score(path) == score

    def test_voice_prefix_is_stripped_before_scoring(self):
        assert sc._priority_score("[voice] main.py") == 4

    def test_python_outranks_everything_else(self):
        files = ["README.md", "a.py", "conf.yaml", "img.png"]
        assert sorted(files, key=sc._priority_score, reverse=True)[0] == "a.py"


class TestChangedFilesFromStat:
    def test_parses_stat_output(self):
        stat = (
            " chat_service.py | 12 ++++----\n"
            " router.py       |  3 +-\n"
            " 2 files changed, 15 insertions(+)\n"
        )
        assert sc._changed_files_from_stat(stat) == ["chat_service.py", "router.py"]

    def test_applies_prefix(self):
        stat = " main.py | 1 +\n"
        assert sc._changed_files_from_stat(stat, prefix="[voice] ") == ["[voice] main.py"]

    def test_ignores_the_summary_line(self):
        stat = " 3 files changed, 10 insertions(+), 2 deletions(-)\n"
        assert sc._changed_files_from_stat(stat) == []

    def test_empty_input(self):
        assert sc._changed_files_from_stat("") == []

    def test_handles_paths_with_directories(self):
        stat = " rag_v1/runtime/context_injector.py | 40 +++++\n"
        assert sc._changed_files_from_stat(stat) == ["rag_v1/runtime/context_injector.py"]


class TestSafeGit:
    def test_returns_command_output(self):
        repo = MagicMock()
        repo.git.execute.return_value = "diff output"
        assert sc._safe_git(repo, "diff") == "diff output"

    def test_failure_becomes_an_inline_marker(self):
        repo = MagicMock()
        repo.git.execute.side_effect = RuntimeError("not a repo")
        out = sc._safe_git(repo, "diff", "--staged")
        assert out.startswith("[git diff --staged failed:")


class TestReadFileCapped:
    def test_missing_file_marker(self, tmp_path):
        out = sc._read_file_capped(tmp_path / "nope.yaml", 100)
        assert "not found" in out

    def test_caps_content(self, tmp_path):
        p = tmp_path / "big.yaml"
        p.write_text("y" * 500, encoding="utf-8")
        assert len(sc._read_file_capped(p, 100)) == 100

    def test_reads_short_file_whole(self, tmp_path):
        p = tmp_path / "s.yaml"
        p.write_text("hello", encoding="utf-8")
        assert sc._read_file_capped(p, 100) == "hello"


class TestScanTodos:
    def test_no_python_files_returns_empty(self):
        assert sc._scan_todos(["README.md"]) == ""

    def test_empty_list_returns_empty(self):
        assert sc._scan_todos([]) == ""

    def test_missing_ripgrep_is_not_fatal(self):
        with (
            patch.object(sc.Path, "exists", return_value=True),
            patch.object(sc.subprocess, "run", side_effect=FileNotFoundError),
        ):
            assert sc._scan_todos(["a.py"]) == ""

    def test_timeout_is_not_fatal(self):
        import subprocess as sp
        with (
            patch.object(sc.Path, "exists", return_value=True),
            patch.object(sc.subprocess, "run",
                         side_effect=sp.TimeoutExpired(cmd="rg", timeout=10)),
        ):
            assert sc._scan_todos(["a.py"]) == ""

    def test_output_is_capped(self):
        with (
            patch.object(sc.Path, "exists", return_value=True),
            patch.object(sc.subprocess, "run",
                         return_value=MagicMock(stdout="x" * 10_000)),
        ):
            assert len(sc._scan_todos(["a.py"])) == 3_000


class TestCollectDiff:
    def test_invalid_repo_degrades_to_empty(self):
        import git as gitmod
        with patch.object(sc.git, "Repo", side_effect=gitmod.InvalidGitRepositoryError("x")):
            assert sc._collect_diff("full", "") == ("", [], [])

    def test_staged_mode_uses_staged_diff(self):
        repo = MagicMock()
        repo.git.execute.return_value = ""
        repo.index.diff.return_value = [MagicMock(a_path="a.py")]
        with patch.object(sc.git, "Repo", return_value=repo):
            _, changed, _ = sc._collect_diff("staged", "")
        assert changed == ["a.py"]

    def test_file_mode_targets_one_path(self, tmp_path):
        repo = MagicMock()
        repo.git.execute.return_value = "the diff"
        with patch.object(sc.git, "Repo", return_value=repo):
            diff, changed, overflow = sc._collect_diff("file", "router.py")
        assert changed == ["router.py"]
        assert overflow == []

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "known defect: _collect_diff('file', '') crashes. "
            "MAIN_ROOT / '' is MAIN_ROOT itself — a directory — so the "
            "`if full_path.exists()` guard passes and read_text() raises "
            "PermissionError on the directory. The `if target else ''` guard "
            "one line above protects the git call but not this branch. "
            "Reachable via `parse_review_command('review the file')` with no "
            "path, which yields mode='file', target=''. Outside the bugs-1-7 "
            "scope for this pass; fix = guard on `target` before touching the "
            "filesystem, or check is_file() rather than exists()."
        ),
    )
    def test_file_mode_with_no_target(self):
        repo = MagicMock()
        repo.git.execute.return_value = ""
        with patch.object(sc.git, "Repo", return_value=repo):
            diff, changed, _ = sc._collect_diff("file", "")
        assert changed == [] and diff == ""

    def test_file_mode_with_no_target_currently_raises(self):
        """Pins the defect above so a fix must update both tests together."""
        repo = MagicMock()
        repo.git.execute.return_value = ""
        with patch.object(sc.git, "Repo", return_value=repo):
            with pytest.raises((PermissionError, IsADirectoryError, OSError)):
                sc._collect_diff("file", "")

    def test_regression_mode_defaults_to_head_prev(self):
        repo = MagicMock()
        repo.git.execute.return_value = ""
        repo.commit.return_value.diff.return_value = []
        with patch.object(sc.git, "Repo", return_value=repo):
            sc._collect_diff("regression", "")
        assert any("HEAD~1..HEAD" in str(c) for c in repo.git.execute.call_args_list)

    def test_regression_mode_uses_supplied_base(self):
        repo = MagicMock()
        repo.git.execute.return_value = ""
        repo.commit.return_value.diff.return_value = []
        with patch.object(sc.git, "Repo", return_value=repo):
            sc._collect_diff("regression", "HEAD~7")
        assert any("HEAD~7..HEAD" in str(c) for c in repo.git.execute.call_args_list)


class TestFullModeDiff:
    def test_budget_overflow_is_recorded(self):
        stat = "\n".join(f" file{i}.py | 100 +++" for i in range(60))
        repo = MagicMock()
        # Every per-file diff is large enough to exhaust the budget quickly.
        repo.git.execute.side_effect = lambda args: (
            stat if "--stat" in args else "d" * sc._PER_FILE_DIFF_LIMIT
        )
        _, all_changed, overflow = sc._full_mode_diff(repo, repo)
        assert len(all_changed) == 120     # 60 main + 60 voice
        assert overflow, "budget cap never triggered"

    def test_per_file_diff_is_capped(self):
        repo = MagicMock()
        repo.git.execute.side_effect = lambda args: (
            " a.py | 5 +\n" if "--stat" in args else "x" * 99_999
        )
        diff, _, _ = sc._full_mode_diff(repo, repo)
        # Two files (main + voice), each capped
        assert len(diff) < 2 * sc._PER_FILE_DIFF_LIMIT + 1_000

    def test_stat_headers_are_always_present(self):
        repo = MagicMock()
        repo.git.execute.return_value = ""
        diff, _, _ = sc._full_mode_diff(repo, repo)
        assert "# Main App — git diff --stat" in diff
        assert "# Voice App — git diff --stat" in diff


# ---------------------------------------------------------------------------
# subprocess_checks
# ---------------------------------------------------------------------------

from review_service.nodes import subprocess_checks as spc


class TestTrim:
    def test_short_text_untouched(self):
        assert spc._trim("hello", 100) == "hello"

    def test_strips_whitespace(self):
        assert spc._trim("  hi  ", 100) == "hi"

    def test_long_text_is_truncated_with_a_notice(self):
        out = spc._trim("x" * 500, 100)
        assert out.startswith("x" * 100)
        assert "400 chars truncated" in out


class TestSubprocessRun:
    def test_missing_tool_is_reported_inline(self):
        with patch.object(spc.asyncio, "create_subprocess_exec", side_effect=FileNotFoundError):
            out = run(spc._run("pyright", ["pyright"], 5))
        assert "not installed" in out

    def test_generic_error_is_reported_inline(self):
        with patch.object(spc.asyncio, "create_subprocess_exec", side_effect=OSError("denied")):
            out = run(spc._run("ruff", ["ruff"], 5))
        assert out.startswith("[ruff: error")

    def test_successful_run_returns_decoded_output(self):
        proc = MagicMock()

        async def _communicate():
            return (b"tool output", None)

        proc.communicate = _communicate

        async def _create(*a, **kw):
            return proc

        with patch.object(spc.asyncio, "create_subprocess_exec", _create):
            assert run(spc._run("ruff", ["ruff"], 5)) == "tool output"

    def test_timeout_kills_the_process(self):
        proc = MagicMock()

        async def _communicate():
            await asyncio.sleep(10)

        proc.communicate = _communicate

        async def _create(*a, **kw):
            return proc

        with patch.object(spc.asyncio, "create_subprocess_exec", _create):
            out = run(spc._run("pytest", ["pytest"], 0.01))
        assert "timed out" in out
        proc.kill.assert_called_once()


class TestSubprocessChecksNode:
    def test_returns_all_five_keys(self):
        async def _fake_run(tool, cmd, timeout):
            return f"{tool} ok"

        with patch.object(spc, "_run", _fake_run):
            out = run(spc.subprocess_checks_node(default_state("staged")))
        assert set(out) == {
            "pyright_output", "ruff_output", "pytest_collect",
            "vulture_output", "ruff_quality_output",
        }

    def test_staged_mode_skips_the_whole_tree_scans(self):
        async def _fake_run(tool, cmd, timeout):
            return f"{tool} ok"

        with patch.object(spc, "_run", _fake_run):
            out = run(spc.subprocess_checks_node(default_state("staged")))
        assert out["vulture_output"] == ""
        assert out["ruff_quality_output"] == ""

    def test_full_mode_runs_the_whole_tree_scans(self):
        called: list[str] = []

        async def _fake_run(tool, cmd, timeout):
            called.append(tool)
            return f"{tool} ok"

        with patch.object(spc, "_run", _fake_run):
            out = run(spc.subprocess_checks_node(default_state("full")))
        assert "vulture" in called
        assert "ruff-quality" in called
        assert out["vulture_output"] == "vulture ok"

    def test_voice_prefixed_files_are_excluded(self):
        captured: list[list[str]] = []

        async def _fake_run(tool, cmd, timeout):
            captured.append(cmd)
            return ""

        s = default_state("staged")
        s["changed_files"] = ["[voice] main.py", "router.py"]
        with patch.object(spc, "_run", _fake_run):
            run(spc.subprocess_checks_node(s))
        assert not any("[voice]" in " ".join(c) for c in captured)


# ---------------------------------------------------------------------------
# human_gate
# ---------------------------------------------------------------------------

class TestHumanGate:
    def test_returns_approved_true(self):
        from review_service.nodes import human_gate as hg
        with patch.object(hg, "interrupt", return_value=True):
            assert hg.human_gate_node(default_state("full")) == {"approved": True}

    def test_returns_approved_false(self):
        from review_service.nodes import human_gate as hg
        with patch.object(hg, "interrupt", return_value=False):
            assert hg.human_gate_node(default_state("full")) == {"approved": False}

    def test_coerces_truthy_resume_values(self):
        from review_service.nodes import human_gate as hg
        with patch.object(hg, "interrupt", return_value="yes"):
            assert hg.human_gate_node(default_state("full"))["approved"] is True

    def test_interrupt_payload_carries_the_synthesis(self):
        from review_service.nodes import human_gate as hg
        s = default_state("full")
        s["synthesis"] = "the findings"
        with patch.object(hg, "interrupt", return_value=True) as intr:
            hg.human_gate_node(s)
        payload = intr.call_args.args[0]
        assert payload["synthesis"] == "the findings"
        assert "prompt" in payload

    def test_interrupt_is_not_swallowed(self):
        """INVARIANT: the interrupt exception must reach LangGraph's runtime."""
        from review_service.nodes import human_gate as hg

        class _Interrupt(Exception):
            pass

        with patch.object(hg, "interrupt", side_effect=_Interrupt):
            with pytest.raises(_Interrupt):
                hg.human_gate_node(default_state("full"))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

from review_service.output import adr_writer, patch_writer, review_writer


@pytest.fixture
def out_dirs(tmp_path, monkeypatch):
    """Redirect all three writers away from the real repo directories."""
    reviews = tmp_path / "reviews"
    patches = tmp_path / "reviews" / "patches"
    adrs = tmp_path / "docs" / "03-DECISIONS"
    monkeypatch.setattr(review_writer, "_REVIEWS_DIR", reviews)
    monkeypatch.setattr(patch_writer, "_PATCHES_DIR", patches)
    monkeypatch.setattr(adr_writer, "_ADR_DIR", adrs)
    return {"reviews": reviews, "patches": patches, "adrs": adrs}


class TestReviewWriter:
    def test_writes_a_file_and_returns_its_path(self, out_dirs):
        s = default_state("full")
        s["synthesis"] = "## Findings\nAll good."
        path = Path(review_writer.write_review_file(s))
        assert path.exists()
        assert "All good." in path.read_text(encoding="utf-8")

    def test_filename_encodes_mode(self, out_dirs):
        path = Path(review_writer.write_review_file(default_state("staged")))
        assert path.name.endswith("-staged-review.md")

    def test_header_reports_counts(self, out_dirs):
        s = default_state("full")
        s["changed_files"] = ["a.py", "b.py", "c.py"]
        s["scope_char_count"] = 12_345
        text = Path(review_writer.write_review_file(s)).read_text(encoding="utf-8")
        assert "| **Changed files** | 3 |" in text
        assert "12,345" in text

    def test_missing_synthesis_gets_a_placeholder(self, out_dirs):
        text = Path(review_writer.write_review_file(default_state("full"))).read_text(
            encoding="utf-8"
        )
        assert "_No synthesis generated._" in text

    def test_overflow_note_only_when_present(self, out_dirs):
        s = default_state("full")
        assert "Overflow files" not in Path(
            review_writer.write_review_file(s)
        ).read_text(encoding="utf-8")
        s["overflow_files"] = ["x.py"]
        assert "Overflow files" in Path(
            review_writer.write_review_file(s)
        ).read_text(encoding="utf-8")

    def test_states_that_nothing_was_applied_automatically(self, out_dirs):
        text = Path(review_writer.write_review_file(default_state("full"))).read_text(
            encoding="utf-8"
        )
        assert "No changes were applied automatically" in text


class TestAdrWriter:
    def test_no_adr_without_risk_markers(self, out_dirs):
        assert adr_writer.write_adr_if_needed(default_state("full")) is None

    def test_no_adr_for_severity_without_architecture_keyword(self, out_dirs):
        s = default_state("full")
        s["architect_findings"] = "[CRITICAL] a typo in a log message"
        assert adr_writer.write_adr_if_needed(s) is None

    def test_no_adr_for_architecture_keyword_without_severity(self, out_dirs):
        s = default_state("full")
        s["architect_findings"] = "[LOW] some tight coupling here"
        assert adr_writer.write_adr_if_needed(s) is None

    def test_writes_adr_when_both_signals_present(self, out_dirs):
        s = default_state("full")
        s["architect_findings"] = "[CRITICAL] tight coupling across module boundary"
        path = adr_writer.write_adr_if_needed(s)
        assert path is not None and Path(path).exists()

    def test_high_severity_also_triggers(self, out_dirs):
        s = default_state("full")
        s["synthesis"] = "[HIGH] architectural layering violation"
        assert adr_writer.write_adr_if_needed(s) is not None

    def test_adr_lists_changed_files(self, out_dirs):
        s = default_state("full")
        s["architect_findings"] = "[HIGH] architectural risk"
        s["changed_files"] = ["chat_service.py"]
        text = Path(adr_writer.write_adr_if_needed(s)).read_text(encoding="utf-8")  # type: ignore[arg-type]
        assert "- chat_service.py" in text

    def test_adr_status_is_proposed(self, out_dirs):
        s = default_state("full")
        s["architect_findings"] = "[HIGH] architectural risk"
        text = Path(adr_writer.write_adr_if_needed(s)).read_text(encoding="utf-8")  # type: ignore[arg-type]
        assert "## Status\nProposed" in text

    def test_extract_section_finds_named_block(self):
        text = "## architecture_risks\nrisk one\nrisk two\n\n## other\nzzz"
        assert "risk one" in adr_writer._extract_section(text, "architecture_risks")
        assert "zzz" not in adr_writer._extract_section(text, "architecture_risks")

    def test_extract_section_missing_returns_empty(self):
        assert adr_writer._extract_section("no sections here", "architecture_risks") == ""


class TestPatchWriter:
    def test_no_synthesis_writes_nothing(self, out_dirs):
        assert patch_writer.write_patch_files(default_state("full")) == []

    def test_named_patch_block_is_written(self, out_dirs):
        s = default_state("full")
        s["synthesis"] = (
            "### Patch: Fix the health check\n"
            "some prose\n"
            "```diff\n--- a/x.py\n+++ b/x.py\n-old\n+new\n```\n"
        )
        paths = patch_writer.write_patch_files(s)
        assert len(paths) == 1
        text = Path(paths[0]).read_text(encoding="utf-8")
        assert "+new" in text
        assert "Suggestion only" in text

    def test_filename_is_slugified_from_the_title(self, out_dirs):
        s = default_state("full")
        s["synthesis"] = "### Patch: Fix The Health Check!\n```diff\n+x\n```"
        assert "fix-the-health-check" in Path(patch_writer.write_patch_files(s)[0]).name

    def test_standalone_diff_used_only_when_no_named_patches(self, out_dirs):
        s = default_state("full")
        s["synthesis"] = "**File**: router.py\n```diff\n+added\n```"
        paths = patch_writer.write_patch_files(s)
        assert len(paths) == 1
        assert "router" in Path(paths[0]).name

    def test_named_patches_win_over_standalone(self, out_dirs):
        s = default_state("full")
        s["synthesis"] = (
            "### Patch: Real One\n```diff\n+a\n```\n"
            "**File**: other.py\n```diff\n+b\n```"
        )
        paths = patch_writer.write_patch_files(s)
        assert len(paths) == 1
        assert "real-one" in Path(paths[0]).name

    def test_multiple_named_patches(self, out_dirs):
        s = default_state("full")
        s["synthesis"] = (
            "### Patch: First\n```diff\n+aaaaaaaaaaaaaaaaaaaaaa\n```\n"
            "### Patch: Second\n```diff\n+bbbbbbbbbbbbbbbbbbbbbb\n```\n"
        )
        assert len(patch_writer.write_patch_files(s)) == 2

    def test_empty_diff_is_skipped(self, out_dirs):
        s = default_state("full")
        s["synthesis"] = "### Patch: Nothing\n```diff\n```"
        assert patch_writer.write_patch_files(s) == []

    @pytest.mark.parametrize(
        "title,slug",
        [
            ("Simple Title", "simple-title"),
            ("With  Multiple   Spaces", "with-multiple-spaces"),
            ("Special!@#$Chars", "specialchars"),
            ("under_scores-and-dashes", "under-scores-and-dashes"),
        ],
    )
    def test_slugify(self, title, slug):
        assert patch_writer._slugify(title) == slug

    def test_slugify_caps_length(self):
        assert len(patch_writer._slugify("word " * 100)) <= 60

    def test_slugify_strips_accents(self):
        assert patch_writer._slugify("Café Naïve") == "cafe-naive"


# ---------------------------------------------------------------------------
# output_writer node
# ---------------------------------------------------------------------------

from review_service.nodes import output_writer as ow


class TestOutputWriterNode:
    def test_unapproved_writes_nothing(self):
        with (
            patch.object(ow, "write_review_file") as wr,
            patch.object(ow, "write_adr_if_needed") as wa,
            patch.object(ow, "write_patch_files") as wp,
        ):
            out = run(ow.output_writer_node(default_state("full")))
        assert out == {"output_paths": []}
        wr.assert_not_called()
        wa.assert_not_called()
        wp.assert_not_called()

    def test_approved_writes_the_review(self):
        s = default_state("full")
        s["approved"] = True
        with (
            patch.object(ow, "write_review_file", return_value="/r.md"),
            patch.object(ow, "write_adr_if_needed", return_value=None),
            patch.object(ow, "write_patch_files", return_value=[]),
        ):
            assert run(ow.output_writer_node(s))["output_paths"] == ["/r.md"]

    def test_adr_path_included_when_written(self):
        s = default_state("full")
        s["approved"] = True
        with (
            patch.object(ow, "write_review_file", return_value="/r.md"),
            patch.object(ow, "write_adr_if_needed", return_value="/adr.md"),
            patch.object(ow, "write_patch_files", return_value=[]),
        ):
            assert run(ow.output_writer_node(s))["output_paths"] == ["/r.md", "/adr.md"]

    def test_patch_paths_appended(self):
        s = default_state("full")
        s["approved"] = True
        with (
            patch.object(ow, "write_review_file", return_value="/r.md"),
            patch.object(ow, "write_adr_if_needed", return_value=None),
            patch.object(ow, "write_patch_files", return_value=["/p1.patch", "/p2.patch"]),
        ):
            assert run(ow.output_writer_node(s))["output_paths"] == [
                "/r.md", "/p1.patch", "/p2.patch",
            ]
