"""
tests/test_evals_bench_and_scorers.py

Unit tests for evals/bench.py, evals/scorers.py and evals/golden.py — Layers 2
and 3 of the model evaluation harness.

The scorer tests are the important ones: they encode this system's real failure
modes (Chinese code-switching, <think> leakage, ungrounded RAG answers) as
executable checks rather than prose in a document.
"""
from __future__ import annotations

import json

import pytest

from evals.bench import (
    BenchRow,
    BenchRun,
    build_llama_bench_argv,
    compare_runs,
    load_run,
    parse_llama_bench_json,
    save_run,
)
from evals.golden import (
    SLICE_TARGETS,
    GoldenItem,
    GoldenSet,
    classify_slice,
    extract_json_payload,
    load_golden_set,
    parse_route_json,
    save_golden_set,
    stratified_sample,
)
from evals.scorers import (
    ScoreReport,
    cjk_ratio,
    compare_reports,
    contains_cjk,
    count_citations,
    has_think_leakage,
    looks_like_refusal,
    score_response,
    score_responses,
)


# ---------------------------------------------------------------------------
# bench — parsing
# ---------------------------------------------------------------------------

def bench_record(**over) -> dict:
    base = {
        "build_commit": "fdc3db9b6", "build_number": 9598,
        "model_filename": "E:/m.gguf", "model_type": "qwen35 27B Q6_K",
        "model_size": 22873411584, "gpu_info": "RTX 5090",
        "flash_attn": 1, "type_k": "q8_0", "type_v": "q8_0",
        "n_ubatch": 512, "devices": "CUDA0",
        "n_prompt": 512, "n_gen": 0, "n_depth": 0,
        "avg_ts": 2846.27, "stddev_ts": 199.31,
        "test_time": "2026-08-06T15:49:10Z",
    }
    base.update(over)
    return base


class TestParseLlamaBenchJson:
    def test_parses_a_clean_payload(self):
        run = parse_llama_bench_json(json.dumps([bench_record()]), label="base")
        assert run.label == "base"
        assert run.build_number == 9598
        assert len(run.rows) == 1

    def test_tolerates_interleaved_progress_lines(self):
        """
        llama-bench writes progress to the same stream as its JSON when both are
        captured together — the real capture this harness was built against had
        'llama-bench: benchmark 1/4' lines inside the array.
        """
        raw = (
            "ggml_cuda_init: found 3 CUDA devices\n"
            "  Device 0: NVIDIA GeForce RTX 5090\n"
            "llama-bench: benchmark 1/4: starting\n"
            + json.dumps([bench_record()]) +
            "\nllama-bench: benchmark 2/4: starting\n"
        )
        assert len(parse_llama_bench_json(raw).rows) == 1

    def test_names_prefill_and_decode_rows(self):
        raw = json.dumps([
            bench_record(n_prompt=512, n_gen=0),
            bench_record(n_prompt=0, n_gen=128, avg_ts=53.96),
        ])
        run = parse_llama_bench_json(raw)
        assert {r.test for r in run.rows} == {"pp512", "tg128"}

    def test_classifies_prefill_vs_decode(self):
        run = parse_llama_bench_json(json.dumps([
            bench_record(n_gen=0), bench_record(n_prompt=0, n_gen=128),
        ]))
        assert len(run.prefill_rows()) == 1
        assert len(run.decode_rows()) == 1
        assert run.prefill_rows()[0].kind == "prefill"
        assert run.decode_rows()[0].kind == "decode"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="no JSON array"):
            parse_llama_bench_json("llama-bench: everything went wrong")

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError, match="malformed"):
            parse_llama_bench_json('[{"build_commit": }]')

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="no measurements"):
            parse_llama_bench_json("[]")

    def test_model_size_converted_to_gib(self):
        run = parse_llama_bench_json(json.dumps([bench_record()]))
        assert run.model_size_gib == pytest.approx(21.30, abs=0.02)

    def test_row_lookup_by_test_and_depth(self):
        run = parse_llama_bench_json(json.dumps([
            bench_record(n_prompt=0, n_gen=128, n_depth=0, avg_ts=53.96),
            bench_record(n_prompt=0, n_gen=128, n_depth=8192, avg_ts=53.41),
        ]))
        assert run.row("tg128", 8192).avg_ts == 53.41
        assert run.row("tg128", 999) is None


class TestBenchRunPersistence:
    def test_round_trip(self, tmp_path):
        run = parse_llama_bench_json(json.dumps([bench_record()]), label="base")
        loaded = load_run(save_run(run, tmp_path / "r.run.json"))
        assert loaded.rows == run.rows
        assert loaded.label == run.label

    def test_load_run_accepts_raw_capture(self, tmp_path):
        p = tmp_path / "raw.json"
        p.write_text("llama-bench: starting\n" + json.dumps([bench_record()]),
                     encoding="utf-8")
        assert len(load_run(p).rows) == 1

    def test_save_creates_parent_directories(self, tmp_path):
        run = parse_llama_bench_json(json.dumps([bench_record()]))
        out = save_run(run, tmp_path / "deep" / "nested" / "r.json")
        assert out.is_file()

    def test_render_includes_config(self):
        run = parse_llama_bench_json(json.dumps([bench_record()]), label="base")
        rendered = run.render()
        assert "fdc3db9b6" in rendered and "q8_0" in rendered and "ubatch=512" in rendered


# ---------------------------------------------------------------------------
# bench — comparison
# ---------------------------------------------------------------------------

def make_run(label: str, rows: list[tuple[str, int, float]]) -> BenchRun:
    return BenchRun(
        label=label, model_filename="m.gguf",
        rows=[
            BenchRow(test=t, n_prompt=0, n_gen=128, n_depth=d, avg_ts=ts, stddev_ts=0.0)
            for t, d, ts in rows
        ],
    )


class TestCompareRuns:
    def test_identical_runs_pass(self):
        a = make_run("base", [("tg128", 0, 50.0)])
        b = make_run("cand", [("tg128", 0, 50.0)])
        assert compare_runs(a, b).passed

    def test_improvement_passes(self):
        a = make_run("base", [("tg128", 0, 50.0)])
        b = make_run("cand", [("tg128", 0, 60.0)])
        cmp = compare_runs(a, b)
        assert cmp.passed
        assert cmp.deltas[0].pct_change == pytest.approx(20.0)

    def test_small_regression_within_tolerance_passes(self):
        a = make_run("base", [("tg128", 0, 50.0)])
        b = make_run("cand", [("tg128", 0, 47.5)])       # -5%
        assert compare_runs(a, b, tolerance_pct=10.0).passed

    def test_large_regression_fails(self):
        a = make_run("base", [("tg128", 0, 50.0)])
        b = make_run("cand", [("tg128", 0, 40.0)])       # -20%
        cmp = compare_runs(a, b, tolerance_pct=10.0)
        assert not cmp.passed and len(cmp.regressions) == 1

    def test_tolerance_is_configurable(self):
        a = make_run("base", [("tg128", 0, 50.0)])
        b = make_run("cand", [("tg128", 0, 40.0)])
        assert compare_runs(a, b, tolerance_pct=25.0).passed

    def test_depth_is_part_of_the_key(self):
        a = make_run("base", [("tg128", 0, 50.0), ("tg128", 8192, 40.0)])
        b = make_run("cand", [("tg128", 0, 50.0), ("tg128", 8192, 40.0)])
        assert len(compare_runs(a, b).deltas) == 2

    def test_unmatched_measurements_are_reported_not_dropped(self):
        """A candidate benchmarked with a different matrix is not comparable."""
        a = make_run("base", [("tg128", 0, 50.0)])
        b = make_run("cand", [("tg256", 0, 50.0)])
        cmp = compare_runs(a, b)
        assert cmp.deltas == []
        assert set(cmp.missing) == {"tg128@0", "tg256@0"}

    def test_zero_baseline_does_not_divide_by_zero(self):
        a = make_run("base", [("tg128", 0, 0.0)])
        b = make_run("cand", [("tg128", 0, 10.0)])
        assert compare_runs(a, b).deltas[0].pct_change == 0.0

    def test_render_flags_regressions(self):
        a = make_run("base", [("tg128", 0, 50.0)])
        b = make_run("cand", [("tg128", 0, 30.0)])
        assert "REGRESSION" in compare_runs(a, b).render()


class TestBuildLlamaBenchArgv:
    def test_mirrors_production_flags(self):
        argv = build_llama_bench_argv("llama-bench.exe", "m.gguf", device="CUDA0")
        assert "-fa" in argv and argv[argv.index("-fa") + 1] == "on"
        assert argv[argv.index("-ctk") + 1] == "q8_0"
        assert argv[argv.index("-dev") + 1] == "CUDA0"

    def test_emits_json(self):
        argv = build_llama_bench_argv("b.exe", "m.gguf", device="CUDA0")
        assert argv[argv.index("-o") + 1] == "json"

    def test_depths_are_comma_joined(self):
        argv = build_llama_bench_argv("b.exe", "m.gguf", device="CUDA0",
                                      depths=(0, 4096, 8192))
        assert argv[argv.index("-d") + 1] == "0,4096,8192"

    def test_flash_attn_can_be_disabled(self):
        argv = build_llama_bench_argv("b.exe", "m.gguf", device="CUDA0", flash_attn=False)
        assert argv[argv.index("-fa") + 1] == "off"

    def test_every_element_is_a_string(self):
        argv = build_llama_bench_argv("b.exe", "m.gguf", device="CUDA0")
        assert all(isinstance(a, str) for a in argv)


# ---------------------------------------------------------------------------
# scorers — CJK (the incumbent's known failure mode)
# ---------------------------------------------------------------------------

class TestCjk:
    def test_pure_english_is_zero(self):
        assert cjk_ratio("The lighthouse keeper watched the storm.") == 0.0

    def test_pure_chinese_is_one(self):
        assert cjk_ratio("这是一个测试") == 1.0

    def test_mixed_is_fractional(self):
        assert 0.0 < cjk_ratio("hello 世界") < 1.0

    def test_whitespace_excluded_from_denominator(self):
        assert cjk_ratio("   \n\t  ") == 0.0

    def test_empty_string(self):
        assert cjk_ratio("") == 0.0

    def test_detects_kana_too(self):
        assert cjk_ratio("ひらがな") > 0

    def test_flag_tolerates_a_single_quoted_ideograph(self):
        """A question *about* Chinese may legitimately quote one character."""
        text = "The character 好 means good. " + "x" * 300
        assert contains_cjk(text) is False

    def test_flag_fires_on_a_code_switched_paragraph(self):
        text = "The keeper watched, 然后他看着海浪拍打着下面的岩石，数着每一个波浪的到来。"
        assert contains_cjk(text) is True

    def test_threshold_is_configurable(self):
        text = "hello " + "世" * 2
        assert contains_cjk(text, threshold=0.9) is False
        assert contains_cjk(text, threshold=0.01) is True


class TestThinkLeakage:
    def test_clean_content(self):
        assert has_think_leakage("The answer is 42.") is False

    def test_opening_tag(self):
        assert has_think_leakage("<think>hmm") is True

    def test_closing_tag_alone(self):
        assert has_think_leakage("stuff</think>answer") is True


class TestCitations:
    def test_numeric_markers(self):
        assert count_citations("Snails have shells [1]. They grow [2].") == 2

    def test_named_markers(self):
        assert count_citations("Per [Reuters], the summit ended.") == 1

    def test_none_present(self):
        assert count_citations("No citations at all here.") == 0

    def test_ignores_single_character_brackets(self):
        """Avoids matching array indexing in code answers like a[i]."""
        assert count_citations("value = a[i]") == 0


class TestRefusal:
    @pytest.mark.parametrize("text", [
        "I'm sorry, but I can't help with that.",
        "I cannot assist with this request.",
        "As an AI language model, I can't provide that.",
        "I must decline.",
    ])
    def test_detects_refusals(self, text):
        assert looks_like_refusal(text) is True

    @pytest.mark.parametrize("text", [
        "Here is how the refusal mechanism works in the router.",
        "I'm sorry to hear that — here is the fix.",
        "The answer is 42.",
    ])
    def test_does_not_flag_ordinary_text(self, text):
        assert looks_like_refusal(text) is False

    def test_only_inspects_the_opening(self):
        assert looks_like_refusal("Here you go. " * 60 + "I cannot help") is False


# ---------------------------------------------------------------------------
# scorers — scorecards and reports
# ---------------------------------------------------------------------------

class TestScoreResponse:
    def test_clean_response_has_no_hard_failures(self):
        card = score_response("A perfectly ordinary English answer.")
        assert card.clean and card.hard_failures == []

    def test_cjk_is_a_hard_failure(self):
        card = score_response("然后他看着海浪拍打着下面的岩石，数着每一个波浪。")
        assert not card.clean
        assert any("cjk" in f for f in card.hard_failures)

    def test_think_leak_is_a_hard_failure(self):
        assert "think_leak" in score_response("<think>x</think>answer").hard_failures

    def test_missing_citations_only_when_expected(self):
        assert score_response("no markers", citations_expected=False).clean
        assert not score_response("no markers", citations_expected=True).clean

    def test_citations_present_when_expected_is_clean(self):
        assert score_response("Grounded [1].", citations_expected=True).clean

    def test_refusal_is_recorded_but_not_a_hard_failure(self):
        """Refusal is a rate to compare, not an absolute veto on one response."""
        card = score_response("I'm sorry, but I can't help with that.")
        assert card.refusal is True
        assert card.clean is True

    def test_carries_identity_fields(self):
        card = score_response("x", prompt_id="g-1", slice_name="creative")
        assert card.prompt_id == "g-1" and card.slice_name == "creative"


class TestScoreReport:
    def _report(self):
        return score_responses([
            {"text": "Clean English answer.", "slice": "creative"},
            {"text": "然后他看着海浪拍打着下面的岩石。", "slice": "creative"},
            {"text": "<think>x</think>y", "slice": "code"},
            {"text": "no markers", "slice": "rag_grounded", "citations_expected": True},
        ], label="cand")

    def test_counts_each_failure_kind(self):
        r = self._report()
        assert len(r.cjk_failures) == 1
        assert len(r.think_leaks) == 1
        assert len(r.missing_citations) == 1

    def test_clean_rate(self):
        assert self._report().clean_rate == pytest.approx(0.25)

    def test_groups_by_slice(self):
        assert set(self._report().by_slice()) == {"creative", "code", "rag_grounded"}

    def test_empty_report_does_not_divide_by_zero(self):
        r = ScoreReport(label="empty")
        assert r.clean_rate == 0.0 and r.refusal_rate == 0.0
        assert r.mean_chars == 0.0 and r.stdev_chars == 0.0

    def test_single_card_stdev_is_zero(self):
        assert score_responses([{"text": "one"}]).stdev_chars == 0.0

    def test_to_dict_is_json_serialisable(self):
        json.dumps(self._report().to_dict())

    def test_render_mentions_the_failure_counts(self):
        assert "CJK failures" in self._report().render()


class TestCompareReports:
    def _report(self, texts, label, **kw):
        return score_responses([{"text": t, **kw} for t in texts], label=label)

    def test_identical_reports_pass(self):
        base = self._report(["clean answer here"], "base")
        cand = self._report(["clean answer here"], "cand")
        assert compare_reports(base, cand).passed

    def test_new_cjk_failure_regresses(self):
        base = self._report(["clean answer here"], "base")
        cand = self._report(["然后他看着海浪拍打着下面的岩石。"], "cand")
        cmp = compare_reports(base, cand)
        assert not cmp.passed
        assert any("CJK" in r for r in cmp.regressions)

    def test_fixing_a_failure_is_an_improvement(self):
        base = self._report(["然后他看着海浪拍打着下面的岩石。"], "base")
        cand = self._report(["clean answer here"], "cand")
        cmp = compare_reports(base, cand)
        assert cmp.passed
        assert any("CJK" in i for i in cmp.improvements)

    def test_new_think_leak_regresses(self):
        base = self._report(["clean"], "base")
        cand = self._report(["<think>x</think>y"], "cand")
        assert not compare_reports(base, cand).passed

    def test_higher_refusal_rate_regresses(self):
        base = self._report(["fine"] * 10, "base")
        cand = self._report(["I'm sorry, but I can't help with that."] * 10, "cand")
        cmp = compare_reports(base, cand)
        assert not cmp.passed
        assert any("refusal" in r for r in cmp.regressions)

    def test_length_collapse_regresses(self):
        base = self._report(["a fairly typical length answer here " * 3,
                             "another answer of roughly similar length " * 3], "base")
        cand = self._report(["tiny", "tiny"], "cand")
        cmp = compare_reports(base, cand)
        assert any("length" in r for r in cmp.regressions)

    def test_zero_stdev_baseline_skips_length_check(self):
        base = self._report(["same"], "base")
        cand = self._report(["a much much much longer response than the baseline"], "cand")
        cmp = compare_reports(base, cand)
        assert any("stdev" in n for n in cmp.notes)

    def test_differing_counts_are_noted(self):
        base = self._report(["a", "b"], "base")
        cand = self._report(["a"], "cand")
        assert any("counts differ" in n for n in compare_reports(base, cand).notes)


# ---------------------------------------------------------------------------
# golden — log mining
# ---------------------------------------------------------------------------

class TestExtractJsonPayload:
    def test_parses_a_route_json_line(self):
        line = 'route_json {"route": "fast", "score": 0, "modality": "text"}'
        assert extract_json_payload(line)["route"] == "fast"

    def test_parses_a_context_injection_line(self):
        line = 'context_injection_json {"brain": "FAST", "rag_chunks": 4}'
        assert extract_json_payload(line)["rag_chunks"] == 4

    def test_returns_none_for_unstructured_lines(self):
        assert extract_json_payload("route | brain=FAST | score=0") is None

    def test_returns_none_for_malformed_json(self):
        assert extract_json_payload("route_json {not json}") is None

    def test_returns_none_for_empty(self):
        assert extract_json_payload("") is None

    def test_returns_none_for_a_json_array(self):
        """Log lines are not a trusted format; only objects are accepted."""
        assert extract_json_payload("route_json [1, 2]") is None

    def test_parse_route_json_rejects_other_records(self):
        assert parse_route_json('context_injection_json {"a": 1}') is None
        assert parse_route_json('route_json {"a": 1}') == {"a": 1}


class TestClassifySlice:
    def test_multimodal_wins_over_topic(self):
        assert classify_slice("write a story", route={"modality": "image"}) == "multimodal"

    def test_rag_grounded_from_injection_record(self):
        assert classify_slice("anything", injection={"rag_chunks": 4}) == "rag_grounded"

    def test_wiki_chars_also_count_as_grounded(self):
        assert classify_slice("anything", injection={"wiki_chars": 900}) == "rag_grounded"

    def test_search_grounded(self):
        assert classify_slice("latest news", route={"search_used": True}) == "search_grounded"

    def test_creative_uses_the_routers_own_hints(self):
        assert classify_slice("write a story about a snail") == "creative"

    def test_code(self):
        assert classify_slice("debug this python traceback") == "code"

    def test_tutoring(self):
        assert classify_slice("teach me long division") == "tutoring"

    def test_philosophy(self):
        assert classify_slice("a question about ethics") == "philosophy"

    def test_knowledge(self):
        assert classify_slice("the history of rome") == "knowledge"

    def test_short_turns_are_voice_shaped(self):
        assert classify_slice("what time is it", route={"input_chars": 15}) == "voice_short"

    def test_fallback_is_general(self):
        long_prompt = "x" * 400
        assert classify_slice(long_prompt, route={"input_chars": 400}) == "general"


class TestStratifiedSample:
    def _items(self, slice_name: str, n: int) -> list[GoldenItem]:
        return [
            GoldenItem(prompt_id=f"{slice_name}-{i}", slice_name=slice_name,
                       prompt=f"p{i}", brain="FAST")
            for i in range(n)
        ]

    def test_caps_each_slice_at_its_target(self):
        items = self._items("creative", 100)
        out = stratified_sample(items, {"creative": 20})
        assert len(out) == 20

    def test_under_filled_slice_returned_whole(self):
        """A thin slice should be visible, not padded from elsewhere."""
        out = stratified_sample(self._items("creative", 3), {"creative": 20})
        assert len(out) == 3

    def test_is_deterministic_for_a_given_seed(self):
        items = self._items("creative", 100)
        a = stratified_sample(items, {"creative": 10}, seed=7)
        b = stratified_sample(items, {"creative": 10}, seed=7)
        assert [i.prompt_id for i in a] == [i.prompt_id for i in b]

    def test_different_seeds_differ(self):
        items = self._items("creative", 100)
        a = stratified_sample(items, {"creative": 10}, seed=1)
        b = stratified_sample(items, {"creative": 10}, seed=2)
        assert [i.prompt_id for i in a] != [i.prompt_id for i in b]

    def test_ignores_slices_not_in_targets(self):
        items = self._items("creative", 5) + self._items("code", 5)
        assert len(stratified_sample(items, {"creative": 5})) == 5

    def test_default_targets_cover_the_documented_slices(self):
        assert "creative" in SLICE_TARGETS and "rag_grounded" in SLICE_TARGETS
        assert sum(SLICE_TARGETS.values()) == 150


class TestGoldenSetPersistence:
    def _set(self) -> GoldenSet:
        return GoldenSet(
            name="v1",
            note="mined from log.sage_kaizen",
            created="2026-08-06",
            items=[
                GoldenItem(prompt_id="g-1", slice_name="creative",
                           prompt="write a story", brain="ARCHITECT"),
                GoldenItem(prompt_id="g-2", slice_name="rag_grounded",
                           prompt="what is a snail", brain="FAST",
                           citations_expected=True, rag_chunks=4),
            ],
        )

    def test_round_trip(self, tmp_path):
        loaded = load_golden_set(save_golden_set(self._set(), tmp_path / "g.jsonl"))
        assert loaded.name == "v1"
        assert loaded.note == "mined from log.sage_kaizen"
        assert len(loaded) == 2
        assert loaded.items[1].citations_expected is True

    def test_first_line_is_a_header(self, tmp_path):
        p = save_golden_set(self._set(), tmp_path / "g.jsonl")
        header = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
        assert header["_header"] is True
        assert header["count"] == 2

    def test_one_item_per_line_for_diffability(self, tmp_path):
        p = save_golden_set(self._set(), tmp_path / "g.jsonl")
        assert len([l for l in p.read_text(encoding="utf-8").splitlines() if l]) == 3

    def test_composition_counts_slices(self):
        assert self._set().composition() == {"creative": 1, "rag_grounded": 1}

    def test_render_shows_targets(self):
        assert "target" in self._set().render()

    def test_unknown_fields_are_ignored_on_load(self, tmp_path):
        """Forward compatibility: a newer set must not break an older reader."""
        p = tmp_path / "g.jsonl"
        p.write_text(json.dumps({
            "prompt_id": "g-1", "slice_name": "creative", "prompt": "x",
            "brain": "FAST", "some_future_field": 123,
        }) + "\n", encoding="utf-8")
        assert load_golden_set(p).items[0].prompt_id == "g-1"

    def test_blank_lines_tolerated(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text("\n" + json.dumps({
            "prompt_id": "g-1", "slice_name": "creative", "prompt": "x", "brain": "FAST",
        }) + "\n\n", encoding="utf-8")
        assert len(load_golden_set(p)) == 1


class TestGoldenSetNotYetFrozen:
    def test_no_golden_set_is_committed_yet(self):
        """
        Deliberate: the log is currently dominated by wiki-ingest work rather
        than representative chat traffic, so a set mined today would over-weight
        whatever was being tested. Delete this test when the set is frozen.
        """
        from pathlib import Path
        golden_dir = Path(__file__).resolve().parent.parent / "benchmarks" / "golden"
        existing = list(golden_dir.glob("*.jsonl")) if golden_dir.is_dir() else []
        assert not existing, (
            f"a golden set now exists ({existing}) — remove this test and add "
            "coverage that loads and validates it instead"
        )
