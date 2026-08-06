"""
Tests for evals/mtp.py — the effective-decode-rate measurement.

The point of this module is a number that gets written into an architecture
document and then used to justify a model swap, so the arithmetic behind it
(acceptance rate, token-weighted throughput, speedup) is asserted here rather
than trusted. The HTTP surface is faked; nothing in this file needs a GPU.
"""
from __future__ import annotations

import json

import pytest

from evals.mtp import (
    DEFAULT_PROMPTS,
    MtpRun,
    SpecCounters,
    TurnResult,
    build_prompt_set,
    fetch_counters,
    load_mtp_run,
    parse_prometheus,
    run_turn,
    save_mtp_run,
    spec_counters,
)


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #

class _Response:
    def __init__(self, status_code: int = 200, text: str = "", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")


class _FakeClient:
    """Serves a scripted sequence of /metrics bodies and one completion."""

    def __init__(self, metrics_bodies, completion=None, metrics_status=200):
        self._metrics = list(metrics_bodies)
        self._completion = completion or {}
        self._metrics_status = metrics_status
        self.posts: list[dict] = []
        self.get_urls: list[str] = []

    def get(self, url, **kwargs):
        self.get_urls.append(url)
        body = self._metrics.pop(0) if self._metrics else ""
        return _Response(status_code=self._metrics_status, text=body)

    def post(self, url, **kwargs):
        self.posts.append({"url": url, **kwargs})
        return _Response(payload=self._completion)


def _metrics_body(drafted: int, accepted: int, drafts: int) -> str:
    return (
        "# HELP llamacpp:spec_decode_num_draft_tokens_total Total draft tokens\n"
        "# TYPE llamacpp:spec_decode_num_draft_tokens_total counter\n"
        f"llamacpp:spec_decode_num_draft_tokens_total {drafted}\n"
        f"llamacpp:spec_decode_num_accepted_tokens_total {accepted}\n"
        f"llamacpp:spec_decode_num_drafts_total {drafts}\n"
    )


# --------------------------------------------------------------------------- #
# parse_prometheus                                                             #
# --------------------------------------------------------------------------- #

class TestParsePrometheus:
    def test_parses_plain_metrics_and_skips_comments(self):
        parsed = parse_prometheus(_metrics_body(100, 80, 100))
        assert parsed["llamacpp:spec_decode_num_draft_tokens_total"] == 100.0
        assert parsed["llamacpp:spec_decode_num_accepted_tokens_total"] == 80.0
        assert not any(k.startswith("#") for k in parsed)

    def test_labelled_series_are_summed_across_labels(self):
        # Accepted-per-position is exposed one series per position; the total is
        # what carries meaning.
        body = (
            'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 60\n'
            'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 20\n'
        )
        parsed = parse_prometheus(body)
        assert parsed["llamacpp:spec_decode_num_accepted_tokens_per_pos_total"] == 80.0

    def test_ignores_blank_and_malformed_lines(self):
        parsed = parse_prometheus("\n\nnot a metric line at all\nfoo 1\n")
        assert parsed == {"foo": 1.0}

    def test_accepts_float_and_scientific_notation(self):
        parsed = parse_prometheus("a 1.5\nb 2e3\n")
        assert parsed["a"] == 1.5
        assert parsed["b"] == 2000.0

    def test_empty_body_yields_empty_mapping(self):
        assert parse_prometheus("") == {}


# --------------------------------------------------------------------------- #
# SpecCounters                                                                 #
# --------------------------------------------------------------------------- #

class TestSpecCounters:
    def test_acceptance_rate(self):
        assert SpecCounters(drafted=100, accepted=80, drafts=100).acceptance_rate == 0.8

    def test_acceptance_rate_is_zero_when_nothing_drafted(self):
        # Division guard: an idle server must read as 0%, not raise.
        assert SpecCounters().acceptance_rate == 0.0

    def test_tokens_per_step_is_one_plus_acceptance(self):
        counters = SpecCounters(drafted=100, accepted=80, drafts=100)
        assert counters.tokens_per_step == pytest.approx(1.8)

    def test_tokens_per_step_bounded_by_two_for_one_head(self):
        # Perfect acceptance with a single MTP head commits 2 tokens per step.
        assert SpecCounters(drafted=50, accepted=50, drafts=50).tokens_per_step == 2.0

    def test_subtraction_gives_the_delta_for_one_turn(self):
        after = SpecCounters(drafted=150, accepted=120, drafts=150)
        before = SpecCounters(drafted=100, accepted=80, drafts=100)
        assert (after - before) == SpecCounters(drafted=50, accepted=40, drafts=50)

    def test_subtraction_clamps_at_zero_after_a_counter_reset(self):
        # A server restart mid-run zeroes the counters; a negative delta would
        # silently poison the run average, so it must clamp instead.
        delta = SpecCounters(drafted=5, accepted=4, drafts=5) - SpecCounters(
            drafted=100, accepted=80, drafts=100
        )
        assert delta == SpecCounters(drafted=0, accepted=0, drafts=0)

    def test_is_active_distinguishes_no_speculation_from_bad_speculation(self):
        assert not SpecCounters().is_active
        assert SpecCounters(drafted=10, accepted=0, drafts=10).is_active

    def test_spec_counters_reads_the_three_metric_names(self):
        counters = spec_counters(parse_prometheus(_metrics_body(7, 5, 7)))
        assert counters == SpecCounters(drafted=7, accepted=5, drafts=7)

    def test_spec_counters_defaults_to_zero_when_absent(self):
        assert spec_counters({}) == SpecCounters()


# --------------------------------------------------------------------------- #
# fetch_counters                                                               #
# --------------------------------------------------------------------------- #

class TestFetchCounters:
    def test_reads_counters_from_the_endpoint(self):
        client = _FakeClient([_metrics_body(10, 8, 10)])
        assert fetch_counters(client, "http://x:8012") == SpecCounters(10, 8, 10)

    def test_strips_trailing_slash_from_base_url(self):
        client = _FakeClient([_metrics_body(1, 1, 1)])
        fetch_counters(client, "http://x:8012/")
        assert client.get_urls == ["http://x:8012/metrics"]

    def test_raises_when_metrics_endpoint_is_disabled(self):
        # llama.cpp answers 501 without --metrics. Returning zeros here would be
        # indistinguishable from a model that never speculates.
        client = _FakeClient([""], metrics_status=501)
        with pytest.raises(RuntimeError, match="--metrics"):
            fetch_counters(client, "http://x:8012")


# --------------------------------------------------------------------------- #
# run_turn                                                                     #
# --------------------------------------------------------------------------- #

def _completion(predicted_n=128, predicted_ms=2000.0, per_second=64.0):
    return {
        "choices": [{"message": {"content": "hi", "reasoning_content": "think"}}],
        "timings": {
            "prompt_n": 20,
            "prompt_ms": 100.0,
            "predicted_n": predicted_n,
            "predicted_ms": predicted_ms,
            "predicted_per_second": per_second,
        },
    }


class TestRunTurn:
    def test_pairs_server_timings_with_the_counter_delta(self):
        client = _FakeClient(
            [_metrics_body(100, 80, 100), _metrics_body(228, 182, 228)],
            completion=_completion(),
        )
        turn = run_turn(client, "http://x:8012", "prompt", label="code")

        assert turn.label == "code"
        assert turn.predicted_tokens == 128
        assert turn.predicted_per_second == 64.0
        assert turn.counters == SpecCounters(drafted=128, accepted=102, drafts=128)

    def test_reads_counters_before_and_after_the_completion(self):
        client = _FakeClient(
            [_metrics_body(0, 0, 0), _metrics_body(1, 1, 1)], completion=_completion()
        )
        run_turn(client, "http://x:8012", "p")
        assert len(client.get_urls) == 2

    def test_requests_a_non_streaming_completion(self):
        # Streaming would fold the HTTP read loop into the measurement; the
        # server's own timings block is the thing being read.
        client = _FakeClient(
            [_metrics_body(0, 0, 0), _metrics_body(0, 0, 0)], completion=_completion()
        )
        run_turn(client, "http://x:8012", "p", max_tokens=256)
        body = client.posts[0]["json"]
        assert body["stream"] is False
        assert body["max_tokens"] == 256

    def test_uses_model_card_sampling_defaults_for_thinking_mode(self):
        client = _FakeClient(
            [_metrics_body(0, 0, 0), _metrics_body(0, 0, 0)], completion=_completion()
        )
        run_turn(client, "http://x:8012", "p")
        body = client.posts[0]["json"]
        assert (body["temperature"], body["top_p"], body["top_k"]) == (0.6, 0.95, 20)

    def test_records_reasoning_length_so_thinking_can_be_confirmed(self):
        client = _FakeClient(
            [_metrics_body(0, 0, 0), _metrics_body(0, 0, 0)], completion=_completion()
        )
        turn = run_turn(client, "http://x:8012", "p")
        assert turn.reasoning_tokens == len("think")

    def test_missing_timings_do_not_raise(self):
        client = _FakeClient(
            [_metrics_body(0, 0, 0), _metrics_body(0, 0, 0)],
            completion={"choices": [{"message": {}}]},
        )
        turn = run_turn(client, "http://x:8012", "p")
        assert turn.predicted_tokens == 0
        assert turn.predicted_per_second == 0.0

    def test_label_defaults_to_a_prompt_prefix(self):
        client = _FakeClient(
            [_metrics_body(0, 0, 0), _metrics_body(0, 0, 0)], completion=_completion()
        )
        turn = run_turn(client, "http://x:8012", "a" * 100)
        assert turn.label == "a" * 32


# --------------------------------------------------------------------------- #
# MtpRun                                                                       #
# --------------------------------------------------------------------------- #

def _turn(label, tokens, ms, per_second, drafted=0, accepted=0, drafts=0):
    return TurnResult(
        label=label,
        predicted_tokens=tokens,
        predicted_ms=ms,
        predicted_per_second=per_second,
        counters=SpecCounters(drafted=drafted, accepted=accepted, drafts=drafts),
    )


class TestMtpRun:
    def test_totals_sum_across_turns(self):
        run = MtpRun(label="r", turns=[
            _turn("a", 100, 1000, 100, drafted=100, accepted=80, drafts=100),
            _turn("b", 100, 1000, 100, drafted=50, accepted=30, drafts=50),
        ])
        assert run.totals == SpecCounters(drafted=150, accepted=110, drafts=150)
        assert run.totals.acceptance_rate == pytest.approx(110 / 150)

    def test_weighted_tg_is_token_weighted_not_a_mean_of_rates(self):
        # 500 tokens in 10 s and 10 tokens in 1 s is 510/11 = 46.4 t/s, not the
        # naive mean of 50 and 10 (30). A short turn must not count as much as
        # a long one.
        run = MtpRun(label="r", turns=[
            _turn("long", 500, 10_000, 50.0),
            _turn("short", 10, 1_000, 10.0),
        ])
        assert run.weighted_tg == pytest.approx(510 / 11)

    def test_weighted_tg_is_zero_for_an_empty_run(self):
        assert MtpRun(label="r").weighted_tg == 0.0

    def test_speedup_divides_effective_by_base(self):
        run = MtpRun(label="r", base_tg=45.0, turns=[_turn("a", 90, 1000, 90.0)])
        assert run.speedup == pytest.approx(2.0)

    def test_speedup_is_zero_without_a_paired_baseline(self):
        run = MtpRun(label="r", turns=[_turn("a", 90, 1000, 90.0)])
        assert run.speedup == 0.0

    def test_render_warns_when_speculative_decoding_never_ran(self):
        run = MtpRun(label="r", turns=[_turn("a", 10, 100, 100.0)])
        assert "WARNING" in run.render()

    def test_render_does_not_warn_when_drafts_were_recorded(self):
        run = MtpRun(label="r", turns=[
            _turn("a", 10, 100, 100.0, drafted=10, accepted=8, drafts=10)
        ])
        assert "WARNING" not in run.render()

    def test_render_reports_speedup_and_ceiling_when_baseline_present(self):
        run = MtpRun(label="r", base_tg=45.0, base_label="baseline", turns=[
            _turn("a", 90, 1000, 90.0, drafted=90, accepted=72, drafts=90)
        ])
        rendered = run.render()
        assert "2.00x" in rendered
        assert "1.80x" in rendered  # ceiling = 1 + acceptance

    def test_round_trips_through_json(self, tmp_path):
        run = MtpRun(label="r", model="m.gguf", base_tg=45.0, turns=[
            _turn("a", 90, 1000, 90.0, drafted=90, accepted=72, drafts=90)
        ])
        path = save_mtp_run(run, tmp_path / "r.mtp.json")
        restored = load_mtp_run(path)

        assert restored.label == "r"
        assert restored.base_tg == 45.0
        assert restored.turns[0].counters == SpecCounters(90, 72, 90)
        assert restored.totals == run.totals

    def test_save_stamps_a_measurement_time(self, tmp_path):
        path = save_mtp_run(MtpRun(label="r"), tmp_path / "r.mtp.json")
        assert json.loads(path.read_text(encoding="utf-8"))["measured_at"]

    def test_save_preserves_an_explicit_measurement_time(self, tmp_path):
        run = MtpRun(label="r", measured_at="2026-01-01T00:00:00Z")
        path = save_mtp_run(run, tmp_path / "r.mtp.json")
        assert json.loads(path.read_text(encoding="utf-8"))["measured_at"] == (
            "2026-01-01T00:00:00Z"
        )


# --------------------------------------------------------------------------- #
# Prompt set                                                                   #
# --------------------------------------------------------------------------- #

class TestPromptSet:
    def test_default_set_is_returned_whole(self):
        assert build_prompt_set(None) == list(DEFAULT_PROMPTS)
        assert build_prompt_set([]) == list(DEFAULT_PROMPTS)

    def test_filters_to_named_slices(self):
        assert [label for label, _ in build_prompt_set(["code"])] == ["code"]

    def test_filter_is_case_insensitive(self):
        assert len(build_prompt_set(["CODE"])) == 1

    def test_unknown_slice_yields_nothing_rather_than_everything(self):
        # run_mtp.py turns an empty set into an error; silently measuring all
        # four slices after a typo would produce a mislabelled result.
        assert build_prompt_set(["nonexistent"]) == []

    def test_slices_cover_distinct_content_types(self):
        # Acceptance rate is content-dependent — code drafts far better than
        # prose — so the set must not collapse to one kind of text.
        labels = {label for label, _ in DEFAULT_PROMPTS}
        assert {"reasoning", "code", "creative"} <= labels
