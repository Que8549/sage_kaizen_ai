"""
tests/test_evals_gguf_and_gates.py

Unit tests for evals/gguf_meta.py and evals/gates.py — Layer 1 of the model
evaluation harness (see Benchmarking_Kaizen_Models.md).

GGUF files are synthesised byte-for-byte in memory rather than read from disk,
so the tests are hermetic and do not need a 20 GB model present.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from evals.gates import (
    DEVICE_CO_TENANTS,
    ROUTER_LABELS,
    GateReport,
    GateResult,
    check_router_label_discipline,
    check_think_contract,
    gate_combined_mmproj,
    gate_context_size,
    gate_mtp_head,
    gate_vram_budget,
    parse_props_context,
    run_static_gates,
    static_gates_architect,
    static_gates_fast,
)
from evals.gguf_meta import GgufError, read_gguf_metadata


# ---------------------------------------------------------------------------
# GGUF synthesis helpers
# ---------------------------------------------------------------------------

def _gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_kv(key: str, value_type: int, payload: bytes) -> bytes:
    return _gguf_string(key) + struct.pack("<I", value_type) + payload


def build_gguf(kv: dict[str, tuple[int, bytes]], *, version: int = 3,
               tensor_count: int = 7, magic: bytes = b"GGUF") -> bytes:
    """Assemble a minimal but valid GGUF header."""
    out = magic + struct.pack("<IQQ", version, tensor_count, len(kv))
    for key, (vtype, payload) in kv.items():
        out += _gguf_kv(key, vtype, payload)
    return out


def write_gguf(tmp_path: Path, name: str, kv: dict[str, tuple[int, bytes]], **kw) -> Path:
    p = tmp_path / name
    p.write_bytes(build_gguf(kv, **kw))
    return p


STR, U32, BOOL, ARR = 8, 4, 7, 9


def _model_kv(arch: str = "qwen2vl", ctx: int = 32768, blocks: int = 28):
    return {
        "general.architecture": (STR, _gguf_string(arch)),
        "general.name": (STR, _gguf_string("test-model")),
        f"{arch}.context_length": (U32, struct.pack("<I", ctx)),
        f"{arch}.block_count": (U32, struct.pack("<I", blocks)),
    }


def _mmproj_kv(vision: bool = True, audio: bool = True, projector: str = "qwen2.5o"):
    return {
        "general.architecture": (STR, _gguf_string("clip")),
        "clip.has_vision_encoder": (BOOL, struct.pack("<?", vision)),
        "clip.has_audio_encoder": (BOOL, struct.pack("<?", audio)),
        "clip.projector_type": (STR, _gguf_string(projector)),
    }


# ---------------------------------------------------------------------------
# gguf_meta
# ---------------------------------------------------------------------------

class TestReadGgufMetadata:
    def test_reads_header_fields(self, tmp_path):
        p = write_gguf(tmp_path, "m.gguf", _model_kv(), version=3, tensor_count=339)
        meta = read_gguf_metadata(p)
        assert meta.version == 3
        assert meta.tensor_count == 339
        assert meta.architecture == "qwen2vl"
        assert meta.name == "test-model"

    def test_missing_file_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_gguf_metadata(tmp_path / "nope.gguf")

    def test_bad_magic_raises_gguferror(self, tmp_path):
        p = tmp_path / "bad.gguf"
        p.write_bytes(b"NOPE" + b"\x00" * 32)
        with pytest.raises(GgufError, match="not a GGUF file"):
            read_gguf_metadata(p)

    def test_truncated_file_raises_gguferror(self, tmp_path):
        p = tmp_path / "trunc.gguf"
        p.write_bytes(b"GGUF" + b"\x00" * 4)
        with pytest.raises(GgufError, match="unexpected end of file"):
            read_gguf_metadata(p)

    def test_implausible_kv_count_is_rejected(self, tmp_path):
        """A non-GGUF file otherwise reads garbage lengths and allocates forever."""
        p = tmp_path / "huge.gguf"
        p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2**40))
        with pytest.raises(GgufError, match="implausible metadata count"):
            read_gguf_metadata(p)

    def test_unknown_value_type_raises(self, tmp_path):
        p = write_gguf(tmp_path, "weird.gguf", {"k": (99, b"")})
        with pytest.raises(GgufError, match="unknown GGUF value type"):
            read_gguf_metadata(p)

    def test_reads_arrays(self, tmp_path):
        payload = struct.pack("<I", U32) + struct.pack("<Q", 3) + struct.pack("<III", 1, 2, 3)
        meta = read_gguf_metadata(write_gguf(tmp_path, "a.gguf", {"nums": (ARR, payload)}))
        assert meta.get("nums") == [1, 2, 3]

    def test_reads_string_arrays(self, tmp_path):
        payload = (struct.pack("<I", STR) + struct.pack("<Q", 2)
                   + _gguf_string("a") + _gguf_string("b"))
        meta = read_gguf_metadata(write_gguf(tmp_path, "s.gguf", {"toks": (ARR, payload)}))
        assert meta.get("toks") == ["a", "b"]

    def test_block_count_found_by_suffix(self, tmp_path):
        meta = read_gguf_metadata(write_gguf(tmp_path, "m.gguf", _model_kv(blocks=65)))
        assert meta.block_count == 65

    def test_block_count_none_when_absent(self, tmp_path):
        kv = {"general.architecture": (STR, _gguf_string("bert"))}
        assert read_gguf_metadata(write_gguf(tmp_path, "m.gguf", kv)).block_count is None

    def test_encoder_flags_default_false(self, tmp_path):
        meta = read_gguf_metadata(write_gguf(tmp_path, "m.gguf", _model_kv()))
        assert meta.has_vision_encoder is False
        assert meta.has_audio_encoder is False
        assert meta.is_multimodal_projector is False

    def test_mmproj_flags(self, tmp_path):
        meta = read_gguf_metadata(write_gguf(tmp_path, "p.gguf", _mmproj_kv()))
        assert meta.has_vision_encoder and meta.has_audio_encoder
        assert meta.is_multimodal_projector
        assert meta.projector_type == "qwen2.5o"

    def test_get_returns_default(self, tmp_path):
        meta = read_gguf_metadata(write_gguf(tmp_path, "m.gguf", _model_kv()))
        assert meta.get("nope", "fallback") == "fallback"


class TestRealModelFiles:
    """
    Validates the reader against the actual incumbent files when present.

    Skipped on a machine without the models — the synthetic tests above carry
    the correctness burden; these guard against a real-world format surprise.
    """

    FAST_MMPROJ = Path("E:/Qwen2.5-Omni-7B-GGUF/mmproj-F16.gguf")
    ARCHITECT = Path("E:/unsloth_Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q6_K.gguf")

    @pytest.mark.skipif(not FAST_MMPROJ.is_file(), reason="FAST mmproj not present")
    def test_fast_mmproj_carries_both_encoders(self):
        meta = read_gguf_metadata(self.FAST_MMPROJ)
        assert meta.has_audio_encoder, "audio upload depends on this"
        assert meta.has_vision_encoder, "image/video upload depends on this"

    @pytest.mark.skipif(not ARCHITECT.is_file(), reason="ARCHITECT model not present")
    def test_architect_has_the_mtp_head(self):
        """block_count 65 = 64 transformer + 1 MTP; brains.yaml spec_draft_n_max=1."""
        meta = read_gguf_metadata(self.ARCHITECT)
        assert meta.architecture == "qwen35"
        assert meta.block_count == 65


# ---------------------------------------------------------------------------
# GateResult / GateReport
# ---------------------------------------------------------------------------

class TestGateResult:
    def test_symbols(self):
        assert GateResult("a", True, "").symbol == "PASS"
        assert GateResult("a", False, "").symbol == "FAIL"
        assert GateResult("a", False, "", blocking=False).symbol == "WARN"

    def test_str_includes_name_and_detail(self):
        assert "vram" in str(GateResult("vram", True, "fits"))
        assert "fits" in str(GateResult("vram", True, "fits"))


class TestGateReport:
    def test_passes_when_all_blocking_pass(self):
        r = GateReport("fast", "m", [GateResult("a", True, "")])
        assert r.passed and not r.blocking_failures

    def test_warning_does_not_block(self):
        r = GateReport("fast", "m", [GateResult("a", False, "", blocking=False)])
        assert r.passed
        assert len(r.warnings) == 1

    def test_blocking_failure_blocks(self):
        r = GateReport("fast", "m", [GateResult("a", False, "boom")])
        assert not r.passed
        assert len(r.blocking_failures) == 1

    def test_to_dict_round_trips_fields(self):
        d = GateReport("fast", "m", [GateResult("a", True, "ok")]).to_dict()
        assert d["brain"] == "fast" and d["passed"] is True
        assert d["results"][0] == {"name": "a", "passed": True, "detail": "ok",
                                   "blocking": True}

    def test_render_reports_verdict(self):
        assert "PASSED" in GateReport("f", "m", [GateResult("a", True, "")]).render()
        assert "BLOCKED" in GateReport("f", "m", [GateResult("a", False, "")]).render()


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------

class TestCombinedMmproj:
    def _names(self, results):
        return {r.name: r.passed for r in results}

    def test_both_encoders_pass(self, tmp_path):
        p = write_gguf(tmp_path, "p.gguf", _mmproj_kv(True, True))
        assert self._names(gate_combined_mmproj(p))["mmproj_combined"] is True

    def test_missing_audio_fails(self, tmp_path):
        p = write_gguf(tmp_path, "p.gguf", _mmproj_kv(vision=True, audio=False))
        names = self._names(gate_combined_mmproj(p))
        assert names["mmproj_audio_encoder"] is False
        assert names["mmproj_combined"] is False

    def test_missing_vision_fails(self, tmp_path):
        p = write_gguf(tmp_path, "p.gguf", _mmproj_kv(vision=False, audio=True))
        assert self._names(gate_combined_mmproj(p))["mmproj_combined"] is False

    def test_missing_file_short_circuits(self, tmp_path):
        results = gate_combined_mmproj(tmp_path / "absent.gguf")
        assert len(results) == 1 and not results[0].passed

    def test_detail_names_the_broken_capability(self, tmp_path):
        p = write_gguf(tmp_path, "p.gguf", _mmproj_kv(audio=False))
        audio = [r for r in gate_combined_mmproj(p) if r.name == "mmproj_audio_encoder"][0]
        assert ".wav" in audio.detail


class TestContextSizeGate:
    def test_sufficient_context_passes(self, tmp_path):
        meta = read_gguf_metadata(write_gguf(tmp_path, "m.gguf",
                                             _model_kv("qwen35", ctx=262144)))
        assert gate_context_size(meta, 131072).passed

    def test_insufficient_context_fails(self, tmp_path):
        meta = read_gguf_metadata(write_gguf(tmp_path, "m.gguf",
                                             _model_kv("qwen35", ctx=32768)))
        assert not gate_context_size(meta, 131072).passed

    def test_exactly_sufficient_passes(self, tmp_path):
        meta = read_gguf_metadata(write_gguf(tmp_path, "m.gguf",
                                             _model_kv("qwen2vl", ctx=32768)))
        assert gate_context_size(meta, 32768).passed

    def test_undeclared_context_warns_rather_than_blocks(self, tmp_path):
        kv = {"general.architecture": (STR, _gguf_string("bert"))}
        meta = read_gguf_metadata(write_gguf(tmp_path, "m.gguf", kv))
        result = gate_context_size(meta, 4096)
        assert not result.passed and result.blocking is False


class TestMtpHeadGate:
    def test_present_head_is_non_blocking_pass(self, tmp_path):
        meta = read_gguf_metadata(write_gguf(tmp_path, "m.gguf",
                                             _model_kv("qwen35", blocks=65)))
        result = gate_mtp_head(meta)
        assert result.passed and result.blocking is False

    def test_undeclared_blocks_warns_only(self, tmp_path):
        kv = {"general.architecture": (STR, _gguf_string("qwen35"))}
        meta = read_gguf_metadata(write_gguf(tmp_path, "m.gguf", kv))
        result = gate_mtp_head(meta)
        assert not result.passed and result.blocking is False

    def test_never_blocks_an_upgrade(self, tmp_path):
        """Losing MTP costs throughput; it is a tradeoff, not a veto."""
        meta = read_gguf_metadata(write_gguf(tmp_path, "m.gguf", _model_kv(blocks=0)))
        assert gate_mtp_head(meta).blocking is False


class TestVramBudgetGate:
    def _model(self, tmp_path, gib: float) -> Path:
        p = tmp_path / "m.gguf"
        p.write_bytes(b"\0" * int(gib * 1024 ** 3 // 1024))   # scaled-down stand-in
        return p

    def test_missing_model_fails(self, tmp_path):
        assert not gate_vram_budget(tmp_path / "absent.gguf", "CUDA0", free_gib=32).passed

    def test_fits_with_headroom(self, tmp_path):
        p = tmp_path / "m.gguf"; p.write_bytes(b"\0" * 1024)
        assert gate_vram_budget(p, "CUDA0", free_gib=30, overhead_gib=5).passed

    def test_insufficient_headroom_fails(self, tmp_path):
        p = tmp_path / "m.gguf"; p.write_bytes(b"\0" * 1024)
        result = gate_vram_budget(p, "CUDA0", free_gib=6, overhead_gib=5, floor_gib=2)
        assert not result.passed

    def test_co_tenants_are_not_deducted_by_default(self, tmp_path):
        """
        free_vram_gib() already excludes resident memory. Deducting the
        co-tenant table on top double-counted and produced a false FAIL on the
        working incumbent config — the bug this default exists to prevent.
        """
        p = tmp_path / "m.gguf"; p.write_bytes(b"\0" * 1024)
        result = gate_vram_budget(p, "CUDA0", free_gib=10, overhead_gib=5)
        assert "co-tenants" not in result.detail
        assert result.passed

    def test_explicit_co_tenants_are_deducted(self, tmp_path):
        p = tmp_path / "m.gguf"; p.write_bytes(b"\0" * 1024)
        result = gate_vram_budget(p, "CUDA0", free_gib=10, overhead_gib=5,
                                  co_tenants={"embed": 4.0})
        assert "planned co-tenants" in result.detail
        assert not result.passed          # 10 - 4 - 5 = 1.0 < floor 2.0

    def test_floor_is_respected(self, tmp_path):
        p = tmp_path / "m.gguf"; p.write_bytes(b"\0" * 1024)
        assert gate_vram_budget(p, "CUDA0", free_gib=7, overhead_gib=5,
                                floor_gib=1.0).passed
        assert not gate_vram_budget(p, "CUDA0", free_gib=7, overhead_gib=5,
                                    floor_gib=3.0).passed

    def test_co_tenant_table_covers_the_three_devices(self):
        assert set(DEVICE_CO_TENANTS) == {"CUDA0", "CUDA1", "CUDA2"}

    def test_co_tenant_table_matches_the_2026_08_24_remap(self):
        """CUDA0 hosts FAST alone; the embed services moved to the CUDA2 eGPU.

        Before the remap CUDA1 was the crowded card (FAST + both wiki embeds +
        CLAP). Now CUDA0 is deliberately empty of co-tenants because it drives
        three monitors, CUDA1 carries only BGE-M3 alongside ARCHITECT, and every
        PyTorch service sits on the otherwise idle RTX 5080.
        """
        assert DEVICE_CO_TENANTS["CUDA0"] == {}
        assert "BGE-M3" in " ".join(DEVICE_CO_TENANTS["CUDA1"])
        cuda2 = " ".join(DEVICE_CO_TENANTS["CUDA2"])
        assert "jina-clip-v2" in cuda2 and "CLAP" in cuda2 and "summarizer" in cuda2
        # The 5080 is 16.3 GiB; the planned set must actually fit.
        assert sum(DEVICE_CO_TENANTS["CUDA2"].values()) < 16.0

    def test_fast_set_passes_a_good_candidate(self, tmp_path):
        model = write_gguf(tmp_path, "m.gguf", _model_kv("qwen2vl", ctx=32768))
        mmproj = write_gguf(tmp_path, "p.gguf", _mmproj_kv())
        assert static_gates_fast(model, mmproj).passed

    def test_fast_set_blocks_when_audio_missing(self, tmp_path):
        model = write_gguf(tmp_path, "m.gguf", _model_kv("qwen2vl", ctx=32768))
        mmproj = write_gguf(tmp_path, "p.gguf", _mmproj_kv(audio=False))
        assert not static_gates_fast(model, mmproj).passed

    def test_fast_set_blocks_on_short_context(self, tmp_path):
        model = write_gguf(tmp_path, "m.gguf", _model_kv("qwen2vl", ctx=8192))
        mmproj = write_gguf(tmp_path, "p.gguf", _mmproj_kv())
        assert not static_gates_fast(model, mmproj).passed

    def test_architect_set_passes_a_good_candidate(self, tmp_path):
        model = write_gguf(tmp_path, "m.gguf", _model_kv("qwen35", ctx=262144, blocks=65))
        assert static_gates_architect(model).passed

    def test_architect_set_blocks_below_128k(self, tmp_path):
        model = write_gguf(tmp_path, "m.gguf", _model_kv("qwen35", ctx=32768, blocks=65))
        assert not static_gates_architect(model).passed

    def test_vram_gate_skipped_when_free_gib_is_none(self, tmp_path):
        model = write_gguf(tmp_path, "m.gguf", _model_kv("qwen35", ctx=262144))
        names = {r.name for r in static_gates_architect(model, free_gib=None).results}
        assert "vram_budget" not in names

    def test_run_static_gates_dispatches(self, tmp_path):
        model = write_gguf(tmp_path, "m.gguf", _model_kv("qwen35", ctx=262144))
        assert run_static_gates("architect", model_path=model).brain == "architect"

    def test_run_static_gates_is_case_insensitive(self, tmp_path):
        model = write_gguf(tmp_path, "m.gguf", _model_kv("qwen35", ctx=262144))
        assert run_static_gates("ARCHITECT", model_path=model).brain == "architect"

    def test_run_static_gates_rejects_unknown_brain(self):
        with pytest.raises(ValueError, match="unknown brain"):
            run_static_gates("summarizer", model_path="x")


# ---------------------------------------------------------------------------
# Live gates
# ---------------------------------------------------------------------------

class TestRouterLabelDiscipline:
    def test_all_clean_labels_pass(self):
        assert check_router_label_discipline(["FAST"] * 20 + ["ARCHITECT"] * 20).passed

    def test_lowercase_and_whitespace_tolerated(self):
        assert check_router_label_discipline(["  fast  ", "architect+search"]).passed

    def test_chatty_model_fails(self):
        completions = ["FAST"] * 8 + ["Sure! This looks like a FAST query."] * 2
        result = check_router_label_discipline(completions)
        assert not result.passed
        assert "Sure!" in result.detail

    def test_empty_input_fails(self):
        assert not check_router_label_discipline([]).passed

    def test_threshold_is_configurable(self):
        completions = ["FAST"] * 9 + ["nonsense"]
        assert not check_router_label_discipline(completions, min_rate=0.95).passed
        assert check_router_label_discipline(completions, min_rate=0.90).passed

    def test_the_four_documented_labels(self):
        assert ROUTER_LABELS == {"FAST", "ARCHITECT", "SEARCH", "ARCHITECT+SEARCH"}

    def test_labels_match_the_router_classifier_prompt(self):
        """The gate must track router._CLASSIFY_SYSTEM, not a drifting copy."""
        import router
        for label in ROUTER_LABELS:
            assert label in router._CLASSIFY_SYSTEM


class TestThinkContract:
    def _payload(self, content: str, reasoning: str | None = "thinking"):
        message: dict = {"content": content}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        return {"choices": [{"message": message}]}

    def test_clean_response_passes_both(self):
        results = check_think_contract(self._payload("The answer."))
        assert all(r.passed for r in results)

    def test_missing_reasoning_field_fails(self):
        results = check_think_contract(self._payload("The answer.", reasoning=None))
        assert not [r for r in results if r.name == "think_reasoning_field"][0].passed

    def test_leaked_think_tag_fails(self):
        results = check_think_contract(self._payload("<think>hmm</think>The answer."))
        leak = [r for r in results if r.name == "think_no_leak_into_content"][0]
        assert not leak.passed
        assert "TTS" in leak.detail

    def test_closing_tag_alone_counts_as_leakage(self):
        results = check_think_contract(self._payload("stuff </think> answer"))
        assert not [r for r in results if r.name == "think_no_leak_into_content"][0].passed

    def test_malformed_payload_fails_cleanly(self):
        assert not check_think_contract({"choices": []})[0].passed
        assert not check_think_contract({})[0].passed


class TestPropsContext:
    def test_reads_top_level_n_ctx(self):
        assert "131,072" in parse_props_context({"n_ctx": 131072}).detail

    def test_reads_nested_n_ctx(self):
        payload = {"default_generation_settings": {"n_ctx": 32768}}
        assert "32,768" in parse_props_context(payload).detail

    def test_missing_n_ctx_warns_only(self):
        result = parse_props_context({})
        assert not result.passed and result.blocking is False
