# Benchmarking Sage Kaizen Models

**Purpose:** decide whether a candidate model is worth upgrading to, *for Sage Kaizen's
specific use cases*. Not to produce a leaderboard number.

**Created:** 2026-08-06.
**Scope:** the six inference models in `config/brains/brains.yaml`, plus the two voice
models in `sage_kaizen_ai_voice`.

---

## 1. Why public benchmarks cannot answer this

MMLU is saturated (88%+) and top models cluster inside measurement noise; the field has
moved to contamination-resistant suites (LiveBench, GPQA-Diamond, SWE-bench Verified).
Even those do not measure what determines whether an upgrade is right *here*:

- Whether a 7B model stays in English through a 2000-token story
  (Qwen2.5-Omni-7B does not — see `router.CREATIVE_HINTS`)
- Whether it emits exactly one of `FAST | ARCHITECT | SEARCH | ARCHITECT+SEARCH`
  when used as the router's classifier (`router.llm_route`)
- Whether `<think>` arrives in a shape `voice_bridge._TtsFilter` can parse
- Whether MTP speculative decoding still yields ~1.8x
- Whether the audio *and* vision encoders survive in one mmproj

A leaderboard win that costs the audio encoder is a **downgrade** for Sage Kaizen.

**Rule: use public benchmarks only to decide what is worth downloading. Decide with a
private eval.**

---

## 2. The models under test

| Model | Role | Port / GPU | What "better" means |
|---|---|---|---|
| Qwen2.5-Omni-7B Q6_K | FAST brain | 8011 / CUDA1 | TTFT, English-only, audio+vision, router-label discipline |
| Qwen3.6-27B-MTP Q6_K | ARCHITECT | 8012 / CUDA0 | sustained t/s, reasoning quality, `<think>` contract, 128K ctx |
| Qwen3-4B Q8_0 | Summarizer | 8013 / CPU | faithfulness to snippets; latency off the critical path |
| BGE-M3 FP16 | Text embed | 8020 / CUDA0 | recall@k — **re-ingest cost dominates** |
| jina-clip-v2 | Wiki/image embed | 8031 / CUDA1 | recall@k + text↔image shared space |
| CLAP htsat-unfused | Audio embed | 8040 / CUDA1 | audio recall@k |
| distil-large-v3.5 | STT (voice) | CPU | WER, latency |
| Kokoro-82M ONNX | TTS (voice) | CPU | subjective; **GPU migration already tried and rejected** |

---

## 3. Four layers, cheapest first

Run in order. A candidate failing Layer 1 never reaches Layer 2.

### Layer 1 — Hard gates (minutes, no judgment)

Binary. Any failure ends the evaluation. Implemented in `evals/gates.py`.

| Gate | FAST | ARCH | How |
|---|:--:|:--:|---|
| VRAM fits **with co-tenants** | ✓ | ✓ | CUDA1 also hosts wiki-embed A + B and CLAP |
| Audio encoder present | ✓ | — | `clip_has_audio_encoder` / mmproj metadata |
| Vision encoder present | ✓ | — | `clip_has_vision_encoder` |
| Both in ONE mmproj | ✓ | — | combined omni encoder |
| Router-label discipline | ✓ | — | 50 prompts → output ∈ the 4 labels |
| `<think>` → `reasoning_content` | — | ✓ | `reasoning_format: deepseek`; `_TtsFilter` depends on it |
| 128K context | — | ✓ | thinking mode requires it |
| Speculative decoding | — | ✓ | MTP head present, or accept the throughput loss knowingly |
| flash-attn + q8_0 KV accepted | ✓ | ✓ | startup log has no "not supported" line |
| No `.bat` / `cmd.exe` / stdout-redirect regressions | ✓ | ✓ | CLAUDE.md §5 invariants |

### Layer 2 — Performance (~30 min, fully objective)

```powershell
llama-bench -m <model> -dev CUDA0 -ngl 999 -sm none -fa on `
            -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 -t 16 `
            -p 512 -n 128 -d 0,8192 -r 3 -o json
```

**Critical methodological limit:** `llama-bench` feeds *randomly generated tokens*, which
do not adequately exercise speculative decoding / MTP. It also exposes no `--spec-*`
flags. So:

| Number | Source | Meaning |
|---|---|---|
| **base tg** | `llama-bench` | decode ceiling *without* MTP |
| **effective tg** | `llama-server` + real prompts | decode *with* MTP |
| **MTP speedup** | effective ÷ base | **measured 1.41x on 2026-08-06** (§10) |

Only measuring `llama-bench` will understate a MTP model and overstate a non-MTP one.
Both numbers are required. `evals/mtp.py` + `scripts/run_mtp.py` produce the second
one; see §10 for the first measurement of the currently deployed model.

The metrics that actually matter for this app:

| Metric | Why | Target |
|---|---|---|
| **TTFT at 8K depth** | real turns carry system prompt + RAG + wiki + search | FAST < 1.5 s |
| **sustained tg** | ARCHITECT thinking runs thousands of tokens | ≥ current baseline |
| **KV-cache-hit TTFT** | `cache_ram` + `slot_prompt_similarity` give ~93% TTFT reduction | must be preserved |
| **MTP acceptance** | sets effective throughput directly | ~80% |
| **voice TTFT** | voice needs first *audio*, not first token | < 800 ms |

Note the asymmetry: **FAST is TTFT-bound; ARCHITECT is throughput-bound.** A model can
win one and lose the other.

### Layer 3 — Quality on a frozen golden set

Build it from **real logged turns**, not invented prompts. `route_json` and
`context_injection_json` are already written to `log.sage_kaizen` on every turn.

Stratify by the taxonomy the code itself defines:

| Slice | n | Selector |
|---|--:|---|
| Creative long-form | 20 | `router.CREATIVE_HINTS` — **the code-switching detector** |
| Code / debug | 20 | `router.CODE_HINTS` |
| Tutoring K-12 | 20 | `chat_service._TEACH_HINTS` |
| Knowledge / history | 15 | `_KNOWLEDGE_HINTS` |
| Philosophy | 10 | `_PHILOSOPHY_HINTS` |
| RAG-grounded | 25 | `rag_chunks > 0` |
| Search-grounded | 15 | `search_used: true` |
| Voice-length | 15 | `input_chars < 150` |
| Multimodal | 10 | `modality != "text"` |

**Freeze it. Version it. Never regenerate it** — a golden set that changes cannot detect
regression.

Scoring, in increasing cost:

1. **Deterministic assertions** (free, every run) — `evals/scorers.py`:
   - **CJK ratio** — would have caught the Chinese code-switching *before* it needed a
     router workaround
   - Citation markers present when RAG context was injected
   - No `<think>` leakage into `content`
   - Length within 2σ of baseline (catches early-stop and runaway generation)
   - Refusal-rate delta — the system prompt is deliberately unrestricted; a new model
     may be more censored
2. **LLM-as-judge, pairwise.** 80–90% agreement with humans at ~1/500th the cost. Position
   bias is 10–15% even on strong judges, worse on small ones, so mitigation is mandatory:
   **evaluate each pair twice with order swapped; a win counts only if the same response
   wins both orderings; disagreements become ties.** Use ARCHITECT to judge FAST
   candidates. Never let a model judge itself (self-preference bias).
3. **Human spot-check** on pairs where the judge disagreed with itself — that is where
   the information is.

### Layer 4 — Live shadow (one week, optional but decisive)

Run the candidate alongside production on the same turns; compare on the real traffic
distribution rather than a sample.

---

## 4. Embedders need a different method entirely

**Embedding models cannot be A/B tested in place.** Changing one invalidates every stored
vector — `rag_chunks`, `wiki_chunks`, `wiki_images`, `image_embeddings`,
`audio_embeddings`. The Wikipedia corpus alone took multi-day ingest runs across three
GPUs, through repeated crashes.

So the decision rule inverts: **migration cost is the dominant term, not the quality
delta.**

| Model | Test | Gate |
|---|---|---|
| BGE-M3 | recall@k on 100 frozen (query → known chunk_id) pairs | must win by a wide margin |
| jina-clip-v2 | same + text↔image cross-modal | shared vector space is the feature |
| CLAP | audio recall@k | smallest corpus, cheapest to redo |

Practical: run a candidate embedder over a **1–2% sample corpus in a parallel table**
first. If recall@5 does not improve by >10 points on your own queries, do not re-ingest.

The **summarizer** has one metric that matters: **faithfulness** — does the summary assert
anything absent from the SearXNG snippets? A hallucinating summarizer poisons the
`<search_context>` block for every downstream turn.

---

## 5. The decision rule

```
UPGRADE IF:
    all Layer-1 gates pass                       (hard veto)
AND no deterministic scorer regresses            (hard veto — CJK, citations, <think>)
AND TTFT/tg within 10% of current, or better     (or quality win large enough to trade)
AND judge win-rate >= 55% on the weighted slices
AND (embedders) recall@5 gain > 10 points
```

Weight slices by actual usage, not evenly. A model that wins on code but loses on
creative writing and tutoring is a downgrade for this system.

---

## 6. Standing status — Watch List triggers

CLAUDE.md §11 listed five conditions for revisiting the FAST brain. **Three have fired**
(verified against the local build, not assumed).

**Build re-verified 2026-08-06 after a llama.cpp update: b10298 (`15586e2d7`)**, 700
commits past the b9598 (`fdc3db9b6`) build the first verification ran against. All four
projectors are still present:

```
PROJECTOR_TYPE_QWEN3A     <- Qwen3-Omni audio    (Watch List #1, the primary path)
PROJECTOR_TYPE_QWEN3VL    <- Qwen3 vision
PROJECTOR_TYPE_GEMMA4A    <- Gemma 4 audio       (Watch List #4)
PROJECTOR_TYPE_VOXTRAL    <- Voxtral             (Watch List #2)
```

Audio support for Qwen3-Omni merged upstream in b8769 (April 2026); this build is well
past it. `clip.h` exposes `has_vision` and `has_audio` on one context, so a combined omni
mmproj is representable.

New in b10298 and relevant to this stack:

- `PROJECTOR_TYPE_QWEN3TTS_GEN` / `_SPKENC` — llama.cpp can now run **Qwen3-TTS**
  (upstream #26254, a breaking change to the `llama-tts` binary). This is the first
  time a TTS path has existed inside the brain stack at all. It does **not** make
  Kokoro-on-GPU any less rejected (that failed for its own reasons), but it is a
  genuinely new option that did not exist when the voice pipeline was designed:
  TTS on CUDA1 through llama-server rather than ONNX on CPU. Unevaluated — no gate
  has been run against it, and voice TTFT (< 800 ms to first *audio*) is the metric
  that would decide it.
- `spec: support eagle3 for qwen3.5 & 3.6` (#24593) — an **alternative speculative
  decoding path for the deployed ARCHITECT model**. With one MTP head the current
  ceiling is 2 tokens per verification step (§10); an Eagle3 draft head can propose
  longer chains. It needs a separate draft model, so it costs VRAM that CUDA0 does
  not obviously have — but it is the only identified route past the measured 1.41x.
- `server: Adding spec-decode counters to /metrics` (#26389) — the reason §10 could
  be measured with an HTTP read instead of log scraping.

Official GGUFs exist: `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF` (Q4_K_M ≈ 18.6 GB).

**Caveats before anyone gets excited:**

- 30B-A3B is MoE — 3B active, so decode should be fast despite size — but 18.6 GB on
  CUDA1 **alongside wiki-embed A + wiki-embed B + CLAP** is a genuine VRAM question.
  That is exactly what Layer 1 exists to answer.
- Take the **Instruct** variant, not Thinking. FAST must not think, or the routing
  latency model collapses (thinking belongs to ARCHITECT).
- Qwen2.5-Omni-7B's Chinese code-switching is the incumbent's known weakness; the CJK
  scorer makes any improvement measurable rather than anecdotal.

---

## 7. Running the harness

```powershell
# Layer 1 — hard gates
python scripts/run_model_gate.py --brain fast
python scripts/run_model_gate.py --brain architect --json

# Layer 2 — performance baseline (servers must be STOPPED; llama-bench loads its own copy)
python scripts/run_bench.py --brain architect --label baseline
python scripts/run_bench.py --brain architect --label candidate --model E:/path/to/new.gguf
python scripts/run_bench.py --compare baseline candidate

# Layer 2b — effective rate WITH speculative decoding (server RUNNING; needs metrics: true)
python scripts/run_mtp.py --brain architect --label mtp-run --base-run baseline --warmup

# Layer 3 — deterministic scorers over a response set
python scripts/run_scorers.py --responses benchmarks/results/<run>.jsonl
```

Results land in `benchmarks/results/` as JSON, one file per run, so any two runs can be
diffed later.

> **Operational note:** `llama-bench` loads its own copy of the model. Running it while
> `llama-server` holds the same model double-allocates VRAM (2 × 21.3 GB > 32 GB on
> CUDA0). `run_bench.py` refuses to start if the target port is listening.

---

## 8. Status

| Step | State |
|---|---|
| 1. Hard gates (`evals/gates.py`) | **built** |
| 2. Performance baseline (`evals/bench.py`) | **built**; ARCHITECT re-baselined on b10298 2026-08-06 |
| 2b. Effective rate w/ MTP (`evals/mtp.py`) | **built and measured** — see §10 |
| 3. Golden set (`evals/golden.py`) | **mining tool built; set not yet frozen** — deferred until chat traffic is representative again (the log is currently dominated by wiki-ingest work) |
| 4. Deterministic scorers (`evals/scorers.py`) | **built** |
| Pairwise judge | not built — Layer 3 tier 2 |
| Live shadow | not built — Layer 4 |

> `scripts/run_bench.py` had never been executed end to end before 2026-08-06: it
> passed `brains.yaml`'s `exe:` (which is `llama-server.exe`) to `llama-bench`, so
> every invocation exited 1. The §9 baseline had been captured by running
> `llama-bench` by hand. Fixed by deriving `llama-bench.exe` from the configured
> server binary, which also keeps both tools pinned to the same build.

---

## 9. ARCHITECT baseline — measured 2026-08-06 (build b9598)

> **Superseded as the operating baseline by §10.1.** The build was updated the
> same day and the numbers below no longer describe the deployed system. Kept
> because the b9598 → b10298 comparison in §10.1 is only meaningful against it.

Run on an idle machine (all llama-servers stopped, CUDA0 holding only the
desktop), with the flags brains.yaml actually serves:

```
build fdc3db9b6 (b9598) | CUDA0 | fa=1 | KV q8_0/q8_0 | ubatch=512
model  qwen35 27B Q6_K  21.30 GiB

  test          depth        t/s     stddev
  pp512             0    2846.27     199.31
  tg128             0      53.96       1.07
  pp512          8192    2661.59      70.99
  tg128          8192      53.41       0.69
```

Two things worth noting.

**Decode barely degrades with depth.** 53.96 → 53.41 t/s from 0 to 8192 tokens
of context is a ~1% drop. Prefill falls ~6%. For a model whose 16 full-attention
layers carry the whole KV burden while 48 DeltaNet layers use fixed recurrent
state, that is the expected shape — and it means long RAG prefixes cost far less
at decode time than a pure-Transformer 27B would.

**The number is ~7x the previously recorded figure, and that needs explaining
before anyone celebrates.** MEMORY.md records 7.3–7.6 t/s with ~80% MTP
acceptance over a ~4.1 t/s base. This run measures **53.96 t/s base decode**.

The measurements are not directly comparable, and the gap is too large to be
overhead alone:

| | old figure | this run |
|---|---|---|
| model | Qwen3.5-27B (dense, pre-swap) | Qwen3.6-27B-MTP (`qwen35` hybrid) |
| measured | end-to-end via server, real turns | `llama-bench`, random tokens |
| includes | HTTP, sampling, thinking, RAG prefill | pure decode, batch 1 |
| MTP | yes | **no** — llama-bench has no `--spec-*` flags |
| date | 2026-05-20 | 2026-08-06 |

A sanity check supports the new number: at 21.30 GiB and ~1.8 TB/s of RTX 5090
bandwidth, the memory-bound ceiling is ~78 t/s, so 53.96 is **69% of
theoretical** — a normal, healthy figure. The old 4.1 t/s base would be 5% of
theoretical, which is the signature of the pathological DeltaNet kernel
behaviour that MEMORY.md's 2026-05-20 entry describes for the *previous* model.

**What this does not yet tell us:** the *effective* server-side rate with MTP,
real prompts and thinking enabled. Until that is measured, the MTP speedup for
the current model is unknown — the 1.79x figure belongs to a model that is no
longer deployed. That measurement is the next thing worth doing, and it is the
one that would reveal whether real turns are leaving throughput on the table.

Stored as `benchmarks/results/architect-baseline.run.json`.

---

## 10. The llama.cpp update, and the effective rate — 2026-08-06

llama.cpp was updated to **b10298 (`15586e2d7`)**, 700 commits past b9598. Same
build configuration (`CMAKE_CUDA_ARCHITECTURES=120`, `GGML_CUDA_FA=ON`,
`GGML_CUDA_GRAPHS=ON`, Release). Everything below was measured on that build.

### 10.1 The new build is measurably slower — 15-22%

Same model, same flags, same idle machine, GPU0 at its stock 600 W limit (no
power cap applied — checked, because a cap would have explained this away):

```
                    b9598      b10298    b10298      change
                            (run 1)   (run 2)
  pp512   depth 0   2846.27   2260.60   2226.70     -20.6%
  tg128   depth 0     53.96     45.62     44.58     -15.5%
  pp512 depth 8192   2661.59   2138.22   2222.64     -19.7%
  tg128 depth 8192     53.41     41.60     42.77     -22.1%
```

Run twice; it reproduces. Prefill and decode both lose ground, which points at
something shared rather than at one kernel. CUDA graphs are still being captured
and reused (the server log reports `graphs reused = 1126` over four turns), so
that is not the cause. The build configuration is unchanged, so this is
upstream, not local.

**Not diagnosed further.** Bisecting 700 commits at roughly ten minutes per
CUDA rebuild is a day of work, and the honest position is that it has not been
paid for yet. What is established is that it reproduces and that it is not a
power, config, or measurement artefact.

**This does not by itself argue for rolling back.** b10298 is what unlocked
Qwen3-TTS, the Eagle3 path for Qwen3.6, and the spec-decode metrics that made
§10.2 measurable — and §11 is a fix worth more than 20% of raw decode. The
tradeoff is real and worth restating when it is next revisited.

Stored as `baseline-b10298.run.json` and `baseline-b10298-repeat.run.json`.

### 10.2 Effective rate with MTP — the number that was missing

Measured against a live `llama-server` with `metrics: true`, four frozen
prompts, 512 tokens each, thinking enabled, model-card sampling
(temp 0.6 / top_p 0.95 / top_k 20):

```
  turn             pred        ms      t/s  drafted  accept    rate
  reasoning         512      7472    68.52      264     247   93.6%
  code              512      7652    66.91      269     242   90.0%
  creative          512      8526    60.05      286     224   78.3%
  architecture      512      8226    62.24      286     225   78.7%
  TOTAL            2048     31877    64.25     1105     938   84.9%

  base tg 45.62 t/s (llama-bench, same build) -> speedup 1.41x
  ceiling at this acceptance rate: 1.85x
```

**Draft acceptance is 84.9%**, above the ~80% the config comments predicted.
The counters are internally consistent: 1105 verification steps committing 938
accepted drafts accounts for 2043 of the 2048 tokens generated, which is what
one MTP head predicts (each step commits the main token plus at most one
accepted draft). The server's own log agrees with `/metrics` to the token
(`draft acceptance = 0.93561 (247 accepted / 264 generated)`), so the harness
is measuring what it claims to.

**Acceptance is strongly content-dependent, and prose is the weak case.**
Reasoning and code draft at 90-94%; creative writing and open-ended
architectural prose at ~78%. Anything measuring MTP on a single prompt type
would report a number that does not generalise — which is why
`DEFAULT_PROMPTS` spans four slices and stays frozen.

**The 1.41x realised against a 1.85x ceiling is the interesting gap.** The
ceiling is what acceptance alone would buy if the MTP head were free; it is
not. Running the head costs extra work in every forward pass, and that
overhead consumes about a quarter of the theoretical gain. 1.41x is a real
win — it is not the 1.79x recorded on 2026-05-20, but that figure belongs to a
different model on a different build measured a different way, and the two
should not be compared.

Getting past 1.41x means proposing longer draft chains, which one MTP head
cannot do. That is what the Eagle3 support noted in §6 is for, and it is the
single most promising unexplored throughput lead for ARCHITECT.

Stored as `benchmarks/results/mtp-b10298.mtp.json`.

---

## 11. The ARCHITECT prompt cache had never stored anything

Found while reading the server log for §10.2, not looked for.

Every prompt-cache write was being rejected:

```
W srv alloc: - prompt state size 469.951 MiB exceeds cache size limit 256.000 MiB, skipping
W srv alloc: - prompt state size 318.825 MiB exceeds cache size limit 256.000 MiB, skipping
W srv alloc: - prompt state size 318.488 MiB exceeds cache size limit 256.000 MiB, skipping
```

Three attempts, three rejections, on **40-token prompts**.

The cause is architectural, and it is the same property that makes this model
cheap at long context. `cache_ram: 256` was sized for a model whose prompt
state grows with token count. Qwen3.6-27B's 48 DeltaNet layers carry a
**fixed-size recurrent state**, so a snapshot costs ~320-470 MiB regardless of
how short the prompt is. 256 MiB could never hold one. The cache had been dead
since the model swap, and the "~93% TTFT reduction" that `brains.yaml` cited
for it was never being realised.

Raised to `cache_ram: 4096` (host RAM only; 192 GB installed; upstream's own
default is now 8192). Verified by displacing a cached prefix and returning to
it — A, then B to evict A, then A again:

```
  A  (cold)              prompt_n=  3216  prompt_ms=   1707.8
  B  (displaces A)       prompt_n=  3616  prompt_ms=   1782.3
  A  (cache restore)     prompt_n=     4  prompt_ms=     99.9
  B  (cache restore)     prompt_n=     4  prompt_ms=    108.0
```

3216 tokens of prefill replaced by 4 — **1707.8 ms → 99.9 ms, a 94.1% TTFT
reduction**, and no rejection lines in the log. That is the documented benefit,
now actually happening.

**Worth noting for the FAST brain too.** It runs `cache_ram: 512` and is a
dense transformer, so it does not have this failure mode for the same reason —
but it has never been checked, and the check is one `grep` of its log for
`exceeds cache size limit` while it serves real turns. Not done here because
FAST was not started.

**General lesson for this harness:** llama-server reports this at `W` level and
carries on serving perfectly well, just slower. No test would have caught it;
no benchmark number would have looked wrong. It was visible only in the log,
and only because the log was read.
