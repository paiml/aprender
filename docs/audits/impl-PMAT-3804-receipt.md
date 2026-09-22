# PMAT-3804 — coder-0.5b Q4_K_M CUDA golden output is gibberish (CPU passes)

**Row:** GH #3804, epic #3710 (0.69.1), under the operator's all-Q4_K-on-CUDA rule.
**Worker:** aprender-d8. **Cop:** aprender-3e.
**Branch:** `PMAT-3804-q5-0-gpu-dispatch` off `origin/main` @ `a9502d992`.
**Tip:** `85d8491f6` = my test/diagnostic commit + `da218127b` cherry-picked clean.

## Verdict

**Root cause named: the FP8 E4M3 batched prefill path.** Not the quant types, not
the 0.5B geometry, not the decode path. Established by a one-variable experiment
in a single binary, not by reading code (§4, §5), and confirmed as a cause rather
than an anecdote by the six-model sweep (§8), where coder-7b also diverges.

**The row does NOT close on the ratified fix alone.** `da218127b` (#3807 (a))
widens a retry that lives behind the F2 guard, and `apr qa`'s golden gate never
reaches that guard — measured in §10, where the gate's verdict is byte-identical
with and without the fix. **gx10 already passes on main** (§9); the outstanding
cell is sm_89.

## Provenance of every measurement below

| | |
|---|---|
| Binary | `apr 0.69.0 (a9502d992)`, pinned via `. scripts/apr_bin.sh` |
| Host | this dev box, **NVIDIA GeForce RTX 4090, sm_89** — the same compute capability as lambda |
| Model | `/home/noah/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` |
| Lock | every GPU command through `gpu-q --prio 0`; builds run off-lock |

**Three binaries appear below and they are NOT the same artifact.** B0 and B1
both self-report `a9502d992`, because apr-cli embeds the git SHA of HEAD and my
worktree's HEAD had not moved yet — #2739's tree-attribution gap, restated. Which
binary produced which number is therefore stated per section rather than inferred
from `--version`:

* **B0 — pristine `origin/main`** (`a9502d992`). §1 only.
* **B1 — B0 + two diagnostics** (`[PMAT-3804]` per-position cosine dump;
  `APR_F2_PROBE_PATH` probe-path override), reports `a9502d992`. §3–§5. Neither
  diagnostic changes any kernel, route or numeric; both are inert unless
  `APR_DEV_TRACE` / `APR_F2_PROBE_PATH` is set, and §1's failure reproduces
  identically under B0 and B1.
* **B2 — B1 + `da218127b`** (the ratified #3807 (a) retry), reports `85d8491f6`,
  snapshotted to a stable path so a later rebuild cannot silently replace it.
  §7b's proof only.

## 1. Reproduced locally — the defect is host-independent

```
gpu-q --prio 0 -- "$APR" qa .../qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
      --skip-throughput --skip-ollama --json     # rc=5
```

| gate | result |
|---|---|
| `golden_output` | **FAIL** — `GPU output failed (CPU passed): gibberish (fragment "```\n" repeats 3+ times)` |
| `gpu_state_isolation` | **FAIL** — `State leak: prompt A produced different output on retry` |
| `gpu_speedup` | pass — GPU 159.5× faster than CPU (419 vs 3 tok/s) |
| `format_parity`, `ptx_parity`, `tensor_contract`, `metadata_plausibility`, `capability_match` | pass |

Byte-identical fragment to the lambda main row in the ticket.

**CORRECTION — I first called this "host-independent" and that was wrong.** The
evidence was this box's RTX 4090 plus lambda's RTX 4090: two instances of the
SAME silicon, which is not two hosts in the sense #3715 means. §9 measures gx10
(GB10, compute capability 12.1) and it **passes on pristine main**. The correct
statement is that the defect is specific to **sm_89 / RTX 4090**. What the
reproduction does buy is that lambda and this box are interchangeable for it, so
the row did not need lambda's queue.

`gpu_state_isolation` is a SECOND defect, not in the ticket. It is not on the
path to this one (§3 shows the diverging forward is deterministic) and is filed
separately rather than allowed to steer this row.

## 2. The file is 12/291 Q4_K — "Q4_K_M" is a filename, not the contents

Parsed from the GGUF tensor table (type id per tensor):

| model | Q4_K | Q6_K | Q5_0 | Q8_0 | F32 | golden |
|---|---|---|---|---|---|---|
| **coder-0.5b-q4_k_m** | **12** | 12 | **133** | 13 | 121 | **FAIL** |
| coder-1.5b-q4_k_m | 169 | 29 | 0 | 0 | 141 | pass |
| coder-7b-q4_k_m | 169 | 29 | 0 | 0 | 141 | pass |
| qwen2.5-1.5b-q4_k_m | 169 | 29 | 0 | 0 | 141 | pass |

K-quants require the row width divisible by `QK_K = 256`. This model's
`embedding_length` is **896**, and 896 = 3.5 × 256. So every 896-wide row fell
back to a 32-element legacy block: `attn_q`/`attn_k`/`attn_output`/`ffn_gate`/
`ffn_up`/`token_embd` are Q5_0, `attn_v` is Q8_0×12 + Q5_0×12, the LM head is
Q8_0. Only `ffn_down` (4864 = 19 × 256) stays a K-quant.

Posted to the issue as
[#3804 comment](https://github.com/paiml/aprender/issues/3804#issuecomment-5771750405),
because it changes what "all Q4_K models work on CUDA" asserts for #3715.

**The ticket's f16 exoneration is invalid.** F16 is not on the GPU whitelist
(`gguf/dtype.rs::gpu_unsupported_quant_qtype` admits `0|2|3|6|8|12|13|14`; F16 is
1), so `Qwen2.5-0.5B-Instruct-f16` never runs the GPU path — its pass is a CPU
pass and says nothing about 0.5B geometry on GPU.

## 3. The diverging forward is DETERMINISTIC (so the bisect is sound)

10 fresh processes, per-position cosine at all 16 probe positions,
`--revalidate` so no receipt short-circuits the guard:

```
pos 0: cosine 0.099921  cpu_argmax 144328  gpu_argmax 80984
pos 1: cosine 0.415256  cpu_argmax  73562  gpu_argmax 20840
pos 2: cosine -0.689155 cpu_argmax    474  gpu_argmax 20840
...
pos15: cosine 0.867281  cpu_argmax    488  gpu_argmax    15
```

**Byte-identical to 6 decimals in all ten runs.** This kills the
uninitialized-memory and race hypotheses for THIS divergence, and
non-deterministic reduction order is ruled out twice over: by the exact
repeatability, and by magnitude — 0.0999 is near-orthogonal, not drift.

It also shows **position 0 is already wrong, and worst of all**. At position 0
attention is a softmax over one key and returns V[0] whatever the scores are, so
a score-path defect is invisible there: attention was not the cause.

## 4. Decode is correct; batched prefill is the defect

Same binary, same model, one env var (`APR_F2_PROBE_PATH`, added for this):

| probe path | pos0 | pos1 | all 16 positions |
|---|---|---|---|
| serial (m=1 decode) | 0.999946 | 0.999869 | cosine ≥ 0.9978, **every argmax matches CPU** |
| batched (m=16 prefill) | 0.099921 | 0.415256 | argmax wrong at every position |

## 5. Inside batched prefill: FP8 is the cause

| config | pos0 | pos1 |
|---|---|---|
| default | 0.099921 | 0.415256 |
| **`FP8_PREFILL=0`** | **0.999948** | **0.999864** |
| `CUBLAS_PREFILL=0` | 0.099921 | 0.415256 |
| `FP8_PREFILL=0 CUBLAS_PREFILL=0` | 0.250566 | −0.474033 |

Disabling FP8 prefill repairs the path. Disabling cuBLAS alone changes nothing,
exactly as the route order predicts: `route_fp8_decode` is tested **before**
`route_cublas` (`cuda/executor/layers/cublas_prefill/attention.rs:1131` vs
`:1166`), so killing cuBLAS leaves the FP8 route armed.

Mechanism: `is_cublas_qtype` is `Q4K || Q6K` only
(`cublas_prefill/attention.rs:1024`), so in this file the ONLY tensors that can
reach FP8 are the 24 `ffn_down` — and the load log agrees exactly:
`[PMAT-053] FP8 weight cache: 24 matrices cached (99.8 MB)`. Every Q5_0/Q8_0
projection falls through to the per-row GEMV fallback
(`layers/batched_qkv.rs:31-53`), which is why §7's real-shape GEMV parity test
passes and why the quant types were a red herring. The FP8 weight path is
per-tensor-absmax scaled (`quant_scale = 448/absmax`,
`cublas_prefill/gemm.rs:38`), so one large outlier in a `ffn_down` collapses most
of the tensor toward zero — consistent with a near-orthogonal result rather than
precision drift.

**The fourth row is a second, distinct defect**: with both FP8 and cuBLAS off the
path is still wrong (0.2506 / −0.4740), so the non-cuBLAS batched fallback is
independently broken for this model. Filed separately; it does not affect the
default configuration.

### The rescue path declined to fire

`f2_should_retry_without_fp8` re-measures on the FP16 prefill only inside a
non-catastrophic band. At 0.4153 this model falls below it, so the code that
would have rescued it declined to, and the model was pushed off the GPU instead.
**This is the defect the remedy targets** — see §7b: the ratified fix widens the
retry to any FP8-batched miss, and §5's own `FP8_PREFILL=0` row is the evidence
that the FP16 path it retries on reaches parity.

## 6. Gate/guard gap — `apr run` refuses what `apr qa` ships

`apr run --gpu` does not emit gibberish at all: its parity guard prints
`GPU output diverges from CPU at position 1 (argmax 20840 != 73562, cosine
0.4153)` and exits **rc=14**. `apr qa`'s golden path has no such guard and reports
gibberish instead. With the serial probe forced, the guard ACCEPTED the GPU,
generation then ran with batched prefill, and emitted the golden failure fragment
` ``` ``` ``` ` verbatim — the gate and the guard disagreeing about the same
model in the same binary, with the gate being the one that ships. Filed against
#3712.

## 7. Code changes on this branch

| file | change |
|---|---|
| `crates/aprender-serve/src/cuda/executor/pmat3804_gemv_shape_parity.rs` | **new.** GPU↔CPU GEMV parity at the model's real shapes: Q5_0 at (n,k) = (896,896)/(128,896)/(4864,896), Q8_0 at (128,896)/(896,896)/(2048,896), against `crate::quantize::dequantize_q5_0/q8_0`. **Both pass** — kept anyway: the prior Q5_0 coverage was k=256,n=4 and Q8_0 had **no** numerical test at any shape while `dtype.rs` whitelists it as "verified". |
| `crates/aprender-serve/src/cuda/executor/q_basic.rs` | `include!` the above |
| `crates/aprender-serve/src/infer/inference_result.rs` | `[PMAT-3804]` per-position cosine dump (dev-trace gated) — the verdict scores positions ≥1 only, so a rejection could not say whether pos0 also diverged, which was the whole diagnosis |
| `crates/aprender-serve/src/infer/inference_result.rs` | `APR_F2_PROBE_PATH=serial\|batched` — judge the path the engine did not resolve, in ONE binary (the `APR_GRAPH_QTYPE_HARDCODE` idiom from #2753) |

## 7b. The remedy: `da218127b` (#3807 (a)), cherry-picked

Ratified by the cop; authored by aprender-37. `f2_should_retry_without_fp8`
stops retrying only inside a non-catastrophic band and re-measures **any** FP8
miss on `via == Batched && fp8_prefill_on`, freeing the FP8 cache first. The
reasoning is exactly what §5 measures: a second measurement cannot turn
orthogonal logits into parity, but **changing precision can** — this model sits
at 0.4153, below the old band, so the code that would have rescued it declined
to and it was pushed to CPU. FP16 reaches 0.999948.

Cherry-picked onto this branch in a clean worktree: `Auto-merging
inference_result.rs`, 1 file, 33+/30−, no conflict, despite both diagnostics
living in that same file. `apr --version` then reports `apr 0.69.0 (85d8491f6)`,
self-attributing to this branch tip rather than to `a9502d992`.

**Routing FP8 off for ffn_down-only models is a possible follow-up, not the
fix** — it is narrower than the retry and would be tuned to one model's tensor
mix. The `is_cublas_qtype` = Q4K||Q6K reasoning and the
`[PMAT-053] FP8 weight cache: 24 matrices cached` line in §5 are what such a
rule would key on.

## 7c. A measurement error of mine, recorded

I rebuilt the shared `target/release/apr` **while the sibling sweep was running
against it**, so that sweep was mixing a pre-fix and a post-fix binary across
models. That is "never label a run by intent — prove the mechanism engaged"
failing in my own hands. The partial results were **discarded unread** rather
than interpreted; a stable binary was snapshotted and the sweep relaunched
against only that. No sibling number in this receipt comes from the mixed run.

Related hazard for others: my builds write to `/home/noah/src/aprender/target`,
so that path's `release/apr` is currently this branch's build, not pristine main.

## 8. The sibling sweep — the FP8 defect is NOT 0.5b-specific

Batched probe, `[PMAT-3804]` dump (which prints the FIRST measurement, before
any retry). Binary B2, `sha256 46741d3f3fe5347ac6f05685`.

| model | pos0 | pos1 | verdict |
|---|---|---|---|
| **coder-0.5b** (this row) | **0.099921** | **0.415256** | diverges, **no retry** — below the band |
| **coder-7b** | 0.996443 | **0.587395** | **diverges under FP8**, retry fired, rescued |
| coder-1.5b | 0.991938 | 0.977885 | accepted |
| qwen2.5-1.5b | 0.864074 | 0.992710 | accepted (pos0 is unscored) |
| Qwen3-1.7B | 0.999920 | 0.998372 | clean |
| Qwen3.5-2B, Qwen3.5-0.8B | n/a | n/a | **not evidence** — `qwen35 hybrid forward, #3090`, the probe never runs |

coder-7b's own log:

```
note: FP8 batched prefill scored min cosine 0.5874 vs CPU (floor 0.95) — re-measuring on the FP16 prefill path
note: FP16 prefill passes (min cosine 0.9834); FP8 stays OFF for this model
```

**A 7B model lands at 0.5874 against a 0.95 floor.** The FP8 E4M3 batched
prefill is broadly lossy, not 0.5b-specific; the 0.5b is simply the one that fell
BELOW the retry band and therefore got no rescue. This is the strongest argument
for #3807 (a), and it also means "FP8 prefill" is a perf claim that silently
degrades to FP16 on at least two of six swept models — worth its own 0.70.0 row.

**Scope answer: this is not the epic.** Every other swept model is already
rescued by the existing retry. #3804 stays a row.

## 9. gx10 — PASSES on pristine main

`apr qa` coder-0.5b on gx10 (**NVIDIA GB10, compute capability 12.1**, aarch64),
binary `apr 0.69.0 (a9502d99)`, `sha256 2cdfd4e4a05cc5d65240401a`, mtime
2026-09-22 01:38:42Z — a binary I did not build:

```
passed: True
  PASS golden_output: 3 golden test cases passed
  PASS gpu_state_isolation: GPU state properly isolated: 3 generations, deterministic replay confirmed
  PASS gpu_speedup: GPU 59.8x faster than CPU (182 vs 3 tok/s)
```

The defect does not reproduce on GB10. #3715's gx10 cell for this model is
satisfied **on main, with no fix**.

**Weaker pin, stated rather than hidden:** that binary lives at
`~/.cargo/bin/apr`, a path another session can rewrite. Its sha256 and mtime
(hours before the run) are recorded, but under the standing rule this is a weaker
pin than the local snapshot and is not presented as equal evidence.

## 10. The ratified fix does NOT close this row — measured

`apr qa` coder-0.5b **with `da218127b` in the binary** (B2):

```
rc=5   golden_output FAIL: GPU output failed (CPU passed):
       golden_output_gpu: gibberish (fragment "```\n" repeats 3+ times)
```

**Byte-identical to pristine main (§1).** Proven, not inferred: grepping that
run's stderr for the guard's and the retry's own lines — `re-measuring`,
`FP16 prefill passes`, `diverges from CPU`, and the `[PMAT-3804]` dump — returns
**zero occurrences**. The F2 guard never executes in `apr qa`'s golden path, so
the retry behind it cannot fire, so widening the retry band cannot change this
gate's verdict. FP8 is still active in that run (`FP8 weight cache: 24 matrices
cached`) — it is the *guard* that is absent, not the path.

This is what #3821 predicted in writing before it was run. **`da218127b` is
necessary and not sufficient**: closing done_when 1 requires the gate to reach the
guard. Scope call belongs to the cop/operator.

## 11. Both-off anomaly reproduced

`FP8_PREFILL=0 CUBLAS_PREFILL=0` → **rc=14** (GPU refused), reproducing §5's
fourth row. The third batched path — neither FP8 nor cuBLAS — diverges
independently. Filed to 0.70.0.

## 12. done_when status

| # | requirement | status |
|---|---|---|
| 1 | `apr qa` golden passes on lambda AND gx10 | **gx10 MET** (§9, on main). **sm_89 NOT MET** (§10) — `da218127b` alone does not close it; the gate must reach the guard |
| 2 | root cause named at file:line + planted mutant turning a test RED | **named** (§4, §5) to route, file and line. **RED-turning mutant still owed** |
| 3 | model-size-specific? sweep the neighbours | **MET** (§8) — not size-specific; coder-7b also diverges at 0.5874 and is rescued only by the retry band |
| 4 | CPU/GPU parity on the fixed path, cited | **partially** — FP16 path reaches 0.999948 (§5) and 0.9834 on coder-7b (§8); not yet cited through a passing `apr qa` |

## 9. Open, and owed

* **A planted mutant that restores the defect and turns a test RED** (done_when
  #2). Still owed; it is the one part of the row I have not delivered.
* The scope decision in §10: whether the gate/guard routing (#3821) returns to
  0.69.1, without which done_when 1 cannot close on sm_89.
* Filed and out of this row: #3821 (verdict lattice / gate-guard), #3822
  (hardcoded Q4K for Q/K/O), #3823 (vacuous GEMV tests), #3824
  (`gpu_state_isolation`), plus the §11 both-off path.
* **Now claimed, with the sweep landed (§8):** FP8 batched prefill is the cause,
  and it is not 0.5b-specific. Before §8 this rested on one model, which by the
  standing rule was an anecdote.
