# PMAT-3804 — coder-0.5b Q4_K_M CUDA golden output is gibberish (CPU passes)

**Row:** GH #3804, epic #3710 (0.69.1), under the operator's all-Q4_K-on-CUDA rule.
**Worker:** aprender-d8. **Cop:** aprender-3e.
**Branch:** `PMAT-3804-q5-0-gpu-dispatch` off `origin/main` @ `a9502d992`.

## Verdict

**Root cause named: the FP8 E4M3 batched prefill path.** Not the quant types, not
the 0.5B geometry, not the decode path. Established by a one-variable experiment
in a single binary, not by reading code.

## Provenance of every measurement below

| | |
|---|---|
| Binary | `apr 0.69.0 (a9502d992)`, pinned via `. scripts/apr_bin.sh` |
| Host | this dev box, **NVIDIA GeForce RTX 4090, sm_89** — the same compute capability as lambda |
| Model | `/home/noah/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` |
| Lock | every GPU command through `gpu-q --prio 0`; builds run off-lock |

Two binaries appear below and they are NOT the same artifact. `apr --version`
reports `a9502d992` for both, because apr-cli embeds the git SHA of HEAD and my
worktree's HEAD never moved (#2739's tree-attribution gap, restated):

* **B0 — pristine `origin/main`.** Used for the reproduction in §1.
* **B1 — B0 + two diagnostics I added** (`[PMAT-3804]` per-position cosine dump;
  `APR_F2_PROBE_PATH` probe-path override). Used for §3–§5. Neither diagnostic
  changes any kernel, route or numeric; both are inert unless `APR_DEV_TRACE` /
  `APR_F2_PROBE_PATH` is set.

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

Byte-identical fragment to the lambda main row in the ticket. **lambda and gx10
are not the variable**, which takes this row out of the GPU queue.

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
**The remedy may therefore be routing (FP8 off when the only cuBLAS-eligible role
is `ffn_down`) rather than a kernel fix** — that is a call for the cop/operator,
not for me.

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

## 8. done_when status

| # | requirement | status |
|---|---|---|
| 1 | `apr qa` golden passes on lambda AND gx10 | **NOT MET** — no fix applied yet; cause named, remedy is a routing/kernel decision |
| 2 | root cause named at file:line + planted mutant turning a test RED | **partially** — named to route and file:line (§5); the RED-turning mutant is owed |
| 3 | model-size-specific? sweep the neighbours | **IN FLIGHT** — sweep running; §2 gives the static composition half |
| 4 | CPU/GPU parity on the fixed path, cited | **not yet** — no fix applied |

## 9. Open, and owed

* Sweep result for done_when #3 (running).
* A planted mutant that restores the defect and turns a test RED (done_when #2).
* gx10 confirmation for done_when #1.
* Three separate filings: the hardcoded-Q4K Q/K/O dispatch, the vacuous GEMV
  tests, the guard/gate gap (#3712).
* **Not claimed:** that the siblings survive FP8. Until the sweep lands, "FP8 is
  the cause" is supported by a one-variable experiment on ONE model, which by
  the standing rule is an anecdote until varied.
