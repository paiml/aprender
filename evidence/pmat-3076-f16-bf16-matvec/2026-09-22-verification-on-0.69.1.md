# #3076 verification on the 0.69.1 release tree — the work is DONE, the issue is OPEN

**Result: #3076 is already fixed and folded. Alfredo's symptom does not reproduce.**
No code was written for this. What follows is the measurement that says so.

## The finding before the measurement

`crates/aprender-serve/src/gguf/inference/float16_dot.rs` opens with `// #3076:` and
documents the fix, the before/after numbers and a receipt directory. It landed in
**`a9502d992`** (`release(0.70): batch-1 — 9 receipted rows`), and
`evidence/pmat-3076-f16-bf16-matvec/` already holds greedy-parity receipts **and
negative-control mutants** for both formats.

**Issue #3076 is still OPEN.** The work shipped; nobody closed the ticket. This is the
"not filed ≠ not built" shape in reverse — *still open ≠ still broken* — and it cost an
assignment. Checking the tree before the ticket is the cheap half of that.

## Measured on this tree, not inferred from the comment

- binary `apr 0.69.1 (b3b3faa33)`, `sha256 d4bfc959042508ab…`, pinned by `scripts/apr_bin.sh`
- host `noah-Lambda-Vector` (Threadripper 7960X, Zen 4, 48 threads), **load 16–21 throughout**
  (another session holds the GPU; every number below is therefore a floor, not a best case)
- decode rate by the original receipt's own method: `(inference_time_ms@64 − @8) / 56`

| model | format | ms/token | tok/s | wall, 16 tokens |
|---|---|---|---|---|
| `Qwen2.5-0.5B-Instruct-f16` | F16 | 53.6 | **18.7** | 2.95 s |
| `Qwen3-0.6B-BF16` | BF16 | 83.5 | **12.0** | — |

The original receipt measured 21.2 tok/s for the f16 at default threads under load 27–42.
18.7 at load 16–21 is the same result on a differently-loaded box, which is what a real
fix looks like a month later.

## Alfredo's repro shape, end to end

His report: `apr chat --gpu <fp16>` → GPU refuses → CPU → *"never produces a token within a
Ctrl-C-worthy wait"*, 100% CPU, no progress output. Run on the f16 model we hold:

```
[GGUF CUDA init failed: Capability mismatch for 'qwen2': missing GPU support for
 [GPU GEMV kernel for the model's quantization type] … will use CPU]
You: [2.2s, ~0 tok/s]
Assistant: Paris
{"backend":{"requested":"default","ran":"cpu","fell_back":true}}
wall = 4.26 s
```

**4.26 seconds, correct answer, and the refusal names itself** (#3075/PMAT-785). The
"appears to hang" symptom is gone on every surface we can drive.

## What this does NOT establish — stated because the gap is real

Alfredo's model is **Phi-3-mini-4k-instruct-fp16 (3.8B)**. We hold no Phi and no f16
larger than 0.6B, so **this was not reproduced at his scale.** At ~8× the weights the
same path predicts roughly 2–3 tok/s: slow, and still not a hang. That is an
extrapolation from the two points above, not a measurement, and it should be confirmed
on his file before the ticket is closed on his behalf.

## A separate finding, carried out of this work

`simd_bf16_matmul` is defined **twice** — `inference/simd_bf16_ops.rs:24` and
`inference/simd_bf16_matmul.rs:24` — and has **no production caller**. Its only
non-definition references are a re-export in `inference/mod.rs` and one integration
test (`tests/benchmark_parity_safetensors.rs:706`). That is #3075's shape: an
implementation nothing dispatches to. It is not what #3076 was about — the live F16/BF16
GGUF path goes through `float16_matmul` → `float16_row_dot`, which is the AVX2/F16C/FMA
kernel above — so this is dead or duplicated code rather than a correctness gap, and it
belongs in its own row.

## Recommendation

1. Close #3076 against `a9502d992`, citing its receipts and these numbers — **or** ask
   Alfredo to re-run on his Phi file first, which is the safer order since he filed it.
2. File the duplicate/uncalled `simd_bf16_matmul` separately.
