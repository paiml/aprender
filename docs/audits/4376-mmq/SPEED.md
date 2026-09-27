# #4376 MMQ int8 slice, step 2: speed of the v1 tensor-core kernel

aprender-79, 2026-09-28. Same cell and model as `ACCURACY.md` (lambda-vector, RTX 4090,
sm_89, `Qwen3.5-4B-Q4_K_M.gguf`, prompts from `docs/audits/srv-fix-002/gx10/prompts.sh`).

- Binary: `apr 0.70.0 (18d5c37582)`, built from this branch with a clean tree
  (`speed-bin-sum.txt`).
- Settings: `--max-tokens 1`, `APR_QWEN35_PREFILL_GEMM` ∈ {f16, dp4a, mmq}, n=3, legs
  interleaved, each run under `/tmp/apr-gpu.lock`. Each number is apr's own
  `[qwen35] batched prefill:` line (`speed-legs.tsv`).
- The mechanism engaged: mmq and dp4a run the same Q8_1 input and differ only in the
  GEMM kernel, and they separate by 1.6–1.8× in every run.

## Result (median of 3, ms)

| prompt | tokens | f16 (default) | dp4a | mmq v1 | mmq / dp4a | mmq / f16 |
|---|---|---|---|---|---|---|
| p2_2k | 2,154 | 316 | 915 | 519 | 1.76× faster | 0.61× |
| p3_8k | 8,119 | 1,057 | 3,316 | 1,760 | 1.88× faster | 0.60× |
| p5_30k | 28,924 | 6,608 | 14,616 | 9,179 | 1.59× faster | 0.72× |

Against llama.cpp `d1d3c3396` pp (13,609 / 14,040 / 12,724 tok/s, `4313-rtx4090`):
mmq v1 is 0.30× / 0.33× / 0.25×; the f16 default is 0.50× / 0.55× / 0.34× in this run.

## Verdict

1. The tensor-core kernel does what the instruction swap promises over DP4A: 1.6–1.9×.
2. **v1 does not beat the f16 default**, so under the ruling (int8 opt-in, f16 default)
   it changes nothing for users yet. It is not prefill parity and is not called that.
3. v1 is the naive form: every warp loads its A/B fragments straight from global memory,
   with no shared-memory staging, no `ldmatrix`, no reuse of a W tile across the
   block's 4 warps, and it re-reads the Q4K scale header per sub-block. That is the next
   step (v2). The int8 leg has to beat f16's GEMM by enough to pay for the Q8_1
   quantize pass before it can be worth anything.
4. At 29k tokens most of the gap to llama.cpp is attention, not the GEMM
   (`4313-rtx4090/RECEIPT.md` §C). The GEMM alone cannot reach 0.8×.

## v2: shared-memory tiles, and f16 for the non-Q4K weights

aprender-79, 2026-09-28. Same cell, model, prompts and loop as v1.

**Kernel (`f83740ae20`).** Each block of 256 threads (8 warps) computes a 64×128 tile.
A and W are staged in shared memory once per super-block and shared by all 8 warps.
Scales are unpacked once per column. The mma accumulator is seeded with `0x4B400000`,
so a single f32 subtract replaces the `cvt`.
- Kernel-level bench (`mma_q4k_gemm_bench_4376`, m=4096, Q8_1 quantize included):
  - n=9216 k=2560: dp4a 10.34 ms, mma 1.60 ms (120.6 TOPS).
  - n=2560 k=9216: dp4a 11.0 ms, mma 1.96 ms (98.7 TOPS).
- The correctness test still matches: |mma−dp4a| ≤ 2.5e-7 relative to max |y|.

End to end the kernel alone barely moved the leg (`speed-legs-v2.tsv`, binary `apr-4376-mmq3`):

| prompt | f16 | dp4a | mmq v1 | mmq v2 |
|---|---|---|---|---|
| p2_2k | 322 | 912 | 519 | 501 |
| p3_8k | 1,042 | 3,308 | 1,760 | 1,672 |
| p5_30k | 6,590 | 14,612 | 9,179 | 8,853 |

**Why.** The mmq arm covers Q4K only. In this Q4_K_M file the other projections fell
through to per-call f32 dequant plus SGEMM:
- `attn_qkv` ×24 and `ssm_out` ×24 are Q5_K.
- `ffn_down` ×16 and `attn_v` ×5 are Q6_K.

That is about 32% of the prefill FLOPs.

**Hybrid (`apr-4376-mmq4`).** In the mmq leg, the non-Q4K projections are now prewarmed
to f16 and run the f16 path. Q4K still runs the mma kernel. The dp4a and f16 legs are
unchanged. The prewarm is 2,192 MiB, against 6,807 MiB for the f16 default.

| prompt | tokens | f16 (default) | mmq hybrid | mmq / f16 | mmq / llama.cpp | f16 / llama.cpp |
|---|---|---|---|---|---|---|
| p2_2k | 2,154 | 313 | 366 | 0.86× | 0.43× | 0.50× |
| p3_8k | 8,119 | 1,038 | 1,178 | 0.88× | 0.49× | 0.56× |
| p5_30k | 28,924 | 6,600 | 7,049 | 0.94× | 0.32× | 0.34× |

(median of 3 ms, `speed-legs-v2-hybrid.tsv`; llama.cpp pp from `4313-rtx4090`.)

**Accuracy of the hybrid** (greedy, 64 tokens, vs f32; `parity-verdicts-v2-hybrid.txt`,
excluding apr's `Completed in` timing line):
- f16 is 5/5 identical.
- mmq is **4/5**. It was 3/5 for the all-int8 dp4a leg in `ACCURACY.md`.
- p5 diverges at byte 155 on a near-tie ("pre-compiled" vs "pre-built" Metal shaders),
  and the rest is a coherent summary.

### v2 verdict

1. The kernel is 6.5× DP4A. It is still slower than cuBLAS f16. At ~100–120 TOPS it is
   below the 4090's f16 tensor rate, far under its int8 rate.
2. The mmq leg is **0.86–0.94× of the f16 default**. It does not beat f16, so the
   ruling stands: opt-in, f16 default. It is not prefill parity. Its one current
   advantage is 4.6 GB less VRAM for prefill weights.
3. Next (v3): make the kernel compute-bound at int8 rate:
   - `ldmatrix` fragment loads, in place of scalar shared-memory reads;
   - `cp.async` double-buffering of the next super-block;
   - larger k per stage, to amortise the three barriers.

   Beating f16 needs well over 200 TOPS at kernel level.
4. Against llama.cpp, attention dominates at 29k tokens. The GEMM alone cannot reach
   0.8× there.
