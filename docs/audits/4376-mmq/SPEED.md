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
