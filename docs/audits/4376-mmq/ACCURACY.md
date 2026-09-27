# #4376 MMQ int8 slice, step 1: accuracy gate for int8 activations in prefill

aprender-79, 2026-09-28. The cell is lambda-vector: RTX 4090, sm_89. The model is
`Qwen3.5-4B-Q4_K_M.gguf`, and the prompts are `docs/audits/srv-fix-002/gx10/prompts.sh`
(p1 ~30 tokens … p5 28,931).

## Why this runs first

#3513 found that int8 activations (DP4A) are catastrophic through the Gated DeltaNet
recurrence in **decode**. An MMQ kernel uses the same numerics as the existing `dp4a`
prefill leg: Q8_1 activations in 32-value blocks, Q4K weights, and an int32 accumulate
(`launch_dp4a_q4k_gemm`, `layers/cublas_prefill/gemm.rs`). Only the instruction differs
(`mma.s8` against `dp4a`). So the `dp4a` leg's output quality is the MMQ kernel's quality
ceiling, and it can be measured before any kernel is written.

## Run

- Binary: `apr 0.70.0 (90d84c86d9)`, which is #4502 `8d021f61e1` plus `ea530bd0b9`.
- Settings: greedy, 64 tokens, `APR_QWEN35_PREFILL_GEMM` ∈ {f32, f16, dp4a}, sequential
  under `/tmp/apr-gpu.lock`.
- Compared: the generated text, byte for byte, against f32.

**The mechanism engaged.** An earlier attempt on `bc433a0f30` (1c's `-main` base) had no
`APR_QWEN35_PREFILL_GEMM` in the tree, so all three legs ran f32. There, all three
measured 11.9 s at 29k tokens and all ten comparisons matched: a vacuous pass. That run
is discarded. In this run the legs separate by timing (`parity-rc.txt`); at 29k tokens:

| leg | time |
|---|---|
| f32 | 11,901 ms |
| f16 | 6,865 ms |
| dp4a | 14,711 ms |

## Result

| prompt | f16 vs f32 | dp4a (int8) vs f32 |
|---|---|---|
| p1_short | identical | identical |
| p2_2k | identical | identical |
| p3_8k | identical | identical |
| p4_16k | identical | differs at byte 55 |
| p5_30k | identical | differs at byte 146 |

**f16: 5/5 identical. int8: 3/5 identical. Both divergences are near-tie word choices,
not damage** (`divergent-outputs.txt`):

- p4 says "pre-built" (f32) where dp4a says "pre-compiled". The rest of the sentence
  matches.
- p5 reorders the same facts into a coherent summary.

## Verdict

1. #3513's decode catastrophe does **not** carry over to int8 in prefill. The outputs are
   coherent at every length up to 29k tokens.
2. int8 activations are **not greedy-exact**. They diverge from f32 on the 2 longest
   prompts, and the shipped f16 default does not diverge on any prompt.
3. An MMQ kernel can beat the f16 default only on speed. On quality it is strictly
   worse. Whether prefill may trade greedy-exactness for speed is a product call.
   Before the kernel is written, the owner has to rule on one of these:
   - int8 opt-in only, like `dp4a` today, or
   - int8 as the default, with a documented near-tie budget.

Evidence: `parity-verdicts.txt`, `parity-rc.txt`, `divergent-outputs.txt`.
