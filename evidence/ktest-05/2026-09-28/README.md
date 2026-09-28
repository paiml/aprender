# KTEST-05 receipts, 2026-09-28 (aprender-6b)

Binary on both hosts: `apr 0.70.0 (8d021f61e)`, the fleet install copied to a pinned path (sha256 prefix in each `version.txt`).
Model: qwen2.5-coder-0.5b-instruct q4_k_m, `BATCHED_PREFILL=1`, `--max-tokens 2` (sanitizer) / `4` (F-4).

| host | F-5 racecheck (barrier / no barrier) | F-4 JIT cubins | F-4 planted | memcheck | racecheck (filtered) | initcheck | synccheck |
|---|---|---|---|---|---|---|---|
| gx10 GB10 sm_121 | 0 / 6 PASS | 39, all sm_121 PASS | expect sm_89 → RED | 0 | 0 | 0 | 0 |
| lambda RTX 4090 sm_89 | 0 / 7 PASS | 42, all sm_89 PASS | expect sm_121 → RED | 0 | 0 | **12,593 + 1,486,112 overflowed** (advisory) | 0 |

- `lambda/san/`: the first run with the kernel filter on every tool. Its initcheck timed out (rc 124, no summary) after 50 reports at `rope_neox_indirect` `LDG [x]`. Those are a filter artifact: `scripts/ktest/fixtures/initcheck_filter_artifact.cu` gives 0 errors unfiltered and 32 with `--kernel-name regex=rope`.
- `lambda/san2/`, `lambda/san3/`: the unfiltered initcheck. Every printed report is a 16-byte read in cuBLASLt's `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_…` (the FP8 prefill GEMM). This is untriaged and handed to the cop for intake. gx10, which does not take that kernel, is clean.
