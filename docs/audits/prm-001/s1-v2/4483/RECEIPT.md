# #4483 RCA-SRV-001 fix 1: tensor-core prefill projections, PRM-S1 v2 receipt

Set `set-a4910a65.jsonl` (sha256 `a4910a65…b30632`), 200 items (50 each at 2k/8k/16k/32k), cell `gx10-cuda` (GB10).
GGUF sha256 `00fe7986…ef11a4`. Competitor: llama.cpp `d1d3c3396`, same session, same GGUF, identical prompt ids.
The rows hold timings, token counts, RSS and the parsed verdict state only. There are no prompts or completions.

**Binary:** `apr 0.70.0`, car `570b89e76` + SRV-TIM-001 + prompt-ids harness = `6cc64fe20`, built from `git archive`
(so `--version` prints `+no-git`); sha256 `04e753ed1482334f653f609c96c2fd23daecff810ef73d23a06999fb0ed15952`.
The fix is the #4313 series already on car (`4d9dbea51`..`8ea9b35be`): f16 tensor-op cuBLAS prefill GEMM, the default since `3f5f252fd`.
The f32 leg is the same binary with `APR_QWEN35_PREFILL_GEMM=f32`.
**Engaged:** the f16 log shows `[qwen35] fp16 prefill weights prewarmed: 6807 MiB`, and the f32 log shows no prewarm.

## Speed (p50 over 200 rows)

| leg | prompt_ms | prompt tps | decode tps | wall ms | prefill ratio vs llama | wall ratio vs llama |
|---|---|---|---|---|---|---|
| apr f16 (fix) | 3976 | 2093 | 31.5 | 12527 | **1.97×** | **2.12×** |
| apr f32 (same binary) | 9302 | 841 | 31.6 | 18231 | 4.61× | 3.09× |
| apr `edb076b27` (before #4313) | 9028 | 903 | 31.1 | 17683 | 4.48× | 3.05× |
| llama.cpp | 2017 | 4032 | 63.0 | 5900 | 1 | 1 |

Prefill p50 per stratum, in ms (f16 / f32 / llama):
- 2k: 439 / 1147 / 273
- 8k: 1714 / 4471 / 966
- 16k: 5679 / 13439 / 2760
- 32k: 13423 / 26017 / 5141

Result: prefill is 2.34× faster than f32 on the same binary. The gap to llama.cpp shrinks from 4.5× to 2.0×.
Decode is unchanged at 0.50× llama. That gap is the next RCA-SRV-001 item, not this fix.

## Quality falsifier (f16 vs f32, paired by `diff_sha256`)

- Parsed verdict state is identical on **200/200** items. The mix is 39 Fail / 161 Pass on both legs, so the match is not vacuous; llama.cpp gives 38/162.
- Output token count is identical on 177/200. Greedy text is therefore **not** bit-identical: f16 accumulation moves near-tie argmaxes.
  Stated tolerance: the verdict must be identical on every item. Met.
