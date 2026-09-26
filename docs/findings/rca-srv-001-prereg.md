# RCA-SRV-001 pre-registration (REX protocol, docs/specifications/review-experiment-protocol.md)

Written 2026-09-26 ~15:30Z, BEFORE any A/B measurement below was taken. Only the S1 rows
(gx10, 33b5e5b24, set a4910a65) and 60's apr-run nsys (4090, 0.69.5-rc.1) had been read.

Subjects: A = apr 33b5e5b24 (the S1 binary, no split decode attention);
B = A + cherry-pick of da3817741 and 5660b0877 (#4273 split decode attention) and nothing else;
L = llama.cpp d1d3c3396 llama-server. Same GGUF (Qwen3.5-4B-Q4_K_M, sha 00fe7986), 4090,
`--context-length 36864`, same request bodies (set a4910a65 items, temp 0, seed 4354, thinking off).
Input varied: 5 items per stratum (2k/8k/16k/32k) = 20 prompts, plus nsys at ~4k and ~22k ctx.

| id | hypothesis | measurement | falsifier (hypothesis REJECTED if) |
|----|------------|-------------|-------------------------------------|
| H1 | The context-proportional decode cost (S1 gx10 slope 5.205 ms/tok per 1k ctx vs llama 0.149) is the unsplit `gdn_decode_attention` | per-row OLS of decode ms/tok on ctx, A vs B, 20 prompts | B's slope is > 50% of A's slope |
| H2 | At short ctx apr decode GPU kernel time/token exceeds llama's by >= 1.5x, led by the quantized GEMVs and `gdn_delta_rule_recurrence` | nsys kern_sum, (max_tokens 129) - (max_tokens 1), per token | kernel ratio apr/llama < 1.3x, or GEMV+recurrence < 50% of the excess |
| H3 | apr decode leaves the GPU idle for a large share of each token (host-side: no CUDA graph by default, full-vocab logits DtoH + CPU sampling per token); llama does not | (decode wall/token) - (GPU kernel time/token) | apr idle share < 15% of its decode wall |
| H4 | apr prefill GPU time is dominated by per-request weight dequant to f32 + f32 SGEMM (no tensor cores, no dequant cache) | nsys kern_sum of max_tokens=1 run minus model-load kernels | dequant + f32 GEMM kernels < 40% of apr prefill kernel time |

A rejected hypothesis is reported as rejected; nothing is re-fitted after the fact.
