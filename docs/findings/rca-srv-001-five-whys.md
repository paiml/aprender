# RCA-SRV-001: five whys, apr serve 4.4x slower than llama.cpp (Qwen3.5-4B, GPU)

Taken over from 5c (apr-rca) by aprender-76 on 2026-09-26. The hypotheses are 5c's H1-H4 in
`rca-srv-001-prereg.md`, registered ~15:30Z before any A/B below existed. Each one is scored
against its falsifier as written. Nothing was re-fitted.

## Subjects and method

| tag | binary | what it is |
|-----|--------|------------|
| A (S1-equivalent) | `apr 0.70.0 (69e12aea2)` | car before the #4304 fold. Same GDN attention as S1 `33b5e5b24`: `git ls-tree 33b5e5b24 crates/aprender-gpu/src/kernels/gdn/` has no `decode_attention_split.rs` |
| B | `apr 0.70.0 (1ac56f258)` | A + fold of #4304 (split decode attention, #4273) + 2 doc comments. Car head `11bcf6c2b` differs only by a rustfmt of one struct literal (`git diff -w`), so B stands for the current car |
| L | llama.cpp `llama-bench` `60b06ab9a` (build 11050) | `-ngl 99 -r 1`, flash-attn auto |

- **Hardware and model:** 4090 (sm_89), `Qwen3.5-4B-Q4_K_M.gguf`.
- **Prompts:** `mid` = 2241 tokens and `long` = 29991 tokens. L runs `-p <n> -n 0` for prefill and `-p 0 -n 128 -d <n>` for decode at the same depth.
- **Profiler:** nsys `cuda_gpu_kern_sum`.
- **apr decode per token:** Σ over kernels of (instances(n=129) − instances(n=1)) × avg(n=129) / 128. The plain (total₁₂₉ − total₁)/128 is contaminated: in B-long the n=129 run's prefill was 27% slower (16.5 s vs 13.0 s, identical instance counts) and leaked 25 ms/tok into "decode".
- **llama prefill:** kernel sums are halved. llama-bench runs one warmup: `gated_delta_net` instances are 240 = 2 × 24 GDN layers × 5 ubatches.
- **llama decode:** compared on wall (`avg_ts`) only. It runs under CUDA graphs, so nsys kern_sum undercounts its kernels.
- **Artifacts:** everything is in `/mnt/nvme-raid0/tmp/ont-gdn/` (`nsys_{car69e,car1ac,llama}_*`, `attn_*`).

## The chain

**WHY 1: why is apr 4.4x slower in S1 on both prefill AND decode?**
There are two independent causes, and the uniform ratio is a coincidence. Decode moves 2x when
#4273 is added (A→B: 20.21 → 10.45 ms/tok at 2.2k ctx). Prefill does not move (A 947 ms vs B 854 ms
kernel time at 2241 tokens, within run-to-run spread of 760–897 ms wall). Prefill runs 8–16
decode-attention launches, so #4273 cannot reach it.

**WHY 2a (decode): why is decode slow?**
S1's binary has no split decode attention (#4273). The unsplit `gdn_decode_attention` costs:

- 7.99 ms/tok at 2.2k ctx, which is 40% of A's decode;
- 127.75 ms/tok at 30k ctx, which is 91% of A's decode.

With the split kernel it costs 0.38 and 2.26 ms/tok. The per-token decode cost grows with context
at 4.31 ms per 1k tokens in A and 0.155 in B, so B's slope is 3.6% of A's. **H1: NOT rejected**
(falsifier: B slope > 50% of A). The residual gap against llama wall is 1.37x at 2.2k (10.45 vs
7.60) and 1.82x at 30k (14.75 vs 8.09). This is 2 ctx points, not 5c's 20-prompt OLS. fb's
in-flight gx10 PRM-S1 run on car `1ac56f258` (set a4910a65) is the full-set replica.

**WHY 2b (prefill): why is prefill slow?**
apr uses 5.1x llama's GPU kernel time at both lengths:

- mid: 854 vs 168 ms
- long: 12812 vs 2523 ms

The gap is almost entirely matmul-shaped (per-run kernel ms):

| prompt | category | apr B | llama | share of gap |
|--------|----------|------:|------:|-------------:|
| mid 2241 | weight GEMM (+dequant / act-quant) | 641 | 103 | **78.4%** |
| mid 2241 | attention | 36 | 3 | 4.7% |
| mid 2241 | GDN delta-rule scan | 66 | 35 | 4.5% |
| mid 2241 | other | 112 | 26 | 12.4% |
| long 29991 | weight GEMM (+dequant / act-quant) | 7005 | 1334 | **55.1%** |
| long 29991 | attention | 4566 | 357 | **40.9%** |
| long 29991 | GDN delta-rule scan | 744 | 489 | 2.5% |
| long 29991 | other | 498 | 343 | 1.5% |

**H4: NOT rejected.** Dequant + f32 GEMM is 65–72% (sgemm) plus 9–15% (dequant) of apr prefill
kernel time, against the falsifier's < 40%. The prefill throughput comes out the same way, which
matches S1's 4.34x prefill ratio:

- apr: 2207–2947 tok/s
- llama `avg_ts`: 10188 tok/s (mid) and 10196 tok/s (long)

**WHY 3: why is the weight GEMM 5x slower?**
apr's prefill projection, `CudaExecutor::qwen35_project_rows` (`crates/aprender-serve/src/cuda/executor/gdn_prefill_ops.rs:148`),
dequantizes every weight to f32 (`q4k_dequant_to_f32`, :109) and then calls `gemm_f32` on a
`CUBLAS_PEDANTIC_MATH` handle (no TF32). nsys shows **zero tensor-core kernels** in apr prefill
(`ampere_sgemm_*` only). llama runs the same projections as `mul_mat_q<Q4_K/Q5_K/Q6_K>`, which is
MMQ with q8_1 activations. Attention (41% of the long gap) has the same shape: apr uses f32 cuBLAS
QKᵀ/PV plus a materialized `causal_mask_softmax`, and llama uses `flash_attn_ext_f16`.

**WHY 4: why f32 with no tensor cores?**
It is deliberate. `gdn_prefill_ops.rs:7-11` says the path avoids `cublas_prefill_gemm`'s
FP8/FP16/WMMA/DP4A legs because "the hybrid feeds its projections into a recurrence, which
compounds low-precision activation error". It cites `Qwen35CudaModel::pin_float_gemv`
(`crates/aprender-serve/src/gguf/cuda/forward_qwen35_cuda.rs:845`).

**WHY 5: why does that pin apply to every low-precision GEMM?**
The pin's own evidence is **one** variant, DP4A with int8 activations. Measured on the 0.8B file,
it puts the DeltaNet layer output 1.1e-2 relative from the CPU reference, against 1e-3 for float.
The docstring's stated purpose is test isolation: "a test that means to measure the Gated
`DeltaNet` kernels, and not the GEMV quantization choice, pins this first". Two measurements were
never taken:

- the precision intermediates (FP16/BF16 tensor-core with f32 accumulate, or TF32);
- an end-to-end quality budget for the hybrid, such as next-token or top-k agreement over a prompt set.

The existing flash-attention leg (`APR_QWEN35_PREFILL_ATTENTION=flash`, f16 inputs and f32
accumulate) produces identical 32-token text on both prompts, but it is **slower**, not faster:

- long: 18.3–19.1 s against 13.6–14.7 s for f32 cuBLAS;
- mid: 0.97–1.10 s against 0.76–0.88 s.

So no fast low-precision leg exists for attention either.

## Root cause

A kernel-isolation test tolerance (DeltaNet output within 1e-3 of CPU, which DP4A missed at
1.1e-2) was generalized into the **production** precision policy for every Qwen3.5 prefill
projection. The qwen35 prefill therefore has no tensor-core GEMM path at all, and its attention
has no tensor-core flash path. Weight GEMM + attention together are 83% (mid) to 96% (long) of the 5.1x prefill
kernel gap on the 4090.

In S1, the decode half of the 4.4x came from a binary cut before split decode attention (#4273).
That is already fixed on car B: decode slope 4.31 → 0.155 ms per 1k ctx.

## Fixes, ranked by measured share of the remaining gap

| # | fix | measured share | ticket (for the cop to mint) |
|---|-----|----------------|------------------------------|
| 1 | qwen35 prefill weight GEMM on tensor cores, e.g. W4A16 fp16-activation WMMA (`fp16_tensor/w4a16_wmma_gemm.rs` exists) or an MMQ path. Gated by a pre-registered end-to-end quality falsifier, not the 1e-3 kernel-isolation bound. Keep `pin_float_gemv` for the isolation tests | 78% of mid / 55% of long prefill gap | PMAT: qwen35 tensor-core prefill projections |
| 2 | qwen35 prefill attention as a real tensor-core flash kernel. The in-tree flash leg is 1.3–1.4x SLOWER than f32 cuBLAS | 41% of long prefill gap, 5% of mid | PMAT: qwen35 flash prefill attention faster than cuBLAS f32 |
| 3 | residual decode after #4273: 1.37–1.82x vs llama wall. Top kernels are `mwv_q4k_gemv` 3.2–3.6, `q5k_gemv_warp_reduce` 2.5, `mwv_q6k_gemv` 2.1–2.5 and `gdn_delta_rule_recurrence` 1.1 ms/tok. The recurrence fix is already on `perf/gdn-recurrence-cuda` d856be9e5, awaiting 59's fold | ≈ 2.8–6.7 ms/tok of decode | PMAT: qwen35 decode GEMV + host idle (H2/H3) |

## Open (not closed by this data)

- **H2** (short-ctx decode kernel ratio ≥ 1.5x, led by GEMV + recurrence): **not scored.** llama's decode kernels run inside CUDA graphs and nsys kern_sum undercounts them. It needs `nsys --cuda-graph-trace=node`.
- **H3** (host idle share ≥ 15% of decode wall): **not scored.** `apr run` prints no decode-only wall. It needs SRV-TIM-001's per-request split from `apr serve`.
- **gx10 replica:** fb's PRM-S1 run on car `1ac56f258` and 41's queued `edb076b27` run are the gx10 replicas. GB10 bandwidth and compute differ from the 4090, so the shares above are 4090 shares.
