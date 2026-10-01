---
spike: 020
idea: llm-decision-classifier
name: qwen35-batched-prefill
type: standard
validates: "Given upstream's Qwen3.5 CPU forward runs token at a time (5–7 s per Kev decision), when a batched prefill runs each layer's projections as one GEMM over the row and only the mixers per token, then Kev probabilities keep spike-017 parity and a short decision drops below 1 s on 6 threads"
verdict: VALIDATED
related: [017, 016]
tags: [qwen3.5, deltanet, prefill, gemm, performance, parity]
---

# Spike 020: Batched Qwen3.5 prefill

## What This Validates
Kev never generates, so its whole cost is prefill. Upstream's own comment in `forward_qwen35.rs`: *"there is no
batched prefill for the hybrid"*. This spike builds one and measures it against spike 017's torch oracle and against
the token-at-a-time path on the same build.

## Research
- Existing F32 "batched" path (`gguf/inference/matmul_fused.rs::fused_matmul_f32`) is a per-row matvec loop, not a
  GEMM, so the prefill calls `trueno::blis` directly.
- GGUF stores W as `[out, in]` row-major, which is exactly the A operand of `C^T = W · X^T`: no weight is copied
  or transposed, only the small activation matrix; `bytemuck::try_cast_slice` borrows the F32 bytes (the workspace
  forbids `unsafe`).
- `delta_rule_recurrence_gqa` loops over value heads independently (key head `h % num_k_heads`), so one-head calls
  of the same function are the same arithmetic in the same order — parallel over heads without changing a bit.

## How to Run
```bash
# needs the spike-016 worktree with upstream-qwen35-batched-prefill.patch (commit 32103ba83 on spike/016-upstream-sync)
CARGO_TARGET_DIR=<repo>/target cargo build --release
M=../017-kev-rust-forward-parity
RAYON_NUM_THREADS=6 <repo>/target/release/qwen35-batched-prefill $M/models/kev-0.8b-merged-f32.gguf \
  $M/models/kev-0.8b-head.safetensors $M/fixtures/kev-0.8b_fixture.json            # parity + control
PHASES=1 RAYON_NUM_THREADS=6 <same> --no-control                                     # phase breakdown on stderr
```

## Investigation Trail
1. **v1 — GEMM projections, everything else as before**: 7–13× over token-at-a-time (85 tokens 6.7 s → 0.73 s,
   exactly the spike-017 GEMM bound). Parity unchanged (1.3e-6). But **6 and 14 threads were equally fast**.
2. **`sample` profile**: 61 % of samples were idle workers (`__psynch_cvwait`), 24 % GEMM, 4.7 % recurrence.
3. **Hypothesis: trueno's thread cap.** `gemm_m_partitions` caps a GEMM below 512 MFLOP at 4 threads — tiers measured
   on a Threadripper (cross-CCD L3), not on Apple or Graviton. Banding W's output rows across the pool myself:
   719 → 639 ms. **Real but small; not the main limit.**
4. **Phase timers** (thread-local, spike-only) settled it at 6 threads, 87 tokens: GEMM 272 ms, strided transposes
   102 ms, **DeltaNet per-token loop 288 ms on ONE thread**, attention 5 ms. The recurrence hid in the profile because
   it runs while five threads wait.
5. **v2 — three-stage DeltaNet** (conv + L2 norm + gates for the whole row sequentially; the recurrence **per head in
   parallel**; gated norm parallel over tokens) **plus 32×32 blocked parallel transposes**. Same arithmetic.
6. **First-row outlier**: the first decision after load once took 2.7 s against ~360 ms for the rest — first touch of
   the mmapped weights. Cold start is spike 019's to measure.

## Results

**Parity (Kev-0.8B, f32 GGUF, 12 rows)**: prefill vs token-at-a-time max |Δh| 1.5e-4 (GEMM summation order);
**vs torch fp32: worst |Δp| 1.3e-6, argmax 12/12** — the same figures as v1 and as spike 017.

**Latency, M4 Pro (`RUN-SCALING.md`)**

| row | token-at-a-time (6 thr) | prefill 1 thr | **prefill 6 thr** | prefill 14 thr |
|---|---|---|---|---|
| 37 tokens | 2.9 s | 0.81 s | **0.20 s** | 0.20 s |
| 87 tokens | 6.6 s | 1.69 s | **0.36–0.38 s** | 0.34 s |
| 915 tokens | 73 s | 17.4 s | **3.9 s** | 3.0 s |

**Where 6 threads go (87 tokens)**: GEMM 252 ms (~340 GFLOP/s), DeltaNet 91 ms, transposes 20 ms, attention 5 ms.
At 915 tokens: GEMM 2.36 s, DeltaNet 0.96 s, attention 0.38 s (L² grows).

**Verdict: VALIDATED ✓.** A batched prefill makes Rust Kev-0.8B a **~0.36 s decision on 6 CPU threads** (18× the
upstream path), with parity untouched. It is ~280 lines on top of upstream's own kernels and is useful to upstream
beyond Kev (time-to-first-token for `apr run`/`apr chat` on the hybrid, which today prefills token by token).

**Limits and open work**
- **F32 only**: BF16/F16/Q8_0 weights fall back to `fused_matmul` (per-row matvec). A BF16 GEMM is required for
  Kev-4B, whose F32 weights (~16 GB) exceed Lambda's 10 GB.
- Kev-4B is ~7× the matmul work of 0.8B (≈3.5B block params): expect ~2.5 s per short decision on 6 threads.
- trueno's FLOP-tiered thread caps are x86-measured; the banded split here sidesteps them for this call site only.
- The state-prefix cache (reuse the state's recurrent/KV state across a request's questions) is not built; a
  3-question request pays the state 3 times.
- Phase timers are spike instrumentation, not for merge.
