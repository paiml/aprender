# M-GPU-MOE-3 scope note — fp-accumulator-order alignment + throughput

**Status:** PR-1 (this PR) ships the per-layer cos falsifier. PR-2+ implement
the fixes.
**Refs:** issue #1583, contract `aprender-contracts/contracts/qwen3-moe-forward-gpu-v1.yaml` v1.7.0
**Predecessors:** M-GPU-MOE-1.x cascade closed at M86 (#1530) — ACTIVE_ALGORITHM_LEVEL

This document captures the entry-point investigation for the M-GPU-MOE-3
cascade and ships the falsifier that makes "L7 cos < 0.99" reproducible
against any commit on `main`.

## What the issue says vs. what's actually true

**Stated**: 7-8 layers (L7, L9, L12, L20, L23, L29, L46) at cos 0.94-0.987
between CPU LAZY-FUSED-MATVEC and GPU `q6k_gemv` warp-shuffle. Need ≥ 0.99
and throughput ≥ 150 tok/s on RTX 4090.

**Verified during scoping:**
- M86 (#1530) closed the L6 `moe_ffn_out` NaN — confirmed on main HEAD.
- The end-to-end final-logits cos test exists
  (`crates/aprender-serve/tests/qwen3_moe_gpu_parity.rs::falsify_qw3_moe_gpu_parity_001_cosine_vs_cpu`,
  `#[ignore]`, requires GGUF + RTX 4090) but it only asserts on the final
  logits — it can't isolate which layer first drops below 0.99.
- **Per-layer GPU traced forward DOES exist**:
  `OwnedQuantizedModelCuda::forward_qwen3_moe_cuda_traced` (and
  `_with_plan`) in
  `crates/aprender-serve/src/gguf/cuda/forward_qwen3_moe_cuda_traced.rs`,
  shipped as M-MOE-SUB-2 step (b) per
  `contracts/trace-moe-gpu-sub-stages-v1.yaml`. Wires into
  `SaveTensorPlan::MoeFfnOut` for per-layer aggregated MoE FFN output dumps.
- **The missing piece was just the falsifier itself.** This PR adds
  `crates/aprender-serve/tests/qwen3_moe_per_layer_gpu_parity.rs` —
  `falsify_qw3_moe_per_layer_001_cosine_per_layer` — which runs CPU
  traced + GPU traced through `SaveTensorPlan::MoeFfnOut`, then computes
  per-layer cos in-process. `#[ignore]`, `--features cuda` only.

## Root cause (confirmed)

Reduction-order divergence between CPU and GPU when summing the same f32
products. f32 + is non-associative.

**CPU (`fused_q6k_parallel_matvec`,
`crates/aprender-serve/src/quantize/q5k_q6k_matvec.rs:34`):**

- `rayon::par_iter` over output rows (each worker computes ONE row)
- Within a row, `fused_q6k_dot_simd`
  (`crates/aprender-serve/src/quantize/fused_q5k_q6k.rs:118`) →
  `fused_q6k_dot_avx2` with **4 independent f32 accumulators** for FMA
  latency hiding, then horizontal-reduces to a scalar at row end.

**GPU (`Q6KGemvKernel` in
`/home/noah/src/trueno/trueno-gpu/src/kernels/quantize/q6k/gemv.rs`):**

- 1 thread block per output row, 32 threads per block.
- Each thread `t` processes ALL super-blocks of its row, accumulating
  `thread_partial` over 8 positions per super-block (positions
  `{t, t+32, t+64, t+96, t+128, t+160, t+192, t+224}`).
- After the super-block loop ends, **warp-shuffle binary-tree reduction**
  across the 32 threads' `acc`s: `shfl_down(16) → +; shfl_down(8) → +;
  shfl_down(4) → +; shfl_down(2) → +; shfl_down(1) → +`.
- Thread 0 writes the final f32.

Different chunking + different reduction tree = different rounding.

## Fix space (ranked by single-PR feasibility)

1. **fp64 accumulator on GPU** (simplest, biggest immediate gain).
   Replace `thread_partial` and `acc` f32 types with f64 in
   `Q6KGemvKernel::build_ptx`. Cost: ~2× register pressure on the
   accumulator, no extra memory traffic, no extra fma latency. Expected
   to push the worst-case cos from 0.94 → ≥ 0.995 without touching
   thread-mapping. **Recommended as PR 1.**

2. **Kahan/Neumaier summation at the warp-shuffle step.** Higher
   precision than fp64 acc; ~5 extra fma per reduction step. Implement
   only if (1) is insufficient.

3. **Contiguous super-block ranges per thread** instead of "all threads
   process all super-blocks at interleaved positions". Brings GPU's
   intra-row order closer to CPU's left-to-right. Larger PTX rewrite;
   needs benchmark to confirm no throughput regression.

4. **Match CPU's 4-accumulator pattern on GPU** — give each thread 4 sub-
   accumulators, reduce horizontally to thread_partial at row end. More
   speculative; likely only relevant if (1)+(2) still don't reach 0.99.

## PR-1 (this PR) — per-layer cos falsifier

`OwnedQuantizedModelCuda::forward_qwen3_moe_cuda_traced` already exists
(M-MOE-SUB-2 step (b)); the missing piece was just the falsifier. This
PR adds `crates/aprender-serve/tests/qwen3_moe_per_layer_gpu_parity.rs`
(`falsify_qw3_moe_per_layer_001_cosine_per_layer`):

- `#[ignore]` + `#![cfg(feature = "cuda")]` — requires cached
  `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` and an RTX 4090 (sm_89).
- Constructs two `SaveTensorPlan`s capturing `MoeFfnOut` for all 48
  layers, one for CPU traced, one for GPU traced.
- Runs `OwnedQuantizedModel::forward_qwen3_moe_traced_with_plan` then
  `OwnedQuantizedModelCuda::forward_qwen3_moe_cuda_traced_with_plan` on
  a 1-token prompt (`[785]`) to bound runtime.
- Reads back the `.bin` files, computes per-layer cosine in f64 to avoid
  cosine-of-cosine-divergence introducing test noise.
- Prints the full 48-element cos vector for diagnosis even on failure.
- Asserts every layer ≥ 0.99.

**Expected outcome of PR-1**: the test FAILS on current `main` (per
issue #1583 the cos for several layers is between 0.94 and 0.987). The
diagnostic output then identifies the actual divergent layers — the
issue body's enumeration ("L7, L9, L12, L20, L23, L29, L46") becomes a
verified-on-main baseline or gets corrected.

How to run once landed:

```
cargo test --release --features cuda \
  -p aprender-serve --test qwen3_moe_per_layer_gpu_parity \
  -- --ignored --nocapture
```

## PR 2+ scope

After PR 1's diagnostic is in place:

- **PR 2**: fp64 accumulator in `Q6KGemvKernel` (upstream `../trueno`).
  Re-run PR 1's test; expect cos floor → ≥ 0.99 across all layers.
- **PR 3**: if (PR 2) insufficient, contiguous super-block chunking.
- **PR 4**: Part 2 of the issue — throughput ≥ 150 tok/s + VRAM ≤ 95%.
  Likely combined with the kernel changes from PR 2/3.
- **PR 5**: bump `qwen3-moe-forward-gpu-v1` contract v1.7.0 → v1.8.0
  ACTIVE_ALGORITHM_LEVEL → **ACTIVE_RUNTIME**.

## Compute lane

- **Lambda-labs** (this host, RTX 4090 24GB sm_89) — runs ALL of the
  above. Cached GGUFs in `/home/noah/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
  (18 GB) and `/home/noah/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/`.
- **gx10** (Blackwell sm_121) — NOT needed for this work; the kernel is
  sm_89-compatible per existing M85 evidence.
- **yoga** (RTX 4060 8GB) — insufficient VRAM for the 17.3 GB Q4_K_M
  GGUF; do not use.

## What's NOT in scope for the cascade

- M-GPU-MOE-2.x wgpu path — blocked on upstream trueno-gpu wgpu kernel
  authoring (issue #1582), not unblocked by anything in M-GPU-MOE-3.
- Heterogeneous distributed training (#393) — orthogonal multi-node
  coordinator work.

## Cascade map

| PR | Scope | Status |
|---|---|---|
| 1 | Per-layer cos falsifier + this scope doc | **this PR** |
| 2 | fp64 acc in upstream `Q6KGemvKernel` (`../trueno`) | pending |
| 3 | Contiguous super-block chunking (if PR-2 insufficient) | pending |
| 4 | Throughput ≥ 150 tok/s + VRAM ≤ 95% | pending |
| 5 | Contract `qwen3-moe-forward-gpu-v1` v1.7.0 → v1.8.0 ACTIVE_RUNTIME | pending |

After PR-1 lands, PR-2 (fp64 acc) can be done independently in
`../trueno/trueno-gpu/src/kernels/quantize/q6k/gemv.rs`, with this PR-1
falsifier as the regression gate.
