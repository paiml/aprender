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

- **PR 2** ✅ **SHIPPED 2026-05-17** (#1737): fp64 accumulators in
  `Q6KGemvKernel`. **Lives in-tree at**
  `crates/aprender-gpu/src/kernels/quantize/q6k/gemv.rs` (the old
  upstream `../trueno` reference is stale — aprender-gpu has subsumed
  trueno-gpu per the monorepo consolidation). Pairs with new helper
  `add_f64_inplace` in `crates/aprender-gpu/src/ptx/builder/inplace_ops.rs`.
- **PR 3** — manual hardware verification step (not a code PR). Ran
  PR 1's falsifier on lambda-vector (RTX 4090, 2026-05-17). **Result:
  47/48 layers cos ≥ 0.99 (PASS). L47 alone fails at cos = 0.961236**
  (3σ outlier from the L40-L46 cluster). All 7 originally-cited
  problem layers (L7/L9/L12/L20/L23/L29/L46) lifted above 0.99 — PR 2
  was a real win. L47 was previously undetected because no in-tree
  per-layer falsifier existed; PR 1 closed that gap and surfaced
  L47. Full evidence: comment on #1583
  ([issuecomment-4470195446](https://github.com/paiml/aprender/issues/1583#issuecomment-4470195446)).
- **PR 3b** ✅ contract amendment `qwen3-moe-forward-gpu-v1`
  v1.7.0 → v1.7.1 capturing PR-3 outcome (PR #1739).
- **PR 3c** ✅ **this update** — scope doc reflects actual landed
  state + L47 sub-cascade.
- **PR 3d** ✅ H(i) qtype-mismatch hypothesis **FALSIFIED**.
  `apr tensors` on `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` shows
  L0, L46, L47 have **identical** tensor shapes AND qtypes (attn_q/k/o
  = Q4_K, attn_v = Q6_K, ffn_down_exps = Q6_K, ffn_gate_exps + ffn_up_exps
  = Q4_K, gate_inp + norms = F32). L47 is NOT a last-layer
  higher-precision pattern. See #1583 comment
  ([issuecomment-4470216021](https://github.com/paiml/aprender/issues/1583#issuecomment-4470216021)).
- **PR 3e** — proposed: routing-divergence falsifier. Current
  dominant hypothesis H(ii): the per-layer cosine is **accumulated**
  drift (CPU L0..L stream vs GPU L0..L stream). By L47 the
  CPU-vs-GPU hidden state has drifted by ~0.002. If at L47 that
  drift straddles a top-k expert boundary (e.g. expert 45 vs 46
  score difference < drift magnitude), CPU and GPU pick different
  expert sets and the FFN output diverges by O(1) — matching the
  0.961 cliff. The falsifier extends `SaveTensorStage::MoeRouter`
  (or adds a sibling stage) to persist the top-k EXPERT INDICES
  alongside the weights, then asserts CPU index set == GPU index
  set at L47. Multi-PR because it touches both CPU
  `forward_qwen3_moe_traced_with_plan` and CUDA
  `forward_qwen3_moe_cuda_traced_with_plan` trace plumbing.
- **PR 3f+** — fix based on PR-3e outcome:
  - If H(ii) is confirmed: route-score tie-breaking in deterministic
    expert ordering (independent of fp accumulation order); or
    fp64 in the MoE gate softmax as well; or expert routing in
    f64 with f32 conversion only post-selection.
  - If H(ii) is dead: investigate per-expert weight cancellation
    pathology at L47's specific input vector (e.g. capture FfnGate
    + FfnUp + FfnSwigl at L47 per-expert).
- **PR 4**: Part 2 of the issue — throughput ≥ 150 tok/s + VRAM ≤ 95%.
  Independent of the L47 sub-cascade; can land in parallel.
- **PR 5**: bump `qwen3-moe-forward-gpu-v1` contract v1.7.1 → v1.8.0
  ACTIVE_ALGORITHM_LEVEL → **ACTIVE_RUNTIME**, after L47 closes AND
  throughput target met.

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

## Cascade map (updated 2026-05-17)

| PR | Scope | Status | PR / evidence |
|---|---|---|---|
| 1  | Per-layer cos falsifier + this scope doc | ✅ shipped | #1713 |
| 2  | fp64 accumulators in `Q6KGemvKernel` (in-tree at `crates/aprender-gpu/src/kernels/quantize/q6k/gemv.rs`) | ✅ shipped | #1737 |
| 3  | Hardware verification on lambda-vector RTX 4090 — 47/48 PASS, L47 surfaces | ✅ ran | #1583 comment-4470195446 |
| 3b | Contract `qwen3-moe-forward-gpu-v1` v1.7.0 → v1.7.1 | ✅ shipped | #1739 |
| 3c | Scope doc update (this update) | ✅ **this PR** | — |
| 3d | H(i) qtype mismatch FALSIFIED | ✅ ran | #1583 comment-4470216021 |
| 3e | Routing-divergence falsifier (H(ii) for L47) | pending | — |
| 3f+| L47 fix based on PR-3e outcome | pending | — |
| 4  | Throughput ≥ 150 tok/s + VRAM ≤ 95% | pending (independent) | — |
| 5  | Contract v1.7.1 → v1.8.0 ACTIVE_RUNTIME | pending (gates: PR-3f+, PR-4) | — |

Note: the original cascade map said PR-2 would land in `../trueno`, but
the monorepo consolidation has subsumed trueno-gpu into in-tree
`crates/aprender-gpu`. PR-2 landed there. PR-3 was originally
"contiguous super-block chunking (if PR-2 insufficient)"; it has been
renumbered to align with the actual landed cascade — PR-3 is now the
manual hardware verification step, and the contingency chunking work
has rolled into the PR-3e / PR-3f+ slot pending the routing-divergence
falsification outcome.
