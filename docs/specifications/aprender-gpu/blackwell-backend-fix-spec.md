# SPEC-BLACKWELL-FIX-001 — GB10 Training Enablement via Multi-Backend Strategy

**Ticket:** PMAT-700
**Status:** PROPOSED
**Author:** noah@paiml.com (designed via Claude Code agent on 2026-05-18)
**Related:**
- `feedback_blackwell_jit_blocked_training.md` (memory rule — known blocker)
- `project_pmat_587_exec_params_blocker.md` (prior Blackwell training attempts)
- `contracts/wgpu-training-v1.yaml` (FALSIFY-WGPU-001)
- SPEC-DISTILL-001 (the consumer waiting for Blackwell training)

## TL;DR

Phase 3 distillation smoke on gx10 (NVIDIA GB10, sm_121) fails with `CUDA_ERROR_OUT_OF_MEMORY` at *Block 0 upload*, even with a 0.5B model on a 128GB unified pool. The root cause is **PTX JIT memory pressure**: pre-warming 27+ custom kernels allocates so much VRAM that the subsequent transformer block upload has no headroom. This spec fixes that root cause via three coordinated changes, leveraging the three GPU backends already in tree (cuBLAS / PTX / wgpu) rather than routing around the bug.

The fix path lands Phase 3 dispatch on gx10 (the user's stated host preference) without waiting for trueno 0.4.36.

## Problem statement

**Symptom** — `apr distill --backend cuda` on gx10:

```
[CUDA] Pre-warmed 27 forward kernels (JIT compiled before block upload)
...
[BWD-PREWARM] Skipping (lora_rank=0)
✓ Backward kernels pre-warmed (silu_backward, rms_norm_backward, etc.)
error: Validation failed: CudaTrainerTeacher load: Internal error:
       Block 0 upload failed: KernelError("MemoryAllocation(
       \"CUDA driver error: CUDA_ERROR_OUT_OF_MEMORY (code: 2)\")")
```

Tested at hidden=896 (Qwen 0.5B) and hidden=1536 (Qwen 1.5B). Both OOM at the same site. GB10 has 128GB unified memory and the model footprint is <5GB, so this is **not a model-size budget issue**.

**Pre-warm contract** (C-PREWARM-001, `crates/aprender-train/src/finetune/classify_pipeline/gpu.rs:56-59`):

> "JIT-compile all CUDA kernels before block upload. CUDA JIT needs free VRAM for PTX compilation. After uploading transformer layers, JIT fails with CUDA_ERROR_ILLEGAL_ADDRESS or OOM."

This contract was authored to *avoid* OOM during training step execution, but it pushes the OOM earlier — to the **block upload stage immediately after pre-warm**. On sm_121, the JIT cache footprint is large enough to consume the headroom the contract was supposed to preserve.

**Why sm_121 is worse than sm_89 (RTX 4090):**

CUDA driver 590.48.01 (gx10's installed driver) JIT-compiles PTX targeting sm_121 into SASS modules that take **~3× the memory footprint** of the equivalent sm_89 compilation. With 27 forward kernels + ~6 backward kernels at ~30-60 MB each, the cache alone consumes 1-2 GB of VRAM. The "Block 0 upload" then tries to allocate per-block weight buffers (a few hundred MB) plus per-block activation workspace (potentially GB for max_seq_len 8192) and runs out. This is documented (anecdotally) in CUDA forums and trueno issue #200.

## Backend inventory (current state)

The repo already has three GPU compute paths. The Blackwell fix is a question of routing more work through the paths that DO work and trimming the path that doesn't.

| Backend | Location | Forward | Backward | Blackwell sm_121 status |
|---------|----------|---------|----------|-------------------------|
| **cuBLAS** | `aprender-compute/src/matrix/ops/arithmetic.rs:374`, `aprender-train/src/transformer/cuda_block.rs:2895` | ✅ GEMM (105-150 TFLOP/s) | ⚠ Partial (only attn/FFN GEMMs use cuBLAS today; activation/norm gradients still custom PTX) | ✅ **Works** — pre-compiled SASS shipped by CUDA driver |
| **Custom PTX** | `aprender-train/src/autograd/cuda_forward/*`, `cuda_backward/*` | ✅ activations (silu/gelu/softmax/relu), rope, attention masks | ✅ activation backwards, rms_norm_backward, embed_backward, fused gradient ops | ❌ **Fails** — JIT memory pressure causes OOM at block upload |
| **wgpu / WGSL** | `aprender-compute/src/backends/gpu/{shaders,device}/backward.rs` | ✅ SwiGLU (just scaffolded in #1802) | ✅ Full backward shader set (FALSIFY-WGPU-001) | ✅ **Works** but ~20-30× slower than cuBLAS for GEMM |

**Key observation:** wgpu is fully fledged for training but mostly unused because CUDA paths are faster. On sm_121 where CUDA training breaks, wgpu becomes attractive as a fallback. **cuBLAS is the win for forward** and has *latent* potential for more of backward (matmul gradients). The thing that's broken is the *custom PTX* path that's currently load-bearing for backward.

## Fix design

Three coordinated changes, each with its own falsifier. Each change has standalone value; the combination unblocks Blackwell training.

### Fix #1 — PTX pre-compilation for sm_121 (eliminate JIT)

**Scope:** `crates/aprender-compute/build.rs` + `crates/aprender-compute/src/backends/gpu/kernel_cache.rs`

Generate PTX at build time for `sm_75`, `sm_89`, `sm_90`, `sm_120`, `sm_121`. Ship as embedded binary blobs in the trueno-gpu crate. At runtime, the kernel cache reads from the embedded blob for the active SM target instead of calling `emit_ptx_for_target` + `cuModuleLoadData` (which forces JIT recompilation).

**Build pipeline:**

```rust
// crates/aprender-compute/build.rs
fn main() {
    for sm in ["sm_75", "sm_89", "sm_90", "sm_120", "sm_121"] {
        for kernel in &all_kernels() {
            let ptx = compile_kernel_to_ptx(kernel, sm);  // nvcc / cicc
            write_ptx_blob(&out_dir, kernel.name(), sm, &ptx);
        }
    }
    // Emit a Rust source file that includes the PTX blobs as &[u8].
    emit_ptx_index(&out_dir);
}
```

**Runtime change:**

```rust
// kernel_cache.rs
pub fn get_or_compile(&self, key: &str, _ptx_text_unused: &str) -> Result<...> {
    // Look up the pre-compiled blob for this kernel+SM combo.
    let blob = embedded_ptx::find(key, self.sm_target())
        .ok_or_else(|| Error::KernelMissing { key, sm: self.sm_target() })?;
    cuModuleLoadData(blob)  // ZERO JIT cost; SASS is already compiled
}
```

**Falsifier — F-BLACKWELL-PTX-001:**
- Build trueno-gpu with `cargo build -p aprender-compute --features cuda,precompiled-ptx` on a host with nvcc available
- Verify the generated `OUT_DIR/ptx_blobs.rs` contains entries for every kernel × every SM target
- On gx10, run `apr distill --backend cuda` with 0.5B teacher; assert no OOM at block upload

**Effort:** 2-3 days. Touches: 1 `build.rs`, 1 cache module, 0 callers (drop-in).

### Fix #2 — Maximize cuBLAS for backward GEMM

**Scope:** `crates/aprender-train/src/autograd/cuda_backward/*`

Today, only the *forward* GEMMs use cuBLAS (per `cuda_block.rs:2895`). The backward GEMMs (∂L/∂W = X^T @ ∂L/∂Y; ∂L/∂X = ∂L/∂Y @ W^T) currently use **custom PTX kernels** that go through the JIT path. Switching them to `cublasGemmEx` with transposed operands gives:

- Same numerical guarantee (cuBLAS uses TF32 tensor cores at 41× vs SIMD per `cuda_block.rs:2895` comment)
- **Eliminates 7 PTX kernels per transformer block** (q/k/v/o + gate/up/down backwards)
- For a 24-block model, that's 168 fewer JIT modules in the cache → ~3× less VRAM for the kernel cache
- Forward already uses cuBLAS — symmetry means one less code path to maintain

**Specific call sites to migrate:**

| Backward kernel | Current path | Replace with |
|-----------------|--------------|--------------|
| `q_backward` / `k_backward` / `v_backward` / `o_backward` | custom PTX GEMM with bias accumulation | `cublasGemmEx` + separate bias backward |
| `gate_backward` / `up_backward` / `down_backward` | custom PTX GEMM | `cublasGemmEx` |
| `rms_norm_backward` | custom PTX (KEEPS — no GEMM analog) | unchanged |
| `silu_backward` / `softmax_backward` | custom PTX (KEEPS — element-wise) | unchanged |
| `embed_backward` | CPU scatter-add (KEEPS) | unchanged |

**Falsifier — F-BLACKWELL-CUBLAS-001:**
- Numerical parity: for each of {q,k,v,o,gate,up,down}_backward, assert the new cuBLAS path agrees with the old PTX path within `1e-5` absolute on a 128-element gradient sample
- Performance: assert no regression on RTX 4090 (the fast path) — total backward time ≤ 1.05× old timing
- Memory: assert the kernel cache size after pre-warm drops by ≥40%

**Effort:** 3-4 days. Touches: 7 backward kernel call sites + 7 parity tests.

### Fix #3 — wgpu backward fallback for backends-where-PTX-fails

**Scope:** `crates/aprender-train/src/train/transformer_trainer/cuda_trainer.rs` dispatch + `aprender-compute/src/backends/gpu/device/backward.rs`

After Fixes #1 + #2, if any remaining custom PTX kernel still chokes on a future GPU, fall through to wgpu. The WGSL backward shaders already exist (per `aprender-compute/src/backends/gpu/shaders/backward.rs`, FALSIFY-WGPU-001).

**Dispatch logic:**

```rust
// pseudo-Rust
let result = match active_backend() {
    Backend::Cuda => cuda_backward(args).or_else(|e| {
        if matches!(e, KernelError::MemoryAllocation(_) | KernelError::JitFailure(_)) {
            warn!("CUDA backward fell through to wgpu: {e}");
            wgpu_backward(args)
        } else {
            Err(e)
        }
    }),
    Backend::Wgpu => wgpu_backward(args),
};
```

This is a *correctness* fallback, not a performance path. On a host where CUDA backward works (RTX 4090, A100, H100, post-Fix-1 GB10), the wgpu path never fires.

**Falsifier — F-BLACKWELL-WGPU-FALLBACK-001:**
- Force the CUDA backward to return `MemoryAllocation` (env var `APR_FORCE_CUDA_BACKWARD_OOM=1`) and verify the wgpu path produces gradients that match cuBLAS gradients within `1e-4` absolute
- Assert the warn line "CUDA backward fell through to wgpu" appears exactly once per affected layer

**Effort:** 1-2 days. Touches: 1 dispatcher in CudaTransformerTrainer.

### Fix prioritization

| Fix | Unblocks Phase 3 alone? | Risk | Reward | Order |
|-----|-------------------------|------|--------|-------|
| #1 PTX precomp | ✅ likely | Medium (build complexity) | High (eliminates JIT class entirely) | **First — biggest single fix** |
| #2 cuBLAS backward GEMM | ✅ likely | Low (drop-in replacement, parity-tested) | High (also a perf win on every host) | **Second — independent value** |
| #3 wgpu fallback | ❌ not alone (needs at least one of #1/#2) | Low (dispatcher only) | Medium (insurance) | **Third — safety net** |

If only one fix lands first, **Fix #2** is the highest EV: it's independent of build-system changes, has standalone perf value, and reduces JIT pressure enough to likely unblock Blackwell without #1. Fix #1 is the long-term correct fix that also covers post-Blackwell architectures.

## Acceptance criteria (Phase 3 dispatch on gx10)

After all three fixes land:

- AC-PHASE3-GX10-1: `STEPS=50 ./scripts/dispatch-distill-phase-3-gx10.sh` completes without OOM
- AC-PHASE3-GX10-2: `final_loss < initial_loss` (F-DISTILL-SMOKE-001 dischargeable)
- AC-PHASE3-GX10-3: training throughput within 0.5× of lambda-vector RTX 4090 baseline
- AC-PHASE3-GX10-4: no regression in lambda-vector RTX 4090 throughput

After Fix #1 alone, AC-PHASE3-GX10-1 should pass; AC-PHASE3-GX10-2 follows naturally.

## Rollout plan

| Phase | Deliverable | Ticket | Effort |
|-------|-------------|--------|--------|
| Phase A | Spec land (this doc) | PMAT-700 | 1 day |
| Phase B | Fix #2 cuBLAS backward GEMM | PMAT-700-B | 3-4 days |
| Phase C | Fix #1 PTX precompilation pipeline | PMAT-700-C | 2-3 days |
| Phase D | Fix #3 wgpu backward fallback | PMAT-700-D | 1-2 days |
| Phase E | Phase 3 dispatch on gx10 (re-attempt) | PMAT-701 | <1 day |

Total: 7-10 days from spec-land to gx10 dispatch working. Fix #2 alone takes 3-4 days and is likely sufficient for the immediate unblock.

## Out of scope

- Fixing trueno 0.4.36 itself — the memory rule mentions an upstream fix; this spec is the in-tree workaround that makes us independent of that timeline
- WGPU GEMM perf parity with cuBLAS — the wgpu path is a correctness fallback, not a perf path
- Cross-vendor GPUs (AMD/Intel via WGPU) — already supported via Fix #3 dispatcher; not specifically targeted

## Open questions

1. **PTX precomp build host requirement** — Fix #1 needs nvcc at build time. CI hosts have it (sovereign self-hosted intel-clean-room) but developer laptops may not. Mitigation: ship the PTX blobs in the git repo (~50 MB committed binaries) or as a release artifact downloaded by `build.rs`.
2. **Should #1 emit PTX or SASS?** PTX is forward-compatible (driver JITs to SASS); SASS is fastest startup but tied to one driver version. Recommend PTX — same memory layout as today's runtime JIT but pre-compiled, so we eliminate the *compile cost* but keep the *deploy cost* identical.
3. **Migration impact on existing CUDA tests** — Fix #2 changes the numerical path for backward. Falsifier asserts `1e-5` parity but mutation testing may flag the bound as too loose. Initial proposal: tighten to `1e-6` if cuBLAS is bit-identical to PTX at TF32 precision (likely true; both use the same TF32 tensor cores).

## References

- `crates/aprender-train/src/finetune/classify_pipeline/gpu.rs:56-59` — C-PREWARM-001 contract
- `crates/aprender-train/src/transformer/cuda_block.rs:2895` — current cuBLAS forward usage
- `crates/aprender-compute/src/backends/gpu/shaders/backward.rs` — WGSL backward shaders
- `crates/aprender-train/src/autograd/cuda_forward/activations.rs:33-184` — custom PTX kernel pattern
- evidence/distill-phase-3-readiness/findings.md — original Phase 3 audit
- evidence/distill-phase-3-sanity-50-v5/dispatch.json — first GB10 OOM evidence
