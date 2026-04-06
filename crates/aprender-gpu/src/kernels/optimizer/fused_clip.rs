//! Fused Gradient Clipping Kernels (ALB-078)
//!
//! Eliminates CPU↔GPU synchronization from per-block gradient clipping by keeping
//! the entire norm-reduce-clip pipeline on GPU.
//!
//! ## Current bottleneck (pre-ALB-078)
//!
//! Per-block clip requires `stream.synchronize()` to download partial sums to CPU,
//! compute the clip scale, then launch clip kernels. With 24 transformer blocks,
//! this creates 24 pipeline stalls per training step.
//!
//! ## Fused pipeline (ALB-078)
//!
//! 1. 9× `SquaredSumKernel` (existing) → write partials to contiguous GPU buffer
//! 2. 1× `ClipScaleReduceKernel` (new) → reduce all partials, compute clip scale on GPU
//! 3. 9× `GradientClipGpuScaleKernel` (new) → read scale from GPU, apply to gradients
//!
//! All kernels on same stream = zero sync points, zero D2H transfers.
//!
//! ## Contract (C-FUSEDCLIP-001)
//!
//! - **Precondition**: partials buffer contains valid squared-sum partial results
//! - **Postcondition**: `output[0] = min(1.0, max_norm / sqrt(sum(partials)))`,
//!   `output[1] = sqrt(sum(partials))` (norm for observability)
//! - **Invariant**: If `sum(partials) == 0`, `output[0] = 1.0` (no clipping)

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// GPU-side clip scale reduction kernel (ALB-078).
///
/// Reads a contiguous buffer of squared-sum partial results (from multiple
/// `SquaredSumKernel` launches), computes the global L2 norm, and writes
/// `clip_scale = min(1.0, max_norm / norm)` to GPU output buffer.
///
/// Single CTA, single thread — the partial buffer is tiny (~1800 f32 for 350M model).
/// Kernel launch overhead (~5μs) dominates; parallel reduction would add complexity
/// with no throughput benefit.
///
/// # Parameters (PTX)
///
/// - `partials_ptr` (u64): pointer to contiguous f32 partial sums
/// - `total_n` (u32): number of partial sum elements
/// - `max_norm` (f32): gradient clipping threshold
/// - `output_ptr` (u64): pointer to output (2 × f32: [clip_scale, grad_norm])
#[derive(Debug, Clone)]
pub struct ClipScaleReduceKernel;

impl Kernel for ClipScaleReduceKernel {
    fn name(&self) -> &str {
        "clip_scale_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("clip_scale_reduce")
            .param(PtxType::U64, "partials_ptr")
            .param(PtxType::U32, "total_n")
            .param(PtxType::F32, "max_norm")
            .param(PtxType::U64, "output_ptr")
            .build(|ctx| {
                let partials_ptr = ctx.load_param_u64("partials_ptr");
                let total_n = ctx.load_param_u32("total_n");
                let max_norm = ctx.load_param_f32("max_norm");
                let output_ptr = ctx.load_param_u64("output_ptr");

                let four = ctx.mov_u32_imm(4);

                // Accumulate all partial sums in f32.
                // Precision: ~1800 partials × eps_f32 ≈ 0.01% relative error.
                // For gradient clipping, this is negligible.
                let total_sq = ctx.mov_f32_imm(0.0);
                let i = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let in_bounds = ctx.setp_lt_u32(i, total_n);
                ctx.branch_if_not(in_bounds, "sum_done");

                let byte_offset = ctx.mul_wide_u32_reg(i, four);
                let addr = ctx.add_u64(partials_ptr, byte_offset);
                let val = ctx.ld_global_f32(addr);
                ctx.add_f32_inplace(total_sq, val);

                let one = ctx.mov_u32_imm(1);
                ctx.add_u32_reg_inplace(i, one);
                ctx.branch("sum_loop");
                ctx.label("sum_done");

                // norm = sqrt(total_sq)
                let norm = ctx.sqrt_f32(total_sq);

                // scale = max_norm / norm
                // IEEE 754: if norm == 0.0, div produces +inf, min(+inf, 1.0) = 1.0
                // This correctly handles the zero-gradient case without branching.
                let raw_scale = ctx.div_f32(max_norm, norm);
                // scale = min(raw_scale, 1.0)
                let one_f32 = ctx.mov_f32_imm(1.0);
                let scale = ctx.min_f32(raw_scale, one_f32);

                // Write output[0] = scale, output[1] = norm
                ctx.st_global_f32(output_ptr, scale);
                let four_u64 = ctx.mov_u64_imm(4);
                let norm_addr = ctx.add_u64(output_ptr, four_u64);
                ctx.st_global_f32(norm_addr, norm);

                ctx.ret();
            })
    }
}

/// GPU-side gradient clipping with scale read from device memory (ALB-078).
///
/// Like [`GradientClipKernel`] but reads the scale factor from a GPU pointer
/// instead of a host-provided f32 parameter. This eliminates the D2H transfer
/// of the clip scale, keeping the entire clip pipeline on GPU.
///
/// The scale pointer is shared across all 9 gradient buffer clip launches —
/// L1 cache ensures the read is effectively free after the first warp.
///
/// If `scale ≈ 1.0` (within 1e-7), the kernel exits without writing,
/// avoiding unnecessary memory bandwidth when no clipping is needed.
///
/// # Parameters (PTX)
///
/// - `grads_ptr` (u64): gradient tensor (updated in-place)
/// - `scale_ptr` (u64): pointer to f32 clip scale (on GPU)
/// - `n` (u32): number of gradient elements
#[derive(Debug, Clone)]
pub struct GradientClipGpuScaleKernel {
    /// Number of gradient elements
    pub n: u32,
}

impl GradientClipGpuScaleKernel {
    /// Create a new GPU-scale gradient clipping kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for GradientClipGpuScaleKernel {
    fn name(&self) -> &str {
        "gradient_clip_gpu_scale"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("gradient_clip_gpu_scale")
            .param(PtxType::U64, "grads_ptr")
            .param(PtxType::U64, "scale_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Bounds check
                let n = ctx.load_param_u32("n");
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Load scale from GPU memory (L1 cache handles broadcast)
                let scale_ptr = ctx.load_param_u64("scale_ptr");
                let scale = ctx.ld_global_f32(scale_ptr);

                // Early exit if scale ≈ 1.0 (no clipping needed)
                // Avoids writing back unchanged values, saving memory bandwidth
                let one = ctx.mov_f32_imm(1.0);
                let diff = ctx.sub_f32(scale, one);
                let abs_diff = ctx.abs_f32(diff);
                let threshold = ctx.mov_f32_imm(1e-7);
                let no_clip = ctx.setp_lt_f32(abs_diff, threshold);
                ctx.branch_if(no_clip, "exit");

                // Load gradient
                let grads_ptr = ctx.load_param_u64("grads_ptr");
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let grad_addr = ctx.add_u64(grads_ptr, offset);
                let grad = ctx.ld_global_f32(grad_addr);

                // Scale gradient
                let grad_new = ctx.mul_f32(grad, scale);

                // Store updated gradient
                ctx.st_global_f32(grad_addr, grad_new);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_scale_reduce_kernel_name() {
        let kernel = ClipScaleReduceKernel;
        assert_eq!(kernel.name(), "clip_scale_reduce");
    }

    #[test]
    fn test_clip_scale_reduce_ptx_generation() {
        let kernel = ClipScaleReduceKernel;
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry clip_scale_reduce"));
        assert!(ptx.contains(".param .u64 partials_ptr"));
        assert!(ptx.contains(".param .u32 total_n"));
        assert!(ptx.contains(".param .f32 max_norm"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        // Verify sqrt for norm computation
        assert!(ptx.contains("sqrt"));
        // Verify min for scale clamping
        assert!(ptx.contains("min"));
        // Verify div for max_norm / norm
        assert!(ptx.contains("div"));
    }

    #[test]
    fn test_gradient_clip_gpu_scale_kernel_name() {
        let kernel = GradientClipGpuScaleKernel::new(1024);
        assert_eq!(kernel.name(), "gradient_clip_gpu_scale");
    }

    #[test]
    fn test_gradient_clip_gpu_scale_ptx_generation() {
        let kernel = GradientClipGpuScaleKernel::new(1024);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry gradient_clip_gpu_scale"));
        assert!(ptx.contains(".param .u64 grads_ptr"));
        assert!(ptx.contains(".param .u64 scale_ptr"));
        assert!(ptx.contains(".param .u32 n"));
        // Scale is loaded from GPU memory, not a param
        assert!(!ptx.contains(".param .f32 scale"));
        // Verify multiplication for scaling
        assert!(ptx.contains("mul.f32"));
    }

    #[test]
    fn test_clip_scale_reduce_barrier_safety() {
        let kernel = ClipScaleReduceKernel;
        // Single-thread kernel has no barriers — must not panic
        let _ptx = kernel.emit_ptx_validated();
    }

    #[test]
    fn test_gradient_clip_gpu_scale_barrier_safety() {
        let kernel = GradientClipGpuScaleKernel::new(1024);
        // No barriers — must not panic
        let _ptx = kernel.emit_ptx_validated();
    }
}
