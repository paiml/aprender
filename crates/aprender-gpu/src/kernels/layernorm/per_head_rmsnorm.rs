//! GH-280: Per-Head RMSNorm kernel for QK normalization (Qwen3)
//!
//! Applies RMSNorm independently to each attention head:
//!
//! ```text
//! For each head h in 0..num_heads:
//!     slice = input[h*head_dim .. (h+1)*head_dim]
//!     rms = sqrt(mean(slice^2) + eps)
//!     output[h*head_dim..(h+1)*head_dim] = slice / rms * gamma
//! ```
//!
//! Gamma weights have shape `[head_dim]` and are shared across all heads.
//! Grid: (num_heads, 1, 1), Block: (32, 1, 1) — one warp per head.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Per-head RMSNorm kernel for QK normalization (Qwen3).
///
/// Each CUDA block (one warp = 32 threads) processes one attention head.
/// `blockIdx.x` selects the head, threads stride over `head_dim` elements.
///
/// For Qwen3-8B: head_dim=128, num_heads=32 (Q) or 8 (K), eps=1e-6.
#[derive(Debug, Clone)]
pub struct PerHeadRmsNormKernel {
    /// Elements per head (128 for Qwen3)
    pub head_dim: u32,
    /// Number of heads (32 for Q, 8 for K)
    pub num_heads: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl PerHeadRmsNormKernel {
    /// Create a new per-head RMSNorm kernel
    #[must_use]
    pub fn new(head_dim: u32, num_heads: u32) -> Self {
        Self { head_dim, num_heads, epsilon: 1e-6 }
    }

    /// Set custom epsilon value
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl Kernel for PerHeadRmsNormKernel {
    fn name(&self) -> &str {
        "per_head_rmsnorm"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let epsilon = self.epsilon;

        // Per-head RMSNorm using warp shuffle (same pattern as RmsNormKernel)
        // Grid: (num_heads, 1, 1) — one block per head
        // Block: (32, 1, 1) — one warp
        // Each thread handles head_dim/32 elements within its head
        PtxKernel::new("per_head_rmsnorm")
            .param(PtxType::U64, "input_ptr") // [num_heads * head_dim]
            .param(PtxType::U64, "output_ptr") // [num_heads * head_dim]
            .param(PtxType::U64, "gamma_ptr") // [head_dim] shared across heads
            .shared_memory(0) // Warp shuffle, no shared memory
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);

                // Load parameters
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Constants
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let four = ctx.mov_u32_imm(4);

                // Compute base offset for this head: head_idx * head_dim * 4 bytes
                let head_elem_offset = ctx.mul_u32_reg(head_idx, head_dim_u32);
                let head_byte_offset = ctx.mul_wide_u32_reg(head_elem_offset, four);
                let head_input_base = ctx.add_u64(input_ptr, head_byte_offset);
                let head_output_base = ctx.add_u64(output_ptr, head_byte_offset);

                // Pass 1: Accumulate sum of squares within this head
                // Each thread processes elements: tid, tid+32, tid+64, ...
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let loop_idx = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(loop_idx, head_dim_u32);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                // Load input[head_offset + idx]
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let elem_addr = ctx.add_u64(head_input_base, elem_offset);
                let val = ctx.ld_global_f32(elem_addr);

                // sq_sum += val * val
                ctx.fma_f32_inplace(sq_sum, val, val);

                // idx += 32 (stride by warp size)
                ctx.add_u32_inplace(idx, 32);
                ctx.branch("sum_loop");

                ctx.label("sum_loop_end");

                // Warp reduce sq_sum
                let shfl16 = ctx.shfl_down_f32(sq_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl16);
                let shfl8 = ctx.shfl_down_f32(sq_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl8);
                let shfl4 = ctx.shfl_down_f32(sq_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl4);
                let shfl2 = ctx.shfl_down_f32(sq_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl2);
                let shfl1 = ctx.shfl_down_f32(sq_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl1);

                // Broadcast final sum to all threads
                let total_sq_sum = ctx.shfl_idx_f32(sq_sum, 0, 0xFFFF_FFFF);

                // Compute RMS = sqrt(mean(x^2) + epsilon) over head_dim
                let head_dim_f32 = ctx.cvt_f32_u32(head_dim_u32);
                let mean_sq = ctx.div_f32(total_sq_sum, head_dim_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let mean_sq_eps = ctx.add_f32(mean_sq, eps);
                let rms_inv = ctx.rsqrt_f32(mean_sq_eps);

                // Pass 2: Normalize and scale
                // output[head_offset+i] = input[head_offset+i] * rms_inv * gamma[i]
                // Note: gamma is indexed by position within head (no head offset)
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, head_dim_u32);
                ctx.branch_if_not(in_bounds2, "exit");

                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let in_addr = ctx.add_u64(head_input_base, elem_offset2);
                // gamma is [head_dim], shared across heads — no head offset
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);
                let out_addr = ctx.add_u64(head_output_base, elem_offset2);

                let inp = ctx.ld_global_f32(in_addr);
                let gamma = ctx.ld_global_f32(gamma_addr);

                // output = input * rms_inv * gamma
                let normalized = ctx.mul_f32(inp, rms_inv);
                let result = ctx.mul_f32(normalized, gamma);

                ctx.st_global_f32(out_addr, result);

                ctx.add_u32_inplace(idx2, 32);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}
