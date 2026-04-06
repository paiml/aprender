//! LayerNorm Backward Kernel
//!
//! Backward (gradient) kernel for Layer Normalization.
//!
//! ## Mathematical Specification
//!
//! Forward:
//! - `y = (x - μ) / σ * γ + β`
//! - where `μ = mean(x)`, `σ = sqrt(var(x) + ε)`
//!
//! Backward (gradients w.r.t. input):
//! - `∂L/∂x_i = γ/σ * (∂L/∂y_i - mean(∂L/∂y) - (x_i - μ)/σ² * mean((x - μ) * ∂L/∂y * γ))`
//!
//! Simplified:
//! - `∂L/∂x = γ/σ * (∂L/∂y - mean(∂L/∂y * γ) - x_norm * mean(x_norm * ∂L/∂y * γ))`
//! - where `x_norm = (x - μ) / σ`
//!
//! ## Implementation
//!
//! Uses warp shuffle reductions for computing means.
//! One warp processes one row (hidden_dim ≤ 32).
//!
//! ## Falsifiable Prediction (P-LN-BACK-001)
//!
//! LayerNorm backward matches finite-difference within ε < 1e-5.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// LayerNorm Backward Kernel (warp-parallel, one row per warp)
///
/// Computes gradients for LayerNorm using warp shuffle reductions.
///
/// # Parameters
/// - `input_ptr`: Original input (x)
/// - `gamma_ptr`: Learned scale parameter (γ)
/// - `mean_ptr`: Mean values from forward pass
/// - `rstd_ptr`: Reciprocal std dev (1/σ) from forward pass
/// - `grad_output_ptr`: Gradient from upstream (∂L/∂y)
/// - `grad_input_ptr`: Output gradient for input (∂L/∂x)
/// - `num_rows`: Number of rows (batch size)
/// - `hidden_dim`: Hidden dimension (must be ≤ 32)
#[derive(Debug, Clone)]
pub struct LayerNormBackwardKernel {
    /// Number of rows (batch size)
    pub num_rows: u32,
    /// Hidden dimension (max 32 for warp reduction)
    pub hidden_dim: u32,
}

impl LayerNormBackwardKernel {
    /// Create a new LayerNorm backward kernel
    ///
    /// # Arguments
    /// - `num_rows`: Batch size
    /// - `hidden_dim`: Hidden dimension (must be ≤ 32)
    ///
    /// # Panics
    /// Panics if `hidden_dim` > 32
    #[must_use]
    pub fn new(num_rows: u32, hidden_dim: u32) -> Self {
        assert!(
            hidden_dim <= 32,
            "hidden_dim must be ≤ 32 for warp reduction"
        );
        Self {
            num_rows,
            hidden_dim,
        }
    }
}

impl Kernel for LayerNormBackwardKernel {
    fn name(&self) -> &str {
        "layer_norm_backward"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_dim = self.hidden_dim;

        PtxKernel::new("layer_norm_backward")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "gamma_ptr")
            .param(PtxType::U64, "mean_ptr")
            .param(PtxType::U64, "rstd_ptr")
            .param(PtxType::U64, "grad_output_ptr")
            .param(PtxType::U64, "grad_input_ptr")
            .param(PtxType::U32, "num_rows")
            .param(PtxType::U32, "hidden_dim")
            .build(move |ctx| {
                // Thread indexing: one warp (32 threads) per row
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let global_tid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Lane within warp (0-31)
                let lane = ctx.and_u32_imm(global_tid, 31);

                // Warp ID (row index)
                let warp_id = ctx.shr_u32_imm(global_tid, 5);

                // Load parameters
                let num_rows_param = ctx.load_param_u32("num_rows");
                let hidden_dim_param = ctx.load_param_u32("hidden_dim");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");
                let mean_ptr = ctx.load_param_u64("mean_ptr");
                let rstd_ptr = ctx.load_param_u64("rstd_ptr");
                let grad_output_ptr = ctx.load_param_u64("grad_output_ptr");
                let grad_input_ptr = ctx.load_param_u64("grad_input_ptr");

                // Bounds check: warp_id < num_rows
                let valid_row = ctx.setp_lt_u32(warp_id, num_rows_param);
                ctx.branch_if_not(valid_row, "exit");

                // Bounds check: lane < hidden_dim
                let valid_lane = ctx.setp_lt_u32(lane, hidden_dim_param);

                // Calculate row base offset
                let row_elem_offset = ctx.mul_lo_u32(warp_id, hidden_dim_param);
                let row_byte_offset = ctx.mul_wide_u32(row_elem_offset, 4);
                let input_row_base = ctx.add_u64(input_ptr, row_byte_offset);
                let grad_out_row_base = ctx.add_u64(grad_output_ptr, row_byte_offset);
                let grad_in_row_base = ctx.add_u64(grad_input_ptr, row_byte_offset);

                // Calculate element address within row
                let lane_offset = ctx.mul_wide_u32(lane, 4);
                let input_addr = ctx.add_u64(input_row_base, lane_offset);
                let gamma_addr = ctx.add_u64(gamma_ptr, lane_offset);
                let grad_out_addr = ctx.add_u64(grad_out_row_base, lane_offset);
                let grad_in_addr = ctx.add_u64(grad_in_row_base, lane_offset);

                // Load mean and rstd (1/σ) for this row
                let row_scalar_offset = ctx.mul_wide_u32(warp_id, 4);
                let mean_addr = ctx.add_u64(mean_ptr, row_scalar_offset);
                let rstd_addr = ctx.add_u64(rstd_ptr, row_scalar_offset);
                let mean = ctx.ld_global_f32(mean_addr);
                let rstd = ctx.ld_global_f32(rstd_addr);

                // Load x[i], γ[i], and ∂L/∂y[i] with bounds checking
                let x_i = ctx.ld_global_f32_predicated(input_addr, valid_lane, 0.0);
                let gamma_i = ctx.ld_global_f32_predicated(gamma_addr, valid_lane, 0.0);
                let grad_y_i = ctx.ld_global_f32_predicated(grad_out_addr, valid_lane, 0.0);

                // Compute x_norm = (x - mean) * rstd
                let x_centered = ctx.sub_f32(x_i, mean);
                let x_norm = ctx.mul_f32(x_centered, rstd);

                // Compute grad_y * gamma
                let grad_y_gamma = ctx.mul_f32(grad_y_i, gamma_i);

                // Warp reduction 1: mean(grad_y * gamma)
                let mut sum1 = grad_y_gamma;
                let warp_mask = 0xFFFF_FFFFu32;

                for offset in [16u32, 8, 4, 2, 1] {
                    if offset < hidden_dim {
                        let shuffled = ctx.shfl_down_f32(sum1, offset, warp_mask);
                        sum1 = ctx.add_f32(sum1, shuffled);
                    }
                }
                let total_sum1 = ctx.shfl_idx_f32(sum1, 0, warp_mask);

                // Warp reduction 2: mean(x_norm * grad_y * gamma)
                let x_norm_grad_gamma = ctx.mul_f32(x_norm, grad_y_gamma);
                let mut sum2 = x_norm_grad_gamma;

                for offset in [16u32, 8, 4, 2, 1] {
                    if offset < hidden_dim {
                        let shuffled = ctx.shfl_down_f32(sum2, offset, warp_mask);
                        sum2 = ctx.add_f32(sum2, shuffled);
                    }
                }
                let total_sum2 = ctx.shfl_idx_f32(sum2, 0, warp_mask);

                // Compute means
                let hidden_dim_f32 = ctx.cvt_f32_u32(hidden_dim_param);
                let mean_grad_gamma = ctx.div_f32(total_sum1, hidden_dim_f32);
                let mean_x_norm_grad_gamma = ctx.div_f32(total_sum2, hidden_dim_f32);

                // Compute grad_x = rstd * gamma * (grad_y - mean(grad_y*gamma) - x_norm * mean(x_norm*grad_y*gamma))
                // = gamma * rstd * (grad_y - mean_grad_gamma - x_norm * mean_x_norm_grad_gamma)
                let correction1 = mean_grad_gamma;
                let correction2 = ctx.mul_f32(x_norm, mean_x_norm_grad_gamma);
                let total_correction = ctx.add_f32(correction1, correction2);
                let adjusted_grad = ctx.sub_f32(grad_y_i, total_correction);
                let gamma_rstd = ctx.mul_f32(gamma_i, rstd);
                let grad_x_i = ctx.mul_f32(gamma_rstd, adjusted_grad);

                // Store result only for valid lanes
                ctx.branch_if_not(valid_lane, "exit");
                ctx.st_global_f32(grad_in_addr, grad_x_i);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_backward_name() {
        let kernel = LayerNormBackwardKernel::new(64, 32);
        assert_eq!(kernel.name(), "layer_norm_backward");
    }

    #[test]
    fn test_layer_norm_backward_ptx_generation() {
        let kernel = LayerNormBackwardKernel::new(64, 32);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry layer_norm_backward"));
        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 gamma_ptr"));
        assert!(ptx.contains(".param .u64 mean_ptr"));
        assert!(ptx.contains(".param .u64 rstd_ptr"));
        assert!(ptx.contains(".param .u64 grad_output_ptr"));
        assert!(ptx.contains(".param .u64 grad_input_ptr"));
        // Verify warp shuffle for reduction
        assert!(ptx.contains("shfl.sync.down"));
        // Verify subtraction for centering (sub.f32 without rounding modifier)
        assert!(ptx.contains("sub.f32"));
    }

    #[test]
    fn test_layer_norm_backward_small_hidden() {
        let kernel = LayerNormBackwardKernel::new(128, 16);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry layer_norm_backward"));
        assert!(ptx.contains("shfl.sync"));
    }

    #[test]
    fn test_layer_norm_backward_barrier_safety() {
        let kernel = LayerNormBackwardKernel::new(64, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "LayerNorm backward should be barrier-safe: {:?}",
            result.violations
        );
    }

    #[test]
    #[should_panic(expected = "hidden_dim must be ≤ 32")]
    fn test_layer_norm_backward_hidden_dim_limit() {
        let _ = LayerNormBackwardKernel::new(64, 64);
    }
}
