//! RMSNorm Backward Kernel
//!
//! Backward (gradient) kernel for Root Mean Square Layer Normalization.
//!
//! ## Mathematical Specification
//!
//! Forward: `y_i = x_i / rms(x) * γ_i` where `rms(x) = sqrt(mean(x²) + ε)`
//!
//! Backward:
//! - `∂L/∂x_i = (1/rms) * (γ_i * ∂L/∂y_i - x_i/rms² * mean(x · ∂L/∂y · γ))`
//! - `∂L/∂γ_i = Σ_batch (∂L/∂y_i * x_i / rms)`
//!
//! ## Implementation
//!
//! Uses warp shuffle reductions for computing mean(x · ∂L/∂y · γ).
//! One warp processes one row (hidden_dim ≤ 32).
//!
//! ## Falsifiable Prediction (P-RMS-BACK-001)
//!
//! RMSNorm backward matches finite-difference within ε < 1e-5.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxAtomic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// RMSNorm Backward Kernel (warp-parallel, one row per warp)
///
/// Computes gradients for RMSNorm using warp shuffle reductions.
///
/// # Parameters
/// - `input_ptr`: Original input (x)
/// - `gamma_ptr`: Learned scale parameter (γ)
/// - `rms_ptr`: RMS values from forward pass
/// - `grad_output_ptr`: Gradient from upstream (∂L/∂y)
/// - `grad_input_ptr`: Output gradient for input (∂L/∂x)
/// - `num_rows`: Number of rows (batch size)
/// - `hidden_dim`: Hidden dimension (must be ≤ 32)
/// - `eps`: Epsilon for numerical stability
#[derive(Debug, Clone)]
pub struct RmsNormBackwardKernel {
    /// Number of rows (batch size)
    pub num_rows: u32,
    /// Hidden dimension (max 32 for warp reduction)
    pub hidden_dim: u32,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl RmsNormBackwardKernel {
    /// Create a new RMSNorm backward kernel
    ///
    /// # Arguments
    /// - `num_rows`: Batch size
    /// - `hidden_dim`: Hidden dimension (must be ≤ 32)
    /// - `eps`: Epsilon for numerical stability
    ///
    /// # Panics
    /// Panics if `hidden_dim` > 32
    #[must_use]
    pub fn new(num_rows: u32, hidden_dim: u32, eps: f32) -> Self {
        assert!(hidden_dim <= 32, "hidden_dim must be ≤ 32 for warp reduction");
        Self { num_rows, hidden_dim, eps }
    }
}

impl Kernel for RmsNormBackwardKernel {
    fn name(&self) -> &str {
        "rms_norm_backward"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_dim = self.hidden_dim;
        let eps = self.eps;

        PtxKernel::new("rms_norm_backward")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "gamma_ptr")
            .param(PtxType::U64, "rms_ptr")
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
                let rms_ptr = ctx.load_param_u64("rms_ptr");
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

                // Load RMS value for this row
                let rms_offset = ctx.mul_wide_u32(warp_id, 4);
                let rms_addr = ctx.add_u64(rms_ptr, rms_offset);
                let rms = ctx.ld_global_f32(rms_addr);

                // Load x[i], γ[i], and ∂L/∂y[i] with bounds checking
                let x_i = ctx.ld_global_f32_predicated(input_addr, valid_lane, 0.0);
                let gamma_i = ctx.ld_global_f32_predicated(gamma_addr, valid_lane, 0.0);
                let grad_y_i = ctx.ld_global_f32_predicated(grad_out_addr, valid_lane, 0.0);

                // Compute x_i * grad_y_i * gamma_i
                let x_grad_gamma = ctx.mul_f32(x_i, grad_y_i);
                let x_grad_gamma = ctx.mul_f32(x_grad_gamma, gamma_i);

                // Warp reduction for mean(x * grad_y * gamma)
                let mut sum = x_grad_gamma;
                let warp_mask = 0xFFFF_FFFFu32;

                for offset in [16u32, 8, 4, 2, 1] {
                    if offset < hidden_dim {
                        let shuffled = ctx.shfl_down_f32(sum, offset, warp_mask);
                        sum = ctx.add_f32(sum, shuffled);
                    }
                }

                // Broadcast sum to all lanes
                let total_sum = ctx.shfl_idx_f32(sum, 0, warp_mask);

                // Compute mean: divide by hidden_dim
                let hidden_dim_f32 = ctx.cvt_f32_u32(hidden_dim_param);
                let mean_term = ctx.div_f32(total_sum, hidden_dim_f32);

                // Compute grad_x_i = (1/rms) * (gamma_i * grad_y_i - x_i/rms² * mean_term)
                let eps_const = ctx.mov_f32_imm(eps);
                let rms_sq = ctx.mul_f32(rms, rms);
                let rms_sq_eps = ctx.add_f32(rms_sq, eps_const);
                let rms_safe = ctx.sqrt_f32(rms_sq_eps);

                // 1 / rms
                let one = ctx.mov_f32_imm(1.0);
                let inv_rms = ctx.div_f32(one, rms_safe);

                // gamma_i * grad_y_i
                let gamma_grad_y = ctx.mul_f32(gamma_i, grad_y_i);

                // x_i / rms² * mean_term (correction)
                let x_over_rms_sq = ctx.div_f32(x_i, rms_sq_eps);
                let correction = ctx.mul_f32(x_over_rms_sq, mean_term);

                // gamma_i * grad_y_i - correction
                let adjusted_grad = ctx.sub_f32(gamma_grad_y, correction);

                // final gradient = (1/rms) * adjusted
                let grad_x_i = ctx.mul_f32(inv_rms, adjusted_grad);

                // Store result only for valid lanes
                ctx.branch_if_not(valid_lane, "exit");
                ctx.st_global_f32(grad_in_addr, grad_x_i);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Batched RMSNorm Backward Kernel (stride-loop, one warp per row, arbitrary hidden_dim)
///
/// Handles hidden_dim > 32 by striding across the row in warp-width (32) steps.
/// Computes RMS from input inline (no pre-computed RMS values needed).
///
/// # Contract (C-BRMS-BACK-001)
///
/// - **Precondition**: input_ptr contains original forward input (x), gamma_ptr contains
///   learned scale (γ), grad_output_ptr contains ∂L/∂y, all buffers have at least
///   num_rows * hidden_dim elements, hidden_dim > 0, num_rows > 0
/// - **Postcondition**: grad_input[r][i] = (1/rms(x[r])) * (γ[i] * ∂L/∂y[r][i] - x[r][i]/rms²
///   * mean(x[r] · ∂L/∂y[r] · γ)) for all r in [0, num_rows), i in [0, hidden_dim)
/// - **Invariant**: Zero CPU-side data transfers; RMS computed inline from input
///
/// # Falsifiable Prediction (P-BRMS-BACK-001)
///
/// Batched RMSNorm backward matches finite-difference within ε < 1e-4 for hidden_dim in {1..1024}.
///
/// # Parameters
/// - `input_ptr`: Original forward input (x) [num_rows, hidden_dim]
/// - `gamma_ptr`: Learned scale parameter (γ) [hidden_dim]
/// - `grad_output_ptr`: Gradient from upstream (∂L/∂y) [num_rows, hidden_dim]
/// - `grad_input_ptr`: Output gradient for input (∂L/∂x) [num_rows, hidden_dim]
/// - `grad_gamma_ptr`: Output gradient for gamma (∂L/∂γ) [hidden_dim] — accumulated via atomicAdd
/// - `num_rows`: Number of rows (batch size × seq_len)
/// - `hidden_dim`: Hidden dimension (no upper limit)
/// - `eps`: Epsilon for numerical stability
#[derive(Debug, Clone)]
pub struct BatchedRmsNormBackwardKernel {
    /// Number of rows to process
    pub num_rows: u32,
    /// Hidden dimension (may exceed 32)
    pub hidden_dim: u32,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl BatchedRmsNormBackwardKernel {
    /// Create a new batched RMSNorm backward kernel
    ///
    /// # Arguments
    /// - `num_rows`: Number of rows to process
    /// - `hidden_dim`: Hidden dimension (no upper limit)
    /// - `eps`: Epsilon for numerical stability
    #[must_use]
    pub fn new(num_rows: u32, hidden_dim: u32, eps: f32) -> Self {
        Self { num_rows, hidden_dim, eps }
    }
}

impl Kernel for BatchedRmsNormBackwardKernel {
    fn name(&self) -> &str {
        "batched_rms_norm_backward"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Contract: dimension-independent-kernels-v1.yaml (FALSIFY-DIM-001)
        // All dimensions loaded from runtime .param — NO baked immediates.
        PtxKernel::new("batched_rms_norm_backward")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "gamma_ptr")
            .param(PtxType::U64, "grad_output_ptr")
            .param(PtxType::U64, "grad_input_ptr")
            .param(PtxType::U64, "grad_gamma_ptr")
            .param(PtxType::U32, "num_rows")
            .param(PtxType::U32, "hidden_dim")
            .param(PtxType::F32, "eps")
            .build(|ctx| {
                // One block per row, one warp (32 threads) per block
                let row_idx = ctx.special_reg(PtxReg::CtaIdX);
                let tid = ctx.special_reg(PtxReg::TidX);

                // Bounds check: row_idx < num_rows
                let num_rows_reg = ctx.load_param_u32("num_rows");
                let valid = ctx.setp_lt_u32(row_idx, num_rows_reg);
                ctx.branch_if_not(valid, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");
                let grad_output_ptr = ctx.load_param_u64("grad_output_ptr");
                let grad_input_ptr = ctx.load_param_u64("grad_input_ptr");
                let grad_gamma_ptr = ctx.load_param_u64("grad_gamma_ptr");
                let hidden_dim_reg = ctx.load_param_u32("hidden_dim");

                // Calculate row base addresses: row_byte_stride = hidden_dim * 4
                let four = ctx.mov_u32_imm(4);
                let row_byte_stride = ctx.mul_lo_u32(hidden_dim_reg, four);
                let row_offset = ctx.mul_wide_u32_reg(row_idx, row_byte_stride);
                let input_row_base = ctx.add_u64(input_ptr, row_offset);
                let grad_out_row_base = ctx.add_u64(grad_output_ptr, row_offset);
                let grad_in_row_base = ctx.add_u64(grad_input_ptr, row_offset);

                // === Pass 1: Compute sum(x²) and sum(x·grad_y·γ) via stride loop ===
                let local_sum_x2 = ctx.mov_f32_imm(0.0);
                let local_sum_xgg = ctx.mov_f32_imm(0.0);
                let i_pass1 = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(i_pass1, tid);
                ctx.label("pass1_loop");
                let done1 = ctx.setp_ge_u32(i_pass1, hidden_dim_reg);
                ctx.branch_if(done1, "pass1_done");

                let offset = ctx.mul_wide_u32_reg(i_pass1, four);
                let x_addr = ctx.add_u64(input_row_base, offset);
                let gy_addr = ctx.add_u64(grad_out_row_base, offset);
                let g_addr = ctx.add_u64(gamma_ptr, offset);

                let x_val = ctx.ld_global_f32(x_addr);
                let gy_val = ctx.ld_global_f32(gy_addr);
                let g_val = ctx.ld_global_f32(g_addr);

                // sum_x2 += x * x
                let x2 = ctx.mul_f32(x_val, x_val);
                ctx.add_f32_inplace(local_sum_x2, x2);

                // sum_xgg += x * grad_y * gamma
                let xgy = ctx.mul_f32(x_val, gy_val);
                let xgyg = ctx.mul_f32(xgy, g_val);
                ctx.add_f32_inplace(local_sum_xgg, xgyg);

                ctx.add_u32_inplace(i_pass1, 32);
                ctx.branch("pass1_loop");

                ctx.label("pass1_done");

                // Warp-reduce sum_x2
                let s16a = ctx.shfl_down_f32(local_sum_x2, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum_x2, s16a);
                let s8a = ctx.shfl_down_f32(local_sum_x2, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum_x2, s8a);
                let s4a = ctx.shfl_down_f32(local_sum_x2, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum_x2, s4a);
                let s2a = ctx.shfl_down_f32(local_sum_x2, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum_x2, s2a);
                let s1a = ctx.shfl_down_f32(local_sum_x2, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum_x2, s1a);
                let sum_x2 = ctx.shfl_idx_f32(local_sum_x2, 0, 0xFFFF_FFFF);

                // Warp-reduce sum_xgg
                let s16b = ctx.shfl_down_f32(local_sum_xgg, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum_xgg, s16b);
                let s8b = ctx.shfl_down_f32(local_sum_xgg, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum_xgg, s8b);
                let s4b = ctx.shfl_down_f32(local_sum_xgg, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum_xgg, s4b);
                let s2b = ctx.shfl_down_f32(local_sum_xgg, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum_xgg, s2b);
                let s1b = ctx.shfl_down_f32(local_sum_xgg, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum_xgg, s1b);
                let sum_xgg = ctx.shfl_idx_f32(local_sum_xgg, 0, 0xFFFF_FFFF);

                // Compute rms = sqrt(mean(x²) + eps)
                let hidden_dim_f32 = ctx.cvt_f32_u32(hidden_dim_reg);
                let mean_x2 = ctx.div_f32(sum_x2, hidden_dim_f32);
                let eps_const = ctx.load_param_f32("eps");
                let variance_eps = ctx.add_f32(mean_x2, eps_const);
                let rms = ctx.sqrt_f32(variance_eps);

                // Compute mean(x · grad_y · gamma) = sum_xgg / hidden_dim
                let mean_xgg = ctx.div_f32(sum_xgg, hidden_dim_f32);

                // === Pass 2: Compute and store grad_x via stride loop ===
                // grad_x[i] = (1/rms) * (gamma[i] * grad_y[i] - x[i] / variance_eps * mean_xgg)
                let one = ctx.mov_f32_imm(1.0);
                let inv_rms = ctx.div_f32(one, rms);
                let i_pass2 = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(i_pass2, tid);
                ctx.label("pass2_loop");
                let done2 = ctx.setp_ge_u32(i_pass2, hidden_dim_reg);
                ctx.branch_if(done2, "exit");

                let offset = ctx.mul_wide_u32_reg(i_pass2, four);
                let x_addr = ctx.add_u64(input_row_base, offset);
                let gy_addr = ctx.add_u64(grad_out_row_base, offset);
                let g_addr = ctx.add_u64(gamma_ptr, offset);
                let gx_addr = ctx.add_u64(grad_in_row_base, offset);

                let x_val = ctx.ld_global_f32(x_addr);
                let gy_val = ctx.ld_global_f32(gy_addr);
                let g_val = ctx.ld_global_f32(g_addr);

                // gamma * grad_y
                let gamma_gy = ctx.mul_f32(g_val, gy_val);

                // x / variance_eps * mean_xgg (correction term)
                let x_over_var = ctx.div_f32(x_val, variance_eps);
                let correction = ctx.mul_f32(x_over_var, mean_xgg);

                // gamma * grad_y - correction
                let adjusted = ctx.sub_f32(gamma_gy, correction);

                // grad_x = (1/rms) * adjusted
                let grad_x = ctx.mul_f32(inv_rms, adjusted);
                ctx.st_global_f32(gx_addr, grad_x);

                // ∂L/∂γ_i += ∂L/∂y[r][i] × x[r][i] / rms
                // Accumulated across rows via atomicAdd (buffer must be zeroed by caller)
                let gg_addr = ctx.add_u64(grad_gamma_ptr, offset);
                let grad_gamma_contrib = ctx.mul_f32(gy_val, x_val);
                let grad_gamma_contrib = ctx.mul_f32(grad_gamma_contrib, inv_rms);
                let _ = ctx.atom_add_global_f32(gg_addr, grad_gamma_contrib);

                ctx.add_u32_inplace(i_pass2, 32);
                ctx.branch("pass2_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_backward_name() {
        let kernel = RmsNormBackwardKernel::new(64, 32, 1e-6);
        assert_eq!(kernel.name(), "rms_norm_backward");
    }

    #[test]
    fn test_rms_norm_backward_ptx_generation() {
        let kernel = RmsNormBackwardKernel::new(64, 32, 1e-6);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry rms_norm_backward"));
        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 gamma_ptr"));
        assert!(ptx.contains(".param .u64 rms_ptr"));
        assert!(ptx.contains(".param .u64 grad_output_ptr"));
        assert!(ptx.contains(".param .u64 grad_input_ptr"));
        // Verify warp shuffle for reduction
        assert!(ptx.contains("shfl.sync.down"));
        // Verify sqrt for RMS - check both rn and approx variants
        assert!(
            ptx.contains("sqrt.rn.f32") || ptx.contains("sqrt"),
            "PTX should contain sqrt: {}",
            ptx
        );
    }

    #[test]
    fn test_rms_norm_backward_small_hidden() {
        let kernel = RmsNormBackwardKernel::new(128, 16, 1e-5);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry rms_norm_backward"));
        assert!(ptx.contains("shfl.sync"));
    }

    #[test]
    fn test_rms_norm_backward_barrier_safety() {
        let kernel = RmsNormBackwardKernel::new(64, 32, 1e-6);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe, "RMSNorm backward should be barrier-safe: {:?}", result.violations);
    }

    #[test]
    #[should_panic(expected = "hidden_dim must be ≤ 32")]
    fn test_rms_norm_backward_hidden_dim_limit() {
        let _ = RmsNormBackwardKernel::new(64, 64, 1e-6);
    }

    // === BatchedRmsNormBackwardKernel tests ===

    #[test]
    fn test_batched_rms_norm_backward_name() {
        let kernel = BatchedRmsNormBackwardKernel::new(64, 128, 1e-6);
        assert_eq!(kernel.name(), "batched_rms_norm_backward");
    }

    #[test]
    fn test_batched_rms_norm_backward_ptx_generation() {
        let kernel = BatchedRmsNormBackwardKernel::new(64, 128, 1e-6);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry batched_rms_norm_backward"));
        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 gamma_ptr"));
        assert!(ptx.contains(".param .u64 grad_output_ptr"));
        assert!(ptx.contains(".param .u64 grad_input_ptr"));
        assert!(ptx.contains(".param .u64 grad_gamma_ptr"));
        assert!(ptx.contains(".param .u32 num_rows"));
        assert!(ptx.contains(".param .u32 hidden_dim"));
        assert!(ptx.contains(".param .f32 eps"));
        // Verify warp shuffle for reductions
        assert!(ptx.contains("shfl.sync.down"));
        assert!(ptx.contains("shfl.sync.idx"));
        // Verify sqrt for RMS
        assert!(
            ptx.contains("sqrt.rn.f32") || ptx.contains("sqrt"),
            "PTX should contain sqrt: {}",
            ptx
        );
    }

    #[test]
    fn test_batched_rms_norm_backward_large_hidden() {
        // hidden_dim=896 (Qwen2-0.5B) — must not panic
        let kernel = BatchedRmsNormBackwardKernel::new(512, 896, 1e-5);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry batched_rms_norm_backward"));
    }

    #[test]
    fn test_batched_rms_norm_backward_small_hidden() {
        // hidden_dim=1 edge case
        let kernel = BatchedRmsNormBackwardKernel::new(4, 1, 1e-6);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry batched_rms_norm_backward"));
    }

    #[test]
    fn test_batched_rms_norm_backward_hidden_32() {
        // Exactly one warp width — no stride needed
        let kernel = BatchedRmsNormBackwardKernel::new(128, 32, 1e-5);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry batched_rms_norm_backward"));
        assert!(ptx.contains("shfl.sync"));
    }

    #[test]
    fn test_batched_rms_norm_backward_hidden_64() {
        // Smallest size that triggers stride-loop (hidden > 32)
        let kernel = BatchedRmsNormBackwardKernel::new(8, 64, 1e-5);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry batched_rms_norm_backward"));
    }

    #[test]
    fn test_batched_rms_norm_backward_barrier_safety() {
        let kernel = BatchedRmsNormBackwardKernel::new(64, 128, 1e-6);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "Batched RMSNorm backward should be barrier-safe: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_batched_rms_norm_backward_clone_and_debug() {
        let kernel = BatchedRmsNormBackwardKernel::new(256, 64, 1e-5);
        let cloned = kernel.clone();
        assert_eq!(kernel.num_rows, cloned.num_rows);
        assert_eq!(kernel.hidden_dim, cloned.hidden_dim);
        assert!((kernel.eps - cloned.eps).abs() < 1e-10);

        let debug_str = format!("{kernel:?}");
        assert!(debug_str.contains("BatchedRmsNormBackwardKernel"));
        assert!(debug_str.contains("256"));
        assert!(debug_str.contains("64"));
    }

    #[test]
    fn test_batched_rms_norm_backward_various_sizes() {
        for (rows, dim) in [(1, 1), (16, 16), (64, 32), (128, 64), (512, 128), (24, 896)] {
            let kernel = BatchedRmsNormBackwardKernel::new(rows, dim, 1e-5);
            let ptx = kernel.emit_ptx();
            assert!(
                ptx.contains(".entry batched_rms_norm_backward"),
                "Failed for rows={rows}, dim={dim}"
            );
        }
    }
}
