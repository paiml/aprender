//! Softmax Backward Kernel
//!
//! Backward (gradient) kernel for softmax activation.
//!
//! ## Mathematical Specification
//!
//! Forward: `y_i = exp(x_i - max) / sum(exp(x_j - max))`
//!
//! Backward: `∂L/∂x_i = y_i · (∂L/∂y_i - Σⱼ ∂L/∂y_j · y_j)`
//!
//! The key insight is that we need to compute `dot(grad_y, y)` for each row,
//! then use it to compute the gradient for each element.
//!
//! ## Implementation
//!
//! Uses warp shuffle reductions for rows that fit within a warp (≤32 elements).
//! For larger rows, this kernel should be called with row_size ≤ 32.
//!
//! ## Falsifiable Prediction (P-SOFT-BACK-001)
//!
//! Softmax backward matches finite-difference within ε < 1e-5.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Softmax Backward Kernel (warp-parallel, one row per warp)
///
/// Computes gradients for softmax using warp shuffle reductions.
///
/// # Parameters
/// - `output_ptr`: Softmax forward output (y)
/// - `grad_output_ptr`: Gradient from upstream (∂L/∂y)
/// - `grad_input_ptr`: Output gradient (∂L/∂x)
/// - `num_rows`: Number of rows
/// - `row_size`: Elements per row (must be ≤ 32 for warp-level reduction)
#[derive(Debug, Clone)]
pub struct SoftmaxBackwardKernel {
    /// Number of rows
    pub num_rows: u32,
    /// Elements per row (max 32 for warp reduction)
    pub row_size: u32,
}

impl SoftmaxBackwardKernel {
    /// Create a new Softmax backward kernel
    ///
    /// # Arguments
    /// - `num_rows`: Number of rows to process
    /// - `row_size`: Elements per row (must be ≤ 32)
    ///
    /// # Panics
    /// Panics if `row_size` > 32
    #[must_use]
    pub fn new(num_rows: u32, row_size: u32) -> Self {
        assert!(row_size <= 32, "row_size must be ≤ 32 for warp reduction");
        Self { num_rows, row_size }
    }
}

impl Kernel for SoftmaxBackwardKernel {
    fn name(&self) -> &str {
        "softmax_backward"
    }

    fn build_ptx(&self) -> PtxKernel {
        let row_size = self.row_size;

        PtxKernel::new("softmax_backward")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U64, "grad_output_ptr")
            .param(PtxType::U64, "grad_input_ptr")
            .param(PtxType::U32, "num_rows")
            .param(PtxType::U32, "row_size")
            .build(move |ctx| {
                // Thread indexing: one warp (32 threads) per row
                // Global warp ID = (blockIdx.x * blockDim.x + threadIdx.x) / 32
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let global_tid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Lane within warp (0-31): lane = global_tid & 31
                let lane = ctx.and_u32_imm(global_tid, 31);

                // Warp ID (row index): warp_id = global_tid >> 5
                let warp_id = ctx.shr_u32_imm(global_tid, 5);

                // Load parameters
                let num_rows = ctx.load_param_u32("num_rows");
                let row_size_param = ctx.load_param_u32("row_size");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let grad_output_ptr = ctx.load_param_u64("grad_output_ptr");
                let grad_input_ptr = ctx.load_param_u64("grad_input_ptr");

                // Bounds check: warp_id < num_rows
                let valid_row = ctx.setp_lt_u32(warp_id, num_rows);
                ctx.branch_if_not(valid_row, "exit");

                // Bounds check: lane < row_size
                let valid_lane = ctx.setp_lt_u32(lane, row_size_param);

                // Calculate row base offset: row_offset = warp_id * row_size * 4
                let row_elem_offset = ctx.mul_lo_u32(warp_id, row_size_param);
                let row_byte_offset = ctx.mul_wide_u32(row_elem_offset, 4);
                let output_row_base = ctx.add_u64(output_ptr, row_byte_offset);
                let grad_out_row_base = ctx.add_u64(grad_output_ptr, row_byte_offset);
                let grad_in_row_base = ctx.add_u64(grad_input_ptr, row_byte_offset);

                // Calculate element address within row: lane * 4
                let lane_offset = ctx.mul_wide_u32(lane, 4);
                let output_addr = ctx.add_u64(output_row_base, lane_offset);
                let grad_out_addr = ctx.add_u64(grad_out_row_base, lane_offset);
                let grad_in_addr = ctx.add_u64(grad_in_row_base, lane_offset);

                // Load y[i] and grad_y[i] with bounds checking
                let y_i = ctx.ld_global_f32_predicated(output_addr, valid_lane, 0.0);
                let grad_y_i = ctx.ld_global_f32_predicated(grad_out_addr, valid_lane, 0.0);

                // Compute local contribution: y_i * grad_y_i
                let local_product = ctx.mul_f32(y_i, grad_y_i);

                // Warp-level reduction to compute dot(grad_y, y) = Σ(y_i * grad_y_i)
                // Using butterfly reduction pattern with shfl.down
                let mut sum = local_product;

                // Full warp mask for shuffle operations
                let warp_mask = 0xFFFF_FFFFu32;

                // Unroll reduction: 16, 8, 4, 2, 1
                for offset in [16u32, 8, 4, 2, 1] {
                    if offset < row_size {
                        let shuffled = ctx.shfl_down_f32(sum, offset, warp_mask);
                        sum = ctx.add_f32(sum, shuffled);
                    }
                }

                // Broadcast sum to all lanes using shfl.idx from lane 0
                let dot_product = ctx.shfl_idx_f32(sum, 0, warp_mask);

                // Compute gradient: grad_x_i = y_i * (grad_y_i - dot_product)
                let diff = ctx.sub_f32(grad_y_i, dot_product);
                let grad_x_i = ctx.mul_f32(y_i, diff);

                // Store result only for valid lanes
                ctx.branch_if_not(valid_lane, "exit");
                ctx.st_global_f32(grad_in_addr, grad_x_i);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Batched Softmax Backward Kernel (stride-loop, one warp per row, arbitrary row_size)
///
/// Handles row_size > 32 by striding across the row in warp-width (32) steps.
/// Mirrors the forward `BatchedSoftmaxKernel` pattern.
///
/// # Contract (C-BSMAX-BACK-001)
///
/// - **Precondition**: output_ptr contains valid softmax output (y), grad_output_ptr contains ∂L/∂y,
///   all buffers have at least total_rows * row_size elements, row_size > 0, total_rows > 0
/// - **Postcondition**: grad_input[r][i] = y[r][i] * (∂L/∂y[r][i] - Σⱼ ∂L/∂y[r][j] * y[r][j])
///   for all r in [0, total_rows), i in [0, row_size)
/// - **Invariant**: Zero CPU-side data transfers; in-place safe (grad_input may alias grad_output
///   because pass 1 reads all elements before pass 2 writes any)
///
/// # Falsifiable Prediction (P-BSMAX-BACK-001)
///
/// Batched softmax backward matches finite-difference within ε < 1e-4 for row_size in {1..512}.
///
/// # Parameters
/// - `output_ptr`: Softmax forward output (y) [total_rows, row_size]
/// - `grad_output_ptr`: Gradient from upstream (∂L/∂y) [total_rows, row_size]
/// - `grad_input_ptr`: Output gradient (∂L/∂x) [total_rows, row_size]
/// - `total_rows`: Number of rows (num_heads * seq_len for attention)
/// - `row_size`: Elements per row (seq_len for attention)
#[derive(Debug, Clone)]
pub struct BatchedSoftmaxBackwardKernel {
    /// Total number of rows to process
    pub total_rows: u32,
    /// Size of each row (may exceed 32)
    pub row_size: u32,
}

impl BatchedSoftmaxBackwardKernel {
    /// Create a new batched softmax backward kernel
    ///
    /// # Arguments
    /// - `total_rows`: Number of rows to process
    /// - `row_size`: Elements per row (no upper limit)
    #[must_use]
    pub const fn new(total_rows: u32, row_size: u32) -> Self {
        Self { total_rows, row_size }
    }
}

impl Kernel for BatchedSoftmaxBackwardKernel {
    fn name(&self) -> &str {
        "batched_softmax_backward"
    }

    fn build_ptx(&self) -> PtxKernel {
        let total_rows = self.total_rows;
        let row_size = self.row_size;

        PtxKernel::new("batched_softmax_backward")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U64, "grad_output_ptr")
            .param(PtxType::U64, "grad_input_ptr")
            .param(PtxType::U32, "total_rows")
            .param(PtxType::U32, "row_size")
            .build(move |ctx| {
                // One block per row, one warp (32 threads) per block
                let row_idx = ctx.special_reg(PtxReg::CtaIdX);
                let tid = ctx.special_reg(PtxReg::TidX);

                // Bounds check: row_idx < total_rows
                let total_rows_reg = ctx.mov_u32_imm(total_rows);
                let valid = ctx.setp_lt_u32(row_idx, total_rows_reg);
                ctx.branch_if_not(valid, "exit");

                let output_ptr = ctx.load_param_u64("output_ptr");
                let grad_output_ptr = ctx.load_param_u64("grad_output_ptr");
                let grad_input_ptr = ctx.load_param_u64("grad_input_ptr");
                let row_size_reg = ctx.mov_u32_imm(row_size);

                // Calculate row base addresses
                let row_offset = ctx.mul_wide_u32(row_idx, row_size * 4);
                let output_row_base = ctx.add_u64(output_ptr, row_offset);
                let grad_out_row_base = ctx.add_u64(grad_output_ptr, row_offset);
                let grad_in_row_base = ctx.add_u64(grad_input_ptr, row_offset);

                let four = ctx.mov_u32_imm(4);

                // === Pass 1: Compute dot(grad_y, y) via stride loop + warp reduce ===
                let local_dot = ctx.mov_f32_imm(0.0);
                let i_dot = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(i_dot, tid);
                ctx.label("dot_loop");
                let dot_done = ctx.setp_ge_u32(i_dot, row_size_reg);
                ctx.branch_if(dot_done, "dot_done");

                let offset = ctx.mul_wide_u32_reg(i_dot, four);
                let y_addr = ctx.add_u64(output_row_base, offset);
                let gy_addr = ctx.add_u64(grad_out_row_base, offset);
                let y_val = ctx.ld_global_f32(y_addr);
                let gy_val = ctx.ld_global_f32(gy_addr);
                let prod = ctx.mul_f32(y_val, gy_val);
                ctx.add_f32_inplace(local_dot, prod);
                ctx.add_u32_inplace(i_dot, 32);
                ctx.branch("dot_loop");

                ctx.label("dot_done");

                // Warp-reduce the dot product
                let dt16 = ctx.shfl_down_f32(local_dot, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_dot, dt16);
                let dt8 = ctx.shfl_down_f32(local_dot, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_dot, dt8);
                let dt4 = ctx.shfl_down_f32(local_dot, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_dot, dt4);
                let dt2 = ctx.shfl_down_f32(local_dot, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_dot, dt2);
                let dt1 = ctx.shfl_down_f32(local_dot, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_dot, dt1);

                // Broadcast row_dot to all lanes
                let row_dot = ctx.shfl_idx_f32(local_dot, 0, 0xFFFF_FFFF);

                // === Pass 2: Compute and store gradients via stride loop ===
                // grad_x[i] = y[i] * (grad_y[i] - row_dot)
                let i_write = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(i_write, tid);
                ctx.label("write_loop");
                let write_done = ctx.setp_ge_u32(i_write, row_size_reg);
                ctx.branch_if(write_done, "exit");

                let offset = ctx.mul_wide_u32_reg(i_write, four);
                let y_addr = ctx.add_u64(output_row_base, offset);
                let gy_addr = ctx.add_u64(grad_out_row_base, offset);
                let gx_addr = ctx.add_u64(grad_in_row_base, offset);

                let y_val = ctx.ld_global_f32(y_addr);
                let gy_val = ctx.ld_global_f32(gy_addr);
                let diff = ctx.sub_f32(gy_val, row_dot);
                let grad_x = ctx.mul_f32(y_val, diff);
                ctx.st_global_f32(gx_addr, grad_x);

                ctx.add_u32_inplace(i_write, 32);
                ctx.branch("write_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_backward_name() {
        let kernel = SoftmaxBackwardKernel::new(64, 32);
        assert_eq!(kernel.name(), "softmax_backward");
    }

    #[test]
    fn test_softmax_backward_ptx_generation() {
        let kernel = SoftmaxBackwardKernel::new(64, 32);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry softmax_backward"));
        // Verify parameters
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u64 grad_output_ptr"));
        assert!(ptx.contains(".param .u64 grad_input_ptr"));
        assert!(ptx.contains(".param .u32 num_rows"));
        // Verify warp shuffle for reduction
        assert!(ptx.contains("shfl.sync.down"));
        // Verify broadcast
        assert!(ptx.contains("shfl.sync.idx"));
    }

    #[test]
    fn test_softmax_backward_small_row() {
        // Test with smaller row size
        let kernel = SoftmaxBackwardKernel::new(128, 16);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry softmax_backward"));
        assert!(ptx.contains("shfl.sync"));
    }

    #[test]
    fn test_softmax_backward_barrier_safety() {
        let kernel = SoftmaxBackwardKernel::new(64, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe, "Softmax backward should be barrier-safe: {:?}", result.violations);
    }

    #[test]
    #[should_panic(expected = "row_size must be ≤ 32")]
    fn test_softmax_backward_row_size_limit() {
        let _ = SoftmaxBackwardKernel::new(64, 64);
    }

    // === BatchedSoftmaxBackwardKernel tests ===

    #[test]
    fn test_batched_softmax_backward_name() {
        let kernel = BatchedSoftmaxBackwardKernel::new(64, 128);
        assert_eq!(kernel.name(), "batched_softmax_backward");
    }

    #[test]
    fn test_batched_softmax_backward_ptx_generation() {
        let kernel = BatchedSoftmaxBackwardKernel::new(64, 128);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry batched_softmax_backward"));
        // Verify parameters
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u64 grad_output_ptr"));
        assert!(ptx.contains(".param .u64 grad_input_ptr"));
        assert!(ptx.contains(".param .u32 total_rows"));
        assert!(ptx.contains(".param .u32 row_size"));
        // Verify warp shuffle for reduction
        assert!(ptx.contains("shfl.sync.down"));
        // Verify broadcast
        assert!(ptx.contains("shfl.sync.idx"));
    }

    #[test]
    fn test_batched_softmax_backward_large_row() {
        // row_size=512 (typical attention seq_len) — must not panic
        let kernel = BatchedSoftmaxBackwardKernel::new(14 * 512, 512);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry batched_softmax_backward"));
    }

    #[test]
    fn test_batched_softmax_backward_small_row() {
        // row_size=1 edge case
        let kernel = BatchedSoftmaxBackwardKernel::new(4, 1);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry batched_softmax_backward"));
    }

    #[test]
    fn test_batched_softmax_backward_row_size_32() {
        // Exactly one warp width — no stride needed
        let kernel = BatchedSoftmaxBackwardKernel::new(128, 32);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry batched_softmax_backward"));
        assert!(ptx.contains("shfl.sync"));
    }

    #[test]
    fn test_batched_softmax_backward_barrier_safety() {
        let kernel = BatchedSoftmaxBackwardKernel::new(64, 128);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "Batched softmax backward should be barrier-safe: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_batched_softmax_backward_clone_and_debug() {
        let kernel = BatchedSoftmaxBackwardKernel::new(256, 64);
        let cloned = kernel.clone();
        assert_eq!(kernel.total_rows, cloned.total_rows);
        assert_eq!(kernel.row_size, cloned.row_size);

        let debug_str = format!("{kernel:?}");
        assert!(debug_str.contains("BatchedSoftmaxBackwardKernel"));
        assert!(debug_str.contains("256"));
        assert!(debug_str.contains("64"));
    }

    #[test]
    fn test_batched_softmax_backward_various_sizes() {
        // Test multiple representative sizes
        for (rows, cols) in [(1, 1), (16, 16), (64, 32), (128, 64), (512, 128), (1024, 512)] {
            let kernel = BatchedSoftmaxBackwardKernel::new(rows, cols);
            let ptx = kernel.emit_ptx();
            assert!(
                ptx.contains(".entry batched_softmax_backward"),
                "Failed for rows={rows}, cols={cols}"
            );
        }
    }
}
