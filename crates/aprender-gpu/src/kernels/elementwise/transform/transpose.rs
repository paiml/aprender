//! Transpose Kernels
//!
//! Matrix transpose operations for attention K^T computation.
//!
//! - [`TransposeKernel`]: Single matrix transpose
//! - [`BatchedTransposeKernel`]: Batched transpose for multi-head attention

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Transpose Kernel: output[j, i] = input[i, j]
///
/// Simple matrix transpose for attention K^T computation.
#[derive(Debug, Clone)]
pub struct TransposeKernel {
    /// Number of rows in input
    pub rows: u32,
    /// Number of columns in input
    pub cols: u32,
}

impl TransposeKernel {
    /// Create a new transpose kernel
    #[must_use]
    pub const fn new(rows: u32, cols: u32) -> Self {
        Self { rows, cols }
    }
}

impl Kernel for TransposeKernel {
    fn name(&self) -> &str {
        "transpose"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Contract: dimension-independent-kernels-v1.yaml (FALSIFY-DIM-004)
        PtxKernel::new("transpose")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "rows")
            .param(PtxType::U32, "cols")
            .param(PtxType::U32, "total_elems")
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let rows = ctx.load_param_u32("rows");
                let cols = ctx.load_param_u32("cols");
                let total = ctx.load_param_u32("total_elems");

                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                let row_idx = ctx.div_u32_reg(gid, cols);
                let col_idx = ctx.rem_u32_reg(gid, cols);

                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(gid, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);

                let out_linear = ctx.mad_lo_u32(col_idx, rows, row_idx);
                let output_offset = ctx.mul_wide_u32_reg(out_linear, four);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// =============================================================================
// Batched Transpose Kernel (WAPR-PERF-008)
// =============================================================================

/// Batched transpose kernel: transpose multiple matrices in one launch
#[derive(Debug, Clone)]
pub struct BatchedTransposeKernel {
    /// Number of batches (e.g., n_heads)
    pub batch: u32,
    /// Input rows (becomes output cols)
    pub rows: u32,
    /// Input cols (becomes output rows)
    pub cols: u32,
}

impl BatchedTransposeKernel {
    /// Create a new batched transpose kernel
    #[must_use]
    pub const fn new(batch: u32, rows: u32, cols: u32) -> Self {
        Self { batch, rows, cols }
    }
}

impl Kernel for BatchedTransposeKernel {
    fn name(&self) -> &str {
        "batched_transpose"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Contract: dimension-independent-kernels-v1.yaml (FALSIFY-DIM-004)
        PtxKernel::new("batched_transpose")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "batch")
            .param(PtxType::U32, "rows")
            .param(PtxType::U32, "cols")
            .param(PtxType::U32, "total_per_batch")
            .build(|ctx| {
                let batch_idx = ctx.special_reg(PtxReg::CtaIdZ);
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let total = ctx.load_param_u32("total_per_batch");
                let in_bounds = ctx.setp_lt_u32(gid, total);
                let batch_param = ctx.load_param_u32("batch");
                let batch_valid = ctx.setp_lt_u32(batch_idx, batch_param);
                let valid = ctx.and_pred(in_bounds, batch_valid);
                ctx.branch_if_not(valid, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let rows = ctx.load_param_u32("rows");
                let cols = ctx.load_param_u32("cols");

                let row = ctx.div_u32_reg(gid, cols);
                let col = ctx.rem_u32_reg(gid, cols);

                // batch_byte_stride = total_per_batch * 4
                let four = ctx.mov_u32_imm(4);
                let batch_byte_stride = ctx.mul_lo_u32(total, four);
                let batch_offset = ctx.mul_wide_u32_reg(batch_idx, batch_byte_stride);
                let in_batch_ptr = ctx.add_u64(input_ptr, batch_offset);
                let out_batch_ptr = ctx.add_u64(output_ptr, batch_offset);

                let in_idx = ctx.mad_lo_u32(row, cols, col);
                let out_idx = ctx.mad_lo_u32(col, rows, row);

                let in_offset = ctx.mul_wide_u32_reg(in_idx, four);
                let out_offset = ctx.mul_wide_u32_reg(out_idx, four);
                let in_addr = ctx.add_u64(in_batch_ptr, in_offset);
                let out_addr = ctx.add_u64(out_batch_ptr, out_offset);

                let val = ctx.ld_global_f32(in_addr);
                ctx.st_global_f32(out_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
