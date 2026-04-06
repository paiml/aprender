//! Batched Element-wise Kernels
//!
//! GPU kernels for batched element-wise operations in multi-head attention.
//!
//! - [`BatchedScaleKernel`]: Scale all elements by a scalar
//! - [`BatchedSoftmaxKernel`]: Row-wise softmax for attention

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Batched scale kernel: multiply all elements by a scalar
#[derive(Debug, Clone)]
pub struct BatchedScaleKernel {
    /// Total number of elements (batch * rows * cols)
    pub n: u32,
}

impl BatchedScaleKernel {
    /// Create a new batched scale kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for BatchedScaleKernel {
    fn name(&self) -> &str {
        "batched_scale"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Contract: dimension-independent-kernels-v1.yaml (FALSIFY-DIM-004)
        PtxKernel::new("batched_scale")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::F32, "scale")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let total_reg = ctx.load_param_u32("n");
                let in_bounds = ctx.setp_lt_u32(gid, total_reg);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let scale = ctx.load_param_f32("scale");

                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                let val = ctx.ld_global_f32(in_addr);
                let scaled = ctx.mul_f32(val, scale);
                ctx.st_global_f32(out_addr, scaled);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Batched softmax kernel: softmax for multiple independent rows
///
/// Uses warp shuffle for reduction. One warp per row.
#[derive(Debug, Clone)]
pub struct BatchedSoftmaxKernel {
    /// Total number of rows to process (batch * n_rows)
    pub total_rows: u32,
    /// Size of each row
    pub row_size: u32,
}

impl BatchedSoftmaxKernel {
    /// Create a new batched softmax kernel
    #[must_use]
    pub const fn new(total_rows: u32, row_size: u32) -> Self {
        Self { total_rows, row_size }
    }
}

impl Kernel for BatchedSoftmaxKernel {
    fn name(&self) -> &str {
        "batched_softmax"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Contract: dimension-independent-kernels-v1.yaml (FALSIFY-DIM-004)
        PtxKernel::new("batched_softmax")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "total_rows")
            .param(PtxType::U32, "row_size")
            .shared_memory(72)
            .build(|ctx| {
                let row_idx = ctx.special_reg(PtxReg::CtaIdX);
                let tid = ctx.special_reg(PtxReg::TidX);

                let total_rows_reg = ctx.load_param_u32("total_rows");
                let valid = ctx.setp_lt_u32(row_idx, total_rows_reg);
                ctx.branch_if_not(valid, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let row_size_reg = ctx.load_param_u32("row_size");

                // row_byte_stride = row_size * 4
                let four = ctx.mov_u32_imm(4);
                let row_byte_stride = ctx.mul_lo_u32(row_size_reg, four);
                let row_offset = ctx.mul_wide_u32_reg(row_idx, row_byte_stride);
                let row_input_ptr = ctx.add_u64(input_ptr, row_offset);
                let row_output_ptr = ctx.add_u64(output_ptr, row_offset);

                let four = ctx.mov_u32_imm(4);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);

                // Pass 1: Find max
                let local_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let i_max = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(i_max, tid);
                ctx.label("max_loop");
                let max_done = ctx.setp_ge_u32(i_max, row_size_reg);
                ctx.branch_if(max_done, "max_done");

                let offset = ctx.mul_wide_u32_reg(i_max, four);
                let addr = ctx.add_u64(row_input_ptr, offset);
                let val = ctx.ld_global_f32(addr);
                ctx.max_f32_inplace(local_max, val);
                ctx.add_u32_inplace(i_max, 32);
                ctx.branch("max_loop");

                ctx.label("max_done");

                // Warp reduce for max
                let tmp16 = ctx.shfl_down_f32(local_max, 16, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp16);
                let tmp8 = ctx.shfl_down_f32(local_max, 8, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp8);
                let tmp4 = ctx.shfl_down_f32(local_max, 4, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp4);
                let tmp2 = ctx.shfl_down_f32(local_max, 2, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp2);
                let tmp1 = ctx.shfl_down_f32(local_max, 1, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp1);

                let row_max = ctx.shfl_idx_f32(local_max, 0, 0xFFFF_FFFF);

                // Pass 2: Sum of exp(x - max)
                let local_sum = ctx.mov_f32_imm(0.0);
                let i_sum = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(i_sum, tid);
                ctx.label("sum_loop");
                let sum_done = ctx.setp_ge_u32(i_sum, row_size_reg);
                ctx.branch_if(sum_done, "sum_done");

                let offset = ctx.mul_wide_u32_reg(i_sum, four);
                let addr = ctx.add_u64(row_input_ptr, offset);
                let val = ctx.ld_global_f32(addr);
                let diff = ctx.sub_f32(val, row_max);
                let exp_arg = ctx.mul_f32(diff, log2e);
                let exp_val = ctx.ex2_f32(exp_arg);
                ctx.add_f32_inplace(local_sum, exp_val);
                ctx.add_u32_inplace(i_sum, 32);
                ctx.branch("sum_loop");

                ctx.label("sum_done");

                // Warp reduce for sum
                let stmp16 = ctx.shfl_down_f32(local_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp16);
                let stmp8 = ctx.shfl_down_f32(local_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp8);
                let stmp4 = ctx.shfl_down_f32(local_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp4);
                let stmp2 = ctx.shfl_down_f32(local_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp2);
                let stmp1 = ctx.shfl_down_f32(local_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp1);

                let row_sum = ctx.shfl_idx_f32(local_sum, 0, 0xFFFF_FFFF);

                // Pass 3: Write normalized values
                let i_write = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(i_write, tid);
                ctx.label("write_loop");
                let write_done = ctx.setp_ge_u32(i_write, row_size_reg);
                ctx.branch_if(write_done, "exit");

                let offset = ctx.mul_wide_u32_reg(i_write, four);
                let in_addr = ctx.add_u64(row_input_ptr, offset);
                let out_addr = ctx.add_u64(row_output_ptr, offset);

                let val = ctx.ld_global_f32(in_addr);
                let diff = ctx.sub_f32(val, row_max);
                let exp_arg = ctx.mul_f32(diff, log2e);
                let exp_val = ctx.ex2_f32(exp_arg);
                let normalized = ctx.div_f32(exp_val, row_sum);
                ctx.st_global_f32(out_addr, normalized);

                ctx.add_u32_inplace(i_write, 32);
                ctx.branch("write_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Cast f32 to f16 kernel: output[i] = cvt.rn.f16.f32(input[i])
///
/// Enables FP16 GEMM dispatch by casting fp32 activations to fp16.
/// 256 threads per block, one element per thread.
#[derive(Debug, Clone)]
pub struct CastF32ToF16Kernel;

impl Kernel for CastF32ToF16Kernel {
    fn name(&self) -> &str {
        "cast_f32_to_f16"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("cast_f32_to_f16")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let n = ctx.load_param_u32("n");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "done");

                // Input: f32 at input_ptr + gid * 4
                let four = ctx.mov_u32_imm(4);
                let in_off = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, in_off);

                // Output: f16 at output_ptr + gid * 2
                let two = ctx.mov_u32_imm(2);
                let out_off = ctx.mul_wide_u32_reg(gid, two);
                let out_addr = ctx.add_u64(output_ptr, out_off);

                // Load f32, convert to f16, store
                let val = ctx.ld_global_f32(in_addr);
                let half = ctx.cvt_f16_f32(val);
                ctx.st_global_f16(out_addr, half);

                ctx.label("done");
                ctx.ret();
            })
    }
}
