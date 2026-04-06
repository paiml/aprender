//! Q8_0 GEMV Kernel
//!
//! 8-bit quantization: 32 int8 values + fp16 scale per block.

use super::{Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Q8_0 quantized GEMV kernel for M=1 decode throughput
///
/// Q8_0 is simpler than Q4K: 32 int8 values + 1 fp16 scale per block.
/// Layout: d (fp16, 2 bytes) + qs[32] (32 int8 values) = 34 bytes per block
/// Dequant: value[i] = d * qs[i]
#[derive(Debug, Clone)]
pub struct Q8_0GemvKernel {
    /// K dimension (input dimension)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl Q8_0GemvKernel {
    /// Create a new Q8_0 GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Get number of blocks per row (ceiling division)
    #[must_use]
    pub const fn num_blocks_per_row(&self) -> u32 {
        (self.k + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE
    }
}

impl Kernel for Q8_0GemvKernel {
    fn name(&self) -> &str {
        "q8_0_gemv_warp_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("q8_0_gemv_warp_reduce")
            .param(PtxType::U64, "y_ptr") // Output vector (N)
            .param(PtxType::U64, "w_ptr") // Q8_0 weights (N x K/32 blocks)
            .param(PtxType::U64, "x_ptr") // Input vector (K)
            .param(PtxType::U32, "k_dim") // K dimension
            .param(PtxType::U32, "n_dim") // N dimension
            .build(|ctx| {
                // Block = 32 threads (one warp), grid = N blocks
                // Each block computes one output element y[block_id]

                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                // Bounds check
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                // Number of blocks per row: ceil(K / 32)
                let k_rounded = ctx.add_u32(k_dim, Q8_0_BLOCK_SIZE - 1);
                let num_blocks = ctx.div_u32(k_rounded, Q8_0_BLOCK_SIZE);

                // Row base address: w_ptr + block_id * num_blocks * 34
                let block_bytes = ctx.mov_u32_imm(Q8_0_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_blocks, block_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Loop over blocks (each thread handles one value per block)
                let blk_idx = ctx.mov_u32_imm(0);

                ctx.label("blk_loop");
                let blk_done = ctx.setp_ge_u32(blk_idx, num_blocks);
                ctx.branch_if(blk_done, "blk_loop_end");

                // Block address = row_base + blk_idx * 34
                let blk_offset = ctx.mul_wide_u32(blk_idx, Q8_0_BLOCK_BYTES);
                let blk_addr = ctx.add_u64(row_base, blk_offset);

                // Load scale d (fp16 at offset 0)
                let d_f16 = ctx.ld_global_f16(blk_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Load quantized value qs[thread_id] (int8 at offset 2 + thread_id)
                let two_64 = ctx.mov_u64_imm(2);
                let qs_base = ctx.add_u64(blk_addr, two_64);
                let tid_64 = ctx.cvt_u64_u32(thread_id);
                let qs_addr = ctx.add_u64(qs_base, tid_64);
                let q_u8 = ctx.ld_global_u8(qs_addr);

                // Convert int8 to signed: treat as signed byte
                // PTX cvt.s32.s8 interprets the byte as signed
                let q_s32 = ctx.cvt_s32_s8(q_u8);
                let q_f32 = ctx.cvt_f32_s32(q_s32);

                // Dequantize: val = d * q
                let dequant = ctx.mul_f32(d, q_f32);

                // Load activation x[blk_idx * 32 + thread_id]
                let blk_k_base = ctx.mul_u32(blk_idx, Q8_0_BLOCK_SIZE);
                let x_idx = ctx.add_u32_reg(blk_k_base, thread_id);

                // Bounds check for last block (K may not be multiple of 32)
                let x_oob = ctx.setp_ge_u32(x_idx, k_dim);
                ctx.branch_if(x_oob, "skip_mul");

                let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                let x_bytes = ctx.mul_u64(x_idx_64, 4);
                let x_addr = ctx.add_u64(x_ptr, x_bytes);
                let x_val = ctx.ld_global_f32(x_addr);

                ctx.fma_f32_inplace(acc, x_val, dequant);

                ctx.label("skip_mul");
                ctx.add_u32_inplace(blk_idx, 1);
                ctx.branch("blk_loop");

                ctx.label("blk_loop_end");

                // Warp reduce
                let tmp16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp16);
                let tmp8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp8);
                let tmp4 = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp4);
                let tmp2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp2);
                let tmp1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp1);

                // Thread 0 writes result
                let one_u32 = ctx.mov_u32_imm(1);
                let is_thread0 = ctx.setp_lt_u32(thread_id, one_u32);
                ctx.branch_if_not(is_thread0, "exit");

                let y_offset = ctx.mul_wide_u32(block_id, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
