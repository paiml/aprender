//! Q4_0 GEMV Kernel
//!
//! 4-bit centered quantization: nibble - 8, with fp16 scale per block.

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Q4_0 block size (32 values per block)
const Q4_0_BLOCK_SIZE: u32 = 32;
/// Q4_0 bytes per block: 2 (d fp16) + 16 (qs) = 18 bytes
const Q4_0_BLOCK_BYTES: u32 = 18;

/// Q4_0 GEMV kernel (fused dequantization + matrix-vector multiply).
///
/// Q4_0 format: 18 bytes per 32 elements (2-byte fp16 scale + 16 bytes packed nibbles).
/// This format is simpler than Q4_K but has a higher compression ratio.
/// Used when GGUF header claims a different qtype but data is actually Q4_0.
///
/// # PTX Implementation
///
/// Each warp processes one output element (row of weight matrix).
/// Lane i processes elements [i, i+32, i+64, ...] within the row.
/// Uses warp shuffle reduction to sum across lanes.
///
/// ```text
/// Block layout: [scale: f16, packed: u8[16]]  (18 bytes for 32 elements)
/// Nibble extraction: low = byte & 0x0F, high = (byte >> 4) & 0x0F
/// Dequant: value = (nibble - 8) * scale
/// ```
#[derive(Debug, Clone)]
pub struct Q4_0GemvKernel {
    /// K dimension (input dimension)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl Q4_0GemvKernel {
    /// Create a new Q4_0 GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Get number of blocks per row (ceiling division)
    #[must_use]
    pub const fn num_blocks_per_row(&self) -> u32 {
        (self.k + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE
    }
}

impl Kernel for Q4_0GemvKernel {
    fn name(&self) -> &str {
        "q4_0_gemv_warp_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("q4_0_gemv_warp_reduce")
            .param(PtxType::U64, "y_ptr") // Output vector (N)
            .param(PtxType::U64, "w_ptr") // Q4_0 weights (N x K/32 blocks)
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
                let k_rounded = ctx.add_u32(k_dim, Q4_0_BLOCK_SIZE - 1);
                let num_blocks = ctx.div_u32(k_rounded, Q4_0_BLOCK_SIZE);

                // Row base address: w_ptr + block_id * num_blocks * 18
                let block_bytes = ctx.mov_u32_imm(Q4_0_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_blocks, block_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Loop over blocks (each thread handles one value per block)
                let blk_idx = ctx.mov_u32_imm(0);

                ctx.label("blk_loop");
                let blk_done = ctx.setp_ge_u32(blk_idx, num_blocks);
                ctx.branch_if(blk_done, "blk_loop_end");

                // Block address = row_base + blk_idx * 18
                let blk_offset = ctx.mul_wide_u32(blk_idx, Q4_0_BLOCK_BYTES);
                let blk_addr = ctx.add_u64(row_base, blk_offset);

                // Load scale d (fp16 at offset 0)
                let d_f16 = ctx.ld_global_f16(blk_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Load nibble for this thread from qs (offset 2)
                // qs layout: 32 4-bit values packed into 16 bytes
                // Nibble index = thread_id, byte index = thread_id / 2
                // Low/high nibble = thread_id % 2
                let two_64 = ctx.mov_u64_imm(2);
                let qs_base = ctx.add_u64(blk_addr, two_64);

                // byte_idx = thread_id / 2
                let byte_idx = ctx.div_u32(thread_id, 2);
                let byte_idx_64 = ctx.cvt_u64_u32(byte_idx);
                let qs_addr = ctx.add_u64(qs_base, byte_idx_64);

                // Load the byte containing our nibble
                let qs_byte = ctx.ld_global_u8(qs_addr);
                let qs_byte_u32 = ctx.cvt_u32_u8(qs_byte);

                // Extract nibble: if thread_id is odd, use high nibble (>> 4)
                // nibble_select = (thread_id % 2) * 4 = (thread_id & 1) << 2
                let one_u32 = ctx.mov_u32_imm(1);
                let nibble_select = ctx.and_u32(thread_id, one_u32);
                let shift_amount = ctx.mul_u32(nibble_select, 4);
                let shifted = ctx.shr_u32(qs_byte_u32, shift_amount);
                let fifteen_u32 = ctx.mov_u32_imm(15);
                let nibble = ctx.and_u32(shifted, fifteen_u32);

                // Center: q_centered = nibble - 8 (result may be negative, -8 to +7)
                let eight_u32 = ctx.mov_u32_imm(8);
                let q_centered = ctx.sub_u32_reg(nibble, eight_u32);

                // Convert to float and dequantize
                let q_f32 = ctx.cvt_f32_s32(q_centered);
                let dequant = ctx.mul_f32(d, q_f32);

                // Load activation x[blk_idx * 32 + thread_id]
                let blk_k_base = ctx.mul_u32(blk_idx, Q4_0_BLOCK_SIZE);
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
