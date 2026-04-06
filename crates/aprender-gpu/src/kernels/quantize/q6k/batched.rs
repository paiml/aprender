//! Batched Q6_K GEMV kernel for M>1 batch processing (PAR-130).

use crate::kernels::quantize::{Kernel, Q6K_SUPER_BLOCK_BYTES, Q6K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

// =============================================================================
// BATCHED Q6_K GEMV KERNEL (PAR-130)
// =============================================================================
//
// Batched version of CoalescedQ6KGemvKernel for M>1 batch processing.
// Eliminates 896 sequential kernel launches for M=32 batch decode.
//
// Strategy:
// - One warp (32 threads) per output row
// - Each thread processes 8 elements per super-block (256/32 = 8)
// - All M batch elements processed within single kernel launch
// - Weights loaded once, reused for all M inputs (L1 cache efficient)
//
// Memory: Q6K = 210 bytes per 256 values = 0.82 bytes/value

/// Batched Q6_K GEMV kernel for batch decode throughput (PAR-130)
///
/// Processes M input vectors against the same weight matrix in one kernel launch.
/// This eliminates M-1 kernel launches per layer, critical for batched decode.
#[derive(Debug, Clone)]
pub struct BatchedQ6KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// M dimension (batch size)
    pub m: u32,
}

impl BatchedQ6KGemvKernel {
    /// Create a new batched Q6_K GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32, m: u32) -> Self {
        Self { k, n, m }
    }

    /// Get number of super-blocks per row
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        (self.k + Q6K_SUPER_BLOCK_SIZE - 1) / Q6K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for BatchedQ6KGemvKernel {
    fn name(&self) -> &str {
        "batched_q6k_gemv_warp_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        let m = self.m;
        PtxKernel::new("batched_q6k_gemv_warp_reduce")
            .param(PtxType::U64, "y_ptr") // Output matrix (M x N)
            .param(PtxType::U64, "w_ptr") // Q6_K weights (N x K/256 super-blocks)
            .param(PtxType::U64, "x_ptr") // Input matrix (M x K)
            .param(PtxType::U32, "k_dim") // K dimension
            .param(PtxType::U32, "n_dim") // N dimension
            .param(PtxType::U32, "m_dim") // M dimension (batch size)
            .build(move |ctx| {
                // Block = 32 threads (one warp), grid = N blocks
                // Each block computes one output row: y[:, block_id]
                //
                // Uses the same Q6K dequantization as Q6KGemvKernel (single-vector):
                // - 8 strided offsets per thread: [0, 32, 64, 96, 128, 160, 192, 224]
                // - Proper Q6K super-block layout with half-blocks, groups, and
                //   correct ql/qh bit combination: ql_nibble | (qh_2bits << 4) - 32

                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                // Bounds check
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let _m_dim = ctx.load_param_u32("m_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                // Initialize M accumulators
                let mut accs = Vec::with_capacity(m as usize);
                for _ in 0..m {
                    accs.push(ctx.mov_f32_imm(0.0));
                }

                // Calculate super-blocks per row
                let k_rounded = ctx.add_u32(k_dim, Q6K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q6K_SUPER_BLOCK_SIZE);

                // Row base address for weights
                let sb_bytes = ctx.mov_u32_imm(Q6K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_end");

                let sb_offset = ctx.mul_wide_u32(sb_idx, Q6K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d (f16 at offset 208)
                let d_offset = ctx.mov_u64_imm(208);
                let d_addr = ctx.add_u64(sb_addr, d_offset);
                let d_f16 = ctx.ld_global_f16(d_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Initialize per-thread partial sums for all M batch elements
                let mut thread_partials = Vec::with_capacity(m as usize);
                for _ in 0..m {
                    thread_partials.push(ctx.mov_f32_imm(0.0));
                }

                // Process 8 values per thread at strided offsets (matching Q6KGemvKernel)
                // Each of 32 threads handles values at: thread_id + [0, 32, 64, 96, 128, 160, 192, 224]
                for offset in [0u32, 32, 64, 96, 128, 160, 192, 224] {
                    let offset_reg = ctx.mov_u32_imm(offset);
                    let val_idx = ctx.add_u32_reg(thread_id, offset_reg);

                    // Q6K super-block layout (from llama.cpp):
                    // 256 values split into two 128-value halves
                    // Each half has 4 groups of 32 values
                    let n_idx = ctx.div_u32(val_idx, 128);
                    let pos = ctx.rem_u32(val_idx, 128);
                    let group = ctx.div_u32(pos, 32);
                    let l = ctx.rem_u32(pos, 32);
                    let is = ctx.div_u32(l, 16);

                    // scale_idx = 8 * n_idx + is + 2 * group
                    let eight = ctx.mov_u32_imm(8);
                    let two = ctx.mov_u32_imm(2);
                    let n_idx_x8 = ctx.mul_u32_reg(n_idx, eight);
                    let group_x2 = ctx.mul_u32_reg(group, two);
                    let scale_idx_temp = ctx.add_u32_reg(n_idx_x8, is);
                    let scale_idx = ctx.add_u32_reg(scale_idx_temp, group_x2);

                    // Load scale (signed i8 at offset 192 + scale_idx)
                    let scales_offset = ctx.mov_u64_imm(192);
                    let scales_base = ctx.add_u64(sb_addr, scales_offset);
                    let scale_idx_64 = ctx.cvt_u64_u32(scale_idx);
                    let scale_addr = ctx.add_u64(scales_base, scale_idx_64);
                    let scale_u8 = ctx.ld_global_u8(scale_addr);
                    let scale_u32 = ctx.cvt_u32_u8(scale_u8);
                    let seven = ctx.mov_u32_imm(7);
                    let sign_bit = ctx.shr_u32(scale_u32, seven);
                    let scale_u32_f32 = ctx.cvt_f32_u32(scale_u32);
                    let sign_bit_f32 = ctx.cvt_f32_u32(sign_bit);
                    let twofiftysix_f32 = ctx.mov_f32_imm(256.0);
                    let correction_f32 = ctx.mul_f32(sign_bit_f32, twofiftysix_f32);
                    let scale_f32 = ctx.sub_f32(scale_u32_f32, correction_f32);

                    // ql_byte_offset = 64 * n_idx + l + 32 * (group & 1)
                    let sixty_four = ctx.mov_u32_imm(64);
                    let thirty_two = ctx.mov_u32_imm(32);
                    let one = ctx.mov_u32_imm(1);
                    let n_idx_x64 = ctx.mul_u32_reg(n_idx, sixty_four);
                    let ql_base = ctx.add_u32_reg(n_idx_x64, l);
                    let group_is_odd = ctx.and_u32(group, one);
                    let ql_offset_add = ctx.mul_u32_reg(group_is_odd, thirty_two);
                    let ql_byte_offset = ctx.add_u32_reg(ql_base, ql_offset_add);

                    // Load ql byte and extract nibble
                    let ql_byte_offset_64 = ctx.cvt_u64_u32(ql_byte_offset);
                    let ql_addr = ctx.add_u64(sb_addr, ql_byte_offset_64);
                    let ql_byte = ctx.ld_global_u8(ql_addr);
                    let ql_byte_32 = ctx.cvt_u32_u8(ql_byte);
                    // nibble_shift = (group / 2) * 4: low nibble for groups 0,1; high for 2,3
                    let group_div_2 = ctx.shr_u32(group, one);
                    let four = ctx.mov_u32_imm(4);
                    let nibble_shift = ctx.mul_u32_reg(group_div_2, four);
                    let ql_shifted = ctx.shr_u32(ql_byte_32, nibble_shift);
                    let mask_0xf = ctx.mov_u32_imm(0xF);
                    let ql_nibble = ctx.and_u32(ql_shifted, mask_0xf);

                    // qh_byte_offset = 32 * n_idx + l
                    let n_idx_x32 = ctx.mul_u32_reg(n_idx, thirty_two);
                    let qh_byte_offset = ctx.add_u32_reg(n_idx_x32, l);

                    // Load qh byte (offset 128 + qh_byte_offset)
                    let qh_base_offset = ctx.mov_u64_imm(128);
                    let qh_base = ctx.add_u64(sb_addr, qh_base_offset);
                    let qh_byte_offset_64 = ctx.cvt_u64_u32(qh_byte_offset);
                    let qh_addr = ctx.add_u64(qh_base, qh_byte_offset_64);
                    let qh_byte = ctx.ld_global_u8(qh_addr);
                    let qh_byte_32 = ctx.cvt_u32_u8(qh_byte);

                    // qh_bit_shift = 2 * group
                    let qh_shift = ctx.mul_u32_reg(group, two);
                    let qh_shifted = ctx.shr_u32(qh_byte_32, qh_shift);
                    let mask_0x3 = ctx.mov_u32_imm(0x3);
                    let qh_2bits = ctx.and_u32(qh_shifted, mask_0x3);

                    // Combine: quant = ql_nibble | (qh_2bits << 4) - 32
                    let qh_shifted_up = ctx.shl_u32(qh_2bits, four);
                    let combined = ctx.or_u32(ql_nibble, qh_shifted_up);
                    let combined_f32 = ctx.cvt_f32_u32(combined);
                    let thirty_two_f32 = ctx.mov_f32_imm(32.0);
                    let quant_signed = ctx.sub_f32(combined_f32, thirty_two_f32);

                    // Dequantize: val = d * scale * quant
                    let d_scale = ctx.mul_f32(d, scale_f32);
                    let dequant = ctx.mul_f32(d_scale, quant_signed);

                    // Calculate K index
                    let sb_k_base = ctx.mul_u32(sb_idx, Q6K_SUPER_BLOCK_SIZE);
                    let k_idx = ctx.add_u32_reg(sb_k_base, val_idx);

                    // GH-215 FIX: Bounds-check for non-256-aligned K dimensions.
                    let in_bounds = ctx.setp_lt_u32(k_idx, k_dim);

                    // Accumulate for all M batch elements
                    for batch_idx in 0..m as usize {
                        // Load x[batch_idx, k_idx]
                        let batch_offset = ctx.mov_u32_imm((batch_idx as u32) * self.k);
                        let x_offset = ctx.add_u32_reg(batch_offset, k_idx);
                        let x_offset_64 = ctx.cvt_u64_u32(x_offset);
                        let x_bytes = ctx.mul_u64(x_offset_64, 4);
                        let x_addr = ctx.add_u64(x_ptr, x_bytes);
                        let x_val = ctx.ld_global_f32_predicated(x_addr, in_bounds, 0.0);

                        ctx.fma_f32_inplace(thread_partials[batch_idx], x_val, dequant);
                    }
                }

                // Accumulate thread partials into main accumulators
                for batch_idx in 0..m as usize {
                    ctx.add_f32_inplace(accs[batch_idx], thread_partials[batch_idx]);
                }

                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_end");

                // Warp reduce each accumulator and store
                for batch_idx in 0..m as usize {
                    let tmp16 = ctx.shfl_down_f32(accs[batch_idx], 16, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(accs[batch_idx], tmp16);
                    let tmp8 = ctx.shfl_down_f32(accs[batch_idx], 8, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(accs[batch_idx], tmp8);
                    let tmp4 = ctx.shfl_down_f32(accs[batch_idx], 4, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(accs[batch_idx], tmp4);
                    let tmp2 = ctx.shfl_down_f32(accs[batch_idx], 2, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(accs[batch_idx], tmp2);
                    let tmp1 = ctx.shfl_down_f32(accs[batch_idx], 1, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(accs[batch_idx], tmp1);
                }

                // Only lane 0 writes
                let one_u32 = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(thread_id, one_u32);
                ctx.branch_if_not(is_lane0, "exit");

                // Write M outputs: y[batch_idx, block_id]
                for batch_idx in 0..m as usize {
                    // y[batch_idx * n + block_id]
                    let batch_row_offset = ctx.mov_u32_imm((batch_idx as u32) * self.n);
                    let y_idx = ctx.add_u32_reg(batch_row_offset, block_id);
                    let y_offset = ctx.mul_wide_u32(y_idx, 4);
                    let y_addr = ctx.add_u64(y_ptr, y_offset);
                    ctx.st_global_f32(y_addr, accs[batch_idx]);
                }

                ctx.label("exit");
                ctx.ret();
            })
    }
}
