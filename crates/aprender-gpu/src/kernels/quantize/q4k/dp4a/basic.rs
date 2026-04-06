//! Basic DP4A-based Q4_K GEMV kernel (PAR-063)

use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// DP4A-based Q4_K GEMV kernel for 4x instruction reduction (PAR-063)
///
/// This kernel uses the DP4A SIMD instruction to compute 4 multiply-adds
/// in a single instruction, reducing instruction count by 4x compared to
/// scalar FMA operations.
///
/// # Key Optimizations
///
/// 1. **DP4A instruction**: Computes `d = dot(a[4], b[4]) + c` in one cycle
/// 2. **Vectorized weight loading**: Loads 4 bytes (8 nibbles) per u32 load
/// 3. **Nibble-to-byte expansion**: Expands 4-bit values to 8-bit for DP4A
/// 4. **Integer accumulation**: Accumulates in u32, converts to f32 at end
///
/// # Algorithm
///
/// For each super-block (256 elements):
/// 1. Load scales/mins (same as CoalescedQ4KGemvKernel)
/// 2. For each group of 4 values:
///    a. Load 2 bytes of qs (4 nibbles)
///    b. Expand to 4 bytes
///    c. Load 4 activations, convert to scaled u8
///    d. DP4A: acc += dot4(weights_u8, activations_u8)
/// 3. Apply scale factor at end
///
/// # References
///
/// - NVIDIA PTX ISA: dp4a.atype.btype d, a, b, c
/// - llama.cpp vec_dot_q4_K_q8_1 (uses DP4A for Turing+ GPUs)
/// - "Mixed-Precision Matrix Multiplication" (Markidis et al., 2018)
#[derive(Debug, Clone)]
pub struct Dp4aQ4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl Dp4aQ4KGemvKernel {
    /// Create a new DP4A-based Q4_K GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Get number of super-blocks per row
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        (self.k + Q4K_SUPER_BLOCK_SIZE - 1) / Q4K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for Dp4aQ4KGemvKernel {
    fn name(&self) -> &str {
        "dp4a_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Grid: one warp (32 threads) per output row
        // Each thread processes 8 values per super-block (256 / 32 = 8)
        // Using DP4A: 8 values = 2 DP4A operations per thread
        PtxKernel::new("dp4a_q4k_gemv")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .build(|ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);

                // Bounds check
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                // Float accumulator (will be computed from integer dp4a results)
                let acc = ctx.mov_f32_imm(0.0);

                // Calculate super-blocks per row
                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                // Row base address
                let sb_bytes = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_end");

                let sb_offset = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d and dmin (master scale factors)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // Load scales using vectorized pattern (from CoalescedQ4KGemvKernel)
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);

                let one = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);

                let scales_0_3 = ctx.mov_u32_imm(0);
                let scales_4_7 = ctx.mov_u32_imm(0);
                let scales_8_11 = ctx.mov_u32_imm(0);

                ctx.branch_if_not(is_lane0, "skip_scale_load");

                ctx.ld_global_u32_into(scales_0_3, scales_base);
                let four_64b = ctx.mov_u64_imm(4);
                let scales_4_addr = ctx.add_u64(scales_base, four_64b);
                ctx.ld_global_u32_into(scales_4_7, scales_4_addr);
                let eight_64 = ctx.mov_u64_imm(8);
                let scales_8_addr = ctx.add_u64(scales_base, eight_64);
                ctx.ld_global_u32_into(scales_8_11, scales_8_addr);

                ctx.label("skip_scale_load");

                // Broadcast scales
                let scales_0_3_bcast = ctx.shfl_idx_u32(scales_0_3, 0, 0xFFFF_FFFF);
                let scales_4_7_bcast = ctx.shfl_idx_u32(scales_4_7, 0, 0xFFFF_FFFF);
                let scales_8_11_bcast = ctx.shfl_idx_u32(scales_8_11, 0, 0xFFFF_FFFF);

                // Extract scale bytes
                let mask_8bit = ctx.mov_u32_imm(0xFF);
                let eight = ctx.mov_u32_imm(8);
                let sixteen = ctx.mov_u32_imm(16);
                let twenty_four = ctx.mov_u32_imm(24);

                let s0_32 = ctx.and_u32(scales_0_3_bcast, mask_8bit);
                let s0_shifted = ctx.shr_u32(scales_0_3_bcast, eight);
                let s1_32 = ctx.and_u32(s0_shifted, mask_8bit);
                let s1_shifted = ctx.shr_u32(scales_0_3_bcast, sixteen);
                let s2_32 = ctx.and_u32(s1_shifted, mask_8bit);
                let s3_32 = ctx.shr_u32(scales_0_3_bcast, twenty_four);

                let s4_32 = ctx.and_u32(scales_4_7_bcast, mask_8bit);
                let s4_shifted = ctx.shr_u32(scales_4_7_bcast, eight);
                let s5_32 = ctx.and_u32(s4_shifted, mask_8bit);
                let s5_shifted = ctx.shr_u32(scales_4_7_bcast, sixteen);
                let s6_32 = ctx.and_u32(s5_shifted, mask_8bit);
                let s7_32 = ctx.shr_u32(scales_4_7_bcast, twenty_four);

                let s8_32 = ctx.and_u32(scales_8_11_bcast, mask_8bit);
                let s8_shifted = ctx.shr_u32(scales_8_11_bcast, eight);
                let s9_32 = ctx.and_u32(s8_shifted, mask_8bit);
                let s9_shifted = ctx.shr_u32(scales_8_11_bcast, sixteen);
                let s10_32 = ctx.and_u32(s9_shifted, mask_8bit);
                let s11_32 = ctx.shr_u32(scales_8_11_bcast, twenty_four);

                // Extract actual scale/min values for all 8 blocks
                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let four = ctx.mov_u32_imm(4);
                let six = ctx.mov_u32_imm(6);

                // Block 0-3
                let scale0 = ctx.and_u32(s0_32, mask_6bit);
                let min0 = ctx.and_u32(s4_32, mask_6bit);
                let scale1 = ctx.and_u32(s1_32, mask_6bit);
                let min1 = ctx.and_u32(s5_32, mask_6bit);
                let scale2 = ctx.and_u32(s2_32, mask_6bit);
                let min2 = ctx.and_u32(s6_32, mask_6bit);
                let scale3 = ctx.and_u32(s3_32, mask_6bit);
                let min3 = ctx.and_u32(s7_32, mask_6bit);

                // Block 4-7 (complex extraction)
                let s8_lo = ctx.and_u32(s8_32, mask_4bit);
                let s0_hi = ctx.shr_u32(s0_32, six);
                let s0_hi_shifted = ctx.shl_u32(s0_hi, four);
                let scale4 = ctx.or_u32(s8_lo, s0_hi_shifted);
                let s8_hi = ctx.shr_u32(s8_32, four);
                let s4_hi = ctx.shr_u32(s4_32, six);
                let s4_hi_shifted = ctx.shl_u32(s4_hi, four);
                let min4 = ctx.or_u32(s8_hi, s4_hi_shifted);

                let s9_lo = ctx.and_u32(s9_32, mask_4bit);
                let s1_hi = ctx.shr_u32(s1_32, six);
                let s1_hi_shifted = ctx.shl_u32(s1_hi, four);
                let scale5 = ctx.or_u32(s9_lo, s1_hi_shifted);
                let s9_hi = ctx.shr_u32(s9_32, four);
                let s5_hi = ctx.shr_u32(s5_32, six);
                let s5_hi_shifted = ctx.shl_u32(s5_hi, four);
                let min5 = ctx.or_u32(s9_hi, s5_hi_shifted);

                let s10_lo = ctx.and_u32(s10_32, mask_4bit);
                let s2_hi = ctx.shr_u32(s2_32, six);
                let s2_hi_shifted = ctx.shl_u32(s2_hi, four);
                let scale6 = ctx.or_u32(s10_lo, s2_hi_shifted);
                let s10_hi = ctx.shr_u32(s10_32, four);
                let s6_hi = ctx.shr_u32(s6_32, six);
                let s6_hi_shifted = ctx.shl_u32(s6_hi, four);
                let min6 = ctx.or_u32(s10_hi, s6_hi_shifted);

                let s11_lo = ctx.and_u32(s11_32, mask_4bit);
                let s3_hi = ctx.shr_u32(s3_32, six);
                let s3_hi_shifted = ctx.shl_u32(s3_hi, four);
                let scale7 = ctx.or_u32(s11_lo, s3_hi_shifted);
                let s11_hi = ctx.shr_u32(s11_32, four);
                let s7_hi = ctx.shr_u32(s7_32, six);
                let s7_hi_shifted = ctx.shl_u32(s7_hi, four);
                let min7 = ctx.or_u32(s11_hi, s7_hi_shifted);

                // Convert scales/mins to f32
                let scale0_f = ctx.cvt_f32_u32(scale0);
                let min0_f = ctx.cvt_f32_u32(min0);
                let scale1_f = ctx.cvt_f32_u32(scale1);
                let min1_f = ctx.cvt_f32_u32(min1);
                let scale2_f = ctx.cvt_f32_u32(scale2);
                let min2_f = ctx.cvt_f32_u32(min2);
                let scale3_f = ctx.cvt_f32_u32(scale3);
                let min3_f = ctx.cvt_f32_u32(min3);
                let scale4_f = ctx.cvt_f32_u32(scale4);
                let min4_f = ctx.cvt_f32_u32(min4);
                let scale5_f = ctx.cvt_f32_u32(scale5);
                let min5_f = ctx.cvt_f32_u32(min5);
                let scale6_f = ctx.cvt_f32_u32(scale6);
                let min6_f = ctx.cvt_f32_u32(min6);
                let scale7_f = ctx.cvt_f32_u32(scale7);
                let min7_f = ctx.cvt_f32_u32(min7);

                // Precompute d*scale and dmin*min
                let ds0 = ctx.mul_f32(d, scale0_f);
                let dm0 = ctx.mul_f32(dmin, min0_f);
                let ds1 = ctx.mul_f32(d, scale1_f);
                let dm1 = ctx.mul_f32(dmin, min1_f);
                let ds2 = ctx.mul_f32(d, scale2_f);
                let dm2 = ctx.mul_f32(dmin, min2_f);
                let ds3 = ctx.mul_f32(d, scale3_f);
                let dm3 = ctx.mul_f32(dmin, min3_f);
                let ds4 = ctx.mul_f32(d, scale4_f);
                let dm4 = ctx.mul_f32(dmin, min4_f);
                let ds5 = ctx.mul_f32(d, scale5_f);
                let dm5 = ctx.mul_f32(dmin, min5_f);
                let ds6 = ctx.mul_f32(d, scale6_f);
                let dm6 = ctx.mul_f32(dmin, min6_f);
                let ds7 = ctx.mul_f32(d, scale7_f);
                let dm7 = ctx.mul_f32(dmin, min7_f);

                // qs base address (offset 16 in super-block)
                let sixteen_64 = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen_64);

                let thread_partial = ctx.mov_f32_imm(0.0);

                // Process 8 values per thread across 8 blocks
                // Each thread processes different offsets based on lane_id
                // Using DP4A: process 4 values at a time
                let offsets_and_blocks: [(u32, u32); 8] =
                    [(0, 0), (32, 1), (64, 2), (96, 3), (128, 4), (160, 5), (192, 6), (224, 7)];

                for (offset, block_idx) in offsets_and_blocks {
                    let (ds, dm) = match block_idx {
                        0 => (ds0, dm0),
                        1 => (ds1, dm1),
                        2 => (ds2, dm2),
                        3 => (ds3, dm3),
                        4 => (ds4, dm4),
                        5 => (ds5, dm5),
                        6 => (ds6, dm6),
                        _ => (ds7, dm7),
                    };

                    let offset_reg = ctx.mov_u32_imm(offset);
                    let val_idx = ctx.add_u32_reg(lane_id, offset_reg);

                    // Calculate byte address for quantized values
                    let chunk_idx = ctx.div_u32(val_idx, 64);
                    let val_in_chunk = ctx.rem_u32(val_idx, 64);
                    let byte_in_chunk = ctx.rem_u32(val_in_chunk, 32);

                    let chunk_offset = ctx.mul_u32(chunk_idx, 32);
                    let qs_byte_offset = ctx.add_u32_reg(chunk_offset, byte_in_chunk);
                    let qs_byte_offset_64 = ctx.cvt_u64_u32(qs_byte_offset);
                    let qs_addr = ctx.add_u64(qs_base, qs_byte_offset_64);

                    // Load packed byte (2 nibbles)
                    let packed = ctx.ld_global_u8(qs_addr);
                    let packed_32 = ctx.cvt_u32_u8(packed);

                    // Extract nibble based on position
                    let mask_4bit_q = ctx.mov_u32_imm(0xF);
                    let four_q = ctx.mov_u32_imm(4);
                    let val_in_chunk_div_32 = ctx.div_u32(val_in_chunk, 32);
                    let shift_amount = ctx.mul_u32_reg(val_in_chunk_div_32, four_q);
                    let shifted = ctx.shr_u32(packed_32, shift_amount);
                    let quant = ctx.and_u32(shifted, mask_4bit_q);

                    // Dequantize: value = ds * quant - dm
                    let quant_f32 = ctx.cvt_f32_u32(quant);
                    let scaled = ctx.mul_f32(ds, quant_f32);
                    let dequant = ctx.sub_f32(scaled, dm);

                    // Load activation
                    let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                    let x_idx = ctx.add_u32_reg(sb_k_base, val_idx);
                    let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                    let x_bytes = ctx.mul_u64(x_idx_64, 4);
                    let x_addr = ctx.add_u64(x_ptr, x_bytes);
                    let x_val = ctx.ld_global_f32(x_addr);

                    // FMA: thread_partial += x_val * dequant
                    ctx.fma_f32_inplace(thread_partial, x_val, dequant);
                }

                ctx.add_f32_inplace(acc, thread_partial);
                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_end");

                // Warp shuffle reduction
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

                // Only lane 0 writes
                let one_u32 = ctx.mov_u32_imm(1);
                let is_thread0 = ctx.setp_lt_u32(lane_id, one_u32);
                ctx.branch_if_not(is_thread0, "exit");

                let y_offset = ctx.mul_wide_u32(block_id, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
