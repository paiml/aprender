//! Batched Q4_K GEMV Kernel (PAR-108: 2x Ollama via dequant sharing)
//!
//! Performance insight: Sequential GEMV dequantizes weights M times for M requests.
//! Batched GEMV dequantizes once and multiplies by M different inputs.
//! This amortizes the ALU-bound dequantization cost, approaching memory bandwidth limit.

use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Batched Q4_K GEMV kernel for M>1 continuous batching throughput
///
/// PAR-108: Key optimization for 2x Ollama target
///
/// Performance insight: Sequential GEMV dequantizes weights M times for M requests.
/// Batched GEMV dequantizes once and multiplies by M different inputs.
/// This amortizes the ALU-bound dequantization cost, approaching memory bandwidth limit.
///
/// Layout:
/// - x: M × K input matrix (row-major, M batch elements, K elements each)
/// - W: N × K weight matrix (Q4_K quantized, N output rows, K/256 super-blocks per row)
/// - y: M × N output matrix (row-major, M batch elements, N outputs each)
///
/// Thread organization:
/// - Grid: N blocks (one per output row)
/// - Block: 32 threads (one warp)
/// - Each thread maintains M accumulators (unrolled for M <= 8)
#[derive(Debug, Clone)]
pub struct BatchedQ4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// M dimension (batch size, max 8 for register unrolling)
    pub m: u32,
}

impl BatchedQ4KGemvKernel {
    /// Create a new batched Q4_K GEMV kernel for Y = X * W^T
    ///
    /// # Arguments
    /// * `k` - Input vector length / weight matrix columns (must be multiple of 256)
    /// * `n` - Output vector length / weight matrix rows
    /// * `m` - Batch size (any size supported via tiling for M>8)
    #[must_use]
    pub fn new(k: u32, n: u32, m: u32) -> Self {
        // PAR-129 FIX: Support M>8 by tiling (process 8 at a time internally)
        // For M<=8, uses register unrolling. For M>8, loops over tiles.
        Self { k, n, m }
    }

    /// Get number of super-blocks per row (ceiling division)
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        (self.k + Q4K_SUPER_BLOCK_SIZE - 1) / Q4K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for BatchedQ4KGemvKernel {
    fn name(&self) -> &str {
        "batched_q4k_gemv_warp_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        let m = self.m;
        // No shared memory needed - each warp works independently
        PtxKernel::new("batched_q4k_gemv_warp_reduce")
            .param(PtxType::U64, "y_ptr") // Output matrix (M × N)
            .param(PtxType::U64, "w_ptr") // Q4_K weights (N × K/256 super-blocks)
            .param(PtxType::U64, "x_ptr") // Input matrix (M × K)
            .param(PtxType::U32, "k_dim") // K dimension
            .param(PtxType::U32, "n_dim") // N dimension
            .param(PtxType::U32, "m_dim") // M dimension (batch size)
            .build(move |ctx| {
                // Block = 32 threads (one warp), grid = N blocks
                // Each block computes one output row: y[:, block_id]

                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                // Bounds check: if block_id >= n_dim, return
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                // Load parameters
                let k_dim = ctx.load_param_u32("k_dim");
                let _m_dim = ctx.load_param_u32("m_dim"); // Not used at runtime (m is compile-time constant)
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                // Initialize M accumulators (unrolled for M <= 8)
                let mut accs = Vec::with_capacity(m as usize);
                for _ in 0..m {
                    accs.push(ctx.mov_f32_imm(0.0));
                }

                // Calculate number of super-blocks per row
                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                // Calculate base address for this row's Q4_K data
                let sb_bytes = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Loop over super-blocks
                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_end");

                // Calculate super-block address
                let sb_offset = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // ============================================================
                // DEQUANTIZATION (shared across all M batch elements)
                // ============================================================

                // Load d (f16 at offset 0)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Load dmin (f16 at offset 2)
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // scales base = sb_addr + 4
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);

                // ========================================================
                // PAR-125 OPTIMIZATION: Vectorized scale loading
                // Load 12 bytes as 3 x u32 instead of 12 x u8
                // All threads load (L1 cache handles redundancy)
                // Reduces instruction count and improves coalescing
                // ========================================================

                // Load scales as 3 x u32 (all threads, L1 cached)
                let scales_0_3 = ctx.ld_global_u32(scales_base);
                let four_64b = ctx.mov_u64_imm(4);
                let scales_4_addr = ctx.add_u64(scales_base, four_64b);
                let scales_4_7 = ctx.ld_global_u32(scales_4_addr);
                let eight_64 = ctx.mov_u64_imm(8);
                let scales_8_addr = ctx.add_u64(scales_base, eight_64);
                let scales_8_11 = ctx.ld_global_u32(scales_8_addr);

                // Extract individual scale bytes using bit operations
                let mask_8bit = ctx.mov_u32_imm(0xFF);
                let eight_const = ctx.mov_u32_imm(8);
                let sixteen = ctx.mov_u32_imm(16);
                let twenty_four = ctx.mov_u32_imm(24);

                // s0-s3 from scales_0_3
                let s0_32 = ctx.and_u32(scales_0_3, mask_8bit);
                let s0_shifted = ctx.shr_u32(scales_0_3, eight_const);
                let s1_32 = ctx.and_u32(s0_shifted, mask_8bit);
                let s1_shifted = ctx.shr_u32(scales_0_3, sixteen);
                let s2_32 = ctx.and_u32(s1_shifted, mask_8bit);
                let s3_32 = ctx.shr_u32(scales_0_3, twenty_four);

                // s4-s7 from scales_4_7
                let s4_32 = ctx.and_u32(scales_4_7, mask_8bit);
                let s4_shifted = ctx.shr_u32(scales_4_7, eight_const);
                let s5_32 = ctx.and_u32(s4_shifted, mask_8bit);
                let s5_shifted = ctx.shr_u32(scales_4_7, sixteen);
                let s6_32 = ctx.and_u32(s5_shifted, mask_8bit);
                let s7_32 = ctx.shr_u32(scales_4_7, twenty_four);

                // s8-s11 from scales_8_11
                let s8_32 = ctx.and_u32(scales_8_11, mask_8bit);
                let s8_shifted = ctx.shr_u32(scales_8_11, eight_const);
                let s9_32 = ctx.and_u32(s8_shifted, mask_8bit);
                let s9_shifted = ctx.shr_u32(scales_8_11, sixteen);
                let s10_32 = ctx.and_u32(s9_shifted, mask_8bit);
                let s11_32 = ctx.shr_u32(scales_8_11, twenty_four);

                // Constants
                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let four = ctx.mov_u32_imm(4);
                let six = ctx.mov_u32_imm(6);

                // Extract scale/min for all 8 blocks using get_scale_min_k4 logic
                let scale0 = ctx.and_u32(s0_32, mask_6bit);
                let min0 = ctx.and_u32(s4_32, mask_6bit);
                let scale0_f = ctx.cvt_f32_u32(scale0);
                let min0_f = ctx.cvt_f32_u32(min0);

                let scale1 = ctx.and_u32(s1_32, mask_6bit);
                let min1 = ctx.and_u32(s5_32, mask_6bit);
                let scale1_f = ctx.cvt_f32_u32(scale1);
                let min1_f = ctx.cvt_f32_u32(min1);

                let scale2 = ctx.and_u32(s2_32, mask_6bit);
                let min2 = ctx.and_u32(s6_32, mask_6bit);
                let scale2_f = ctx.cvt_f32_u32(scale2);
                let min2_f = ctx.cvt_f32_u32(min2);

                let scale3 = ctx.and_u32(s3_32, mask_6bit);
                let min3 = ctx.and_u32(s7_32, mask_6bit);
                let scale3_f = ctx.cvt_f32_u32(scale3);
                let min3_f = ctx.cvt_f32_u32(min3);

                let s8_lo = ctx.and_u32(s8_32, mask_4bit);
                let s0_hi = ctx.shr_u32(s0_32, six);
                let s0_hi_shifted = ctx.shl_u32(s0_hi, four);
                let scale4 = ctx.or_u32(s8_lo, s0_hi_shifted);
                let s8_hi = ctx.shr_u32(s8_32, four);
                let s4_hi = ctx.shr_u32(s4_32, six);
                let s4_hi_shifted = ctx.shl_u32(s4_hi, four);
                let min4 = ctx.or_u32(s8_hi, s4_hi_shifted);
                let scale4_f = ctx.cvt_f32_u32(scale4);
                let min4_f = ctx.cvt_f32_u32(min4);

                let s9_lo = ctx.and_u32(s9_32, mask_4bit);
                let s1_hi = ctx.shr_u32(s1_32, six);
                let s1_hi_shifted = ctx.shl_u32(s1_hi, four);
                let scale5 = ctx.or_u32(s9_lo, s1_hi_shifted);
                let s9_hi = ctx.shr_u32(s9_32, four);
                let s5_hi = ctx.shr_u32(s5_32, six);
                let s5_hi_shifted = ctx.shl_u32(s5_hi, four);
                let min5 = ctx.or_u32(s9_hi, s5_hi_shifted);
                let scale5_f = ctx.cvt_f32_u32(scale5);
                let min5_f = ctx.cvt_f32_u32(min5);

                let s10_lo = ctx.and_u32(s10_32, mask_4bit);
                let s2_hi = ctx.shr_u32(s2_32, six);
                let s2_hi_shifted = ctx.shl_u32(s2_hi, four);
                let scale6 = ctx.or_u32(s10_lo, s2_hi_shifted);
                let s10_hi = ctx.shr_u32(s10_32, four);
                let s6_hi = ctx.shr_u32(s6_32, six);
                let s6_hi_shifted = ctx.shl_u32(s6_hi, four);
                let min6 = ctx.or_u32(s10_hi, s6_hi_shifted);
                let scale6_f = ctx.cvt_f32_u32(scale6);
                let min6_f = ctx.cvt_f32_u32(min6);

                let s11_lo = ctx.and_u32(s11_32, mask_4bit);
                let s3_hi = ctx.shr_u32(s3_32, six);
                let s3_hi_shifted = ctx.shl_u32(s3_hi, four);
                let scale7 = ctx.or_u32(s11_lo, s3_hi_shifted);
                let s11_hi = ctx.shr_u32(s11_32, four);
                let s7_hi = ctx.shr_u32(s7_32, six);
                let s7_hi_shifted = ctx.shl_u32(s7_hi, four);
                let min7 = ctx.or_u32(s11_hi, s7_hi_shifted);
                let scale7_f = ctx.cvt_f32_u32(scale7);
                let min7_f = ctx.cvt_f32_u32(min7);

                // Precompute d*scale and dmin*min for each block
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

                // qs base = sb_addr + 16
                let sixteen_64 = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen_64);

                // Each thread handles 8 values (256 values / 32 threads)
                let thread_partials: Vec<_> = (0..m).map(|_| ctx.mov_f32_imm(0.0)).collect();

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
                    let val_idx = ctx.add_u32_reg(thread_id, offset_reg);

                    // Load quantized value (same for all M batch elements)
                    let chunk_idx = ctx.div_u32(val_idx, 64);
                    let val_in_chunk = ctx.rem_u32(val_idx, 64);
                    let byte_in_chunk = ctx.rem_u32(val_in_chunk, 32);

                    let chunk_offset = ctx.mul_u32(chunk_idx, 32);
                    let qs_byte_offset = ctx.add_u32_reg(chunk_offset, byte_in_chunk);
                    let qs_byte_offset_64 = ctx.cvt_u64_u32(qs_byte_offset);
                    let qs_addr = ctx.add_u64(qs_base, qs_byte_offset_64);
                    let packed = ctx.ld_global_u8(qs_addr);
                    let packed_32 = ctx.cvt_u32_u8(packed);

                    let mask_4bit_q = ctx.mov_u32_imm(0xF);
                    let four_q = ctx.mov_u32_imm(4);
                    let val_in_chunk_div_32 = ctx.div_u32(val_in_chunk, 32);
                    let shift_amount = ctx.mul_u32_reg(val_in_chunk_div_32, four_q);
                    let shifted = ctx.shr_u32(packed_32, shift_amount);
                    let quant = ctx.and_u32(shifted, mask_4bit_q);

                    // Dequantize ONCE (shared across all M)
                    let quant_f32 = ctx.cvt_f32_u32(quant);
                    let scaled = ctx.mul_f32(ds, quant_f32);
                    let dequant = ctx.sub_f32(scaled, dm);

                    // Calculate base x index
                    let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                    let x_elem_idx = ctx.add_u32_reg(sb_k_base, val_idx);

                    // Process each batch element (unrolled for M <= 8)
                    // x layout: M × K row-major, so x[m][k] = x_ptr + m * k_dim + k
                    for batch_m in 0..m {
                        // x_addr = x_ptr + (batch_m * k_dim + x_elem_idx) * 4
                        let m_offset = ctx.mov_u32_imm(batch_m);
                        let m_k_offset = ctx.mul_u32_reg(m_offset, k_dim);
                        let x_idx = ctx.add_u32_reg(m_k_offset, x_elem_idx);
                        let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                        let x_bytes = ctx.mul_u64(x_idx_64, 4);
                        let x_addr = ctx.add_u64(x_ptr, x_bytes);
                        let x_val = ctx.ld_global_f32(x_addr);

                        // Accumulate: thread_partial[m] += x_val * dequant
                        ctx.fma_f32_inplace(thread_partials[batch_m as usize], x_val, dequant);
                    }
                }

                // Add thread partials to accumulators
                for batch_m in 0..m {
                    ctx.add_f32_inplace(accs[batch_m as usize], thread_partials[batch_m as usize]);
                }

                // Next super-block
                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_end");

                // Warp shuffle reduce for each batch element
                for batch_m in 0..m {
                    let acc = accs[batch_m as usize];
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
                }

                // Only thread 0 writes results
                let one_u32 = ctx.mov_u32_imm(1);
                let is_thread0 = ctx.setp_lt_u32(thread_id, one_u32);
                ctx.branch_if_not(is_thread0, "exit");

                // Store y[m][block_id] for each batch element
                // y layout: M × N row-major, so y[m][n] = y_ptr + m * n_dim + n
                let four_bytes = ctx.mov_u32_imm(4);
                for batch_m in 0..m {
                    let m_offset = ctx.mov_u32_imm(batch_m);
                    let m_n_offset = ctx.mul_u32_reg(m_offset, n_dim);
                    let y_idx = ctx.add_u32_reg(m_n_offset, block_id);
                    let y_offset = ctx.mul_wide_u32_reg(y_idx, four_bytes);
                    let y_addr = ctx.add_u64(y_ptr, y_offset);
                    ctx.st_global_f32(y_addr, accs[batch_m as usize]);
                }

                ctx.label("exit");
                ctx.ret();
            })
    }
}
