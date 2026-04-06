//! Q4_K Dequantization Kernel (PMAT-024)
//!
//! Dequantizes Q4_K weight data from GPU memory to dense F32,
//! enabling cuBLAS GEMM for prefill (M > 1) operations.
//!
//! # Motivation
//!
//! During prefill, llama.cpp uses cuBLAS GEMM (reading weights once).
//! realizr's batched GEMV tiles M at 8, re-reading weights per tile.
//! This kernel + cuBLAS GEMM closes the 18.6x prefill gap.
//!
//! # Launch Configuration
//!
//! - Grid: (N, num_super_blocks_per_row) — one block per super-block
//! - Block: 32 threads (one warp) — each thread writes 8 values
//! - Output: row-major F32 [N × K]

use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Q4_K dequantization kernel for cuBLAS GEMM prefill (PMAT-024)
///
/// Reads Q4_K super-blocks from GPU memory, writes dense F32 output.
/// One warp (32 threads) per super-block, each thread writes 8 values.
///
/// # Arguments (kernel params)
///
/// * `out_ptr` - Output F32 buffer [N × K]
/// * `w_ptr` - Q4K weight data [N × ceil(K/256) × 144 bytes]
/// * `k_dim` - K dimension (columns)
/// * `n_dim` - N dimension (rows)
#[derive(Debug, Clone)]
pub struct Q4KDequantKernel {
    /// K dimension (must be multiple of 256)
    pub k: u32,
    /// N dimension (number of rows)
    pub n: u32,
}

impl Q4KDequantKernel {
    /// Create a new Q4K dequant kernel
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

impl Kernel for Q4KDequantKernel {
    fn name(&self) -> &str {
        "q4k_dequant_to_f32"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("q4k_dequant_to_f32")
            .param(PtxType::U64, "out_ptr") // F32 output [N × K]
            .param(PtxType::U64, "w_ptr") // Q4K weights
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .build(|ctx| {
                // blockIdx.x = row index (0..N)
                // blockIdx.y = super-block index within row (0..num_sb)
                // threadIdx.x = thread in warp (0..31)
                let row_id = ctx.special_reg(PtxReg::CtaIdX);
                let sb_idx = ctx.special_reg(PtxReg::CtaIdY);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");

                // Bounds check
                let oob = ctx.setp_ge_u32(row_id, n_dim);
                ctx.branch_if(oob, "exit");

                let out_ptr = ctx.load_param_u64("out_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");

                // Number of super-blocks per row
                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_sb = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                // Bounds check for sb_idx
                let sb_oob = ctx.setp_ge_u32(sb_idx, num_sb);
                ctx.branch_if(sb_oob, "exit");

                // Calculate super-block address
                // row_addr = w_ptr + row_id * num_sb * 144
                let sb_bytes = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_sb, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(row_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);
                let sb_offset = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d (f16 at offset 0)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Load dmin (f16 at offset 2)
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // Load all 12 scale bytes
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);

                let s0 = ctx.ld_global_u8(scales_base);
                let s0_32 = ctx.cvt_u32_u8(s0);
                let one_64 = ctx.mov_u64_imm(1);
                let s1_addr = ctx.add_u64(scales_base, one_64);
                let s1 = ctx.ld_global_u8(s1_addr);
                let s1_32 = ctx.cvt_u32_u8(s1);
                let two_64 = ctx.mov_u64_imm(2);
                let s2_addr = ctx.add_u64(scales_base, two_64);
                let s2 = ctx.ld_global_u8(s2_addr);
                let s2_32 = ctx.cvt_u32_u8(s2);
                let three_64 = ctx.mov_u64_imm(3);
                let s3_addr = ctx.add_u64(scales_base, three_64);
                let s3 = ctx.ld_global_u8(s3_addr);
                let s3_32 = ctx.cvt_u32_u8(s3);
                let four_64b = ctx.mov_u64_imm(4);
                let s4_addr = ctx.add_u64(scales_base, four_64b);
                let s4 = ctx.ld_global_u8(s4_addr);
                let s4_32 = ctx.cvt_u32_u8(s4);
                let five_64 = ctx.mov_u64_imm(5);
                let s5_addr = ctx.add_u64(scales_base, five_64);
                let s5 = ctx.ld_global_u8(s5_addr);
                let s5_32 = ctx.cvt_u32_u8(s5);
                let six_64 = ctx.mov_u64_imm(6);
                let s6_addr = ctx.add_u64(scales_base, six_64);
                let s6 = ctx.ld_global_u8(s6_addr);
                let s6_32 = ctx.cvt_u32_u8(s6);
                let seven_64 = ctx.mov_u64_imm(7);
                let s7_addr = ctx.add_u64(scales_base, seven_64);
                let s7 = ctx.ld_global_u8(s7_addr);
                let s7_32 = ctx.cvt_u32_u8(s7);
                let eight_64 = ctx.mov_u64_imm(8);
                let s8_addr = ctx.add_u64(scales_base, eight_64);
                let s8 = ctx.ld_global_u8(s8_addr);
                let s8_32 = ctx.cvt_u32_u8(s8);
                let nine_64 = ctx.mov_u64_imm(9);
                let s9_addr = ctx.add_u64(scales_base, nine_64);
                let s9 = ctx.ld_global_u8(s9_addr);
                let s9_32 = ctx.cvt_u32_u8(s9);
                let ten_64 = ctx.mov_u64_imm(10);
                let s10_addr = ctx.add_u64(scales_base, ten_64);
                let s10 = ctx.ld_global_u8(s10_addr);
                let s10_32 = ctx.cvt_u32_u8(s10);
                let eleven_64 = ctx.mov_u64_imm(11);
                let s11_addr = ctx.add_u64(scales_base, eleven_64);
                let s11 = ctx.ld_global_u8(s11_addr);
                let s11_32 = ctx.cvt_u32_u8(s11);

                // Constants for scale unpacking
                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let four = ctx.mov_u32_imm(4);
                let six = ctx.mov_u32_imm(6);

                // Unpack scales and mins (identical to Q4KGemvKernel)
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

                // qs base = sb_addr + 16
                let sixteen_64 = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen_64);

                // Output base for this super-block:
                // out_ptr + (row_id * k_dim + sb_idx * 256) * 4
                let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                let row_k = ctx.mul_u32_reg(row_id, k_dim);
                let out_k_base = ctx.add_u32_reg(row_k, sb_k_base);
                let out_k_base_64 = ctx.cvt_u64_u32(out_k_base);
                let out_k_bytes = ctx.mul_u64(out_k_base_64, 4);
                let out_base = ctx.add_u64(out_ptr, out_k_bytes);

                // Each thread writes 8 values (256/32 = 8 per thread)
                let offsets_and_blocks: [(u32, u32); 8] =
                    [(0, 0), (32, 1), (64, 2), (96, 3), (128, 4), (160, 5), (192, 6), (224, 7)];

                for (i, (offset, block_idx)) in offsets_and_blocks.iter().enumerate() {
                    let (ds, dm) = match *block_idx {
                        0 => (ds0, dm0),
                        1 => (ds1, dm1),
                        2 => (ds2, dm2),
                        3 => (ds3, dm3),
                        4 => (ds4, dm4),
                        5 => (ds5, dm5),
                        6 => (ds6, dm6),
                        _ => (ds7, dm7),
                    };

                    let offset_reg = ctx.mov_u32_imm(*offset);
                    let val_idx = ctx.add_u32_reg(thread_id, offset_reg);

                    // Extract nibble from qs
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

                    // Dequantize: val = ds * quant - dm
                    let quant_f32 = ctx.cvt_f32_u32(quant);
                    let scaled = ctx.mul_f32(ds, quant_f32);
                    let dequant = ctx.sub_f32(scaled, dm);

                    // Bounds-check: don't write beyond K
                    let global_k = ctx.add_u32_reg(sb_k_base, val_idx);
                    let out_of_bounds = ctx.setp_ge_u32(global_k, k_dim);
                    let skip_label = format!("skip_store_{i}");
                    ctx.branch_if(out_of_bounds, &skip_label);

                    // Write to output: out_base + val_idx * 4
                    let val_idx_64 = ctx.cvt_u64_u32(val_idx);
                    let val_bytes = ctx.mul_u64(val_idx_64, 4);
                    let out_addr = ctx.add_u64(out_base, val_bytes);
                    ctx.st_global_f32(out_addr, dequant);

                    ctx.label(&skip_label);
                }

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// PMAT-065: Q4K → FP16 direct dequantization kernel
///
/// Same algorithm as Q4KDequantKernel but outputs FP16 instead of F32.
/// Half the output bandwidth (2 B/elem vs 4 B/elem), enabling L2-cached
/// HGEMM prefill: dequant Q4K → FP16 temp (fits in L2) → cuBLAS reads from L2.
///
/// # Arguments (kernel params)
///
/// * `out_ptr` - Output FP16 buffer [N × K] (u16 elements)
/// * `w_ptr` - Q4K weight data [N × ceil(K/256) × 144 bytes]
/// * `k_dim` - K dimension (columns)
/// * `n_dim` - N dimension (rows)
#[derive(Debug, Clone)]
pub struct Q4KDequantFp16Kernel {
    /// K dimension (must be multiple of 256)
    pub k: u32,
    /// N dimension (number of rows)
    pub n: u32,
}

impl Q4KDequantFp16Kernel {
    /// Creates a new Q4K dequantization-to-FP16 kernel with the given dimensions.
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Returns the number of Q4K super-blocks per row along the K dimension.
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        (self.k + Q4K_SUPER_BLOCK_SIZE - 1) / Q4K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for Q4KDequantFp16Kernel {
    fn name(&self) -> &str {
        "q4k_dequant_to_f16"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("q4k_dequant_to_f16")
            .param(PtxType::U64, "out_ptr") // FP16 output [N × K]
            .param(PtxType::U64, "w_ptr") // Q4K weights
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .build(|ctx| {
                let row_id = ctx.special_reg(PtxReg::CtaIdX);
                let sb_idx = ctx.special_reg(PtxReg::CtaIdY);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");

                let oob = ctx.setp_ge_u32(row_id, n_dim);
                ctx.branch_if(oob, "exit");

                let out_ptr = ctx.load_param_u64("out_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");

                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_sb = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                let sb_oob = ctx.setp_ge_u32(sb_idx, num_sb);
                ctx.branch_if(sb_oob, "exit");

                // Super-block address (identical to F32 version)
                let sb_bytes = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_sb, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(row_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);
                let sb_offset = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d and dmin (identical)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // Load all 12 scale bytes (identical to F32 version)
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);

                let s0 = ctx.ld_global_u8(scales_base);
                let s0_32 = ctx.cvt_u32_u8(s0);
                let one_64 = ctx.mov_u64_imm(1);
                let s1_addr = ctx.add_u64(scales_base, one_64);
                let s1 = ctx.ld_global_u8(s1_addr);
                let s1_32 = ctx.cvt_u32_u8(s1);
                let two_64 = ctx.mov_u64_imm(2);
                let s2_addr = ctx.add_u64(scales_base, two_64);
                let s2 = ctx.ld_global_u8(s2_addr);
                let s2_32 = ctx.cvt_u32_u8(s2);
                let three_64 = ctx.mov_u64_imm(3);
                let s3_addr = ctx.add_u64(scales_base, three_64);
                let s3 = ctx.ld_global_u8(s3_addr);
                let s3_32 = ctx.cvt_u32_u8(s3);
                let four_64b = ctx.mov_u64_imm(4);
                let s4_addr = ctx.add_u64(scales_base, four_64b);
                let s4 = ctx.ld_global_u8(s4_addr);
                let s4_32 = ctx.cvt_u32_u8(s4);
                let five_64 = ctx.mov_u64_imm(5);
                let s5_addr = ctx.add_u64(scales_base, five_64);
                let s5 = ctx.ld_global_u8(s5_addr);
                let s5_32 = ctx.cvt_u32_u8(s5);
                let six_64 = ctx.mov_u64_imm(6);
                let s6_addr = ctx.add_u64(scales_base, six_64);
                let s6 = ctx.ld_global_u8(s6_addr);
                let s6_32 = ctx.cvt_u32_u8(s6);
                let seven_64 = ctx.mov_u64_imm(7);
                let s7_addr = ctx.add_u64(scales_base, seven_64);
                let s7 = ctx.ld_global_u8(s7_addr);
                let s7_32 = ctx.cvt_u32_u8(s7);
                let eight_64 = ctx.mov_u64_imm(8);
                let s8_addr = ctx.add_u64(scales_base, eight_64);
                let s8 = ctx.ld_global_u8(s8_addr);
                let s8_32 = ctx.cvt_u32_u8(s8);
                let nine_64 = ctx.mov_u64_imm(9);
                let s9_addr = ctx.add_u64(scales_base, nine_64);
                let s9 = ctx.ld_global_u8(s9_addr);
                let s9_32 = ctx.cvt_u32_u8(s9);
                let ten_64 = ctx.mov_u64_imm(10);
                let s10_addr = ctx.add_u64(scales_base, ten_64);
                let s10 = ctx.ld_global_u8(s10_addr);
                let s10_32 = ctx.cvt_u32_u8(s10);
                let eleven_64 = ctx.mov_u64_imm(11);
                let s11_addr = ctx.add_u64(scales_base, eleven_64);
                let s11 = ctx.ld_global_u8(s11_addr);
                let s11_32 = ctx.cvt_u32_u8(s11);

                // Unpack scales and mins (identical)
                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let four = ctx.mov_u32_imm(4);
                let six = ctx.mov_u32_imm(6);

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

                // Precompute d*scale and dmin*min (identical)
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

                let sixteen_64 = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen_64);

                // Output base: out_ptr + (row_id * k_dim + sb_idx * 256) * 2
                // (×2 for FP16 instead of ×4 for F32)
                let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                let row_k = ctx.mul_u32_reg(row_id, k_dim);
                let out_k_base = ctx.add_u32_reg(row_k, sb_k_base);
                let out_k_base_64 = ctx.cvt_u64_u32(out_k_base);
                let out_k_bytes = ctx.mul_u64(out_k_base_64, 2); // ×2 for FP16
                let out_base = ctx.add_u64(out_ptr, out_k_bytes);

                // Each thread writes 8 values (256/32 = 8 per thread)
                let offsets_and_blocks: [(u32, u32); 8] =
                    [(0, 0), (32, 1), (64, 2), (96, 3), (128, 4), (160, 5), (192, 6), (224, 7)];

                for (i, (offset, block_idx)) in offsets_and_blocks.iter().enumerate() {
                    let (ds, dm) = match *block_idx {
                        0 => (ds0, dm0),
                        1 => (ds1, dm1),
                        2 => (ds2, dm2),
                        3 => (ds3, dm3),
                        4 => (ds4, dm4),
                        5 => (ds5, dm5),
                        6 => (ds6, dm6),
                        _ => (ds7, dm7),
                    };

                    let offset_reg = ctx.mov_u32_imm(*offset);
                    let val_idx = ctx.add_u32_reg(thread_id, offset_reg);

                    // Extract nibble from qs (identical)
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

                    // Dequantize: val = ds * quant - dm
                    let quant_f32 = ctx.cvt_f32_u32(quant);
                    let scaled = ctx.mul_f32(ds, quant_f32);
                    let dequant = ctx.sub_f32(scaled, dm);

                    // Bounds-check: don't write beyond K
                    let global_k = ctx.add_u32_reg(sb_k_base, val_idx);
                    let out_of_bounds = ctx.setp_ge_u32(global_k, k_dim);
                    let skip_label = format!("skip_store_{i}");
                    ctx.branch_if(out_of_bounds, &skip_label);

                    // Convert F32 → FP16 and write: out_base + val_idx * 2
                    let dequant_f16 = ctx.cvt_f16_f32(dequant);
                    let val_idx_64 = ctx.cvt_u64_u32(val_idx);
                    let val_bytes = ctx.mul_u64(val_idx_64, 2); // ×2 for FP16
                    let out_addr = ctx.add_u64(out_base, val_bytes);
                    ctx.st_global_f16(out_addr, dequant_f16);

                    ctx.label(&skip_label);
                }

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4k_dequant_kernel_emits_ptx() {
        let kernel = Q4KDequantKernel::new(1536, 1536);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("q4k_dequant_to_f32"));
        assert!(ptx.contains(".entry"));
    }

    #[test]
    fn test_q4k_dequant_kernel_name() {
        let kernel = Q4KDequantKernel::new(256, 16);
        assert_eq!(kernel.name(), "q4k_dequant_to_f32");
    }

    #[test]
    fn test_num_super_blocks_per_row() {
        let kernel = Q4KDequantKernel::new(1536, 1536);
        assert_eq!(kernel.num_super_blocks_per_row(), 6);

        let kernel = Q4KDequantKernel::new(8960, 1536);
        assert_eq!(kernel.num_super_blocks_per_row(), 35);
    }

    #[test]
    fn test_q4k_dequant_fp16_kernel_emits_ptx() {
        let kernel = Q4KDequantFp16Kernel::new(1536, 1536);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("q4k_dequant_to_f16"));
        assert!(ptx.contains(".entry"));
        // Should have cvt.rn.f16.f32 for FP16 output
        assert!(ptx.contains("cvt.rn.f16.f32"));
        // Should NOT have st.global.f32 (FP16 stores use b16)
        assert!(ptx.contains("st.global.b16"));
    }

    #[test]
    fn test_q4k_dequant_fp16_kernel_name() {
        let kernel = Q4KDequantFp16Kernel::new(256, 16);
        assert_eq!(kernel.name(), "q4k_dequant_to_f16");
    }
}
