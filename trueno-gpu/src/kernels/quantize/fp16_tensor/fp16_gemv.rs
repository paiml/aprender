//! PAR-032: FP16 INPUT/OUTPUT Q4K GEMV KERNEL

use crate::kernels::quantize::{Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// FP16 input/output Q4_K GEMV kernel for decode throughput optimization
///
/// This kernel reduces memory bandwidth by 2x for both input and output:
/// - Standard Q4K GEMV: FP32 input -> Q4K matmul -> FP32 output
/// - FP16 Q4K GEMV: FP16 input -> FP32 compute -> FP16 output
///
/// Memory bandwidth savings:
/// - Input: hidden_size x 2 bytes vs hidden_size x 4 bytes (2x)
/// - Output: output_size x 2 bytes vs output_size x 4 bytes (2x)
/// - Total: 4x bandwidth reduction for activations
///
/// Internal computation remains FP32 for numerical stability.
///
/// # Grid Configuration
///
/// - Block: 32 threads (one warp)
/// - Grid: N blocks (one per output element)
#[derive(Debug, Clone)]
pub struct Fp16Q4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl Fp16Q4KGemvKernel {
    /// Create a new FP16 Q4_K GEMV kernel for y = W * x
    ///
    /// # Arguments
    /// * `k` - Input vector length / weight matrix rows (must be multiple of 256)
    /// * `n` - Output vector length / weight matrix columns
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }
}

impl Kernel for Fp16Q4KGemvKernel {
    fn name(&self) -> &str {
        "fp16_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        // PAR-032: FP16 I/O Q4K GEMV - 2x bandwidth savings vs FP32
        PtxKernel::new("fp16_q4k_gemv")
            .param(PtxType::U64, "y_ptr") // Output vector FP16 (N)
            .param(PtxType::U64, "w_ptr") // Q4_K weights (N x K/256 super-blocks)
            .param(PtxType::U64, "x_ptr") // Input vector FP16 (K)
            .param(PtxType::U32, "k_dim") // K dimension
            .param(PtxType::U32, "n_dim") // N dimension
            .build(|ctx| {
                // Block = 32 threads (one warp), grid = N blocks
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                // Bounds check
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                // Load parameters
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Number of super-blocks per row: ceil(K / 256) for GGUF
                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                // Row base address for Q4_K data
                let sb_bytes = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Loop over super-blocks
                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_end");

                // Super-block address
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

                // scales base = sb_addr + 4
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);

                // Load all 12 scale bytes
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

                // Constants for scale extraction
                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let four = ctx.mov_u32_imm(4);
                let six = ctx.mov_u32_imm(6);

                // Extract scale/min for blocks 0-3
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

                // Extract scale/min for blocks 4-7
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

                // Thread partial sum
                let thread_partial = ctx.mov_f32_imm(0.0);

                // Process 8 values per thread (256 values / 32 threads)
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

                    // Value index within super-block
                    let offset_reg = ctx.mov_u32_imm(offset);
                    let val_idx = ctx.add_u32_reg(thread_id, offset_reg);

                    // Load 4-bit quantized value
                    let chunk_idx = ctx.div_u32(val_idx, 64);
                    let val_in_chunk = ctx.rem_u32(val_idx, 64);
                    let byte_in_chunk = ctx.rem_u32(val_in_chunk, 32);
                    let chunk_offset = ctx.mul_u32(chunk_idx, 32);
                    let qs_byte_offset = ctx.add_u32_reg(chunk_offset, byte_in_chunk);
                    let qs_byte_offset_64 = ctx.cvt_u64_u32(qs_byte_offset);
                    let qs_addr = ctx.add_u64(qs_base, qs_byte_offset_64);
                    let packed = ctx.ld_global_u8(qs_addr);
                    let packed_32 = ctx.cvt_u32_u8(packed);

                    // Extract nibble
                    let mask_4bit_q = ctx.mov_u32_imm(0xF);
                    let four_q = ctx.mov_u32_imm(4);
                    let val_in_chunk_div_32 = ctx.div_u32(val_in_chunk, 32);
                    let shift_amount = ctx.mul_u32_reg(val_in_chunk_div_32, four_q);
                    let shifted = ctx.shr_u32(packed_32, shift_amount);
                    let quant = ctx.and_u32(shifted, mask_4bit_q);

                    // Dequantize
                    let quant_f32 = ctx.cvt_f32_u32(quant);
                    let scaled = ctx.mul_f32(ds, quant_f32);
                    let dequant = ctx.sub_f32(scaled, dm);

                    // PAR-032: Load FP16 input (2x bandwidth savings)
                    let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                    let x_idx = ctx.add_u32_reg(sb_k_base, val_idx);
                    let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                    let x_bytes = ctx.mul_u64(x_idx_64, 2); // FP16 = 2 bytes
                    let x_addr = ctx.add_u64(x_ptr, x_bytes);
                    let x_val_f16 = ctx.ld_global_f16(x_addr);
                    let x_val = ctx.cvt_f32_f16(x_val_f16);

                    // Accumulate
                    ctx.fma_f32_inplace(thread_partial, x_val, dequant);
                }

                ctx.add_f32_inplace(acc, thread_partial);

                // Next super-block
                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_end");

                // Warp shuffle reduce
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

                // Only thread 0 writes result
                let one_u32 = ctx.mov_u32_imm(1);
                let is_thread0 = ctx.setp_lt_u32(thread_id, one_u32);
                ctx.branch_if_not(is_thread0, "exit");

                // PAR-032: Store FP16 output (2x bandwidth savings)
                let acc_f16 = ctx.cvt_f16_f32(acc);
                let y_offset = ctx.mul_wide_u32(block_id, 2); // FP16 = 2 bytes
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f16(y_addr, acc_f16);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
