//! Q5_K FUSED GEMV KERNEL (PAR-003)

use super::super::{Kernel, Q5K_SUPER_BLOCK_BYTES, Q5K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Q5_K quantized GEMV kernel for M=1 decode throughput
#[derive(Debug, Clone)]
pub struct Q5KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl Q5KGemvKernel {
    /// Create a new Q5_K GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }
}

impl Kernel for Q5KGemvKernel {
    fn name(&self) -> &str {
        "q5k_gemv_warp_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        build_q5k("q5k_gemv_warp_reduce", 1)
    }
}

/// Batched Q5_K GEMV (#5279): `m` activation rows against one weight matrix,
/// `y[r] = W · x[r]` for `r < m`.
///
/// Each weight value is dequantized once and applied to all `m` rows. Every
/// row keeps its own partial sums, added in exactly the order of
/// [`Q5KGemvKernel`], so row `r` of the output is bitwise equal to that kernel
/// run on `x[r]` alone.
///
/// Layout: `x` is `m × k` and `y` is `m × n`, both row-major f32. Launch it as
/// the single-row kernel: `n` blocks of 32 threads.
#[derive(Debug, Clone)]
pub struct BatchedQ5KGemvKernel {
    /// K dimension (input dimension)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// M dimension: activation rows per launch
    pub m: u32,
}

impl Kernel for BatchedQ5KGemvKernel {
    fn name(&self) -> &str {
        "batched_q5k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        build_q5k("batched_q5k_gemv", self.m)
    }
}

/// The Q5_K GEMV over `rows` activation rows. With `rows == 1` it emits the
/// single-row kernel's PTX unchanged.
fn build_q5k(name: &str, rows: u32) -> PtxKernel {
    PtxKernel::new(name)
        .param(PtxType::U64, "y_ptr")
        .param(PtxType::U64, "w_ptr")
        .param(PtxType::U64, "x_ptr")
        .param(PtxType::U32, "k_dim")
        .param(PtxType::U32, "n_dim")
        .build(move |ctx| {
            let block_id = ctx.special_reg(PtxReg::CtaIdX);
            let thread_id = ctx.special_reg(PtxReg::TidX);

            let n_dim = ctx.load_param_u32("n_dim");
            let oob = ctx.setp_ge_u32(block_id, n_dim);
            ctx.branch_if(oob, "exit");

            let k_dim = ctx.load_param_u32("k_dim");
            let y_ptr = ctx.load_param_u64("y_ptr");
            let w_ptr = ctx.load_param_u64("w_ptr");
            let x_ptr = ctx.load_param_u64("x_ptr");
            // Row r of x starts r × k floats in.
            let x_rows: Vec<_> = (0..rows)
                .map(|r| {
                    if r == 0 {
                        return x_ptr;
                    }
                    let off = ctx.mul_wide_u32(k_dim, 4 * r);
                    ctx.add_u64(x_ptr, off)
                })
                .collect();

            let accs: Vec<_> = (0..rows).map(|_| ctx.mov_f32_imm(0.0)).collect();
            // Ceiling division: (k + 255) / 256 for GGUF super-block count
            let k_rounded = ctx.add_u32(k_dim, Q5K_SUPER_BLOCK_SIZE - 1);
            let num_super_blocks = ctx.div_u32(k_rounded, Q5K_SUPER_BLOCK_SIZE);

            let sb_bytes = ctx.mov_u32_imm(Q5K_SUPER_BLOCK_BYTES);
            let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
            let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
            let row_base = ctx.add_u64(w_ptr, row_offset);

            let sb_idx = ctx.mov_u32_imm(0);

            ctx.label("sb_loop");
            let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
            ctx.branch_if(sb_done, "sb_loop_end");

            let sb_offset = ctx.mul_wide_u32(sb_idx, Q5K_SUPER_BLOCK_BYTES);
            let sb_addr = ctx.add_u64(row_base, sb_offset);

            // Load d and dmin
            let d_f16 = ctx.ld_global_f16(sb_addr);
            let d = ctx.cvt_f32_f16(d_f16);
            let two = ctx.mov_u64_imm(2);
            let dmin_addr = ctx.add_u64(sb_addr, two);
            let dmin_f16 = ctx.ld_global_f16(dmin_addr);
            let dmin = ctx.cvt_f32_f16(dmin_f16);

            // Each thread handles 8 values
            let partials: Vec<_> = (0..rows).map(|_| ctx.mov_f32_imm(0.0)).collect();
            let offsets: [u32; 8] = [0, 32, 64, 96, 128, 160, 192, 224];

            for offset in offsets {
                let offset_reg = ctx.mov_u32_imm(offset);
                let val_idx = ctx.add_u32_reg(thread_id, offset_reg);

                let sub_block = ctx.div_u32(val_idx, 32);

                // Extract scale and min using llama.cpp get_scale_min_k4 logic:
                // For j < 4: scale = scales[j] & 0x3F, min = scales[j+4] & 0x3F
                // For j >= 4: scale = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
                //             min = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);

                // Check if sub_block < 4
                let four_u32 = ctx.mov_u32_imm(4);
                let is_simple = ctx.setp_lt_u32(sub_block, four_u32);

                // Load scales[sub_block] and scales[sub_block + 4]
                let sub_block_64 = ctx.cvt_u64_u32(sub_block);
                let scales_j_addr = ctx.add_u64(scales_base, sub_block_64);
                let scales_j = ctx.ld_global_u8(scales_j_addr);
                let scales_j_32 = ctx.cvt_u32_u8(scales_j);

                let sub_block_plus_4 = ctx.add_u32_reg(sub_block, four_u32);
                let sub_block_plus_4_64 = ctx.cvt_u64_u32(sub_block_plus_4);
                let scales_j4_addr = ctx.add_u64(scales_base, sub_block_plus_4_64);
                let scales_j4 = ctx.ld_global_u8(scales_j4_addr);
                let scales_j4_32 = ctx.cvt_u32_u8(scales_j4);

                // Simple case (j < 4): scale = scales[j] & 0x3F, min = scales[j+4] & 0x3F
                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let scale_simple = ctx.and_u32(scales_j_32, mask_6bit);
                let min_simple = ctx.and_u32(scales_j4_32, mask_6bit);

                // Complex case (j >= 4): need scales[j-4] and scales[j+4]
                // Safe subtraction: for sub_block < 4, use 0 to avoid underflow
                let zero_safe = ctx.mov_u32_imm(0);
                let sub_block_minus_4_raw = ctx.sub_u32_reg(sub_block, four_u32);
                let sub_block_minus_4 = ctx.selp_u32(is_simple, zero_safe, sub_block_minus_4_raw);
                let sub_block_minus_4_64 = ctx.cvt_u64_u32(sub_block_minus_4);
                let scales_jm4_addr = ctx.add_u64(scales_base, sub_block_minus_4_64);
                let scales_jm4 = ctx.ld_global_u8(scales_jm4_addr);
                let scales_jm4_32 = ctx.cvt_u32_u8(scales_jm4);

                // Complex: scale = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let six = ctx.mov_u32_imm(6);
                let s_j4_lo = ctx.and_u32(scales_j4_32, mask_4bit);
                let s_jm4_hi = ctx.shr_u32(scales_jm4_32, six);
                let s_jm4_hi_shifted = ctx.shl_u32(s_jm4_hi, four_u32);
                let scale_complex = ctx.or_u32(s_j4_lo, s_jm4_hi_shifted);

                // Complex: min = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
                let s_j4_hi = ctx.shr_u32(scales_j4_32, four_u32);
                let s_j_hi = ctx.shr_u32(scales_j_32, six);
                let s_j_hi_shifted = ctx.shl_u32(s_j_hi, four_u32);
                let min_complex = ctx.or_u32(s_j4_hi, s_j_hi_shifted);

                // Select between simple and complex based on sub_block < 4
                let scale_6bit = ctx.selp_u32(is_simple, scale_simple, scale_complex);
                let min_6bit = ctx.selp_u32(is_simple, min_simple, min_complex);

                let scale_f32 = ctx.cvt_f32_u32(scale_6bit);
                let min_f32 = ctx.cvt_f32_u32(min_6bit);

                // Load low 4-bit from qs (offset 48: d=2 + dmin=2 + scales=12 + qh=32)
                let chunk_idx = ctx.div_u32(val_idx, 64);
                let val_in_chunk = ctx.rem_u32(val_idx, 64);
                let byte_in_chunk = ctx.rem_u32(val_in_chunk, 32);

                let qs_offset_64 = ctx.mov_u64_imm(48);
                let qs_base = ctx.add_u64(sb_addr, qs_offset_64);
                let chunk_offset = ctx.mul_u32(chunk_idx, 32);
                let qs_byte_offset = ctx.add_u32_reg(chunk_offset, byte_in_chunk);
                let qs_byte_offset_64 = ctx.cvt_u64_u32(qs_byte_offset);
                let qs_addr = ctx.add_u64(qs_base, qs_byte_offset_64);
                let packed = ctx.ld_global_u8(qs_addr);
                let packed_32 = ctx.cvt_u32_u8(packed);

                let four = ctx.mov_u32_imm(4);
                let mask_4bit = ctx.mov_u32_imm(0xF);
                // Branch-free nibble selection: shift = 4 * (val_in_chunk / 32)
                let val_in_chunk_div_32 = ctx.div_u32(val_in_chunk, 32);
                let shift_amount = ctx.mul_u32_reg(val_in_chunk_div_32, four);
                let shifted = ctx.shr_u32(packed_32, shift_amount);
                let ql = ctx.and_u32(shifted, mask_4bit);

                // Load high bit from qh (offset 16: d=2 + dmin=2 + scales=12)
                let qh_offset = ctx.mov_u64_imm(16);
                let qh_base = ctx.add_u64(sb_addr, qh_offset);
                // ggml block_q5_K: the fifth bit of value l in sub-block s is bit s of
                // qh[l], i.e. byte val_idx % 32, bit val_idx / 32 (#3111). It was read
                // as a sequential bitmask, byte val_idx / 8, bit val_idx % 8.
                let qh_byte_idx = ctx.rem_u32(val_idx, 32);
                let qh_bit_idx = ctx.div_u32(val_idx, 32);
                let qh_byte_idx_64 = ctx.cvt_u64_u32(qh_byte_idx);
                let qh_addr = ctx.add_u64(qh_base, qh_byte_idx_64);
                let qh_byte = ctx.ld_global_u8(qh_addr);
                let qh_byte_32 = ctx.cvt_u32_u8(qh_byte);
                let qh_shifted = ctx.shr_u32(qh_byte_32, qh_bit_idx);
                let mask_1bit = ctx.mov_u32_imm(1);
                let qh = ctx.and_u32(qh_shifted, mask_1bit);

                // Combine: quant = ql + 16 * qh (5-bit: 0-31)
                let sixteen_u32 = ctx.mov_u32_imm(16);
                let qh_scaled = ctx.mul_u32_reg(qh, sixteen_u32);
                let quant = ctx.add_u32_reg(ql, qh_scaled);

                // Dequantize
                let quant_f32 = ctx.cvt_f32_u32(quant);
                let d_scale = ctx.mul_f32(d, scale_f32);
                let scaled = ctx.mul_f32(d_scale, quant_f32);
                let dmin_min = ctx.mul_f32(dmin, min_f32);
                let dequant = ctx.sub_f32(scaled, dmin_min);

                // Load activation x[sb_idx * 256 + val_idx]
                // GH-215 FIX: Bounds-check for non-256-aligned K dimensions.
                // The last super-block may address indices beyond k_dim; treat
                // those as 0.0 to avoid out-of-bounds GPU memory reads.
                let sb_k_base = ctx.mul_u32(sb_idx, Q5K_SUPER_BLOCK_SIZE);
                let x_idx = ctx.add_u32_reg(sb_k_base, val_idx);
                let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                let x_bytes = ctx.mul_u64(x_idx_64, 4);
                let x_addr = ctx.add_u64(x_rows[0], x_bytes);
                let in_bounds = ctx.setp_lt_u32(x_idx, k_dim);
                // The same dequantized weight against every row's activation.
                for (r, &partial) in partials.iter().enumerate() {
                    let addr = if r == 0 {
                        x_addr
                    } else {
                        ctx.add_u64(x_rows[r], x_bytes)
                    };
                    let x_val = ctx.ld_global_f32_predicated(addr, in_bounds, 0.0);
                    ctx.fma_f32_inplace(partial, x_val, dequant);
                }
            }

            for (&acc, &partial) in accs.iter().zip(&partials) {
                ctx.add_f32_inplace(acc, partial);
            }
            ctx.add_u32_inplace(sb_idx, 1);
            ctx.branch("sb_loop");

            ctx.label("sb_loop_end");

            // Warp reduce, each row on its own
            for &acc in &accs {
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

            let one_u32 = ctx.mov_u32_imm(1);
            let is_thread0 = ctx.setp_lt_u32(thread_id, one_u32);
            ctx.branch_if_not(is_thread0, "exit");

            // Row r of y starts r × n floats in.
            for (r, &acc) in (0u32..).zip(&accs) {
                let out_idx = if r == 0 {
                    block_id
                } else {
                    let row_start = ctx.mul_u32(n_dim, r);
                    ctx.add_u32_reg(row_start, block_id)
                };
                let y_offset = ctx.mul_wide_u32(out_idx, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, acc);
            }

            ctx.label("exit");
            ctx.ret();
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batched_one_row_is_the_single_row_kernel() {
        let single = Q5KGemvKernel::new(2560, 8192).emit_ptx();
        let batched = BatchedQ5KGemvKernel {
            k: 2560,
            n: 8192,
            m: 1,
        }
        .emit_ptx();
        assert_eq!(
            batched.replace("batched_q5k_gemv", "q5k_gemv_warp_reduce"),
            single
        );
    }

    #[test]
    fn test_batched_loads_every_row() {
        let one = BatchedQ5KGemvKernel {
            k: 2560,
            n: 8192,
            m: 1,
        }
        .emit_ptx();
        let four = BatchedQ5KGemvKernel {
            k: 2560,
            n: 8192,
            m: 4,
        }
        .emit_ptx();
        let loads = |p: &str| p.matches("ld.global.f32").count();
        assert_eq!(loads(&four), 4 * loads(&one));
        let stores = |p: &str| p.matches("st.global.f32").count();
        assert_eq!(stores(&four), 4);
    }
}
