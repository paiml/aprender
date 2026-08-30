//! Q6_K Dequantization Kernel (PMAT-026)
//!
//! Dequantizes Q6_K weight data from GPU memory to dense F32,
//! enabling cuBLAS GEMM for prefill (M > 1) operations.
//!
//! # Q6_K Layout (210 bytes per 256 values)
//!
//! - ql[128]: bytes 0-127, low 4-bits packed 2 per byte
//! - qh[64]: bytes 128-191, high 2-bits packed 4 per byte
//! - scales[16]: bytes 192-207, signed i8 per 16-element sub-block
//! - d: bytes 208-209, f16 scale factor
//!
//! # Dequant formula
//!
//! For element at position `idx` (0..255):
//!   q6 = ql_nibble | (qh_2bits << 4)
//!   value = d * scales[idx / 16] * (q6 - 32)
//!
//! # Launch Configuration
//!
//! - Grid: (N, num_super_blocks_per_row)
//! - Block: 32 threads (one warp), each thread writes 8 values
//! - Output: row-major F32 [N × K]

use crate::kernels::quantize::{Kernel, Q6K_SUPER_BLOCK_BYTES, Q6K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Q6_K dequantization kernel for cuBLAS GEMM prefill (PMAT-026)
#[derive(Debug, Clone)]
pub struct Q6KDequantKernel {
    /// K dimension (must be multiple of 256)
    pub k: u32,
    /// N dimension (number of rows)
    pub n: u32,
}

impl Q6KDequantKernel {
    /// Create a new Q6K dequantization kernel for the given dimensions.
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Number of Q6K super-blocks per row (ceiling division).
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        (self.k + Q6K_SUPER_BLOCK_SIZE - 1) / Q6K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for Q6KDequantKernel {
    fn name(&self) -> &str {
        "q6k_dequant_to_f32"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("q6k_dequant_to_f32")
            .param(PtxType::U64, "out_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .build(|ctx| {
                // blockIdx.x = row index (0..N)
                // blockIdx.y = super-block index within row
                // threadIdx.x = thread in warp (0..31)
                let row_id = ctx.special_reg(PtxReg::CtaIdX);
                let sb_idx = ctx.special_reg(PtxReg::CtaIdY);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");

                let oob = ctx.setp_ge_u32(row_id, n_dim);
                ctx.branch_if(oob, "exit");

                let out_ptr = ctx.load_param_u64("out_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");

                let k_rounded = ctx.add_u32(k_dim, Q6K_SUPER_BLOCK_SIZE - 1);
                let num_sb = ctx.div_u32(k_rounded, Q6K_SUPER_BLOCK_SIZE);

                let sb_oob = ctx.setp_ge_u32(sb_idx, num_sb);
                ctx.branch_if(sb_oob, "exit");

                // Super-block address: w_ptr + row_id * num_sb * 210 + sb_idx * 210
                let sb_bytes = ctx.mov_u32_imm(Q6K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_sb, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(row_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);
                let sb_offset = ctx.mul_wide_u32(sb_idx, Q6K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d (f16 at offset 208)
                let d_offset = ctx.mov_u64_imm(208);
                let d_addr = ctx.add_u64(sb_addr, d_offset);
                let d_f16 = ctx.ld_global_f16(d_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Load all 16 scales (i8 at offset 192-207)
                let scales_offset = ctx.mov_u64_imm(192);
                let scales_base = ctx.add_u64(sb_addr, scales_offset);

                // Load scales as i8 → f32 (sign-extend u8 → s32 → f32)
                let mut scale_f32s = Vec::with_capacity(16);
                for i in 0..16u64 {
                    let s_off = ctx.mov_u64_imm(i);
                    let s_addr = ctx.add_u64(scales_base, s_off);
                    let s_u8 = ctx.ld_global_u8(s_addr);
                    // cvt_s32_s8 handles: load u8, sign-extend (if >= 128, subtract 256)
                    let s_i32 = ctx.cvt_s32_s8(s_u8);
                    let s_f32 = ctx.cvt_f32_s32(s_i32);
                    // Precompute d * scale
                    let ds = ctx.mul_f32(d, s_f32);
                    scale_f32s.push(ds);
                }

                // ql base at offset 0, qh base at offset 128
                let ql_base = sb_addr;
                let qh_offset = ctx.mov_u64_imm(128);
                let qh_base = ctx.add_u64(sb_addr, qh_offset);

                // Output base: out_ptr + (row_id * k_dim + sb_idx * 256) * 4
                let sb_k_base = ctx.mul_u32(sb_idx, Q6K_SUPER_BLOCK_SIZE);
                let row_k = ctx.mul_u32_reg(row_id, k_dim);
                let out_k_base = ctx.add_u32_reg(row_k, sb_k_base);
                let out_k_base_64 = ctx.cvt_u64_u32(out_k_base);
                let out_k_bytes = ctx.mul_u64(out_k_base_64, 4);
                let out_base = ctx.add_u64(out_ptr, out_k_bytes);

                let mask_0f = ctx.mov_u32_imm(0x0F);
                let four_u32 = ctx.mov_u32_imm(4);
                let const_32_f = ctx.mov_f32_imm(32.0);
                let sixteen = ctx.mov_u32_imm(16);

                // Each thread writes 8 values: thread_id, thread_id+32, ..., thread_id+224
                for step in 0..8u32 {
                    let offset = step * 32;
                    let offset_reg = ctx.mov_u32_imm(offset);
                    let val_idx = ctx.add_u32_reg(thread_id, offset_reg);

                    // Bounds check
                    let global_k = ctx.add_u32_reg(sb_k_base, val_idx);
                    let out_of_bounds = ctx.setp_ge_u32(global_k, k_dim);
                    let skip_label = format!("skip_store_{step}");
                    ctx.branch_if(out_of_bounds, &skip_label);

                    // Q6K dequant: extract 6-bit value from ql + qh
                    //
                    // The Q6K layout packs 256 values into two 128-value halves.
                    // For half h (0 or 1), within each half there are 4 groups of 32:
                    //   group g, lane l (0..31):
                    //     ql_byte = ql[64*h + 32*(g & 1) + l]
                    //     ql_nibble = g < 2 ? (ql_byte & 0xF) : (ql_byte >> 4)
                    //     qh_byte = qh[32*h + l]
                    //     qh_bits = (qh_byte >> (2*g)) & 0x3
                    //     q6 = ql_nibble | (qh_bits << 4)
                    //     sub_block = 8*h + 2*g + l/16
                    //     value = d * scales[sub_block] * (q6 - 32)
                    //
                    // aprender#2770: the two bits of `g` mean DIFFERENT things and this
                    // kernel had them the wrong way round -- it took the byte offset from
                    // `g >> 1` and the nibble from `g & 1`. ggml's dequantize_row_q6_K
                    // emits, per lane l:
                    //     q1 = ql[l     ] & 0xF | qh[l] >> 0   -> y[l     ]   (g = 0)
                    //     q2 = ql[l + 32] & 0xF | qh[l] >> 2   -> y[l + 32]  (g = 1)
                    //     q3 = ql[l     ] >>  4 | qh[l] >> 4   -> y[l + 64]  (g = 2)
                    //     q4 = ql[l + 32] >>  4 | qh[l] >> 6   -> y[l + 96]  (g = 3)
                    // so the +32 byte step belongs to `g & 1` and the high nibble to
                    // `g >= 2`. Swapping them is correct for g = 0 and g = 3, where both
                    // bits agree, and exchanges the low four bits of groups 1 and 2 --
                    // while their qh bits and sub-block scale stay correct. Half of every
                    // super-block therefore came out near-but-not-right rather than
                    // broken, and the model kept emitting fluent text on a different
                    // continuation. Q6_K is attn_v, ffn_down and output in Q4_K_M, and
                    // this kernel is the ONLY Q6_K reader on the cuBLAS GEMM path (FP8,
                    // FP16 and FP32 all build from its output), so the error was
                    // invariant to GEMM precision -- which is what ruled precision out.
                    // The Q6_K GEMV kernel (q6k/gemv.rs) always had this right, which is
                    // why m < 4 was unaffected.
                    // Guarded by tests/falsify_q6k_dequant_parity_2770.rs.

                    let half = step / 4; // 0 for steps 0-3, 1 for steps 4-7
                    let group = step % 4; // 0-3 within each half

                    // ql_byte_idx = 64*half + 32*(group & 1) + thread_id
                    let ql_byte_offset = 64 * half + 32 * (group % 2);
                    let ql_off_reg = ctx.mov_u32_imm(ql_byte_offset);
                    let ql_idx = ctx.add_u32_reg(ql_off_reg, thread_id);
                    let ql_idx_64 = ctx.cvt_u64_u32(ql_idx);
                    let ql_addr = ctx.add_u64(ql_base, ql_idx_64);
                    let ql_byte = ctx.ld_global_u8(ql_addr);
                    let ql_u32 = ctx.cvt_u32_u8(ql_byte);

                    // Extract nibble: low for groups 0-1, high for groups 2-3
                    let ql_nibble = if group < 2 {
                        ctx.and_u32(ql_u32, mask_0f)
                    } else {
                        ctx.shr_u32(ql_u32, four_u32)
                    };

                    // qh_byte_idx = 32*half + thread_id
                    let qh_byte_offset = 32 * half;
                    let qh_off_reg = ctx.mov_u32_imm(qh_byte_offset);
                    let qh_idx = ctx.add_u32_reg(qh_off_reg, thread_id);
                    let qh_idx_64 = ctx.cvt_u64_u32(qh_idx);
                    let qh_addr = ctx.add_u64(qh_base, qh_idx_64);
                    let qh_byte = ctx.ld_global_u8(qh_addr);
                    let qh_u32 = ctx.cvt_u32_u8(qh_byte);

                    // Extract 2 bits: (qh >> (2*group)) & 0x3
                    let qh_shift = ctx.mov_u32_imm(2 * group);
                    let qh_shifted = ctx.shr_u32(qh_u32, qh_shift);
                    let mask_03 = ctx.mov_u32_imm(0x03);
                    let qh_2bits = ctx.and_u32(qh_shifted, mask_03);

                    // q6 = ql_nibble | (qh_2bits << 4)
                    let qh_hi = ctx.shl_u32(qh_2bits, four_u32);
                    let q6 = ctx.or_u32(ql_nibble, qh_hi);

                    // dequant: d * scale * (q6 - 32)
                    let q6_f32 = ctx.cvt_f32_u32(q6);
                    let q6_centered = ctx.sub_f32(q6_f32, const_32_f);

                    // sub_block index = 8*half + 2*group + thread_id/16
                    // Since thread_id is 0..31, thread_id/16 is 0 or 1
                    // We need to handle this dynamically
                    // sub_block = 8*half + 2*group + thread_id/16
                    // thread_id/16 is 0 for lanes 0-15, 1 for lanes 16-31

                    // Select scale: we have 16 precomputed d*scale values
                    // sub_block is 0..15, pick the right one with selp chain
                    // Use a selp cascade for the 2 possible values
                    let sb_base = (8 * half + 2 * group) as usize;
                    let ds_lo = scale_f32s[sb_base]; // for lanes 0-15
                    let ds_hi = scale_f32s[sb_base + 1]; // for lanes 16-31
                    let is_hi = ctx.setp_ge_u32(thread_id, sixteen);
                    let ds = ctx.selp_f32(is_hi, ds_hi, ds_lo);

                    let dequant = ctx.mul_f32(ds, q6_centered);

                    // Write output
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q6k_dequant_kernel_emits_ptx() {
        let kernel = Q6KDequantKernel::new(1536, 256);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("q6k_dequant_to_f32"));
        assert!(ptx.contains(".entry"));
    }

    #[test]
    fn test_q6k_dequant_kernel_name() {
        let kernel = Q6KDequantKernel::new(256, 16);
        assert_eq!(kernel.name(), "q6k_dequant_to_f32");
    }

    #[test]
    fn test_num_super_blocks_per_row() {
        let kernel = Q6KDequantKernel::new(1536, 256);
        assert_eq!(kernel.num_super_blocks_per_row(), 6);

        let kernel = Q6KDequantKernel::new(4096, 1536);
        assert_eq!(kernel.num_super_blocks_per_row(), 16);
    }
}
