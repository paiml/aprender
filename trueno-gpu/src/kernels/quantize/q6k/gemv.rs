//! Q6_K basic GEMV kernel with warp shuffle reduction (PAR-003).

use crate::kernels::quantize::{Kernel, Q6K_SUPER_BLOCK_BYTES, Q6K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

// =============================================================================
// Q6_K FUSED GEMV KERNEL (PAR-003)
// =============================================================================

/// Q6_K quantized GEMV kernel for M=1 decode throughput
#[derive(Debug, Clone)]
pub struct Q6KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl Q6KGemvKernel {
    /// Create a new Q6_K GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }
}

impl Kernel for Q6KGemvKernel {
    fn name(&self) -> &str {
        "q6k_gemv_warp_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Q6_K super-block layout (210 bytes for 256 values):
        // - ql[128]: bytes 0-127, low 4-bits packed 2 per byte
        // - qh[64]: bytes 128-191, high 2-bits packed 4 per byte
        // - scales[16]: bytes 192-207, signed i8 per 16-element sub-block
        // - d: bytes 208-209, f16 scale factor
        //
        // Q6_K dequant formula (from llama.cpp):
        // For 256 values, processed in two 128-value halves (n=0, n=128):
        //   For each half, 4 groups of 32 values at positions l, l+32, l+64, l+96
        //   q1: ql[l] low nibble + qh[l] bits 0-1, shifted left 4
        //   q2: ql[l+32] low nibble + qh[l] bits 2-3, shifted left 4
        //   q3: ql[l] high nibble + qh[l] bits 4-5, shifted left 4
        //   q4: ql[l+32] high nibble + qh[l] bits 6-7, shifted left 4
        //   quant = q_combined - 32 (signed range -32 to +31)
        //   scale = scales[8*half + l/16 + 2*group] (signed i8)
        //   dequant = d * scale * quant
        PtxKernel::new("q6k_gemv_warp_reduce")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .build(|ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);
                // Ceiling division: (k + 255) / 256 for GGUF super-block count
                let k_rounded = ctx.add_u32(k_dim, Q6K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q6K_SUPER_BLOCK_SIZE);

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

                // Each thread handles 8 values at offsets 0, 32, 64, 96, 128, 160, 192, 224
                let thread_partial = ctx.mov_f32_imm(0.0);

                // Process each of 8 values per thread
                // For val_idx = thread_id + offset (offset in [0, 32, 64, 96, 128, 160, 192, 224]):
                //   n_idx = val_idx / 128 (0 or 1, which 128-block half)
                //   pos = val_idx % 128 (position within 128-block)
                //   group = pos / 32 (0, 1, 2, or 3)
                //   l = pos % 32 (0-31)
                //   is = l / 16 (0 or 1)
                //
                //   scale_idx = 8 * n_idx + is + 2 * group
                //   ql_byte_offset = 64 * n_idx + l + (32 if group in [1, 3] else 0)
                //   ql_use_high_nibble = (group >= 2)
                //   qh_byte_offset = 32 * n_idx + l
                //   qh_bit_shift = 2 * group

                for offset in [0u32, 32, 64, 96, 128, 160, 192, 224] {
                    let offset_reg = ctx.mov_u32_imm(offset);
                    let val_idx = ctx.add_u32_reg(thread_id, offset_reg);

                    // n_idx = val_idx / 128
                    let n_idx = ctx.div_u32(val_idx, 128);
                    // pos = val_idx % 128
                    let pos = ctx.rem_u32(val_idx, 128);
                    // group = pos / 32
                    let group = ctx.div_u32(pos, 32);
                    // l = pos % 32
                    let l = ctx.rem_u32(pos, 32);
                    // is = l / 16
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
                    // Convert u8 to signed i8 then to f32
                    // i8 is stored as u8, reinterpret: if >= 128, subtract 256
                    // Using: scale_f32 = (scale_u8 as f32) - 256.0 * (scale_u8 >> 7)
                    let scale_u32 = ctx.cvt_u32_u8(scale_u8);
                    let seven = ctx.mov_u32_imm(7);
                    let sign_bit = ctx.shr_u32(scale_u32, seven); // 0 or 1
                    let scale_u32_f32 = ctx.cvt_f32_u32(scale_u32);
                    let sign_bit_f32 = ctx.cvt_f32_u32(sign_bit);
                    let twofiftysix_f32 = ctx.mov_f32_imm(256.0);
                    let correction_f32 = ctx.mul_f32(sign_bit_f32, twofiftysix_f32);
                    let scale_f32 = ctx.sub_f32(scale_u32_f32, correction_f32);

                    // ql_byte_offset = 64 * n_idx + l + (32 * group_is_odd)
                    // where group_is_odd = group & 1
                    let sixty_four = ctx.mov_u32_imm(64);
                    let thirty_two = ctx.mov_u32_imm(32);
                    let one = ctx.mov_u32_imm(1);
                    let n_idx_x64 = ctx.mul_u32_reg(n_idx, sixty_four);
                    let ql_base = ctx.add_u32_reg(n_idx_x64, l);
                    let group_is_odd = ctx.and_u32(group, one);
                    let ql_offset_add = ctx.mul_u32_reg(group_is_odd, thirty_two);
                    let ql_byte_offset = ctx.add_u32_reg(ql_base, ql_offset_add);

                    // Load ql byte
                    let ql_byte_offset_64 = ctx.cvt_u64_u32(ql_byte_offset);
                    let ql_addr = ctx.add_u64(sb_addr, ql_byte_offset_64);
                    let ql_byte = ctx.ld_global_u8(ql_addr);
                    let ql_byte_32 = ctx.cvt_u32_u8(ql_byte);

                    // Extract nibble: low if group < 2, high if group >= 2
                    // nibble_shift = (group / 2) * 4 = (group >> 1) * 4
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

                    // Dequantize: val = d × scale × quant
                    let d_scale = ctx.mul_f32(d, scale_f32);
                    let dequant = ctx.mul_f32(d_scale, quant_signed);

                    // Load activation x[sb_idx * 256 + val_idx]
                    // GH-215 FIX: Bounds-check for non-256-aligned K dimensions.
                    // The last super-block may address indices beyond k_dim; treat
                    // those as 0.0 to avoid out-of-bounds GPU memory reads.
                    let sb_k_base = ctx.mul_u32(sb_idx, Q6K_SUPER_BLOCK_SIZE);
                    let x_idx = ctx.add_u32_reg(sb_k_base, val_idx);
                    let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                    let x_bytes = ctx.mul_u64(x_idx_64, 4);
                    let x_addr = ctx.add_u64(x_ptr, x_bytes);
                    let in_bounds = ctx.setp_lt_u32(x_idx, k_dim);
                    let x_val = ctx.ld_global_f32_predicated(x_addr, in_bounds, 0.0);

                    ctx.fma_f32_inplace(thread_partial, x_val, dequant);
                }

                ctx.add_f32_inplace(acc, thread_partial);
                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_end");

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
