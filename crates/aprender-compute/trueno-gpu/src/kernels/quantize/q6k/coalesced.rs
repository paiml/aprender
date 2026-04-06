//! PAR-066: Coalesced Q6_K GEMV kernel with vectorized scale loading.

use crate::kernels::quantize::{Kernel, Q6K_SUPER_BLOCK_BYTES, Q6K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

// =============================================================================
// PAR-066: COALESCED Q6_K GEMV KERNEL
// =============================================================================

/// Coalesced Q6_K GEMV kernel with vectorized scale loading (PAR-066)
///
/// Five-Whys Root Cause: Q6KGemvKernel uses single-byte loads for all 16 scales,
/// causing 16 separate memory transactions per super-block. This kernel loads
/// all scales as 4 x u32 via lane 0, then broadcasts via warp shuffle.
///
/// # Memory Access Pattern
///
/// **Before (Q6KGemvKernel):** 16 × ld_global_u8 = 16 memory transactions
/// **After (Coalesced):** 4 × ld_global_u32 + warp shuffle = 4 transactions
///
/// # Q6_K Layout (210 bytes per 256 values)
///
/// - ql[128]: bytes 0-127, low 4-bits packed 2 per byte
/// - qh[64]: bytes 128-191, high 2-bits packed 4 per byte
/// - scales[16]: bytes 192-207, signed i8 per 16-element sub-block
/// - d: bytes 208-209, f16 scale factor
///
/// # Performance Target
///
/// - Qwen2 1.5B FFN down_proj uses Q6_K (bottleneck identified in PAR-065)
/// - Expected 20-30% improvement from reduced memory transactions
#[derive(Debug, Clone)]
pub struct CoalescedQ6KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl CoalescedQ6KGemvKernel {
    /// Create a new coalesced Q6_K GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Get number of super-blocks per row
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        (self.k + Q6K_SUPER_BLOCK_SIZE - 1) / Q6K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for CoalescedQ6KGemvKernel {
    fn name(&self) -> &str {
        "coalesced_q6k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("coalesced_q6k_gemv")
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

                let acc = ctx.mov_f32_imm(0.0);

                // Calculate super-blocks per row
                let k_rounded = ctx.add_u32(k_dim, Q6K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q6K_SUPER_BLOCK_SIZE);

                // Row base address
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

                // ========================================================
                // PAR-066 OPTIMIZATION: Byte-wise scale loading + warp shuffle
                // Q6K super-blocks are 210 bytes (NOT 4-byte aligned!)
                // So we use byte loads + warp shuffle to share scales
                // Lanes 0-15 each load one scale byte, then broadcast via shuffle
                // ========================================================
                let scales_base_offset = ctx.mov_u64_imm(192);
                let scales_base = ctx.add_u64(sb_addr, scales_base_offset);

                // Each of lanes 0-15 loads one scale byte
                // Lanes 16-31 will get their values via warp shuffle
                let lane_mod_16 = ctx.rem_u32(lane_id, 16);
                let lane_offset = ctx.cvt_u64_u32(lane_mod_16);
                let scale_addr = ctx.add_u64(scales_base, lane_offset);

                // Load scale byte for this lane (lanes 0-15) or 0 (lanes 16-31)
                let my_scale_byte = ctx.mov_u32_imm(0);
                let sixteen_const = ctx.mov_u32_imm(16);
                let is_low_lane = ctx.setp_lt_u32(lane_id, sixteen_const);
                ctx.branch_if_not(is_low_lane, "skip_scale_load");
                let scale_u8 = ctx.ld_global_u8(scale_addr);
                let scale_u32 = ctx.cvt_u32_u8(scale_u8);
                ctx.mov_u32_reg(my_scale_byte, scale_u32);
                ctx.label("skip_scale_load");

                // Broadcast all 16 scales via warp shuffle
                // Each lane gets scale[0..15] by shuffling from lanes 0..15
                let s0_u32 = ctx.shfl_idx_u32(my_scale_byte, 0, 0xFFFF_FFFF);
                let s1_u32 = ctx.shfl_idx_u32(my_scale_byte, 1, 0xFFFF_FFFF);
                let s2_u32 = ctx.shfl_idx_u32(my_scale_byte, 2, 0xFFFF_FFFF);
                let s3_u32 = ctx.shfl_idx_u32(my_scale_byte, 3, 0xFFFF_FFFF);
                let s4_u32 = ctx.shfl_idx_u32(my_scale_byte, 4, 0xFFFF_FFFF);
                let s5_u32 = ctx.shfl_idx_u32(my_scale_byte, 5, 0xFFFF_FFFF);
                let s6_u32 = ctx.shfl_idx_u32(my_scale_byte, 6, 0xFFFF_FFFF);
                let s7_u32 = ctx.shfl_idx_u32(my_scale_byte, 7, 0xFFFF_FFFF);
                let s8_u32 = ctx.shfl_idx_u32(my_scale_byte, 8, 0xFFFF_FFFF);
                let s9_u32 = ctx.shfl_idx_u32(my_scale_byte, 9, 0xFFFF_FFFF);
                let s10_u32 = ctx.shfl_idx_u32(my_scale_byte, 10, 0xFFFF_FFFF);
                let s11_u32 = ctx.shfl_idx_u32(my_scale_byte, 11, 0xFFFF_FFFF);
                let s12_u32 = ctx.shfl_idx_u32(my_scale_byte, 12, 0xFFFF_FFFF);
                let s13_u32 = ctx.shfl_idx_u32(my_scale_byte, 13, 0xFFFF_FFFF);
                let s14_u32 = ctx.shfl_idx_u32(my_scale_byte, 14, 0xFFFF_FFFF);
                let s15_u32 = ctx.shfl_idx_u32(my_scale_byte, 15, 0xFFFF_FFFF);

                // For packing into u32, create combined values (for reference only)
                let _scales_0_3_bcast = s0_u32; // placeholder for old code compatibility
                let _scales_4_7_bcast = s4_u32;
                let _scales_8_11_bcast = s8_u32;
                let _scales_12_15_bcast = s12_u32;

                // Convert individual scale bytes to signed f32
                // Scale bytes are already in s0_u32..s15_u32 from warp shuffle above
                // Convert u8 to signed i8 as f32: if >= 128, subtract 256
                let seven = ctx.mov_u32_imm(7);
                let twofiftysix_f32 = ctx.mov_f32_imm(256.0);

                // Helper: convert u8 to signed f32
                // sign = (val >> 7), correction = sign * 256, result = val - correction
                let s0_sign = ctx.shr_u32(s0_u32, seven);
                let s0_f32_raw = ctx.cvt_f32_u32(s0_u32);
                let s0_sign_f32 = ctx.cvt_f32_u32(s0_sign);
                let s0_correction = ctx.mul_f32(s0_sign_f32, twofiftysix_f32);
                let scale0 = ctx.sub_f32(s0_f32_raw, s0_correction);

                let s1_sign = ctx.shr_u32(s1_u32, seven);
                let s1_f32_raw = ctx.cvt_f32_u32(s1_u32);
                let s1_sign_f32 = ctx.cvt_f32_u32(s1_sign);
                let s1_correction = ctx.mul_f32(s1_sign_f32, twofiftysix_f32);
                let scale1 = ctx.sub_f32(s1_f32_raw, s1_correction);

                let s2_sign = ctx.shr_u32(s2_u32, seven);
                let s2_f32_raw = ctx.cvt_f32_u32(s2_u32);
                let s2_sign_f32 = ctx.cvt_f32_u32(s2_sign);
                let s2_correction = ctx.mul_f32(s2_sign_f32, twofiftysix_f32);
                let scale2 = ctx.sub_f32(s2_f32_raw, s2_correction);

                let s3_sign = ctx.shr_u32(s3_u32, seven);
                let s3_f32_raw = ctx.cvt_f32_u32(s3_u32);
                let s3_sign_f32 = ctx.cvt_f32_u32(s3_sign);
                let s3_correction = ctx.mul_f32(s3_sign_f32, twofiftysix_f32);
                let scale3 = ctx.sub_f32(s3_f32_raw, s3_correction);

                let s4_sign = ctx.shr_u32(s4_u32, seven);
                let s4_f32_raw = ctx.cvt_f32_u32(s4_u32);
                let s4_sign_f32 = ctx.cvt_f32_u32(s4_sign);
                let s4_correction = ctx.mul_f32(s4_sign_f32, twofiftysix_f32);
                let scale4 = ctx.sub_f32(s4_f32_raw, s4_correction);

                let s5_sign = ctx.shr_u32(s5_u32, seven);
                let s5_f32_raw = ctx.cvt_f32_u32(s5_u32);
                let s5_sign_f32 = ctx.cvt_f32_u32(s5_sign);
                let s5_correction = ctx.mul_f32(s5_sign_f32, twofiftysix_f32);
                let scale5 = ctx.sub_f32(s5_f32_raw, s5_correction);

                let s6_sign = ctx.shr_u32(s6_u32, seven);
                let s6_f32_raw = ctx.cvt_f32_u32(s6_u32);
                let s6_sign_f32 = ctx.cvt_f32_u32(s6_sign);
                let s6_correction = ctx.mul_f32(s6_sign_f32, twofiftysix_f32);
                let scale6 = ctx.sub_f32(s6_f32_raw, s6_correction);

                let s7_sign = ctx.shr_u32(s7_u32, seven);
                let s7_f32_raw = ctx.cvt_f32_u32(s7_u32);
                let s7_sign_f32 = ctx.cvt_f32_u32(s7_sign);
                let s7_correction = ctx.mul_f32(s7_sign_f32, twofiftysix_f32);
                let scale7 = ctx.sub_f32(s7_f32_raw, s7_correction);

                let s8_sign = ctx.shr_u32(s8_u32, seven);
                let s8_f32_raw = ctx.cvt_f32_u32(s8_u32);
                let s8_sign_f32 = ctx.cvt_f32_u32(s8_sign);
                let s8_correction = ctx.mul_f32(s8_sign_f32, twofiftysix_f32);
                let scale8 = ctx.sub_f32(s8_f32_raw, s8_correction);

                let s9_sign = ctx.shr_u32(s9_u32, seven);
                let s9_f32_raw = ctx.cvt_f32_u32(s9_u32);
                let s9_sign_f32 = ctx.cvt_f32_u32(s9_sign);
                let s9_correction = ctx.mul_f32(s9_sign_f32, twofiftysix_f32);
                let scale9 = ctx.sub_f32(s9_f32_raw, s9_correction);

                let s10_sign = ctx.shr_u32(s10_u32, seven);
                let s10_f32_raw = ctx.cvt_f32_u32(s10_u32);
                let s10_sign_f32 = ctx.cvt_f32_u32(s10_sign);
                let s10_correction = ctx.mul_f32(s10_sign_f32, twofiftysix_f32);
                let scale10 = ctx.sub_f32(s10_f32_raw, s10_correction);

                let s11_sign = ctx.shr_u32(s11_u32, seven);
                let s11_f32_raw = ctx.cvt_f32_u32(s11_u32);
                let s11_sign_f32 = ctx.cvt_f32_u32(s11_sign);
                let s11_correction = ctx.mul_f32(s11_sign_f32, twofiftysix_f32);
                let scale11 = ctx.sub_f32(s11_f32_raw, s11_correction);

                let s12_sign = ctx.shr_u32(s12_u32, seven);
                let s12_f32_raw = ctx.cvt_f32_u32(s12_u32);
                let s12_sign_f32 = ctx.cvt_f32_u32(s12_sign);
                let s12_correction = ctx.mul_f32(s12_sign_f32, twofiftysix_f32);
                let scale12 = ctx.sub_f32(s12_f32_raw, s12_correction);

                let s13_sign = ctx.shr_u32(s13_u32, seven);
                let s13_f32_raw = ctx.cvt_f32_u32(s13_u32);
                let s13_sign_f32 = ctx.cvt_f32_u32(s13_sign);
                let s13_correction = ctx.mul_f32(s13_sign_f32, twofiftysix_f32);
                let scale13 = ctx.sub_f32(s13_f32_raw, s13_correction);

                let s14_sign = ctx.shr_u32(s14_u32, seven);
                let s14_f32_raw = ctx.cvt_f32_u32(s14_u32);
                let s14_sign_f32 = ctx.cvt_f32_u32(s14_sign);
                let s14_correction = ctx.mul_f32(s14_sign_f32, twofiftysix_f32);
                let scale14 = ctx.sub_f32(s14_f32_raw, s14_correction);

                let s15_sign = ctx.shr_u32(s15_u32, seven);
                let s15_f32_raw = ctx.cvt_f32_u32(s15_u32);
                let s15_sign_f32 = ctx.cvt_f32_u32(s15_sign);
                let s15_correction = ctx.mul_f32(s15_sign_f32, twofiftysix_f32);
                let scale15 = ctx.sub_f32(s15_f32_raw, s15_correction);

                // Precompute d * scale for all 16 scales
                let ds0 = ctx.mul_f32(d, scale0);
                let ds1 = ctx.mul_f32(d, scale1);
                let ds2 = ctx.mul_f32(d, scale2);
                let ds3 = ctx.mul_f32(d, scale3);
                let ds4 = ctx.mul_f32(d, scale4);
                let ds5 = ctx.mul_f32(d, scale5);
                let ds6 = ctx.mul_f32(d, scale6);
                let ds7 = ctx.mul_f32(d, scale7);
                let ds8 = ctx.mul_f32(d, scale8);
                let ds9 = ctx.mul_f32(d, scale9);
                let ds10 = ctx.mul_f32(d, scale10);
                let ds11 = ctx.mul_f32(d, scale11);
                let ds12 = ctx.mul_f32(d, scale12);
                let ds13 = ctx.mul_f32(d, scale13);
                let ds14 = ctx.mul_f32(d, scale14);
                let ds15 = ctx.mul_f32(d, scale15);

                // Process 8 values per thread at offsets 0, 32, 64, 96, 128, 160, 192, 224
                // PAR-066 OPTIMIZATION: Scale index is deterministic per offset
                // scale_idx = 8 * n_idx + is + 2 * group
                // For lanes 0-15: is=0, so scale_idx = 8*n_idx + 2*group
                // For lanes 16-31: is=1, so scale_idx = 8*n_idx + 2*group + 1
                // This means each offset needs only 2 ds values, selected by lane_id < 16
                let thread_partial = ctx.mov_f32_imm(0.0);
                let thirty_two_f32 = ctx.mov_f32_imm(32.0);

                // Hardcoded ds pairs for each offset (determined by n_idx, group):
                // offset 0:   n=0, g=0, base=0  -> ds0 or ds1
                // offset 32:  n=0, g=1, base=2  -> ds2 or ds3
                // offset 64:  n=0, g=2, base=4  -> ds4 or ds5
                // offset 96:  n=0, g=3, base=6  -> ds6 or ds7
                // offset 128: n=1, g=0, base=8  -> ds8 or ds9
                // offset 160: n=1, g=1, base=10 -> ds10 or ds11
                // offset 192: n=1, g=2, base=12 -> ds12 or ds13
                // offset 224: n=1, g=3, base=14 -> ds14 or ds15

                // Precompute ds_selected for each offset using conditional add
                // ds_selected = is_low_lane ? ds[base] : ds[base+1]
                // Use FMA: ds[base] + (is_high_lane * (ds[base+1] - ds[base]))
                let ds_diff_0 = ctx.sub_f32(ds1, ds0);
                let ds_diff_1 = ctx.sub_f32(ds3, ds2);
                let ds_diff_2 = ctx.sub_f32(ds5, ds4);
                let ds_diff_3 = ctx.sub_f32(ds7, ds6);
                let ds_diff_4 = ctx.sub_f32(ds9, ds8);
                let ds_diff_5 = ctx.sub_f32(ds11, ds10);
                let ds_diff_6 = ctx.sub_f32(ds13, ds12);
                let ds_diff_7 = ctx.sub_f32(ds15, ds14);

                // Compute lane_is (0 for lanes 0-15, 1 for lanes 16-31)
                // div_u32 takes a constant, so use 16 directly
                let lane_is = ctx.div_u32(lane_id, 16);
                let lane_is_f32 = ctx.cvt_f32_u32(lane_is);

                // ds_selected[i] = ds[base_i] + lane_is_f32 * ds_diff_i
                let ds_sel_0 = ctx.fma_f32(lane_is_f32, ds_diff_0, ds0);
                let ds_sel_1 = ctx.fma_f32(lane_is_f32, ds_diff_1, ds2);
                let ds_sel_2 = ctx.fma_f32(lane_is_f32, ds_diff_2, ds4);
                let ds_sel_3 = ctx.fma_f32(lane_is_f32, ds_diff_3, ds6);
                let ds_sel_4 = ctx.fma_f32(lane_is_f32, ds_diff_4, ds8);
                let ds_sel_5 = ctx.fma_f32(lane_is_f32, ds_diff_5, ds10);
                let ds_sel_6 = ctx.fma_f32(lane_is_f32, ds_diff_6, ds12);
                let ds_sel_7 = ctx.fma_f32(lane_is_f32, ds_diff_7, ds14);

                // Process each of 8 offsets with hardcoded parameters
                // (offset, n_idx, group, ds_selected)
                let offset_params: [(u32, u32, u32); 8] = [
                    (0, 0, 0),
                    (32, 0, 1),
                    (64, 0, 2),
                    (96, 0, 3),
                    (128, 1, 0),
                    (160, 1, 1),
                    (192, 1, 2),
                    (224, 1, 3),
                ];

                for (i, (offset, n_idx_val, group_val)) in offset_params.iter().enumerate() {
                    let offset_reg = ctx.mov_u32_imm(*offset);
                    let val_idx = ctx.add_u32_reg(lane_id, offset_reg);

                    // Select the precomputed ds_selected for this offset
                    let ds_selected = match i {
                        0 => ds_sel_0,
                        1 => ds_sel_1,
                        2 => ds_sel_2,
                        3 => ds_sel_3,
                        4 => ds_sel_4,
                        5 => ds_sel_5,
                        6 => ds_sel_6,
                        _ => ds_sel_7,
                    };

                    // l = lane_id (since all offsets are multiples of 32)
                    let l = lane_id;

                    // n_idx and group are compile-time constants for this offset
                    let n_idx = ctx.mov_u32_imm(*n_idx_val);
                    let group = ctx.mov_u32_imm(*group_val);

                    // ql_byte_offset = 64 * n_idx + l + (32 * group_is_odd)
                    let sixty_four = ctx.mov_u32_imm(64);
                    let thirty_two = ctx.mov_u32_imm(32);
                    let one_32 = ctx.mov_u32_imm(1);
                    let n_idx_x64 = ctx.mul_u32_reg(n_idx, sixty_four);
                    let ql_base = ctx.add_u32_reg(n_idx_x64, l);
                    let group_is_odd = ctx.and_u32(group, one_32);
                    let ql_offset_add = ctx.mul_u32_reg(group_is_odd, thirty_two);
                    let ql_byte_offset = ctx.add_u32_reg(ql_base, ql_offset_add);

                    // Load ql byte
                    let ql_byte_offset_64 = ctx.cvt_u64_u32(ql_byte_offset);
                    let ql_addr = ctx.add_u64(sb_addr, ql_byte_offset_64);
                    let ql_byte = ctx.ld_global_u8(ql_addr);
                    let ql_byte_32 = ctx.cvt_u32_u8(ql_byte);

                    // Extract nibble: low if group < 2, high if group >= 2
                    let group_div_2 = ctx.shr_u32(group, one_32);
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
                    let two = ctx.mov_u32_imm(2);
                    let qh_shift = ctx.mul_u32_reg(group, two);
                    let qh_shifted = ctx.shr_u32(qh_byte_32, qh_shift);
                    let mask_0x3 = ctx.mov_u32_imm(0x3);
                    let qh_2bits = ctx.and_u32(qh_shifted, mask_0x3);

                    // Combine: quant = ql_nibble | (qh_2bits << 4) - 32
                    let qh_shifted_up = ctx.shl_u32(qh_2bits, four);
                    let combined = ctx.or_u32(ql_nibble, qh_shifted_up);
                    let combined_f32 = ctx.cvt_f32_u32(combined);
                    let quant_signed = ctx.sub_f32(combined_f32, thirty_two_f32);

                    // Dequantize: val = d * scale * quant = ds_selected * quant
                    let dequant = ctx.mul_f32(ds_selected, quant_signed);

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
