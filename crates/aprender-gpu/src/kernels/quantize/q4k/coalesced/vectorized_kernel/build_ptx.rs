//! PTX code generation for the vectorized Q4_K GEMV kernel
//!
//! Coalesced u32 loads for high memory bandwidth (PAR-069).
//! Each thread loads 4 consecutive bytes (8 nibbles = 8 Q4 values).
//! 32 threads x 4 bytes = 128 bytes per warp transaction (perfectly coalesced).

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use super::VectorizedQ4KGemvKernel;
use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

impl Kernel for VectorizedQ4KGemvKernel {
    fn name(&self) -> &str {
        "vectorized_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Grid: one warp (32 threads) per output row
        // Each thread loads 4 bytes = 8 nibbles = 8 values per super-block
        // Total: 32 threads x 8 values = 256 values = 1 super-block per iteration
        PtxKernel::new("vectorized_q4k_gemv")
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
                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                // Row base address
                let sb_bytes = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop_v");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_end_v");

                let sb_offset = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d and dmin (all lanes)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // ========================================================
                // PAR-069: VECTORIZED SCALE LOADING (from CoalescedQ4K)
                // Lane 0 loads scales as 3 x u32, broadcasts via shuffle
                // ========================================================
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);
                let one = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);

                let scales_0_3 = ctx.mov_u32_imm(0);
                let scales_4_7 = ctx.mov_u32_imm(0);
                let scales_8_11 = ctx.mov_u32_imm(0);

                ctx.branch_if_not(is_lane0, "skip_scale_load_v");
                ctx.ld_global_u32_into(scales_0_3, scales_base);
                let four_64b = ctx.mov_u64_imm(4);
                let scales_4_addr = ctx.add_u64(scales_base, four_64b);
                ctx.ld_global_u32_into(scales_4_7, scales_4_addr);
                let eight_64 = ctx.mov_u64_imm(8);
                let scales_8_addr = ctx.add_u64(scales_base, eight_64);
                ctx.ld_global_u32_into(scales_8_11, scales_8_addr);
                ctx.label("skip_scale_load_v");

                let scales_0_3_bcast = ctx.shfl_idx_u32(scales_0_3, 0, 0xFFFF_FFFF);
                let scales_4_7_bcast = ctx.shfl_idx_u32(scales_4_7, 0, 0xFFFF_FFFF);
                let scales_8_11_bcast = ctx.shfl_idx_u32(scales_8_11, 0, 0xFFFF_FFFF);

                // Extract individual scale bytes
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

                // Constants for scale/min extraction
                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let four_const = ctx.mov_u32_imm(4);
                let six = ctx.mov_u32_imm(6);

                // Extract scale/min for all 8 blocks
                // Block 0-3: simple extraction
                let scale0 = ctx.and_u32(s0_32, mask_6bit);
                let min0 = ctx.and_u32(s4_32, mask_6bit);
                let scale1 = ctx.and_u32(s1_32, mask_6bit);
                let min1 = ctx.and_u32(s5_32, mask_6bit);
                let scale2 = ctx.and_u32(s2_32, mask_6bit);
                let min2 = ctx.and_u32(s6_32, mask_6bit);
                let scale3 = ctx.and_u32(s3_32, mask_6bit);
                let min3 = ctx.and_u32(s7_32, mask_6bit);

                // Block 4-7: complex extraction (6-bit packed)
                let s8_lo = ctx.and_u32(s8_32, mask_4bit);
                let s0_hi = ctx.shr_u32(s0_32, six);
                let s0_hi_shifted = ctx.shl_u32(s0_hi, four_const);
                let scale4 = ctx.or_u32(s8_lo, s0_hi_shifted);
                let s8_hi = ctx.shr_u32(s8_32, four_const);
                let s4_hi = ctx.shr_u32(s4_32, six);
                let s4_hi_shifted = ctx.shl_u32(s4_hi, four_const);
                let min4 = ctx.or_u32(s8_hi, s4_hi_shifted);

                let s9_lo = ctx.and_u32(s9_32, mask_4bit);
                let s1_hi = ctx.shr_u32(s1_32, six);
                let s1_hi_shifted = ctx.shl_u32(s1_hi, four_const);
                let scale5 = ctx.or_u32(s9_lo, s1_hi_shifted);
                let s9_hi = ctx.shr_u32(s9_32, four_const);
                let s5_hi = ctx.shr_u32(s5_32, six);
                let s5_hi_shifted = ctx.shl_u32(s5_hi, four_const);
                let min5 = ctx.or_u32(s9_hi, s5_hi_shifted);

                let s10_lo = ctx.and_u32(s10_32, mask_4bit);
                let s2_hi = ctx.shr_u32(s2_32, six);
                let s2_hi_shifted = ctx.shl_u32(s2_hi, four_const);
                let scale6 = ctx.or_u32(s10_lo, s2_hi_shifted);
                let s10_hi = ctx.shr_u32(s10_32, four_const);
                let s6_hi = ctx.shr_u32(s6_32, six);
                let s6_hi_shifted = ctx.shl_u32(s6_hi, four_const);
                let min6 = ctx.or_u32(s10_hi, s6_hi_shifted);

                let s11_lo = ctx.and_u32(s11_32, mask_4bit);
                let s3_hi = ctx.shr_u32(s3_32, six);
                let s3_hi_shifted = ctx.shl_u32(s3_hi, four_const);
                let scale7 = ctx.or_u32(s11_lo, s3_hi_shifted);
                let s11_hi = ctx.shr_u32(s11_32, four_const);
                let s7_hi = ctx.shr_u32(s7_32, six);
                let s7_hi_shifted = ctx.shl_u32(s7_hi, four_const);
                let min7 = ctx.or_u32(s11_hi, s7_hi_shifted);

                // Convert to f32 and precompute d*scale, dmin*min for all blocks
                let scale0_f = ctx.cvt_f32_u32(scale0);
                let min0_f = ctx.cvt_f32_u32(min0);
                let ds0 = ctx.mul_f32(d, scale0_f);
                let dm0 = ctx.mul_f32(dmin, min0_f);

                let scale1_f = ctx.cvt_f32_u32(scale1);
                let min1_f = ctx.cvt_f32_u32(min1);
                let ds1 = ctx.mul_f32(d, scale1_f);
                let dm1 = ctx.mul_f32(dmin, min1_f);

                let scale2_f = ctx.cvt_f32_u32(scale2);
                let min2_f = ctx.cvt_f32_u32(min2);
                let ds2 = ctx.mul_f32(d, scale2_f);
                let dm2 = ctx.mul_f32(dmin, min2_f);

                let scale3_f = ctx.cvt_f32_u32(scale3);
                let min3_f = ctx.cvt_f32_u32(min3);
                let ds3 = ctx.mul_f32(d, scale3_f);
                let dm3 = ctx.mul_f32(dmin, min3_f);

                let scale4_f = ctx.cvt_f32_u32(scale4);
                let min4_f = ctx.cvt_f32_u32(min4);
                let ds4 = ctx.mul_f32(d, scale4_f);
                let dm4 = ctx.mul_f32(dmin, min4_f);

                let scale5_f = ctx.cvt_f32_u32(scale5);
                let min5_f = ctx.cvt_f32_u32(min5);
                let ds5 = ctx.mul_f32(d, scale5_f);
                let dm5 = ctx.mul_f32(dmin, min5_f);

                let scale6_f = ctx.cvt_f32_u32(scale6);
                let min6_f = ctx.cvt_f32_u32(min6);
                let ds6 = ctx.mul_f32(d, scale6_f);
                let dm6 = ctx.mul_f32(dmin, min6_f);

                let scale7_f = ctx.cvt_f32_u32(scale7);
                let min7_f = ctx.cvt_f32_u32(min7);
                let ds7 = ctx.mul_f32(d, scale7_f);
                let dm7 = ctx.mul_f32(dmin, min7_f);

                // ============================================================
                // PAR-069: COALESCED WEIGHT LOADING
                // Each thread loads 4 consecutive bytes (u32) = 8 nibbles
                // 32 threads x 4 bytes = 128 bytes per warp (1 memory transaction!)
                // ============================================================
                let sixteen_64 = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen_64);

                let four = ctx.mov_u32_imm(4);
                let thread_byte_offset = ctx.mul_u32_reg(lane_id, four);
                let thread_byte_offset_64 = ctx.cvt_u64_u32(thread_byte_offset);
                let qs_addr = ctx.add_u64(qs_base, thread_byte_offset_64);

                // COALESCED u32 LOAD: 32 threads x 4 bytes = 128 bytes per transaction
                let packed_u32 = ctx.ld_global_u32(qs_addr);

                // Unpack 8 nibbles from 4 bytes
                let nib0 = ctx.and_u32(packed_u32, mask_4bit);
                let shift4 = ctx.mov_u32_imm(4);
                let nib1 = ctx.shr_u32(packed_u32, shift4);
                let nib1 = ctx.and_u32(nib1, mask_4bit);
                let shift8_const = ctx.mov_u32_imm(8);
                let nib2 = ctx.shr_u32(packed_u32, shift8_const);
                let nib2 = ctx.and_u32(nib2, mask_4bit);
                let shift12 = ctx.mov_u32_imm(12);
                let nib3 = ctx.shr_u32(packed_u32, shift12);
                let nib3 = ctx.and_u32(nib3, mask_4bit);
                let shift16_const = ctx.mov_u32_imm(16);
                let nib4 = ctx.shr_u32(packed_u32, shift16_const);
                let nib4 = ctx.and_u32(nib4, mask_4bit);
                let shift20 = ctx.mov_u32_imm(20);
                let nib5 = ctx.shr_u32(packed_u32, shift20);
                let nib5 = ctx.and_u32(nib5, mask_4bit);
                let shift24_const = ctx.mov_u32_imm(24);
                let nib6 = ctx.shr_u32(packed_u32, shift24_const);
                let nib6 = ctx.and_u32(nib6, mask_4bit);
                let shift28 = ctx.mov_u32_imm(28);
                let nib7 = ctx.shr_u32(packed_u32, shift28);

                // ============================================================
                // CORRECTNESS-002 FIX: Q4K DEINTERLEAVED NIBBLE LAYOUT
                // ============================================================

                // Compute chunk index (which 64-value block we're in)
                let three_const = ctx.mov_u32_imm(3);
                let chunk_idx = ctx.shr_u32(lane_id, three_const); // lane_id / 8

                // Compute scale indices for low and high nibbles
                let low_scale_idx = ctx.shl_u32(chunk_idx, one); // chunk * 2
                let high_scale_idx = ctx.add_u32(low_scale_idx, 1); // chunk * 2 + 1

                // Select low scale (for nib0, nib2, nib4, nib6)
                let ds_low = ds0;
                let dm_low = dm0;
                let is_low1 = ctx.setp_eq_u32(low_scale_idx, one);
                let ds_low = ctx.selp_f32(is_low1, ds1, ds_low);
                let dm_low = ctx.selp_f32(is_low1, dm1, dm_low);
                let two_u32 = ctx.mov_u32_imm(2);
                let is_low2 = ctx.setp_eq_u32(low_scale_idx, two_u32);
                let ds_low = ctx.selp_f32(is_low2, ds2, ds_low);
                let dm_low = ctx.selp_f32(is_low2, dm2, dm_low);
                let three_u32 = ctx.mov_u32_imm(3);
                let is_low3 = ctx.setp_eq_u32(low_scale_idx, three_u32);
                let ds_low = ctx.selp_f32(is_low3, ds3, ds_low);
                let dm_low = ctx.selp_f32(is_low3, dm3, dm_low);
                let is_low4 = ctx.setp_eq_u32(low_scale_idx, four);
                let ds_low = ctx.selp_f32(is_low4, ds4, ds_low);
                let dm_low = ctx.selp_f32(is_low4, dm4, dm_low);
                let five_u32 = ctx.mov_u32_imm(5);
                let is_low5 = ctx.setp_eq_u32(low_scale_idx, five_u32);
                let ds_low = ctx.selp_f32(is_low5, ds5, ds_low);
                let dm_low = ctx.selp_f32(is_low5, dm5, dm_low);
                let six_u32 = ctx.mov_u32_imm(6);
                let is_low6 = ctx.setp_eq_u32(low_scale_idx, six_u32);
                let ds_low = ctx.selp_f32(is_low6, ds6, ds_low);
                let dm_low = ctx.selp_f32(is_low6, dm6, dm_low);
                let seven_u32 = ctx.mov_u32_imm(7);
                let is_low7 = ctx.setp_eq_u32(low_scale_idx, seven_u32);
                let ds_low = ctx.selp_f32(is_low7, ds7, ds_low);
                let dm_low = ctx.selp_f32(is_low7, dm7, dm_low);

                // Select high scale (for nib1, nib3, nib5, nib7)
                let ds_high = ds0;
                let dm_high = dm0;
                let is_high1 = ctx.setp_eq_u32(high_scale_idx, one);
                let ds_high = ctx.selp_f32(is_high1, ds1, ds_high);
                let dm_high = ctx.selp_f32(is_high1, dm1, dm_high);
                let is_high2 = ctx.setp_eq_u32(high_scale_idx, two_u32);
                let ds_high = ctx.selp_f32(is_high2, ds2, ds_high);
                let dm_high = ctx.selp_f32(is_high2, dm2, dm_high);
                let is_high3 = ctx.setp_eq_u32(high_scale_idx, three_u32);
                let ds_high = ctx.selp_f32(is_high3, ds3, ds_high);
                let dm_high = ctx.selp_f32(is_high3, dm3, dm_high);
                let is_high4 = ctx.setp_eq_u32(high_scale_idx, four);
                let ds_high = ctx.selp_f32(is_high4, ds4, ds_high);
                let dm_high = ctx.selp_f32(is_high4, dm4, dm_high);
                let is_high5 = ctx.setp_eq_u32(high_scale_idx, five_u32);
                let ds_high = ctx.selp_f32(is_high5, ds5, ds_high);
                let dm_high = ctx.selp_f32(is_high5, dm5, dm_high);
                let is_high6 = ctx.setp_eq_u32(high_scale_idx, six_u32);
                let ds_high = ctx.selp_f32(is_high6, ds6, ds_high);
                let dm_high = ctx.selp_f32(is_high6, dm6, dm_high);
                let is_high7 = ctx.setp_eq_u32(high_scale_idx, seven_u32);
                let ds_high = ctx.selp_f32(is_high7, ds7, ds_high);
                let dm_high = ctx.selp_f32(is_high7, dm7, dm_high);

                // Convert nibbles to f32
                let nib0_f = ctx.cvt_f32_u32(nib0);
                let nib1_f = ctx.cvt_f32_u32(nib1);
                let nib2_f = ctx.cvt_f32_u32(nib2);
                let nib3_f = ctx.cvt_f32_u32(nib3);
                let nib4_f = ctx.cvt_f32_u32(nib4);
                let nib5_f = ctx.cvt_f32_u32(nib5);
                let nib6_f = ctx.cvt_f32_u32(nib6);
                let nib7_f = ctx.cvt_f32_u32(nib7);

                // Dequantize with CORRECT scale selection
                let dq0 = ctx.mul_f32(ds_low, nib0_f);
                let dq0 = ctx.sub_f32(dq0, dm_low);
                let dq1 = ctx.mul_f32(ds_high, nib1_f);
                let dq1 = ctx.sub_f32(dq1, dm_high);
                let dq2 = ctx.mul_f32(ds_low, nib2_f);
                let dq2 = ctx.sub_f32(dq2, dm_low);
                let dq3 = ctx.mul_f32(ds_high, nib3_f);
                let dq3 = ctx.sub_f32(dq3, dm_high);
                let dq4 = ctx.mul_f32(ds_low, nib4_f);
                let dq4 = ctx.sub_f32(dq4, dm_low);
                let dq5 = ctx.mul_f32(ds_high, nib5_f);
                let dq5 = ctx.sub_f32(dq5, dm_high);
                let dq6 = ctx.mul_f32(ds_low, nib6_f);
                let dq6 = ctx.sub_f32(dq6, dm_low);
                let dq7 = ctx.mul_f32(ds_high, nib7_f);
                let dq7 = ctx.sub_f32(dq7, dm_high);

                // ============================================================
                // CORRECTNESS-002 FIX: CORRECT ACTIVATION INDICES
                // ============================================================
                let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                let sixty_four = ctx.mov_u32_imm(64);
                let chunk_base = ctx.mul_u32_reg(chunk_idx, sixty_four);
                let chunk_start = ctx.add_u32_reg(sb_k_base, chunk_base);

                // byte_in_chunk = (lane_id % 8) * 4
                let seven_mask = ctx.mov_u32_imm(7);
                let lane_in_chunk = ctx.and_u32(lane_id, seven_mask);
                let byte_in_chunk = ctx.shl_u32(lane_in_chunk, two_u32);

                // Base for low nibbles
                let low_base = ctx.add_u32_reg(chunk_start, byte_in_chunk);
                // Base for high nibbles
                let thirty_two = ctx.mov_u32_imm(32);
                let high_base = ctx.add_u32_reg(chunk_start, thirty_two);
                let high_base = ctx.add_u32_reg(high_base, byte_in_chunk);

                let thread_partial = ctx.mov_f32_imm(0.0);

                // LOW nibbles: values at low_base + 0, 1, 2, 3
                let x_idx0_64 = ctx.cvt_u64_u32(low_base);
                let x_off0 = ctx.mul_u64(x_idx0_64, 4);
                let x_addr0 = ctx.add_u64(x_ptr, x_off0);
                let x0 = ctx.ld_global_f32(x_addr0);
                ctx.fma_f32_inplace(thread_partial, x0, dq0);

                let x_idx2 = ctx.add_u32(low_base, 1);
                let x_idx2_64 = ctx.cvt_u64_u32(x_idx2);
                let x_off2 = ctx.mul_u64(x_idx2_64, 4);
                let x_addr2 = ctx.add_u64(x_ptr, x_off2);
                let x2 = ctx.ld_global_f32(x_addr2);
                ctx.fma_f32_inplace(thread_partial, x2, dq2);

                let x_idx4 = ctx.add_u32(low_base, 2);
                let x_idx4_64 = ctx.cvt_u64_u32(x_idx4);
                let x_off4 = ctx.mul_u64(x_idx4_64, 4);
                let x_addr4 = ctx.add_u64(x_ptr, x_off4);
                let x4 = ctx.ld_global_f32(x_addr4);
                ctx.fma_f32_inplace(thread_partial, x4, dq4);

                let x_idx6 = ctx.add_u32(low_base, 3);
                let x_idx6_64 = ctx.cvt_u64_u32(x_idx6);
                let x_off6 = ctx.mul_u64(x_idx6_64, 4);
                let x_addr6 = ctx.add_u64(x_ptr, x_off6);
                let x6 = ctx.ld_global_f32(x_addr6);
                ctx.fma_f32_inplace(thread_partial, x6, dq6);

                // HIGH nibbles: values at high_base + 0, 1, 2, 3
                let x_idx1_64 = ctx.cvt_u64_u32(high_base);
                let x_off1 = ctx.mul_u64(x_idx1_64, 4);
                let x_addr1 = ctx.add_u64(x_ptr, x_off1);
                let x1 = ctx.ld_global_f32(x_addr1);
                ctx.fma_f32_inplace(thread_partial, x1, dq1);

                let x_idx3 = ctx.add_u32(high_base, 1);
                let x_idx3_64 = ctx.cvt_u64_u32(x_idx3);
                let x_off3 = ctx.mul_u64(x_idx3_64, 4);
                let x_addr3 = ctx.add_u64(x_ptr, x_off3);
                let x3 = ctx.ld_global_f32(x_addr3);
                ctx.fma_f32_inplace(thread_partial, x3, dq3);

                let x_idx5 = ctx.add_u32(high_base, 2);
                let x_idx5_64 = ctx.cvt_u64_u32(x_idx5);
                let x_off5 = ctx.mul_u64(x_idx5_64, 4);
                let x_addr5 = ctx.add_u64(x_ptr, x_off5);
                let x5 = ctx.ld_global_f32(x_addr5);
                ctx.fma_f32_inplace(thread_partial, x5, dq5);

                let x_idx7 = ctx.add_u32(high_base, 3);
                let x_idx7_64 = ctx.cvt_u64_u32(x_idx7);
                let x_off7 = ctx.mul_u64(x_idx7_64, 4);
                let x_addr7 = ctx.add_u64(x_ptr, x_off7);
                let x7 = ctx.ld_global_f32(x_addr7);
                ctx.fma_f32_inplace(thread_partial, x7, dq7);

                ctx.add_f32_inplace(acc, thread_partial);
                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop_v");

                ctx.label("sb_loop_end_v");

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
                let is_lane0_final = ctx.setp_lt_u32(lane_id, one);
                ctx.branch_if_not(is_lane0_final, "exit");

                let y_offset = ctx.mul_wide_u32(block_id, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
