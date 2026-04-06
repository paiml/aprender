//! DP4A Q6_K GEMV kernel for decode throughput on Orin
//!
//! Vectorized Q6K dequantization with dp4a integer dot products.
//! Replaces scalar per-element processing with 4-element vectorized dp4a,
//! reducing instruction count by ~4x in the inner loop.
//!
//! # Design by Contract
//!
//! ## Preconditions
//! - `k` > 0 (need not be multiple of 256 — Q8 quantization zero-pads)
//! - `n` > 0
//! - `num_warps` ∈ {1, 2, 3, 4, 6, 8}
//! - `w_ptr`: N × ceil(K/256) × 210 bytes of Q6_K super-blocks (row-major)
//! - `q8_ptr`: ceil(K/32) × 36 bytes of Q8_1 quantized activations
//! - `y_ptr`: N writable f32 output values
//!
//! ## Postcondition
//! For each row r ∈ [0, N):
//!   y[r] = Σ_{sb} Σ_{j} dequant_q6k(w[r][sb], j) × dequant_q8(q8, sb*256+j)
//! where dequant_q6k follows the GGML Q6_K formula:
//!   quant = (ql_nibble | (qh_2bits << 4)) - 32
//!   value = d × scale[sub_block] × quant
//!
//! ## Key improvement over MultiWarpQ6KGemvKernel
//! - 4 consecutive Q6K values loaded as int32 (1 load vs 4 byte loads)
//! - dp4a.u32.s32 computes 4 multiply-adds per instruction
//! - Bias term (−32) handled via: result = dp4a(q6k, q8) − 32 × dp4a(1s, q8)
//! - Binary tree scale selection (7 selps vs 56 selps in linear chain)
//!
//! ## Invariants
//! - Each warp processes super-blocks [warp_id, warp_id+num_warps, ...]
//! - Intra-warp reduction via shfl.down.f32
//! - Cross-warp reduction via shared memory with bar.sync 0

use crate::kernels::quantize::{Kernel, Q6K_SUPER_BLOCK_BYTES, Q6K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// DP4A Q6_K GEMV kernel
///
/// Uses dp4a.u32.s32 integer dot products with pre-quantized Q8_1 activations.
/// Processes 4 Q6K values per dp4a instruction for ~4x instruction reduction.
pub struct Dp4aQ6KGemvKernel {
    /// K dimension (input dimension)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Number of warps per block
    pub num_warps: u32,
}

impl Dp4aQ6KGemvKernel {
    /// Create a new dp4a Q6K GEMV kernel with default warp count (3).
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, num_warps: 3 }
    }

    /// Create a new dp4a Q6K GEMV kernel with a custom warp count.
    #[must_use]
    pub fn with_warps(k: u32, n: u32, num_warps: u32) -> Self {
        debug_assert!(
            matches!(num_warps, 1 | 2 | 3 | 4 | 6 | 8),
            "num_warps should be in {{1,2,3,4,6,8}}, got {num_warps}"
        );
        Self { k, n, num_warps }
    }
}

impl Kernel for Dp4aQ6KGemvKernel {
    fn name(&self) -> &str {
        "dp4a_q6k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let smem_size = (num_warps * 4) as usize;

        PtxKernel::new("dp4a_q6k_gemv")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "q8_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .shared_memory(smem_size)
            .max_regs(255)
            .build(move |ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);
                // GH-174: Grid-stride loop — read gridDim.x for stride
                let grid_dim = ctx.special_reg(PtxReg::NctaIdX);

                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let q8_ptr = ctx.load_param_u64("q8_ptr");

                // ceil(k / 256)
                let k_rounded = ctx.add_u32(k_dim, Q6K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q6K_SUPER_BLOCK_SIZE);

                // Row stride in bytes (constant across rows)
                let sb_bytes_c = ctx.mov_u32_imm(Q6K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes_c);

                // GH-174: Grid-stride outer loop over rows
                // row_idx starts at block_id, increments by gridDim.x each iteration
                let row_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(row_idx, block_id);

                ctx.label("dp4a_q6k_row_loop");
                let row_oob = ctx.setp_ge_u32(row_idx, n_dim);
                ctx.branch_if(row_oob, "dp4a_q6k_exit");

                // Row base address
                let row_offset = ctx.mul_wide_u32_reg(row_idx, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // GH-175: Prefetch next row's ql data to L2 while computing current row.
                // Critical for LM head (n=151936): 593 rows per block, ~700 cycles per row
                // gives enough lead time to cover ~400 cycle DRAM latency.
                let next_row = ctx.add_u32_reg(row_idx, grid_dim);
                let next_offset = ctx.mul_wide_u32_reg(next_row, row_bytes);
                let next_base = ctx.add_u64(w_ptr, next_offset);
                ctx.prefetch_global_l2(next_base);

                let acc = ctx.mov_f32_imm(0.0);

                // === Per-lane constants (derived from lane_id) ===
                //
                // Thread layout per iteration (128 elements per half):
                //   32 lanes × 4 elements/lane = 128 elements
                //   sub_block_local = lane_id / 4    (0-7, which sub-block in the half)
                //   pos = (lane_id % 4) * 4           (0,4,8,12 within sub-block)
                //
                // Q6K addressing:
                //   group_of_32 = sub_block_local / 2   (0-3)
                //   group_is_odd = sub_block_local & 1
                //   pos_in_group = group_is_odd * 16 + pos
                //   ql_byte = 64*n_idx + 32*(group_of_32 & 1) + pos_in_group
                //   nibble_shift = (group_of_32 / 2) * 4   (0 or 4)
                //   qh_byte = 128 + 32*n_idx + pos_in_group
                //   qh_shift = 2 * group_of_32              (0,2,4,6)

                let one_c = ctx.mov_u32_imm(1);
                let two_c = ctx.mov_u32_imm(2);
                let four_c = ctx.mov_u32_imm(4);
                let sixteen_c = ctx.mov_u32_imm(16);
                let thirty_two_c = ctx.mov_u32_imm(32);

                // sub_block_local = lane_id >> 2
                let sub_block_local = ctx.shr_u32(lane_id, two_c);
                // pos = (lane_id & 3) << 2
                let three_c = ctx.mov_u32_imm(3);
                let lane_mod4 = ctx.and_u32(lane_id, three_c);
                let pos = ctx.shl_u32(lane_mod4, two_c);
                // group_of_32 = sub_block_local >> 1
                let group_of_32 = ctx.shr_u32(sub_block_local, one_c);
                // group_is_odd = sub_block_local & 1
                let group_is_odd = ctx.and_u32(sub_block_local, one_c);
                // pos_in_group = group_is_odd * 16 + pos
                let gio_x16 = ctx.mul_u32_reg(group_is_odd, sixteen_c);
                let pos_in_group = ctx.add_u32_reg(gio_x16, pos);
                // ql_base_in_half = 32 * (group_of_32 & 1) + pos_in_group
                let group_low_bit = ctx.and_u32(group_of_32, one_c);
                let ql_group_offset = ctx.mul_u32_reg(group_low_bit, thirty_two_c);
                let ql_base_in_half = ctx.add_u32_reg(ql_group_offset, pos_in_group);
                // nibble_shift = (group_of_32 >> 1) << 2   (0 or 4)
                let group_div2 = ctx.shr_u32(group_of_32, one_c);
                let nibble_shift = ctx.shl_u32(group_div2, two_c);
                // qh_shift = group_of_32 << 1              (0, 2, 4, or 6)
                let qh_shift = ctx.shl_u32(group_of_32, one_c);

                // Constants for dp4a
                let mask_0f = ctx.mov_u32_imm(0x0F0F_0F0F);
                let mask_03 = ctx.mov_u32_imm(0x0303_0303);
                let ones_packed = ctx.mov_u32_imm(0x0101_0101);
                let five_c = ctx.mov_u32_imm(5);

                // Binary tree scale selection bits (constant per lane)
                let bit0 = group_is_odd;
                let bit1_val = ctx.and_u32(group_of_32, one_c);
                let bit2_val = ctx.shr_u32(sub_block_local, two_c);
                let zero_c = ctx.mov_u32_imm(0);
                let p_bit0 = ctx.setp_ne_u32(bit0, zero_c);
                let p_bit1 = ctx.setp_ne_u32(bit1_val, zero_c);
                let p_bit2 = ctx.setp_ne_u32(bit2_val, zero_c);

                // === Super-block loop ===
                let sb_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(sb_idx, warp_id);
                let nw_reg = ctx.mov_u32_imm(num_warps);

                ctx.label("dp4a_q6k_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "dp4a_q6k_sb_end");

                // Super-block address
                let sb_off = ctx.mul_wide_u32(sb_idx, Q6K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // Load d (f16 at offset 208)
                let d_offset = ctx.mov_u64_imm(208);
                let d_addr = ctx.add_u64(sb_addr, d_offset);
                let d_f16 = ctx.ld_global_f16(d_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // === Scale loading: lanes 0-15 each load one i8 scale ===
                let scales_base_offset = ctx.mov_u64_imm(192);
                let scales_base = ctx.add_u64(sb_addr, scales_base_offset);
                let lane_mod_16 = ctx.rem_u32(lane_id, 16);
                let lane_offset_64 = ctx.cvt_u64_u32(lane_mod_16);
                let scale_addr = ctx.add_u64(scales_base, lane_offset_64);

                let my_scale_byte = ctx.mov_u32_imm(0);
                let is_low_lane = ctx.setp_lt_u32(lane_id, sixteen_c);
                ctx.branch_if_not(is_low_lane, "dp4a_q6k_skip_scale");
                let scale_u8 = ctx.ld_global_u8(scale_addr);
                let scale_u32 = ctx.cvt_u32_u8(scale_u8);
                ctx.mov_u32_reg(my_scale_byte, scale_u32);
                ctx.label("dp4a_q6k_skip_scale");

                // Broadcast all 16 scales via warp shuffle
                let mut scale_regs = Vec::with_capacity(16);
                for i in 0..16u32 {
                    scale_regs.push(ctx.shfl_idx_u32(my_scale_byte, i, 0xFFFF_FFFF));
                }

                // Convert i8 scales to signed f32 and precompute d*scale
                let seven_c = ctx.mov_u32_imm(7);
                let twofiftysix_f32 = ctx.mov_f32_imm(256.0);
                let mut ds = Vec::with_capacity(16);
                for &sr in &scale_regs {
                    let sign_bit = ctx.shr_u32(sr, seven_c);
                    let raw_f32 = ctx.cvt_f32_u32(sr);
                    let sign_f32 = ctx.cvt_f32_u32(sign_bit);
                    let correction = ctx.mul_f32(sign_f32, twofiftysix_f32);
                    let signed_f32 = ctx.sub_f32(raw_f32, correction);
                    ds.push(ctx.mul_f32(d, signed_f32));
                }

                // Q8 base for this super-block: q8_ptr + sb_idx * 8 * 36
                let eight_c = ctx.mov_u32_imm(8);
                let sb_q8_blocks = ctx.mul_u32_reg(sb_idx, eight_c);
                let thirty_six_c = ctx.mov_u32_imm(36);
                let q8_sb_offset = ctx.mul_wide_u32_reg(sb_q8_blocks, thirty_six_c);
                let q8_sb_base = ctx.add_u64(q8_ptr, q8_sb_offset);

                // === Process 2 halves: n_idx=0 (elements 0-127), n_idx=1 (128-255) ===
                for n_idx in 0..2u32 {
                    // ql byte offset = 64*n_idx + ql_base_in_half
                    let ql_full_offset =
                        if n_idx == 0 { ql_base_in_half } else { ctx.add_u32(ql_base_in_half, 64) };
                    let ql_off_64 = ctx.cvt_u64_u32(ql_full_offset);
                    let ql_addr = ctx.add_u64(sb_addr, ql_off_64);
                    // GH-129: Use unaligned load — Q6K super-blocks are 210 bytes
                    // (not 4-byte aligned), causing MISALIGNED_ADDRESS on sm_87.
                    let ql_int32 = ctx.ld_global_u32_unaligned(ql_addr);

                    // Extract nibbles: shift by nibble_shift (0 or 4), mask 0x0F
                    let ql_shifted = ctx.shr_u32(ql_int32, nibble_shift);
                    let ql_nibs = ctx.and_u32(ql_shifted, mask_0f);

                    // qh byte offset = 128 + 32*n_idx + pos_in_group
                    let qh_base_val = 128 + 32 * n_idx;
                    let qh_full_offset = ctx.add_u32(pos_in_group, qh_base_val);
                    let qh_off_64 = ctx.cvt_u64_u32(qh_full_offset);
                    let qh_addr = ctx.add_u64(sb_addr, qh_off_64);
                    // GH-129: Use unaligned load — same Q6K alignment issue.
                    let qh_int32 = ctx.ld_global_u32_unaligned(qh_addr);

                    // Extract 2-bit pairs: shift by qh_shift (0,2,4,6), mask 0x03
                    let qh_shifted = ctx.shr_u32(qh_int32, qh_shift);
                    let qh_2bits = ctx.and_u32(qh_shifted, mask_03);

                    // Combine: q6k_unsigned = ql_nibs | (qh_2bits << 4)
                    let qh_up = ctx.shl_u32(qh_2bits, four_c);
                    let combined = ctx.or_u32(ql_nibs, qh_up);

                    // Q8 block index within super-block = n_idx*4 + sub_block_local/2
                    let sbl_div2 = ctx.shr_u32(sub_block_local, one_c);
                    let q8_block_idx = ctx.add_u32(sbl_div2, n_idx * 4);

                    // Q8 qs address
                    let q8_block_off = ctx.mul_wide_u32_reg(q8_block_idx, thirty_six_c);
                    let q8_block_addr = ctx.add_u64(q8_sb_base, q8_block_off);
                    let pig_64 = ctx.cvt_u64_u32(pos_in_group);
                    let q8_qs_addr = ctx.add_u64(q8_block_addr, pig_64);
                    let q8_int32 = ctx.ld_global_u32(q8_qs_addr);

                    // dp4a: unsigned Q6K × signed Q8 dot product
                    let dot_acc = ctx.mov_u32_imm(0);
                    ctx.dp4a_u32_s32_inplace(dot_acc, combined, q8_int32);

                    // dp4a: sum of Q8 bytes (for −32 bias term)
                    let sum_acc = ctx.mov_u32_imm(0);
                    ctx.dp4a_u32_s32_inplace(sum_acc, ones_packed, q8_int32);

                    // int_result = dot - 32*sum = dot - (sum << 5)
                    let sum_x32 = ctx.shl_u32(sum_acc, five_c);
                    let int_result = ctx.sub_u32(dot_acc, sum_x32);

                    // Convert to f32
                    let result_f32 = ctx.cvt_f32_s32(int_result);

                    // Load Q8 d scale (f16 at offset 32 in Q8 block)
                    let q8_d_off = ctx.mov_u64_imm(32);
                    let q8_d_addr = ctx.add_u64(q8_block_addr, q8_d_off);
                    let q8_d_f16 = ctx.ld_global_f16(q8_d_addr);
                    let q8_d = ctx.cvt_f32_f16(q8_d_f16);

                    // Binary tree scale selection: ds[n_idx*8 + sub_block_local]
                    // sub_block_local = lane_id/4 ∈ {0..7}
                    // bit0 = sbl & 1, bit1 = (sbl>>1) & 1, bit2 = sbl >> 2
                    let base = n_idx as usize * 8;

                    // Level 1: select by bit0 (odd/even sub-block)
                    let t_01 = ctx.selp_f32(p_bit0, ds[base + 1], ds[base]);
                    let t_23 = ctx.selp_f32(p_bit0, ds[base + 3], ds[base + 2]);
                    let t_45 = ctx.selp_f32(p_bit0, ds[base + 5], ds[base + 4]);
                    let t_67 = ctx.selp_f32(p_bit0, ds[base + 7], ds[base + 6]);

                    // Level 2: select by bit1
                    let t_03 = ctx.selp_f32(p_bit1, t_23, t_01);
                    let t_47 = ctx.selp_f32(p_bit1, t_67, t_45);

                    // Level 3: select by bit2
                    let ds_selected = ctx.selp_f32(p_bit2, t_47, t_03);

                    // acc += ds_selected * q8_d * result_f32
                    let scale_product = ctx.mul_f32(ds_selected, q8_d);
                    ctx.fma_f32_inplace(acc, scale_product, result_f32);
                }

                // Stride by num_warps
                ctx.add_u32_reg_inplace(sb_idx, nw_reg);
                ctx.branch("dp4a_q6k_sb_loop");

                ctx.label("dp4a_q6k_sb_end");

                // === Phase 1: Intra-warp reduction via shuffle ===
                let t16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t16);
                let t8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t8);
                let t4_r = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t4_r);
                let t2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t2);
                let t1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t1);

                // === Phase 2: Cross-warp reduction via shared memory ===
                let is_l0 = ctx.setp_eq_u32(lane_id, zero_c);
                ctx.branch_if_not(is_l0, "dp4a_q6k_skip_sm");

                let wo = ctx.mul_u32_reg(warp_id, four_c);
                let sa = ctx.cvt_u64_u32(wo);
                ctx.st_shared_f32(sa, acc);

                ctx.label("dp4a_q6k_skip_sm");
                ctx.bar_sync(0);

                let is_t0 = ctx.setp_eq_u32(thread_id, zero_c);
                ctx.branch_if_not(is_t0, "dp4a_q6k_skip_store");

                let fs = ctx.mov_f32_imm(0.0);
                for w in 0..num_warps {
                    let wo = ctx.mov_u64_imm(u64::from(w * 4));
                    let pv = ctx.ld_shared_f32(wo);
                    ctx.add_f32_inplace(fs, pv);
                }

                // GH-174: Store to row_idx (not block_id) for grid-stride
                let yo = ctx.mul_wide_u32(row_idx, 4);
                let ya = ctx.add_u64(y_ptr, yo);
                ctx.st_global_f32(ya, fs);

                ctx.label("dp4a_q6k_skip_store");

                // GH-174: Advance to next row (grid-stride)
                ctx.add_u32_reg_inplace(row_idx, grid_dim);
                // Barrier before next row iteration to protect shared memory
                ctx.bar_sync(0);
                ctx.branch("dp4a_q6k_row_loop");

                ctx.label("dp4a_q6k_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dp4a_q6k_builds_qwen25() {
        let kernel = Dp4aQ6KGemvKernel::new(1536, 1536);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".visible .entry dp4a_q6k_gemv"));
        assert!(ptx.contains("dp4a.u32.s32"), "Must use dp4a instructions");
        assert!(ptx.contains("bar.sync"), "Must have barrier for cross-warp safety");
        // GH-175: Prefetch next row's data in grid-stride loop
        assert!(ptx.contains("prefetch.global.L2"), "Must prefetch next row data");
        // GH-131: bfi.b32 used for unaligned Q6K loads (replaces shl+or assembly)
        assert!(ptx.contains("bfi.b32"), "Must use bfi.b32 for unaligned byte packing");
    }

    #[test]
    fn test_dp4a_q6k_builds_lm_head() {
        // LM head: n=151936, k=1536 — the main optimization target
        let kernel = Dp4aQ6KGemvKernel::new(1536, 151936);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".visible .entry dp4a_q6k_gemv"));
    }

    #[test]
    fn test_dp4a_q6k_parameters() {
        let kernel = Dp4aQ6KGemvKernel::new(256, 64);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("y_ptr"));
        assert!(ptx.contains("w_ptr"));
        assert!(ptx.contains("q8_ptr"), "Must have Q8_1 activation pointer");
        assert!(ptx.contains("k_dim"));
        assert!(ptx.contains("n_dim"));
    }

    #[test]
    fn test_dp4a_q6k_shared_memory() {
        for warps in [1, 2, 3, 4, 6, 8] {
            let kernel = Dp4aQ6KGemvKernel::with_warps(256, 64, warps);
            let ptx_kernel = kernel.build_ptx();
            assert_eq!(
                ptx_kernel.shared_memory_bytes(),
                (warps * 4) as usize,
                "Shared memory must be {warps} warps × 4 bytes"
            );
        }
    }

    #[test]
    fn test_dp4a_q6k_warp_variants() {
        for warps in [1, 2, 3, 4, 6, 8] {
            let kernel = Dp4aQ6KGemvKernel::with_warps(1536, 1536, warps);
            let ptx = kernel.emit_ptx();
            assert!(ptx.contains(".visible .entry"), "Must produce valid PTX for {warps} warps");
        }
    }

    #[test]
    fn test_dp4a_q6k_name() {
        let k = Dp4aQ6KGemvKernel::new(1536, 1536);
        assert_eq!(k.name(), "dp4a_q6k_gemv");
    }
}
