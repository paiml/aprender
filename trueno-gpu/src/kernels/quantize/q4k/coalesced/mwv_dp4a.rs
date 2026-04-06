use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Multi-warp DP4A Q4_K GEMV kernel (PAR-082-V4)
///
/// Combines MWV multi-warp parallelism with DP4A integer dot products:
/// - **Activations**: Pre-quantized to Q8_1 format (36 bytes/block: 32 qs + f16 scale + f16 sum)
/// - **DP4A**: `dp4a.u32.s32` computes 4 multiply-adds of u4×s8 per instruction
/// - **Instruction reduction**: ~65 → ~20 instructions per inner loop iteration (3.3x)
/// - **Memory reduction**: 8 scattered f32 loads → 2 coalesced u32 Q8 loads
///
/// # Q4K × Q8_1 dot product
///
/// For each sub-block of 32 values:
///   `result = q8_d * (d * scale * dp4a_sum - dmin * min * q8_byte_sum)`
/// where `dp4a_sum = Σ(nibble_i × q8_byte_i)` and `q8_byte_sum = Σ(q8_byte_i)`.
///
/// Reference: NVIDIA PTX ISA §9.7.8.11 dp4a (Compute Capability ≥ 6.1)
/// Citation: Markidis et al., "NVIDIA Tensor Core Programmability, Performance &
///           Precision," IEEE IPDPSW 2018, DOI:10.1109/IPDPSW.2018.00091
pub struct MwvDp4aQ4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Number of warps per block (default: 3)
    pub num_warps: u32,
}

impl MwvDp4aQ4KGemvKernel {
    /// Create with 3 warps (96 threads), empirically optimal on RTX 4090
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, num_warps: 3 }
    }
}

impl Kernel for MwvDp4aQ4KGemvKernel {
    fn name(&self) -> &str {
        "mwv_dp4a_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let smem_size = (num_warps * 4) as usize;

        PtxKernel::new("mwv_dp4a_q4k_gemv")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "q8_ptr") // Q8_1 quantized activations (NOT f32)
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .shared_memory(smem_size)
            .max_regs(255)
            .build(move |ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);
                let grid_dim = ctx.special_reg(PtxReg::NctaIdX);

                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let q8_ptr = ctx.load_param_u64("q8_ptr");

                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                let sb_bytes_c = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes_c);

                // ===== HOISTED CONSTANTS (GH-174-V2) =====
                // All constants moved before the loop to reduce per-iteration instruction count.
                // PTX JIT may or may not hoist these; explicit placement guarantees it.
                let c_one = ctx.mov_u32_imm(1);
                let c_two = ctx.mov_u32_imm(2);
                let c_four = ctx.mov_u32_imm(4);
                let c_six = ctx.mov_u32_imm(6);
                let c_eight = ctx.mov_u32_imm(8);
                let c_sixteen = ctx.mov_u32_imm(16);
                let c_thirty_six = ctx.mov_u32_imm(36);
                let c_mask_0f = ctx.mov_u32_imm(0x0F0F_0F0F);
                let c_mask_3f = ctx.mov_u32_imm(0x3F3F_3F3F);
                let c_mask_03 = ctx.mov_u32_imm(0x0303_0303);
                let c_mask_ff = ctx.mov_u32_imm(0xFF);
                let c_ones_dp4a = ctx.mov_u32_imm(0x0101_0101);
                let c_two_64 = ctx.mov_u64_imm(2);
                let c_four_64 = ctx.mov_u64_imm(4);
                let c_eight_64 = ctx.mov_u64_imm(8);
                let c_sixteen_64 = ctx.mov_u64_imm(16);
                let c_thirty_two_64 = ctx.mov_u64_imm(32);
                let c_thirty_six_64 = ctx.mov_u64_imm(36);

                // ===== HOISTED PER-THREAD INVARIANTS =====
                // These depend on lane_id but are constant across all loop iterations.
                let c_three = ctx.mov_u32_imm(3);
                let ci = ctx.shr_u32(lane_id, c_three); // lane_id / 8
                let c_seven = ctx.mov_u32_imm(7);
                let lic = ctx.and_u32(lane_id, c_seven); // lane_id & 7
                let ci2 = ctx.shl_u32(ci, c_one); // ci * 2
                let tbo = ctx.mul_u32_reg(lane_id, c_four); // lane_id * 4
                let tbo64 = ctx.cvt_u64_u32(tbo);
                let lic_x4 = ctx.mul_u32_reg(lic, c_four);
                let lic_x4_64 = ctx.cvt_u64_u32(lic_x4);

                // Per-thread scale extraction invariants
                let ci_mod2 = ctx.and_u32(ci, c_one);
                let byte_shift = ctx.mul_u32_reg(ci_mod2, c_sixteen); // (ci%2)*16
                let byte_shift_hi = ctx.add_u32_reg(byte_shift, c_eight); // byte_shift+8
                let p_hi = ctx.setp_ge_u32(ci, c_two); // ci >= 2

                // Lane 0 predicate for scale loading
                let is_lane0 = ctx.setp_lt_u32(lane_id, c_one);

                // GH-174: Grid-stride outer loop over rows
                let row_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(row_idx, block_id);

                ctx.label("dp4a_row_loop");
                let row_oob = ctx.setp_ge_u32(row_idx, n_dim);
                ctx.branch_if(row_oob, "dp4a_exit");

                let row_offset = ctx.mul_wide_u32_reg(row_idx, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // GH-175: Prefetch next row
                let next_row = ctx.add_u32_reg(row_idx, grid_dim);
                let next_offset = ctx.mul_wide_u32_reg(next_row, row_bytes);
                let next_base = ctx.add_u64(w_ptr, next_offset);
                let pf_addr0 = ctx.add_u64(next_base, c_sixteen_64);
                ctx.prefetch_global_l2(pf_addr0);

                let acc = ctx.mov_f32_imm(0.0);

                let sb_idx_z = ctx.mov_u32_imm(0);
                let sb_idx = ctx.add_u32_reg(sb_idx_z, warp_id);
                let nw_reg = ctx.mov_u32_imm(num_warps);

                ctx.label("dp4a_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "dp4a_sb_end");

                let sb_off = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // Load d and dmin
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let dmin_addr = ctx.add_u64(sb_addr, c_two_64);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // Scale loading: lane 0 loads 3 x u32, broadcasts
                let scales_base = ctx.add_u64(sb_addr, c_four_64);

                let sc03r = ctx.mov_u32_imm(0);
                let sc47r = ctx.mov_u32_imm(0);
                let sc811r = ctx.mov_u32_imm(0);

                ctx.branch_if_not(is_lane0, "dp4a_skip_sc");
                ctx.ld_global_u32_into(sc03r, scales_base);
                let s4a = ctx.add_u64(scales_base, c_four_64);
                ctx.ld_global_u32_into(sc47r, s4a);
                let s8a = ctx.add_u64(scales_base, c_eight_64);
                ctx.ld_global_u32_into(sc811r, s8a);
                ctx.label("dp4a_skip_sc");

                let sc03 = ctx.shfl_idx_u32(sc03r, 0, 0xFFFF_FFFF);
                let sc47 = ctx.shfl_idx_u32(sc47r, 0, 0xFFFF_FFFF);
                let sc811 = ctx.shfl_idx_u32(sc811r, 0, 0xFFFF_FFFF);

                // GH-173: Parallel byte-masked scale extraction
                // Blocks 0-3: mask low 6 bits
                let sc_lo4 = ctx.and_u32(sc03, c_mask_3f);
                let mn_lo4 = ctx.and_u32(sc47, c_mask_3f);

                // Blocks 4-7: combine low 4 + high 2 bits
                let sc_hi_low = ctx.and_u32(sc811, c_mask_0f);
                let t = ctx.shr_u32(sc03, c_six);
                let t = ctx.and_u32(t, c_mask_03);
                let sc_hi_top = ctx.shl_u32(t, c_four);
                let sc_hi4 = ctx.or_u32(sc_hi_low, sc_hi_top);

                let mn_hi_raw = ctx.shr_u32(sc811, c_four);
                let mn_hi_low = ctx.and_u32(mn_hi_raw, c_mask_0f);
                let t = ctx.shr_u32(sc47, c_six);
                let t = ctx.and_u32(t, c_mask_03);
                let mn_hi_top = ctx.shl_u32(t, c_four);
                let mn_hi4 = ctx.or_u32(mn_hi_low, mn_hi_top);

                // === DP4A HOT PATH ===

                // COALESCED u32 LOAD of Q4K weights (uses hoisted tbo64)
                let qs_base = ctx.add_u64(sb_addr, c_sixteen_64);
                let qa = ctx.add_u64(qs_base, tbo64);
                let packed = ctx.ld_global_u32(qa);

                // Extract nibble packs (uses hoisted c_mask_0f, c_four)
                let low_nibs = ctx.and_u32(packed, c_mask_0f);
                let high_shifted = ctx.shr_u32(packed, c_four);
                let high_nibs = ctx.and_u32(high_shifted, c_mask_0f);

                // Q8 block indices (uses hoisted ci2, c_eight, c_thirty_six)
                let sb8 = ctx.mul_u32_reg(sb_idx, c_eight);
                let blk_low = ctx.add_u32_reg(sb8, ci2);

                let q8_off_low = ctx.mul_wide_u32_reg(blk_low, c_thirty_six);
                let q8_off_high = ctx.add_u64(q8_off_low, c_thirty_six_64);

                // Load Q8 packed bytes (uses hoisted lic_x4_64)
                let q8_base_low = ctx.add_u64(q8_ptr, q8_off_low);
                let q8_addr_low = ctx.add_u64(q8_base_low, lic_x4_64);
                let q8_low = ctx.ld_global_u32(q8_addr_low);

                let q8_base_high = ctx.add_u64(q8_ptr, q8_off_high);
                let q8_addr_high = ctx.add_u64(q8_base_high, lic_x4_64);
                let q8_high = ctx.ld_global_u32(q8_addr_high);

                // DP4A: dot product of nibbles × q8 bytes
                let dot_low = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(dot_low, low_nibs, q8_low);
                let dot_high = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(dot_high, high_nibs, q8_high);

                // DP4A: sum of Q8 bytes (for min term, uses hoisted c_ones_dp4a)
                let sum_low = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(sum_low, c_ones_dp4a, q8_low);
                let sum_high = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(sum_high, c_ones_dp4a, q8_high);

                // Load Q8 scales (uses hoisted c_thirty_two_64)
                let q8_d_addr_low = ctx.add_u64(q8_base_low, c_thirty_two_64);
                let q8_d_low_f16 = ctx.ld_global_f16(q8_d_addr_low);
                let q8_d_low = ctx.cvt_f32_f16(q8_d_low_f16);

                let q8_d_addr_high = ctx.add_u64(q8_base_high, c_thirty_two_64);
                let q8_d_high_f16 = ctx.ld_global_f16(q8_d_addr_high);
                let q8_d_high = ctx.cvt_f32_f16(q8_d_high_f16);

                // Per-thread byte extraction (uses hoisted p_hi, byte_shift, byte_shift_hi, c_mask_ff)
                let sc_src = ctx.selp_u32(p_hi, sc_hi4, sc_lo4);
                let mn_src = ctx.selp_u32(p_hi, mn_hi4, mn_lo4);

                let t = ctx.shr_u32(sc_src, byte_shift);
                let sl = ctx.and_u32(t, c_mask_ff);
                let t = ctx.shr_u32(mn_src, byte_shift);
                let ml_u = ctx.and_u32(t, c_mask_ff);

                let t = ctx.shr_u32(sc_src, byte_shift_hi);
                let sh = ctx.and_u32(t, c_mask_ff);
                let t = ctx.shr_u32(mn_src, byte_shift_hi);
                let mh_u = ctx.and_u32(t, c_mask_ff);

                // Convert scales to f32 and multiply by d/dmin
                let sl_f = ctx.cvt_f32_u32(sl);
                let dl = ctx.mul_f32(d, sl_f);
                let ml_f = ctx.cvt_f32_u32(ml_u);
                let ml = ctx.mul_f32(dmin, ml_f);
                let sh_f = ctx.cvt_f32_u32(sh);
                let dh = ctx.mul_f32(d, sh_f);
                let mh_f = ctx.cvt_f32_u32(mh_u);
                let mh = ctx.mul_f32(dmin, mh_f);

                // Convert DP4A results to f32
                let dot_low_f = ctx.cvt_f32_s32(dot_low);
                let dot_high_f = ctx.cvt_f32_s32(dot_high);
                let sum_low_f = ctx.cvt_f32_s32(sum_low);
                let sum_high_f = ctx.cvt_f32_s32(sum_high);

                // LOW contribution: q8_d * (d*scale * dot - dmin*min * sum)
                let t1 = ctx.mul_f32(dl, dot_low_f);
                let t2 = ctx.mul_f32(ml, sum_low_f);
                let t3 = ctx.sub_f32(t1, t2);
                let t4 = ctx.mul_f32(q8_d_low, t3);
                ctx.add_f32_inplace(acc, t4);

                // HIGH contribution: q8_d * (d*scale * dot - dmin*min * sum)
                let t1 = ctx.mul_f32(dh, dot_high_f);
                let t2 = ctx.mul_f32(mh, sum_high_f);
                let t3 = ctx.sub_f32(t1, t2);
                let t4 = ctx.mul_f32(q8_d_high, t3);
                ctx.add_f32_inplace(acc, t4);

                // Stride by num_warps
                ctx.add_u32_reg_inplace(sb_idx, nw_reg);
                ctx.branch("dp4a_sb_loop");

                ctx.label("dp4a_sb_end");

                // Phase 1: Intra-warp reduction
                let t16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t16);
                let t8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t8);
                let t4 = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t4);
                let t2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t2);
                let t1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t1);

                // Phase 2: Cross-warp reduction via shared memory
                let z = ctx.mov_u32_imm(0);
                let is_l0 = ctx.setp_eq_u32(lane_id, z);
                ctx.branch_if_not(is_l0, "dp4a_skip_sm");

                let wo = ctx.mul_u32_reg(warp_id, c_four);
                let sa = ctx.cvt_u64_u32(wo);
                ctx.st_shared_f32(sa, acc);

                ctx.label("dp4a_skip_sm");
                ctx.bar_sync(0);

                let is_t0 = ctx.setp_eq_u32(thread_id, z);
                ctx.branch_if_not(is_t0, "dp4a_skip_store");

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

                ctx.label("dp4a_skip_store");

                // GH-174: Advance to next row (grid-stride)
                ctx.add_u32_reg_inplace(row_idx, grid_dim);
                ctx.bar_sync(0);
                ctx.branch("dp4a_row_loop");

                ctx.label("dp4a_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_emits() {
        let k = MwvDp4aQ4KGemvKernel::new(1536, 256);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("mwv_dp4a_q4k_gemv"));
        assert!(ptx.contains("dp4a.u32.s32"));
    }

    #[test]
    fn dump_ptx() {
        let k = MwvDp4aQ4KGemvKernel::new(1536, 256);
        let ptx = k.emit_ptx();
        eprintln!("{ptx}");
    }
}
