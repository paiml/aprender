use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory, PtxSync};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Fused Gate+Up+SwiGLU HW DP4A Q4_K GEMV kernel (PMAT-034)
///
/// Computes `result[i] = silu(dot(W_gate[i], x)) * dot(W_up[i], x)` in a single
/// kernel pass, eliminating:
/// - 1 kernel launch (gate + up → fused)
/// - 1 SwiGLU kernel launch
/// - 4 global memory passes on intermediate buffers (2 writes + 2 reads)
///
/// Based on `HalfWarpDp4aQ4KGemvKernel` (GH-176) with dual accumulators.
///
/// # Key optimization: shared Q8 data
///
/// Both gate and up projections use the same Q8-quantized input activation.
/// The Q8 byte sums (for dmin offset) are computed ONCE and reused for both,
/// saving 4 DP4A instructions per super-block vs running two separate kernels.
///
/// # Instruction budget
///
/// - Separate kernels: 2 × 108 = 216 insn/SB (+ SwiGLU kernel + 4 global mem passes)
/// - Fused kernel: ~193 insn/SB (Q8 loads + sums shared, SwiGLU in-register)
/// - Savings: ~11% fewer instructions + eliminates SwiGLU kernel + 4 intermediate buffer passes
///
/// # Contracts
///
/// - **C1**: Same half-warp thread mapping as HW DP4A (16 threads/SB)
/// - **C2**: Same value coverage (16 threads × 16 values = 256)
/// - **C3**: Inner loop ≤ 200 insn for 16 values of BOTH gate and up (vs 2×108=216 separate)
/// - **C4**: Same half-warp reduction, applied to both accumulators
/// - **C5**: SwiGLU computed in-register by thread 0: silu(gate) * up
pub struct FusedGateUpSwigluHwDp4aQ4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (intermediate size, output dimension)
    pub n: u32,
    /// Number of warps per block (default: 3, giving 6 half-warps).
    /// PMAT-089: 4 warps FALSIFIED (register pressure, -2% decode). 3 is optimal.
    pub num_warps: u32,
}

impl FusedGateUpSwigluHwDp4aQ4KGemvKernel {
    /// Create a new fused gate+up+SwiGLU HW DP4A Q4K GEMV kernel with default 3 warps.
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, num_warps: 3 }
    }
}

impl Kernel for FusedGateUpSwigluHwDp4aQ4KGemvKernel {
    fn name(&self) -> &str {
        "fused_gate_up_swiglu_hw_dp4a_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let num_half_warps = num_warps * 2;
        // Shared memory: 2 accumulators (gate + up) per half-warp
        let smem_size = (num_half_warps * 2 * 4) as usize;

        PtxKernel::new("fused_gate_up_swiglu_hw_dp4a_q4k_gemv")
            .param(PtxType::U64, "y_ptr") // Output: SwiGLU result (N)
            .param(PtxType::U64, "wg_ptr") // Q4K gate weights
            .param(PtxType::U64, "wu_ptr") // Q4K up weights
            .param(PtxType::U64, "q8_ptr") // Q8-quantized input activation
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .shared_memory(smem_size)
            .max_regs(255)
            .build(move |ctx| {
                // ===== Thread identity =====
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);
                let grid_dim = ctx.special_reg(PtxReg::NctaIdX);

                // ===== Parameters =====
                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let wg_ptr = ctx.load_param_u64("wg_ptr");
                let wu_ptr = ctx.load_param_u64("wu_ptr");
                let q8_ptr = ctx.load_param_u64("q8_ptr");

                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_sb = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);
                let sb_bytes_reg = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_sb, sb_bytes_reg);

                // ===== C1: Half-warp thread mapping =====
                let half_lane = ctx.and_u32_imm(lane_id, 15);
                let half_warp_in_warp = ctx.shr_u32_imm(lane_id, 4);
                let warp_x2 = ctx.shl_u32_imm(warp_id, 1);
                let half_warp_id = ctx.add_u32_reg(warp_x2, half_warp_in_warp);
                let num_hw = ctx.mov_u32_imm(num_half_warps);

                // ===== C2: Per-thread data mapping =====
                let bq8_group = ctx.shr_u32_imm(half_lane, 2);
                let lane_in_group = ctx.and_u32_imm(half_lane, 3);
                let bq8_offset = ctx.shl_u32_imm(bq8_group, 1);

                // Q4K qs offset: 16 (header) + 16 * bq8_offset + 4 * lane_in_group
                let t1 = ctx.shl_u32_imm(bq8_offset, 4);
                let t2 = ctx.shl_u32_imm(lane_in_group, 2);
                let q4_local = ctx.add_u32_reg(t1, t2);
                let q4_off = ctx.add_u32(q4_local, 16);
                let q4_off_64 = ctx.cvt_u64_u32(q4_off);

                // Q8 per-thread offsets
                let c_36_u32 = ctx.mov_u32_imm(36);
                let bq8_bytes = ctx.mul_u32_reg(bq8_offset, c_36_u32);
                let bq8_bytes_64 = ctx.cvt_u64_u32(bq8_bytes);
                let lig_x4 = ctx.shl_u32_imm(lane_in_group, 2);
                let lig_x4_64 = ctx.cvt_u64_u32(lig_x4);

                // Hoisted 64-bit constants
                let c_2_64 = ctx.mov_u64_imm(2);
                let c_4_64 = ctx.mov_u64_imm(4);
                let c_8_64 = ctx.mov_u64_imm(8);
                let c_16_64 = ctx.mov_u64_imm(16);
                let c_32_64 = ctx.mov_u64_imm(32);
                let c_36_64 = ctx.mov_u64_imm(36);
                let c_288 = ctx.mov_u32_imm(288);

                // Scale extraction invariants
                let ci_mod2 = ctx.and_u32_imm(bq8_group, 1);
                let c_16_u32 = ctx.mov_u32_imm(16);
                let byte_shift = ctx.mul_u32_reg(ci_mod2, c_16_u32);
                let c_8_u32 = ctx.mov_u32_imm(8);
                let byte_shift_hi = ctx.add_u32_reg(byte_shift, c_8_u32);
                let c_2_u32 = ctx.mov_u32_imm(2);
                let p_hi = ctx.setp_ge_u32(bq8_group, c_2_u32);

                // DP4A constant
                let c_ones = ctx.mov_u32_imm(0x0101_0101);

                // PMAT-029: Hoist repeated bitmask constants before inner loop
                let c_mask_6bit = ctx.mov_u32_imm(0x3F3F_3F3F);
                let c_mask_4bit = ctx.mov_u32_imm(0x0F0F_0F0F);
                let c_mask_2bit = ctx.mov_u32_imm(0x0303_0303);

                // ===== Grid-stride row loop =====
                let row_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(row_idx, block_id);

                ctx.label("fgs_row_loop");
                let row_oob = ctx.setp_ge_u32(row_idx, n_dim);
                ctx.branch_if(row_oob, "fgs_exit");

                // Row bases for gate and up weights
                let row_off = ctx.mul_wide_u32_reg(row_idx, row_bytes);
                let wg_row_base = ctx.add_u64(wg_ptr, row_off);
                let wu_row_base = ctx.add_u64(wu_ptr, row_off);

                // Dual accumulators
                let acc_gate = ctx.mov_f32_imm(0.0);
                let acc_up = ctx.mov_f32_imm(0.0);

                // SB loop: each half-warp processes 1 SB, stride by num_half_warps
                let sb_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(sb_idx, half_warp_id);

                ctx.label("fgs_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_sb);
                ctx.branch_if(sb_done, "fgs_sb_end");

                // Super-block offset
                let sb_off = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let wg_sb_addr = ctx.add_u64(wg_row_base, sb_off);
                let wu_sb_addr = ctx.add_u64(wu_row_base, sb_off);

                // ===== Load d, dmin for GATE =====
                let dg_f16 = ctx.ld_global_f16(wg_sb_addr);
                let dg = ctx.cvt_f32_f16(dg_f16);
                let dming_addr = ctx.add_u64(wg_sb_addr, c_2_64);
                let dming_f16 = ctx.ld_global_f16(dming_addr);
                let dming = ctx.cvt_f32_f16(dming_f16);
                let neg_dming = ctx.neg_f32(dming);

                // ===== Load d, dmin for UP =====
                let du_f16 = ctx.ld_global_f16(wu_sb_addr);
                let du = ctx.cvt_f32_f16(du_f16);
                let dminu_addr = ctx.add_u64(wu_sb_addr, c_2_64);
                let dminu_f16 = ctx.ld_global_f16(dminu_addr);
                let dminu = ctx.cvt_f32_f16(dminu_f16);
                let neg_dminu = ctx.neg_f32(dminu);

                // ===== Scale loading for GATE (all threads, L1 coalesced) =====
                let scg_base = ctx.add_u64(wg_sb_addr, c_4_64);
                let scg03 = ctx.ld_global_u32(scg_base);
                let scg47_addr = ctx.add_u64(scg_base, c_4_64);
                let scg47 = ctx.ld_global_u32(scg47_addr);
                let scg811_addr = ctx.add_u64(scg_base, c_8_64);
                let scg811 = ctx.ld_global_u32(scg811_addr);

                // ===== Scale loading for UP =====
                let scu_base = ctx.add_u64(wu_sb_addr, c_4_64);
                let scu03 = ctx.ld_global_u32(scu_base);
                let scu47_addr = ctx.add_u64(scu_base, c_4_64);
                let scu47 = ctx.ld_global_u32(scu47_addr);
                let scu811_addr = ctx.add_u64(scu_base, c_8_64);
                let scu811 = ctx.ld_global_u32(scu811_addr);

                // ===== GH-173: Parallel byte-masked scale extraction — GATE =====
                // PMAT-029: Use hoisted constant registers
                let scg_lo4 = ctx.and_u32(scg03, c_mask_6bit);
                let mng_lo4 = ctx.and_u32(scg47, c_mask_6bit);
                let scg_hi_low = ctx.and_u32(scg811, c_mask_4bit);
                let t = ctx.shr_u32_imm(scg03, 6);
                let t = ctx.and_u32(t, c_mask_2bit);
                let scg_hi_top = ctx.shl_u32_imm(t, 4);
                let scg_hi4 = ctx.or_u32(scg_hi_low, scg_hi_top);
                let mng_hi_raw = ctx.shr_u32_imm(scg47, 6);
                let mng_hi_low = ctx.and_u32(mng_hi_raw, c_mask_4bit);
                let t = ctx.shr_u32_imm(scg47, 6);
                let t = ctx.and_u32(t, c_mask_2bit);
                let mng_hi_top = ctx.shl_u32_imm(t, 4);
                let mng_hi4 = ctx.or_u32(mng_hi_low, mng_hi_top);

                let scg_src = ctx.selp_u32(p_hi, scg_hi4, scg_lo4);
                let mng_src = ctx.selp_u32(p_hi, mng_hi4, mng_lo4);
                // PMAT-039: BFE replaces shr+and (4 insn saved per SB)
                let scg0 = ctx.bfe_u32_reg(scg_src, byte_shift, 8);
                let scg1 = ctx.bfe_u32_reg(scg_src, byte_shift_hi, 8);
                let mng0 = ctx.bfe_u32_reg(mng_src, byte_shift, 8);
                let mng1 = ctx.bfe_u32_reg(mng_src, byte_shift_hi, 8);

                // ===== Scale extraction — UP =====
                let scu_lo4 = ctx.and_u32(scu03, c_mask_6bit);
                let mnu_lo4 = ctx.and_u32(scu47, c_mask_6bit);
                let scu_hi_low = ctx.and_u32(scu811, c_mask_4bit);
                let t = ctx.shr_u32_imm(scu03, 6);
                let t = ctx.and_u32(t, c_mask_2bit);
                let scu_hi_top = ctx.shl_u32_imm(t, 4);
                let scu_hi4 = ctx.or_u32(scu_hi_low, scu_hi_top);
                let mnu_hi_raw = ctx.shr_u32_imm(scu47, 6);
                let mnu_hi_low = ctx.and_u32(mnu_hi_raw, c_mask_4bit);
                let t = ctx.shr_u32_imm(scu47, 6);
                let t = ctx.and_u32(t, c_mask_2bit);
                let mnu_hi_top = ctx.shl_u32_imm(t, 4);
                let mnu_hi4 = ctx.or_u32(mnu_hi_low, mnu_hi_top);

                let scu_src = ctx.selp_u32(p_hi, scu_hi4, scu_lo4);
                let mnu_src = ctx.selp_u32(p_hi, mnu_hi4, mnu_lo4);
                // PMAT-039: BFE replaces shr+and (4 insn saved per SB)
                let scu0 = ctx.bfe_u32_reg(scu_src, byte_shift, 8);
                let scu1 = ctx.bfe_u32_reg(scu_src, byte_shift_hi, 8);
                let mnu0 = ctx.bfe_u32_reg(mnu_src, byte_shift, 8);
                let mnu1 = ctx.bfe_u32_reg(mnu_src, byte_shift_hi, 8);

                // ===== Load Q4K data: GATE =====
                let q4g_addr = ctx.add_u64(wg_sb_addr, q4_off_64);
                let vg0 = ctx.ld_global_u32(q4g_addr);
                let vg1_addr = ctx.add_u64(q4g_addr, c_16_64);
                let vg1 = ctx.ld_global_u32(vg1_addr);

                // ===== Load Q4K data: UP =====
                let q4u_addr = ctx.add_u64(wu_sb_addr, q4_off_64);
                let vu0 = ctx.ld_global_u32(q4u_addr);
                let vu1_addr = ctx.add_u64(q4u_addr, c_16_64);
                let vu1 = ctx.ld_global_u32(vu1_addr);

                // ===== Q8 block base (SHARED between gate and up) =====
                let q8_sb_off = ctx.mul_wide_u32_reg(sb_idx, c_288);
                let q8_sb_base = ctx.add_u64(q8_ptr, q8_sb_off);
                let q8_blk = ctx.add_u64(q8_sb_base, bq8_bytes_64);
                let q8_data = ctx.add_u64(q8_blk, lig_x4_64);

                // ===== QR=0: Low nibbles =====
                let vg0_lo = ctx.and_u32(vg0, c_mask_4bit);
                let vg1_lo = ctx.and_u32(vg1, c_mask_4bit);
                let vu0_lo = ctx.and_u32(vu0, c_mask_4bit);
                let vu1_lo = ctx.and_u32(vu1, c_mask_4bit);

                let u0_lo = ctx.ld_global_u32(q8_data);
                let u1_lo_addr = ctx.add_u64(q8_data, c_16_64);
                let u1_lo = ctx.ld_global_u32(u1_lo_addr);

                // Gate DP4A: dot_gate = vg0_lo . u0 + vg1_lo . u1
                let dotg0 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(dotg0, vg0_lo, u0_lo);
                ctx.dp4a_u32_s32_inplace(dotg0, vg1_lo, u1_lo);

                // Up DP4A: dot_up = vu0_lo . u0 + vu1_lo . u1
                let dotu0 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(dotu0, vu0_lo, u0_lo);
                ctx.dp4a_u32_s32_inplace(dotu0, vu1_lo, u1_lo);

                // Q8 byte sum (SHARED): sum = ones . u0 + ones . u1
                let sum0 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(sum0, c_ones, u0_lo);
                ctx.dp4a_u32_s32_inplace(sum0, c_ones, u1_lo);

                // Q8 scale
                let q8_d0_addr = ctx.add_u64(q8_blk, c_32_64);
                let q8_d0_f16 = ctx.ld_global_f16(q8_d0_addr);
                let q8_d0 = ctx.cvt_f32_f16(q8_d0_f16);

                // Gate accumulate: acc_gate += q8_d * (dg*scg*dot - dming*mng*sum)
                let sdotg0 = ctx.mul_lo_s32(scg0, dotg0);
                let msumg0 = ctx.mul_lo_s32(mng0, sum0);
                let sdotg0_f = ctx.cvt_f32_s32(sdotg0);
                let msumg0_f = ctx.cvt_f32_s32(msumg0);
                let tg1 = ctx.mul_f32(dg, sdotg0_f);
                let tg3 = ctx.fma_f32(neg_dming, msumg0_f, tg1);
                let q8_d0_tg3 = ctx.mul_f32(q8_d0, tg3);
                ctx.add_f32_inplace(acc_gate, q8_d0_tg3);

                // Up accumulate: acc_up += q8_d * (du*scu*dot - dminu*mnu*sum)
                let sdotu0 = ctx.mul_lo_s32(scu0, dotu0);
                let msumu0 = ctx.mul_lo_s32(mnu0, sum0);
                let sdotu0_f = ctx.cvt_f32_s32(sdotu0);
                let msumu0_f = ctx.cvt_f32_s32(msumu0);
                let tu1 = ctx.mul_f32(du, sdotu0_f);
                let tu3 = ctx.fma_f32(neg_dminu, msumu0_f, tu1);
                let q8_d0_tu3 = ctx.mul_f32(q8_d0, tu3);
                ctx.add_f32_inplace(acc_up, q8_d0_tu3);

                // ===== QR=1: High nibbles =====
                let vg0_hi = ctx.shr_u32_imm(vg0, 4);
                let vg0_hi = ctx.and_u32(vg0_hi, c_mask_4bit);
                let vg1_hi = ctx.shr_u32_imm(vg1, 4);
                let vg1_hi = ctx.and_u32(vg1_hi, c_mask_4bit);
                let vu0_hi = ctx.shr_u32_imm(vu0, 4);
                let vu0_hi = ctx.and_u32(vu0_hi, c_mask_4bit);
                let vu1_hi = ctx.shr_u32_imm(vu1, 4);
                let vu1_hi = ctx.and_u32(vu1_hi, c_mask_4bit);

                // Q8 block +1 (36 bytes later)
                let q8_blk_hi = ctx.add_u64(q8_blk, c_36_64);
                let q8_data_hi = ctx.add_u64(q8_blk_hi, lig_x4_64);

                let u0_hi = ctx.ld_global_u32(q8_data_hi);
                let u1_hi_addr = ctx.add_u64(q8_data_hi, c_16_64);
                let u1_hi = ctx.ld_global_u32(u1_hi_addr);

                // Gate DP4A high nibbles
                let dotg1 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(dotg1, vg0_hi, u0_hi);
                ctx.dp4a_u32_s32_inplace(dotg1, vg1_hi, u1_hi);

                // Up DP4A high nibbles
                let dotu1 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(dotu1, vu0_hi, u0_hi);
                ctx.dp4a_u32_s32_inplace(dotu1, vu1_hi, u1_hi);

                // Q8 byte sum high (SHARED)
                let sum1 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(sum1, c_ones, u0_hi);
                ctx.dp4a_u32_s32_inplace(sum1, c_ones, u1_hi);

                let q8_d1_addr = ctx.add_u64(q8_blk_hi, c_32_64);
                let q8_d1_f16 = ctx.ld_global_f16(q8_d1_addr);
                let q8_d1 = ctx.cvt_f32_f16(q8_d1_f16);

                // Gate accumulate QR=1
                let sdotg1 = ctx.mul_lo_s32(scg1, dotg1);
                let msumg1 = ctx.mul_lo_s32(mng1, sum1);
                let sdotg1_f = ctx.cvt_f32_s32(sdotg1);
                let msumg1_f = ctx.cvt_f32_s32(msumg1);
                let tg1b = ctx.mul_f32(dg, sdotg1_f);
                let tg3b = ctx.fma_f32(neg_dming, msumg1_f, tg1b);
                let q8_d1_tg3b = ctx.mul_f32(q8_d1, tg3b);
                ctx.add_f32_inplace(acc_gate, q8_d1_tg3b);

                // Up accumulate QR=1
                let sdotu1 = ctx.mul_lo_s32(scu1, dotu1);
                let msumu1 = ctx.mul_lo_s32(mnu1, sum1);
                let sdotu1_f = ctx.cvt_f32_s32(sdotu1);
                let msumu1_f = ctx.cvt_f32_s32(msumu1);
                let tu1b = ctx.mul_f32(du, sdotu1_f);
                let tu3b = ctx.fma_f32(neg_dminu, msumu1_f, tu1b);
                let q8_d1_tu3b = ctx.mul_f32(q8_d1, tu3b);
                ctx.add_f32_inplace(acc_up, q8_d1_tu3b);

                // Next SB
                ctx.add_u32_reg_inplace(sb_idx, num_hw);
                ctx.branch("fgs_sb_loop");

                ctx.label("fgs_sb_end");

                // ===== C4: Half-warp reduction for BOTH accumulators =====
                // Gate reduction
                let t = ctx.shfl_down_f32(acc_gate, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_gate, t);
                let t = ctx.shfl_down_f32(acc_gate, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_gate, t);
                let t = ctx.shfl_down_f32(acc_gate, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_gate, t);
                let t = ctx.shfl_down_f32(acc_gate, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_gate, t);

                // Up reduction
                let t = ctx.shfl_down_f32(acc_up, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_up, t);
                let t = ctx.shfl_down_f32(acc_up, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_up, t);
                let t = ctx.shfl_down_f32(acc_up, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_up, t);
                let t = ctx.shfl_down_f32(acc_up, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_up, t);

                // Half-warp lane 0 stores both partials to shared memory
                let z = ctx.mov_u32_imm(0);
                let is_hl0 = ctx.setp_eq_u32(half_lane, z);
                ctx.branch_if_not(is_hl0, "fgs_skip_sm");

                // Layout: [gate_0, gate_1, ..., gate_(nhw-1), up_0, up_1, ..., up_(nhw-1)]
                let sm_gate_off = ctx.shl_u32_imm(half_warp_id, 2);
                let sm_gate_addr = ctx.cvt_u64_u32(sm_gate_off);
                ctx.st_shared_f32(sm_gate_addr, acc_gate);

                let nhw_bytes = ctx.mov_u32_imm(num_half_warps * 4);
                let sm_up_off = ctx.add_u32_reg(sm_gate_off, nhw_bytes);
                let sm_up_addr = ctx.cvt_u64_u32(sm_up_off);
                ctx.st_shared_f32(sm_up_addr, acc_up);

                ctx.label("fgs_skip_sm");
                ctx.bar_sync(0);

                // ===== PMAT-089: Parallel warp-0 reduction + C5: SwiGLU =====
                let is_warp0 = ctx.setp_eq_u32(warp_id, z);
                ctx.branch_if_not(is_warp0, "fgs_skip_store");

                let in_range = ctx.setp_lt_u32_imm(lane_id, num_half_warps);
                let zero_f = ctx.mov_f32_imm(0.0);

                // Reduce gate partials: thread i loads smem[i] if i < nhw
                let gate_sm_off = ctx.shl_u32_imm(lane_id, 2);
                let gate_sm_addr = ctx.cvt_u64_u32(gate_sm_off);
                let gate_loaded = ctx.ld_shared_f32(gate_sm_addr);
                let gate_partial = ctx.selp_f32(in_range, gate_loaded, zero_f);

                let t = ctx.shfl_down_f32(gate_partial, 4, 0xFFFF_FFFF);
                let gate_partial = ctx.add_f32(gate_partial, t);
                let t = ctx.shfl_down_f32(gate_partial, 2, 0xFFFF_FFFF);
                let gate_partial = ctx.add_f32(gate_partial, t);
                let t = ctx.shfl_down_f32(gate_partial, 1, 0xFFFF_FFFF);
                let gate_sum = ctx.add_f32(gate_partial, t);

                // Reduce up partials: thread i loads smem[nhw + i] if i < nhw
                let nhw_bytes = ctx.mov_u32_imm(num_half_warps * 4);
                let up_sm_off = ctx.add_u32_reg(gate_sm_off, nhw_bytes);
                let up_sm_addr = ctx.cvt_u64_u32(up_sm_off);
                let up_loaded = ctx.ld_shared_f32(up_sm_addr);
                let up_partial = ctx.selp_f32(in_range, up_loaded, zero_f);

                let t = ctx.shfl_down_f32(up_partial, 4, 0xFFFF_FFFF);
                let up_partial = ctx.add_f32(up_partial, t);
                let t = ctx.shfl_down_f32(up_partial, 2, 0xFFFF_FFFF);
                let up_partial = ctx.add_f32(up_partial, t);
                let t = ctx.shfl_down_f32(up_partial, 1, 0xFFFF_FFFF);
                let up_sum = ctx.add_f32(up_partial, t);

                // Thread 0: SwiGLU + store
                let is_t0 = ctx.setp_eq_u32(lane_id, z);
                ctx.branch_if_not(is_t0, "fgs_skip_store");

                // SwiGLU: result = silu(gate) * up = gate * sigmoid(gate) * up
                // sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + 2^(-x * log2(e)))
                let neg_gate = ctx.neg_f32(gate_sum);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let neg_gate_log2e = ctx.mul_f32(neg_gate, log2e);
                let exp_neg = ctx.ex2_f32(neg_gate_log2e);
                let one = ctx.mov_f32_imm(1.0);
                let one_plus_exp = ctx.add_f32(one, exp_neg);
                let sigmoid = ctx.rcp_f32(one_plus_exp);
                let silu = ctx.mul_f32(gate_sum, sigmoid);
                let result = ctx.mul_f32(silu, up_sum);

                let y_off = ctx.mul_wide_u32(row_idx, 4);
                let y_addr = ctx.add_u64(y_ptr, y_off);
                ctx.st_global_f32(y_addr, result);

                ctx.label("fgs_skip_store");

                // Next row (grid-stride)
                ctx.add_u32_reg_inplace(row_idx, grid_dim);
                ctx.bar_sync(0);
                ctx.branch("fgs_row_loop");

                ctx.label("fgs_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_emits_valid() {
        let k = FusedGateUpSwigluHwDp4aQ4KGemvKernel::new(1536, 4096);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("fused_gate_up_swiglu_hw_dp4a_q4k_gemv"), "kernel name");
        assert!(ptx.contains("dp4a.u32.s32"), "DP4A instruction");
        assert!(ptx.contains("ex2.approx"), "exp for sigmoid");
        assert!(ptx.contains("rcp.approx"), "rcp for sigmoid");
    }

    #[test]
    fn test_dual_accumulator_structure() {
        let k = FusedGateUpSwigluHwDp4aQ4KGemvKernel::new(1536, 4096);
        let ptx = k.emit_ptx();
        // Should have wg_ptr and wu_ptr params
        assert!(ptx.contains("wg_ptr"), "gate weight pointer");
        assert!(ptx.contains("wu_ptr"), "up weight pointer");
    }

    #[test]
    fn test_instruction_density() {
        let k = FusedGateUpSwigluHwDp4aQ4KGemvKernel::new(1536, 4096);
        let ptx = k.emit_ptx();

        let sb_loop_start = ptx.find("fgs_sb_loop:").expect("sb_loop label");
        let sb_loop_end = ptx.find("fgs_sb_end:").expect("sb_end label");
        let inner = &ptx[sb_loop_start..sb_loop_end];

        let insn_count = inner.matches(';').count();

        // C3: fused kernel should be <= 200 insn for 16 values of BOTH gate and up
        // vs 2 × 108 = 216 for separate kernels (saves ~10% instructions + eliminates SwiGLU kernel)
        assert!(
            insn_count <= 200,
            "C3 violated: inner loop has {} instructions (limit 200)",
            insn_count
        );

        // Must be less than 2× single kernel
        let savings_pct = (1.0 - insn_count as f64 / 216.0) * 100.0;
        eprintln!(
            "[C3] Fused inner loop: {} insn for 16 values of gate+up ({:.0}% savings vs 2×108=216)",
            insn_count, savings_pct
        );
    }

    #[test]
    fn test_value_coverage_contract() {
        // Same as HW DP4A: 16 threads × 16 values = 256 = Q4K_SUPER_BLOCK_SIZE
        assert_eq!(16 * 16, Q4K_SUPER_BLOCK_SIZE as usize);
    }

    #[test]
    fn dump_ptx() {
        let k = FusedGateUpSwigluHwDp4aQ4KGemvKernel::new(1536, 4096);
        let ptx = k.emit_ptx();
        eprintln!("{ptx}");
    }
}
