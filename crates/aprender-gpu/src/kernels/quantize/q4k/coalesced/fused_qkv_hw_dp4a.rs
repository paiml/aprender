use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory, PtxSync};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Fused K+V HW DP4A Q4_K GEMV kernel (trueno#237, PMAT-452)
///
/// Computes `k[i] = dot(W_k[i], x)` and `v[i] = dot(W_v[i], x)` in a single
/// kernel pass, eliminating:
/// - 1 kernel launch (K + V → fused)
/// - 2 global memory reads of Q8 input (shared between K and V)
///
/// Based on `FusedGateUpSwigluHwDp4aQ4KGemvKernel` (PMAT-034) with dual accumulators.
/// Unlike gate+up, outputs are raw dot products (no activation function).
///
/// # GQA note
///
/// Q projection has different output dim (num_heads × head_dim) than K/V
/// (num_kv_heads × head_dim). Q stays as a separate kernel launch.
/// K and V have identical dims, making them a natural fusion pair.
///
/// # Instruction budget
///
/// - Separate kernels: 2 × 108 = 216 insn/SB
/// - Fused kernel: ~170 insn/SB (Q8 loads + sums shared, no activation)
/// - Savings: ~21% fewer instructions + 1 fewer kernel launch per layer
///
/// # Contracts
///
/// - **C1**: Same half-warp thread mapping as HW DP4A (16 threads/SB)
/// - **C2**: Same value coverage (16 threads × 16 values = 256)
/// - **C3**: Inner loop ≤ 200 insn for 16 values of BOTH K and V
/// - **C4**: Same half-warp reduction, applied to both accumulators
/// - **C5**: Thread 0 stores both K and V outputs (no activation)
pub struct FusedQKVHwDp4aQ4KGemvKernel {
    /// Hidden dimension (input)
    pub hidden_dim: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of Q heads
    pub num_q_heads: u32,
    /// Number of KV heads
    pub num_kv_heads: u32,
}

impl FusedQKVHwDp4aQ4KGemvKernel {
    /// Create a new fused K+V DP4A kernel.
    #[must_use]
    pub fn new(hidden_dim: u32, head_dim: u32, num_q_heads: u32, num_kv_heads: u32) -> Self {
        Self { hidden_dim, head_dim, num_q_heads, num_kv_heads }
    }

    /// KV output dimension (num_kv_heads × head_dim)
    #[must_use]
    pub fn kv_dim(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }
}

impl Kernel for FusedQKVHwDp4aQ4KGemvKernel {
    fn name(&self) -> &str {
        "fused_qkv_hw_dp4a_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps: u32 = 3;
        let num_half_warps = num_warps * 2;
        // Shared memory: 2 accumulators (K + V) per half-warp
        let smem_size = (num_half_warps * 2 * 4) as usize;

        PtxKernel::new("fused_qkv_hw_dp4a_q4k_gemv")
            .param(PtxType::U64, "x_ptr") // Q8-quantized input activation (shared)
            .param(PtxType::U64, "wk_ptr") // Q4K K weights
            .param(PtxType::U64, "wv_ptr") // Q4K V weights
            .param(PtxType::U64, "y_k_ptr") // K projection output
            .param(PtxType::U64, "y_v_ptr") // V projection output
            .param(PtxType::U32, "k_dim") // Input dimension (hidden_dim)
            .param(PtxType::U32, "n_dim") // Output dimension (kv_dim)
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
                let q8_ptr = ctx.load_param_u64("x_ptr");
                let wk_ptr = ctx.load_param_u64("wk_ptr");
                let wv_ptr = ctx.load_param_u64("wv_ptr");
                let y_k_ptr = ctx.load_param_u64("y_k_ptr");
                let y_v_ptr = ctx.load_param_u64("y_v_ptr");

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

                ctx.label("fkv_row_loop");
                let row_oob = ctx.setp_ge_u32(row_idx, n_dim);
                ctx.branch_if(row_oob, "fkv_exit");

                // Row bases for K and V weights
                let row_off = ctx.mul_wide_u32_reg(row_idx, row_bytes);
                let wk_row_base = ctx.add_u64(wk_ptr, row_off);
                let wv_row_base = ctx.add_u64(wv_ptr, row_off);

                // Dual accumulators
                let acc_k = ctx.mov_f32_imm(0.0);
                let acc_v = ctx.mov_f32_imm(0.0);

                // SB loop: each half-warp processes 1 SB, stride by num_half_warps
                let sb_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(sb_idx, half_warp_id);

                ctx.label("fkv_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_sb);
                ctx.branch_if(sb_done, "fkv_sb_end");

                // Super-block offset
                let sb_off = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let wk_sb_addr = ctx.add_u64(wk_row_base, sb_off);
                let wv_sb_addr = ctx.add_u64(wv_row_base, sb_off);

                // ===== Load d, dmin for K =====
                let dk_f16 = ctx.ld_global_f16(wk_sb_addr);
                let dk = ctx.cvt_f32_f16(dk_f16);
                let dmink_addr = ctx.add_u64(wk_sb_addr, c_2_64);
                let dmink_f16 = ctx.ld_global_f16(dmink_addr);
                let dmink = ctx.cvt_f32_f16(dmink_f16);
                let neg_dmink = ctx.neg_f32(dmink);

                // ===== Load d, dmin for V =====
                let dv_f16 = ctx.ld_global_f16(wv_sb_addr);
                let dv = ctx.cvt_f32_f16(dv_f16);
                let dminv_addr = ctx.add_u64(wv_sb_addr, c_2_64);
                let dminv_f16 = ctx.ld_global_f16(dminv_addr);
                let dminv = ctx.cvt_f32_f16(dminv_f16);
                let neg_dminv = ctx.neg_f32(dminv);

                // ===== Scale loading for K =====
                let sck_base = ctx.add_u64(wk_sb_addr, c_4_64);
                let sck03 = ctx.ld_global_u32(sck_base);
                let sck47_addr = ctx.add_u64(sck_base, c_4_64);
                let sck47 = ctx.ld_global_u32(sck47_addr);
                let sck811_addr = ctx.add_u64(sck_base, c_8_64);
                let sck811 = ctx.ld_global_u32(sck811_addr);

                // ===== Scale loading for V =====
                let scv_base = ctx.add_u64(wv_sb_addr, c_4_64);
                let scv03 = ctx.ld_global_u32(scv_base);
                let scv47_addr = ctx.add_u64(scv_base, c_4_64);
                let scv47 = ctx.ld_global_u32(scv47_addr);
                let scv811_addr = ctx.add_u64(scv_base, c_8_64);
                let scv811 = ctx.ld_global_u32(scv811_addr);

                // ===== Scale extraction — K =====
                let sck_lo4 = ctx.and_u32(sck03, c_mask_6bit);
                let mnk_lo4 = ctx.and_u32(sck47, c_mask_6bit);
                let sck_hi_low = ctx.and_u32(sck811, c_mask_4bit);
                let t = ctx.shr_u32_imm(sck03, 6);
                let t = ctx.and_u32(t, c_mask_2bit);
                let sck_hi_top = ctx.shl_u32_imm(t, 4);
                let sck_hi4 = ctx.or_u32(sck_hi_low, sck_hi_top);
                let mnk_hi_raw = ctx.shr_u32_imm(sck47, 6);
                let mnk_hi_low = ctx.and_u32(mnk_hi_raw, c_mask_4bit);
                let t = ctx.shr_u32_imm(sck47, 6);
                let t = ctx.and_u32(t, c_mask_2bit);
                let mnk_hi_top = ctx.shl_u32_imm(t, 4);
                let mnk_hi4 = ctx.or_u32(mnk_hi_low, mnk_hi_top);

                let sck_src = ctx.selp_u32(p_hi, sck_hi4, sck_lo4);
                let mnk_src = ctx.selp_u32(p_hi, mnk_hi4, mnk_lo4);
                let sck0 = ctx.bfe_u32_reg(sck_src, byte_shift, 8);
                let sck1 = ctx.bfe_u32_reg(sck_src, byte_shift_hi, 8);
                let mnk0 = ctx.bfe_u32_reg(mnk_src, byte_shift, 8);
                let mnk1 = ctx.bfe_u32_reg(mnk_src, byte_shift_hi, 8);

                // ===== Scale extraction — V =====
                let scv_lo4 = ctx.and_u32(scv03, c_mask_6bit);
                let mnv_lo4 = ctx.and_u32(scv47, c_mask_6bit);
                let scv_hi_low = ctx.and_u32(scv811, c_mask_4bit);
                let t = ctx.shr_u32_imm(scv03, 6);
                let t = ctx.and_u32(t, c_mask_2bit);
                let scv_hi_top = ctx.shl_u32_imm(t, 4);
                let scv_hi4 = ctx.or_u32(scv_hi_low, scv_hi_top);
                let mnv_hi_raw = ctx.shr_u32_imm(scv47, 6);
                let mnv_hi_low = ctx.and_u32(mnv_hi_raw, c_mask_4bit);
                let t = ctx.shr_u32_imm(scv47, 6);
                let t = ctx.and_u32(t, c_mask_2bit);
                let mnv_hi_top = ctx.shl_u32_imm(t, 4);
                let mnv_hi4 = ctx.or_u32(mnv_hi_low, mnv_hi_top);

                let scv_src = ctx.selp_u32(p_hi, scv_hi4, scv_lo4);
                let mnv_src = ctx.selp_u32(p_hi, mnv_hi4, mnv_lo4);
                let scv0 = ctx.bfe_u32_reg(scv_src, byte_shift, 8);
                let scv1 = ctx.bfe_u32_reg(scv_src, byte_shift_hi, 8);
                let mnv0 = ctx.bfe_u32_reg(mnv_src, byte_shift, 8);
                let mnv1 = ctx.bfe_u32_reg(mnv_src, byte_shift_hi, 8);

                // ===== Load Q4K data: K =====
                let q4k_addr = ctx.add_u64(wk_sb_addr, q4_off_64);
                let vk0 = ctx.ld_global_u32(q4k_addr);
                let vk1_addr = ctx.add_u64(q4k_addr, c_16_64);
                let vk1 = ctx.ld_global_u32(vk1_addr);

                // ===== Load Q4K data: V =====
                let q4v_addr = ctx.add_u64(wv_sb_addr, q4_off_64);
                let vv0 = ctx.ld_global_u32(q4v_addr);
                let vv1_addr = ctx.add_u64(q4v_addr, c_16_64);
                let vv1 = ctx.ld_global_u32(vv1_addr);

                // ===== Q8 block base (SHARED between K and V) =====
                let q8_sb_off = ctx.mul_wide_u32_reg(sb_idx, c_288);
                let q8_sb_base = ctx.add_u64(q8_ptr, q8_sb_off);
                let q8_blk = ctx.add_u64(q8_sb_base, bq8_bytes_64);
                let q8_data = ctx.add_u64(q8_blk, lig_x4_64);

                // ===== QR=0: Low nibbles =====
                let vk0_lo = ctx.and_u32(vk0, c_mask_4bit);
                let vk1_lo = ctx.and_u32(vk1, c_mask_4bit);
                let vv0_lo = ctx.and_u32(vv0, c_mask_4bit);
                let vv1_lo = ctx.and_u32(vv1, c_mask_4bit);

                let u0_lo = ctx.ld_global_u32(q8_data);
                let u1_lo_addr = ctx.add_u64(q8_data, c_16_64);
                let u1_lo = ctx.ld_global_u32(u1_lo_addr);

                // K DP4A: dot_k = vk0_lo . u0 + vk1_lo . u1
                let dotk0 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(dotk0, vk0_lo, u0_lo);
                ctx.dp4a_u32_s32_inplace(dotk0, vk1_lo, u1_lo);

                // V DP4A: dot_v = vv0_lo . u0 + vv1_lo . u1
                let dotv0 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(dotv0, vv0_lo, u0_lo);
                ctx.dp4a_u32_s32_inplace(dotv0, vv1_lo, u1_lo);

                // Q8 byte sum (SHARED): sum = ones . u0 + ones . u1
                let sum0 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(sum0, c_ones, u0_lo);
                ctx.dp4a_u32_s32_inplace(sum0, c_ones, u1_lo);

                // Q8 scale
                let q8_d0_addr = ctx.add_u64(q8_blk, c_32_64);
                let q8_d0_f16 = ctx.ld_global_f16(q8_d0_addr);
                let q8_d0 = ctx.cvt_f32_f16(q8_d0_f16);

                // K accumulate: acc_k += q8_d * (dk*sck*dot - dmink*mnk*sum)
                let sdotk0 = ctx.mul_lo_s32(sck0, dotk0);
                let msumk0 = ctx.mul_lo_s32(mnk0, sum0);
                let sdotk0_f = ctx.cvt_f32_s32(sdotk0);
                let msumk0_f = ctx.cvt_f32_s32(msumk0);
                let tk1 = ctx.mul_f32(dk, sdotk0_f);
                let tk3 = ctx.fma_f32(neg_dmink, msumk0_f, tk1);
                let q8_d0_tk3 = ctx.mul_f32(q8_d0, tk3);
                ctx.add_f32_inplace(acc_k, q8_d0_tk3);

                // V accumulate: acc_v += q8_d * (dv*scv*dot - dminv*mnv*sum)
                let sdotv0 = ctx.mul_lo_s32(scv0, dotv0);
                let msumv0 = ctx.mul_lo_s32(mnv0, sum0);
                let sdotv0_f = ctx.cvt_f32_s32(sdotv0);
                let msumv0_f = ctx.cvt_f32_s32(msumv0);
                let tv1 = ctx.mul_f32(dv, sdotv0_f);
                let tv3 = ctx.fma_f32(neg_dminv, msumv0_f, tv1);
                let q8_d0_tv3 = ctx.mul_f32(q8_d0, tv3);
                ctx.add_f32_inplace(acc_v, q8_d0_tv3);

                // ===== QR=1: High nibbles =====
                let vk0_hi = ctx.shr_u32_imm(vk0, 4);
                let vk0_hi = ctx.and_u32(vk0_hi, c_mask_4bit);
                let vk1_hi = ctx.shr_u32_imm(vk1, 4);
                let vk1_hi = ctx.and_u32(vk1_hi, c_mask_4bit);
                let vv0_hi = ctx.shr_u32_imm(vv0, 4);
                let vv0_hi = ctx.and_u32(vv0_hi, c_mask_4bit);
                let vv1_hi = ctx.shr_u32_imm(vv1, 4);
                let vv1_hi = ctx.and_u32(vv1_hi, c_mask_4bit);

                // Q8 block +1 (36 bytes later)
                let q8_blk_hi = ctx.add_u64(q8_blk, c_36_64);
                let q8_data_hi = ctx.add_u64(q8_blk_hi, lig_x4_64);

                let u0_hi = ctx.ld_global_u32(q8_data_hi);
                let u1_hi_addr = ctx.add_u64(q8_data_hi, c_16_64);
                let u1_hi = ctx.ld_global_u32(u1_hi_addr);

                // K DP4A high nibbles
                let dotk1 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(dotk1, vk0_hi, u0_hi);
                ctx.dp4a_u32_s32_inplace(dotk1, vk1_hi, u1_hi);

                // V DP4A high nibbles
                let dotv1 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(dotv1, vv0_hi, u0_hi);
                ctx.dp4a_u32_s32_inplace(dotv1, vv1_hi, u1_hi);

                // Q8 byte sum high (SHARED)
                let sum1 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(sum1, c_ones, u0_hi);
                ctx.dp4a_u32_s32_inplace(sum1, c_ones, u1_hi);

                let q8_d1_addr = ctx.add_u64(q8_blk_hi, c_32_64);
                let q8_d1_f16 = ctx.ld_global_f16(q8_d1_addr);
                let q8_d1 = ctx.cvt_f32_f16(q8_d1_f16);

                // K accumulate QR=1
                let sdotk1 = ctx.mul_lo_s32(sck1, dotk1);
                let msumk1 = ctx.mul_lo_s32(mnk1, sum1);
                let sdotk1_f = ctx.cvt_f32_s32(sdotk1);
                let msumk1_f = ctx.cvt_f32_s32(msumk1);
                let tk1b = ctx.mul_f32(dk, sdotk1_f);
                let tk3b = ctx.fma_f32(neg_dmink, msumk1_f, tk1b);
                let q8_d1_tk3b = ctx.mul_f32(q8_d1, tk3b);
                ctx.add_f32_inplace(acc_k, q8_d1_tk3b);

                // V accumulate QR=1
                let sdotv1 = ctx.mul_lo_s32(scv1, dotv1);
                let msumv1 = ctx.mul_lo_s32(mnv1, sum1);
                let sdotv1_f = ctx.cvt_f32_s32(sdotv1);
                let msumv1_f = ctx.cvt_f32_s32(msumv1);
                let tv1b = ctx.mul_f32(dv, sdotv1_f);
                let tv3b = ctx.fma_f32(neg_dminv, msumv1_f, tv1b);
                let q8_d1_tv3b = ctx.mul_f32(q8_d1, tv3b);
                ctx.add_f32_inplace(acc_v, q8_d1_tv3b);

                // Next SB
                ctx.add_u32_reg_inplace(sb_idx, num_hw);
                ctx.branch("fkv_sb_loop");

                ctx.label("fkv_sb_end");

                // ===== C4: Half-warp reduction for BOTH accumulators =====
                // K reduction
                let t = ctx.shfl_down_f32(acc_k, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_k, t);
                let t = ctx.shfl_down_f32(acc_k, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_k, t);
                let t = ctx.shfl_down_f32(acc_k, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_k, t);
                let t = ctx.shfl_down_f32(acc_k, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_k, t);

                // V reduction
                let t = ctx.shfl_down_f32(acc_v, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_v, t);
                let t = ctx.shfl_down_f32(acc_v, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_v, t);
                let t = ctx.shfl_down_f32(acc_v, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_v, t);
                let t = ctx.shfl_down_f32(acc_v, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_v, t);

                // Half-warp lane 0 stores both partials to shared memory
                let z = ctx.mov_u32_imm(0);
                let is_hl0 = ctx.setp_eq_u32(half_lane, z);
                ctx.branch_if_not(is_hl0, "fkv_skip_sm");

                // Layout: [k_0..k_(nhw-1), v_0..v_(nhw-1)]
                let sm_k_off = ctx.shl_u32_imm(half_warp_id, 2);
                let sm_k_addr = ctx.cvt_u64_u32(sm_k_off);
                ctx.st_shared_f32(sm_k_addr, acc_k);

                let nhw_bytes = ctx.mov_u32_imm(num_half_warps * 4);
                let sm_v_off = ctx.add_u32_reg(sm_k_off, nhw_bytes);
                let sm_v_addr = ctx.cvt_u64_u32(sm_v_off);
                ctx.st_shared_f32(sm_v_addr, acc_v);

                ctx.label("fkv_skip_sm");
                ctx.bar_sync(0);

                // ===== Warp-0 reduction + C5: store K and V =====
                let is_warp0 = ctx.setp_eq_u32(warp_id, z);
                ctx.branch_if_not(is_warp0, "fkv_skip_store");

                let in_range = ctx.setp_lt_u32_imm(lane_id, num_half_warps);
                let zero_f = ctx.mov_f32_imm(0.0);

                // Reduce K partials
                let k_sm_off = ctx.shl_u32_imm(lane_id, 2);
                let k_sm_addr = ctx.cvt_u64_u32(k_sm_off);
                let k_loaded = ctx.ld_shared_f32(k_sm_addr);
                let k_partial = ctx.selp_f32(in_range, k_loaded, zero_f);

                let t = ctx.shfl_down_f32(k_partial, 4, 0xFFFF_FFFF);
                let k_partial = ctx.add_f32(k_partial, t);
                let t = ctx.shfl_down_f32(k_partial, 2, 0xFFFF_FFFF);
                let k_partial = ctx.add_f32(k_partial, t);
                let t = ctx.shfl_down_f32(k_partial, 1, 0xFFFF_FFFF);
                let k_sum = ctx.add_f32(k_partial, t);

                // Reduce V partials
                let nhw_bytes = ctx.mov_u32_imm(num_half_warps * 4);
                let v_sm_off = ctx.add_u32_reg(k_sm_off, nhw_bytes);
                let v_sm_addr = ctx.cvt_u64_u32(v_sm_off);
                let v_loaded = ctx.ld_shared_f32(v_sm_addr);
                let v_partial = ctx.selp_f32(in_range, v_loaded, zero_f);

                let t = ctx.shfl_down_f32(v_partial, 4, 0xFFFF_FFFF);
                let v_partial = ctx.add_f32(v_partial, t);
                let t = ctx.shfl_down_f32(v_partial, 2, 0xFFFF_FFFF);
                let v_partial = ctx.add_f32(v_partial, t);
                let t = ctx.shfl_down_f32(v_partial, 1, 0xFFFF_FFFF);
                let v_sum = ctx.add_f32(v_partial, t);

                // Thread 0: store K and V outputs (no activation function)
                let is_t0 = ctx.setp_eq_u32(lane_id, z);
                ctx.branch_if_not(is_t0, "fkv_skip_store");

                let y_off = ctx.mul_wide_u32(row_idx, 4);
                let y_k_addr = ctx.add_u64(y_k_ptr, y_off);
                ctx.st_global_f32(y_k_addr, k_sum);
                let y_v_addr = ctx.add_u64(y_v_ptr, y_off);
                ctx.st_global_f32(y_v_addr, v_sum);

                ctx.label("fkv_skip_store");

                // Next row (grid-stride)
                ctx.add_u32_reg_inplace(row_idx, grid_dim);
                ctx.bar_sync(0);
                ctx.branch("fkv_row_loop");

                ctx.label("fkv_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_emits_valid() {
        let k = FusedQKVHwDp4aQ4KGemvKernel::new(1536, 128, 12, 2);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("fused_qkv_hw_dp4a_q4k_gemv"), "kernel name");
        assert!(ptx.contains("dp4a.u32.s32"), "DP4A instruction");
        // No sigmoid/exp — unlike gate+up, this is raw dot product
        assert!(!ptx.contains("ex2.approx"), "no activation function");
    }

    #[test]
    fn test_dual_accumulator_structure() {
        let k = FusedQKVHwDp4aQ4KGemvKernel::new(1536, 128, 12, 2);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("wk_ptr"), "K weight pointer");
        assert!(ptx.contains("wv_ptr"), "V weight pointer");
        assert!(ptx.contains("y_k_ptr"), "K output pointer");
        assert!(ptx.contains("y_v_ptr"), "V output pointer");
    }

    #[test]
    fn test_instruction_density() {
        let k = FusedQKVHwDp4aQ4KGemvKernel::new(1536, 128, 12, 2);
        let ptx = k.emit_ptx();

        let sb_loop_start = ptx.find("fkv_sb_loop:").expect("sb_loop label");
        let sb_loop_end = ptx.find("fkv_sb_end:").expect("sb_end label");
        let inner = &ptx[sb_loop_start..sb_loop_end];

        let insn_count = inner.matches(';').count();

        // C3: fused kernel should be <= 200 insn for 16 values of BOTH K and V
        // Should be slightly less than gate+up (~170 vs ~193) since no SwiGLU
        assert!(
            insn_count <= 200,
            "C3 violated: inner loop has {} instructions (limit 200)",
            insn_count
        );

        let savings_pct = (1.0 - insn_count as f64 / 216.0) * 100.0;
        eprintln!(
            "[C3] Fused K+V inner loop: {} insn ({:.0}% savings vs 2×108=216)",
            insn_count, savings_pct
        );
    }

    #[test]
    fn test_kv_dim() {
        let k = FusedQKVHwDp4aQ4KGemvKernel::new(1536, 128, 12, 2);
        assert_eq!(k.kv_dim(), 256);
    }
}
