use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory, PtxSync};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// GH-141: Batched Half-warp DP4A Q4_K GEMV kernel
///
/// Extends HalfWarpDp4aQ4KGemvKernel for M>1 batch elements:
/// - Dequantizes Q4K weights ONCE per super-block (shared across M)
/// - Loads M Q8_1 activation vectors per SB
/// - M DP4A dot products per QR iteration
/// - M accumulators reduced via half-warp + cross-warp shared memory
///
/// # Five-Whys Root Cause (GH-141)
///
/// 1. WHY is c=4 decode 0.64x llama.cpp? → cuBLAS SGEMM at M=4 reads FP32 (4 B/elem)
/// 2. WHY use cuBLAS SGEMM? → Batched Q4K GEMV (single warp) is slower (35 tok/s)
/// 3. WHY is single-warp batched slow? → 32 threads/block, FP32 activations, no DP4A
/// 4. WHY no DP4A batched? → Only M=1 DP4A kernels existed
/// 5. WHY not extend HwDp4a to M>1? → This kernel does exactly that
///
/// # Provable Contracts
///
/// - **C1**: Same thread mapping as HwDp4a (16 threads/SB, half-warp structure)
/// - **C2**: Same value coverage (256 values/SB via 4 groups × 4 threads × 16 values)
/// - **C3**: M accumulators per thread, compile-time unrolled
/// - **C4**: M reductions via shared memory (M × num_half_warps entries)
/// - **C5**: Q8_1 layout: M sequential vectors, each K/32 blocks × 36 bytes
pub struct BatchedHwDp4aQ4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Batch size M (number of sequences)
    pub m: u32,
    /// Number of warps per block (default: 3, giving 6 half-warps).
    /// PMAT-089: 4 warps FALSIFIED (register pressure, -2% decode). 3 is optimal.
    pub num_warps: u32,
}

impl BatchedHwDp4aQ4KGemvKernel {
    /// Create a new batched HW DP4A Q4K GEMV kernel.
    pub fn new(k: u32, n: u32, m: u32) -> Self {
        Self { k, n, m, num_warps: 3 }
    }
}

impl Kernel for BatchedHwDp4aQ4KGemvKernel {
    fn name(&self) -> &str {
        "batched_hw_dp4a_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let num_half_warps = num_warps * 2;
        let m = self.m;
        // Shared memory: M accumulators per half-warp (f32 each)
        let smem_size = (num_half_warps * m * 4) as usize;

        PtxKernel::new("batched_hw_dp4a_q4k_gemv")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "q8_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .param(PtxType::U32, "m_dim")
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
                let w_ptr = ctx.load_param_u64("w_ptr");
                let q8_ptr = ctx.load_param_u64("q8_ptr");

                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_sb = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);
                let sb_bytes_reg = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_sb, sb_bytes_reg);

                // ===== C1: Half-warp thread mapping (same as HwDp4a) =====
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

                // Q8 per-thread offsets (precomputed before loop)
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

                // Scale extraction invariants (ci = bq8_group)
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

                // ===== C5: Q8_1 stride per batch element =====
                // Each activation vector: num_sb * 8 * 36 = num_sb * 288 bytes
                let q8_vec_stride = ctx.mul_u32_reg(num_sb, c_288);
                let _q8_vec_stride_64 = ctx.cvt_u64_u32(q8_vec_stride);

                // ===== C3: M accumulators (compile-time unrolled) =====
                let f32_zero = ctx.mov_f32_imm(0.0);
                let mut accs = Vec::with_capacity(m as usize);
                for _ in 0..m {
                    accs.push(ctx.mov_f32_imm(0.0));
                }

                // ===== Grid-stride row loop =====
                let row_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(row_idx, block_id);

                ctx.label("bhw_row_loop");
                let row_oob = ctx.setp_ge_u32(row_idx, n_dim);
                ctx.branch_if(row_oob, "bhw_exit");

                let row_off = ctx.mul_wide_u32_reg(row_idx, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_off);

                // Zero accumulators for this row
                for acc in &accs {
                    ctx.mov_f32_reg(*acc, f32_zero);
                }

                // SB loop: each half-warp processes 1 SB, stride by num_half_warps
                let sb_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(sb_idx, half_warp_id);

                ctx.label("bhw_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_sb);
                ctx.branch_if(sb_done, "bhw_sb_end");

                // Super-block base (Q4K weights — loaded ONCE, shared across M)
                let sb_off = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // Load d, dmin (f16 -> f32) — shared
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let dmin_addr = ctx.add_u64(sb_addr, c_2_64);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);
                let neg_dmin = ctx.neg_f32(dmin);

                // ===== Scale loading (shared across M) =====
                let sc_base = ctx.add_u64(sb_addr, c_4_64);
                let sc03 = ctx.ld_global_u32(sc_base);
                let sc47_addr = ctx.add_u64(sc_base, c_4_64);
                let sc47 = ctx.ld_global_u32(sc47_addr);
                let sc811_addr = ctx.add_u64(sc_base, c_8_64);
                let sc811 = ctx.ld_global_u32(sc811_addr);

                // Scale extraction (same as HwDp4a, PMAT-029 hoisted constants)
                let sc_lo4 = ctx.and_u32(sc03, c_mask_6bit);
                let mn_lo4 = ctx.and_u32(sc47, c_mask_6bit);
                let sc_hi_low = ctx.and_u32(sc811, c_mask_4bit);
                let t = ctx.shr_u32_imm(sc03, 6);
                let t = ctx.and_u32(t, c_mask_2bit);
                let sc_hi_top = ctx.shl_u32_imm(t, 4);
                let sc_hi4 = ctx.or_u32(sc_hi_low, sc_hi_top);

                let mn_hi_raw = ctx.shr_u32_imm(sc811, 4);
                let mn_hi_low = ctx.and_u32(mn_hi_raw, c_mask_4bit);
                let t = ctx.shr_u32_imm(sc47, 6);
                let t = ctx.and_u32(t, c_mask_2bit);
                let mn_hi_top = ctx.shl_u32_imm(t, 4);
                let mn_hi4 = ctx.or_u32(mn_hi_low, mn_hi_top);

                let sc_src = ctx.selp_u32(p_hi, sc_hi4, sc_lo4);
                let mn_src = ctx.selp_u32(p_hi, mn_hi4, mn_lo4);

                let sc0 = ctx.bfe_u32_reg(sc_src, byte_shift, 8);
                let sc1 = ctx.bfe_u32_reg(sc_src, byte_shift_hi, 8);
                let mn0 = ctx.bfe_u32_reg(mn_src, byte_shift, 8);
                let mn1 = ctx.bfe_u32_reg(mn_src, byte_shift_hi, 8);

                // ===== Load Q4K data (shared across M) =====
                let q4_addr = ctx.add_u64(sb_addr, q4_off_64);
                let v0 = ctx.ld_global_u32(q4_addr);
                let v1_addr = ctx.add_u64(q4_addr, c_16_64);
                let v1 = ctx.ld_global_u32(v1_addr);

                // Pre-extract nibbles (shared across M, PMAT-029 hoisted constants)
                let v0_lo = ctx.and_u32(v0, c_mask_4bit);
                let v1_lo = ctx.and_u32(v1, c_mask_4bit);
                let v0_hi = ctx.shr_u32_imm(v0, 4);
                let v0_hi = ctx.and_u32(v0_hi, c_mask_4bit);
                let v1_hi = ctx.shr_u32_imm(v1, 4);
                let v1_hi = ctx.and_u32(v1_hi, c_mask_4bit);

                // ===== Per-batch-element Q8 loading + DP4A (compile-time unrolled) =====
                let q8_sb_off_base = ctx.mul_wide_u32_reg(sb_idx, c_288);

                for mi in 0..m {
                    // Q8 base for batch element mi
                    let q8_m_off = if mi == 0 {
                        ctx.mov_u64_imm(0)
                    } else {
                        let mi_reg = ctx.mov_u32_imm(mi);
                        ctx.mul_wide_u32_reg(mi_reg, q8_vec_stride)
                    };
                    let q8_m_base = ctx.add_u64(q8_ptr, q8_m_off);
                    let q8_sb_base = ctx.add_u64(q8_m_base, q8_sb_off_base);
                    let q8_blk = ctx.add_u64(q8_sb_base, bq8_bytes_64);
                    let q8_data = ctx.add_u64(q8_blk, lig_x4_64);

                    // ===== QR=0: Low nibbles =====
                    let u0_lo = ctx.ld_global_u32(q8_data);
                    let u1_lo_addr = ctx.add_u64(q8_data, c_16_64);
                    let u1_lo = ctx.ld_global_u32(u1_lo_addr);

                    let dot0 = ctx.mov_u32_imm(0);
                    ctx.dp4a_u32_s32_inplace(dot0, v0_lo, u0_lo);
                    ctx.dp4a_u32_s32_inplace(dot0, v1_lo, u1_lo);

                    let sum0 = ctx.mov_u32_imm(0);
                    ctx.dp4a_u32_s32_inplace(sum0, c_ones, u0_lo);
                    ctx.dp4a_u32_s32_inplace(sum0, c_ones, u1_lo);

                    let q8_d0_addr = ctx.add_u64(q8_blk, c_32_64);
                    let q8_d0_f16 = ctx.ld_global_f16(q8_d0_addr);
                    let q8_d0 = ctx.cvt_f32_f16(q8_d0_f16);

                    let sdot0 = ctx.mul_lo_s32(sc0, dot0);
                    let msum0 = ctx.mul_lo_s32(mn0, sum0);
                    let sdot0_f = ctx.cvt_f32_s32(sdot0);
                    let msum0_f = ctx.cvt_f32_s32(msum0);
                    let t1 = ctx.mul_f32(d, sdot0_f);
                    let t3 = ctx.fma_f32(neg_dmin, msum0_f, t1);
                    let q8_d0_t3 = ctx.mul_f32(q8_d0, t3);
                    ctx.add_f32_inplace(accs[mi as usize], q8_d0_t3);

                    // ===== QR=1: High nibbles =====
                    let q8_blk_hi = ctx.add_u64(q8_blk, c_36_64);
                    let q8_data_hi = ctx.add_u64(q8_blk_hi, lig_x4_64);

                    let u0_hi = ctx.ld_global_u32(q8_data_hi);
                    let u1_hi_addr = ctx.add_u64(q8_data_hi, c_16_64);
                    let u1_hi = ctx.ld_global_u32(u1_hi_addr);

                    let dot1 = ctx.mov_u32_imm(0);
                    ctx.dp4a_u32_s32_inplace(dot1, v0_hi, u0_hi);
                    ctx.dp4a_u32_s32_inplace(dot1, v1_hi, u1_hi);

                    let sum1 = ctx.mov_u32_imm(0);
                    ctx.dp4a_u32_s32_inplace(sum1, c_ones, u0_hi);
                    ctx.dp4a_u32_s32_inplace(sum1, c_ones, u1_hi);

                    let q8_d1_addr = ctx.add_u64(q8_blk_hi, c_32_64);
                    let q8_d1_f16 = ctx.ld_global_f16(q8_d1_addr);
                    let q8_d1 = ctx.cvt_f32_f16(q8_d1_f16);

                    let sdot1 = ctx.mul_lo_s32(sc1, dot1);
                    let msum1 = ctx.mul_lo_s32(mn1, sum1);
                    let sdot1_f = ctx.cvt_f32_s32(sdot1);
                    let msum1_f = ctx.cvt_f32_s32(msum1);
                    let t1 = ctx.mul_f32(d, sdot1_f);
                    let t3 = ctx.fma_f32(neg_dmin, msum1_f, t1);
                    let q8_d1_t3 = ctx.mul_f32(q8_d1, t3);
                    ctx.add_f32_inplace(accs[mi as usize], q8_d1_t3);
                }

                // Stride by num_half_warps
                ctx.add_u32_reg_inplace(sb_idx, num_hw);
                ctx.branch("bhw_sb_loop");

                ctx.label("bhw_sb_end");

                // ===== C4: Half-warp reduction for M accumulators =====
                for acc in &accs {
                    let t = ctx.shfl_down_f32(*acc, 8, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(*acc, t);
                    let t = ctx.shfl_down_f32(*acc, 4, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(*acc, t);
                    let t = ctx.shfl_down_f32(*acc, 2, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(*acc, t);
                    let t = ctx.shfl_down_f32(*acc, 1, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(*acc, t);
                }

                // Half-warp lane 0 stores M values to shared memory
                let z = ctx.mov_u32_imm(0);
                let is_hl0 = ctx.setp_eq_u32(half_lane, z);
                ctx.branch_if_not(is_hl0, "bhw_skip_sm");

                for (mi, acc) in accs.iter().enumerate() {
                    // sm_offset = (half_warp_id * M + mi) * 4
                    let hw_m = ctx.mul_u32(half_warp_id, m);
                    let idx = ctx.add_u32(hw_m, mi as u32);
                    let sm_off = ctx.shl_u32_imm(idx, 2);
                    let sm_addr = ctx.cvt_u64_u32(sm_off);
                    ctx.st_shared_f32(sm_addr, *acc);
                }

                ctx.label("bhw_skip_sm");
                ctx.bar_sync(0);

                // ===== PMAT-089: Parallel warp-0 reduction for M batch elements =====
                // Warp 0 threads load from smem and shfl_down reduce.
                let is_warp0 = ctx.setp_eq_u32(warp_id, z);
                ctx.branch_if_not(is_warp0, "bhw_skip_store");

                let in_range = ctx.setp_lt_u32_imm(lane_id, num_half_warps);
                let zero_f = ctx.mov_f32_imm(0.0);

                // Lane 0 predicate for store (computed once outside loop)
                let is_l0 = ctx.setp_eq_u32(lane_id, z);

                for mi in 0..m {
                    // Thread i loads smem[i * M + mi] if i < num_half_warps
                    // Layout: smem[hw * M + mi], stride = M between half-warps
                    let mi_off = ctx.mov_u32_imm(mi);
                    let hw_m = ctx.mul_u32(lane_id, m);
                    let idx = ctx.add_u32_reg(hw_m, mi_off);
                    let sm_off = ctx.shl_u32_imm(idx, 2);
                    let sm_addr = ctx.cvt_u64_u32(sm_off);
                    let loaded = ctx.ld_shared_f32(sm_addr);
                    let partial = ctx.selp_f32(in_range, loaded, zero_f);

                    // Warp shuffle reduce: 3 steps for up to 8 half-warps
                    let t = ctx.shfl_down_f32(partial, 4, 0xFFFF_FFFF);
                    let partial = ctx.add_f32(partial, t);
                    let t = ctx.shfl_down_f32(partial, 2, 0xFFFF_FFFF);
                    let partial = ctx.add_f32(partial, t);
                    let t = ctx.shfl_down_f32(partial, 1, 0xFFFF_FFFF);
                    let result = ctx.add_f32(partial, t);

                    // Lane 0 stores the final sum (predicated branch per mi)
                    let skip_label = format!("bhw_skip_mi{mi}");
                    ctx.branch_if_not(is_l0, &skip_label);

                    // y[mi * N + row] — row-major output for batch element mi
                    // realizr#220 FIX: Compute element index FIRST, then multiply by 4 for byte offset.
                    // Previous code mixed elements (mi*n_dim) with bytes (row*4), causing
                    // batch elements to overwrite each other's output.
                    let mi_reg = ctx.mov_u32_imm(mi);
                    let y_mi_base = ctx.mul_u32_reg(mi_reg, n_dim);
                    let y_idx = ctx.add_u32_reg(y_mi_base, row_idx);
                    let y_off = ctx.mul_wide_u32(y_idx, 4);
                    let y_addr = ctx.add_u64(y_ptr, y_off);
                    ctx.st_global_f32(y_addr, result);

                    ctx.label(&skip_label);
                }

                ctx.label("bhw_skip_store");

                // Next row (grid-stride)
                ctx.add_u32_reg_inplace(row_idx, grid_dim);
                ctx.bar_sync(0);
                ctx.branch("bhw_row_loop");

                ctx.label("bhw_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_emits_batched_hw_dp4a() {
        let k = BatchedHwDp4aQ4KGemvKernel::new(1536, 256, 4);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("batched_hw_dp4a_q4k_gemv"), "kernel name present");
        assert!(ptx.contains("dp4a.u32.s32"), "DP4A instruction present");
        // Should have half-warp identity
        assert!(ptx.contains("and.b32"), "half_lane = lane_id & 15");
    }

    #[test]
    fn test_batched_m2() {
        // Verify M=2 kernel also compiles
        let k = BatchedHwDp4aQ4KGemvKernel::new(1536, 256, 2);
        let ptx = k.emit_ptx();
        assert!(!ptx.is_empty());
    }

    #[test]
    fn test_batched_m8() {
        // Verify M=8 kernel compiles
        let k = BatchedHwDp4aQ4KGemvKernel::new(1536, 256, 8);
        let ptx = k.emit_ptx();
        assert!(!ptx.is_empty());
    }
}
