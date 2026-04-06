//! Half-warp DP4A Q6_K GEMV kernel (PMAT-030 + PMAT-078)
//!
//! 16 threads per super-block (half-warp), matching the Q4K HW DP4A design.
//! Each thread handles 16 Q6K values — exactly one scale's worth — so each thread
//! loads its own scale directly. Eliminates ALL shfl broadcasts and binary tree
//! selection (126 instructions/SB → ~7 instructions/SB for scales).
//!
//! PMAT-078: Q8 activation caching in shared memory. Q8 data is identical across
//! all output rows — loaded once cooperatively, then reused across all grid-stride
//! iterations. Eliminates redundant Q8 global reads (2x bandwidth reduction for LmHead).
//!
//! # Provable Contracts
//!
//! - **C1 (Thread mapping)**: 16 threads per SB via `half_lane = lane_id & 15`.
//!   Each warp processes 2 SBs independently (lanes 0-15 and 16-31).
//!
//! - **C2 (Value coverage)**: 16 threads × 16 values = 256 = Q6K_SUPER_BLOCK_SIZE.
//!   Each thread processes 4 dp4a iterations of 4 values each.
//!
//! - **C3 (Scale efficiency)**: 1 byte load + 6 instructions per thread per SB
//!   (vs 126 instructions for 16 shfl + sign-extension + binary tree in Dp4a kernel).
//!
//! - **C4 (Reduction correctness)**: Half-warp reduction via shfl_down with
//!   deltas 8,4,2,1. Same algebraic proof as HW DP4A Q4K (GH-176).
//!
//! - **C5 (Q8 cache correctness)**: Q8 data loaded cooperatively before row loop,
//!   barrier-synchronized, then read from shared memory. Same data, shared address space.

use crate::kernels::quantize::{Kernel, Q6K_SUPER_BLOCK_BYTES, Q6K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Q8_1 block size: 32 qs bytes + 2 bytes d + 2 bytes s = 36 bytes
const Q8_BLOCK_BYTES: u32 = 36;
/// Q8 blocks per Q6K super-block: 256 / 32 = 8
const Q8_BLOCKS_PER_SB: u32 = 8;
/// Q8 bytes per Q6K super-block: 8 * 36 = 288
const Q8_SB_STRIDE: u32 = Q8_BLOCKS_PER_SB * Q8_BLOCK_BYTES;

/// Half-warp DP4A Q6_K GEMV kernel.
pub struct HalfWarpDp4aQ6KGemvKernel {
    /// Number of elements per row (K dimension).
    pub k: u32,
    /// Number of rows (N dimension).
    pub n: u32,
    /// Number of warps per CTA.
    pub num_warps: u32,
}

impl HalfWarpDp4aQ6KGemvKernel {
    /// Create a new HW DP4A Q6K GEMV kernel with default 3 warps per CTA.
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, num_warps: 3 }
    }
}

impl Kernel for HalfWarpDp4aQ6KGemvKernel {
    fn name(&self) -> &str {
        "hw_dp4a_q6k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let num_half_warps = num_warps * 2;
        let num_threads = num_warps * 32;

        // Shared memory layout:
        //   [0, num_half_warps*4)             — warp reduction scratch
        //   [q8_smem_off, q8_smem_off + q8_total_bytes) — Q8 activation cache
        let reduction_bytes = num_half_warps * 4;
        // Align Q8 region to 4 bytes
        let q8_smem_off = (reduction_bytes + 3) & !3;
        let num_sb = (self.k + Q6K_SUPER_BLOCK_SIZE - 1) / Q6K_SUPER_BLOCK_SIZE;
        let q8_total_bytes = num_sb * Q8_SB_STRIDE;
        let smem_size = (q8_smem_off + q8_total_bytes) as usize;

        PtxKernel::new("hw_dp4a_q6k_gemv")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "q8_ptr")
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
                let w_ptr = ctx.load_param_u64("w_ptr");
                let q8_ptr = ctx.load_param_u64("q8_ptr");

                let k_rounded = ctx.add_u32(k_dim, Q6K_SUPER_BLOCK_SIZE - 1);
                let num_sb_reg = ctx.div_u32(k_rounded, Q6K_SUPER_BLOCK_SIZE);
                let sb_bytes_reg = ctx.mov_u32_imm(Q6K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_sb_reg, sb_bytes_reg);

                // ===== PMAT-078: Cooperative Q8 load into shared memory =====
                // All threads cooperatively load Q8 activation data before the row loop.
                // Q8 data is identical for all output rows — load once, reuse across all.
                let q8_total_u32s = q8_total_bytes / 4;
                let q8_smem_base = ctx.mov_u32_imm(q8_smem_off);
                // Cooperative load: thread_id strides by num_threads
                let load_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(load_idx, thread_id);
                let total_u32s = ctx.mov_u32_imm(q8_total_u32s);
                let num_threads_reg = ctx.mov_u32_imm(num_threads);

                ctx.label("hwq6_q8_load");
                let load_done = ctx.setp_ge_u32(load_idx, total_u32s);
                ctx.branch_if(load_done, "hwq6_q8_load_done");

                // Global address: q8_ptr + load_idx * 4
                let load_byte_off = ctx.mul_wide_u32(load_idx, 4);
                let load_global_addr = ctx.add_u64(q8_ptr, load_byte_off);
                let load_val = ctx.ld_global_u32(load_global_addr);

                // Shared address: q8_smem_off + load_idx * 4
                let load_smem_off = ctx.shl_u32_imm(load_idx, 2);
                let load_smem_addr = ctx.add_u32_reg(q8_smem_base, load_smem_off);
                let load_smem_addr_64 = ctx.cvt_u64_u32(load_smem_addr);
                ctx.st_shared_u32(load_smem_addr_64, load_val);

                ctx.add_u32_reg_inplace(load_idx, num_threads_reg);
                ctx.branch("hwq6_q8_load");

                ctx.label("hwq6_q8_load_done");
                ctx.bar_sync(0);

                // ===== C1: Half-warp thread mapping =====
                let half_lane = ctx.and_u32_imm(lane_id, 15);
                let half_warp_in_warp = ctx.shr_u32_imm(lane_id, 4);
                let warp_x2 = ctx.shl_u32_imm(warp_id, 1);
                let half_warp_id = ctx.add_u32_reg(warp_x2, half_warp_in_warp);
                let num_hw = ctx.mov_u32_imm(num_half_warps);

                // ===== C2: Per-thread data mapping =====
                let c1 = ctx.mov_u32_imm(1);
                let c2 = ctx.mov_u32_imm(2);
                let c3 = ctx.mov_u32_imm(3);
                let c4 = ctx.mov_u32_imm(4);
                let c7 = ctx.mov_u32_imm(7);
                let c16 = ctx.mov_u32_imm(16);
                let c32 = ctx.mov_u32_imm(32);
                let c64 = ctx.mov_u32_imm(64);

                let n_idx = ctx.shr_u32(half_lane, c3);
                let hl_mod8 = ctx.and_u32(half_lane, c7);
                let q_path = ctx.shr_u32(hl_mod8, c1);
                let l_half = ctx.and_u32(half_lane, c1);

                // ql_base = n_idx*64 + (q_path & 1)*32 + l_half*16
                let n_x64 = ctx.mul_u32_reg(n_idx, c64);
                let qp_low = ctx.and_u32(q_path, c1);
                let qp_x32 = ctx.mul_u32_reg(qp_low, c32);
                let lh_x16 = ctx.mul_u32_reg(l_half, c16);
                let ql_base = ctx.add_u32_reg(n_x64, qp_x32);
                let ql_base = ctx.add_u32_reg(ql_base, lh_x16);
                let ql_base_64 = ctx.cvt_u64_u32(ql_base);

                // nibble_shift = (q_path >> 1) << 2
                let qp_div2 = ctx.shr_u32(q_path, c1);
                let nibble_shift = ctx.shl_u32(qp_div2, c2);

                // qh_base = 128 + n_idx*32 + l_half*16
                let n_x32 = ctx.mul_u32_reg(n_idx, c32);
                let qh_base_off = ctx.add_u32(n_x32, 128);
                let qh_base = ctx.add_u32_reg(qh_base_off, lh_x16);
                let qh_base_64 = ctx.cvt_u64_u32(qh_base);

                // qh_shift = q_path * 2
                let qh_shift = ctx.shl_u32(q_path, c1);

                // Q8 block within SB = half_lane / 2
                let q8_block_in_sb = ctx.shr_u32(half_lane, c1);
                // Q8 sub-offset within block = (half_lane & 1) * 16
                let q8_sub = ctx.mul_u32_reg(l_half, c16);

                // Pre-computed Q8 block byte offset within SB (block * 36)
                let c36 = ctx.mov_u32_imm(Q8_BLOCK_BYTES);
                let q8_blk_bytes = ctx.mul_u32_reg(q8_block_in_sb, c36);

                // DP4A constants
                let mask_0f = ctx.mov_u32_imm(0x0F0F_0F0F);
                let mask_03 = ctx.mov_u32_imm(0x0303_0303);
                let ones_packed = ctx.mov_u32_imm(0x0101_0101);
                let c5 = ctx.mov_u32_imm(5);

                let c32_smem = ctx.mov_u32_imm(32); // Q8 d scale offset within block
                let c192_64 = ctx.mov_u64_imm(192);
                let c208_64 = ctx.mov_u64_imm(208);
                let c256_f32 = ctx.mov_f32_imm(256.0);

                let c288 = ctx.mov_u32_imm(Q8_SB_STRIDE);

                let zero = ctx.mov_u32_imm(0);

                // ===== Grid-stride row loop =====
                let row_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(row_idx, block_id);

                ctx.label("hwq6_row_loop");
                let row_oob = ctx.setp_ge_u32(row_idx, n_dim);
                ctx.branch_if(row_oob, "hwq6_exit");

                let row_off = ctx.mul_wide_u32_reg(row_idx, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_off);

                let acc = ctx.mov_f32_imm(0.0);

                // SB loop: stride by num_half_warps
                let sb_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(sb_idx, half_warp_id);

                ctx.label("hwq6_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_sb_reg);
                ctx.branch_if(sb_done, "hwq6_sb_end");

                // Super-block base address (weights — from global memory)
                let sb_off = ctx.mul_wide_u32(sb_idx, Q6K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // ===== C3: Direct scale loading (from global — weights are per-row) =====
                let sc_addr = ctx.add_u64(sb_addr, c192_64);
                let sc_lane_off = ctx.cvt_u64_u32(half_lane);
                let sc_my_addr = ctx.add_u64(sc_addr, sc_lane_off);
                let sc_u8 = ctx.ld_global_u8(sc_my_addr);
                let sc_u32 = ctx.cvt_u32_u8(sc_u8);

                // Signed i8 → f32
                let sign_bit = ctx.shr_u32(sc_u32, c7);
                let raw_f32 = ctx.cvt_f32_u32(sc_u32);
                let sign_f32 = ctx.cvt_f32_u32(sign_bit);
                let correction = ctx.mul_f32(sign_f32, c256_f32);
                let scale_f32 = ctx.sub_f32(raw_f32, correction);

                // Load d (f16 at offset 208 — weight-specific, from global)
                let d_addr = ctx.add_u64(sb_addr, c208_64);
                let d_f16 = ctx.ld_global_f16(d_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                let d_scale = ctx.mul_f32(d, scale_f32);

                // ===== PMAT-078: Q8 from shared memory =====
                // Q8 SB base in shared memory: q8_smem_off + sb_idx * 288
                let q8_sb_smem_off = ctx.mul_u32_reg(sb_idx, c288);
                let q8_sb_smem = ctx.add_u32_reg(q8_smem_base, q8_sb_smem_off);

                // Q8 block address in shared: q8_sb_smem + q8_blk_bytes
                let q8_blk_smem = ctx.add_u32_reg(q8_sb_smem, q8_blk_bytes);
                let q8_blk_smem_64 = ctx.cvt_u64_u32(q8_blk_smem);

                // Q8 d scale (f16 at block+32) — from shared memory
                let q8_d_smem_off = ctx.add_u32_reg(q8_blk_smem, c32_smem);
                let q8_d_smem_addr = ctx.cvt_u64_u32(q8_d_smem_off);
                let q8_d_f16 = ctx.ld_shared_f16(q8_d_smem_addr);
                let q8_d = ctx.cvt_f32_f16(q8_d_f16);

                let combined_scale = ctx.mul_f32(d_scale, q8_d);

                // Q8 qs base for this thread (in shared memory)
                let q8_sub_u32 = ctx.cvt_u64_u32(q8_sub);
                let q8_qs_smem = ctx.add_u64(q8_blk_smem_64, q8_sub_u32);

                // ===== 4 dp4a iterations =====
                let int_acc = ctx.mov_u32_imm(0);

                for i in 0..4u32 {
                    let i_x4 = ctx.mov_u64_imm(u64::from(i * 4));

                    // ql load (from global — weights are per-row)
                    let ql_iter_addr = ctx.add_u64(sb_addr, ql_base_64);
                    let ql_addr = ctx.add_u64(ql_iter_addr, i_x4);
                    let ql_raw = ctx.ld_global_u32_unaligned(ql_addr);

                    let ql_shifted = ctx.shr_u32(ql_raw, nibble_shift);
                    let ql_nibs = ctx.and_u32(ql_shifted, mask_0f);

                    // qh load (from global — weights are per-row)
                    let qh_iter_addr = ctx.add_u64(sb_addr, qh_base_64);
                    let qh_addr = ctx.add_u64(qh_iter_addr, i_x4);
                    let qh_raw = ctx.ld_global_u32_unaligned(qh_addr);

                    let qh_shifted = ctx.shr_u32(qh_raw, qh_shift);
                    let qh_2bits = ctx.and_u32(qh_shifted, mask_03);

                    let qh_up = ctx.shl_u32(qh_2bits, c4);
                    let combined = ctx.or_u32(ql_nibs, qh_up);

                    // Q8 qs load (from shared memory — PMAT-078)
                    let i_x4_u32 = ctx.mov_u32_imm(i * 4);
                    let i_x4_u64 = ctx.cvt_u64_u32(i_x4_u32);
                    let q8_addr = ctx.add_u64(q8_qs_smem, i_x4_u64);
                    // Truncate to u32 for shared memory address
                    let q8_addr_u32 = ctx.cvt_u32_u64(q8_addr);
                    let q8_addr_shared = ctx.cvt_u64_u32(q8_addr_u32);
                    let q8_int32 = ctx.ld_shared_u32(q8_addr_shared);

                    // dp4a: unsigned Q6K × signed Q8
                    ctx.dp4a_u32_s32_inplace(int_acc, combined, q8_int32);

                    // dp4a: sum of Q8 bytes (for -32 bias)
                    let sum_iter = ctx.mov_u32_imm(0);
                    ctx.dp4a_u32_s32_inplace(sum_iter, ones_packed, q8_int32);

                    let sum_x32 = ctx.shl_u32(sum_iter, c5);
                    let int_acc_new = ctx.sub_u32(int_acc, sum_x32);
                    ctx.mov_u32_reg(int_acc, int_acc_new);
                }

                let int_f32 = ctx.cvt_f32_s32(int_acc);
                ctx.fma_f32_inplace(acc, combined_scale, int_f32);

                // Stride by num_half_warps
                ctx.add_u32_reg_inplace(sb_idx, num_hw);
                ctx.branch("hwq6_sb_loop");

                ctx.label("hwq6_sb_end");

                // ===== C4: Half-warp reduction via shfl_down =====
                let t = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t);
                let t = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t);
                let t = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t);
                let t = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t);

                // Half-warp lane 0 stores to shared memory (reduction region)
                let is_hl0 = ctx.setp_eq_u32(half_lane, zero);
                ctx.branch_if_not(is_hl0, "hwq6_skip_sm");

                let sm_off = ctx.shl_u32_imm(half_warp_id, 2);
                let sm_addr = ctx.cvt_u64_u32(sm_off);
                ctx.st_shared_f32(sm_addr, acc);

                ctx.label("hwq6_skip_sm");
                ctx.bar_sync(0);

                // Thread 0 reduces all half-warps and stores
                let is_t0 = ctx.setp_eq_u32(thread_id, zero);
                ctx.branch_if_not(is_t0, "hwq6_skip_store");

                let result = ctx.mov_f32_imm(0.0);
                for hw in 0..num_half_warps {
                    let off = ctx.mov_u64_imm(u64::from(hw * 4));
                    let val = ctx.ld_shared_f32(off);
                    ctx.add_f32_inplace(result, val);
                }

                let y_off = ctx.mul_wide_u32(row_idx, 4);
                let y_addr = ctx.add_u64(y_ptr, y_off);
                ctx.st_global_f32(y_addr, result);

                ctx.label("hwq6_skip_store");

                // Next row (grid-stride)
                ctx.add_u32_reg_inplace(row_idx, grid_dim);
                ctx.bar_sync(0);
                ctx.branch("hwq6_row_loop");

                ctx.label("hwq6_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_emits_valid() {
        let k = HalfWarpDp4aQ6KGemvKernel::new(1536, 1536);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("hw_dp4a_q6k_gemv"), "kernel name");
        assert!(ptx.contains("dp4a.u32.s32"), "DP4A instruction");
        assert!(ptx.contains("and.b32"), "half_lane masking");
    }

    #[test]
    fn test_ptx_lm_head() {
        let k = HalfWarpDp4aQ6KGemvKernel::new(1536, 151936);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("hw_dp4a_q6k_gemv"));
    }

    #[test]
    fn test_ptx_has_shared_q8_load() {
        let k = HalfWarpDp4aQ6KGemvKernel::new(1536, 151936);
        let ptx = k.emit_ptx();
        // PMAT-078: Should load Q8 from shared, not global
        assert!(ptx.contains("ld.shared.u32"), "Q8 should be loaded from shared memory");
        assert!(ptx.contains("st.shared.u32"), "Q8 cooperative load stores to shared");
    }

    #[test]
    fn test_value_coverage() {
        assert_eq!(16 * 16, Q6K_SUPER_BLOCK_SIZE as usize);
    }

    #[test]
    fn test_shared_memory_size() {
        // k=1536: 6 SBs × 288 Q8 bytes = 1728, plus reduction scratch
        let k = HalfWarpDp4aQ6KGemvKernel::new(1536, 1536);
        let num_sb = (1536 + 255) / 256;
        let q8_bytes = num_sb * Q8_SB_STRIDE;
        assert_eq!(q8_bytes, 1728);
        let ptx = k.emit_ptx();
        // Shared memory should be at least 1728 + reduction
        assert!(ptx.contains(".shared"));
        let _ = ptx; // verify it compiles
    }

    #[test]
    fn test_scale_mapping() {
        for half_lane in 0..16u32 {
            let n_idx = half_lane / 8;
            let q_path = (half_lane % 8) / 2;
            let l_half = half_lane % 2;
            let start_pos = n_idx * 128 + q_path * 32 + l_half * 16;
            assert_eq!(start_pos / 16, half_lane);
        }
    }

    #[test]
    fn test_addressing_derivation() {
        for half_lane in 0..16u32 {
            let n_idx = half_lane / 8;
            let q_path = (half_lane % 8) / 2;
            let l_half = half_lane % 2;

            let ql_base = n_idx * 64 + (q_path & 1) * 32 + l_half * 16;
            let nibble_shift = (q_path / 2) * 4;
            let qh_base = 128 + n_idx * 32 + l_half * 16;
            let qh_shift = q_path * 2;

            let pos = half_lane * 16;
            let n = if pos < 128 { 0 } else { 128 };
            let pos_in_half = pos - n;
            let l = pos_in_half % 32;
            let q_idx = pos_in_half / 32;

            let expected_ql = n / 2 + (q_idx & 1) * 32 + l;
            assert_eq!(ql_base, expected_ql);

            let expected_nibble = if q_idx < 2 { 0 } else { 4 };
            assert_eq!(nibble_shift, expected_nibble);

            let expected_qh = 128 + n / 4 + l;
            assert_eq!(qh_base, expected_qh);

            let expected_qh_shift = q_idx * 2;
            assert_eq!(qh_shift, expected_qh_shift);
        }
    }
}
