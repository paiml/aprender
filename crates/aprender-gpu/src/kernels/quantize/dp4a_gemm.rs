//! DP4A Q4K×Q8 GEMM Kernel (PMAT-066)
//!
//! Fused dequantization-free GEMM for prefill: C[M,N] = A_q8[M,K] × W_q4k[N,K]^T
//!
//! Operates directly on Q4K nibbles × Q8 activations via DP4A instruction,
//! eliminating FP16 dequantization. 3.56x memory bandwidth reduction vs HGEMM.
//!
//! ## Five-Whys Root Cause (PMAT-066)
//!
//! 1. WHY is prefill TTFT bandwidth-bound? → HGEMM reads FP16 weights (2 B/elem)
//! 2. WHY read FP16? → cuBLAS HGEMM requires FP16 input format
//! 3. WHY use cuBLAS HGEMM? → WMMA approach still dequants Q4K→FP16 in SHMEM (PMAT-045)
//! 4. WHY dequant to FP16? → WMMA requires FP16×FP16→FP32 operands
//! 5. WHY not DP4A? → **This kernel: Q4K nibbles × Q8 int8 via dp4a.u32.s32**
//!
//! ## Architecture
//!
//! ```text
//! Grid: (ceil(N/num_hw), ceil(M/TILE_M))
//! Block: num_warps * 32 threads
//!
//! Each half-warp (16 threads) → 1 N column × TILE_M M rows
//! Q4K weights loaded ONCE per super-block, reused across TILE_M rows
//! No shared memory needed (intra-half-warp shuffle reduction only)
//! ```
//!
//! ## Memory Traffic
//!
//! - Q4K reads: N × K × 0.5625 B/elem (3.56x less than FP16)
//! - Q8 reads: M × K × 1.125 B/elem (Q8_1: 36 B per 32 values)
//! - Output writes: M × N × 4 B/elem (f32)

use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxSync};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// DP4A Q4K×Q8 GEMM kernel for prefill (PMAT-066)
///
/// # Provable Contracts
///
/// - **C1**: Each half-warp (16 threads) processes all K super-blocks for one N column
/// - **C2**: TILE_M accumulators per thread, weight data shared across M rows
/// - **C3**: Intra-half-warp shuffle reduction (no shared memory, no barriers)
/// - **C4**: Output C[m, n] in f32 row-major, bounds-checked for partial M tiles
/// - **C5**: Same DP4A + scale correction as HalfWarpDp4aQ4KGemvKernel (proven correct)
pub struct Dp4aQ4KGemmKernel {
    /// M dimension (number of rows / sequence length)
    pub m: u32,
    /// N dimension (output dimension / number of columns)
    pub n: u32,
    /// K dimension (inner dimension, must be multiple of 256)
    pub k: u32,
    /// Number of warps per block (default: 4 → 8 half-warps → 8 N columns/block)
    pub num_warps: u32,
    /// M-dimension tile size (compile-time unrolled, default: 4)
    pub tile_m: u32,
}

impl Dp4aQ4KGemmKernel {
    /// Create a new DP4A Q4K GEMM kernel.
    ///
    /// Requires pre-quantized Q8_1 activations (use `Q8QuantizeKernel` first).
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            num_warps: 4,
            tile_m: 4,
        }
    }

    /// Set M-dimension tile size (compile-time unrolled).
    #[must_use]
    pub const fn with_tile_m(mut self, tile_m: u32) -> Self {
        self.tile_m = tile_m;
        self
    }

    /// Set number of warps per block.
    #[must_use]
    pub const fn with_num_warps(mut self, num_warps: u32) -> Self {
        self.num_warps = num_warps;
        self
    }
}

impl Kernel for Dp4aQ4KGemmKernel {
    fn name(&self) -> &str {
        "dp4a_q4k_gemm"
    }

    #[allow(clippy::too_many_lines)]
    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let num_half_warps = num_warps * 2;
        let tile_m = self.tile_m;

        PtxKernel::new("dp4a_q4k_gemm")
            .param(PtxType::U64, "y_ptr") // C[M, N] output (f32)
            .param(PtxType::U64, "w_ptr") // Q4K weights W[N, K/256, 144B]
            .param(PtxType::U64, "q8_ptr") // Q8_1 activations A_q8[M, K/32, 36B]
            .param(PtxType::U32, "m_dim")
            .param(PtxType::U32, "n_dim")
            .param(PtxType::U32, "k_dim")
            .build(move |ctx| {
                // ===== Thread identity =====
                let block_x = ctx.special_reg(PtxReg::CtaIdX);
                let block_y = ctx.special_reg(PtxReg::CtaIdY);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);

                // ===== Parameters =====
                let m_dim = ctx.load_param_u32("m_dim");
                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let q8_ptr = ctx.load_param_u64("q8_ptr");

                // ===== K dimension setup =====
                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_sb = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);
                let sb_bytes_reg = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_sb, sb_bytes_reg);

                // ===== Half-warp identity (C1) =====
                let half_lane = ctx.and_u32_imm(lane_id, 15);
                let half_warp_in_warp = ctx.shr_u32_imm(lane_id, 4);
                let warp_x2 = ctx.shl_u32_imm(warp_id, 1);
                let half_warp_id = ctx.add_u32_reg(warp_x2, half_warp_in_warp);

                // ===== N column for this half-warp =====
                let num_hw_imm = ctx.mov_u32_imm(num_half_warps);
                let n_block_off = ctx.mul_u32_reg(block_x, num_hw_imm);
                let n_col = ctx.add_u32_reg(n_block_off, half_warp_id);

                // Bounds check N
                let n_oob = ctx.setp_ge_u32(n_col, n_dim);
                ctx.branch_if(n_oob, "gemm_exit");

                // ===== M base for this tile =====
                let tile_m_imm = ctx.mov_u32_imm(tile_m);
                let m_base = ctx.mul_u32_reg(block_y, tile_m_imm);

                // ===== Q4K row base for this N column =====
                let row_off = ctx.mul_wide_u32_reg(n_col, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_off);

                // ===== Q8 stride per activation row =====
                let c_288 = ctx.mov_u32_imm(288);
                let q8_vec_stride = ctx.mul_u32_reg(num_sb, c_288);

                // ===== Per-thread data mapping (same as HW DP4A GEMV) =====
                let bq8_group = ctx.shr_u32_imm(half_lane, 2);
                let lane_in_group = ctx.and_u32_imm(half_lane, 3);
                let bq8_offset = ctx.shl_u32_imm(bq8_group, 1);

                // Q4K qs offset: 16 + bq8_offset * 16 + lane_in_group * 4
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
                let c_288_u32 = ctx.mov_u32_imm(288);

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

                // ===== C2: TILE_M accumulators (compile-time unrolled) =====
                let mut accs = Vec::with_capacity(tile_m as usize);
                for _ in 0..tile_m {
                    accs.push(ctx.mov_f32_imm(0.0));
                }

                // ===== Super-block loop (sequential, all SBs) =====
                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("gemm_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_sb);
                ctx.branch_if(sb_done, "gemm_sb_end");

                // Super-block base (Q4K weights — loaded ONCE, shared across M)
                let sb_off = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // Load d, dmin (f16 → f32)
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

                // Scale extraction (PMAT-029 hoisted constants)
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

                // ===== Load Q4K nibbles (shared across M) =====
                let q4_addr = ctx.add_u64(sb_addr, q4_off_64);
                let v0 = ctx.ld_global_u32(q4_addr);
                let v1_addr = ctx.add_u64(q4_addr, c_16_64);
                let v1 = ctx.ld_global_u32(v1_addr);

                let v0_lo = ctx.and_u32(v0, c_mask_4bit);
                let v1_lo = ctx.and_u32(v1, c_mask_4bit);
                let v0_hi = ctx.shr_u32_imm(v0, 4);
                let v0_hi = ctx.and_u32(v0_hi, c_mask_4bit);
                let v1_hi = ctx.shr_u32_imm(v1, 4);
                let v1_hi = ctx.and_u32(v1_hi, c_mask_4bit);

                // ===== Per-row Q8 loading + DP4A (C2: compile-time unrolled) =====
                let q8_sb_off_base = ctx.mul_wide_u32_reg(sb_idx, c_288_u32);

                for mi in 0..tile_m {
                    // Bounds check: skip if m_base + mi >= m_dim
                    let m_row = ctx.add_u32(m_base, mi);
                    let m_oob = ctx.setp_ge_u32(m_row, m_dim);
                    let skip_label = format!("gemm_skip_m{mi}");
                    ctx.branch_if(m_oob, &skip_label);

                    // Q8 base for row m_row
                    let q8_m_off = ctx.mul_wide_u32_reg(m_row, q8_vec_stride);
                    let q8_m_base = ctx.add_u64(q8_ptr, q8_m_off);
                    let q8_sb_base = ctx.add_u64(q8_m_base, q8_sb_off_base);
                    let q8_blk = ctx.add_u64(q8_sb_base, bq8_bytes_64);
                    let q8_data = ctx.add_u64(q8_blk, lig_x4_64);

                    // === QR=0: Low nibbles ===
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

                    // === QR=1: High nibbles ===
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

                    ctx.label(&skip_label);
                }

                // Next super-block
                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("gemm_sb_loop");

                ctx.label("gemm_sb_end");

                // ===== C3: Half-warp shuffle reduction =====
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

                // ===== C4: Lane 0 of each half-warp stores results =====
                let z = ctx.mov_u32_imm(0);
                let is_hl0 = ctx.setp_eq_u32(half_lane, z);
                ctx.branch_if_not(is_hl0, "gemm_exit");

                for mi in 0..tile_m {
                    let m_row = ctx.add_u32(m_base, mi);
                    let m_oob = ctx.setp_ge_u32(m_row, m_dim);
                    let store_skip = format!("gemm_store_skip{mi}");
                    ctx.branch_if(m_oob, &store_skip);

                    // C[m_row, n_col] = byte_off((m_row * n_dim + n_col) * 4)
                    let row_elem = ctx.mul_u32_reg(m_row, n_dim);
                    let elem_idx = ctx.add_u32_reg(row_elem, n_col);
                    let byte_off = ctx.mul_wide_u32(elem_idx, 4);
                    let y_addr = ctx.add_u64(y_ptr, byte_off);
                    ctx.st_global_f32(y_addr, accs[mi as usize]);

                    ctx.label(&store_skip);
                }

                ctx.label("gemm_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_name() {
        let k = Dp4aQ4KGemmKernel::new(100, 1536, 1536);
        assert_eq!(k.name(), "dp4a_q4k_gemm");
    }

    #[test]
    fn test_ptx_emits() {
        let k = Dp4aQ4KGemmKernel::new(100, 1536, 1536);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("dp4a_q4k_gemm"), "kernel name present");
        assert!(ptx.contains("dp4a.u32.s32"), "DP4A instruction present");
        assert!(
            ptx.contains("shfl.sync.down"),
            "half-warp shuffle reduction"
        );
    }

    #[test]
    fn test_no_shared_memory() {
        let k = Dp4aQ4KGemmKernel::new(100, 1536, 1536);
        let ptx = k.emit_ptx();
        // No shared memory allocation (no bar.sync needed)
        assert!(
            !ptx.contains("bar.sync"),
            "no barriers needed (each HW independent)"
        );
    }

    #[test]
    fn test_barrier_safety() {
        let k = Dp4aQ4KGemmKernel::new(100, 1536, 1536);
        k.validate_barrier_safety()
            .expect("barrier safety validation");
    }

    #[test]
    fn test_2d_grid_dispatch() {
        let k = Dp4aQ4KGemmKernel::new(100, 1536, 1536);
        let ptx = k.emit_ptx();
        // Uses both ctaid.x (N tiling) and ctaid.y (M tiling)
        assert!(ptx.contains("ctaid.x"), "uses blockIdx.x for N");
        assert!(ptx.contains("ctaid.y"), "uses blockIdx.y for M");
    }

    #[test]
    fn test_tile_m_8() {
        let k = Dp4aQ4KGemmKernel::new(200, 1536, 1536).with_tile_m(8);
        let ptx = k.emit_ptx();
        assert!(!ptx.is_empty());
        assert!(ptx.contains("dp4a.u32.s32"));
    }

    #[test]
    fn test_ffn_dimensions() {
        // Qwen 1.5B FFN dimensions: K=1536, N=8960
        let k = Dp4aQ4KGemmKernel::new(50, 8960, 1536);
        let ptx = k.emit_ptx();
        assert!(!ptx.is_empty());
    }
}
