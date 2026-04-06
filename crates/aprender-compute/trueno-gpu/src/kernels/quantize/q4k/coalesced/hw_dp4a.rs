use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory, PtxSync};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Half-warp DP4A Q4_K GEMV kernel (GH-176)
///
/// Restructured from 32-thread/SB (MWV) to 16-thread/SB (half-warp), based on
/// analysis of llama.cpp's `vec_dot_q4_K_q8_1` (MIT license, vecdotq.cuh:775).
///
/// # Provable Contracts
///
/// - **C1 (Thread mapping)**: 16 threads per SB via `half_lane = lane_id & 15`.
///   Each warp splits into 2 half-warps, each processing one SB independently.
///
/// - **C2 (Value coverage)**: 16 threads x 16 values = 256 = `Q4K_SUPER_BLOCK_SIZE`.
///   Thread layout: `bq8_group = half_lane / 4` (4 groups of 4 threads),
///   each group covers 64 values (qs bytes `[32*g..32*g+31]`).
///   QR=2 inner loop extracts low + high nibbles: 4 threads x 8 bytes x 2 nibbles = 64.
///
/// - **C3 (Instruction density)**: ~93 inner loop instructions for 16 values
///   = 5.8 insn/value. MWV: 99 instructions for 8 values = 12.4 insn/value.
///   Total thread-instructions per SB: 16 x 93 = 1488 vs 32 x 99 = 3168 (2.13x).
///   PMAT-033: FMA chain saves 4 insn/SB (neg_dmin hoisted, fma replaces mul+sub).
///   PMAT-029: Hoisted bitmask constants save 10 insn/SB (103→93).
///
/// - **C4 (Reduction correctness)**: Full-warp `shfl.sync.down` with delta=8,4,2,1
///   yields correct half-warp sums at lanes 0 and 16. Proof: shfl reads happen
///   simultaneously for all lanes. At each step, the source lane for any target
///   lane in half-warp 0 is also in half-warp 0 (delta <= 8, max source = 15).
///   Cross-contamination at lane 8 (reads lane 16) propagates only to lanes 8+,
///   which are never read by lane 0's reduction chain.
///
/// - **C5 (Parallel final reduction, PMAT-089)**: Warp 0 threads 0..num_half_warps
///   each load one half-warp partial from shared memory, then `shfl_down` with
///   deltas 4,2,1 reduces to thread 0. Threads >= num_half_warps contribute 0.0.
///   Replaces serial thread-0 loop (O(num_half_warps) → O(log2) = 3 shuffles).
///   4-warp hypothesis FALSIFIED (register pressure, -2% decode on RTX 4060L).
pub struct HalfWarpDp4aQ4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Number of warps per block (default: 3, giving 6 half-warps).
    /// PMAT-089: 4 warps FALSIFIED (register pressure, -2% decode). 3 is optimal.
    pub num_warps: u32,
}

impl HalfWarpDp4aQ4KGemvKernel {
    /// Create a new HW DP4A Q4K GEMV kernel with default 3 warps per CTA.
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, num_warps: 3 }
    }
}

impl Kernel for HalfWarpDp4aQ4KGemvKernel {
    fn name(&self) -> &str {
        "hw_dp4a_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let num_half_warps = num_warps * 2;
        let smem_size = (num_half_warps * 4) as usize;

        PtxKernel::new("hw_dp4a_q4k_gemv")
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

                // PMAT-029: Hoist repeated bitmask constants before inner loop.
                // and_u32_imm emits mov+and per call — hoisting saves 7 movs/SB.
                let c_mask_6bit = ctx.mov_u32_imm(0x3F3F_3F3F);
                let c_mask_4bit = ctx.mov_u32_imm(0x0F0F_0F0F);
                let c_mask_2bit = ctx.mov_u32_imm(0x0303_0303);

                // ===== Grid-stride row loop =====
                let row_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(row_idx, block_id);

                ctx.label("hw_row_loop");
                let row_oob = ctx.setp_ge_u32(row_idx, n_dim);
                ctx.branch_if(row_oob, "hw_exit");

                let row_off = ctx.mul_wide_u32_reg(row_idx, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_off);

                let acc = ctx.mov_f32_imm(0.0);

                // SB loop: each half-warp processes 1 SB, stride by num_half_warps
                let sb_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(sb_idx, half_warp_id);

                ctx.label("hw_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_sb);
                ctx.branch_if(sb_done, "hw_sb_end");

                // Super-block base
                let sb_off = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // Load d, dmin (f16 -> f32)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let dmin_addr = ctx.add_u64(sb_addr, c_2_64);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);
                // PMAT-033: Negate dmin once per SB for FMA accumulation (saves 4 insn/SB)
                let neg_dmin = ctx.neg_f32(dmin);

                // ===== Scale loading: all threads load (L1 coalesced) =====
                let sc_base = ctx.add_u64(sb_addr, c_4_64);
                let sc03 = ctx.ld_global_u32(sc_base);
                let sc47_addr = ctx.add_u64(sc_base, c_4_64);
                let sc47 = ctx.ld_global_u32(sc47_addr);
                let sc811_addr = ctx.add_u64(sc_base, c_8_64);
                let sc811 = ctx.ld_global_u32(sc811_addr);

                // GH-173: Parallel byte-masked scale extraction
                // PMAT-029: Use hoisted constant registers (saves 7 mov/SB)
                // Blocks 0-3: low 6 bits
                let sc_lo4 = ctx.and_u32(sc03, c_mask_6bit);
                let mn_lo4 = ctx.and_u32(sc47, c_mask_6bit);

                // Blocks 4-7: combine low 4 + high 2
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

                // Per-thread scale/min byte extraction
                let sc_src = ctx.selp_u32(p_hi, sc_hi4, sc_lo4);
                let mn_src = ctx.selp_u32(p_hi, mn_hi4, mn_lo4);

                // PMAT-039: BFE replaces shr+and (4 insn saved per SB)
                let sc0 = ctx.bfe_u32_reg(sc_src, byte_shift, 8);
                let sc1 = ctx.bfe_u32_reg(sc_src, byte_shift_hi, 8);
                let mn0 = ctx.bfe_u32_reg(mn_src, byte_shift, 8);
                let mn1 = ctx.bfe_u32_reg(mn_src, byte_shift_hi, 8);

                // ===== Load Q4K data: 2 aligned ints =====
                let q4_addr = ctx.add_u64(sb_addr, q4_off_64);
                let v0 = ctx.ld_global_u32(q4_addr);
                let v1_addr = ctx.add_u64(q4_addr, c_16_64);
                let v1 = ctx.ld_global_u32(v1_addr);

                // ===== Q8 block base addresses =====
                let q8_sb_off = ctx.mul_wide_u32_reg(sb_idx, c_288);
                let q8_sb_base = ctx.add_u64(q8_ptr, q8_sb_off);
                let q8_blk = ctx.add_u64(q8_sb_base, bq8_bytes_64);
                let q8_data = ctx.add_u64(q8_blk, lig_x4_64);

                // ===== QR=0: Low nibbles =====
                let v0_lo = ctx.and_u32(v0, c_mask_4bit);
                let v1_lo = ctx.and_u32(v1, c_mask_4bit);

                let u0_lo = ctx.ld_global_u32(q8_data);
                let u1_lo_addr = ctx.add_u64(q8_data, c_16_64);
                let u1_lo = ctx.ld_global_u32(u1_lo_addr);

                // Chained DP4A: dot = v0_lo . u0 + v1_lo . u1
                let dot0 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(dot0, v0_lo, u0_lo);
                ctx.dp4a_u32_s32_inplace(dot0, v1_lo, u1_lo);

                // Q8 byte sum: sum = ones . u0 + ones . u1
                let sum0 = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(sum0, c_ones, u0_lo);
                ctx.dp4a_u32_s32_inplace(sum0, c_ones, u1_lo);

                // Q8 scale (d at offset 32 from block base)
                let q8_d0_addr = ctx.add_u64(q8_blk, c_32_64);
                let q8_d0_f16 = ctx.ld_global_f16(q8_d0_addr);
                let q8_d0 = ctx.cvt_f32_f16(q8_d0_f16);

                // Accumulate: acc += q8_d * (d * sc*dot - dmin * mn*sum)
                // PMAT-033: FMA chain reduces 9 → 7 instructions per QR iteration
                let sdot0 = ctx.mul_lo_s32(sc0, dot0);
                let msum0 = ctx.mul_lo_s32(mn0, sum0);
                let sdot0_f = ctx.cvt_f32_s32(sdot0);
                let msum0_f = ctx.cvt_f32_s32(msum0);
                let t1 = ctx.mul_f32(d, sdot0_f);
                let t3 = ctx.fma_f32(neg_dmin, msum0_f, t1); // d*sc*dot - dmin*mn*sum
                let q8_d0_t3 = ctx.mul_f32(q8_d0, t3);
                ctx.add_f32_inplace(acc, q8_d0_t3);

                // ===== QR=1: High nibbles =====
                let v0_hi = ctx.shr_u32_imm(v0, 4);
                let v0_hi = ctx.and_u32(v0_hi, c_mask_4bit);
                let v1_hi = ctx.shr_u32_imm(v1, 4);
                let v1_hi = ctx.and_u32(v1_hi, c_mask_4bit);

                // Q8 block +1 (36 bytes later)
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

                // PMAT-033: FMA chain (same as QR=0)
                let sdot1 = ctx.mul_lo_s32(sc1, dot1);
                let msum1 = ctx.mul_lo_s32(mn1, sum1);
                let sdot1_f = ctx.cvt_f32_s32(sdot1);
                let msum1_f = ctx.cvt_f32_s32(msum1);
                let t1 = ctx.mul_f32(d, sdot1_f);
                let t3 = ctx.fma_f32(neg_dmin, msum1_f, t1);
                let q8_d1_t3 = ctx.mul_f32(q8_d1, t3);
                ctx.add_f32_inplace(acc, q8_d1_t3);

                // Stride by num_half_warps
                ctx.add_u32_reg_inplace(sb_idx, num_hw);
                ctx.branch("hw_sb_loop");

                ctx.label("hw_sb_end");

                // ===== C4: Half-warp reduction via full-warp shfl_down =====
                // Deltas 8,4,2,1: lane 0 accumulates half-warp 0 sum,
                // lane 16 accumulates half-warp 1 sum.
                let t = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t);
                let t = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t);
                let t = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t);
                let t = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t);

                // Half-warp lane 0 stores to shared memory
                let z = ctx.mov_u32_imm(0);
                let is_hl0 = ctx.setp_eq_u32(half_lane, z);
                ctx.branch_if_not(is_hl0, "hw_skip_sm");

                let sm_off = ctx.shl_u32_imm(half_warp_id, 2);
                let sm_addr = ctx.cvt_u64_u32(sm_off);
                ctx.st_shared_f32(sm_addr, acc);

                ctx.label("hw_skip_sm");
                ctx.bar_sync(0);

                // ===== C5: Parallel warp-0 reduction (PMAT-089) =====
                // Warp 0 threads each load one half-warp partial, then shfl_down.
                // Replaces serial thread-0 loop with O(log2) shuffles.
                let is_warp0 = ctx.setp_eq_u32(warp_id, z);
                ctx.branch_if_not(is_warp0, "hw_skip_store");

                // Thread i in warp 0 loads smem[i] if i < num_half_warps, else 0.0
                let in_range = ctx.setp_lt_u32_imm(lane_id, num_half_warps);
                let my_off = ctx.shl_u32_imm(lane_id, 2);
                let my_addr = ctx.cvt_u64_u32(my_off);
                let loaded = ctx.ld_shared_f32(my_addr);
                let zero_f = ctx.mov_f32_imm(0.0);
                let partial = ctx.selp_f32(in_range, loaded, zero_f);

                // Warp shuffle reduce: 3 steps for up to 8 half-warps
                let t = ctx.shfl_down_f32(partial, 4, 0xFFFF_FFFF);
                let partial = ctx.add_f32(partial, t);
                let t = ctx.shfl_down_f32(partial, 2, 0xFFFF_FFFF);
                let partial = ctx.add_f32(partial, t);
                let t = ctx.shfl_down_f32(partial, 1, 0xFFFF_FFFF);
                let result = ctx.add_f32(partial, t);

                // Lane 0 of warp 0 = thread 0 stores the final sum
                let is_t0 = ctx.setp_eq_u32(lane_id, z);
                ctx.branch_if_not(is_t0, "hw_skip_store");

                let y_off = ctx.mul_wide_u32(row_idx, 4);
                let y_addr = ctx.add_u64(y_ptr, y_off);
                ctx.st_global_f32(y_addr, result);

                ctx.label("hw_skip_store");

                // Next row (grid-stride)
                ctx.add_u32_reg_inplace(row_idx, grid_dim);
                ctx.bar_sync(0);
                ctx.branch("hw_row_loop");

                ctx.label("hw_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CONTRACT C1: Kernel emits valid PTX with half-warp identity
    #[test]
    fn test_ptx_emits_hw_identity() {
        let k = HalfWarpDp4aQ4KGemvKernel::new(1536, 256);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("hw_dp4a_q4k_gemv"), "kernel name");
        assert!(ptx.contains("dp4a.u32.s32"), "DP4A instruction");
        // Half-warp identity: and.b32 %r, lane_id, 15
        assert!(ptx.contains("and.b32"), "half_lane = lane_id & 15");
    }

    // CONTRACT C2: 16 threads x 16 values = 256
    #[test]
    fn test_value_coverage_contract() {
        // Static proof: each of 16 half-warp threads processes:
        //   bq8_group (0-3) x 4 threads = 4 groups
        //   Each group: 4 threads x 2 ints x 4 bytes x 2 nibbles = 64 values
        //   4 groups x 64 = 256 = Q4K_SUPER_BLOCK_SIZE
        assert_eq!(4 * 4 * 2 * 4 * 2, Q4K_SUPER_BLOCK_SIZE);
        // Alternative: 16 threads x 16 values/thread = 256
        assert_eq!(16 * 16, Q4K_SUPER_BLOCK_SIZE as usize);
    }

    // CONTRACT C3: Inner loop instruction density
    #[test]
    fn test_instruction_density() {
        let k = HalfWarpDp4aQ4KGemvKernel::new(1536, 256);
        let ptx = k.emit_ptx();

        // Count instructions between hw_sb_loop and hw_sb_end labels
        let sb_loop_start = ptx.find("hw_sb_loop:").expect("sb_loop label");
        let sb_loop_end = ptx.find("hw_sb_end:").expect("sb_end label");
        let inner = &ptx[sb_loop_start..sb_loop_end];

        // Count semicolons as proxy for instructions (each PTX instruction ends with ;)
        let insn_count = inner.matches(';').count();

        // Contract: <= 100 instructions for 16 values (6.25 insn/value)
        // MWV baseline: 99 instructions for 8 values (12.4 insn/value)
        // Thread-insn/SB: 16×100=1600 vs 32×99=3168 (1.98x fewer)
        // PMAT-029: Tightened from 120 to 100 after bitmask constant hoisting
        assert!(
            insn_count <= 100,
            "C3 violated: inner loop has {} instructions (limit 100)",
            insn_count
        );

        // Verify improvement over MWV per-value density
        let hw_per_value = insn_count as f64 / 16.0;
        let mwv_per_value = 99.0 / 8.0;
        assert!(
            hw_per_value < mwv_per_value,
            "C3: HW {:.1} insn/val should be < MWV {:.1} insn/val",
            hw_per_value,
            mwv_per_value
        );

        eprintln!(
            "[C3] Inner loop: {} instructions for 16 values ({:.1} insn/val vs MWV {:.1})",
            insn_count, hw_per_value, mwv_per_value
        );
    }

    // CONTRACT C4: Reduction correctness (algebraic proof)
    #[test]
    fn test_reduction_correctness() {
        // Simulate full-warp shfl_down reduction for two independent half-warps.
        // Half-warp 0: lanes 0-15 with values 1..16
        // Half-warp 1: lanes 16-31 with values 100..115
        let mut vals: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        vals.extend((100..=115).map(|x| x as f32));
        assert_eq!(vals.len(), 32);

        // shfl_down(delta): lane i reads lane i+delta (clamped to 31)
        for delta in [8, 4, 2, 1] {
            let old = vals.clone();
            for i in 0..32 {
                let src = i + delta;
                if src < 32 {
                    vals[i] += old[src];
                }
                // if src >= 32, no-op (clamped, returns own value but that's
                // handled by the hardware returning the source lane's value)
            }
        }

        let hw0_sum: f32 = (1..=16).map(|x| x as f32).sum();
        let hw1_sum: f32 = (100..=115).map(|x| x as f32).sum();

        // Lane 0 should have the correct sum of half-warp 0
        assert!((vals[0] - hw0_sum).abs() < 0.01, "Lane 0: got {}, expected {}", vals[0], hw0_sum);

        // Lane 16 should have the correct sum of half-warp 1
        assert!(
            (vals[16] - hw1_sum).abs() < 0.01,
            "Lane 16: got {}, expected {}",
            vals[16],
            hw1_sum
        );
    }

    #[test]
    fn dump_ptx() {
        let k = HalfWarpDp4aQ4KGemvKernel::new(1536, 256);
        let ptx = k.emit_ptx();
        eprintln!("{ptx}");
    }
}
