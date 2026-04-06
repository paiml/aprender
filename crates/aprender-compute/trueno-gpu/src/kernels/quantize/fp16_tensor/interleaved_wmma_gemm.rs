//! PMAT-091: Coalesced WMMA Q4K GEMM reading column-interleaved weights
//!
//! Same architecture as [`MultiWarpTensorCoreQ4KGemmKernel`] (4 warps, 32×32
//! output tiles, WMMA 16×16×16), but Phase 2 (B loading) rewired for the
//! interleaved weight layout from `q4k::interleaved::repack_q4k_interleaved`.
//!
//! Key coalescing change: thread-to-element mapping uses column-first order
//! (`col = tid % 32`) so adjacent threads in a warp access adjacent columns.
//! With byte-interleaved qs, 16 adjacent columns map to 16 consecutive bytes
//! → single 128-byte cache line transaction.
//!
//! Grid: (ceil(N/32), ceil(M/32)), Block: 128 threads (4 warps)
//! SHMEM: 2048 bytes (A[32×16 FP16] + B[16×32 FP16])
//!
//! Expected bandwidth improvement: 5-8× for B loads vs uncoalesced Q4K access.

use crate::kernels::quantize::q4k::interleaved::{
    INTERLEAVED_TILE_BYTES, TILE_DMIN_OFFSET, TILE_D_OFFSET, TILE_QS_OFFSET, TILE_SCALES_OFFSET,
};
use crate::kernels::quantize::Q4K_SUPER_BLOCK_SIZE;
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, WmmaLayout};

/// Coalesced WMMA Q4K GEMM kernel reading interleaved weights.
///
/// Drop-in replacement for [`MultiWarpTensorCoreQ4KGemmKernel`] when weights
/// have been repacked with `repack_q4k_interleaved`. Same grid/block config.
///
/// Input: FP32 activations [M, K]
/// Weights: Interleaved Q4K [N_tiles × (K/256) × 2304B] tiles
/// Output: FP32 [M, N]
#[derive(Debug, Clone)]
pub struct InterleavedWmmaQ4KGemmKernel {
    /// Batch size (M) — sequence length for prefill, batch size for decode
    pub m: u32,
    /// Output dimension (N) — hidden_dim or intermediate_dim
    pub n: u32,
    /// Input dimension (K) — must be multiple of 256 for Q4K super-blocks
    pub k: u32,
}

impl InterleavedWmmaQ4KGemmKernel {
    /// Create kernel for dimensions M×K×N (K must be multiple of 256)
    #[must_use]
    pub fn new(m: u32, k: u32, n: u32) -> Self {
        Self { m, n, k }
    }

    /// Number of Q4K super-blocks along K dimension
    #[must_use]
    pub fn num_super_blocks(&self) -> u32 {
        (self.k + Q4K_SUPER_BLOCK_SIZE - 1) / Q4K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for InterleavedWmmaQ4KGemmKernel {
    fn name(&self) -> &str {
        "interleaved_wmma_q4k_gemm"
    }

    #[allow(clippy::too_many_lines)]
    fn build_ptx(&self) -> PtxKernel {
        let n_const = self.n;
        let k_const = self.k;
        let num_sb = self.num_super_blocks();
        let n_k_tiles = k_const / 16;

        // SHMEM: A[32×16 FP16] + B[16×32 FP16] = 1024 + 1024 = 2048 bytes
        let smem_a_size: u32 = 32 * 16 * 2; // 1024 bytes
        let smem_b_offset: u32 = smem_a_size; // B starts at byte 1024
        let smem_bytes = (smem_a_size + 16 * 32 * 2) as usize; // 2048 total

        PtxKernel::new("interleaved_wmma_q4k_gemm")
            .max_regs(96)
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_interleaved_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m_param")
            .param(PtxType::U32, "n_param")
            .param(PtxType::U32, "k_param")
            .shared_memory(smem_bytes)
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

                // Constants
                let c_0 = ctx.mov_u32_imm(0);
                let c_1 = ctx.mov_u32_imm(1);
                let c_2 = ctx.mov_u32_imm(2);
                let c_4_u32 = ctx.mov_u32_imm(4);
                let c_8_u32 = ctx.mov_u32_imm(8);
                let c_15 = ctx.mov_u32_imm(15);
                let c_16 = ctx.mov_u32_imm(16);
                let c_31 = ctx.mov_u32_imm(31);
                let c_32 = ctx.mov_u32_imm(32);
                let c_128 = ctx.mov_u32_imm(128);
                let c_255 = ctx.mov_u32_imm(255);

                // Block tile origins (32×32 per block)
                let tile_col = ctx.mul_u32_reg(ctaid_x, c_32);
                let tile_row = ctx.mul_u32_reg(ctaid_y, c_32);

                let m_param = ctx.load_param_u32("m_param");
                let n_param = ctx.load_param_u32("n_param");
                let k_param = ctx.load_param_u32("k_param");

                // Bounds — skip entire out-of-range blocks
                let row_oob = ctx.setp_ge_u32(tile_row, m_param);
                ctx.branch_if(row_oob, "exit");
                let col_oob = ctx.setp_ge_u32(tile_col, n_param);
                ctx.branch_if(col_oob, "exit");

                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_interleaved_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Warp-level dispatch: warp_id = tid / 32, lane = tid % 32
                let shift_5 = ctx.mov_u32_imm(5);
                let warp_id = ctx.shr_u32(tid, shift_5);
                let warp_row = ctx.shr_u32(warp_id, c_1); // 0 or 1
                let warp_col = ctx.and_u32(warp_id, c_1); // 0 or 1

                // SHMEM bases
                let smem_a_base = c_0;
                let smem_b_base = ctx.mov_u32_imm(smem_b_offset);

                // Initialize WMMA FP32 accumulators (per-warp)
                let frag_c = ctx.wmma_init_c_zero();

                let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);
                let k_tile_idx = ctx.mov_u32_imm(0);
                let num_sb_reg = ctx.mov_u32_imm(num_sb);

                // Clamp limits
                let m_minus_1 = ctx.sub_u32_reg(m_param, c_1);
                let n_minus_1 = ctx.sub_u32_reg(n_param, c_1);
                let k_minus_1 = ctx.sub_u32_reg(k_param, c_1);

                // Shift/mask constants
                let shift_4 = c_4_u32;
                let shift_8 = c_8_u32;
                let mask_0f = c_15;
                let mask_3f = ctx.mov_u32_imm(0x3F);
                let mask_03 = ctx.mov_u32_imm(0x03);
                let mask_0f_4 = ctx.mov_u32_imm(0x0F);
                let c_6_u32 = ctx.mov_u32_imm(6);
                let c_3_u32 = ctx.mov_u32_imm(3);

                // Interleaved tile constants
                let tile_qs_off = ctx.mov_u32_imm(TILE_QS_OFFSET as u32);
                let tile_d_off = ctx.mov_u32_imm(TILE_D_OFFSET as u32);
                let tile_dmin_off = ctx.mov_u32_imm(TILE_DMIN_OFFSET as u32);
                let tile_scales_off = ctx.mov_u32_imm(TILE_SCALES_OFFSET as u32);
                let c_12 = ctx.mov_u32_imm(12);

                // A loading: same element mapping as multi-warp kernel
                // 128 threads × 4 elements = 512 A elements (32×16)
                let my_start_a = ctx.mul_u32_reg(tid, c_4_u32); // 0, 4, 8, ..., 508

                // ===== K-tile loop =====
                ctx.label("k_tile_loop");
                let k_done = ctx.setp_ge_u32(k_tile_idx, n_k_tiles_reg);
                ctx.branch_if(k_done, "k_tile_end");

                let k_offset = ctx.mul_u32_reg(k_tile_idx, c_16);

                // ====== PHASE 1: Load A[32×16] FP32→FP16 SHMEM (row-major) ======
                // Identical to multi-warp kernel — A loading doesn't change
                let load_i = ctx.mov_u32_imm(0);
                ctx.label("load_a_loop");
                let load_a_done = ctx.setp_ge_u32(load_i, c_4_u32);
                ctx.branch_if(load_a_done, "load_a_end");

                let elem_a = ctx.add_u32_reg(my_start_a, load_i);
                let row_in_tile = ctx.shr_u32(elem_a, shift_4); // /16
                let k_in_tile = ctx.and_u32(elem_a, c_15); // %16

                // SHMEM addr (row-major): elem * 2 bytes
                let smem_a_off = ctx.mul_u32_reg(elem_a, c_2);
                let smem_a_addr = ctx.add_u32_reg(smem_a_base, smem_a_off);

                // Global coords
                let global_row = ctx.add_u32_reg(tile_row, row_in_tile);
                let global_k_a = ctx.add_u32_reg(k_offset, k_in_tile);

                // Clamp and load
                let clamped_row = ctx.min_u32(global_row, m_minus_1);
                let clamped_k = ctx.min_u32(global_k_a, k_minus_1);

                let a_row_off = ctx.mul_wide_u32_reg(clamped_row, k_param);
                let a_k_off = ctx.cvt_u64_u32(clamped_k);
                let a_elem_off = ctx.add_u64(a_row_off, a_k_off);
                let a_byte_off = ctx.mul_u64(a_elem_off, 4);
                let a_addr = ctx.add_u64(a_ptr, a_byte_off);

                let a_val_f32 = ctx.ld_global_f32(a_addr);

                // Zero OOB
                let row_valid = ctx.setp_lt_u32(global_row, m_param);
                let k_valid = ctx.setp_lt_u32(global_k_a, k_param);
                let zero_f32 = ctx.mov_f32_imm(0.0);
                let a_masked = ctx.selp_f32(row_valid, a_val_f32, zero_f32);
                let a_masked2 = ctx.selp_f32(k_valid, a_masked, zero_f32);

                let a_f16 = ctx.cvt_f16_f32(a_masked2);
                ctx.st_shared_f16(smem_a_addr, a_f16);

                ctx.add_u32_inplace(load_i, 1);
                ctx.branch("load_a_loop");
                ctx.label("load_a_end");

                // ====== PHASE 2: Dequant B[16×32] Q4K→FP16 SHMEM (col-major) ======
                // PMAT-091: Column-first thread mapping for coalesced interleaved access
                //
                // 512 elements, 128 threads, 4 elements per thread.
                // Mapping: elem_b = j * 128 + tid
                //   col_in_tile = elem_b % 32 = tid % 32  (since 128 % 32 == 0)
                //   k_in_tile   = elem_b / 32
                //
                // Adjacent threads → adjacent columns → coalesced qs byte access.
                // B SHMEM layout: col-major [16_k × 32_col], addr = col * 16 + k
                let load_j = ctx.mov_u32_imm(0);
                ctx.label("load_b_loop");
                let load_b_done = ctx.setp_ge_u32(load_j, c_4_u32);
                ctx.branch_if(load_b_done, "load_b_end");

                // Column-first mapping: elem_b = j * 128 + tid
                let j_times_128 = ctx.mul_u32_reg(load_j, c_128);
                let elem_b = ctx.add_u32_reg(j_times_128, tid);
                let col_in_tile = ctx.and_u32(elem_b, c_31); // % 32
                let k_in_tile_b = ctx.shr_u32(elem_b, shift_5); // / 32

                // SHMEM addr: B is col-major [16_k × 32_col]
                // addr = (col_in_tile * 16 + k_in_tile_b) * 2
                let smem_b_elem = ctx.mul_u32_reg(col_in_tile, c_16);
                let smem_b_elem = ctx.add_u32_reg(smem_b_elem, k_in_tile_b);
                let smem_b_off = ctx.mul_u32_reg(smem_b_elem, c_2);
                let smem_b_addr = ctx.add_u32_reg(smem_b_base, smem_b_off);

                // Global coords
                let global_col = ctx.add_u32_reg(tile_col, col_in_tile);
                let global_k_b = ctx.add_u32_reg(k_offset, k_in_tile_b);

                // Interleaved tile addressing:
                // n_tile_idx = global_col / 16
                // col_within_itile = global_col % 16
                // tile_base = b_ptr + (n_tile_idx * num_sb + sb_idx) * 2304
                let clamped_col = ctx.min_u32(global_col, n_minus_1);
                let clamped_n_tile = ctx.shr_u32(clamped_col, shift_4);
                let clamped_col_in_itile = ctx.and_u32(clamped_col, c_15);

                // sb_idx from global K
                let sb_idx = ctx.shr_u32(global_k_b, shift_8);
                let k_within_sb = ctx.and_u32(global_k_b, c_255);

                // Tile base address = b_ptr + (n_tile_idx * num_sb + sb_idx) * 2304
                let tile_sb_off = ctx.mul_u32_reg(clamped_n_tile, num_sb_reg);
                let tile_sb_off = ctx.add_u32_reg(tile_sb_off, sb_idx);
                let tile_byte_off = ctx.mul_wide_u32(tile_sb_off, INTERLEAVED_TILE_BYTES as u32);
                let tile_base = ctx.add_u64(b_ptr, tile_byte_off);

                // ---- Load d (FP16): tile_base + TILE_D_OFFSET + col_within_itile * 2 ----
                let d_col_off = ctx.mul_u32_reg(clamped_col_in_itile, c_2);
                let d_off_32 = ctx.add_u32_reg(tile_d_off, d_col_off);
                let d_off_64 = ctx.cvt_u64_u32(d_off_32);
                let d_addr = ctx.add_u64(tile_base, d_off_64);
                let d_f16 = ctx.ld_global_f16(d_addr);
                let d_val = ctx.cvt_f32_f16(d_f16);

                // ---- Load dmin (FP16): tile_base + TILE_DMIN_OFFSET + col_within_itile * 2 ----
                let dmin_col_off = ctx.mul_u32_reg(clamped_col_in_itile, c_2);
                let dmin_off_32 = ctx.add_u32_reg(tile_dmin_off, dmin_col_off);
                let dmin_off_64 = ctx.cvt_u64_u32(dmin_off_32);
                let dmin_addr = ctx.add_u64(tile_base, dmin_off_64);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin_val = ctx.cvt_f32_f16(dmin_f16);

                // ---- Scale/min extraction from interleaved scales ----
                // scales_base = tile_base + TILE_SCALES_OFFSET + col_within_itile * 12
                // Within each column's 12 bytes, same GGML Q4K split format:
                //   bytes 0-3:  scale[0..3] & 0x3F (low)
                //   bytes 4-7:  min[0..3] & 0x3F (low)
                //   bytes 8-11: combo for SB 4-7
                let sub_block = ctx.shr_u32(k_within_sb, shift_5); // /32
                let val_in_sub = ctx.and_u32(k_within_sb, c_31); // %32

                let is_high = ctx.setp_ge_u32(sub_block, c_4_u32);
                let i_hi_raw = ctx.sub_u32_reg(sub_block, c_4_u32);
                let i_hi = ctx.min_u32(i_hi_raw, c_3_u32);

                let col_scales_off = ctx.mul_u32_reg(clamped_col_in_itile, c_12);
                let scales_base_off = ctx.add_u32_reg(tile_scales_off, col_scales_off);
                let scales_base_64 = ctx.cvt_u64_u32(scales_base_off);
                let scales_base = ctx.add_u64(tile_base, scales_base_64);

                // Low path (SB 0-3): scale = scales[sb] & 0x3F
                let sub_block_64 = ctx.cvt_u64_u32(sub_block);
                let lo_sc_addr = ctx.add_u64(scales_base, sub_block_64);
                let lo_sc_byte = ctx.ld_global_u8(lo_sc_addr);
                let lo_sc_32 = ctx.cvt_u32_u8(lo_sc_byte);
                let lo_scale = ctx.and_u32(lo_sc_32, mask_3f);

                let four_64 = ctx.mov_u64_imm(4);
                let lo_mn_base = ctx.add_u64(scales_base, four_64);
                let lo_mn_addr = ctx.add_u64(lo_mn_base, sub_block_64);
                let lo_mn_byte = ctx.ld_global_u8(lo_mn_addr);
                let lo_mn_32 = ctx.cvt_u32_u8(lo_mn_byte);
                let lo_min = ctx.and_u32(lo_mn_32, mask_3f);

                // High path (SB 4-7): combo bytes at offset 8
                let i_hi_64 = ctx.cvt_u64_u32(i_hi);
                let eight_64 = ctx.mov_u64_imm(8);
                let combo_base = ctx.add_u64(scales_base, eight_64);
                let combo_addr = ctx.add_u64(combo_base, i_hi_64);
                let combo_byte = ctx.ld_global_u8(combo_addr);
                let combo_32 = ctx.cvt_u32_u8(combo_byte);

                let sc_low4 = ctx.and_u32(combo_32, mask_0f_4);

                let hi_sc_addr = ctx.add_u64(scales_base, i_hi_64);
                let hi_sc_byte = ctx.ld_global_u8(hi_sc_addr);
                let hi_sc_32 = ctx.cvt_u32_u8(hi_sc_byte);
                let sc_shifted = ctx.shr_u32(hi_sc_32, c_6_u32);
                let sc_high2 = ctx.and_u32(sc_shifted, mask_03);
                let sc_high_pos = ctx.shl_u32(sc_high2, c_4_u32);
                let hi_scale = ctx.or_u32(sc_low4, sc_high_pos);

                let mn_shifted = ctx.shr_u32(combo_32, c_4_u32);
                let mn_low4 = ctx.and_u32(mn_shifted, mask_0f_4);

                let hi_mn_base = ctx.add_u64(scales_base, four_64);
                let hi_mn_addr = ctx.add_u64(hi_mn_base, i_hi_64);
                let hi_mn_byte = ctx.ld_global_u8(hi_mn_addr);
                let hi_mn_32 = ctx.cvt_u32_u8(hi_mn_byte);
                let mn_hi_shifted = ctx.shr_u32(hi_mn_32, c_6_u32);
                let mn_high2 = ctx.and_u32(mn_hi_shifted, mask_03);
                let mn_high_pos = ctx.shl_u32(mn_high2, c_4_u32);
                let hi_min = ctx.or_u32(mn_low4, mn_high_pos);

                // Select based on sub-block index
                let scale_int = ctx.selp_u32(is_high, hi_scale, lo_scale);
                let min_int = ctx.selp_u32(is_high, hi_min, lo_min);

                let scale_f32 = ctx.cvt_f32_u32(scale_int);
                let min_f32 = ctx.cvt_f32_u32(min_int);

                let d_scale = ctx.mul_f32(d_val, scale_f32);
                let dmin_min = ctx.mul_f32(dmin_val, min_f32);

                // ---- qs nibble from interleaved layout ----
                // byte_idx = pair * 32 + val_in_sub  (same as original)
                // qs_addr = tile_base + TILE_QS_OFFSET + byte_idx * 16 + col_within_itile
                // → 16 adjacent threads (different cols) access 16 consecutive bytes!
                let pair = ctx.shr_u32(sub_block, c_1);
                let nibble_sel = ctx.and_u32(sub_block, c_1);
                let nibble_shift = ctx.mul_u32_reg(nibble_sel, c_4_u32);

                let pair_byte_base = ctx.mul_u32_reg(pair, c_32);
                let byte_idx = ctx.add_u32_reg(pair_byte_base, val_in_sub);

                // Interleaved: qs_addr = tile_base + QS_OFFSET + byte_idx * 16 + col
                let byte_idx_times_16 = ctx.mul_u32_reg(byte_idx, c_16);
                let qs_off = ctx.add_u32_reg(tile_qs_off, byte_idx_times_16);
                let qs_off = ctx.add_u32_reg(qs_off, clamped_col_in_itile);
                let qs_off_64 = ctx.cvt_u64_u32(qs_off);
                let qs_addr = ctx.add_u64(tile_base, qs_off_64);
                let packed = ctx.ld_global_u8(qs_addr);
                let packed_32 = ctx.cvt_u32_u8(packed);

                let shifted_qs = ctx.shr_u32(packed_32, nibble_shift);
                let quant = ctx.and_u32(shifted_qs, mask_0f);
                let quant_f32 = ctx.cvt_f32_u32(quant);

                // Dequantize: val = d * scale * quant - dmin * min
                let weighted = ctx.mul_f32(d_scale, quant_f32);
                let dequant = ctx.sub_f32(weighted, dmin_min);

                // Zero OOB elements
                let col_valid = ctx.setp_lt_u32(global_col, n_param);
                let k_valid_b = ctx.setp_lt_u32(global_k_b, k_param);
                let zero_b = ctx.mov_f32_imm(0.0);
                let dequant_m = ctx.selp_f32(col_valid, dequant, zero_b);
                let dequant_m2 = ctx.selp_f32(k_valid_b, dequant_m, zero_b);

                let b_f16 = ctx.cvt_f16_f32(dequant_m2);
                ctx.st_shared_f16(smem_b_addr, b_f16);

                ctx.add_u32_inplace(load_j, 1);
                ctx.branch("load_b_loop");
                ctx.label("load_b_end");

                // Barrier: all loading complete
                ctx.bar_sync(0);

                // ====== PHASE 3: WMMA 16×16×16 per warp ======
                // Each warp loads from its own 16×16 sub-tile of A and B
                let smem_generic = ctx.shared_base_addr();

                // A fragment: warp_row selects upper or lower 16 rows
                let c_512 = ctx.mov_u32_imm(512);
                let a_warp_byte_off = ctx.mul_u32_reg(warp_row, c_512);
                let a_warp_off_64 = ctx.cvt_u64_u32(a_warp_byte_off);
                let smem_a_warp = ctx.add_u64(smem_generic, a_warp_off_64);
                let frag_a = ctx.wmma_load_a_f16(smem_a_warp, 16, WmmaLayout::RowMajor);

                // B fragment: warp_col selects left or right 16 columns
                let b_warp_off = ctx.mul_u32_reg(warp_col, c_512);
                let b_base_plus_warp = ctx.add_u32_reg(smem_b_base, b_warp_off);
                let b_warp_off_64 = ctx.cvt_u64_u32(b_base_plus_warp);
                let smem_b_warp = ctx.add_u64(smem_generic, b_warp_off_64);
                let frag_b = ctx.wmma_load_b_f16(smem_b_warp, 16, WmmaLayout::ColMajor);

                // WMMA: D = A × B + C
                let frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);

                // Accumulate D → C for next K-tile
                for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                    ctx.mov_f32_reg(*c_reg, *d_reg);
                }

                // Barrier before next K-tile overwrites SHMEM
                ctx.bar_sync(1);

                ctx.add_u32_inplace(k_tile_idx, 1);
                ctx.branch("k_tile_loop");
                ctx.label("k_tile_end");

                // ====== Store C[16×16] FP32 to global (per warp) ======
                let c_16_reg = ctx.mov_u32_imm(16);
                let out_row_off = ctx.mul_u32_reg(warp_row, c_16_reg);
                let out_row = ctx.add_u32_reg(tile_row, out_row_off);
                let out_col_off = ctx.mul_u32_reg(warp_col, c_16_reg);
                let out_col = ctx.add_u32_reg(tile_col, out_col_off);

                let c_row_off = ctx.mul_wide_u32_reg(out_row, n_param);
                let c_row_bytes = ctx.mul_u64(c_row_off, 4);
                let c_col_off = ctx.cvt_u64_u32(out_col);
                let c_col_bytes = ctx.mul_u64(c_col_off, 4);
                let c_tile_addr = ctx.add_u64(c_ptr, c_row_bytes);
                let c_tile_addr = ctx.add_u64(c_tile_addr, c_col_bytes);

                ctx.wmma_store_d_f32(c_tile_addr, &frag_c, n_const, WmmaLayout::RowMajor);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interleaved_wmma_kernel_name() {
        let k = InterleavedWmmaQ4KGemmKernel::new(4, 1536, 1536);
        assert_eq!(k.name(), "interleaved_wmma_q4k_gemm");
    }

    #[test]
    fn test_interleaved_wmma_kernel_emits_ptx() {
        let k = InterleavedWmmaQ4KGemmKernel::new(4, 1536, 1536);
        let ptx = k.emit_ptx();
        assert!(ptx.contains(".entry interleaved_wmma_q4k_gemm"));
        assert!(ptx.contains(".maxnreg 96"));
        assert!(ptx.contains("wmma.load.a.sync.aligned"));
        assert!(ptx.contains("wmma.load.b.sync.aligned"));
        assert!(ptx.contains("wmma.mma.sync.aligned"));
        assert!(ptx.contains("wmma.store.d.sync.aligned"));
    }

    #[test]
    fn test_interleaved_wmma_shared_memory() {
        let k = InterleavedWmmaQ4KGemmKernel::new(4, 1536, 1536);
        let ptx = k.emit_ptx();
        // 2048 bytes = 32×16×2 (A) + 16×32×2 (B)
        assert!(ptx.contains("smem[2048]"), "SHMEM should be 2048 bytes for 32×32 tile");
    }

    #[test]
    fn test_interleaved_wmma_barrier_safety() {
        let k = InterleavedWmmaQ4KGemmKernel::new(4, 1536, 1536);
        assert!(k.validate_barrier_safety().is_ok());
    }

    #[test]
    fn test_interleaved_wmma_ffn_dimensions() {
        // FFN shape: M=4 (batched decode), K=1536, N=8960
        let k = InterleavedWmmaQ4KGemmKernel::new(4, 1536, 8960);
        let ptx = k.emit_ptx();
        assert!(ptx.contains(".entry interleaved_wmma_q4k_gemm"));
    }

    #[test]
    fn test_interleaved_wmma_num_super_blocks() {
        let k = InterleavedWmmaQ4KGemmKernel::new(4, 1536, 1536);
        assert_eq!(k.num_super_blocks(), 6); // 1536 / 256 = 6
    }
}
