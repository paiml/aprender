//! PMAT-054B: W4A16 WMMA Q4K GEMM with pre-computed scales.
//!
//! Uses the W4A16 weight format (pre-computed FP16 effective scales) to reduce
//! GPU-side dequantization from ~20 to ~5 instructions per element.
//!
//! vs PMAT-091 interleaved kernel:
//! - Phase 2 (B loading): 5-8 global loads + 15 ALU → 3 loads + 4 ALU per element
//! - Effective scale/min loaded as FP16 (pre-computed on CPU, no GGML decode)
//! - qs nibble access identical (byte-interleaved, coalesced)
//!
//! Same architecture: 32×32 output tiles, 128 threads (4 warps, 2×2 layout),
//! WMMA 16×16×16 FP16 multiply + FP32 accumulate.
//!
//! Grid: (ceil(N/32), ceil(M/32)), Block: 128 threads
//! SHMEM: 2048 bytes (A[32×16 FP16] + B[16×32 FP16])

use crate::kernels::quantize::q4k::w4a16::{
    W4A16_MIN_OFFSET, W4A16_QS_OFFSET, W4A16_SCALE_OFFSET, W4A16_TILE_BYTES,
};
use crate::kernels::quantize::Q4K_SUPER_BLOCK_SIZE;
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, WmmaLayout};

/// W4A16 WMMA Q4K GEMM kernel with pre-computed scales.
///
/// Drop-in replacement for [`InterleavedWmmaQ4KGemmKernel`] when weights
/// have been repacked with `repack_q4k_w4a16`. Same grid/block config.
///
/// Input: FP32 activations [M, K]
/// Weights: W4A16 Q4K [N_tiles × (K/256) × 2560B] tiles
/// Output: FP32 [M, N]
#[derive(Debug, Clone)]
pub struct W4a16WmmaQ4KGemmKernel {
    /// Batch size (M)
    pub m: u32,
    /// Output dimension (N)
    pub n: u32,
    /// Input dimension (K) — must be multiple of 256
    pub k: u32,
}

impl W4a16WmmaQ4KGemmKernel {
    /// Creates a new W4A16 WMMA Q4K GEMM kernel with the given dimensions.
    #[must_use]
    pub fn new(m: u32, k: u32, n: u32) -> Self {
        Self { m, n, k }
    }

    /// Returns the number of Q4K super-blocks along the K dimension.
    #[must_use]
    pub fn num_super_blocks(&self) -> u32 {
        (self.k + Q4K_SUPER_BLOCK_SIZE - 1) / Q4K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for W4a16WmmaQ4KGemmKernel {
    fn name(&self) -> &str {
        "w4a16_wmma_q4k_gemm"
    }

    #[allow(clippy::too_many_lines)]
    fn build_ptx(&self) -> PtxKernel {
        let n_const = self.n;
        let k_const = self.k;
        let num_sb = self.num_super_blocks();
        let n_k_tiles = k_const / 16;

        // SHMEM: A[32×16 FP16] + B[16×32 FP16] = 1024 + 1024 = 2048 bytes
        let smem_a_size: u32 = 32 * 16 * 2;
        let smem_b_offset: u32 = smem_a_size;
        let smem_bytes = (smem_a_size + 16 * 32 * 2) as usize;

        PtxKernel::new("w4a16_wmma_q4k_gemm")
            .max_regs(96)
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_w4a16_ptr")
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
                let c_15 = ctx.mov_u32_imm(15);
                let c_16 = ctx.mov_u32_imm(16);
                let c_31 = ctx.mov_u32_imm(31);
                let c_32 = ctx.mov_u32_imm(32);
                let c_128 = ctx.mov_u32_imm(128);

                // Block tile origins
                let tile_col = ctx.mul_u32_reg(ctaid_x, c_32);
                let tile_row = ctx.mul_u32_reg(ctaid_y, c_32);

                let m_param = ctx.load_param_u32("m_param");
                let n_param = ctx.load_param_u32("n_param");
                let k_param = ctx.load_param_u32("k_param");

                // Skip out-of-range blocks
                let row_oob = ctx.setp_ge_u32(tile_row, m_param);
                ctx.branch_if(row_oob, "exit");
                let col_oob = ctx.setp_ge_u32(tile_col, n_param);
                ctx.branch_if(col_oob, "exit");

                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_w4a16_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Warp dispatch
                let shift_5 = ctx.mov_u32_imm(5);
                let warp_id = ctx.shr_u32(tid, shift_5);
                let warp_row = ctx.shr_u32(warp_id, c_1);
                let warp_col = ctx.and_u32(warp_id, c_1);

                // SHMEM bases
                let smem_a_base = c_0;
                let smem_b_base = ctx.mov_u32_imm(smem_b_offset);

                // Initialize accumulators
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
                let shift_8 = ctx.mov_u32_imm(8);
                let mask_0f = c_15;
                let c_255 = ctx.mov_u32_imm(255);

                // W4A16 tile constants
                let w4_scale_off = ctx.mov_u32_imm(W4A16_SCALE_OFFSET as u32);
                let w4_min_off = ctx.mov_u32_imm(W4A16_MIN_OFFSET as u32);
                let w4_qs_off = ctx.mov_u32_imm(W4A16_QS_OFFSET as u32);

                // A loading element mapping
                let my_start_a = ctx.mul_u32_reg(tid, c_4_u32);

                // ===== K-tile loop =====
                ctx.label("k_tile_loop");
                let k_done = ctx.setp_ge_u32(k_tile_idx, n_k_tiles_reg);
                ctx.branch_if(k_done, "k_tile_end");

                let k_offset = ctx.mul_u32_reg(k_tile_idx, c_16);

                // ====== PHASE 1: Load A[32×16] FP32→FP16 SHMEM ======
                // (Identical to PMAT-091 — A loading doesn't change)
                let load_i = ctx.mov_u32_imm(0);
                ctx.label("load_a_loop");
                let load_a_done = ctx.setp_ge_u32(load_i, c_4_u32);
                ctx.branch_if(load_a_done, "load_a_end");

                let elem_a = ctx.add_u32_reg(my_start_a, load_i);
                let row_in_tile = ctx.shr_u32(elem_a, shift_4);
                let k_in_tile = ctx.and_u32(elem_a, c_15);

                let smem_a_off = ctx.mul_u32_reg(elem_a, c_2);
                let smem_a_addr = ctx.add_u32_reg(smem_a_base, smem_a_off);

                let global_row = ctx.add_u32_reg(tile_row, row_in_tile);
                let global_k_a = ctx.add_u32_reg(k_offset, k_in_tile);

                let clamped_row = ctx.min_u32(global_row, m_minus_1);
                let clamped_k = ctx.min_u32(global_k_a, k_minus_1);

                let a_row_off = ctx.mul_wide_u32_reg(clamped_row, k_param);
                let a_k_off = ctx.cvt_u64_u32(clamped_k);
                let a_elem_off = ctx.add_u64(a_row_off, a_k_off);
                let a_byte_off = ctx.mul_u64(a_elem_off, 4);
                let a_addr = ctx.add_u64(a_ptr, a_byte_off);

                let a_val_f32 = ctx.ld_global_f32(a_addr);

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

                // ====== PHASE 2: W4A16 Dequant B[16×32] → FP16 SHMEM ======
                // PMAT-054B: Pre-computed scales eliminate GGML decode overhead.
                // Per element: 3 global loads + 4 ALU (vs 5-8 loads + 15 ALU in PMAT-091).
                let load_j = ctx.mov_u32_imm(0);
                ctx.label("load_b_loop");
                let load_b_done = ctx.setp_ge_u32(load_j, c_4_u32);
                ctx.branch_if(load_b_done, "load_b_end");

                // Column-first mapping (same as PMAT-091)
                let j_times_128 = ctx.mul_u32_reg(load_j, c_128);
                let elem_b = ctx.add_u32_reg(j_times_128, tid);
                let col_in_tile = ctx.and_u32(elem_b, c_31); // % 32
                let k_in_tile_b = ctx.shr_u32(elem_b, shift_5); // / 32

                // SHMEM B addr (col-major)
                let smem_b_elem = ctx.mul_u32_reg(col_in_tile, c_16);
                let smem_b_elem = ctx.add_u32_reg(smem_b_elem, k_in_tile_b);
                let smem_b_off = ctx.mul_u32_reg(smem_b_elem, c_2);
                let smem_b_addr = ctx.add_u32_reg(smem_b_base, smem_b_off);

                // Global coords
                let global_col = ctx.add_u32_reg(tile_col, col_in_tile);
                let global_k_b = ctx.add_u32_reg(k_offset, k_in_tile_b);

                // W4A16 tile addressing
                let clamped_col = ctx.min_u32(global_col, n_minus_1);
                let clamped_n_tile = ctx.shr_u32(clamped_col, shift_4); // /16
                let clamped_col_in_itile = ctx.and_u32(clamped_col, c_15); // %16

                // Super-block index and K within super-block
                let sb_idx = ctx.shr_u32(global_k_b, shift_8); // /256
                let k_within_sb = ctx.and_u32(global_k_b, c_255); // %256

                // Sub-block index: k_within_sb / 32
                let sub_block = ctx.shr_u32(k_within_sb, shift_5); // /32
                                                                   // Value within sub-block: k_within_sb % 32
                let val_in_sub = ctx.and_u32(k_within_sb, c_31); // %32

                // Tile base address
                let tile_sb_off = ctx.mul_u32_reg(clamped_n_tile, num_sb_reg);
                let tile_sb_off = ctx.add_u32_reg(tile_sb_off, sb_idx);
                let tile_byte_off = ctx.mul_wide_u32(tile_sb_off, W4A16_TILE_BYTES as u32);
                let tile_base = ctx.add_u64(b_ptr, tile_byte_off);

                // ---- Load pre-computed eff_scale (1 FP16 load) ----
                // eff_scale[sub_block * 16 + col_in_itile] at SCALE_OFFSET
                let scale_sb_off = ctx.mul_u32_reg(sub_block, c_16);
                let scale_idx = ctx.add_u32_reg(scale_sb_off, clamped_col_in_itile);
                let scale_byte_off = ctx.mul_u32_reg(scale_idx, c_2);
                let scale_off = ctx.add_u32_reg(w4_scale_off, scale_byte_off);
                let scale_off_64 = ctx.cvt_u64_u32(scale_off);
                let scale_addr = ctx.add_u64(tile_base, scale_off_64);
                let eff_scale_f16 = ctx.ld_global_f16(scale_addr);
                let eff_scale = ctx.cvt_f32_f16(eff_scale_f16);

                // ---- Load pre-computed eff_min (1 FP16 load) ----
                let min_byte_off = ctx.mul_u32_reg(scale_idx, c_2);
                let min_off = ctx.add_u32_reg(w4_min_off, min_byte_off);
                let min_off_64 = ctx.cvt_u64_u32(min_off);
                let min_addr = ctx.add_u64(tile_base, min_off_64);
                let eff_min_f16 = ctx.ld_global_f16(min_addr);
                let eff_min = ctx.cvt_f32_f16(eff_min_f16);

                // ---- Load qs nibble (1 byte load, coalesced) ----
                // Same pair/nibble mapping as Q4K:
                // pair = sub_block / 2, nibble_sel = sub_block % 1
                let pair = ctx.shr_u32(sub_block, c_1);
                let nibble_sel = ctx.and_u32(sub_block, c_1);
                let nibble_shift = ctx.mul_u32_reg(nibble_sel, c_4_u32);

                let pair_byte_base = ctx.mul_u32_reg(pair, c_32);
                let byte_idx = ctx.add_u32_reg(pair_byte_base, val_in_sub);

                // Interleaved: qs_addr = tile_base + QS_OFFSET + byte_idx * 16 + col
                let byte_idx_times_16 = ctx.mul_u32_reg(byte_idx, c_16);
                let qs_off = ctx.add_u32_reg(w4_qs_off, byte_idx_times_16);
                let qs_off = ctx.add_u32_reg(qs_off, clamped_col_in_itile);
                let qs_off_64 = ctx.cvt_u64_u32(qs_off);
                let qs_addr = ctx.add_u64(tile_base, qs_off_64);
                let packed = ctx.ld_global_u8(qs_addr);
                let packed_32 = ctx.cvt_u32_u8(packed);

                // Extract nibble
                let shifted_qs = ctx.shr_u32(packed_32, nibble_shift);
                let quant = ctx.and_u32(shifted_qs, mask_0f);
                let quant_f32 = ctx.cvt_f32_u32(quant);

                // Dequant: val = quant * eff_scale - eff_min (2 FP ops)
                let weighted = ctx.mul_f32(eff_scale, quant_f32);
                let dequant = ctx.sub_f32(weighted, eff_min);

                // Zero OOB
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
                let smem_generic = ctx.shared_base_addr();

                let c_512 = ctx.mov_u32_imm(512);
                let a_warp_byte_off = ctx.mul_u32_reg(warp_row, c_512);
                let a_warp_off_64 = ctx.cvt_u64_u32(a_warp_byte_off);
                let smem_a_warp = ctx.add_u64(smem_generic, a_warp_off_64);
                let frag_a = ctx.wmma_load_a_f16(smem_a_warp, 16, WmmaLayout::RowMajor);

                let b_warp_off = ctx.mul_u32_reg(warp_col, c_512);
                let b_base_plus_warp = ctx.add_u32_reg(smem_b_base, b_warp_off);
                let b_warp_off_64 = ctx.cvt_u64_u32(b_base_plus_warp);
                let smem_b_warp = ctx.add_u64(smem_generic, b_warp_off_64);
                let frag_b = ctx.wmma_load_b_f16(smem_b_warp, 16, WmmaLayout::ColMajor);

                let frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);

                for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                    ctx.mov_f32_reg(*c_reg, *d_reg);
                }

                // Barrier before next K-tile
                ctx.bar_sync(1);

                ctx.add_u32_inplace(k_tile_idx, 1);
                ctx.branch("k_tile_loop");
                ctx.label("k_tile_end");

                // ====== Store C[16×16] FP32 to global ======
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
    fn test_w4a16_wmma_kernel_name() {
        let k = W4a16WmmaQ4KGemmKernel::new(4, 1536, 1536);
        assert_eq!(k.name(), "w4a16_wmma_q4k_gemm");
    }

    #[test]
    fn test_w4a16_wmma_kernel_emits_ptx() {
        let k = W4a16WmmaQ4KGemmKernel::new(4, 1536, 1536);
        let ptx = k.emit_ptx();
        assert!(ptx.contains(".entry w4a16_wmma_q4k_gemm"));
        assert!(ptx.contains(".maxnreg 96"));
        assert!(ptx.contains("wmma.load.a.sync.aligned"));
        assert!(ptx.contains("wmma.load.b.sync.aligned"));
        assert!(ptx.contains("wmma.mma.sync.aligned"));
        assert!(ptx.contains("wmma.store.d.sync.aligned"));
    }

    #[test]
    fn test_w4a16_wmma_shared_memory() {
        let k = W4a16WmmaQ4KGemmKernel::new(4, 1536, 1536);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("smem[2048]"), "SHMEM should be 2048 bytes");
    }

    #[test]
    fn test_w4a16_wmma_barrier_safety() {
        let k = W4a16WmmaQ4KGemmKernel::new(4, 1536, 1536);
        assert!(k.validate_barrier_safety().is_ok());
    }

    #[test]
    fn test_w4a16_wmma_ffn_dimensions() {
        let k = W4a16WmmaQ4KGemmKernel::new(4, 1536, 8960);
        let ptx = k.emit_ptx();
        assert!(ptx.contains(".entry w4a16_wmma_q4k_gemm"));
    }

    #[test]
    fn test_w4a16_wmma_num_super_blocks() {
        let k = W4a16WmmaQ4KGemmKernel::new(4, 1536, 1536);
        assert_eq!(k.num_super_blocks(), 6);
    }

    #[test]
    fn test_w4a16_wmma_fewer_registers_than_interleaved() {
        // W4A16 kernel should use fewer instructions than PMAT-091 interleaved
        // kernel because it doesn't decode GGML scales on GPU.
        use crate::kernels::quantize::fp16_tensor::InterleavedWmmaQ4KGemmKernel;

        let w4 = W4a16WmmaQ4KGemmKernel::new(4, 1536, 1536);
        let il = InterleavedWmmaQ4KGemmKernel::new(4, 1536, 1536);
        let w4_ptx = w4.emit_ptx();
        let il_ptx = il.emit_ptx();

        // W4A16 should produce shorter PTX (fewer instructions)
        assert!(
            w4_ptx.len() < il_ptx.len(),
            "W4A16 PTX ({} bytes) should be shorter than interleaved ({} bytes)",
            w4_ptx.len(),
            il_ptx.len()
        );
    }
}
