//! PMAT-064: Q4K WMMA GEMM — tensor core GEMM reading Q4K weights directly
//!
//! Dequantizes Q4K super-blocks to FP16 in shared memory, then uses WMMA
//! 16×16×16 tensor core tiles for the matmul. Reads 0.5625 B/elem (Q4K)
//! instead of 2 B/elem (FP16 HGEMM) — 3.56× bandwidth savings.
//!
//! Grid: (ceil(N/16), ceil(M/16)), Block: 32 threads (1 warp)
//! SHMEM: 1024 bytes (A[16×16] FP16 + B[16×16] FP16)
//!
//! A (activations): [M × K] FP32 global → FP16 in SHMEM
//! B (weights): [N × K/256 × 144B] Q4K global → dequant FP16 in SHMEM
//! C (output): [M × N] FP32 global
//!
//! Computes: C[m,n] = sum_k A[m,k] × W[n,k]  (W transposed via col-major B)

use crate::kernels::quantize::{Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, WmmaLayout};

/// Q4K WMMA GEMM kernel — tensor cores with direct Q4K weight reads
///
/// Replaces cuBLAS HGEMM prefill. Same correctness, 3.56× less bandwidth.
/// Requires sm_70+ for WMMA tensor cores.
///
/// Input: FP32 activations [M, K]
/// Weights: Q4K [N × (K/256) × 144B] super-blocks
/// Output: FP32 [M, N]
#[derive(Debug, Clone)]
pub struct TensorCoreQ4KGemmKernel {
    /// Batch size (M) — sequence length for prefill
    pub m: u32,
    /// Output dimension (N) — hidden_dim or intermediate_dim
    pub n: u32,
    /// Input dimension (K) — must be multiple of 256 for Q4K super-blocks
    pub k: u32,
}

impl TensorCoreQ4KGemmKernel {
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

impl Kernel for TensorCoreQ4KGemmKernel {
    fn name(&self) -> &str {
        "tensor_core_q4k_gemm"
    }

    #[allow(clippy::too_many_lines)]
    fn build_ptx(&self) -> PtxKernel {
        let n_const = self.n;
        let k_const = self.k;
        let num_sb = self.num_super_blocks();
        let n_k_tiles = k_const / 16;

        // SHMEM: A tile [16×16 FP16] + B tile [16×16 FP16] = 1024 bytes
        let smem_bytes = 16 * 16 * 2 * 2;

        PtxKernel::new("tensor_core_q4k_gemm")
            .max_regs(96)
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_quant_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m_param")
            .param(PtxType::U32, "n_param")
            .param(PtxType::U32, "k_param")
            .shared_memory(smem_bytes as usize)
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
                let c_255 = ctx.mov_u32_imm(255);

                let tile_col = ctx.mul_u32_reg(ctaid_x, c_16);
                let tile_row = ctx.mul_u32_reg(ctaid_y, c_16);

                let m_param = ctx.load_param_u32("m_param");
                let n_param = ctx.load_param_u32("n_param");
                let k_param = ctx.load_param_u32("k_param");

                // Bounds — skip out-of-range tiles
                let row_oob = ctx.setp_ge_u32(tile_row, m_param);
                ctx.branch_if(row_oob, "exit");
                let col_oob = ctx.setp_ge_u32(tile_col, n_param);
                ctx.branch_if(col_oob, "exit");

                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_quant_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // SHMEM offsets
                let smem_a_base = c_0;
                let smem_b_base = ctx.mov_u32_imm(512);

                // Initialize WMMA FP32 accumulator
                let frag_c = ctx.wmma_init_c_zero();

                let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);
                let k_tile_idx = ctx.mov_u32_imm(0);
                let num_sb_reg = ctx.mov_u32_imm(num_sb);

                // Clamp limits
                let m_minus_1 = ctx.sub_u32_reg(m_param, c_1);
                let n_minus_1 = ctx.sub_u32_reg(n_param, c_1);
                let k_minus_1 = ctx.sub_u32_reg(k_param, c_1);

                // Shift constants for div/mod by power-of-2
                let shift_4 = c_4_u32;
                let shift_5 = ctx.mov_u32_imm(5);
                let shift_8 = c_8_u32;
                let mask_0f = c_15;
                let mask_3f = ctx.mov_u32_imm(0x3F);
                let mask_03 = ctx.mov_u32_imm(0x03);
                let mask_0f_4 = ctx.mov_u32_imm(0x0F);
                let c_6_u32 = ctx.mov_u32_imm(6);
                let c_3_u32 = ctx.mov_u32_imm(3);

                // ===== K-tile loop =====
                ctx.label("k_tile_loop");
                let k_done = ctx.setp_ge_u32(k_tile_idx, n_k_tiles_reg);
                ctx.branch_if(k_done, "k_tile_end");

                let k_offset = ctx.mul_u32_reg(k_tile_idx, c_16);

                // Thread work: 32 threads × 8 elems = 256 = 16×16
                let my_start = ctx.mul_u32_reg(tid, c_8_u32);

                // ====== PHASE 1: Load A[16×16] FP32→FP16 SHMEM ======
                let load_i = ctx.mov_u32_imm(0);
                ctx.label("load_a_loop");
                let load_a_done = ctx.setp_ge_u32(load_i, c_8_u32);
                ctx.branch_if(load_a_done, "load_a_end");

                let elem_a = ctx.add_u32_reg(my_start, load_i);
                let row_in_tile = ctx.shr_u32(elem_a, shift_4);
                let k_in_tile = ctx.and_u32(elem_a, c_15);

                // SHMEM addr (row-major): elem * 2
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

                // ====== PHASE 2: Dequant B[16×16] Q4K→FP16 SHMEM (col-major) ======
                // B layout: SHMEM[col_in_tile * 16 + k_in_tile] = W[n, k]
                let load_j = ctx.mov_u32_imm(0);
                ctx.label("load_b_loop");
                let load_b_done = ctx.setp_ge_u32(load_j, c_8_u32);
                ctx.branch_if(load_b_done, "load_b_end");

                let elem_b = ctx.add_u32_reg(my_start, load_j);
                let col_in_tile = ctx.shr_u32(elem_b, shift_4);
                let k_in_tile_b = ctx.and_u32(elem_b, c_15);

                // SHMEM addr (col-major order: col*16 + k)
                let smem_b_off = ctx.mul_u32_reg(elem_b, c_2);
                let smem_b_addr = ctx.add_u32_reg(smem_b_base, smem_b_off);

                // Global coords
                let global_col = ctx.add_u32_reg(tile_col, col_in_tile);
                let global_k_b = ctx.add_u32_reg(k_offset, k_in_tile_b);

                let clamped_col = ctx.min_u32(global_col, n_minus_1);

                // Q4K addressing: sb_idx = global_k / 256
                let sb_idx = ctx.shr_u32(global_k_b, shift_8);
                let k_within_sb = ctx.and_u32(global_k_b, c_255);
                let sub_block = ctx.shr_u32(k_within_sb, shift_5); // /32
                let val_in_sub = ctx.and_u32(k_within_sb, c_31); // %32

                // Super-block address
                let col_sb_off = ctx.mul_u32_reg(clamped_col, num_sb_reg);
                let total_sb_off = ctx.add_u32_reg(col_sb_off, sb_idx);
                let sb_byte_off = ctx.mul_wide_u32(total_sb_off, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(b_ptr, sb_byte_off);

                // Load d, dmin
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d_val = ctx.cvt_f32_f16(d_f16);
                let dmin_offset_64 = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, dmin_offset_64);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin_val = ctx.cvt_f32_f16(dmin_f16);

                // ---- Scale/min extraction (GGML Q4K split format) ----
                let is_high = ctx.setp_ge_u32(sub_block, c_4_u32);
                let i_hi_raw = ctx.sub_u32_reg(sub_block, c_4_u32);
                let i_hi = ctx.min_u32(i_hi_raw, c_3_u32);

                let scales_offset_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, scales_offset_64);

                // Low path (SB 0-3)
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

                // High path (SB 4-7)
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

                // Select
                let scale_int = ctx.selp_u32(is_high, hi_scale, lo_scale);
                let min_int = ctx.selp_u32(is_high, hi_min, lo_min);

                let scale_f32 = ctx.cvt_f32_u32(scale_int);
                let min_f32 = ctx.cvt_f32_u32(min_int);

                let d_scale = ctx.mul_f32(d_val, scale_f32);
                let dmin_min = ctx.mul_f32(dmin_val, min_f32);

                // ---- qs nibble ----
                let pair = ctx.shr_u32(sub_block, c_1);
                let nibble_sel = ctx.and_u32(sub_block, c_1);
                let nibble_shift = ctx.mul_u32_reg(nibble_sel, c_4_u32);

                let pair_byte_base = ctx.mul_u32_reg(pair, c_32);
                let qs_offset = ctx.add_u32_reg(pair_byte_base, val_in_sub);
                let qs_offset_16 = ctx.add_u32_reg(qs_offset, c_16);
                let qs_offset_64 = ctx.cvt_u64_u32(qs_offset_16);
                let qs_addr = ctx.add_u64(sb_addr, qs_offset_64);
                let packed = ctx.ld_global_u8(qs_addr);
                let packed_32 = ctx.cvt_u32_u8(packed);

                let shifted_qs = ctx.shr_u32(packed_32, nibble_shift);
                let quant = ctx.and_u32(shifted_qs, mask_0f);
                let quant_f32 = ctx.cvt_f32_u32(quant);

                // Dequantize
                let weighted = ctx.mul_f32(d_scale, quant_f32);
                let dequant = ctx.sub_f32(weighted, dmin_min);

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

                ctx.bar_sync(0);

                // ====== PHASE 3: WMMA 16×16×16 ======
                let smem_generic = ctx.shared_base_addr();

                let frag_a = ctx.wmma_load_a_f16(smem_generic, 16, WmmaLayout::RowMajor);

                let smem_b_off_u64 = ctx.cvt_u64_u32(smem_b_base);
                let smem_b_ptr = ctx.add_u64(smem_generic, smem_b_off_u64);
                let frag_b = ctx.wmma_load_b_f16(smem_b_ptr, 16, WmmaLayout::ColMajor);

                let frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);

                // Accumulate D → C
                for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                    ctx.mov_f32_reg(*c_reg, *d_reg);
                }

                ctx.bar_sync(1);

                ctx.add_u32_inplace(k_tile_idx, 1);
                ctx.branch("k_tile_loop");
                ctx.label("k_tile_end");

                // ====== Store C[16×16] FP32 to global ======
                let c_row_off = ctx.mul_wide_u32_reg(tile_row, n_param);
                let c_row_bytes = ctx.mul_u64(c_row_off, 4);
                let c_col_off = ctx.cvt_u64_u32(tile_col);
                let c_col_bytes = ctx.mul_u64(c_col_off, 4);
                let c_tile_addr = ctx.add_u64(c_ptr, c_row_bytes);
                let c_tile_addr = ctx.add_u64(c_tile_addr, c_col_bytes);

                ctx.wmma_store_d_f32(c_tile_addr, &frag_c, n_const, WmmaLayout::RowMajor);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
