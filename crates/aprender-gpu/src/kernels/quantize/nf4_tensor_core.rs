// =============================================================================
// PMAT-479: NF4 TENSOR CORE GEMM — WMMA 16×16×16 with NF4 dequantization
// =============================================================================
//
// Dequantizes NF4 blocks to FP16 in shared memory, then uses WMMA tensor
// cores for matmul. Expected 5-40x compute improvement over naive tiled GEMM.
//
// Contract: nf4-tensor-core-gemm-v1.yaml

use super::nf4::nf4_register_lut_lookup;
use super::nf4_cpu::{NF4_BLOCK_SIZE, NF4_LUT};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, WmmaLayout};

const NF4_BLOCK_SIZE_U32: u32 = NF4_BLOCK_SIZE as u32;

/// NF4 tensor core GEMM — WMMA with inline NF4 dequantization.
///
/// Computes `C[M×N] = A[M×K] @ dequant(B_nf4[K×N])` using WMMA 16×16×16.
///
/// Phase 1: Load A[16×16] FP32→FP16 into shared memory (row-major)
/// Phase 2: Dequant B_nf4[16×16] to FP16 into shared memory (col-major)
/// Phase 3: WMMA mma.sync with FP32 accumulator
/// Phase 4: Store C[16×16] FP32 to global memory
///
/// Grid: (ceil(N/16), ceil(M/16)), Block: 32 threads (1 warp)
/// SHMEM: 1024 bytes (A[16×16] FP16 + B[16×16] FP16)
#[derive(Debug, Clone)]
pub struct Nf4TensorCoreGemmKernel {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl Nf4TensorCoreGemmKernel {
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    #[must_use]
    pub const fn num_k_blocks(&self) -> u32 {
        self.k / NF4_BLOCK_SIZE_U32
    }
}

impl Kernel for Nf4TensorCoreGemmKernel {
    fn name(&self) -> &str {
        "nf4_tensor_core_gemm"
    }

    #[allow(clippy::too_many_lines)]
    fn build_ptx(&self) -> PtxKernel {
        let k_const = self.k;
        let n_const = self.n;
        let n_k_tiles = k_const / 16;
        let num_k_blocks = k_const / NF4_BLOCK_SIZE_U32;
        let smem_bytes = 16 * 16 * 2 * 2; // A[16×16]+B[16×16] in FP16

        PtxKernel::new("nf4_tensor_core_gemm")
            .max_regs(96)
            .param(PtxType::U64, "a_ptr") // Activations [M×K] FP32
            .param(PtxType::U64, "scales_ptr") // NF4 scales [N × num_k_blocks] f32
            .param(PtxType::U64, "data_ptr") // NF4 packed nibbles
            .param(PtxType::U64, "c_ptr") // Output [M×N] FP32
            .param(PtxType::U32, "m_param")
            .param(PtxType::U32, "n_param")
            .param(PtxType::U32, "k_param")
            .shared_memory(smem_bytes as usize)
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX); // N tile
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY); // M tile

                let c_0 = ctx.mov_u32_imm(0);
                let c_1 = ctx.mov_u32_imm(1);
                let c_2 = ctx.mov_u32_imm(2);
                let c_4 = ctx.mov_u32_imm(4);
                let c_8 = ctx.mov_u32_imm(8);
                let c_15 = ctx.mov_u32_imm(15);
                let c_16 = ctx.mov_u32_imm(16);

                let tile_col = ctx.mul_u32_reg(ctaid_x, c_16);
                let tile_row = ctx.mul_u32_reg(ctaid_y, c_16);

                let m_param = ctx.load_param_u32("m_param");
                let n_param = ctx.load_param_u32("n_param");
                let k_param = ctx.load_param_u32("k_param");

                // Bounds check — skip OOB tiles
                let row_oob = ctx.setp_ge_u32(tile_row, m_param);
                ctx.branch_if(row_oob, "exit");
                let col_oob = ctx.setp_ge_u32(tile_col, n_param);
                ctx.branch_if(col_oob, "exit");

                let a_ptr = ctx.load_param_u64("a_ptr");
                let scales_ptr = ctx.load_param_u64("scales_ptr");
                let data_ptr = ctx.load_param_u64("data_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                let smem_a_base = c_0;
                let smem_b_base = ctx.mov_u32_imm(512); // 16×16×2 = 512 bytes

                // NF4 codebook in registers
                let lut: [_; 16] = std::array::from_fn(|i| ctx.mov_f32_imm(NF4_LUT[i]));

                // Initialize WMMA accumulator
                let frag_c = ctx.wmma_init_c_zero();

                let m_minus_1 = ctx.sub_u32_reg(m_param, c_1);
                let n_minus_1 = ctx.sub_u32_reg(n_param, c_1);
                let k_minus_1 = ctx.sub_u32_reg(k_param, c_1);
                let num_kb_reg = ctx.mov_u32_imm(num_k_blocks);

                let k_tile_idx = ctx.mov_u32_imm(0);
                let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);

                // ===== K-tile loop (process 16 columns of K per iteration) =====
                ctx.label("k_tile_loop");
                let k_done = ctx.setp_ge_u32(k_tile_idx, n_k_tiles_reg);
                ctx.branch_if(k_done, "k_tile_end");

                let k_offset = ctx.mul_u32_reg(k_tile_idx, c_16);

                // 32 threads × 8 elements = 256 = 16×16
                let my_start = ctx.mul_u32_reg(tid, c_8);

                // ====== PHASE 1: Load A[16×16] FP32 → FP16 SHMEM (row-major) ======
                let load_i = ctx.mov_u32_imm(0);
                ctx.label("load_a");
                let la_done = ctx.setp_ge_u32(load_i, c_8);
                ctx.branch_if(la_done, "load_a_end");

                let elem_a = ctx.add_u32_reg(my_start, load_i);
                let row_in_tile = ctx.shr_u32(elem_a, c_4); // /16
                let k_in_tile = ctx.and_u32(elem_a, c_15); // %16

                let smem_a_off = ctx.mul_u32_reg(elem_a, c_2);
                let smem_a_addr = ctx.add_u32_reg(smem_a_base, smem_a_off);

                let global_row = ctx.add_u32_reg(tile_row, row_in_tile);
                let global_k = ctx.add_u32_reg(k_offset, k_in_tile);
                let cr = ctx.min_u32(global_row, m_minus_1);
                let ck = ctx.min_u32(global_k, k_minus_1);

                let a_row_off = ctx.mul_wide_u32_reg(cr, k_param);
                let a_k_off = ctx.cvt_u64_u32(ck);
                let a_elem_off = ctx.add_u64(a_row_off, a_k_off);
                let a_byte_off = ctx.mul_u64(a_elem_off, 4);
                let a_addr = ctx.add_u64(a_ptr, a_byte_off);
                let a_f32 = ctx.ld_global_f32(a_addr);

                // Zero OOB
                let rv = ctx.setp_lt_u32(global_row, m_param);
                let kv = ctx.setp_lt_u32(global_k, k_param);
                let zero_f32 = ctx.mov_f32_imm(0.0);
                let a_m = ctx.selp_f32(rv, a_f32, zero_f32);
                let a_m2 = ctx.selp_f32(kv, a_m, zero_f32);
                let a_f16 = ctx.cvt_f16_f32(a_m2);
                ctx.st_shared_f16(smem_a_addr, a_f16);

                ctx.add_u32_inplace(load_i, 1);
                ctx.branch("load_a");
                ctx.label("load_a_end");

                // ====== PHASE 2: Dequant B_nf4[16×16] → FP16 SHMEM (col-major) ======
                let load_j = ctx.mov_u32_imm(0);
                ctx.label("load_b");
                let lb_done = ctx.setp_ge_u32(load_j, c_8);
                ctx.branch_if(lb_done, "load_b_end");

                let elem_b = ctx.add_u32_reg(my_start, load_j);
                let col_in_tile = ctx.shr_u32(elem_b, c_4);
                let k_in_tile_b = ctx.and_u32(elem_b, c_15);

                let smem_b_off = ctx.mul_u32_reg(elem_b, c_2);
                let smem_b_addr = ctx.add_u32_reg(smem_b_base, smem_b_off);

                let global_col = ctx.add_u32_reg(tile_col, col_in_tile);
                let global_k_b = ctx.add_u32_reg(k_offset, k_in_tile_b);
                let cc = ctx.min_u32(global_col, n_minus_1);

                // NF4 addressing: block_idx = global_k / 64, elem_in_block = global_k % 64
                let blk_idx = ctx.div_u32(global_k_b, NF4_BLOCK_SIZE_U32);
                let elem_in_blk = ctx.rem_u32(global_k_b, NF4_BLOCK_SIZE_U32);

                // Scale: scales_ptr[col * num_k_blocks + blk_idx]
                let col_blk_off = ctx.mul_u32_reg(cc, num_kb_reg);
                let scale_idx = ctx.add_u32_reg(col_blk_off, blk_idx);
                let scale_byte_off = ctx.mul_wide_u32_reg(scale_idx, c_4);
                let scale_addr = ctx.add_u64(scales_ptr, scale_byte_off);
                let scale = ctx.ld_global_f32(scale_addr);

                // Data: data_ptr[col * num_k_blocks * 32 + blk_idx * 32 + elem_in_blk / 2]
                let c_32 = ctx.mov_u32_imm(32);
                let col_data_base = ctx.mul_u32_reg(cc, num_kb_reg);
                let col_data_base = ctx.mul_u32_reg(col_data_base, c_32);
                let blk_data_off = ctx.mul_u32_reg(blk_idx, c_32);
                let byte_idx = ctx.div_u32(elem_in_blk, 2);
                let data_off = ctx.add_u32_reg(col_data_base, blk_data_off);
                let data_off = ctx.add_u32_reg(data_off, byte_idx);
                let data_off_64 = ctx.cvt_u64_u32(data_off);
                let data_addr = ctx.add_u64(data_ptr, data_off_64);
                let packed = ctx.ld_global_u8(data_addr);
                let packed_u32 = ctx.cvt_u32_u8(packed);

                // Extract nibble
                let is_high = ctx.rem_u32(elem_in_blk, 2);
                let shift = ctx.mul_u32_reg(is_high, c_4);
                let shifted = ctx.shr_u32(packed_u32, shift);
                let mask4 = ctx.mov_u32_imm(0xF);
                let nibble = ctx.and_u32(shifted, mask4);

                // NF4 LUT lookup + dequant
                let codebook_val = nf4_register_lut_lookup(ctx, nibble, &lut);
                let weight_f32 = ctx.mul_f32(scale, codebook_val);

                // Zero OOB
                let cv = ctx.setp_lt_u32(global_col, n_param);
                let w_m = ctx.selp_f32(cv, weight_f32, zero_f32);
                let w_m2 = ctx.selp_f32(kv, w_m, zero_f32);
                let w_f16 = ctx.cvt_f16_f32(w_m2);
                ctx.st_shared_f16(smem_b_addr, w_f16);

                ctx.add_u32_inplace(load_j, 1);
                ctx.branch("load_b");
                ctx.label("load_b_end");

                // Barrier before WMMA
                ctx.bar_sync(0);

                // ====== PHASE 3: WMMA mma.sync ======
                let frag_a = ctx.wmma_load_a_f16(smem_a_base, 16, WmmaLayout::RowMajor);
                let frag_b = ctx.wmma_load_b_f16(smem_b_base, 16, WmmaLayout::ColMajor);
                let frag_c = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);

                // Barrier before next tile
                ctx.bar_sync(1);

                ctx.add_u32_inplace(k_tile_idx, 1);
                ctx.branch("k_tile_loop");
                ctx.label("k_tile_end");

                // ====== PHASE 4: Store C[16×16] FP32 ======
                // C row-major: c_ptr + (tile_row * N + tile_col) * 4
                let c_tile_off = ctx.mul_wide_u32_reg(tile_row, n_param);
                let c_col_off = ctx.cvt_u64_u32(tile_col);
                let c_base = ctx.add_u64(c_tile_off, c_col_off);
                let c_base = ctx.mul_u64(c_base, 4);
                let c_addr = ctx.add_u64(c_ptr, c_base);

                ctx.wmma_store_d_f32(c_addr, &frag_c, n_const, WmmaLayout::RowMajor);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nf4_tc_gemm_name() {
        let k = Nf4TensorCoreGemmKernel::new(128, 1536, 1536);
        assert_eq!(k.name(), "nf4_tensor_core_gemm");
    }

    #[test]
    fn test_nf4_tc_gemm_ptx_emits() {
        let k = Nf4TensorCoreGemmKernel::new(128, 1536, 1536);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("nf4_tensor_core_gemm"));
        assert!(ptx.contains("wmma")); // tensor core
        assert!(ptx.contains("selp")); // NF4 LUT
    }
}
