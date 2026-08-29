// =============================================================================
// NF4 BACKWARD TENSOR CORE GEMM — WMMA 16×16×16 backward with NF4 dequant
// =============================================================================
//
// Backward analog of forward Nf4TensorCoreGemmKernel (PMAT-479).
// Computes: grad_A[M×K] = grad_out[M×N] @ dequant(B_nf4[K×N])^T
//
// Forward: C = A @ B        (shared dim = K)
// Backward: grad_A = grad_out @ B^T  (shared dim = N)
//
// Key difference: B is transposed. NF4 data layout is unchanged — we access
// the same B_nf4[k, n] elements but with transposed SHMEM addressing.
//
// Contract: nf4-backward-tensor-core-gemm-v1.yaml
// Refs: PMAT-481, PMAT-484, trueno#236

use crate::kernels::quantize::{nf4_register_lut_lookup, NF4_BLOCK_SIZE, NF4_LUT};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, WmmaLayout};

const NF4_BLOCK_SIZE_U32: u32 = NF4_BLOCK_SIZE as u32;

/// NF4 backward tensor core GEMM — WMMA with inline NF4 dequantization.
///
/// Computes `grad_A[M×K] = grad_out[M×N] @ dequant(B_nf4[K×N])^T` using WMMA 16×16×16.
///
/// Phase 1: Load grad_out[16×16] FP32→FP16 into shared memory (row-major)
/// Phase 2: Dequant B_nf4[16×16] to FP16 into shared memory (col-major for B^T)
/// Phase 3: WMMA mma.sync with FP32 accumulator
/// Phase 4: Store grad_A[16×16] FP32 to global memory
///
/// Grid: (ceil(K/16), ceil(M/16)), Block: 32 threads (1 warp)
/// SHMEM: 1024 bytes (grad_out[16×16] FP16 + B^T[16×16] FP16)
///
/// Eliminates separate dequant kernel + generic cuBLAS GEMM per backward projection.
/// 28 layers × 7 projections = 196 kernel launches saved per training step.
#[derive(Debug, Clone)]
pub struct Nf4TensorCoreGemmBackwardAKernel {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl Nf4TensorCoreGemmBackwardAKernel {
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    #[must_use]
    pub const fn num_k_blocks(&self) -> u32 {
        self.k / NF4_BLOCK_SIZE_U32
    }
}

impl Kernel for Nf4TensorCoreGemmBackwardAKernel {
    fn name(&self) -> &str {
        "nf4_tensor_core_gemm_backward_a"
    }

    #[allow(clippy::too_many_lines)]
    fn build_ptx(&self) -> PtxKernel {
        // B_nf4 is [K, N]. Backward shared dim = N, output dim = K.
        let k_const = self.k;
        let n_const = self.n;
        let n_n_tiles = n_const / 16; // tiles in shared dimension (N)
        let num_k_blocks = k_const / NF4_BLOCK_SIZE_U32;
        let smem_bytes = 16 * 16 * 2 * 2; // grad_out[16×16] + B^T[16×16] in FP16

        PtxKernel::new("nf4_tensor_core_gemm_backward_a")
            .max_regs(96)
            .param(PtxType::U64, "grad_out_ptr") // grad_out [M×N] FP32
            .param(PtxType::U64, "scales_ptr") // NF4 scales [N × num_k_blocks] f32
            .param(PtxType::U64, "data_ptr") // NF4 packed nibbles
            .param(PtxType::U64, "grad_a_ptr") // Output grad_A [M×K] FP32
            .param(PtxType::U32, "m_param")
            .param(PtxType::U32, "n_param")
            .param(PtxType::U32, "k_param")
            .shared_memory(smem_bytes as usize)
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX); // K tile (output col)
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY); // M tile (output row)

                let c_0 = ctx.mov_u32_imm(0);
                let c_1 = ctx.mov_u32_imm(1);
                let c_2 = ctx.mov_u32_imm(2);
                let c_4 = ctx.mov_u32_imm(4);
                let c_8 = ctx.mov_u32_imm(8);
                let c_15 = ctx.mov_u32_imm(15);
                let c_16 = ctx.mov_u32_imm(16);

                // Output tile position: grad_A[tile_row..tile_row+16, tile_k..tile_k+16]
                let tile_k = ctx.mul_u32_reg(ctaid_x, c_16); // K output column
                let tile_row = ctx.mul_u32_reg(ctaid_y, c_16); // M output row

                let m_param = ctx.load_param_u32("m_param");
                let n_param = ctx.load_param_u32("n_param");
                let k_param = ctx.load_param_u32("k_param");

                // Bounds check — skip OOB tiles
                let row_oob = ctx.setp_ge_u32(tile_row, m_param);
                ctx.branch_if(row_oob, "exit");
                let k_oob = ctx.setp_ge_u32(tile_k, k_param);
                ctx.branch_if(k_oob, "exit");

                let grad_out_ptr = ctx.load_param_u64("grad_out_ptr");
                let scales_ptr = ctx.load_param_u64("scales_ptr");
                let data_ptr = ctx.load_param_u64("data_ptr");
                let grad_a_ptr = ctx.load_param_u64("grad_a_ptr");

                let smem_a_base = c_0; // grad_out tile
                let smem_b_base = ctx.mov_u32_imm(512); // B^T tile (16×16×2 = 512 bytes)

                // NF4 codebook in registers (19 selp instructions)
                let lut: [_; 16] = std::array::from_fn(|i| ctx.mov_f32_imm(NF4_LUT[i]));

                // Initialize WMMA accumulator to zero
                let frag_c = ctx.wmma_init_c_zero();

                let m_minus_1 = ctx.sub_u32_reg(m_param, c_1);
                let n_minus_1 = ctx.sub_u32_reg(n_param, c_1);
                let k_minus_1 = ctx.sub_u32_reg(k_param, c_1);
                let num_kb_reg = ctx.mov_u32_imm(num_k_blocks);

                let n_tile_idx = ctx.mov_u32_imm(0);
                let n_n_tiles_reg = ctx.mov_u32_imm(n_n_tiles);

                // ===== N-tile loop (shared dimension, process 16 columns of N per iteration) =====
                ctx.label("n_tile_loop");
                let n_done = ctx.setp_ge_u32(n_tile_idx, n_n_tiles_reg);
                ctx.branch_if(n_done, "n_tile_end");

                let n_offset = ctx.mul_u32_reg(n_tile_idx, c_16);

                // 32 threads × 8 elements = 256 = 16×16
                let my_start = ctx.mul_u32_reg(tid, c_8);

                // ====== PHASE 1: Load grad_out[16×16] FP32 → FP16 SHMEM (row-major) ======
                // grad_out[M, N], row-major, stride = N
                let load_i = ctx.mov_u32_imm(0);
                ctx.label("load_grad_out");
                let la_done = ctx.setp_ge_u32(load_i, c_8);
                ctx.branch_if(la_done, "load_grad_out_end");

                let elem_a = ctx.add_u32_reg(my_start, load_i);
                let row_in_tile = ctx.shr_u32(elem_a, c_4); // /16
                let n_in_tile = ctx.and_u32(elem_a, c_15); // %16

                let smem_a_off = ctx.mul_u32_reg(elem_a, c_2);
                let smem_a_addr = ctx.add_u32_reg(smem_a_base, smem_a_off);

                let global_row = ctx.add_u32_reg(tile_row, row_in_tile);
                let global_n = ctx.add_u32_reg(n_offset, n_in_tile);
                let cr = ctx.min_u32(global_row, m_minus_1);
                let cn = ctx.min_u32(global_n, n_minus_1);

                // grad_out[global_row, global_n] at stride N
                let go_row_off = ctx.mul_wide_u32_reg(cr, n_param);
                let go_n_off = ctx.cvt_u64_u32(cn);
                let go_elem_off = ctx.add_u64(go_row_off, go_n_off);
                let go_byte_off = ctx.mul_u64(go_elem_off, 4);
                let go_addr = ctx.add_u64(grad_out_ptr, go_byte_off);
                let go_f32 = ctx.ld_global_f32(go_addr);

                // Zero OOB elements
                let rv = ctx.setp_lt_u32(global_row, m_param);
                let nv = ctx.setp_lt_u32(global_n, n_param);
                let zero_f32 = ctx.mov_f32_imm(0.0);
                let go_m = ctx.selp_f32(rv, go_f32, zero_f32);
                let go_m2 = ctx.selp_f32(nv, go_m, zero_f32);
                let go_f16 = ctx.cvt_f16_f32(go_m2);
                ctx.st_shared_f16(smem_a_addr, go_f16);

                ctx.add_u32_inplace(load_i, 1);
                ctx.branch("load_grad_out");
                ctx.label("load_grad_out_end");

                // ====== PHASE 2: Dequant B_nf4 → FP16 SHMEM as B^T col-major ======
                // B^T[n, k] = B[k, n] = dequant(B_nf4[k, n])
                // Col-major B^T: SHMEM_B[k_in_tile * 16 + n_in_tile] = B[tile_k + k_in_tile, n_offset + n_in_tile]
                let load_j = ctx.mov_u32_imm(0);
                ctx.label("load_b_bwd");
                let lb_done = ctx.setp_ge_u32(load_j, c_8);
                ctx.branch_if(lb_done, "load_b_bwd_end");

                let elem_b = ctx.add_u32_reg(my_start, load_j);
                // Decompose: k_out_in_tile = elem_b / 16, n_shared_in_tile = elem_b % 16
                let k_out_in_tile = ctx.shr_u32(elem_b, c_4);
                let n_shared_in_tile = ctx.and_u32(elem_b, c_15);

                let smem_b_off = ctx.mul_u32_reg(elem_b, c_2);
                let smem_b_addr = ctx.add_u32_reg(smem_b_base, smem_b_off);

                // Global B coordinates: B[tile_k + k_out_in_tile, n_offset + n_shared_in_tile]
                let global_k_out = ctx.add_u32_reg(tile_k, k_out_in_tile);
                let global_n_shared = ctx.add_u32_reg(n_offset, n_shared_in_tile);
                let _ck = ctx.min_u32(global_k_out, k_minus_1);
                let cn_b = ctx.min_u32(global_n_shared, n_minus_1);

                // NF4 addressing: block_idx = global_k_out / 64
                let blk_idx = ctx.div_u32(global_k_out, NF4_BLOCK_SIZE_U32);
                let elem_in_blk = ctx.rem_u32(global_k_out, NF4_BLOCK_SIZE_U32);

                // Scale: scales_ptr[global_n * num_k_blocks + blk_idx]
                let n_blk_off = ctx.mul_u32_reg(cn_b, num_kb_reg);
                let scale_idx = ctx.add_u32_reg(n_blk_off, blk_idx);
                let scale_byte_off = ctx.mul_wide_u32_reg(scale_idx, c_4);
                let scale_addr = ctx.add_u64(scales_ptr, scale_byte_off);
                let scale = ctx.ld_global_f32(scale_addr);

                // Data: data_ptr[global_n * num_k_blocks * 32 + blk_idx * 32 + elem_in_blk / 2]
                let c_32 = ctx.mov_u32_imm(32);
                let n_data_base = ctx.mul_u32_reg(cn_b, num_kb_reg);
                let n_data_base = ctx.mul_u32_reg(n_data_base, c_32);
                let blk_data_off = ctx.mul_u32_reg(blk_idx, c_32);
                let byte_idx = ctx.div_u32(elem_in_blk, 2);
                let data_off = ctx.add_u32_reg(n_data_base, blk_data_off);
                let data_off = ctx.add_u32_reg(data_off, byte_idx);
                let data_off_64 = ctx.cvt_u64_u32(data_off);
                let data_addr = ctx.add_u64(data_ptr, data_off_64);
                let packed = ctx.ld_global_u8(data_addr);
                let packed_u32 = ctx.cvt_u32_u8(packed);

                // Extract nibble (same as forward)
                let is_high = ctx.rem_u32(elem_in_blk, 2);
                let shift = ctx.mul_u32_reg(is_high, c_4);
                let shifted = ctx.shr_u32(packed_u32, shift);
                let mask4 = ctx.mov_u32_imm(0xF);
                let nibble = ctx.and_u32(shifted, mask4);

                // NF4 LUT lookup + dequant
                let codebook_val = nf4_register_lut_lookup(ctx, nibble, &lut);
                let weight_f32 = ctx.mul_f32(scale, codebook_val);

                // Zero OOB
                let kv = ctx.setp_lt_u32(global_k_out, k_param);
                let nv_b = ctx.setp_lt_u32(global_n_shared, n_param);
                let w_m = ctx.selp_f32(kv, weight_f32, zero_f32);
                let w_m2 = ctx.selp_f32(nv_b, w_m, zero_f32);
                let w_f16 = ctx.cvt_f16_f32(w_m2);
                ctx.st_shared_f16(smem_b_addr, w_f16);

                ctx.add_u32_inplace(load_j, 1);
                ctx.branch("load_b_bwd");
                ctx.label("load_b_bwd_end");

                // Barrier before WMMA
                ctx.bar_sync(0);

                // ====== PHASE 3: WMMA mma.sync ======
                // frag_a = grad_out tile (row-major, 16×16)
                // frag_b = B^T tile (col-major, 16×16)
                // grad_A[tile] += grad_out[tile] @ B^T[tile]
                let frag_a = ctx.wmma_load_a_f16(smem_a_base, 16, WmmaLayout::RowMajor);
                let frag_b = ctx.wmma_load_b_f16(smem_b_base, 16, WmmaLayout::ColMajor);
                let frag_c = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);

                // Barrier before next tile
                ctx.bar_sync(1);

                ctx.add_u32_inplace(n_tile_idx, 1);
                ctx.branch("n_tile_loop");
                ctx.label("n_tile_end");

                // ====== PHASE 4: Store grad_A[16×16] FP32 ======
                // grad_A row-major: grad_a_ptr + (tile_row * K + tile_k) * 4
                let ga_tile_off = ctx.mul_wide_u32_reg(tile_row, k_param);
                let ga_k_off = ctx.cvt_u64_u32(tile_k);
                let ga_base = ctx.add_u64(ga_tile_off, ga_k_off);
                let ga_base = ctx.mul_u64(ga_base, 4);
                let ga_addr = ctx.add_u64(grad_a_ptr, ga_base);

                ctx.wmma_store_d_f32(ga_addr, &frag_c, k_const, WmmaLayout::RowMajor);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nf4_tc_gemm_backward_a_name() {
        let k = Nf4TensorCoreGemmBackwardAKernel::new(128, 1536, 1536);
        assert_eq!(k.name(), "nf4_tensor_core_gemm_backward_a");
    }

    #[test]
    fn test_nf4_tc_gemm_backward_a_ptx_emits() {
        let k = Nf4TensorCoreGemmBackwardAKernel::new(128, 1536, 1536);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("nf4_tensor_core_gemm_backward_a"));
        assert!(ptx.contains("wmma")); // tensor core
        assert!(ptx.contains("selp")); // NF4 LUT
        assert!(ptx.contains("grad_out_ptr"));
        assert!(ptx.contains("grad_a_ptr"));
    }

    #[test]
    fn test_nf4_tc_gemm_backward_a_ptx_valid() {
        let k = Nf4TensorCoreGemmBackwardAKernel::new(128, 1536, 1536);
        let ptx = k.emit_ptx();
        assert!(ptx.contains(".entry"));
        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".target"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn test_nf4_tc_gemm_backward_a_barrier_safety() {
        let k = Nf4TensorCoreGemmBackwardAKernel::new(128, 1536, 1536);
        let result = k.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "NF4 backward TC GEMM should be barrier-safe: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_nf4_tc_gemm_backward_a_num_k_blocks() {
        let k = Nf4TensorCoreGemmBackwardAKernel::new(128, 1536, 1536);
        assert_eq!(k.num_k_blocks(), 1536 / 64); // 24 blocks
    }

    #[test]
    fn test_nf4_tc_gemm_backward_a_clone_debug() {
        let k = Nf4TensorCoreGemmBackwardAKernel::new(64, 256, 128);
        let cloned = k.clone();
        assert_eq!(k.m, cloned.m);
        assert_eq!(k.n, cloned.n);
        assert_eq!(k.k, cloned.k);

        let debug = format!("{k:?}");
        assert!(debug.contains("Nf4TensorCoreGemmBackwardAKernel"));
        assert!(debug.contains("64"));
    }

    #[test]
    fn test_nf4_tc_gemm_backward_a_small_dims() {
        // Edge case: small dimensions (still must be multiples of 16 for WMMA)
        let k = Nf4TensorCoreGemmBackwardAKernel::new(16, 64, 64);
        let ptx = k.emit_ptx();
        assert!(ptx.contains(".entry"));
    }

    #[test]
    fn test_nf4_tc_gemm_backward_a_qwen_dims() {
        // Qwen 1.5B dimensions
        // Q/K/V projection backward: grad_out[S, H] @ W[H, H]^T, H=1536
        let q_bwd = Nf4TensorCoreGemmBackwardAKernel::new(512, 1536, 1536);
        assert!(q_bwd.emit_ptx().contains(".entry"));

        // K/V projection backward (GQA): grad_out[S, D_kv] @ W[D_kv, H]^T, D_kv=256
        let kv_bwd = Nf4TensorCoreGemmBackwardAKernel::new(512, 256, 1536);
        assert!(kv_bwd.emit_ptx().contains(".entry"));

        // Gate/Up projection backward: grad_out[S, I] @ W[I, H]^T, I=4608
        let gate_bwd = Nf4TensorCoreGemmBackwardAKernel::new(512, 4608, 1536);
        assert!(gate_bwd.emit_ptx().contains(".entry"));

        // Down projection backward: grad_out[S, H] @ W[H, I]^T
        let down_bwd = Nf4TensorCoreGemmBackwardAKernel::new(512, 1536, 4608);
        assert!(down_bwd.emit_ptx().contains(".entry"));
    }
}
