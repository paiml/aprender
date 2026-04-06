// =============================================================================
// PMAT-475: FUSED RMSNORM + NF4 GEMV KERNEL
// =============================================================================
//
// Replicates the Q4K fused RMSNorm+GEMV pattern (PAR-030) for NF4 weights.
// Eliminates global memory roundtrip between RMSNorm output and GEMV input.
//
// Contract: nf4-fused-rmsnorm-gemv-v1.yaml
//   F-NF4-RMS-001: Fused output matches separate RMSNorm + NF4 GEMV within 1e-4
//   F-NF4-RMS-002: Zero intermediate global memory writes for normed output
//   F-NF4-RMS-003: Throughput >= 1.05x separate path

use super::super::nf4_cpu::{NF4_BLOCK_SIZE, NF4_LUT};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// NF4 block size as u32.
const NF4_BLOCK_SIZE_U32: u32 = NF4_BLOCK_SIZE as u32;
/// Bytes per NF4 block: 32 packed nibbles
const NF4_BLOCK_DATA_BYTES: u32 = (NF4_BLOCK_SIZE / 2) as u32;

/// Fused RMSNorm + NF4 GEMV kernel for training forward pass.
///
/// Computes `y = NF4_GEMV(W_nf4, RMSNorm(x, gamma))` in a single kernel:
/// - Phase 1: Cooperative RMSNorm (x → shared memory)
/// - Phase 2: NF4 dequant+GEMV from shared memory (zero DRAM roundtrip for normed input)
///
/// Memory bandwidth savings per call:
/// - Eliminates: k × 4 bytes write + k × 4 bytes read (RMSNorm→GEMV)
/// - For Qwen 1.5B (k=1536): saves 12 KB per projection
///
/// # Grid Configuration
///
/// - Block: 256 threads (cooperatively normalize, then GEMV)
/// - Grid: (n, 1, 1) — one block per output element
/// - Shared memory: k × 4 + 32 bytes (normed input + warp partials)
#[derive(Debug, Clone)]
pub struct FusedRmsNormNf4GemvKernel {
    /// K dimension (hidden size, must be divisible by 64)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Epsilon for RMSNorm numerical stability
    pub epsilon: f32,
}

impl FusedRmsNormNf4GemvKernel {
    /// Create a new fused RMSNorm + NF4 GEMV kernel.
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, epsilon: 1e-5 }
    }

    /// Set custom epsilon for RMSNorm.
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Number of NF4 blocks per column (K / 64).
    #[must_use]
    pub const fn num_blocks_per_col(&self) -> u32 {
        self.k / NF4_BLOCK_SIZE_U32
    }
}

impl Kernel for FusedRmsNormNf4GemvKernel {
    fn name(&self) -> &str {
        "fused_rmsnorm_nf4_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let k = self.k;
        let epsilon = self.epsilon;
        let num_k_blocks = k / NF4_BLOCK_SIZE_U32;

        // Shared memory: [0, k*4) = normalized input, [k*4, k*4+32) = warp partials
        let smem_size = (k * 4 + 32) as usize;

        PtxKernel::new("fused_rmsnorm_nf4_gemv")
            .param(PtxType::U64, "y_ptr") // Output [N]
            .param(PtxType::U64, "scales_ptr") // NF4 scales [N * num_k_blocks] f32
            .param(PtxType::U64, "data_ptr") // NF4 packed nibbles [N * num_k_blocks * 32] u8
            .param(PtxType::U64, "x_ptr") // Input [K] (unnormalized)
            .param(PtxType::U64, "gamma_ptr") // RMSNorm weights [K]
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .shared_memory(smem_size)
            .build(move |ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                // Bounds check
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                // Load parameters
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let scales_ptr = ctx.load_param_u64("scales_ptr");
                let data_ptr = ctx.load_param_u64("data_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                let four = ctx.mov_u32_imm(4);
                let one = ctx.mov_u32_imm(1);

                // ================================================================
                // PHASE 1: RMSNorm — cooperative load + normalize in shared memory
                // ================================================================
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("load_loop");
                let loop_idx = ctx.add_u32_reg(idx, thread_id);
                let in_bounds = ctx.setp_lt_u32(loop_idx, k_dim);
                ctx.branch_if_not(in_bounds, "load_loop_end");

                // Load x[loop_idx] and store to shared memory
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let x_addr = ctx.add_u64(x_ptr, elem_offset);
                let x_val = ctx.ld_global_f32(x_addr);
                ctx.st_shared_f32(elem_offset, x_val);

                // Accumulate sum of squares
                ctx.fma_f32_inplace(sq_sum, x_val, x_val);

                ctx.add_u32_inplace(idx, 256);
                ctx.branch("load_loop");
                ctx.label("load_loop_end");

                // Warp-level reduction of sq_sum
                let shfl16 = ctx.shfl_down_f32(sq_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl16);
                let shfl8 = ctx.shfl_down_f32(sq_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl8);
                let shfl4 = ctx.shfl_down_f32(sq_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl4);
                let shfl2 = ctx.shfl_down_f32(sq_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl2);
                let shfl1 = ctx.shfl_down_f32(sq_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl1);

                // Warp lane/id
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);

                // Store warp partial sums to shared memory after input buffer
                let k_bytes = ctx.mul_u32_reg(k_dim, four);
                let warp_sum_offset = ctx.mul_wide_u32_reg(warp_id, four);
                let k_bytes_64 = ctx.cvt_u64_u32(k_bytes);
                let warp_sum_addr = ctx.add_u64(k_bytes_64, warp_sum_offset);

                ctx.branch_if_not(is_lane0, "skip_warp_write");
                ctx.st_shared_f32(warp_sum_addr, sq_sum);
                ctx.label("skip_warp_write");

                ctx.bar_sync(0);

                // Thread 0: sum warp partials → rms_inv
                let is_thread0 = ctx.setp_lt_u32(thread_id, one);
                let total_sq = ctx.mov_f32_imm(0.0);

                ctx.branch_if_not(is_thread0, "skip_rms_reduce");
                for warp in 0..8u32 {
                    let w_off = ctx.mov_u64_imm((warp * 4) as u64);
                    let w_addr = ctx.add_u64(k_bytes_64, w_off);
                    let w_sum = ctx.ld_shared_f32(w_addr);
                    ctx.add_f32_inplace(total_sq, w_sum);
                }

                let k_f32 = ctx.cvt_f32_u32(k_dim);
                let mean_sq = ctx.div_f32(total_sq, k_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let mean_sq_eps = ctx.add_f32(mean_sq, eps);
                let rms_inv = ctx.rsqrt_f32(mean_sq_eps);

                // Broadcast rms_inv via shared memory offset 0
                let zero_off = ctx.mov_u64_imm(0);
                ctx.st_shared_f32(zero_off, rms_inv);

                ctx.label("skip_rms_reduce");
                ctx.bar_sync(1);

                // All threads load rms_inv
                let rms_inv_off = ctx.mov_u64_imm(0);
                let rms_inv_val = ctx.ld_shared_f32(rms_inv_off);

                // Normalize x in shared memory: x_norm = x * rms_inv * gamma
                let idx2 = ctx.mov_u32_imm(0);
                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, thread_id);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, k_dim);
                ctx.branch_if_not(in_bounds2, "norm_loop_end");

                let elem_off2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let x_smem = ctx.ld_shared_f32(elem_off2);
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_off2);
                let gamma_val = ctx.ld_global_f32(gamma_addr);
                let normalized = ctx.mul_f32(x_smem, rms_inv_val);
                let scaled = ctx.mul_f32(normalized, gamma_val);
                ctx.st_shared_f32(elem_off2, scaled);

                ctx.add_u32_inplace(idx2, 256);
                ctx.branch("norm_loop");
                ctx.label("norm_loop_end");

                ctx.bar_sync(2);

                // ================================================================
                // PHASE 2: NF4 GEMV using normed input from shared memory
                // ================================================================
                // Load NF4 codebook into 16 f32 registers (zero memory access)
                let lut_regs: [_; 16] = std::array::from_fn(|i| ctx.mov_f32_imm(NF4_LUT[i]));

                let acc = ctx.mov_f64_imm_zero(); // f64 accumulator for precision

                // Row base for this output element (block_id)
                let num_k_blocks_reg = ctx.mov_u32_imm(num_k_blocks);

                // Scale row base: scales_ptr + block_id * num_k_blocks * 4
                let scale_row_elems = ctx.mul_u32_reg(block_id, num_k_blocks_reg);
                let scale_row_offset = ctx.mul_wide_u32_reg(scale_row_elems, four);
                let scale_row_base = ctx.add_u64(scales_ptr, scale_row_offset);

                // Data row base: data_ptr + block_id * num_k_blocks * 32
                let thirty_two = ctx.mov_u32_imm(NF4_BLOCK_DATA_BYTES);
                let data_row_elems = ctx.mul_u32_reg(block_id, num_k_blocks_reg);
                let data_row_blocks = ctx.mul_u32_reg(data_row_elems, thirty_two);
                let data_row_offset = ctx.cvt_u64_u32(data_row_blocks);
                let data_row_base = ctx.add_u64(data_ptr, data_row_offset);

                // Each thread handles NF4 blocks at stride 256/64 = 4 blocks apart
                // But we need per-thread accumulation over all K elements
                // Strategy: each thread processes K/256 elements (strided by 256)
                let elem_idx = ctx.mov_u32_imm(0);

                ctx.label("nf4_loop");
                let e_idx = ctx.add_u32_reg(elem_idx, thread_id);
                let in_bounds3 = ctx.setp_lt_u32(e_idx, k_dim);
                ctx.branch_if_not(in_bounds3, "nf4_loop_end");

                // Which NF4 block does this element belong to?
                let block_idx = ctx.div_u32(e_idx, NF4_BLOCK_SIZE_U32);
                let elem_in_block = ctx.rem_u32(e_idx, NF4_BLOCK_SIZE_U32);

                // Load scale for this block
                let scale_off = ctx.mul_wide_u32_reg(block_idx, four);
                let scale_addr = ctx.add_u64(scale_row_base, scale_off);
                let scale = ctx.ld_global_f32(scale_addr);

                // Load packed byte: data[block_idx * 32 + elem_in_block / 2]
                let byte_idx = ctx.div_u32(elem_in_block, 2);
                let block_data_off = ctx.mul_u32_reg(block_idx, thirty_two);
                let byte_off_in_data = ctx.add_u32_reg(block_data_off, byte_idx);
                let byte_off_64 = ctx.cvt_u64_u32(byte_off_in_data);
                let byte_addr = ctx.add_u64(data_row_base, byte_off_64);
                let packed_byte = ctx.ld_global_u8(byte_addr);
                let packed_u32 = ctx.cvt_u32_u8(packed_byte);

                // Extract nibble (low or high)
                let is_high = ctx.rem_u32(elem_in_block, 2);
                let shift_amt = ctx.mul_u32_reg(is_high, four);
                let shifted = ctx.shr_u32(packed_u32, shift_amt);
                let mask_4bit = ctx.mov_u32_imm(0xF);
                let nibble = ctx.and_u32(shifted, mask_4bit);

                // Register LUT lookup: NF4_LUT[nibble]
                let codebook_val =
                    super::super::nf4::nf4_register_lut_lookup(ctx, nibble, &lut_regs);

                // Dequantize: weight = scale * codebook_val
                let weight = ctx.mul_f32(scale, codebook_val);

                // Load normed activation from shared memory
                let x_smem_off = ctx.mul_wide_u32_reg(e_idx, four);
                let x_norm = ctx.ld_shared_f32(x_smem_off);

                // FMA into f64 accumulator: acc += weight * x_norm
                ctx.fma_f64_acc_inplace(acc, x_norm, weight);

                ctx.add_u32_inplace(elem_idx, 256);
                ctx.branch("nf4_loop");
                ctx.label("nf4_loop_end");

                // ================================================================
                // PHASE 3: Warp reduction and final output
                // ================================================================
                // Convert f64 acc to f32 for reduction
                let acc_f32 = ctx.cvt_f32_f64_rn(acc);

                // Warp reduction
                let w_shfl16 = ctx.shfl_down_f32(acc_f32, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_f32, w_shfl16);
                let w_shfl8 = ctx.shfl_down_f32(acc_f32, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_f32, w_shfl8);
                let w_shfl4 = ctx.shfl_down_f32(acc_f32, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_f32, w_shfl4);
                let w_shfl2 = ctx.shfl_down_f32(acc_f32, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_f32, w_shfl2);
                let w_shfl1 = ctx.shfl_down_f32(acc_f32, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_f32, w_shfl1);

                // Store warp partials
                let warp_acc_off = ctx.mul_wide_u32_reg(warp_id, four);
                let warp_acc_addr = ctx.add_u64(k_bytes_64, warp_acc_off);

                ctx.branch_if_not(is_lane0, "skip_warp_acc");
                ctx.st_shared_f32(warp_acc_addr, acc_f32);
                ctx.label("skip_warp_acc");

                ctx.bar_sync(3);

                // Thread 0: sum warps and store output
                ctx.branch_if_not(is_thread0, "exit");

                let final_val = ctx.mov_f32_imm(0.0);
                for warp in 0..8u32 {
                    let w_off = ctx.mov_u64_imm((warp * 4) as u64);
                    let w_addr = ctx.add_u64(k_bytes_64, w_off);
                    let w_acc = ctx.ld_shared_f32(w_addr);
                    ctx.add_f32_inplace(final_val, w_acc);
                }

                // Store y[block_id]
                let y_off = ctx.mul_wide_u32(block_id, 4);
                let y_addr = ctx.add_u64(y_ptr, y_off);
                ctx.st_global_f32(y_addr, final_val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_rmsnorm_nf4_gemv_name() {
        let kernel = FusedRmsNormNf4GemvKernel::new(1536, 1536);
        assert_eq!(kernel.name(), "fused_rmsnorm_nf4_gemv");
    }

    #[test]
    fn test_fused_rmsnorm_nf4_gemv_ptx_emits() {
        let kernel = FusedRmsNormNf4GemvKernel::new(1536, 1536);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("fused_rmsnorm_nf4_gemv"));
        assert!(ptx.contains("scales_ptr"));
        assert!(ptx.contains("gamma_ptr"));
        assert!(ptx.contains("selp")); // NF4 register LUT
        assert!(ptx.contains("rsqrt")); // RMSNorm
    }

    #[test]
    fn test_fused_rmsnorm_nf4_gemv_num_blocks() {
        let kernel = FusedRmsNormNf4GemvKernel::new(1536, 256);
        assert_eq!(kernel.num_blocks_per_col(), 24); // 1536 / 64
    }
}
