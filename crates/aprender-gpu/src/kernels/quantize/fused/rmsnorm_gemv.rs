// =============================================================================
// PAR-030: FUSED RMSNORM + Q4K GEMV KERNEL
// =============================================================================

use super::super::{Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Fused RMSNorm + Q4_K GEMV kernel for decode throughput optimization
///
/// This kernel eliminates the global memory roundtrip between RMSNorm and GEMV:
/// - Standard flow: RMSNorm -> global write -> global read -> Q4K GEMV
/// - Fused flow: RMSNorm in shared memory -> Q4K GEMV from shared memory
///
/// Memory bandwidth savings:
/// - Eliminates: hidden_size x 4 bytes write + hidden_size x 4 bytes read per output
/// - For Qwen 3B (hidden=3584): saves 28KB per GEMV call
///
/// # Grid Configuration
///
/// - Block: 256 threads (processes hidden_size elements cooperatively)
/// - Grid: N blocks (one per output element)
/// - Shared memory: hidden_size x 4 bytes for normalized input cache
#[derive(Debug, Clone)]
pub struct FusedRmsNormQ4KGemvKernel {
    /// K dimension (hidden size, input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Epsilon for RMSNorm numerical stability
    pub epsilon: f32,
}

impl FusedRmsNormQ4KGemvKernel {
    /// Create a new fused RMSNorm + Q4_K GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self {
            k,
            n,
            epsilon: 1e-5,
        }
    }

    /// Set custom epsilon value for RMSNorm
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl Kernel for FusedRmsNormQ4KGemvKernel {
    fn name(&self) -> &str {
        "fused_rmsnorm_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let k = self.k;
        let epsilon = self.epsilon;

        // Shared memory layout:
        // - [0, k*4): Normalized input vector (k floats)
        // - [k*4, k*4+32): Warp partial sums (8 floats for 8 warps)
        let smem_size = (k * 4 + 32) as usize;

        PtxKernel::new("fused_rmsnorm_q4k_gemv")
            .param(PtxType::U64, "y_ptr") // Output vector (N)
            .param(PtxType::U64, "w_ptr") // Q4_K weights (N x K/256 super-blocks)
            .param(PtxType::U64, "x_ptr") // Input vector (K) - raw, not normalized
            .param(PtxType::U64, "gamma_ptr") // RMSNorm scale weights (K)
            .param(PtxType::U32, "k_dim") // K dimension
            .param(PtxType::U32, "n_dim") // N dimension
            .shared_memory(smem_size)
            .build(move |ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                // Bounds check: if block_id >= n_dim, exit early
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                // Load parameters
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Constants
                let four = ctx.mov_u32_imm(4);
                let one = ctx.mov_u32_imm(1);

                // ================================================================
                // PHASE 1: Cooperatively load input and compute sum of squares
                // ================================================================
                // Each thread handles k/256 elements: thread_id, thread_id+256, ...
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("load_loop");
                let loop_idx = ctx.add_u32_reg(idx, thread_id);
                let in_bounds = ctx.setp_lt_u32(loop_idx, k_dim);
                ctx.branch_if_not(in_bounds, "load_loop_end");

                // Load x[loop_idx]
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let x_addr = ctx.add_u64(x_ptr, elem_offset);
                let x_val = ctx.ld_global_f32(x_addr);

                // Store to shared memory (will normalize later)
                ctx.st_shared_f32(elem_offset, x_val);

                // Accumulate sq_sum += x_val * x_val
                ctx.fma_f32_inplace(sq_sum, x_val, x_val);

                // idx += 256 (stride by block size)
                ctx.add_u32_inplace(idx, 256);
                ctx.branch("load_loop");

                ctx.label("load_loop_end");

                // Block-level reduction for sq_sum using warp shuffles
                // First, warp-level reduction within each warp
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

                // Store warp partial sums to shared memory (reuse end of buffer)
                // Using bytes at offset k*4 to k*4+32 for 8 warp sums
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);

                // Warp-local storage region after input buffer: k_dim * 4 + warp_id * 4
                let k_bytes = ctx.mul_u32_reg(k_dim, four);
                let warp_sum_offset = ctx.mul_wide_u32_reg(warp_id, four);
                let k_bytes_64 = ctx.cvt_u64_u32(k_bytes);
                let warp_sum_addr = ctx.add_u64(k_bytes_64, warp_sum_offset);

                // Only lane 0 of each warp writes
                ctx.branch_if_not(is_lane0, "skip_warp_write");
                ctx.st_shared_f32(warp_sum_addr, sq_sum);
                ctx.label("skip_warp_write");

                // Synchronize all threads
                ctx.bar_sync(0);

                // Thread 0 sums warp partial results
                let is_thread0 = ctx.setp_lt_u32(thread_id, one);
                let total_sq_sum = ctx.mov_f32_imm(0.0);

                ctx.branch_if_not(is_thread0, "skip_final_reduce");

                // Sum 8 warp partial sums
                for warp in 0..8u32 {
                    let warp_offset = ctx.mov_u64_imm((warp * 4) as u64);
                    let addr = ctx.add_u64(k_bytes_64, warp_offset);
                    let warp_sum = ctx.ld_shared_f32(addr);
                    ctx.add_f32_inplace(total_sq_sum, warp_sum);
                }

                // Compute rms_inv = rsqrt(mean_sq + epsilon)
                let k_f32 = ctx.cvt_f32_u32(k_dim);
                let mean_sq = ctx.div_f32(total_sq_sum, k_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let mean_sq_eps = ctx.add_f32(mean_sq, eps);
                let rms_inv = ctx.rsqrt_f32(mean_sq_eps);

                // Store rms_inv to shared memory for broadcast
                let rms_inv_offset = ctx.mov_u64_imm(0); // Reuse offset 0 temporarily
                ctx.st_shared_f32(rms_inv_offset, rms_inv);

                ctx.label("skip_final_reduce");

                // Synchronize to ensure rms_inv is available
                ctx.bar_sync(1);

                // All threads load rms_inv
                let rms_inv_broadcast_offset = ctx.mov_u64_imm(0);
                let rms_inv_val = ctx.ld_shared_f32(rms_inv_broadcast_offset);

                // ================================================================
                // PHASE 2: Normalize input in shared memory
                // ================================================================
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, thread_id);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, k_dim);
                ctx.branch_if_not(in_bounds2, "norm_loop_end");

                // Load x from shared memory
                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let x_smem = ctx.ld_shared_f32(elem_offset2);

                // Load gamma
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);
                let gamma = ctx.ld_global_f32(gamma_addr);

                // Normalize: x_norm = x * rms_inv * gamma
                let normalized = ctx.mul_f32(x_smem, rms_inv_val);
                let scaled = ctx.mul_f32(normalized, gamma);

                // Store back to shared memory
                ctx.st_shared_f32(elem_offset2, scaled);

                ctx.add_u32_inplace(idx2, 256);
                ctx.branch("norm_loop");

                ctx.label("norm_loop_end");

                // Synchronize before GEMV phase
                ctx.bar_sync(2);

                // ================================================================
                // PHASE 3: Q4K GEMV using normalized input from shared memory
                // ================================================================
                // Each block computes one output y[block_id]
                // All threads cooperate to process super-blocks

                let acc = ctx.mov_f32_imm(0.0);
                // Ceiling division: (k + 255) / 256 for GGUF super-block count
                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                // Calculate row base for this output element
                let sb_bytes = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Each thread handles elements: thread_id/32 within warp, strided
                // For 256 threads and 256 values per super-block: each thread gets 1 value
                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_end");

                let sb_offset = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d and dmin
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // Load scales (12 bytes at offset 4)
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);

                // Each thread processes one value at position thread_id within super-block
                // Determine sub-block (0-7)
                let sub_block = ctx.div_u32(thread_id, 32);

                // Load scale bytes for this sub-block
                // Scale extraction (same as Q4KGemvKernel):
                // Blocks 0-3: scale = scales[sub_block] & 63
                // Blocks 4-7: more complex extraction
                let four_cmp = ctx.mov_u32_imm(4);
                let sub_block_lt_4 = ctx.setp_lt_u32(sub_block, four_cmp);

                // Load necessary scale bytes
                let sub_block_64 = ctx.cvt_u64_u32(sub_block);
                let scale_byte_addr = ctx.add_u64(scales_base, sub_block_64);
                let scale_byte = ctx.ld_global_u8(scale_byte_addr);
                let scale_byte_32 = ctx.cvt_u32_u8(scale_byte);

                let four_reg = ctx.mov_u32_imm(4);
                let sub_block_plus_4 = ctx.add_u32_reg(sub_block, four_reg);
                let sub_block_plus_4_64 = ctx.cvt_u64_u32(sub_block_plus_4);
                let min_byte_addr = ctx.add_u64(scales_base, sub_block_plus_4_64);
                let min_byte = ctx.ld_global_u8(min_byte_addr);
                let min_byte_32 = ctx.cvt_u32_u8(min_byte);

                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let six = ctx.mov_u32_imm(6);

                // Simple path for blocks 0-3
                let scale_simple = ctx.and_u32(scale_byte_32, mask_6bit);
                let min_simple = ctx.and_u32(min_byte_32, mask_6bit);

                // Complex path for blocks 4-7 (using conditional moves)
                // CORRECTNESS-001: Fixed per GGML Q4_K spec
                // For blocks 4-7: index = sub_block - 4
                // scale = (scales[8+index] & 0xF) | ((scales[index] >> 6) << 4)
                // min = (scales[8+index] >> 4) | ((scales[4+index] >> 6) << 4)
                let eight_64 = ctx.mov_u64_imm(8);
                let scales_8_base = ctx.add_u64(scales_base, eight_64);
                // Safe subtraction: for sub_block < 4, use 0 to avoid underflow
                // (the loaded value won't be used anyway due to selp)
                let sub_block_minus_4_raw = ctx.sub_u32_reg(sub_block, four_reg);
                let zero_safe_fused = ctx.mov_u32_imm(0);
                let sub_block_minus_4 =
                    ctx.selp_u32(sub_block_lt_4, zero_safe_fused, sub_block_minus_4_raw);
                let sub_block_minus_4_64 = ctx.cvt_u64_u32(sub_block_minus_4);
                let scales_8_addr = ctx.add_u64(scales_8_base, sub_block_minus_4_64);
                let s8_byte = ctx.ld_global_u8(scales_8_addr);
                let s8_byte_32 = ctx.cvt_u32_u8(s8_byte);

                // Load scales[index] = scales[sub_block - 4] for scale high bits
                let scale_hi_src_addr = ctx.add_u64(scales_base, sub_block_minus_4_64);
                let scale_hi_src_byte = ctx.ld_global_u8(scale_hi_src_addr);
                let scale_hi_src_32 = ctx.cvt_u32_u8(scale_hi_src_byte);

                // scale = (scales[8+index] & 0xF) | ((scales[index] >> 6) << 4)
                let s8_lo = ctx.and_u32(s8_byte_32, mask_4bit);
                let s0_hi = ctx.shr_u32(scale_hi_src_32, six);
                let s0_hi_shifted = ctx.shl_u32(s0_hi, four_reg);
                let scale_complex = ctx.or_u32(s8_lo, s0_hi_shifted);

                // min = (scales[8+index] >> 4) | ((scales[4+index] >> 6) << 4)
                // scales[4+index] = scales[sub_block], which is scale_byte_32
                let s8_hi = ctx.shr_u32(s8_byte_32, four_reg);
                let s4_hi = ctx.shr_u32(scale_byte_32, six);
                let s4_hi_shifted = ctx.shl_u32(s4_hi, four_reg);
                let min_complex = ctx.or_u32(s8_hi, s4_hi_shifted);

                // Select based on sub_block < 4
                let scale = ctx.selp_u32(sub_block_lt_4, scale_simple, scale_complex);
                let min = ctx.selp_u32(sub_block_lt_4, min_simple, min_complex);

                let scale_f = ctx.cvt_f32_u32(scale);
                let min_f = ctx.cvt_f32_u32(min);

                // Precompute d*scale and dmin*min
                let ds = ctx.mul_f32(d, scale_f);
                let dm = ctx.mul_f32(dmin, min_f);

                // Load quantized value from qs (offset 16)
                let sixteen_64 = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen_64);

                // qs layout: values are packed in 64-value chunks
                let chunk_idx = ctx.div_u32(thread_id, 64);
                let val_in_chunk = ctx.rem_u32(thread_id, 64);
                let byte_in_chunk = ctx.rem_u32(val_in_chunk, 32);

                let chunk_offset = ctx.mul_u32(chunk_idx, 32);
                let qs_byte_offset = ctx.add_u32_reg(chunk_offset, byte_in_chunk);
                let qs_byte_offset_64 = ctx.cvt_u64_u32(qs_byte_offset);
                let qs_addr = ctx.add_u64(qs_base, qs_byte_offset_64);
                let packed = ctx.ld_global_u8(qs_addr);
                let packed_32 = ctx.cvt_u32_u8(packed);

                // Extract nibble (low or high)
                let val_in_chunk_div_32 = ctx.div_u32(val_in_chunk, 32);
                let shift_amount = ctx.mul_u32_reg(val_in_chunk_div_32, four_reg);
                let shifted = ctx.shr_u32(packed_32, shift_amount);
                let quant = ctx.and_u32(shifted, mask_4bit);

                // Dequantize: val = d*scale*quant - dmin*min
                let quant_f32 = ctx.cvt_f32_u32(quant);
                let scaled_q = ctx.mul_f32(ds, quant_f32);
                let dequant = ctx.sub_f32(scaled_q, dm);

                // Load normalized activation from shared memory
                let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                let x_idx = ctx.add_u32_reg(sb_k_base, thread_id);
                let x_smem_offset = ctx.mul_wide_u32_reg(x_idx, four);
                let x_norm_val = ctx.ld_shared_f32(x_smem_offset);

                // Accumulate: acc += x_norm * dequant
                ctx.fma_f32_inplace(acc, x_norm_val, dequant);

                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_end");

                // Block-level reduction of acc using warp shuffles
                let shfl16_acc = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, shfl16_acc);
                let shfl8_acc = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, shfl8_acc);
                let shfl4_acc = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, shfl4_acc);
                let shfl2_acc = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, shfl2_acc);
                let shfl1_acc = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, shfl1_acc);

                // Store warp partial to shared memory for final reduction
                let warp_acc_offset = ctx.mul_wide_u32_reg(warp_id, four);
                let warp_acc_addr = ctx.add_u64(k_bytes_64, warp_acc_offset);

                ctx.branch_if_not(is_lane0, "skip_warp_acc_write");
                ctx.st_shared_f32(warp_acc_addr, acc);
                ctx.label("skip_warp_acc_write");

                ctx.bar_sync(3);

                // Thread 0 computes final result
                ctx.branch_if_not(is_thread0, "exit");

                let final_acc = ctx.mov_f32_imm(0.0);
                for warp in 0..8u32 {
                    let warp_offset = ctx.mov_u64_imm((warp * 4) as u64);
                    let addr = ctx.add_u64(k_bytes_64, warp_offset);
                    let warp_acc = ctx.ld_shared_f32(addr);
                    ctx.add_f32_inplace(final_acc, warp_acc);
                }

                // Store y[block_id]
                let y_offset = ctx.mul_wide_u32(block_id, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, final_acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
