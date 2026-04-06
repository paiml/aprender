//! RMSNorm and Vectorized RMSNorm kernels

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// RMSNorm kernel (Root Mean Square Layer Normalization)
///
/// RMSNorm(x) = x / sqrt(mean(x^2) + epsilon) * gamma
///
/// Simpler and faster than LayerNorm - no mean centering, just scaling.
/// Used by LLaMA, Mistral, and other modern LLMs.
///
/// # PAR-023: Async pipeline support
///
/// This kernel is designed for chaining with other operations without sync.
#[derive(Debug, Clone)]
pub struct RmsNormKernel {
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl RmsNormKernel {
    /// Create a new RMSNorm kernel
    #[must_use]
    pub fn new(hidden_size: u32) -> Self {
        Self {
            hidden_size,
            epsilon: 1e-5,
        }
    }

    /// Set custom epsilon value
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl Kernel for RmsNormKernel {
    fn name(&self) -> &str {
        "rmsnorm"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_size = self.hidden_size;
        let epsilon = self.epsilon;

        // RMSNorm for single row (batch=1) using warp shuffle
        // Grid: 1 block, Block: 32 threads (one warp)
        // Each thread handles hidden_size/32 elements
        PtxKernel::new("rmsnorm")
            .param(PtxType::U64, "input_ptr") // Input vector
            .param(PtxType::U64, "output_ptr") // Output vector (can be same as input)
            .param(PtxType::U64, "gamma_ptr") // Scale weights
            .shared_memory(0) // Warp shuffle, no shared memory
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Constants
                let hidden_u32 = ctx.mov_u32_imm(hidden_size);
                let four = ctx.mov_u32_imm(4);

                // Accumulate sum of squares
                // Each thread processes elements: tid, tid+32, tid+64, ...
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let loop_idx = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(loop_idx, hidden_u32);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                // Load input[idx]
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let elem_addr = ctx.add_u64(input_ptr, elem_offset);
                let val = ctx.ld_global_f32(elem_addr);

                // sq_sum += val * val
                ctx.fma_f32_inplace(sq_sum, val, val);

                // idx += 32 (stride by warp size)
                ctx.add_u32_inplace(idx, 32);
                ctx.branch("sum_loop");

                ctx.label("sum_loop_end");

                // Warp reduce sq_sum
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

                // Broadcast final sum to all threads
                let total_sq_sum = ctx.shfl_idx_f32(sq_sum, 0, 0xFFFF_FFFF);

                // Compute RMS = sqrt(mean(x^2) + epsilon)
                let hidden_f32 = ctx.cvt_f32_u32(hidden_u32);
                let mean_sq = ctx.div_f32(total_sq_sum, hidden_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let mean_sq_eps = ctx.add_f32(mean_sq, eps);
                let rms_inv = ctx.rsqrt_f32(mean_sq_eps);

                // Normalize and scale: output[i] = (input[i] * rms_inv) * gamma[i]
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, hidden_u32);
                ctx.branch_if_not(in_bounds2, "exit");

                // Load input[idx] and gamma[idx]
                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let in_addr = ctx.add_u64(input_ptr, elem_offset2);
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);
                let out_addr = ctx.add_u64(output_ptr, elem_offset2);

                let inp = ctx.ld_global_f32(in_addr);
                let gamma = ctx.ld_global_f32(gamma_addr);

                // output = input * rms_inv * gamma
                let normalized = ctx.mul_f32(inp, rms_inv);
                let result = ctx.mul_f32(normalized, gamma);

                ctx.st_global_f32(out_addr, result);

                ctx.add_u32_inplace(idx2, 32);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// PAR-081: Vectorized RMSNorm kernel with 256 threads
///
/// Achieves ~10x speedup over single-warp RmsNormKernel by:
/// 1. Using 256 threads (8 warps) instead of 32
/// 2. Shared memory for cross-warp reduction
/// 3. Better memory coalescing
///
/// Expected: 23.5us -> ~2-3us for hidden_size=1536
#[derive(Debug, Clone)]
pub struct VectorizedRmsNormKernel {
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl VectorizedRmsNormKernel {
    /// Create a new vectorized RMSNorm kernel
    #[must_use]
    pub fn new(hidden_size: u32) -> Self {
        Self {
            hidden_size,
            epsilon: 1e-5,
        }
    }

    /// Set custom epsilon value
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl Kernel for VectorizedRmsNormKernel {
    fn name(&self) -> &str {
        "rmsnorm_vectorized"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_size = self.hidden_size;
        let epsilon = self.epsilon;

        // Strategy:
        // - Block: 256 threads (8 warps)
        // - Each thread handles ceil(hidden_size / 256) elements
        // - Pass 1: Compute sum of squares using shared memory for warp reduction
        // - Pass 2: Normalize with the computed RMS inverse
        //
        // Shared memory layout:
        // - warp_sums[8] - partial sums from each warp

        PtxKernel::new("rmsnorm_vectorized")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U64, "gamma_ptr")
            .shared_memory(8 * 4) // 8 warp partial sums (f32)
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let warp_id = ctx.div_u32(tid, 32);
                let lane_id = ctx.rem_u32(tid, 32);

                // Load parameters
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Constants
                let hidden_u32 = ctx.mov_u32_imm(hidden_size);
                let four = ctx.mov_u32_imm(4);
                let _thread_count = ctx.mov_u32_imm(256);

                // Pass 1: Accumulate sum of squares
                // Each thread processes elements: tid, tid+256, tid+512, ...
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let loop_idx = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(loop_idx, hidden_u32);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                // Load input[loop_idx]
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let elem_addr = ctx.add_u64(input_ptr, elem_offset);
                let val = ctx.ld_global_f32(elem_addr);

                // sq_sum += val * val
                ctx.fma_f32_inplace(sq_sum, val, val);

                // idx += 256 (stride by block size)
                ctx.add_u32_inplace(idx, 256);
                ctx.branch("sum_loop");

                ctx.label("sum_loop_end");

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

                // Constants for comparisons
                let zero = ctx.mov_u32_imm(0);
                let eight = ctx.mov_u32_imm(8);
                let thirty_two = ctx.mov_u32_imm(32);

                // Lane 0 of each warp writes to shared memory
                let lane_zero = ctx.setp_eq_u32(lane_id, zero);
                let warp_smem_off = ctx.mul_u32(warp_id, 4);
                ctx.branch_if_not(lane_zero, "skip_smem_write");
                ctx.st_shared_f32(warp_smem_off, sq_sum);
                ctx.label("skip_smem_write");

                // Sync all warps
                ctx.bar_sync(0);

                // First warp reduces across all warp sums
                let is_first_warp = ctx.setp_lt_u32(tid, thirty_two);

                ctx.branch_if_not(is_first_warp, "skip_final_reduce");

                // Load warp partial sum if lane < 8
                let lane_valid = ctx.setp_lt_u32(lane_id, eight);
                let lane_smem_off = ctx.mul_u32(lane_id, 4);
                // Initialize to 0, load real value if valid
                let warp_partial = ctx.mov_f32_imm(0.0);
                ctx.branch_if_not(lane_valid, "skip_warp_load");
                let loaded_val = ctx.ld_shared_f32(lane_smem_off);
                // Use add with 0 to copy (no move_f32_inplace available)
                let _zero_f32 = ctx.mov_f32_imm(0.0);
                ctx.add_f32_inplace(warp_partial, loaded_val);
                ctx.label("skip_warp_load");

                // Reduce 8 values (only first 8 lanes participate)
                let red4 = ctx.shfl_down_f32(warp_partial, 4, 0xFFFF_FFFF);
                let partial = ctx.add_f32(warp_partial, red4);
                let red2 = ctx.shfl_down_f32(partial, 2, 0xFFFF_FFFF);
                let partial = ctx.add_f32(partial, red2);
                let red1 = ctx.shfl_down_f32(partial, 1, 0xFFFF_FFFF);
                let final_sum = ctx.add_f32(partial, red1);

                // Lane 0 writes total to shared memory slot 0
                let smem_zero = ctx.mov_u32_imm(0);
                ctx.branch_if_not(lane_zero, "skip_final_write");
                ctx.st_shared_f32(smem_zero, final_sum);
                ctx.label("skip_final_write");

                ctx.label("skip_final_reduce");

                // Sync again before all threads read
                ctx.bar_sync(1);

                // All threads read the total sum from slot 0
                let smem_read_zero = ctx.mov_u32_imm(0);
                let total = ctx.ld_shared_f32(smem_read_zero);

                // Compute RMS inverse: 1 / sqrt(mean(x^2) + epsilon)
                let hidden_f32 = ctx.cvt_f32_u32(hidden_u32);
                let mean_sq = ctx.div_f32(total, hidden_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let mean_sq_eps = ctx.add_f32(mean_sq, eps);
                let rms_inv = ctx.rsqrt_f32(mean_sq_eps);

                // Pass 2: Normalize and scale
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, hidden_u32);
                ctx.branch_if_not(in_bounds2, "exit");

                // Load input and gamma
                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let in_addr = ctx.add_u64(input_ptr, elem_offset2);
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);
                let out_addr = ctx.add_u64(output_ptr, elem_offset2);

                let inp = ctx.ld_global_f32(in_addr);
                let gamma = ctx.ld_global_f32(gamma_addr);

                // output = input * rms_inv * gamma
                let normalized = ctx.mul_f32(inp, rms_inv);
                let result = ctx.mul_f32(normalized, gamma);

                ctx.st_global_f32(out_addr, result);

                ctx.add_u32_inplace(idx2, 256);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}
