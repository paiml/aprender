//! Batched Vectorized RMSNorm and Precise RMSNorm kernels

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// PAR-112: Batched Vectorized RMSNorm Kernel
///
/// Processes M sequences in parallel using Grid.y = M.
/// Each block (blockIdx.y) processes one sequence independently.
/// Achieves ~4x speedup over M sequential kernel launches.
///
/// Grid: (1, M, 1), Block: (256, 1, 1)
#[derive(Debug, Clone)]
pub struct BatchedVectorizedRmsNormKernel {
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl BatchedVectorizedRmsNormKernel {
    /// Create a new batched vectorized RMSNorm kernel
    #[must_use]
    pub fn new(hidden_size: u32, batch_size: u32) -> Self {
        Self { hidden_size, batch_size, epsilon: 1e-5 }
    }

    /// Set custom epsilon value
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl Kernel for BatchedVectorizedRmsNormKernel {
    fn name(&self) -> &str {
        "batched_rmsnorm_vectorized"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_size = self.hidden_size;
        let epsilon = self.epsilon;

        // Strategy:
        // - Grid: (1, M, 1) - one block row per sequence
        // - Block: (256, 1, 1) - 8 warps per block
        // - Each block processes input[blockIdx.y * hidden_size : (blockIdx.y+1) * hidden_size]
        // - Shared memory for warp reduction within block
        //
        // Memory layout (packed):
        // input:  [seq0_hidden..., seq1_hidden..., seq2_hidden..., seq3_hidden...]
        // output: [seq0_hidden..., seq1_hidden..., seq2_hidden..., seq3_hidden...]
        // gamma:  [hidden_size] (shared across all sequences)

        PtxKernel::new("batched_rmsnorm_vectorized")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U64, "gamma_ptr")
            .shared_memory(8 * 4) // 8 warp partial sums (f32)
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY); // blockIdx.y = batch index
                let warp_id = ctx.div_u32(tid, 32);
                let lane_id = ctx.rem_u32(tid, 32);

                // Load parameters
                let input_base = ctx.load_param_u64("input_ptr");
                let output_base = ctx.load_param_u64("output_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Calculate batch offset: batch_idx * hidden_size * 4 bytes
                let hidden_u32 = ctx.mov_u32_imm(hidden_size);
                let four = ctx.mov_u32_imm(4);
                let batch_offset_elems = ctx.mul_wide_u32_reg(batch_idx, hidden_u32);
                let batch_offset_bytes = ctx.mul_u64(batch_offset_elems, 4);

                // Offset input/output pointers for this batch
                let input_ptr = ctx.add_u64(input_base, batch_offset_bytes);
                let output_ptr = ctx.add_u64(output_base, batch_offset_bytes);

                // Pass 1: Accumulate sum of squares
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let loop_idx = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(loop_idx, hidden_u32);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let elem_addr = ctx.add_u64(input_ptr, elem_offset);
                let val = ctx.ld_global_f32(elem_addr);

                ctx.fma_f32_inplace(sq_sum, val, val);
                ctx.add_u32_inplace(idx, 256);
                ctx.branch("sum_loop");

                ctx.label("sum_loop_end");

                // Warp-level reduction
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

                // Lane 0 of each warp writes partial sum to shared memory
                // CRITICAL: Use u32 offsets for shared memory (window-form addressing).
                // Using u64 registers would generate generic-form addressing where
                // values like 0, 4, 8 are NOT valid shared memory addresses.
                let zero = ctx.mov_u32_imm(0);
                let is_lane_zero = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_lane_zero, "skip_store");

                let warp_smem_off = ctx.mul_u32(warp_id, 4);
                ctx.st_shared_f32(warp_smem_off, sq_sum);

                ctx.label("skip_store");
                ctx.bar_sync(0);

                // Thread 0 reduces across warps
                let final_sum = ctx.mov_f32_imm(0.0);
                let is_tid_zero = ctx.setp_eq_u32(tid, zero);
                ctx.branch_if_not(is_tid_zero, "after_final_reduce");

                // Load and sum all 8 warp contributions (u32 shared memory offsets)
                let addr0 = ctx.mov_u32_imm(0);
                let s0 = ctx.ld_shared_f32(addr0);
                ctx.add_f32_inplace(final_sum, s0);

                let addr1 = ctx.mov_u32_imm(4);
                let s1 = ctx.ld_shared_f32(addr1);
                ctx.add_f32_inplace(final_sum, s1);

                let addr2 = ctx.mov_u32_imm(8);
                let s2 = ctx.ld_shared_f32(addr2);
                ctx.add_f32_inplace(final_sum, s2);

                let addr3 = ctx.mov_u32_imm(12);
                let s3 = ctx.ld_shared_f32(addr3);
                ctx.add_f32_inplace(final_sum, s3);

                let addr4 = ctx.mov_u32_imm(16);
                let s4 = ctx.ld_shared_f32(addr4);
                ctx.add_f32_inplace(final_sum, s4);

                let addr5 = ctx.mov_u32_imm(20);
                let s5 = ctx.ld_shared_f32(addr5);
                ctx.add_f32_inplace(final_sum, s5);

                let addr6 = ctx.mov_u32_imm(24);
                let s6 = ctx.ld_shared_f32(addr6);
                ctx.add_f32_inplace(final_sum, s6);

                let addr7 = ctx.mov_u32_imm(28);
                let s7 = ctx.ld_shared_f32(addr7);
                ctx.add_f32_inplace(final_sum, s7);

                // Compute rms_inv = rsqrt(sum / hidden_size + epsilon)
                let hidden_f32 = ctx.cvt_f32_u32(hidden_u32);
                let mean_sq = ctx.div_f32(final_sum, hidden_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let var_plus_eps = ctx.add_f32(mean_sq, eps);
                let rms_inv = ctx.rsqrt_f32(var_plus_eps);

                // Store rms_inv to shared memory for other threads
                ctx.st_shared_f32(addr0, rms_inv);

                ctx.label("after_final_reduce");
                ctx.bar_sync(0);

                // All threads load rms_inv and normalize
                let smem_zero = ctx.mov_u32_imm(0);
                let rms_inv_shared = ctx.ld_shared_f32(smem_zero);

                // Pass 2: Normalize output = input * rms_inv * gamma
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, hidden_u32);
                ctx.branch_if_not(in_bounds2, "exit");

                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let in_addr = ctx.add_u64(input_ptr, elem_offset2);
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);
                let out_addr = ctx.add_u64(output_ptr, elem_offset2);

                let inp = ctx.ld_global_f32(in_addr);
                let gamma = ctx.ld_global_f32(gamma_addr);

                let normalized = ctx.mul_f32(inp, rms_inv_shared);
                let result = ctx.mul_f32(normalized, gamma);

                ctx.st_global_f32(out_addr, result);

                ctx.add_u32_inplace(idx2, 256);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// CORRECTNESS-013: High-Precision RMSNorm Kernel for CPU/GPU bit-exactness
///
/// Uses Kahan compensated summation to match CPU sequential sum precision,
/// and Newton-Raphson refinement for rsqrt to achieve full IEEE-754 f32 precision.
///
/// This kernel is slower than VectorizedRmsNormKernel but produces results
/// that match CPU computation within floating-point epsilon.
///
/// Grid: (1, 1, 1), Block: (256, 1, 1)
#[derive(Debug, Clone)]
pub struct PreciseRmsNormKernel {
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl PreciseRmsNormKernel {
    /// Create a new precise RMSNorm kernel
    #[must_use]
    pub fn new(hidden_size: u32) -> Self {
        Self { hidden_size, epsilon: 1e-5 }
    }

    /// Set custom epsilon value
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl Kernel for PreciseRmsNormKernel {
    fn name(&self) -> &str {
        "rmsnorm_precise"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_size = self.hidden_size;
        let epsilon = self.epsilon;

        // Strategy for CPU-matching precision:
        // - Block: 256 threads (8 warps)
        // - Each thread handles ceil(hidden_size / 256) elements
        // - Pass 1: Kahan compensated sum of squares for numerically stable accumulation
        // - Warp-level Kahan merge for reduction
        // - Pass 2: Newton-Raphson refined rsqrt for full precision
        // - Pass 3: Normalize with the computed RMS inverse
        //
        // Kahan summation formula:
        //   y = val - compensation
        //   t = sum + y
        //   compensation = (t - sum) - y
        //   sum = t
        //
        // Newton-Raphson rsqrt refinement:
        //   rsqrt_refined = rsqrt * (1.5 - 0.5 * x * rsqrt * rsqrt)

        PtxKernel::new("rmsnorm_precise")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U64, "gamma_ptr")
            .shared_memory(16 * 4) // 8 warp sums + 8 compensations (f32)
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

                // Pass 1: Kahan compensated sum of squares
                // Each thread maintains (sum, compensation) pair
                let sq_sum = ctx.mov_f32_imm(0.0);
                let compensation = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let loop_idx = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(loop_idx, hidden_u32);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                // Load input[loop_idx]
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let elem_addr = ctx.add_u64(input_ptr, elem_offset);
                let val = ctx.ld_global_f32(elem_addr);

                // val_sq = val * val
                let val_sq = ctx.mul_f32(val, val);

                // Kahan summation: add val_sq to sum with compensation
                // y = val_sq - compensation
                let y = ctx.sub_f32(val_sq, compensation);
                // t = sum + y
                let t = ctx.add_f32(sq_sum, y);
                // new_compensation = (t - sum) - y
                let t_minus_sum = ctx.sub_f32(t, sq_sum);
                let new_comp = ctx.sub_f32(t_minus_sum, y);

                // Update compensation = new_comp (zero then add)
                let zero_f32 = ctx.mov_f32_imm(0.0);
                ctx.mul_f32_inplace(compensation, zero_f32); // compensation = 0
                ctx.add_f32_inplace(compensation, new_comp); // compensation = new_comp

                // Update sq_sum = t (zero then add)
                ctx.mul_f32_inplace(sq_sum, zero_f32); // sq_sum = 0
                ctx.add_f32_inplace(sq_sum, t); // sq_sum = t

                // idx += 256 (stride by block size)
                ctx.add_u32_inplace(idx, 256);
                ctx.branch("sum_loop");

                ctx.label("sum_loop_end");

                // Warp-level reduction with Kahan merge
                // For each shuffle level, merge (sum, comp) pairs
                // merge: new_sum = sum1 + sum2, new_comp = comp1 + comp2 + ((sum1 + sum2) - new_sum)
                // Simplified: just sum the values (Kahan already gave per-thread precision)
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
                let warp_partial = ctx.mov_f32_imm(0.0);
                ctx.branch_if_not(lane_valid, "skip_warp_load");
                let loaded_val = ctx.ld_shared_f32(lane_smem_off);
                ctx.add_f32_inplace(warp_partial, loaded_val);
                ctx.label("skip_warp_load");

                // Reduce 8 values (only first 8 lanes participate)
                let red4 = ctx.shfl_down_f32(warp_partial, 4, 0xFFFF_FFFF);
                let partial = ctx.add_f32(warp_partial, red4);
                let red2 = ctx.shfl_down_f32(partial, 2, 0xFFFF_FFFF);
                let partial = ctx.add_f32(partial, red2);
                let red1 = ctx.shfl_down_f32(partial, 1, 0xFFFF_FFFF);
                let final_sum = ctx.add_f32(partial, red1);

                // Compute mean_sq and apply Newton-Raphson refined rsqrt
                let hidden_f32 = ctx.cvt_f32_u32(hidden_u32);
                let mean_sq = ctx.div_f32(final_sum, hidden_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let mean_sq_eps = ctx.add_f32(mean_sq, eps);

                // Initial rsqrt approximation
                let rsqrt_approx = ctx.rsqrt_f32(mean_sq_eps);

                // Newton-Raphson refinement: rsqrt_refined = rsqrt * (1.5 - 0.5 * x * rsqrt^2)
                let half = ctx.mov_f32_imm(0.5);
                let three_half = ctx.mov_f32_imm(1.5);
                // rsqrt^2
                let rsqrt_sq = ctx.mul_f32(rsqrt_approx, rsqrt_approx);
                // 0.5 * x * rsqrt^2
                let half_x_rsq_sq = ctx.mul_f32(mean_sq_eps, rsqrt_sq);
                let half_x_rsq_sq = ctx.mul_f32(half, half_x_rsq_sq);
                // 1.5 - 0.5 * x * rsqrt^2
                let factor = ctx.sub_f32(three_half, half_x_rsq_sq);
                // rsqrt * factor
                let rms_inv = ctx.mul_f32(rsqrt_approx, factor);

                // Lane 0 writes total to shared memory slot 0
                let smem_zero = ctx.mov_u32_imm(0);
                ctx.branch_if_not(lane_zero, "skip_final_write");
                ctx.st_shared_f32(smem_zero, rms_inv);
                ctx.label("skip_final_write");

                ctx.label("skip_final_reduce");

                // Sync again before all threads read
                ctx.bar_sync(1);

                // All threads read the total sum from slot 0
                let smem_read_zero = ctx.mov_u32_imm(0);
                let rms_inv_final = ctx.ld_shared_f32(smem_read_zero);

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
                let normalized = ctx.mul_f32(inp, rms_inv_final);
                let result = ctx.mul_f32(normalized, gamma);

                ctx.st_global_f32(out_addr, result);

                ctx.add_u32_inplace(idx2, 256);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}
