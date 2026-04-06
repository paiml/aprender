//! Residual Connection Kernels
//!
//! Kernels for residual connections in transformer architectures.
//!
//! - `ResidualAddKernel`: Element-wise addition for residual connections
//! - `BatchedResidualAddKernel`: Batched version processing M sequences
//! - `FusedResidualRmsNormKernel`: Fused residual add + RMSNorm
//! - `BatchedFusedResidualRmsNormKernel`: PMAT-092 batched fused residual+RMSNorm

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Residual Add Kernel: output = input1 + input2
///
/// Element-wise addition for residual connections in transformers.
/// Used for: x = x + attn(x) and x = x + ffn(x)
///
/// # Parameters
///
/// - `input1_ptr`: First input vector (u64 pointer)
/// - `input2_ptr`: Second input vector (u64 pointer)
/// - `output_ptr`: Output vector (u64 pointer, can alias input1 or input2)
/// - `n`: Number of elements (u32)
///
/// # Grid Configuration
///
/// - Block: 256 threads
/// - Grid: ceil(n / 256) blocks
#[derive(Debug, Clone)]
pub struct ResidualAddKernel {
    /// Number of elements
    pub n: u32,
}

impl ResidualAddKernel {
    /// Create a new residual add kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for ResidualAddKernel {
    fn name(&self) -> &str {
        "residual_add"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Simple element-wise addition
        // Each thread processes one element
        // Block: 256 threads, Grid: ceil(n/256)
        PtxKernel::new("residual_add")
            .param(PtxType::U64, "input1_ptr")
            .param(PtxType::U64, "input2_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let n = ctx.load_param_u32("n");
                let input1_ptr = ctx.load_param_u64("input1_ptr");
                let input2_ptr = ctx.load_param_u64("input2_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate address (gid * 4 bytes)
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let addr1 = ctx.add_u64(input1_ptr, offset);
                let addr2 = ctx.add_u64(input2_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load both values
                let val1 = ctx.ld_global_f32(addr1);
                let val2 = ctx.ld_global_f32(addr2);

                // Add
                let result = ctx.add_f32(val1, val2);

                // Store
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// ============================================================================
// PAR-114: Batched Residual Add Kernel (processes M sequences in parallel)
// ============================================================================

/// Batched Residual Add: output[m] = input1[m] + input2[m] for m in 0..M
///
/// Processes M sequences in parallel using Grid.y for batch index.
///
/// # Parameters
///
/// - `input1_ptr`: First packed input [M × n]
/// - `input2_ptr`: Second packed input [M × n]
/// - `output_ptr`: Output [M × n]
/// - `n`: Elements per sequence
///
/// # Grid Configuration
///
/// - Grid: (ceil(n/256), batch_size, 1)
/// - Block: (256, 1, 1)
#[derive(Debug, Clone)]
pub struct BatchedResidualAddKernel {
    /// Elements per sequence
    pub n: u32,
    /// Batch size (M)
    pub batch_size: u32,
}

impl BatchedResidualAddKernel {
    /// Create a new batched residual add kernel
    #[must_use]
    pub const fn new(n: u32, batch_size: u32) -> Self {
        Self { n, batch_size }
    }
}

impl Kernel for BatchedResidualAddKernel {
    fn name(&self) -> &str {
        "batched_residual_add"
    }

    fn build_ptx(&self) -> PtxKernel {
        let n = self.n;

        PtxKernel::new("batched_residual_add")
            .param(PtxType::U64, "input1_ptr")
            .param(PtxType::U64, "input2_ptr")
            .param(PtxType::U64, "output_ptr")
            .build(move |ctx| {
                // Global thread ID within the sequence
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY); // Grid.y = sequence index
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let local_gid = ctx.mad_lo_u32(ctaid_x, ntid, tid);

                // Load parameters
                let input1_ptr = ctx.load_param_u64("input1_ptr");
                let input2_ptr = ctx.load_param_u64("input2_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check within sequence
                let n_val = ctx.mov_u32_imm(n);
                let in_bounds = ctx.setp_lt_u32(local_gid, n_val);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate global element index: batch_idx × n + local_gid
                let batch_offset = ctx.mul_lo_u32(batch_idx, n_val);
                let gid = ctx.add_u32_reg(batch_offset, local_gid);

                // Calculate byte address (gid × 4 bytes)
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let addr1 = ctx.add_u64(input1_ptr, offset);
                let addr2 = ctx.add_u64(input2_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load both values
                let val1 = ctx.ld_global_f32(addr1);
                let val2 = ctx.ld_global_f32(addr2);

                // Add
                let result = ctx.add_f32(val1, val2);

                // Store
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Fused Residual Add + RMSNorm Kernel
///
/// Combines residual addition and RMSNorm in a single kernel pass.
/// Reduces memory bandwidth by avoiding intermediate writes.
///
/// output = rmsnorm(input1 + input2, gamma, epsilon)
///
/// # PAR-023: This fused kernel eliminates one memory round-trip
#[derive(Debug, Clone)]
pub struct FusedResidualRmsNormKernel {
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl FusedResidualRmsNormKernel {
    /// Create a new fused residual+rmsnorm kernel
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

impl Kernel for FusedResidualRmsNormKernel {
    fn name(&self) -> &str {
        "fused_residual_rmsnorm"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_size = self.hidden_size;
        let epsilon = self.epsilon;

        // Fused residual add + RMSNorm for single row using warp shuffle
        // Grid: 1 block, Block: 32 threads (one warp)
        PtxKernel::new("fused_residual_rmsnorm")
            .param(PtxType::U64, "residual_ptr") // Residual input
            .param(PtxType::U64, "input_ptr") // Input to add
            .param(PtxType::U64, "output_ptr") // Output (can alias residual)
            .param(PtxType::U64, "gamma_ptr") // Scale weights
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                let residual_ptr = ctx.load_param_u64("residual_ptr");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Constants
                let hidden_u32 = ctx.mov_u32_imm(hidden_size);
                let four = ctx.mov_u32_imm(4);

                // ===== Phase 1: Add residual and accumulate sum of squares =====
                // Each thread processes elements: tid, tid+32, tid+64, ...
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let loop_idx = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(loop_idx, hidden_u32);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                // Load residual[idx] and input[idx]
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let res_addr = ctx.add_u64(residual_ptr, elem_offset);
                let inp_addr = ctx.add_u64(input_ptr, elem_offset);

                let res_val = ctx.ld_global_f32(res_addr);
                let inp_val = ctx.ld_global_f32(inp_addr);

                // sum_val = residual + input
                let sum_val = ctx.add_f32(res_val, inp_val);

                // sq_sum += sum_val * sum_val
                ctx.fma_f32_inplace(sq_sum, sum_val, sum_val);

                // Store intermediate sum for phase 2
                // Using output buffer as scratch (will be overwritten)
                let out_addr = ctx.add_u64(output_ptr, elem_offset);
                ctx.st_global_f32(out_addr, sum_val);

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

                // ===== Phase 2: Normalize and scale =====
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, hidden_u32);
                ctx.branch_if_not(in_bounds2, "exit");

                // Load sum_val from output buffer and gamma
                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let out_addr2 = ctx.add_u64(output_ptr, elem_offset2);
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);

                let sum_val2 = ctx.ld_global_f32(out_addr2);
                let gamma = ctx.ld_global_f32(gamma_addr);

                // output = sum_val * rms_inv * gamma
                let normalized = ctx.mul_f32(sum_val2, rms_inv);
                let result = ctx.mul_f32(normalized, gamma);

                ctx.st_global_f32(out_addr2, result);

                ctx.add_u32_inplace(idx2, 32);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// ============================================================================
// PMAT-092: Batched Fused Residual Add + RMSNorm Kernel
// ============================================================================

/// Batched Fused Residual Add + RMSNorm: saves 28 kernel launches per decode step.
///
/// For each batch element m in 0..M:
///   residual_out[m] = residual[m] + input[m]
///   normed_out[m]   = rmsnorm(residual_out[m], gamma, epsilon)
///
/// Pass 1 reads residual + input, computes their sum, stores to residual_out,
/// and accumulates sum-of-squares for RMSNorm. Pass 2 reads from residual_out,
/// normalizes, and stores to normed_out.
///
/// Grid: (1, M, 1), Block: (256, 1, 1) — one block per batch element via ctaid.y.
#[derive(Debug, Clone)]
pub struct BatchedFusedResidualRmsNormKernel {
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl BatchedFusedResidualRmsNormKernel {
    /// Create a new batched fused residual+RMSNorm kernel
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

impl Kernel for BatchedFusedResidualRmsNormKernel {
    fn name(&self) -> &str {
        "batched_fused_residual_rmsnorm"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_size = self.hidden_size;
        let epsilon = self.epsilon;

        // Grid: (1, M, 1) — one block per batch element
        // Block: (256, 1, 1) — 8 warps per block
        // Shared memory: 8 warp partial sums (f32)
        //
        // Parameters:
        //   residual_ptr:     [M × hidden_size] residual stream (read)
        //   input_ptr:        [M × hidden_size] input to add (read)
        //   residual_out_ptr: [M × hidden_size] sum output (write, then read in pass 2)
        //   normed_out_ptr:   [M × hidden_size] normalized output (write)
        //   gamma_ptr:        [hidden_size] shared across all sequences (read)

        PtxKernel::new("batched_fused_residual_rmsnorm")
            .param(PtxType::U64, "residual_ptr")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "residual_out_ptr")
            .param(PtxType::U64, "normed_out_ptr")
            .param(PtxType::U64, "gamma_ptr")
            .shared_memory(8 * 4) // 8 warp partial sums
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);
                let warp_id = ctx.div_u32(tid, 32);
                let lane_id = ctx.rem_u32(tid, 32);

                // Load parameters
                let residual_base = ctx.load_param_u64("residual_ptr");
                let input_base = ctx.load_param_u64("input_ptr");
                let res_out_base = ctx.load_param_u64("residual_out_ptr");
                let norm_out_base = ctx.load_param_u64("normed_out_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Batch offset: batch_idx * hidden_size * 4 bytes
                let hidden_u32 = ctx.mov_u32_imm(hidden_size);
                let four = ctx.mov_u32_imm(4);
                let batch_offset_elems = ctx.mul_wide_u32_reg(batch_idx, hidden_u32);
                let batch_offset_bytes = ctx.mul_u64(batch_offset_elems, 4);

                let residual_ptr = ctx.add_u64(residual_base, batch_offset_bytes);
                let input_ptr = ctx.add_u64(input_base, batch_offset_bytes);
                let res_out_ptr = ctx.add_u64(res_out_base, batch_offset_bytes);
                let norm_out_ptr = ctx.add_u64(norm_out_base, batch_offset_bytes);

                // ===== Pass 1: Fused residual add + sum-of-squares =====
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let loop_idx = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(loop_idx, hidden_u32);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let res_addr = ctx.add_u64(residual_ptr, elem_offset);
                let inp_addr = ctx.add_u64(input_ptr, elem_offset);
                let res_out_addr = ctx.add_u64(res_out_ptr, elem_offset);

                let res_val = ctx.ld_global_f32(res_addr);
                let inp_val = ctx.ld_global_f32(inp_addr);

                // sum_val = residual + input
                let sum_val = ctx.add_f32(res_val, inp_val);

                // Store sum to residual_out (serves as scratch for pass 2)
                ctx.st_global_f32(res_out_addr, sum_val);

                // Accumulate sum of squares
                ctx.fma_f32_inplace(sq_sum, sum_val, sum_val);

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

                // Lane 0 writes partial sum to shared memory
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

                // Load and sum all 8 warp contributions
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

                // rms_inv = rsqrt(sum / hidden_size + epsilon)
                let hidden_f32 = ctx.cvt_f32_u32(hidden_u32);
                let mean_sq = ctx.div_f32(final_sum, hidden_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let var_plus_eps = ctx.add_f32(mean_sq, eps);
                let rms_inv = ctx.rsqrt_f32(var_plus_eps);

                // Store rms_inv to shared memory for other threads
                ctx.st_shared_f32(addr0, rms_inv);

                ctx.label("after_final_reduce");
                ctx.bar_sync(0);

                // All threads load rms_inv
                let smem_zero = ctx.mov_u32_imm(0);
                let rms_inv_shared = ctx.ld_shared_f32(smem_zero);

                // ===== Pass 2: Normalize from residual_out, store to normed_out =====
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, hidden_u32);
                ctx.branch_if_not(in_bounds2, "exit");

                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let res_out_read_addr = ctx.add_u64(res_out_ptr, elem_offset2);
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);
                let norm_out_addr = ctx.add_u64(norm_out_ptr, elem_offset2);

                let sum_val2 = ctx.ld_global_f32(res_out_read_addr);
                let gamma = ctx.ld_global_f32(gamma_addr);

                let normalized = ctx.mul_f32(sum_val2, rms_inv_shared);
                let result = ctx.mul_f32(normalized, gamma);

                ctx.st_global_f32(norm_out_addr, result);

                ctx.add_u32_inplace(idx2, 256);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_add_kernel_name() {
        let kernel = ResidualAddKernel::new(2048);
        assert_eq!(kernel.name(), "residual_add");
    }

    #[test]
    fn test_residual_add_ptx_generation() {
        let kernel = ResidualAddKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify parameters
        assert!(ptx.contains(".param .u64 input1_ptr"));
        assert!(ptx.contains(".param .u64 input2_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u32 n"));

        // Verify basic structure
        assert!(ptx.contains(".entry residual_add"));
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn test_fused_residual_rmsnorm_kernel_name() {
        let kernel = FusedResidualRmsNormKernel::new(2048);
        assert_eq!(kernel.name(), "fused_residual_rmsnorm");
    }

    #[test]
    fn test_fused_residual_rmsnorm_ptx_generation() {
        let kernel = FusedResidualRmsNormKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify parameters
        assert!(ptx.contains(".param .u64 residual_ptr"));
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u64 gamma_ptr"));

        // Verify warp shuffle operations (for reduction)
        assert!(ptx.contains("shfl.sync.down"));
        assert!(ptx.contains("shfl.sync.idx"));

        // Verify rsqrt for RMS normalization
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn test_batched_residual_add_kernel() {
        let kernel = BatchedResidualAddKernel::new(2048, 4);
        assert_eq!(kernel.name(), "batched_residual_add");

        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry batched_residual_add"));
        assert!(ptx.contains("add.f32"));
    }

    #[test]
    fn test_batched_fused_residual_rmsnorm_kernel_name() {
        let kernel = BatchedFusedResidualRmsNormKernel::new(1536, 4);
        assert_eq!(kernel.name(), "batched_fused_residual_rmsnorm");
    }

    #[test]
    fn test_batched_fused_residual_rmsnorm_ptx_generation() {
        let kernel = BatchedFusedResidualRmsNormKernel::new(1536, 4);
        let ptx = kernel.emit_ptx();

        // Verify 5 parameters
        assert!(ptx.contains(".param .u64 residual_ptr"));
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 residual_out_ptr"));
        assert!(ptx.contains(".param .u64 normed_out_ptr"));
        assert!(ptx.contains(".param .u64 gamma_ptr"));

        // Verify uses ctaid.y for batch dispatch
        assert!(ptx.contains("ctaid.y"));

        // Verify warp shuffle + rsqrt
        assert!(ptx.contains("shfl.sync.down"));
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn test_batched_fused_residual_rmsnorm_shared_memory() {
        let kernel = BatchedFusedResidualRmsNormKernel::new(1536, 4);
        let ptx = kernel.emit_ptx();
        // 8 warps × 4 bytes = 32 bytes shared memory
        assert!(ptx.contains(".shared .align 16 .b8 smem[32]"));
    }
}
