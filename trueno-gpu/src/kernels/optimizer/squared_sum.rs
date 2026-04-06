//! GPU Squared Sum Reduction Kernel (KAIZEN-049)
//!
//! Computes `sum(x[i]^2)` for a flat f32 tensor using block-level parallel reduction.
//! Used to compute L2 gradient norms on GPU, eliminating large D2H transfers.
//!
//! ## Algorithm
//!
//! 1. Each block (256 threads) processes `ceil(n / num_blocks)` elements via striding
//! 2. Each thread accumulates `x[i]^2` over its strided chunk
//! 3. Warp-level shuffle reduction combines 32 thread values
//! 4. Lane 0 of each warp writes to shared memory
//! 5. First warp reduces 8 warp partials
//! 6. Thread 0 writes block result to `output[block_id]`
//!
//! ## Usage
//!
//! ```ignore
//! // Launch with num_blocks blocks, 256 threads per block
//! // Output: num_blocks f32 partial sums
//! // Host: sum partials in f64, sqrt → L2 norm
//! ```

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Block-level squared sum reduction kernel.
///
/// Each block produces one partial sum. Host sums partials for global result.
///
/// # Contract (C-SQSUM-001)
///
/// - **Precondition**: `n > 0`, `output` has `num_blocks` elements
/// - **Postcondition**: `output[bid] = sum(input[i]^2)` for elements assigned to block `bid`
/// - **Invariant**: No f64 on GPU (consumer GPUs have 1/32 throughput); f32 accumulation
///   with block-level granularity bounds relative error to O(block_elements × epsilon_f32)
#[derive(Debug, Clone)]
pub struct SquaredSumKernel {
    /// Number of elements in input tensor
    pub n: u32,
}

impl SquaredSumKernel {
    /// Create a new squared sum kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }

    /// Block size (threads per block)
    #[must_use]
    pub const fn block_size(&self) -> u32 {
        256
    }

    /// Number of blocks to launch
    #[must_use]
    pub fn num_blocks(&self) -> u32 {
        // Each thread handles at least 4 elements for efficiency
        let elements_per_block = self.block_size() * 4;
        let needed = (self.n + elements_per_block - 1) / elements_per_block;
        // Cap at 256 blocks — more than enough for any practical size
        needed.min(256)
    }
}

impl Kernel for SquaredSumKernel {
    fn name(&self) -> &str {
        "squared_sum_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Shared memory: 8 f32 warp partials = 32 bytes
        // Pattern follows VectorizedRmsNormKernel (proven correct).
        PtxKernel::new("squared_sum_reduce")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "n")
            .shared_memory(8 * 4) // 8 warps × 4 bytes
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let bid = ctx.special_reg(PtxReg::CtaIdX);
                let block_dim = ctx.special_reg(PtxReg::NtidX);
                let num_blocks = ctx.special_reg(PtxReg::NctaIdX);

                let warp_id = ctx.div_u32(tid, 32);
                let lane_id = ctx.rem_u32(tid, 32);

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let n = ctx.load_param_u32("n");

                // Total threads = num_blocks * block_dim
                let total_threads = ctx.mul_lo_u32(num_blocks, block_dim);

                // Global thread ID = bid * block_dim + tid
                let global_id = ctx.mad_lo_u32(bid, block_dim, tid);

                // Constants
                let four = ctx.mov_u32_imm(4);
                let zero_u32 = ctx.mov_u32_imm(0);
                let eight = ctx.mov_u32_imm(8);

                // --- Phase 1: Thread-level accumulation ---
                // Each thread strides over the input, accumulating x[i]^2
                let sq_sum = ctx.mov_f32_imm(0.0);
                let i = ctx.mov_u32_imm(0);

                // i starts at global_id
                ctx.add_u32_reg_inplace(i, global_id);

                ctx.label("acc_loop");
                let in_bounds = ctx.setp_lt_u32(i, n);
                ctx.branch_if_not(in_bounds, "acc_done");

                // Load input[i]
                let byte_offset = ctx.mul_wide_u32_reg(i, four);
                let addr = ctx.add_u64(input_ptr, byte_offset);
                let val = ctx.ld_global_f32(addr);

                // sq_sum += val * val
                ctx.fma_f32_inplace(sq_sum, val, val);

                // i += total_threads
                ctx.add_u32_reg_inplace(i, total_threads);
                ctx.branch("acc_loop");
                ctx.label("acc_done");

                // --- Phase 2: Warp-level shuffle reduction ---
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

                // --- Phase 3: Warp partials → shared memory ---
                // Lane 0 of each warp writes to shared memory
                let lane_zero = ctx.setp_eq_u32(lane_id, zero_u32);
                let warp_smem_off = ctx.mul_u32(warp_id, 4);
                ctx.branch_if_not(lane_zero, "skip_smem_write");
                ctx.st_shared_f32(warp_smem_off, sq_sum);
                ctx.label("skip_smem_write");

                ctx.bar_sync(0);

                // --- Phase 4: First warp reduces shared memory ---
                let thirty_two = ctx.mov_u32_imm(32);
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

                // Reduce 8 values (offsets 4, 2, 1)
                let red4 = ctx.shfl_down_f32(warp_partial, 4, 0xFFFF_FFFF);
                let partial = ctx.add_f32(warp_partial, red4);
                let red2 = ctx.shfl_down_f32(partial, 2, 0xFFFF_FFFF);
                let partial = ctx.add_f32(partial, red2);
                let red1 = ctx.shfl_down_f32(partial, 1, 0xFFFF_FFFF);
                let final_sum = ctx.add_f32(partial, red1);

                // Thread 0 writes block result to output[bid]
                let is_tid0 = ctx.setp_eq_u32(tid, zero_u32);
                ctx.branch_if_not(is_tid0, "skip_final_reduce");

                let bid_offset = ctx.mul_wide_u32_reg(bid, four);
                let out_addr = ctx.add_u64(output_ptr, bid_offset);
                ctx.st_global_f32(out_addr, final_sum);

                ctx.label("skip_final_reduce");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_sum_kernel_name() {
        let kernel = SquaredSumKernel::new(1024);
        assert_eq!(kernel.name(), "squared_sum_reduce");
    }

    #[test]
    fn test_squared_sum_num_blocks() {
        // Small: 1024 elements, 256 threads * 4 = 1024 per block → 1 block
        assert_eq!(SquaredSumKernel::new(1024).num_blocks(), 1);

        // Medium: 100K elements → ceil(100K / 1024) = 98 blocks
        assert_eq!(SquaredSumKernel::new(100_000).num_blocks(), 98);

        // Large: 32M elements → capped at 256 blocks
        assert_eq!(SquaredSumKernel::new(32_000_000).num_blocks(), 256);
    }

    #[test]
    fn test_squared_sum_ptx_generation() {
        let kernel = SquaredSumKernel::new(1024);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry squared_sum_reduce"));
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u32 n"));
        // Verify fused multiply-add (for x*x accumulation)
        assert!(ptx.contains("fma.rn.f32"));
        // Verify warp shuffle
        assert!(ptx.contains("shfl.sync.down"));
        // Verify shared memory
        assert!(ptx.contains(".shared"));
    }

    #[test]
    fn test_squared_sum_barrier_safety() {
        let kernel = SquaredSumKernel::new(1024);
        // Must not panic — validates no early-exit-before-barrier
        let _ptx = kernel.emit_ptx_validated();
    }
}
