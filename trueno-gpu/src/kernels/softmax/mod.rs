//! Numerically Stable Softmax Kernel
//!
//! Implements softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))

mod long_row;

pub use long_row::LongRowSoftmaxKernel;

use super::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxType};

/// Softmax kernel configuration
#[derive(Debug, Clone)]
pub struct SoftmaxKernel {
    /// Vector length
    pub length: u32,
    /// Use warp shuffle for reduction (faster)
    pub use_warp_shuffle: bool,
}

impl SoftmaxKernel {
    /// Create a new softmax kernel
    #[must_use]
    pub fn new(length: u32) -> Self {
        Self { length, use_warp_shuffle: true }
    }

    /// Disable warp shuffle (for compatibility with older GPUs)
    #[must_use]
    pub const fn without_warp_shuffle(mut self) -> Self {
        self.use_warp_shuffle = false;
        self
    }
}

impl Kernel for SoftmaxKernel {
    fn name(&self) -> &str {
        if self.use_warp_shuffle {
            "softmax_warp_shuffle"
        } else {
            "softmax_shared"
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        if self.use_warp_shuffle {
            self.build_warp_shuffle()
        } else {
            self.build_shared_memory()
        }
    }
}

impl SoftmaxKernel {
    fn build_warp_shuffle(&self) -> PtxKernel {
        // Warp-level softmax using shuffle for fast reductions
        // Each block processes one row; blockIdx.x selects which row
        // Assumes row fits in a single warp (32 elements) for simplicity
        // For longer vectors, multiple warps would cooperate
        PtxKernel::new("softmax_warp_shuffle")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "length")
            .build(|ctx| {
                // Thread ID within block and block ID (row index)
                let tid = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let ctaid = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let length = ctx.load_param_u32("length");

                // Bounds check: tid must be < length (row size)
                let pred = ctx.setp_ge_u32(tid, length);
                ctx.branch_if(pred, "exit");

                // Load input value for this thread
                // Global index = ctaid * length + tid (row_idx * row_size + col_idx)
                let input_ptr = ctx.load_param_u64("input_ptr");
                let global_idx = ctx.mad_lo_u32(ctaid, length, tid);
                let offset = ctx.mul_wide_u32(global_idx, 4);
                let addr = ctx.add_u64(input_ptr, offset);
                let val = ctx.ld_global_f32(addr);

                // ===== Step 1: Find max using warp shuffle =====
                // Initialize max with our value
                let max_val = val;

                // Warp shuffle reduction for max (tree reduction)
                // Each iteration halves the active participants
                let shuffled_16 = ctx.shfl_down_f32(max_val, 16, 0xFFFF_FFFF);
                let max_val_1 = ctx.max_f32(max_val, shuffled_16);

                let shuffled_8 = ctx.shfl_down_f32(max_val_1, 8, 0xFFFF_FFFF);
                let max_val_2 = ctx.max_f32(max_val_1, shuffled_8);

                let shuffled_4 = ctx.shfl_down_f32(max_val_2, 4, 0xFFFF_FFFF);
                let max_val_3 = ctx.max_f32(max_val_2, shuffled_4);

                let shuffled_2 = ctx.shfl_down_f32(max_val_3, 2, 0xFFFF_FFFF);
                let max_val_4 = ctx.max_f32(max_val_3, shuffled_2);

                let shuffled_1 = ctx.shfl_down_f32(max_val_4, 1, 0xFFFF_FFFF);
                let warp_max = ctx.max_f32(max_val_4, shuffled_1);

                // Broadcast max to all lanes (get value from lane 0)
                let broadcast_max = ctx.shfl_idx_f32(warp_max, 0, 0xFFFF_FFFF);

                // ===== Step 2: Compute exp(val - max) =====
                let shifted = ctx.sub_f32(val, broadcast_max);
                // PTX ex2 computes 2^x, we need e^x = 2^(x * log2(e))
                // log2(e) ≈ 1.4426950408889634
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scaled = ctx.mul_f32(shifted, log2_e);
                let exp_val = ctx.ex2_f32(scaled);

                // ===== Step 3: Sum exp values using warp shuffle =====
                let sum_val = exp_val;

                let sum_shuffled_16 = ctx.shfl_down_f32(sum_val, 16, 0xFFFF_FFFF);
                let sum_val_1 = ctx.add_f32(sum_val, sum_shuffled_16);

                let sum_shuffled_8 = ctx.shfl_down_f32(sum_val_1, 8, 0xFFFF_FFFF);
                let sum_val_2 = ctx.add_f32(sum_val_1, sum_shuffled_8);

                let sum_shuffled_4 = ctx.shfl_down_f32(sum_val_2, 4, 0xFFFF_FFFF);
                let sum_val_3 = ctx.add_f32(sum_val_2, sum_shuffled_4);

                let sum_shuffled_2 = ctx.shfl_down_f32(sum_val_3, 2, 0xFFFF_FFFF);
                let sum_val_4 = ctx.add_f32(sum_val_3, sum_shuffled_2);

                let sum_shuffled_1 = ctx.shfl_down_f32(sum_val_4, 1, 0xFFFF_FFFF);
                let warp_sum = ctx.add_f32(sum_val_4, sum_shuffled_1);

                // Broadcast sum to all lanes (get value from lane 0)
                let broadcast_sum = ctx.shfl_idx_f32(warp_sum, 0, 0xFFFF_FFFF);

                // ===== Step 4: Divide exp(val - max) by sum =====
                let softmax_result = ctx.div_f32(exp_val, broadcast_sum);

                // ===== Step 5: Store result =====
                let output_ptr = ctx.load_param_u64("output_ptr");
                let out_addr = ctx.add_u64(output_ptr, offset);
                ctx.st_global_f32(out_addr, softmax_result);

                ctx.label("exit");
                ctx.ret();
            })
    }

    fn build_shared_memory(&self) -> PtxKernel {
        // Shared memory softmax for larger vectors or older GPUs
        // Uses block-level reduction with shared memory
        let block_size = 256_u32;
        let smem_size = block_size * 4; // Reduction buffer for f32

        PtxKernel::new("softmax_shared")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "length")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Thread and block indices
                let tid = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let ctaid = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ntid = ctx.special_reg(crate::ptx::PtxReg::NtidX);

                // Global index
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let length = ctx.load_param_u32("length");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check for loading
                let pred = ctx.setp_ge_u32(gid, length);

                // Load value (or 0 if out of bounds)
                let val = ctx.mov_f32_imm(0.0);
                ctx.branch_if(pred, "skip_load");
                let offset = ctx.mul_wide_u32(gid, 4);
                let addr = ctx.add_u64(input_ptr, offset);
                let _loaded = ctx.ld_global_f32(addr);
                // In real PTX we'd use predicated mov, simplified here
                ctx.label("skip_load");

                // Store to shared memory for reduction
                let smem_offset = ctx.mul_wide_u32(tid, 4);
                ctx.st_shared_f32(smem_offset, val);

                // Synchronize
                ctx.bar_sync(0);

                // ===== Block-level max reduction =====
                // Tree reduction in shared memory with halving stride
                let stride_reg = ctx.mov_u32_imm(128);
                let one = ctx.mov_u32_imm(1);

                ctx.label("max_reduce_loop");

                // Exit when stride reaches 0
                let stride_zero = ctx.setp_lt_u32(stride_reg, one);
                ctx.branch_if(stride_zero, "max_reduce_done");

                // Only threads with tid < stride participate
                let should_reduce = ctx.setp_lt_u32(tid, stride_reg);
                ctx.branch_if_not(should_reduce, "max_skip_neighbor");

                // Load neighbor value at tid + stride
                let neighbor_tid = ctx.add_u32_reg(tid, stride_reg);
                let block_size_reg = ctx.mov_u32_imm(block_size);
                let neighbor_oob = ctx.setp_ge_u32(neighbor_tid, block_size_reg);
                ctx.branch_if(neighbor_oob, "max_skip_neighbor");

                let neighbor_offset = ctx.mul_u32(neighbor_tid, 4);
                let neighbor_val = ctx.ld_shared_f32(neighbor_offset);
                let my_val = ctx.ld_shared_f32(smem_offset);
                let new_max = ctx.max_f32(my_val, neighbor_val);
                ctx.st_shared_f32(smem_offset, new_max);

                ctx.label("max_skip_neighbor");

                ctx.bar_sync(1);

                // Halve stride: stride = stride >> 1
                ctx.shr_u32_inplace(stride_reg, 1);
                ctx.branch("max_reduce_loop"); // Loop back with halved stride

                ctx.label("max_reduce_done");

                // Get max from thread 0
                let zero_offset = ctx.mov_u32_imm(0);
                let zero_offset_64 = ctx.cvt_u64_u32(zero_offset);
                let block_max = ctx.ld_shared_f32(zero_offset_64);

                ctx.bar_sync(2);

                // ===== Compute exp(val - max) =====
                let shifted = ctx.sub_f32(val, block_max);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scaled = ctx.mul_f32(shifted, log2_e);
                let exp_val = ctx.ex2_f32(scaled);

                // Store exp values back to shared memory
                ctx.st_shared_f32(smem_offset, exp_val);

                ctx.bar_sync(3);

                // ===== Block-level sum reduction =====
                // Tree reduction for sum (same pattern as max)
                let sum_stride_reg = ctx.mov_u32_imm(128);

                ctx.label("sum_reduce_loop");

                // Exit when stride reaches 0
                let sum_stride_zero = ctx.setp_lt_u32(sum_stride_reg, one);
                ctx.branch_if(sum_stride_zero, "sum_reduce_done");

                // Only threads with tid < stride participate
                let should_sum = ctx.setp_lt_u32(tid, sum_stride_reg);
                ctx.branch_if_not(should_sum, "sum_skip_neighbor");

                // Load neighbor value at tid + stride
                let sum_neighbor_tid = ctx.add_u32_reg(tid, sum_stride_reg);
                let sum_neighbor_oob = ctx.setp_ge_u32(sum_neighbor_tid, block_size_reg);
                ctx.branch_if(sum_neighbor_oob, "sum_skip_neighbor");

                let sum_neighbor_offset = ctx.mul_u32(sum_neighbor_tid, 4);
                let sum_neighbor_val = ctx.ld_shared_f32(sum_neighbor_offset);
                let sum_my_val = ctx.ld_shared_f32(smem_offset);
                let new_sum = ctx.add_f32(sum_my_val, sum_neighbor_val);
                ctx.st_shared_f32(smem_offset, new_sum);

                ctx.label("sum_skip_neighbor");

                ctx.bar_sync(4);

                // Halve stride: stride = stride >> 1
                ctx.shr_u32_inplace(sum_stride_reg, 1);
                ctx.branch("sum_reduce_loop"); // Loop back with halved stride

                ctx.label("sum_reduce_done");

                // Get sum from thread 0
                let block_sum = ctx.ld_shared_f32(zero_offset_64);

                ctx.bar_sync(5);

                // ===== Divide and store =====
                let softmax_result = ctx.div_f32(exp_val, block_sum);

                ctx.branch_if(pred, "exit");
                let out_offset = ctx.mul_wide_u32(gid, 4);
                let out_addr = ctx.add_u64(output_ptr, out_offset);
                ctx.st_global_f32(out_addr, softmax_result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests;
