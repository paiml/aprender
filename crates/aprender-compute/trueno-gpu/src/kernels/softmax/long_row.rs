//! Long Row Softmax Kernel (for rows > 32 elements)
//!
//! Uses multi-warp reduction with grid-stride loops for rows that exceed
//! a single warp width.

use super::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxType};

/// Softmax kernel for long rows (> 32 elements)
///
/// Uses multi-warp reduction with grid-stride loops:
/// - Each block handles one row
/// - Up to 256 threads (8 warps) per block
/// - Each thread processes multiple elements in grid-stride pattern
/// - Warp-level reduction, then inter-warp reduction via shared memory
#[derive(Debug, Clone)]
pub struct LongRowSoftmaxKernel {
    /// Row size (number of elements per row)
    pub row_size: u32,
}

impl LongRowSoftmaxKernel {
    /// Create a new long row softmax kernel
    #[must_use]
    pub fn new(row_size: u32) -> Self {
        Self { row_size }
    }
}

impl Kernel for LongRowSoftmaxKernel {
    fn name(&self) -> &str {
        "softmax_long_row"
    }

    fn build_ptx(&self) -> PtxKernel {
        // FULL SOFTMAX: exp(x - max) / sum(exp(x - max))
        let block_size = 256_u32;
        let n_warps = block_size / 32;
        // Shared memory: 8 warp maxes + 1 global max + 8 warp sums + 1 global sum = 72 bytes
        let smem_size = (n_warps * 2 + 2) * 4;

        PtxKernel::new("softmax_long_row")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "row_size")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Thread indexing
                let tid = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let ctaid = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ntid = ctx.special_reg(crate::ptx::PtxReg::NtidX);

                let row_size = ctx.load_param_u32("row_size");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Compute warp_id and lane_id
                let lane_mask = ctx.mov_u32_imm(31);
                let lane_id = ctx.and_u32(tid, lane_mask);
                let warp_id = ctx.shr_u32_imm(tid, 5); // tid / 32

                // Row offset
                let row_offset = ctx.mul_lo_u32(ctaid, row_size);
                let row_offset_bytes = ctx.mul_wide_u32(row_offset, 4);
                let row_in_ptr = ctx.add_u64(input_ptr, row_offset_bytes);
                let row_out_ptr = ctx.add_u64(output_ptr, row_offset_bytes);

                // =========================================================
                // Phase 1: Find max using grid-stride loop + multi-warp reduction
                // =========================================================
                let neg_inf = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let local_max = neg_inf;

                // Grid-stride loop: idx = tid; idx < row_size; idx += ntid
                let idx = ctx.add_u32(tid, 0); // Copy tid to new register

                ctx.label("max_loop");
                let done_max = ctx.setp_ge_u32(idx, row_size);
                ctx.branch_if(done_max, "max_loop_done");

                // Load input[idx]
                let byte_offset = ctx.mul_wide_u32(idx, 4);
                let load_addr = ctx.add_u64(row_in_ptr, byte_offset);
                let val = ctx.ld_global_f32(load_addr);

                // local_max = max(local_max, val)
                ctx.max_f32_inplace(local_max, val);

                // idx += ntid
                ctx.add_u32_reg_inplace(idx, ntid);
                ctx.branch("max_loop");

                ctx.label("max_loop_done");

                // Warp-level max reduction using shuffles
                let shuffled_16 = ctx.shfl_down_f32(local_max, 16, 0xFFFF_FFFF);
                let warp_max_1 = ctx.max_f32(local_max, shuffled_16);
                let shuffled_8 = ctx.shfl_down_f32(warp_max_1, 8, 0xFFFF_FFFF);
                let warp_max_2 = ctx.max_f32(warp_max_1, shuffled_8);
                let shuffled_4 = ctx.shfl_down_f32(warp_max_2, 4, 0xFFFF_FFFF);
                let warp_max_3 = ctx.max_f32(warp_max_2, shuffled_4);
                let shuffled_2 = ctx.shfl_down_f32(warp_max_3, 2, 0xFFFF_FFFF);
                let warp_max_4 = ctx.max_f32(warp_max_3, shuffled_2);
                let shuffled_1 = ctx.shfl_down_f32(warp_max_4, 1, 0xFFFF_FFFF);
                let warp_max = ctx.max_f32(warp_max_4, shuffled_1);

                // Lane 0 of each warp stores warp max to shared memory
                let zero = ctx.mov_u32_imm(0);
                let is_lane_0 = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_lane_0, "skip_store_warp_max");
                let smem_offset = ctx.mul_u32(warp_id, 4);
                let smem_offset_64 = ctx.cvt_u64_u32(smem_offset);
                ctx.st_shared_f32(smem_offset_64, warp_max);
                ctx.label("skip_store_warp_max");

                // Synchronize
                ctx.bar_sync(0);

                // Warp 0 (all 32 lanes) reduce across warp maxes
                // ALL lanes must participate in shuffles to avoid deadlock!
                let is_warp_0 = ctx.setp_eq_u32(warp_id, zero);
                ctx.branch_if_not(is_warp_0, "skip_inter_warp_max");

                // Lanes 0-7 load valid warp maxes, lanes 8-31 load duplicates
                // using lane_id & 7 to clamp index to 0-7 (duplicates don't affect max)
                let seven = ctx.mov_u32_imm(7);
                let lane_id_clamped = ctx.and_u32(lane_id, seven);
                let lane_smem_offset = ctx.mul_u32(lane_id_clamped, 4);
                let lane_smem_64 = ctx.cvt_u64_u32(lane_smem_offset);
                let loaded_warp_max = ctx.ld_shared_f32(lane_smem_64);

                // Reduce 8 values using shuffles (all 32 lanes participate)
                let inter_4 = ctx.shfl_down_f32(loaded_warp_max, 4, 0xFFFF_FFFF);
                let inter_max_1 = ctx.max_f32(loaded_warp_max, inter_4);
                let inter_2 = ctx.shfl_down_f32(inter_max_1, 2, 0xFFFF_FFFF);
                let inter_max_2 = ctx.max_f32(inter_max_1, inter_2);
                let inter_1 = ctx.shfl_down_f32(inter_max_2, 1, 0xFFFF_FFFF);
                let global_max = ctx.max_f32(inter_max_2, inter_1);

                // Lane 0 stores global max at shared[8] (offset 32 bytes)
                let is_lane_0_check = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_lane_0_check, "skip_store_global_max");
                let global_max_offset = ctx.mov_u32_imm(32); // 8 * 4 bytes
                let global_max_offset_64 = ctx.cvt_u64_u32(global_max_offset);
                ctx.st_shared_f32(global_max_offset_64, global_max);
                ctx.label("skip_store_global_max");

                ctx.label("skip_inter_warp_max");

                // Synchronize
                ctx.bar_sync(1);

                // All threads load global max
                let global_max_read_offset = ctx.mov_u32_imm(32);
                let global_max_read_64 = ctx.cvt_u64_u32(global_max_read_offset);
                let global_max_val = ctx.ld_shared_f32(global_max_read_64);

                // =========================================================
                // Phase 2: Compute sum(exp(x - max)) using grid-stride loop
                // =========================================================
                let local_sum = ctx.mov_f32_imm(0.0);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);

                let idx2 = ctx.add_u32(tid, 0);
                ctx.label("sum_loop");
                let done_sum = ctx.setp_ge_u32(idx2, row_size);
                ctx.branch_if(done_sum, "sum_loop_done");

                // Load input[idx]
                let byte_offset2 = ctx.mul_wide_u32(idx2, 4);
                let load_addr2 = ctx.add_u64(row_in_ptr, byte_offset2);
                let val2 = ctx.ld_global_f32(load_addr2);

                // exp_val = exp(val - global_max) = 2^((val - max) * log2(e))
                let shifted = ctx.sub_f32(val2, global_max_val);
                let scaled = ctx.mul_f32(shifted, log2_e);
                let exp_val = ctx.ex2_f32(scaled);

                // local_sum += exp_val
                ctx.add_f32_inplace(local_sum, exp_val);

                ctx.add_u32_reg_inplace(idx2, ntid);
                ctx.branch("sum_loop");

                ctx.label("sum_loop_done");

                // Warp-level sum reduction using shuffles
                let sum_shuffled_16 = ctx.shfl_down_f32(local_sum, 16, 0xFFFF_FFFF);
                let warp_sum_1 = ctx.add_f32(local_sum, sum_shuffled_16);
                let sum_shuffled_8 = ctx.shfl_down_f32(warp_sum_1, 8, 0xFFFF_FFFF);
                let warp_sum_2 = ctx.add_f32(warp_sum_1, sum_shuffled_8);
                let sum_shuffled_4 = ctx.shfl_down_f32(warp_sum_2, 4, 0xFFFF_FFFF);
                let warp_sum_3 = ctx.add_f32(warp_sum_2, sum_shuffled_4);
                let sum_shuffled_2 = ctx.shfl_down_f32(warp_sum_3, 2, 0xFFFF_FFFF);
                let warp_sum_4 = ctx.add_f32(warp_sum_3, sum_shuffled_2);
                let sum_shuffled_1 = ctx.shfl_down_f32(warp_sum_4, 1, 0xFFFF_FFFF);
                let warp_sum = ctx.add_f32(warp_sum_4, sum_shuffled_1);

                // Lane 0 of each warp stores warp sum to shared memory at offset 36+ bytes
                ctx.branch_if_not(is_lane_0, "skip_store_warp_sum");
                let sum_smem_base = ctx.mov_u32_imm(36); // after global_max
                let four = ctx.mov_u32_imm(4);
                let sum_smem_offset = ctx.mad_lo_u32(warp_id, four, sum_smem_base);
                let sum_smem_64 = ctx.cvt_u64_u32(sum_smem_offset);
                ctx.st_shared_f32(sum_smem_64, warp_sum);
                ctx.label("skip_store_warp_sum");

                // Synchronize
                ctx.bar_sync(2);

                // Warp 0 (all 32 lanes) reduce across warp sums
                ctx.branch_if_not(is_warp_0, "skip_inter_warp_sum");

                // Lanes 0-7 load valid warp sums, lanes 8-31 load duplicates
                let seven2 = ctx.mov_u32_imm(7);
                let lane_id_clamped2 = ctx.and_u32(lane_id, seven2);
                let sum_base2 = ctx.mov_u32_imm(36);
                let four2 = ctx.mov_u32_imm(4);
                let sum_lane_offset = ctx.mad_lo_u32(lane_id_clamped2, four2, sum_base2);
                let sum_lane_64 = ctx.cvt_u64_u32(sum_lane_offset);
                let loaded_warp_sum = ctx.ld_shared_f32(sum_lane_64);

                // Reduce 8 values using shuffles (all 32 lanes participate)
                let sum_inter_4 = ctx.shfl_down_f32(loaded_warp_sum, 4, 0xFFFF_FFFF);
                let inter_sum_1 = ctx.add_f32(loaded_warp_sum, sum_inter_4);
                let sum_inter_2 = ctx.shfl_down_f32(inter_sum_1, 2, 0xFFFF_FFFF);
                let inter_sum_2 = ctx.add_f32(inter_sum_1, sum_inter_2);
                let sum_inter_1 = ctx.shfl_down_f32(inter_sum_2, 1, 0xFFFF_FFFF);
                let global_sum = ctx.add_f32(inter_sum_2, sum_inter_1);

                // Lane 0 stores global sum at shared[17] (offset 68 bytes)
                let is_lane_0_sum = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_lane_0_sum, "skip_store_global_sum");
                let global_sum_offset = ctx.mov_u32_imm(68);
                let global_sum_offset_64 = ctx.cvt_u64_u32(global_sum_offset);
                ctx.st_shared_f32(global_sum_offset_64, global_sum);
                ctx.label("skip_store_global_sum");

                ctx.label("skip_inter_warp_sum");

                // Synchronize
                ctx.bar_sync(3);

                // All threads load global sum
                let global_sum_read_offset = ctx.mov_u32_imm(68);
                let global_sum_read_64 = ctx.cvt_u64_u32(global_sum_read_offset);
                let global_sum_val = ctx.ld_shared_f32(global_sum_read_64);

                // =========================================================
                // Phase 3: Normalize and write output: exp(x - max) / sum
                // =========================================================
                let idx3 = ctx.add_u32(tid, 0);
                ctx.label("write_loop");
                let done_write = ctx.setp_ge_u32(idx3, row_size);
                ctx.branch_if(done_write, "write_loop_done");

                // Load input[idx]
                let byte_offset3 = ctx.mul_wide_u32(idx3, 4);
                let load_addr3 = ctx.add_u64(row_in_ptr, byte_offset3);
                let val3 = ctx.ld_global_f32(load_addr3);

                // exp_val = exp(val - global_max)
                let shifted3 = ctx.sub_f32(val3, global_max_val);
                let scaled3 = ctx.mul_f32(shifted3, log2_e);
                let exp_val3 = ctx.ex2_f32(scaled3);

                // softmax_val = exp_val / global_sum
                let softmax_val = ctx.div_f32(exp_val3, global_sum_val);

                // Store result
                let out_addr = ctx.add_u64(row_out_ptr, byte_offset3);
                ctx.st_global_f32(out_addr, softmax_val);

                ctx.add_u32_reg_inplace(idx3, ntid);
                ctx.branch("write_loop");

                ctx.label("write_loop_done");
                ctx.ret();
            })
    }
}
