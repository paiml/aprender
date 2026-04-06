//! Flash Decoding Reduce Kernel - combines partial attention results from chunks.

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

use super::FLASH_DECODE_CHUNK_SIZE;

/// PAR-118: Flash Decoding reduction kernel
///
/// Reduces partial attention results from multiple chunks into final output.
/// Uses online softmax rescaling to combine results correctly.
///
/// Memory layout:
/// - partials: [M, num_heads, num_chunks, head_dim + 2]
/// - output: [M, num_heads, head_dim]
/// - seq_lens: [M] - sequence lengths (to compute actual num_chunks)
#[derive(Debug, Clone)]
pub struct FlashDecodingReduceKernel {
    /// Head dimension
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Chunk size used in chunk kernel
    pub chunk_size: u32,
}

impl FlashDecodingReduceKernel {
    /// Create a new Flash Decoding reduce kernel
    #[must_use]
    pub fn new(head_dim: u32, num_heads: u32, batch_size: u32) -> Self {
        Self { head_dim, num_heads, batch_size, chunk_size: FLASH_DECODE_CHUNK_SIZE }
    }
}

impl Kernel for FlashDecodingReduceKernel {
    fn name(&self) -> &str {
        "flash_decoding_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let chunk_size = self.chunk_size;
        let _batch_size = self.batch_size;

        // Grid: (num_heads, batch_size, 1)
        // Block: (32, 1, 1) - one warp per block
        //
        // Each block reduces all chunks for one (head, batch) pair

        PtxKernel::new("flash_decoding_reduce")
            .param(PtxType::U64, "partials_ptr") // [M, num_heads, max_chunks, head_dim + 2]
            .param(PtxType::U64, "output_ptr") // [M, num_heads, head_dim]
            .param(PtxType::U64, "seq_lens_ptr") // [M] sequence lengths
            .param(PtxType::U32, "max_chunks") // Maximum number of chunks
            .shared_memory(0)
            .build(move |ctx| {
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                let partials_ptr = ctx.load_param_u64("partials_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let seq_lens_ptr = ctx.load_param_u64("seq_lens_ptr");
                let max_chunks = ctx.load_param_u32("max_chunks");

                let four = ctx.mov_u32_imm(4);

                // Load seq_len and compute actual number of chunks
                let batch_idx_bytes = ctx.mul_wide_u32_reg(batch_idx, four);
                let seq_len_addr = ctx.add_u64(seq_lens_ptr, batch_idx_bytes);
                let seq_len = ctx.ld_global_u32(seq_len_addr);
                // num_chunks = (seq_len + chunk_size - 1) / chunk_size
                let seq_plus_chunk_m1 = ctx.add_u32(seq_len, chunk_size - 1);
                let num_chunks = ctx.div_u32(seq_plus_chunk_m1, chunk_size);

                // Compute partials base for this (batch, head) pair
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let num_heads_u32 = ctx.mov_u32_imm(num_heads);
                let head_dim_plus_2 = ctx.mov_u32_imm(head_dim + 2);
                let partial_stride = ctx.mul_lo_u32(max_chunks, head_dim_plus_2);
                let batch_partial_stride = ctx.mul_lo_u32(num_heads_u32, partial_stride);
                let batch_partial_off = ctx.mul_lo_u32(batch_idx, batch_partial_stride);
                let head_partial_off = ctx.mul_lo_u32(head_idx, partial_stride);
                let partial_base_off = ctx.add_u32_reg(batch_partial_off, head_partial_off);
                let partial_base_off_bytes = ctx.mul_wide_u32_reg(partial_base_off, four);
                let partial_base = ctx.add_u64(partials_ptr, partial_base_off_bytes);

                // First pass: find global max across all chunks
                let global_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let chunk_iter = ctx.mov_u32_imm(0);
                ctx.label("reduce_max_loop");
                let max_loop_cond = ctx.setp_lt_u32(chunk_iter, num_chunks);
                ctx.branch_if_not(max_loop_cond, "reduce_max_loop_end");

                // Load this chunk's max_score (at offset head_dim within chunk)
                let chunk_off = ctx.mul_lo_u32(chunk_iter, head_dim_plus_2);
                let max_elem_off = ctx.add_u32(chunk_off, head_dim);
                let max_elem_off_bytes = ctx.mul_wide_u32_reg(max_elem_off, four);
                let chunk_max_addr = ctx.add_u64(partial_base, max_elem_off_bytes);
                // Only lane 0 needs to read (broadcast later)
                let zero_lane = ctx.mov_u32_imm(0);
                let is_lane0 = ctx.setp_eq_u32(lane_id, zero_lane);
                let chunk_max =
                    ctx.ld_global_f32_predicated(chunk_max_addr, is_lane0, f32::NEG_INFINITY);
                // Broadcast from lane 0 to all lanes
                let chunk_max = ctx.shfl_idx_f32(chunk_max, 0, 0xFFFF_FFFF);
                ctx.max_f32_inplace(global_max, chunk_max);

                ctx.add_u32_inplace(chunk_iter, 1);
                ctx.branch("reduce_max_loop");
                ctx.label("reduce_max_loop_end");

                // Second pass: accumulate rescaled outputs
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let global_sum = ctx.mov_f32_imm(0.0);
                let acc0 = ctx.mov_f32_imm(0.0);
                let acc1 = ctx.mov_f32_imm(0.0);
                let acc2 = ctx.mov_f32_imm(0.0);
                let acc3 = ctx.mov_f32_imm(0.0);

                let lane_plus_32 = ctx.add_u32(lane_id, 32);
                let lane_plus_64 = ctx.add_u32(lane_id, 64);
                let lane_plus_96 = ctx.add_u32(lane_id, 96);
                let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
                let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
                let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
                let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);

                let chunk_iter2 = ctx.mov_u32_imm(0);
                ctx.label("reduce_acc_loop");
                let acc_loop_cond = ctx.setp_lt_u32(chunk_iter2, num_chunks);
                ctx.branch_if_not(acc_loop_cond, "reduce_acc_loop_end");

                // Load chunk max and sum_exp
                let chunk_off2 = ctx.mul_lo_u32(chunk_iter2, head_dim_plus_2);
                let max_elem_off2 = ctx.add_u32(chunk_off2, head_dim);
                let sum_elem_off2 = ctx.add_u32(chunk_off2, head_dim);
                let sum_elem_off2 = ctx.add_u32(sum_elem_off2, 1);
                let max_off_bytes2 = ctx.mul_wide_u32_reg(max_elem_off2, four);
                let sum_off_bytes2 = ctx.mul_wide_u32_reg(sum_elem_off2, four);
                let chunk_max_addr2 = ctx.add_u64(partial_base, max_off_bytes2);
                let chunk_sum_addr2 = ctx.add_u64(partial_base, sum_off_bytes2);

                let chunk_max2 =
                    ctx.ld_global_f32_predicated(chunk_max_addr2, is_lane0, f32::NEG_INFINITY);
                let chunk_sum2 = ctx.ld_global_f32_predicated(chunk_sum_addr2, is_lane0, 0.0);
                let chunk_max2 = ctx.shfl_idx_f32(chunk_max2, 0, 0xFFFF_FFFF);
                let chunk_sum2 = ctx.shfl_idx_f32(chunk_sum2, 0, 0xFFFF_FFFF);

                // Skip empty chunks (max = -inf)
                let neg_inf_check = ctx.mov_f32_imm(-1e30);
                let is_valid = ctx.setp_gt_f32(chunk_max2, neg_inf_check);
                ctx.branch_if_not(is_valid, "reduce_skip_chunk");

                // Compute scale factor: exp(chunk_max - global_max)
                let max_diff = ctx.sub_f32(chunk_max2, global_max);
                let max_diff_log2 = ctx.mul_f32(max_diff, log2e);
                let scale_factor = ctx.ex2_f32(max_diff_log2);

                // Accumulate scaled sum_exp
                let scaled_sum = ctx.mul_f32(chunk_sum2, scale_factor);
                ctx.add_f32_inplace(global_sum, scaled_sum);

                // Load and accumulate scaled output
                let chunk_base_off_bytes = ctx.mul_wide_u32_reg(chunk_off2, four);
                let chunk_base = ctx.add_u64(partial_base, chunk_base_off_bytes);

                let out0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
                let out0_addr = ctx.add_u64(chunk_base, out0_off_bytes);
                let out0 = ctx.ld_global_f32_predicated(out0_addr, in_bounds0, 0.0);
                let scaled_out0 = ctx.mul_f32(out0, scale_factor);
                ctx.add_f32_inplace(acc0, scaled_out0);

                let out1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
                let out1_addr = ctx.add_u64(chunk_base, out1_off_bytes);
                let out1 = ctx.ld_global_f32_predicated(out1_addr, in_bounds1, 0.0);
                let scaled_out1 = ctx.mul_f32(out1, scale_factor);
                ctx.add_f32_inplace(acc1, scaled_out1);

                let out2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
                let out2_addr = ctx.add_u64(chunk_base, out2_off_bytes);
                let out2 = ctx.ld_global_f32_predicated(out2_addr, in_bounds2, 0.0);
                let scaled_out2 = ctx.mul_f32(out2, scale_factor);
                ctx.add_f32_inplace(acc2, scaled_out2);

                let out3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
                let out3_addr = ctx.add_u64(chunk_base, out3_off_bytes);
                let out3 = ctx.ld_global_f32_predicated(out3_addr, in_bounds3, 0.0);
                let scaled_out3 = ctx.mul_f32(out3, scale_factor);
                ctx.add_f32_inplace(acc3, scaled_out3);

                ctx.label("reduce_skip_chunk");
                ctx.add_u32_inplace(chunk_iter2, 1);
                ctx.branch("reduce_acc_loop");
                ctx.label("reduce_acc_loop_end");

                // Final normalization: output = acc / global_sum
                let one = ctx.mov_f32_imm(1.0);
                let inv_sum = ctx.div_f32(one, global_sum);
                ctx.mul_f32_inplace(acc0, inv_sum);
                ctx.mul_f32_inplace(acc1, inv_sum);
                ctx.mul_f32_inplace(acc2, inv_sum);
                ctx.mul_f32_inplace(acc3, inv_sum);

                // Compute output offset: batch_idx * num_heads * head_dim + head_idx * head_dim
                let batch_head_stride = ctx.mul_lo_u32(num_heads_u32, head_dim_u32);
                let batch_off = ctx.mul_lo_u32(batch_idx, batch_head_stride);
                let head_off = ctx.mul_lo_u32(head_idx, head_dim_u32);
                let out_base_off = ctx.add_u32_reg(batch_off, head_off);
                let out_base_off_bytes = ctx.mul_wide_u32_reg(out_base_off, four);
                let out_base = ctx.add_u64(output_ptr, out_base_off_bytes);

                // Store output
                let final_out0_addr = ctx.add_u64(out_base, out0_off_bytes);
                ctx.branch_if_not(in_bounds0, "reduce_skip_store0");
                ctx.st_global_f32(final_out0_addr, acc0);
                ctx.label("reduce_skip_store0");

                let final_out1_addr = ctx.add_u64(out_base, out1_off_bytes);
                ctx.branch_if_not(in_bounds1, "reduce_skip_store1");
                ctx.st_global_f32(final_out1_addr, acc1);
                ctx.label("reduce_skip_store1");

                let final_out2_addr = ctx.add_u64(out_base, out2_off_bytes);
                ctx.branch_if_not(in_bounds2, "reduce_skip_store2");
                ctx.st_global_f32(final_out2_addr, acc2);
                ctx.label("reduce_skip_store2");

                let final_out3_addr = ctx.add_u64(out_base, out3_off_bytes);
                ctx.branch_if_not(in_bounds3, "reduce_skip_store3");
                ctx.st_global_f32(final_out3_addr, acc3);
                ctx.label("reduce_skip_store3");

                ctx.ret();
            })
    }
}
