//! PTX generation for Multi-Warp Incremental Attention Kernel

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

use super::MultiWarpIncrementalAttentionKernel;

impl Kernel for MultiWarpIncrementalAttentionKernel {
    fn name(&self) -> &str {
        if self.indirect_seq_len {
            "multi_warp_attention_indirect"
        } else {
            "multi_warp_attention"
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let num_warps = self.num_warps_per_head;
        let indirect = self.indirect_seq_len;

        // Shared memory for cross-warp reduction:
        // - max_scores[num_warps]: local max per warp (offset 0)
        // - sum_exps[num_warps]: local sum_exp per warp (offset num_warps*4)
        // - global_max: 1 float (offset num_warps*8) - CORRECTNESS-013: separate from local values
        // - global_sum: 1 float (offset num_warps*8+4)
        // - outputs[num_warps * head_dim]: partial outputs per warp (offset num_warps*8+8)
        let smem_size = (num_warps * 2 + 2 + num_warps * head_dim) * 4;

        let kernel_name = if indirect {
            "multi_warp_attention_indirect"
        } else {
            "multi_warp_attention"
        };

        let mut builder = PtxKernel::new(kernel_name)
            .param(PtxType::U64, "q_ptr")
            .param(PtxType::U64, "k_ptr")
            .param(PtxType::U64, "v_ptr")
            .param(PtxType::U64, "out_ptr");

        builder = if indirect {
            builder.param(PtxType::U64, "seq_len_ptr")
        } else {
            builder.param(PtxType::U32, "seq_len")
        };

        builder.shared_memory(smem_size as usize).build(move |ctx| {
            // Grid: (num_heads, 1, 1) - one block per head
            // Block: (32 * num_warps, 1, 1) - multiple warps per block
            let q_head_idx = ctx.special_reg(PtxReg::CtaIdX);
            let tid = ctx.special_reg(PtxReg::TidX);

            // Compute warp_idx and lane_id from tid
            // warp_idx = tid / 32, lane_id = tid % 32
            let warp_idx = ctx.div_u32(tid, 32);
            let lane_id = ctx.rem_u32(tid, 32);

            // Load seq_len
            let seq_len = if indirect {
                let seq_len_ptr = ctx.load_param_u64("seq_len_ptr");
                ctx.ld_global_u32(seq_len_ptr)
            } else {
                ctx.load_param_u32("seq_len")
            };

            let q_ptr = ctx.load_param_u64("q_ptr");
            let k_ptr = ctx.load_param_u64("k_ptr");
            let v_ptr = ctx.load_param_u64("v_ptr");
            let out_ptr = ctx.load_param_u64("out_ptr");

            // Constants
            let four = ctx.mov_u32_imm(4);
            let head_dim_u32 = ctx.mov_u32_imm(head_dim);
            let num_warps_u32 = ctx.mov_u32_imm(num_warps);

            // Compute chunk boundaries for this warp
            // chunk_size = ceil(seq_len / num_warps) = (seq_len + num_warps - 1) / num_warps
            let seq_plus_nw = ctx.add_u32(seq_len, num_warps - 1);
            let chunk_size = ctx.div_u32(seq_plus_nw, num_warps);

            let start_pos = ctx.mul_lo_u32(warp_idx, chunk_size);
            let end_pos = ctx.add_u32_reg(start_pos, chunk_size);
            let end_pos = ctx.min_u32(end_pos, seq_len);

            // Compute Q/output head offset
            let q_head_off = ctx.mul_lo_u32(q_head_idx, head_dim_u32);
            let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
            let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);
            let out_head_ptr = ctx.add_u64(out_ptr, q_head_off_bytes);

            // GQA: kv_head_idx = q_head_idx * num_kv_heads / num_heads
            let kv_head_idx = ctx.mul_u32(q_head_idx, num_kv_heads);
            let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);

            // K/V: kv_head_idx * max_seq_len * head_dim
            let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
            let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
            let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
            let k_head_ptr = ctx.add_u64(k_ptr, kv_head_off_bytes);
            let v_head_ptr = ctx.add_u64(v_ptr, kv_head_off_bytes);

            // Load Q values (persistent across loop)
            let q0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
            let q0_addr = ctx.add_u64(q_head_ptr, q0_off_bytes);
            let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
            let q0 = ctx.ld_global_f32_predicated(q0_addr, in_bounds0, 0.0);

            let lane_plus_32 = ctx.add_u32(lane_id, 32);
            let q1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
            let q1_addr = ctx.add_u64(q_head_ptr, q1_off_bytes);
            let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
            let q1 = ctx.ld_global_f32_predicated(q1_addr, in_bounds1, 0.0);

            let lane_plus_64 = ctx.add_u32(lane_id, 64);
            let q2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
            let q2_addr = ctx.add_u64(q_head_ptr, q2_off_bytes);
            let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
            let q2 = ctx.ld_global_f32_predicated(q2_addr, in_bounds2, 0.0);

            let lane_plus_96 = ctx.add_u32(lane_id, 96);
            let q3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
            let q3_addr = ctx.add_u64(q_head_ptr, q3_off_bytes);
            let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);
            let q3 = ctx.ld_global_f32_predicated(q3_addr, in_bounds3, 0.0);

            // Initialize accumulators
            let out0 = ctx.mov_f32_imm(0.0);
            let out1 = ctx.mov_f32_imm(0.0);
            let out2 = ctx.mov_f32_imm(0.0);
            let out3 = ctx.mov_f32_imm(0.0);
            let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
            let sum_exp = ctx.mov_f32_imm(0.0);

            let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
            let scale_reg = ctx.mov_f32_imm(scale);

            // Loop over this warp's chunk of positions
            let pos = ctx.add_u32(start_pos, 0);

            ctx.label("chunk_loop");
            let loop_cond = ctx.setp_lt_u32(pos, end_pos);
            ctx.branch_if_not(loop_cond, "chunk_loop_end");

            // Load K[pos]
            let k_pos_off = ctx.mul_lo_u32(pos, head_dim_u32);
            let k0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
            let k0_off_bytes = ctx.mul_wide_u32_reg(k0_elem_off, four);
            let k0_addr = ctx.add_u64(k_head_ptr, k0_off_bytes);
            let k0 = ctx.ld_global_f32_predicated(k0_addr, in_bounds0, 0.0);

            let k1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
            let k1_off_bytes = ctx.mul_wide_u32_reg(k1_elem_off, four);
            let k1_addr = ctx.add_u64(k_head_ptr, k1_off_bytes);
            let k1 = ctx.ld_global_f32_predicated(k1_addr, in_bounds1, 0.0);

            let k2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
            let k2_off_bytes = ctx.mul_wide_u32_reg(k2_elem_off, four);
            let k2_addr = ctx.add_u64(k_head_ptr, k2_off_bytes);
            let k2 = ctx.ld_global_f32_predicated(k2_addr, in_bounds2, 0.0);

            let k3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
            let k3_off_bytes = ctx.mul_wide_u32_reg(k3_elem_off, four);
            let k3_addr = ctx.add_u64(k_head_ptr, k3_off_bytes);
            let k3 = ctx.ld_global_f32_predicated(k3_addr, in_bounds3, 0.0);

            // Dot product Q·K
            let dot = ctx.mul_f32(q0, k0);
            let dot = ctx.fma_f32(q1, k1, dot);
            let dot = ctx.fma_f32(q2, k2, dot);
            let dot = ctx.fma_f32(q3, k3, dot);

            // Warp-reduce
            let dot16 = ctx.shfl_down_f32(dot, 16, 0xFFFF_FFFF);
            let dot = ctx.add_f32(dot, dot16);
            let dot8 = ctx.shfl_down_f32(dot, 8, 0xFFFF_FFFF);
            let dot = ctx.add_f32(dot, dot8);
            let dot4 = ctx.shfl_down_f32(dot, 4, 0xFFFF_FFFF);
            let dot = ctx.add_f32(dot, dot4);
            let dot2 = ctx.shfl_down_f32(dot, 2, 0xFFFF_FFFF);
            let dot = ctx.add_f32(dot, dot2);
            let dot1 = ctx.shfl_down_f32(dot, 1, 0xFFFF_FFFF);
            let dot = ctx.add_f32(dot, dot1);
            let score = ctx.shfl_idx_f32(dot, 0, 0xFFFF_FFFF);
            let score = ctx.mul_f32(score, scale_reg);

            // Online softmax update
            let new_max = ctx.max_f32(max_score, score);
            let max_diff = ctx.sub_f32(max_score, new_max);
            let max_diff_scaled = ctx.mul_f32(max_diff, log2e);
            let correction = ctx.ex2_f32(max_diff_scaled);
            let score_diff = ctx.sub_f32(score, new_max);
            let score_diff_scaled = ctx.mul_f32(score_diff, log2e);
            let exp_score = ctx.ex2_f32(score_diff_scaled);

            // Load V[pos]
            let v0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
            let v0_off_bytes = ctx.mul_wide_u32_reg(v0_elem_off, four);
            let v0_addr = ctx.add_u64(v_head_ptr, v0_off_bytes);
            let v0 = ctx.ld_global_f32_predicated(v0_addr, in_bounds0, 0.0);

            let v1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
            let v1_off_bytes = ctx.mul_wide_u32_reg(v1_elem_off, four);
            let v1_addr = ctx.add_u64(v_head_ptr, v1_off_bytes);
            let v1 = ctx.ld_global_f32_predicated(v1_addr, in_bounds1, 0.0);

            let v2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
            let v2_off_bytes = ctx.mul_wide_u32_reg(v2_elem_off, four);
            let v2_addr = ctx.add_u64(v_head_ptr, v2_off_bytes);
            let v2 = ctx.ld_global_f32_predicated(v2_addr, in_bounds2, 0.0);

            let v3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
            let v3_off_bytes = ctx.mul_wide_u32_reg(v3_elem_off, four);
            let v3_addr = ctx.add_u64(v_head_ptr, v3_off_bytes);
            let v3 = ctx.ld_global_f32_predicated(v3_addr, in_bounds3, 0.0);

            // Update state
            ctx.max_f32_inplace(max_score, score);
            ctx.mul_f32_inplace(sum_exp, correction);
            ctx.add_f32_inplace(sum_exp, exp_score);

            ctx.mul_f32_inplace(out0, correction);
            ctx.fma_f32_inplace(out0, exp_score, v0);
            ctx.mul_f32_inplace(out1, correction);
            ctx.fma_f32_inplace(out1, exp_score, v1);
            ctx.mul_f32_inplace(out2, correction);
            ctx.fma_f32_inplace(out2, exp_score, v2);
            ctx.mul_f32_inplace(out3, correction);
            ctx.fma_f32_inplace(out3, exp_score, v3);

            ctx.add_u32_inplace(pos, 1);
            ctx.branch("chunk_loop");

            ctx.label("chunk_loop_end");

            // Store local max and sum_exp to shared memory for cross-warp reduction
            // smem layout: [max_0..max_n, sum_0..sum_n, out_0..out_n*head_dim]
            // Shared memory uses offset from 0, not a base pointer in PTX
            let warp_off = ctx.mul_u32(warp_idx, 4);
            let warp_off_64 = ctx.cvt_u64_u32(warp_off);
            let max_off_base = ctx.mov_u64_imm(0);
            let max_addr = ctx.add_u64(max_off_base, warp_off_64);

            let sum_off_base = ctx.mov_u64_imm((num_warps * 4) as u64);
            let sum_addr = ctx.add_u64(sum_off_base, warp_off_64);

            // Only lane 0 writes to shared memory
            let zero_u32 = ctx.mov_u32_imm(0);
            let is_lane0 = ctx.setp_eq_u32(lane_id, zero_u32);
            ctx.branch_if_not(is_lane0, "skip_smem_write");
            ctx.st_shared_f32(max_addr, max_score);
            ctx.st_shared_f32(sum_addr, sum_exp);
            ctx.label("skip_smem_write");

            // Barrier for all threads in this block
            ctx.bar_sync(0);

            // Lane 0 of warp 0 reduces global max across warps
            let is_warp0 = ctx.setp_eq_u32(warp_idx, zero_u32);
            let is_warp0_lane0 = ctx.and_pred(is_warp0, is_lane0);

            ctx.branch_if_not(is_warp0_lane0, "skip_reduce");

            // Compute global max
            let global_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
            let reduce_i = ctx.mov_u32_imm(0);
            ctx.label("reduce_max_loop");
            let reduce_cond = ctx.setp_lt_u32(reduce_i, num_warps_u32);
            ctx.branch_if_not(reduce_cond, "reduce_max_done");

            let i_off = ctx.mul_u32(reduce_i, 4);
            let i_off_64 = ctx.cvt_u64_u32(i_off);
            let max_i_addr = ctx.add_u64(max_off_base, i_off_64);
            let max_i = ctx.ld_shared_f32(max_i_addr);
            ctx.max_f32_inplace(global_max, max_i);
            ctx.add_u32_inplace(reduce_i, 1);
            ctx.branch("reduce_max_loop");

            ctx.label("reduce_max_done");

            // Compute global sum_exp = sum(sum_exp_i * exp(max_i - global_max))
            let global_sum = ctx.mov_f32_imm(0.0);
            let reduce_i = ctx.mov_u32_imm(0);
            ctx.label("reduce_sum_loop");
            let reduce_cond = ctx.setp_lt_u32(reduce_i, num_warps_u32);
            ctx.branch_if_not(reduce_cond, "reduce_sum_done");

            let i_off = ctx.mul_u32(reduce_i, 4);
            let i_off_64 = ctx.cvt_u64_u32(i_off);
            let max_i_addr = ctx.add_u64(max_off_base, i_off_64);
            let max_i = ctx.ld_shared_f32(max_i_addr);
            let sum_i_addr = ctx.add_u64(sum_off_base, i_off_64);
            let sum_i = ctx.ld_shared_f32(sum_i_addr);

            let diff = ctx.sub_f32(max_i, global_max);
            let diff_scaled = ctx.mul_f32(diff, log2e);
            let correction = ctx.ex2_f32(diff_scaled);
            let corrected_sum = ctx.mul_f32(sum_i, correction);
            ctx.add_f32_inplace(global_sum, corrected_sum);

            ctx.add_u32_inplace(reduce_i, 1);
            ctx.branch("reduce_sum_loop");

            ctx.label("reduce_sum_done");

            // CORRECTNESS-013: Store global_max and global_sum to SEPARATE locations
            // Separate offsets prevent overwriting warp 0's local max via max_off_base (0)
            let global_max_off = ctx.mov_u64_imm((num_warps * 8) as u64);
            let global_sum_off = ctx.mov_u64_imm((num_warps * 8 + 4) as u64);
            ctx.st_shared_f32(global_max_off, global_max);
            ctx.st_shared_f32(global_sum_off, global_sum);

            ctx.label("skip_reduce");

            // Barrier to wait for reduction
            ctx.bar_sync(1);

            // CORRECTNESS-013: Load global max and sum from dedicated locations
            let global_max_off = ctx.mov_u64_imm((num_warps * 8) as u64);
            let global_sum_off = ctx.mov_u64_imm((num_warps * 8 + 4) as u64);
            let global_max = ctx.ld_shared_f32(global_max_off);
            let global_sum = ctx.ld_shared_f32(global_sum_off);

            // Compute correction for this warp's contribution
            let my_max = ctx.ld_shared_f32(max_addr);
            let my_diff = ctx.sub_f32(my_max, global_max);
            let my_diff_scaled = ctx.mul_f32(my_diff, log2e);
            let my_correction = ctx.ex2_f32(my_diff_scaled);

            // Correct and normalize this warp's output
            let one = ctx.mov_f32_imm(1.0);
            let inv_sum = ctx.div_f32(one, global_sum);
            let final_scale = ctx.mul_f32(my_correction, inv_sum);

            ctx.mul_f32_inplace(out0, final_scale);
            ctx.mul_f32_inplace(out1, final_scale);
            ctx.mul_f32_inplace(out2, final_scale);
            ctx.mul_f32_inplace(out3, final_scale);

            // Store corrected outputs to shared memory for warp 0 to sum
            // CORRECTNESS-013: Output area starts at offset: num_warps*8 + 8 (after global max/sum)
            // Each warp stores head_dim * 4 bytes
            let out_area_base = ctx.mov_u32_imm(num_warps * 8 + 8);
            let warp_out_offset = ctx.mul_u32(warp_idx, head_dim * 4);
            let out_base = ctx.add_u32_reg(out_area_base, warp_out_offset);

            // Store out0-out3 based on lane positions
            let lane_off_0 = ctx.mul_u32(lane_id, 4);
            let out0_smem_off = ctx.add_u32_reg(out_base, lane_off_0);
            let out0_smem_addr = ctx.cvt_u64_u32(out0_smem_off);
            ctx.branch_if_not(in_bounds0, "skip_store_out0");
            ctx.st_shared_f32(out0_smem_addr, out0);
            ctx.label("skip_store_out0");

            let lane_off_1 = ctx.mul_u32(lane_plus_32, 4);
            let out1_smem_off = ctx.add_u32_reg(out_base, lane_off_1);
            let out1_smem_addr = ctx.cvt_u64_u32(out1_smem_off);
            ctx.branch_if_not(in_bounds1, "skip_store_out1");
            ctx.st_shared_f32(out1_smem_addr, out1);
            ctx.label("skip_store_out1");

            let lane_off_2 = ctx.mul_u32(lane_plus_64, 4);
            let out2_smem_off = ctx.add_u32_reg(out_base, lane_off_2);
            let out2_smem_addr = ctx.cvt_u64_u32(out2_smem_off);
            ctx.branch_if_not(in_bounds2, "skip_store_out2");
            ctx.st_shared_f32(out2_smem_addr, out2);
            ctx.label("skip_store_out2");

            let lane_off_3 = ctx.mul_u32(lane_plus_96, 4);
            let out3_smem_off = ctx.add_u32_reg(out_base, lane_off_3);
            let out3_smem_addr = ctx.cvt_u64_u32(out3_smem_off);
            ctx.branch_if_not(in_bounds3, "skip_store_out3");
            ctx.st_shared_f32(out3_smem_addr, out3);
            ctx.label("skip_store_out3");

            // Barrier to ensure all warps stored their outputs
            ctx.bar_sync(2);

            // Only warp 0 sums outputs and stores to global memory
            ctx.branch_if_not(is_warp0, "skip_final_sum");

            // Sum outputs from all warps for each element this warp handles
            // Thread i in warp 0 sums element i across all warps
            let final_out0 = ctx.mov_f32_imm(0.0);
            let sum_w = ctx.mov_u32_imm(0);
            ctx.label("sum_warps_loop0");
            let sum_cond = ctx.setp_lt_u32(sum_w, num_warps_u32);
            ctx.branch_if_not(sum_cond, "sum_warps_done0");

            let w_out_offset = ctx.mul_u32(sum_w, head_dim * 4);
            let w_out_base = ctx.add_u32_reg(out_area_base, w_out_offset);
            let elem_off = ctx.mul_u32(lane_id, 4);
            let elem_addr_off = ctx.add_u32_reg(w_out_base, elem_off);
            let elem_addr = ctx.cvt_u64_u32(elem_addr_off);
            let elem_val = ctx.ld_shared_f32(elem_addr);
            ctx.add_f32_inplace(final_out0, elem_val);
            ctx.add_u32_inplace(sum_w, 1);
            ctx.branch("sum_warps_loop0");

            ctx.label("sum_warps_done0");

            // Store final output to global memory
            let out0_addr = ctx.add_u64(out_head_ptr, q0_off_bytes);
            ctx.branch_if_not(in_bounds0, "skip_final_store0");
            ctx.st_global_f32(out0_addr, final_out0);
            ctx.label("skip_final_store0");

            // Repeat for elements 32-63 (out1)
            let final_out1 = ctx.mov_f32_imm(0.0);
            let sum_w = ctx.mov_u32_imm(0);
            ctx.label("sum_warps_loop1");
            let sum_cond = ctx.setp_lt_u32(sum_w, num_warps_u32);
            ctx.branch_if_not(sum_cond, "sum_warps_done1");

            let w_out_offset = ctx.mul_u32(sum_w, head_dim * 4);
            let w_out_base = ctx.add_u32_reg(out_area_base, w_out_offset);
            let elem_off = ctx.mul_u32(lane_plus_32, 4);
            let elem_addr_off = ctx.add_u32_reg(w_out_base, elem_off);
            let elem_addr = ctx.cvt_u64_u32(elem_addr_off);
            let elem_val = ctx.ld_shared_f32(elem_addr);
            ctx.add_f32_inplace(final_out1, elem_val);
            ctx.add_u32_inplace(sum_w, 1);
            ctx.branch("sum_warps_loop1");

            ctx.label("sum_warps_done1");

            let out1_addr = ctx.add_u64(out_head_ptr, q1_off_bytes);
            ctx.branch_if_not(in_bounds1, "skip_final_store1");
            ctx.st_global_f32(out1_addr, final_out1);
            ctx.label("skip_final_store1");

            // Repeat for elements 64-95 (out2)
            let final_out2 = ctx.mov_f32_imm(0.0);
            let sum_w = ctx.mov_u32_imm(0);
            ctx.label("sum_warps_loop2");
            let sum_cond = ctx.setp_lt_u32(sum_w, num_warps_u32);
            ctx.branch_if_not(sum_cond, "sum_warps_done2");

            let w_out_offset = ctx.mul_u32(sum_w, head_dim * 4);
            let w_out_base = ctx.add_u32_reg(out_area_base, w_out_offset);
            let elem_off = ctx.mul_u32(lane_plus_64, 4);
            let elem_addr_off = ctx.add_u32_reg(w_out_base, elem_off);
            let elem_addr = ctx.cvt_u64_u32(elem_addr_off);
            let elem_val = ctx.ld_shared_f32(elem_addr);
            ctx.add_f32_inplace(final_out2, elem_val);
            ctx.add_u32_inplace(sum_w, 1);
            ctx.branch("sum_warps_loop2");

            ctx.label("sum_warps_done2");

            let out2_addr = ctx.add_u64(out_head_ptr, q2_off_bytes);
            ctx.branch_if_not(in_bounds2, "skip_final_store2");
            ctx.st_global_f32(out2_addr, final_out2);
            ctx.label("skip_final_store2");

            // Repeat for elements 96-127 (out3)
            let final_out3 = ctx.mov_f32_imm(0.0);
            let sum_w = ctx.mov_u32_imm(0);
            ctx.label("sum_warps_loop3");
            let sum_cond = ctx.setp_lt_u32(sum_w, num_warps_u32);
            ctx.branch_if_not(sum_cond, "sum_warps_done3");

            let w_out_offset = ctx.mul_u32(sum_w, head_dim * 4);
            let w_out_base = ctx.add_u32_reg(out_area_base, w_out_offset);
            let elem_off = ctx.mul_u32(lane_plus_96, 4);
            let elem_addr_off = ctx.add_u32_reg(w_out_base, elem_off);
            let elem_addr = ctx.cvt_u64_u32(elem_addr_off);
            let elem_val = ctx.ld_shared_f32(elem_addr);
            ctx.add_f32_inplace(final_out3, elem_val);
            ctx.add_u32_inplace(sum_w, 1);
            ctx.branch("sum_warps_loop3");

            ctx.label("sum_warps_done3");

            let out3_addr = ctx.add_u64(out_head_ptr, q3_off_bytes);
            ctx.branch_if_not(in_bounds3, "skip_final_store3");
            ctx.st_global_f32(out3_addr, final_out3);
            ctx.label("skip_final_store3");

            ctx.label("skip_final_sum");

            ctx.ret();
        })
    }
}
