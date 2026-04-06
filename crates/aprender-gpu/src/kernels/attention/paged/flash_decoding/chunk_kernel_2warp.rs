//! trueno#253: 2-warp Flash Decoding Chunk Kernel
//!
//! Same algorithm as chunk_kernel.rs but with 2 warps per block (64 threads).
//! Each warp handles half of head_dim (64 elements instead of 128).
//! Cross-warp Q·K reduction via shared memory (8 bytes).
//!
//! Expected: +100% occupancy (4.7% → 9.4%) on RTX 4090 at M=1 decode.
//! Grid stays the same: (num_heads, batch_size, num_chunks).

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

use super::FLASH_DECODE_CHUNK_SIZE;

/// trueno#253: 2-warp variant of FlashDecodingChunkKernel.
/// Block = (32, 2, 1). Shared memory = 8 bytes (2 floats for partial Q·K sums).
#[derive(Debug, Clone)]
pub struct FlashDecodingChunkKernel2Warp {
    pub max_seq_len: u32,
    pub head_dim: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub batch_size: u32,
    pub chunk_size: u32,
    pub scale: f32,
}

impl FlashDecodingChunkKernel2Warp {
    #[must_use]
    pub fn new(
        max_seq_len: u32,
        head_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
        batch_size: u32,
    ) -> Self {
        assert!(head_dim == 128, "2-warp kernel requires head_dim=128");
        Self {
            max_seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            batch_size,
            chunk_size: FLASH_DECODE_CHUNK_SIZE,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    #[must_use]
    pub fn num_chunks(&self, seq_len: u32) -> u32 {
        (seq_len + self.chunk_size - 1) / self.chunk_size
    }

    #[must_use]
    pub fn partials_size_per_head(&self, max_chunks: u32) -> u32 {
        max_chunks * (self.head_dim + 2)
    }
}

impl Kernel for FlashDecodingChunkKernel2Warp {
    fn name(&self) -> &str {
        "flash_decoding_chunk_2warp"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let chunk_size = self.chunk_size;

        // Grid: (num_heads, batch_size, num_chunks) — same as 1-warp
        // Block: (32, 2, 1) — 2 warps per block
        // Shared memory: 8 bytes (2 floats for partial Q·K dot products)
        PtxKernel::new("flash_decoding_chunk_2warp")
            .param(PtxType::U64, "q_ptr")
            .param(PtxType::U64, "k_ptrs_ptr")
            .param(PtxType::U64, "v_ptrs_ptr")
            .param(PtxType::U64, "partials_ptr")
            .param(PtxType::U64, "seq_lens_ptr")
            .param(PtxType::U32, "max_chunks")
            .shared_memory(8) // 2 floats for cross-warp reduction
            .build(move |ctx| {
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);
                let chunk_idx = ctx.special_reg(PtxReg::CtaIdZ);
                let lane_id = ctx.special_reg(PtxReg::TidX);
                let warp_id = ctx.special_reg(PtxReg::TidY); // 0 or 1

                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptrs_ptr = ctx.load_param_u64("k_ptrs_ptr");
                let v_ptrs_ptr = ctx.load_param_u64("v_ptrs_ptr");
                let partials_ptr = ctx.load_param_u64("partials_ptr");
                let seq_lens_ptr = ctx.load_param_u64("seq_lens_ptr");
                let max_chunks_param = ctx.load_param_u32("max_chunks");

                let four = ctx.mov_u32_imm(4);
                let eight = ctx.mov_u32_imm(8);

                // Load seq_len for this batch element
                let batch_idx_bytes = ctx.mul_wide_u32_reg(batch_idx, four);
                let seq_len_addr = ctx.add_u64(seq_lens_ptr, batch_idx_bytes);
                let seq_len = ctx.ld_global_u32(seq_len_addr);

                // Chunk boundaries
                let chunk_size_u32 = ctx.mov_u32_imm(chunk_size);
                let chunk_start = ctx.mul_lo_u32(chunk_idx, chunk_size_u32);
                let chunk_end_raw = ctx.add_u32(chunk_start, chunk_size);
                let chunk_end = ctx.min_u32(chunk_end_raw, seq_len);

                let has_work = ctx.setp_lt_u32(chunk_start, seq_len);
                ctx.branch_if_not(has_work, "fd2w_empty");

                // Load K/V cache pointers
                let batch_ptr_off = ctx.mul_wide_u32_reg(batch_idx, eight);
                let k_ptr_addr = ctx.add_u64(k_ptrs_ptr, batch_ptr_off);
                let v_ptr_addr = ctx.add_u64(v_ptrs_ptr, batch_ptr_off);
                let k_cache_ptr = ctx.ld_global_u64(k_ptr_addr);
                let v_cache_ptr = ctx.ld_global_u64(v_ptr_addr);

                // Q offset: batch * num_heads * head_dim + head * head_dim
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let num_heads_u32 = ctx.mov_u32_imm(num_heads);
                let batch_head_stride = ctx.mul_lo_u32(num_heads_u32, head_dim_u32);
                let batch_off = ctx.mul_lo_u32(batch_idx, batch_head_stride);
                let head_off = ctx.mul_lo_u32(head_idx, head_dim_u32);
                let q_head_off = ctx.add_u32_reg(batch_off, head_off);
                let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
                let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);

                // GQA: KV head index
                let kv_head_idx = ctx.mul_u32(head_idx, num_kv_heads);
                let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);
                let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
                let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
                let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
                let k_head_ptr = ctx.add_u64(k_cache_ptr, kv_head_off_bytes);
                let v_head_ptr = ctx.add_u64(v_cache_ptr, kv_head_off_bytes);

                // trueno#253: Each warp handles 64 elements of head_dim=128
                // Warp 0: elements [lane_id, lane_id+32] (0..63)
                // Warp 1: elements [64+lane_id, 64+lane_id+32] (64..127)
                let warp_base = ctx.mul_u32(warp_id, 64); // 0 or 64
                let elem0 = ctx.add_u32_reg(warp_base, lane_id); // warp0: 0..31, warp1: 64..95
                let thirty_two = ctx.mov_u32_imm(32);
                let elem1 = ctx.add_u32_reg(elem0, thirty_two); // warp0: 32..63, warp1: 96..127

                let in_bounds0 = ctx.setp_lt_u32(elem0, head_dim_u32);
                let in_bounds1 = ctx.setp_lt_u32(elem1, head_dim_u32);

                // Load Q (2 elements per thread)
                let q0_off_bytes = ctx.mul_wide_u32_reg(elem0, four);
                let q0_addr = ctx.add_u64(q_head_ptr, q0_off_bytes);
                let q0 = ctx.ld_global_f32_predicated(q0_addr, in_bounds0, 0.0);

                let q1_off_bytes = ctx.mul_wide_u32_reg(elem1, four);
                let q1_addr = ctx.add_u64(q_head_ptr, q1_off_bytes);
                let q1 = ctx.ld_global_f32_predicated(q1_addr, in_bounds1, 0.0);

                // Accumulators
                let out0 = ctx.mov_f32_imm(0.0);
                let out1 = ctx.mov_f32_imm(0.0);
                let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let sum_exp = ctx.mov_f32_imm(0.0);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scale_reg = ctx.mov_f32_imm(scale);

                // Shared memory offsets (u32, like rmsnorm pattern)
                // smem[0] = warp 0 partial, smem[4] = warp 1 partial
                let smem_off_warp = ctx.mul_u32(warp_id, 4); // u32: 0 or 4
                let smem_off_0 = ctx.mov_u32_imm(0);
                let smem_off_1 = ctx.mov_u32_imm(4);

                // Main loop
                let pos = chunk_start;
                ctx.label("fd2w_loop");
                let loop_cond = ctx.setp_lt_u32(pos, chunk_end);
                ctx.branch_if_not(loop_cond, "fd2w_loop_end");

                // Load K[pos] (2 elements per thread, same split as Q)
                let k_pos_off = ctx.mul_lo_u32(pos, head_dim_u32);
                let k0_elem_off = ctx.add_u32_reg(k_pos_off, elem0);
                let k0_off_bytes = ctx.mul_wide_u32_reg(k0_elem_off, four);
                let k0_addr = ctx.add_u64(k_head_ptr, k0_off_bytes);
                let k0 = ctx.ld_global_f32_predicated(k0_addr, in_bounds0, 0.0);

                let k1_elem_off = ctx.add_u32_reg(k_pos_off, elem1);
                let k1_off_bytes = ctx.mul_wide_u32_reg(k1_elem_off, four);
                let k1_addr = ctx.add_u64(k_head_ptr, k1_off_bytes);
                let k1 = ctx.ld_global_f32_predicated(k1_addr, in_bounds1, 0.0);

                // Partial dot product within warp (2 elements per thread)
                let dot = ctx.mul_f32(q0, k0);
                ctx.fma_f32_inplace(dot, q1, k1);

                // Warp reduce within each warp
                for delta in [16, 8, 4, 2, 1] {
                    let other = ctx.shfl_down_f32(dot, delta, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(dot, other);
                }

                // Cross-warp reduction via shared memory (u32 offsets)
                // Lane 0 of each warp stores its partial to smem[warp_id*4]
                let zero_u32 = ctx.mov_u32_imm(0);
                let is_lane0 = ctx.setp_eq_u32(lane_id, zero_u32);
                ctx.branch_if_not(is_lane0, "fd2w_skip_st");
                ctx.st_shared_f32(smem_off_warp, dot);
                ctx.label("fd2w_skip_st");

                ctx.bar_sync(0);

                // All threads read both partials and combine
                let zero_warp = ctx.mov_u32_imm(0);
                let is_warp0 = ctx.setp_eq_u32(warp_id, zero_warp);
                let partial0 = ctx.ld_shared_f32(smem_off_0);
                let partial1 = ctx.ld_shared_f32(smem_off_1);
                let total_dot = ctx.add_f32(partial0, partial1);

                // Broadcast combined score: warp0 lane0 writes, all read after barrier
                ctx.branch_if_not(is_lane0, "fd2w_skip_bcast");
                ctx.branch_if_not(is_warp0, "fd2w_skip_bcast");
                ctx.st_shared_f32(smem_off_0, total_dot);
                ctx.label("fd2w_skip_bcast");
                ctx.bar_sync(1);
                let score_raw = ctx.ld_shared_f32(smem_off_0);

                // Scale score
                let score = ctx.mul_f32(score_raw, scale_reg);

                // Online softmax update (same for both warps)
                let old_max = ctx.mov_f32_imm(0.0);
                ctx.mov_f32_reg(old_max, max_score);
                ctx.max_f32_inplace(max_score, score);
                let score_minus_max = ctx.sub_f32(score, max_score);
                let score_log2 = ctx.mul_f32(score_minus_max, log2e);
                let exp_score = ctx.ex2_f32(score_log2);

                let old_minus_new = ctx.sub_f32(old_max, max_score);
                let log2_old = ctx.mul_f32(old_minus_new, log2e);
                let correction = ctx.ex2_f32(log2_old);
                ctx.mul_f32_inplace(sum_exp, correction);
                ctx.add_f32_inplace(sum_exp, exp_score);

                // Rescale and accumulate V
                ctx.mul_f32_inplace(out0, correction);
                ctx.mul_f32_inplace(out1, correction);

                let v0_addr = ctx.add_u64(v_head_ptr, k0_off_bytes);
                let v0 = ctx.ld_global_f32_predicated(v0_addr, in_bounds0, 0.0);
                ctx.fma_f32_inplace(out0, exp_score, v0);

                let v1_addr = ctx.add_u64(v_head_ptr, k1_off_bytes);
                let v1 = ctx.ld_global_f32_predicated(v1_addr, in_bounds1, 0.0);
                ctx.fma_f32_inplace(out1, exp_score, v1);

                ctx.add_u32_inplace(pos, 1);
                ctx.branch("fd2w_loop");
                ctx.label("fd2w_loop_end");

                // Store partials: each thread writes its 2 elements
                let head_dim_plus_2 = ctx.mov_u32_imm(head_dim + 2);
                let partial_stride = ctx.mul_lo_u32(max_chunks_param, head_dim_plus_2);
                let batch_partial_stride = ctx.mul_lo_u32(num_heads_u32, partial_stride);
                let batch_partial_off = ctx.mul_lo_u32(batch_idx, batch_partial_stride);
                let head_partial_off = ctx.mul_lo_u32(head_idx, partial_stride);
                let chunk_partial_off = ctx.mul_lo_u32(chunk_idx, head_dim_plus_2);
                let partial_off = ctx.add_u32_reg(batch_partial_off, head_partial_off);
                let partial_off = ctx.add_u32_reg(partial_off, chunk_partial_off);
                let partial_off_bytes = ctx.mul_wide_u32_reg(partial_off, four);
                let partial_base = ctx.add_u64(partials_ptr, partial_off_bytes);

                let out0_addr = ctx.add_u64(partial_base, q0_off_bytes);
                ctx.branch_if_not(in_bounds0, "fd2w_skip_out0");
                ctx.st_global_f32(out0_addr, out0);
                ctx.label("fd2w_skip_out0");

                let out1_addr = ctx.add_u64(partial_base, q1_off_bytes);
                ctx.branch_if_not(in_bounds1, "fd2w_skip_out1");
                ctx.st_global_f32(out1_addr, out1);
                ctx.label("fd2w_skip_out1");

                // Metadata: only lane 0 of warp 0 stores max_score and sum_exp
                let is_writer = ctx.and_pred(is_lane0, is_warp0);
                ctx.branch_if_not(is_writer, "fd2w_skip_meta");
                let max_off = ctx.mov_u32_imm(head_dim);
                let max_off_bytes = ctx.mul_wide_u32_reg(max_off, four);
                let max_addr = ctx.add_u64(partial_base, max_off_bytes);
                ctx.st_global_f32(max_addr, max_score);
                let sum_off = ctx.mov_u32_imm(head_dim + 1);
                let sum_off_bytes = ctx.mul_wide_u32_reg(sum_off, four);
                let sum_addr = ctx.add_u64(partial_base, sum_off_bytes);
                ctx.st_global_f32(sum_addr, sum_exp);
                ctx.label("fd2w_skip_meta");

                ctx.ret();

                // Empty chunk handler
                ctx.label("fd2w_empty");
                let head_dim_plus_2_e = ctx.mov_u32_imm(head_dim + 2);
                let partial_stride_e = ctx.mul_lo_u32(max_chunks_param, head_dim_plus_2_e);
                let batch_partial_stride_e = ctx.mul_lo_u32(num_heads_u32, partial_stride_e);
                let batch_partial_off_e = ctx.mul_lo_u32(batch_idx, batch_partial_stride_e);
                let head_partial_off_e = ctx.mul_lo_u32(head_idx, partial_stride_e);
                let chunk_partial_off_e = ctx.mul_lo_u32(chunk_idx, head_dim_plus_2_e);
                let partial_off_e = ctx.add_u32_reg(batch_partial_off_e, head_partial_off_e);
                let partial_off_e = ctx.add_u32_reg(partial_off_e, chunk_partial_off_e);
                let partial_off_bytes_e = ctx.mul_wide_u32_reg(partial_off_e, four);
                let partial_base_e = ctx.add_u64(partials_ptr, partial_off_bytes_e);

                let zero_u32_e = ctx.mov_u32_imm(0);
                let is_lane0_e = ctx.setp_eq_u32(lane_id, zero_u32_e);
                let is_warp0_e = ctx.setp_eq_u32(warp_id, zero_u32_e);
                let is_writer_e = ctx.and_pred(is_lane0_e, is_warp0_e);
                ctx.branch_if_not(is_writer_e, "fd2w_empty_skip");
                let neg_inf = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let max_off_e = ctx.mov_u32_imm(head_dim);
                let max_off_bytes_e = ctx.mul_wide_u32_reg(max_off_e, four);
                let max_addr_e = ctx.add_u64(partial_base_e, max_off_bytes_e);
                ctx.st_global_f32(max_addr_e, neg_inf);
                let zero_f32 = ctx.mov_f32_imm(0.0);
                let sum_off_e = ctx.mov_u32_imm(head_dim + 1);
                let sum_off_bytes_e = ctx.mul_wide_u32_reg(sum_off_e, four);
                let sum_addr_e = ctx.add_u64(partial_base_e, sum_off_bytes_e);
                ctx.st_global_f32(sum_addr_e, zero_f32);
                ctx.label("fd2w_empty_skip");
                ctx.ret();
            })
    }
}
