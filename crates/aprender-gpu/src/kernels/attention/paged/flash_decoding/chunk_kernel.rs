//! Flash Decoding Chunk Kernel - processes individual KV cache chunks in parallel.

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

use super::FLASH_DECODE_CHUNK_SIZE;

/// PAR-118: Flash Decoding kernel for split-K attention
///
/// Splits the KV cache into chunks and processes them in parallel.
/// Requires a separate reduction kernel to combine partial results.
///
/// Memory layout:
/// - q: [M, num_heads, head_dim] - contiguous query vectors
/// - k_ptrs: [M] - array of M pointers to K caches
/// - v_ptrs: [M] - array of M pointers to V caches
/// - partials: [M, num_heads, num_chunks, head_dim + 2] - partial results
///   - [0..head_dim]: weighted output (sum of exp_score * V)
///   - [head_dim]: max_score for this chunk
///   - [head_dim + 1]: sum_exp for this chunk
/// - seq_lens: [M] - array of M sequence lengths
#[derive(Debug, Clone)]
pub struct FlashDecodingChunkKernel {
    /// Maximum sequence length to support
    pub max_seq_len: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Chunk size for split-K
    pub chunk_size: u32,
    /// Scaling factor for attention scores
    pub scale: f32,
}

impl FlashDecodingChunkKernel {
    /// Create a new Flash Decoding chunk kernel
    #[must_use]
    pub fn new(
        max_seq_len: u32,
        head_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
        batch_size: u32,
    ) -> Self {
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

    /// Get the number of chunks for a given sequence length
    #[must_use]
    pub fn num_chunks(&self, seq_len: u32) -> u32 {
        (seq_len + self.chunk_size - 1) / self.chunk_size
    }

    /// Get the size of the partials buffer per (head, batch) pair
    /// Layout: [num_chunks, head_dim + 2]
    #[must_use]
    pub fn partials_size_per_head(&self, max_chunks: u32) -> u32 {
        max_chunks * (self.head_dim + 2)
    }
}

impl Kernel for FlashDecodingChunkKernel {
    fn name(&self) -> &str {
        "flash_decoding_chunk"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let chunk_size = self.chunk_size;
        let _batch_size = self.batch_size;

        // Grid: (num_heads, batch_size, num_chunks)
        // Block: (32, 1, 1) - one warp per block
        //
        // Each block handles one (head, batch, chunk) triple
        // chunk_idx = blockIdx.z selects which chunk of the sequence
        // batch_idx = blockIdx.y selects which sequence
        // head_idx = blockIdx.x selects which Q head

        PtxKernel::new("flash_decoding_chunk")
            .param(PtxType::U64, "q_ptr") // [M, num_heads, head_dim]
            .param(PtxType::U64, "k_ptrs_ptr") // [M] array of K cache pointers
            .param(PtxType::U64, "v_ptrs_ptr") // [M] array of V cache pointers
            .param(PtxType::U64, "partials_ptr") // [M, num_heads, num_chunks, head_dim + 2]
            .param(PtxType::U64, "seq_lens_ptr") // [M] array of sequence lengths
            .param(PtxType::U32, "max_chunks") // Maximum number of chunks
            .shared_memory(0)
            .build(move |ctx| {
                // Get indices
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);
                let chunk_idx = ctx.special_reg(PtxReg::CtaIdZ);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                // Load parameters
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

                // Compute chunk boundaries
                let chunk_size_u32 = ctx.mov_u32_imm(chunk_size);
                let chunk_start = ctx.mul_lo_u32(chunk_idx, chunk_size_u32);
                let chunk_end_raw = ctx.add_u32(chunk_start, chunk_size); // Use literal for add_u32
                                                                          // Clamp chunk_end to seq_len
                let chunk_end = ctx.min_u32(chunk_end_raw, seq_len); // Both are VirtualReg

                // Early exit if chunk_start >= seq_len (this chunk has no work)
                let has_work = ctx.setp_lt_u32(chunk_start, seq_len);
                ctx.branch_if_not(has_work, "flash_decode_chunk_empty");

                // Load K and V cache pointers for this batch element
                let batch_ptr_off = ctx.mul_wide_u32_reg(batch_idx, eight);
                let k_ptr_addr = ctx.add_u64(k_ptrs_ptr, batch_ptr_off);
                let v_ptr_addr = ctx.add_u64(v_ptrs_ptr, batch_ptr_off);
                let k_cache_ptr = ctx.ld_global_u64(k_ptr_addr);
                let v_cache_ptr = ctx.ld_global_u64(v_ptr_addr);

                // Compute Q offset: batch_idx * num_heads * head_dim + head_idx * head_dim
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let num_heads_u32 = ctx.mov_u32_imm(num_heads);
                let batch_head_stride = ctx.mul_lo_u32(num_heads_u32, head_dim_u32);
                let batch_off = ctx.mul_lo_u32(batch_idx, batch_head_stride);
                let head_off = ctx.mul_lo_u32(head_idx, head_dim_u32);
                let q_head_off = ctx.add_u32_reg(batch_off, head_off);
                let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
                let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);

                // GQA: Compute KV head index
                let kv_head_idx = ctx.mul_u32(head_idx, num_kv_heads);
                let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);

                // K/V: kv_head_idx * max_seq_len * head_dim
                let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
                let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
                let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
                let k_head_ptr = ctx.add_u64(k_cache_ptr, kv_head_off_bytes);
                let v_head_ptr = ctx.add_u64(v_cache_ptr, kv_head_off_bytes);

                // Load Q values (head_dim up to 128, 4 elements per lane)
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

                // Initialize accumulators for this chunk
                let out0 = ctx.mov_f32_imm(0.0);
                let out1 = ctx.mov_f32_imm(0.0);
                let out2 = ctx.mov_f32_imm(0.0);
                let out3 = ctx.mov_f32_imm(0.0);

                // Online softmax state for this chunk
                let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let sum_exp = ctx.mov_f32_imm(0.0);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scale_reg = ctx.mov_f32_imm(scale);

                // Loop over positions in this chunk [chunk_start, chunk_end)
                let pos = chunk_start;
                ctx.label("flash_decode_chunk_loop");
                let loop_cond = ctx.setp_lt_u32(pos, chunk_end);
                ctx.branch_if_not(loop_cond, "flash_decode_chunk_loop_end");

                // Load K[pos] and compute Q·K dot product
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
                ctx.fma_f32_inplace(dot, q1, k1);
                ctx.fma_f32_inplace(dot, q2, k2);
                ctx.fma_f32_inplace(dot, q3, k3);

                // Warp reduce
                for delta in [16, 8, 4, 2, 1] {
                    let other = ctx.shfl_down_f32(dot, delta, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(dot, other);
                }

                // PAR-118-FIX: Broadcast reduced dot product from lane 0 to all lanes.
                // After shfl_down reduction, only lane 0 has the correct sum.
                // All lanes need the score for softmax and V accumulation.
                // Without this broadcast, lanes 1-31 compute exp(wrong_partial_dot)
                // and weight V values incorrectly -> garbage output.
                let dot = ctx.shfl_idx_f32(dot, 0, 0xFFFF_FFFF);

                // Scale score
                let score = ctx.mul_f32(dot, scale_reg);

                // Online softmax update
                // CRITICAL: Copy max_score to a NEW register before in-place update.
                // `let old_max = max_score` would alias the same VirtualReg, so after
                // max_f32_inplace clobbers max_score, old_max would also be clobbered,
                // making correction = exp2(0) = 1.0 always (no rescaling).
                let old_max = ctx.mov_f32_imm(0.0);
                ctx.mov_f32_reg(old_max, max_score);
                ctx.max_f32_inplace(max_score, score);
                let score_minus_max = ctx.sub_f32(score, max_score);
                let score_log2 = ctx.mul_f32(score_minus_max, log2e);
                let exp_score = ctx.ex2_f32(score_log2);

                // Rescale sum_exp if max changed
                let old_minus_new = ctx.sub_f32(old_max, max_score);
                let log2_old = ctx.mul_f32(old_minus_new, log2e);
                let correction = ctx.ex2_f32(log2_old);
                ctx.mul_f32_inplace(sum_exp, correction);
                ctx.add_f32_inplace(sum_exp, exp_score);

                // Rescale existing output
                ctx.mul_f32_inplace(out0, correction);
                ctx.mul_f32_inplace(out1, correction);
                ctx.mul_f32_inplace(out2, correction);
                ctx.mul_f32_inplace(out3, correction);

                // Load V[pos] and accumulate (NOT normalized yet)
                let v0_addr = ctx.add_u64(v_head_ptr, k0_off_bytes);
                let v0 = ctx.ld_global_f32_predicated(v0_addr, in_bounds0, 0.0);
                ctx.fma_f32_inplace(out0, exp_score, v0);

                let v1_addr = ctx.add_u64(v_head_ptr, k1_off_bytes);
                let v1 = ctx.ld_global_f32_predicated(v1_addr, in_bounds1, 0.0);
                ctx.fma_f32_inplace(out1, exp_score, v1);

                let v2_addr = ctx.add_u64(v_head_ptr, k2_off_bytes);
                let v2 = ctx.ld_global_f32_predicated(v2_addr, in_bounds2, 0.0);
                ctx.fma_f32_inplace(out2, exp_score, v2);

                let v3_addr = ctx.add_u64(v_head_ptr, k3_off_bytes);
                let v3 = ctx.ld_global_f32_predicated(v3_addr, in_bounds3, 0.0);
                ctx.fma_f32_inplace(out3, exp_score, v3);

                ctx.add_u32_inplace(pos, 1);
                ctx.branch("flash_decode_chunk_loop");

                ctx.label("flash_decode_chunk_loop_end");

                // Compute partials offset:
                // partials_ptr + (batch_idx * num_heads * max_chunks + head_idx * max_chunks + chunk_idx) * (head_dim + 2) * 4
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

                // Store weighted output (out0..out3)
                let out0_addr = ctx.add_u64(partial_base, q0_off_bytes);
                ctx.branch_if_not(in_bounds0, "flash_decode_skip_out0");
                ctx.st_global_f32(out0_addr, out0);
                ctx.label("flash_decode_skip_out0");

                let out1_addr = ctx.add_u64(partial_base, q1_off_bytes);
                ctx.branch_if_not(in_bounds1, "flash_decode_skip_out1");
                ctx.st_global_f32(out1_addr, out1);
                ctx.label("flash_decode_skip_out1");

                let out2_addr = ctx.add_u64(partial_base, q2_off_bytes);
                ctx.branch_if_not(in_bounds2, "flash_decode_skip_out2");
                ctx.st_global_f32(out2_addr, out2);
                ctx.label("flash_decode_skip_out2");

                let out3_addr = ctx.add_u64(partial_base, q3_off_bytes);
                ctx.branch_if_not(in_bounds3, "flash_decode_skip_out3");
                ctx.st_global_f32(out3_addr, out3);
                ctx.label("flash_decode_skip_out3");

                // Store max_score at offset head_dim (only lane 0)
                let zero_u32 = ctx.mov_u32_imm(0);
                let is_lane0 = ctx.setp_eq_u32(lane_id, zero_u32);
                ctx.branch_if_not(is_lane0, "flash_decode_skip_meta");
                let max_off = ctx.mov_u32_imm(head_dim);
                let max_off_bytes = ctx.mul_wide_u32_reg(max_off, four);
                let max_addr = ctx.add_u64(partial_base, max_off_bytes);
                ctx.st_global_f32(max_addr, max_score);

                // Store sum_exp at offset head_dim + 1
                let sum_off = ctx.mov_u32_imm(head_dim + 1);
                let sum_off_bytes = ctx.mul_wide_u32_reg(sum_off, four);
                let sum_addr = ctx.add_u64(partial_base, sum_off_bytes);
                ctx.st_global_f32(sum_addr, sum_exp);
                ctx.label("flash_decode_skip_meta");

                ctx.ret();

                // Empty chunk handler - store sentinel values
                ctx.label("flash_decode_chunk_empty");
                // Same partial offset calculation
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

                // Store -inf for max_score (sentinel for empty chunk)
                let zero_u32_e = ctx.mov_u32_imm(0);
                let is_lane0_e = ctx.setp_eq_u32(lane_id, zero_u32_e);
                ctx.branch_if_not(is_lane0_e, "flash_decode_empty_done");
                let neg_inf = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let max_off_e = ctx.mov_u32_imm(head_dim);
                let max_off_bytes_e = ctx.mul_wide_u32_reg(max_off_e, four);
                let max_addr_e = ctx.add_u64(partial_base_e, max_off_bytes_e);
                ctx.st_global_f32(max_addr_e, neg_inf);

                // Store 0 for sum_exp
                let zero = ctx.mov_f32_imm(0.0);
                let sum_off_e = ctx.mov_u32_imm(head_dim + 1);
                let sum_off_bytes_e = ctx.mul_wide_u32_reg(sum_off_e, four);
                let sum_addr_e = ctx.add_u64(partial_base_e, sum_off_bytes_e);
                ctx.st_global_f32(sum_addr_e, zero);
                ctx.label("flash_decode_empty_done");

                ctx.ret();
            })
    }
}
