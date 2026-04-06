//! Batched Incremental Attention Kernel (PAR-118)

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// PAR-118: Batched Incremental Attention for M sequences in parallel
///
/// Processes M independent sequences in a single kernel launch, reducing
/// kernel launch overhead from 3M to 3 per layer (batched KV scatter + batched attention).
///
/// Grid: (num_heads, batch_size, 1)
/// Block: (32, 1, 1) - one warp per head
///
/// Memory layout:
/// - q: [M, num_heads, head_dim] - contiguous query vectors
/// - k_ptrs: [M] - array of M pointers to K caches
/// - v_ptrs: [M] - array of M pointers to V caches
/// - output: [M, num_heads, head_dim] - contiguous output
/// - seq_lens: [M] - array of M sequence lengths (indirect mode)
#[derive(Debug, Clone)]
pub struct BatchedIncrementalAttentionKernel {
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
    /// Scaling factor for attention scores
    pub scale: f32,
}

impl BatchedIncrementalAttentionKernel {
    /// Create a new batched incremental attention kernel
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
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

impl Kernel for BatchedIncrementalAttentionKernel {
    fn name(&self) -> &str {
        "batched_incremental_attention"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let _batch_size = self.batch_size;

        // Grid: (num_heads, batch_size, 1)
        // Block: (32, 1, 1) - one warp per block
        //
        // Each block handles one (head, batch) pair
        // batch_idx = blockIdx.y selects which sequence
        // head_idx = blockIdx.x selects which Q head

        PtxKernel::new("batched_incremental_attention")
            .param(PtxType::U64, "q_ptr") // [M, num_heads, head_dim]
            .param(PtxType::U64, "k_ptrs_ptr") // [M] array of K cache pointers
            .param(PtxType::U64, "v_ptrs_ptr") // [M] array of V cache pointers
            .param(PtxType::U64, "out_ptr") // [M, num_heads, head_dim]
            .param(PtxType::U64, "seq_lens_ptr") // [M] array of sequence lengths
            .shared_memory(0)
            .build(move |ctx| {
                // Get indices
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptrs_ptr = ctx.load_param_u64("k_ptrs_ptr");
                let v_ptrs_ptr = ctx.load_param_u64("v_ptrs_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let seq_lens_ptr = ctx.load_param_u64("seq_lens_ptr");

                // Load seq_len for this batch element
                let four = ctx.mov_u32_imm(4);
                let eight = ctx.mov_u32_imm(8);
                let batch_idx_bytes = ctx.mul_wide_u32_reg(batch_idx, four);
                let seq_len_addr = ctx.add_u64(seq_lens_ptr, batch_idx_bytes);
                let seq_len = ctx.ld_global_u32(seq_len_addr);

                // Load K and V cache pointers for this batch element
                let batch_ptr_off = ctx.mul_wide_u32_reg(batch_idx, eight);
                let k_ptr_addr = ctx.add_u64(k_ptrs_ptr, batch_ptr_off);
                let v_ptr_addr = ctx.add_u64(v_ptrs_ptr, batch_ptr_off);
                let k_cache_ptr = ctx.ld_global_u64(k_ptr_addr);
                let v_cache_ptr = ctx.ld_global_u64(v_ptr_addr);

                // Compute Q/output offset: batch_idx * num_heads * head_dim + head_idx * head_dim
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let num_heads_u32 = ctx.mov_u32_imm(num_heads);
                let batch_head_stride = ctx.mul_lo_u32(num_heads_u32, head_dim_u32);
                let batch_off = ctx.mul_lo_u32(batch_idx, batch_head_stride);
                let head_off = ctx.mul_lo_u32(head_idx, head_dim_u32);
                let q_head_off = ctx.add_u32_reg(batch_off, head_off);
                let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
                let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);
                let out_head_ptr = ctx.add_u64(out_ptr, q_head_off_bytes);

                // GQA: Compute KV head index
                let kv_head_idx = ctx.mul_u32(head_idx, num_kv_heads);
                let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);

                // K/V: kv_head_idx * max_seq_len * head_dim
                let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
                let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
                let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
                let k_head_ptr = ctx.add_u64(k_cache_ptr, kv_head_off_bytes);
                let v_head_ptr = ctx.add_u64(v_cache_ptr, kv_head_off_bytes);

                // Load Q values (same as IncrementalAttentionKernel)
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

                // Online softmax state
                let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let sum_exp = ctx.mov_f32_imm(0.0);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scale_reg = ctx.mov_f32_imm(scale);

                // Loop over sequence positions
                let pos = ctx.mov_u32_imm(0);
                ctx.label("batched_seq_loop");
                let loop_cond = ctx.setp_lt_u32(pos, seq_len);
                ctx.branch_if_not(loop_cond, "batched_seq_loop_end");

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

                // Warp reduce - use full warp mask for all 32 threads
                for delta in [16, 8, 4, 2, 1] {
                    let other = ctx.shfl_down_f32(dot, delta, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(dot, other);
                }

                // PAR-118-FIX: Broadcast reduced dot product from lane 0 to all lanes.
                // After shfl_down reduction, only lane 0 has the correct sum.
                // All lanes need the score for softmax and V accumulation.
                let dot = ctx.shfl_idx_f32(dot, 0, 0xFFFF_FFFF);

                // Scale score
                let score = ctx.mul_f32(dot, scale_reg);

                // Online softmax update
                let old_max = max_score;
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

                // Load V[pos] and accumulate
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
                ctx.branch("batched_seq_loop");

                ctx.label("batched_seq_loop_end");

                // Normalize output
                let one = ctx.mov_f32_imm(1.0);
                let inv_sum = ctx.div_f32(one, sum_exp);
                ctx.mul_f32_inplace(out0, inv_sum);
                ctx.mul_f32_inplace(out1, inv_sum);
                ctx.mul_f32_inplace(out2, inv_sum);
                ctx.mul_f32_inplace(out3, inv_sum);

                // Store output
                let out0_addr = ctx.add_u64(out_head_ptr, q0_off_bytes);
                ctx.branch_if_not(in_bounds0, "batched_skip_store0");
                ctx.st_global_f32(out0_addr, out0);
                ctx.label("batched_skip_store0");

                let out1_addr = ctx.add_u64(out_head_ptr, q1_off_bytes);
                ctx.branch_if_not(in_bounds1, "batched_skip_store1");
                ctx.st_global_f32(out1_addr, out1);
                ctx.label("batched_skip_store1");

                let out2_addr = ctx.add_u64(out_head_ptr, q2_off_bytes);
                ctx.branch_if_not(in_bounds2, "batched_skip_store2");
                ctx.st_global_f32(out2_addr, out2);
                ctx.label("batched_skip_store2");

                let out3_addr = ctx.add_u64(out_head_ptr, q3_off_bytes);
                ctx.branch_if_not(in_bounds3, "batched_skip_store3");
                ctx.st_global_f32(out3_addr, out3);
                ctx.label("batched_skip_store3");

                ctx.ret();
            })
    }
}
