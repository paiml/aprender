//! Incremental Attention Kernel for M=1 Autoregressive Decoding (PAR-020)

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

// =============================================================================
// PAR-020: Incremental Attention Kernel for M=1 Autoregressive Decoding
// =============================================================================

/// Incremental attention kernel for single-query autoregressive decoding (PAR-020)
///
/// Optimized for the critical path of LLM token generation where each new token
/// requires attention over the entire KV cache with a single query vector.
///
/// # Memory Layout
///
/// - Q: [head_dim] - single query vector for current position
/// - K: [seq_len, head_dim] - cached keys (GPU-resident)
/// - V: [seq_len, head_dim] - cached values (GPU-resident)
/// - Output: [head_dim] - weighted sum of values
///
/// # Algorithm
///
/// 1. Compute attention scores: score[i] = dot(Q, K[i]) * scale
/// 2. Apply causal mask (positions > current are masked)
/// 3. Online softmax: max_score, sum_exp tracked incrementally
/// 4. Compute weighted V sum: output = sum(softmax[i] * V[i])
///
/// # Performance
///
/// - Avoids materializing [seq_len, seq_len] attention matrix
/// - Uses warp shuffle for efficient parallel reduction
/// - Designed for GPU-resident KV cache (no D2H transfer)
/// - Target: O(seq_len * head_dim) memory, O(seq_len * head_dim) compute
#[derive(Debug, Clone)]
pub struct IncrementalAttentionKernel {
    /// Maximum sequence length to support
    pub max_seq_len: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Number of key-value heads (for GQA, <= num_heads)
    pub num_kv_heads: u32,
    /// Scaling factor for attention scores (1/sqrt(head_dim))
    pub scale: f32,
    /// PAR-061: Read seq_len from device memory (for CUDA graph compatibility)
    pub indirect_seq_len: bool,
}

impl IncrementalAttentionKernel {
    /// Create new incremental attention kernel (MHA - num_kv_heads = num_heads)
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `head_dim` - Dimension per attention head
    /// * `num_heads` - Number of attention heads
    #[must_use]
    pub fn new(max_seq_len: u32, head_dim: u32, num_heads: u32) -> Self {
        Self::with_gqa(max_seq_len, head_dim, num_heads, num_heads)
    }

    /// Create new incremental attention kernel with GQA support (PAR-021)
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `head_dim` - Dimension per attention head
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    #[must_use]
    pub fn with_gqa(max_seq_len: u32, head_dim: u32, num_heads: u32, num_kv_heads: u32) -> Self {
        Self {
            max_seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            scale: 1.0 / (head_dim as f32).sqrt(),
            indirect_seq_len: false,
        }
    }

    /// PAR-061: Enable indirect seq_len mode (reads from device memory)
    /// Required for CUDA graph compatibility
    #[must_use]
    pub fn with_indirect_seq_len(mut self, indirect: bool) -> Self {
        self.indirect_seq_len = indirect;
        self
    }

    /// Check if this kernel is configured for GQA
    #[must_use]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads != self.num_heads
    }
}

impl Kernel for IncrementalAttentionKernel {
    fn name(&self) -> &str {
        // PAR-061: Different kernel name for indirect mode
        if self.indirect_seq_len {
            "incremental_attention_indirect"
        } else {
            "incremental_attention"
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let indirect = self.indirect_seq_len;

        // Kernel strategy (PAR-020 + PAR-021 GQA):
        // - Grid: (num_heads, 1, 1) - one block per Q head
        // - Block: (32, 1, 1) - one warp per block
        // - Each warp computes attention for one Q head using online softmax
        //
        // Memory layout:
        // - q: [num_heads, head_dim] - query vectors for current position
        // - k: [num_kv_heads, max_seq_len, head_dim] - key cache (GPU-resident)
        // - v: [num_kv_heads, max_seq_len, head_dim] - value cache (GPU-resident)
        // - output: [num_heads, head_dim] - attention output
        //
        // GQA mapping (PAR-021):
        // - Each Q head uses kv_head_idx = q_head_idx * num_kv_heads / num_heads
        // - For MHA: kv_head_idx = q_head_idx
        // - For GQA: multiple Q heads share the same KV head
        //
        // Algorithm:
        // 1. Thread i loads Q[lane_id], Q[lane_id+32], ... (strided)
        // 2. Loop over seq positions, computing Q·K dot product per position
        // 3. Warp-reduce dot product using shfl_down
        // 4. Online softmax: track running max and sum_exp
        // 5. Accumulate weighted V vectors
        // 6. Normalize and store output

        // PAR-061: Use different kernel name and parameter type for indirect mode
        let kernel_name =
            if indirect { "incremental_attention_indirect" } else { "incremental_attention" };

        let mut builder = PtxKernel::new(kernel_name)
            .param(PtxType::U64, "q_ptr")
            .param(PtxType::U64, "k_ptr")
            .param(PtxType::U64, "v_ptr")
            .param(PtxType::U64, "out_ptr");

        // PAR-061: Indirect mode takes seq_len_ptr (U64), direct mode takes seq_len (U32)
        builder = if indirect {
            builder.param(PtxType::U64, "seq_len_ptr")
        } else {
            builder.param(PtxType::U32, "seq_len")
        };

        builder
            .shared_memory(0) // Register-only, warp shuffle for reduction
            .build(move |ctx| {
                // Get indices
                let q_head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                // PAR-061: In indirect mode, load seq_len from device memory
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

                // Pre-compute constants
                let four = ctx.mov_u32_imm(4);
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);

                // Compute Q/output head offset
                // Q/output: q_head_idx * head_dim
                let q_head_off = ctx.mul_lo_u32(q_head_idx, head_dim_u32);
                let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
                let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);
                let out_head_ptr = ctx.add_u64(out_ptr, q_head_off_bytes);

                // PAR-021 GQA: Compute KV head index
                // kv_head_idx = q_head_idx * num_kv_heads / num_heads
                // This maps multiple Q heads to the same KV head
                // Use literal values since they're known at kernel build time
                let kv_head_idx = ctx.mul_u32(q_head_idx, num_kv_heads);
                let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);

                // K/V: kv_head_idx * max_seq_len * head_dim
                let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
                let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
                let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
                let k_head_ptr = ctx.add_u64(k_ptr, kv_head_off_bytes);
                let v_head_ptr = ctx.add_u64(v_ptr, kv_head_off_bytes);

                // CORRECTNESS-002: Each thread handles 4 elements (strided by 32) for head_dim up to 128
                // Thread 0 handles [0,32,64,96], thread 1 handles [1,33,65,97], etc.
                // Supports head_dim: 32, 64, 96, 128

                // Load Q values into registers (persistent across seq loop)
                // Using predicated loads for bounds checking
                // CORRECTNESS-002: Support head_dim up to 128 (4 elements per thread)
                let q0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
                let q0_addr = ctx.add_u64(q_head_ptr, q0_off_bytes);
                let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
                let q0 = ctx.ld_global_f32_predicated(q0_addr, in_bounds0, 0.0);

                // Second element (if head_dim > 32)
                let lane_plus_32 = ctx.add_u32(lane_id, 32);
                let q1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
                let q1_addr = ctx.add_u64(q_head_ptr, q1_off_bytes);
                let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
                let q1 = ctx.ld_global_f32_predicated(q1_addr, in_bounds1, 0.0);

                // CORRECTNESS-002: Third element (if head_dim > 64)
                let lane_plus_64 = ctx.add_u32(lane_id, 64);
                let q2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
                let q2_addr = ctx.add_u64(q_head_ptr, q2_off_bytes);
                let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
                let q2 = ctx.ld_global_f32_predicated(q2_addr, in_bounds2, 0.0);

                // CORRECTNESS-002: Fourth element (if head_dim > 96)
                let lane_plus_96 = ctx.add_u32(lane_id, 96);
                let q3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
                let q3_addr = ctx.add_u64(q_head_ptr, q3_off_bytes);
                let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);
                let q3 = ctx.ld_global_f32_predicated(q3_addr, in_bounds3, 0.0);

                // Initialize output accumulators
                let out0 = ctx.mov_f32_imm(0.0);
                let out1 = ctx.mov_f32_imm(0.0);
                // CORRECTNESS-002: Additional accumulators for head_dim > 64
                let out2 = ctx.mov_f32_imm(0.0);
                let out3 = ctx.mov_f32_imm(0.0);

                // Online softmax state
                let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let sum_exp = ctx.mov_f32_imm(0.0);

                // Log2(e) for exp approximation via ex2
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scale_reg = ctx.mov_f32_imm(scale);

                // Loop counter
                let pos = ctx.mov_u32_imm(0);

                ctx.label("seq_loop");

                // Check loop condition
                let loop_cond = ctx.setp_lt_u32(pos, seq_len);
                ctx.branch_if_not(loop_cond, "seq_loop_end");

                // Compute K offset for this position: pos * head_dim
                let k_pos_off = ctx.mul_lo_u32(pos, head_dim_u32);

                // Load K[pos, lane_id] and K[pos, lane_id+32]
                let k0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
                let k0_off_bytes = ctx.mul_wide_u32_reg(k0_elem_off, four);
                let k0_addr = ctx.add_u64(k_head_ptr, k0_off_bytes);
                let k0 = ctx.ld_global_f32_predicated(k0_addr, in_bounds0, 0.0);

                let k1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
                let k1_off_bytes = ctx.mul_wide_u32_reg(k1_elem_off, four);
                let k1_addr = ctx.add_u64(k_head_ptr, k1_off_bytes);
                let k1 = ctx.ld_global_f32_predicated(k1_addr, in_bounds1, 0.0);

                // CORRECTNESS-002: Load K[pos, lane_id+64] and K[pos, lane_id+96]
                let k2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
                let k2_off_bytes = ctx.mul_wide_u32_reg(k2_elem_off, four);
                let k2_addr = ctx.add_u64(k_head_ptr, k2_off_bytes);
                let k2 = ctx.ld_global_f32_predicated(k2_addr, in_bounds2, 0.0);

                let k3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
                let k3_off_bytes = ctx.mul_wide_u32_reg(k3_elem_off, four);
                let k3_addr = ctx.add_u64(k_head_ptr, k3_off_bytes);
                let k3 = ctx.ld_global_f32_predicated(k3_addr, in_bounds3, 0.0);

                // Compute partial dot product: q0*k0 + q1*k1 + q2*k2 + q3*k3
                // CORRECTNESS-002: Now handles full head_dim=128
                let dot_partial = ctx.mul_f32(q0, k0);
                let dot_partial = ctx.fma_f32(q1, k1, dot_partial);
                let dot_partial = ctx.fma_f32(q2, k2, dot_partial);
                let dot_partial = ctx.fma_f32(q3, k3, dot_partial);

                // Warp-reduce the dot product using shfl.down
                // sum += shfl_down(sum, 16)
                // sum += shfl_down(sum, 8)
                // sum += shfl_down(sum, 4)
                // sum += shfl_down(sum, 2)
                // sum += shfl_down(sum, 1)
                let dot16 = ctx.shfl_down_f32(dot_partial, 16, 0xFFFF_FFFF);
                let dot_partial = ctx.add_f32(dot_partial, dot16);
                let dot8 = ctx.shfl_down_f32(dot_partial, 8, 0xFFFF_FFFF);
                let dot_partial = ctx.add_f32(dot_partial, dot8);
                let dot4 = ctx.shfl_down_f32(dot_partial, 4, 0xFFFF_FFFF);
                let dot_partial = ctx.add_f32(dot_partial, dot4);
                let dot2 = ctx.shfl_down_f32(dot_partial, 2, 0xFFFF_FFFF);
                let dot_partial = ctx.add_f32(dot_partial, dot2);
                let dot1 = ctx.shfl_down_f32(dot_partial, 1, 0xFFFF_FFFF);
                let dot_reduced = ctx.add_f32(dot_partial, dot1);

                // Broadcast result to all threads via shfl.idx lane 0
                let dot_broadcast = ctx.shfl_idx_f32(dot_reduced, 0, 0xFFFF_FFFF);

                // Scale the attention score
                let score = ctx.mul_f32(dot_broadcast, scale_reg);

                // Online softmax update (Milakov & Gimelshein 2018):
                // new_max = max(old_max, score)
                // correction = exp(old_max - new_max)
                // sum_exp = sum_exp * correction + exp(score - new_max)
                // output = output * correction + exp(score - new_max) * V

                let new_max = ctx.max_f32(max_score, score);

                // exp(old_max - new_max) using 2^(x * log2(e))
                let max_diff = ctx.sub_f32(max_score, new_max);
                let max_diff_scaled = ctx.mul_f32(max_diff, log2e);
                let correction = ctx.ex2_f32(max_diff_scaled);

                // exp(score - new_max)
                let score_diff = ctx.sub_f32(score, new_max);
                let score_diff_scaled = ctx.mul_f32(score_diff, log2e);
                let exp_score = ctx.ex2_f32(score_diff_scaled);

                // Load V[pos, lane_id] and V[pos, lane_id+32]
                let v0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
                let v0_off_bytes = ctx.mul_wide_u32_reg(v0_elem_off, four);
                let v0_addr = ctx.add_u64(v_head_ptr, v0_off_bytes);
                let v0 = ctx.ld_global_f32_predicated(v0_addr, in_bounds0, 0.0);

                let v1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
                let v1_off_bytes = ctx.mul_wide_u32_reg(v1_elem_off, four);
                let v1_addr = ctx.add_u64(v_head_ptr, v1_off_bytes);
                let v1 = ctx.ld_global_f32_predicated(v1_addr, in_bounds1, 0.0);

                // CORRECTNESS-002: Load V[pos, lane_id+64] and V[pos, lane_id+96]
                let v2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
                let v2_off_bytes = ctx.mul_wide_u32_reg(v2_elem_off, four);
                let v2_addr = ctx.add_u64(v_head_ptr, v2_off_bytes);
                let v2 = ctx.ld_global_f32_predicated(v2_addr, in_bounds2, 0.0);

                let v3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
                let v3_off_bytes = ctx.mul_wide_u32_reg(v3_elem_off, four);
                let v3_addr = ctx.add_u64(v_head_ptr, v3_off_bytes);
                let v3 = ctx.ld_global_f32_predicated(v3_addr, in_bounds3, 0.0);

                // Update loop state using in-place operations
                // Online softmax: max_score = max(max_score, score)
                ctx.max_f32_inplace(max_score, score);

                // sum_exp = sum_exp * correction + exp_score
                ctx.mul_f32_inplace(sum_exp, correction);
                ctx.add_f32_inplace(sum_exp, exp_score);

                // out = out * correction + exp_score * V
                ctx.mul_f32_inplace(out0, correction);
                ctx.fma_f32_inplace(out0, exp_score, v0);
                ctx.mul_f32_inplace(out1, correction);
                ctx.fma_f32_inplace(out1, exp_score, v1);
                // CORRECTNESS-002: Update out2 and out3
                ctx.mul_f32_inplace(out2, correction);
                ctx.fma_f32_inplace(out2, exp_score, v2);
                ctx.mul_f32_inplace(out3, correction);
                ctx.fma_f32_inplace(out3, exp_score, v3);

                // Increment position
                ctx.add_u32_inplace(pos, 1);
                ctx.branch("seq_loop");

                ctx.label("seq_loop_end");

                // Normalize output: out /= sum_exp
                // Use reciprocal approximation for speed
                let one = ctx.mov_f32_imm(1.0);
                let inv_sum = ctx.div_f32(one, sum_exp);

                ctx.mul_f32_inplace(out0, inv_sum);
                ctx.mul_f32_inplace(out1, inv_sum);
                // CORRECTNESS-002: Normalize out2 and out3
                ctx.mul_f32_inplace(out2, inv_sum);
                ctx.mul_f32_inplace(out3, inv_sum);

                // Store output (only for valid indices)
                // Thread writes to output[head_idx, lane_id]
                let out0_addr = ctx.add_u64(out_head_ptr, q0_off_bytes);
                ctx.branch_if_not(in_bounds0, "skip_store0");
                ctx.st_global_f32(out0_addr, out0);
                ctx.label("skip_store0");

                let out1_addr = ctx.add_u64(out_head_ptr, q1_off_bytes);
                ctx.branch_if_not(in_bounds1, "skip_store1");
                ctx.st_global_f32(out1_addr, out1);
                ctx.label("skip_store1");

                // CORRECTNESS-002: Store out2 and out3
                let out2_addr = ctx.add_u64(out_head_ptr, q2_off_bytes);
                ctx.branch_if_not(in_bounds2, "skip_store2");
                ctx.st_global_f32(out2_addr, out2);
                ctx.label("skip_store2");

                let out3_addr = ctx.add_u64(out_head_ptr, q3_off_bytes);
                ctx.branch_if_not(in_bounds3, "skip_store3");
                ctx.st_global_f32(out3_addr, out3);
                ctx.label("skip_store3");

                ctx.ret();
            })
    }
}
