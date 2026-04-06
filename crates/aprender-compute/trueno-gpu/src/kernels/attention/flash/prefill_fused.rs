//! PMAT-069: Fused prefill attention kernel for inference.
//!
//! Replaces 5 cuBLAS + PTX launches per layer (2 QK^T + 1 softmax + 2 Attn×V)
//! with a single fused kernel launch per layer. For 28-layer Qwen 2.5:
//! 140 launches → 28 launches. Saves ~7.5ms TTFT on RTX 4060 Laptop.
//!
//! Features:
//! - GQA support (num_q_heads ≠ num_kv_heads) via heads_per_kv parameter
//! - Packed QKV layout (q_stride, kv_stride) — zero-copy from projection output
//! - Online softmax (no N×N materialization, O(M) registers)
//! - Causal masking (autoregressive inference)
//!
//! Grid: (num_q_heads, 1, 1) — one block per Q head
//! Block: (32, 1, 1) — one warp per block
//! Shared memory: 0 bytes (all in registers, K/V via L2 cache)
//!
//! Algorithm (per block, per Q head):
//! 1. Map Q head → KV head: kv_head = head_idx / heads_per_kv
//! 2. For each Q row i = 0..M-1:
//!    a. Load Q[i, lane*EPL..lane*EPL+EPL-1]
//!    b. For j = 0..i (causal): dot(Q[i], K[j]) via warp reduce → score s
//!    c. Online softmax update: max, correction, p = exp(s - max)
//!    d. O[d] = O[d] * correction + p * V[j, d]
//! 3. Normalize O /= l_i, store to global
//!
//! EPL = elements per lane = head_dim / 32

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Fused prefill attention kernel for inference with GQA.
///
/// Processes all M tokens for one Q head per block using online softmax.
/// K/V are read from global memory with L2 cache reuse across GQA groups.
#[derive(Debug, Clone)]
pub struct PrefillAttentionKernel {
    /// Head dimension (must be multiple of 32, typically 64 or 128)
    pub head_dim: u32,
    /// Number of Q heads per KV head (for GQA). E.g., 6 for 12Q/2KV.
    pub heads_per_kv: u32,
}

impl PrefillAttentionKernel {
    /// Create a new fused prefill attention kernel.
    ///
    /// - `head_dim`: must be a multiple of 32
    /// - `heads_per_kv`: GQA ratio (num_q_heads / num_kv_heads)
    #[must_use]
    pub fn new(head_dim: u32, heads_per_kv: u32) -> Self {
        assert!(head_dim % 32 == 0, "head_dim must be multiple of 32");
        assert!(heads_per_kv > 0, "heads_per_kv must be > 0");
        Self { head_dim, heads_per_kv }
    }
}

impl Kernel for PrefillAttentionKernel {
    fn name(&self) -> &str {
        "fused_prefill_attention_causal"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let heads_per_kv = self.heads_per_kv;
        let epl = head_dim / 32; // elements per lane
        let scale = 1.0 / (head_dim as f32).sqrt();

        PtxKernel::new("fused_prefill_attention_causal")
            .param(PtxType::U64, "q_ptr") // Q[M, q_stride] FP32
            .param(PtxType::U64, "k_ptr") // K[M, kv_stride] FP32
            .param(PtxType::U64, "v_ptr") // V[M, kv_stride] FP32
            .param(PtxType::U64, "o_ptr") // O[M, q_stride] FP32
            .param(PtxType::U32, "m_param") // sequence length
            .param(PtxType::U32, "q_stride") // Q stride in elements
            .param(PtxType::U32, "kv_stride") // KV stride in elements
            .param(PtxType::U32, "num_q_heads") // for bounds check
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);

                // Load parameters
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let o_ptr = ctx.load_param_u64("o_ptr");
                let m_param = ctx.load_param_u32("m_param");
                let q_stride = ctx.load_param_u32("q_stride");
                let kv_stride = ctx.load_param_u32("kv_stride");
                let num_q_heads = ctx.load_param_u32("num_q_heads");

                // Bounds check
                let head_valid = ctx.setp_lt_u32(head_idx, num_q_heads);
                ctx.branch_if_not(head_valid, "exit");

                // GQA: kv_head = head_idx / heads_per_kv (compile-time constant)
                let kv_head = ctx.div_u32(head_idx, heads_per_kv);

                // Head offsets (in bytes)
                let head_dim_reg = ctx.mov_u32_imm(head_dim);
                let q_head_elem_off = ctx.mul_u32_reg(head_idx, head_dim_reg);
                let q_head_byte_off = ctx.mul_wide_u32(q_head_elem_off, 4);
                let kv_head_elem_off = ctx.mul_u32_reg(kv_head, head_dim_reg);
                let kv_head_byte_off = ctx.mul_wide_u32(kv_head_elem_off, 4);

                // Lane element range: [lane_base, lane_base + epl)
                let lane_base = ctx.mul_u32(tid, epl);

                // Constants
                let scale_reg = ctx.mov_f32_imm(scale);
                let log2e_reg = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let zero_f32 = ctx.mov_f32_imm(0.0);
                let neg_inf = ctx.mov_f32_imm(f32::NEG_INFINITY);

                // Pre-compute lane byte offset from head start
                let lane_byte_off = ctx.mul_wide_u32(lane_base, 4);

                // ===== Allocate per-row accumulators (reused across rows) =====
                let mut q_regs = Vec::with_capacity(epl as usize);
                let mut o_regs = Vec::with_capacity(epl as usize);
                for _ in 0..epl {
                    q_regs.push(ctx.mov_f32_imm(0.0));
                    o_regs.push(ctx.mov_f32_imm(0.0));
                }
                let m_i = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let l_i = ctx.mov_f32_imm(0.0);

                // ===== Row loop: for row_i = 0..M =====
                let row_i = ctx.mov_u32_imm(0);
                ctx.label("row_start");
                let row_done = ctx.setp_ge_u32(row_i, m_param);
                ctx.branch_if(row_done, "row_end");

                // Reset accumulators for this row
                ctx.mov_f32_reg(m_i, neg_inf);
                ctx.mov_f32_reg(l_i, zero_f32);
                for o in &o_regs {
                    ctx.mov_f32_reg(*o, zero_f32);
                }

                // Q row base address
                let q_row_off = ctx.mul_wide_u32_reg(row_i, q_stride);
                let q_row_off_bytes = ctx.mul_u64(q_row_off, 4);
                let q_row_base = ctx.add_u64(q_ptr, q_row_off_bytes);
                let q_row_base = ctx.add_u64(q_row_base, q_head_byte_off);
                let q_lane_base = ctx.add_u64(q_row_base, lane_byte_off);

                // Load Q elements for this lane
                for e in 0..epl as usize {
                    let addr = if e == 0 {
                        q_lane_base
                    } else {
                        let off = ctx.mov_u64_imm((e * 4) as u64);
                        ctx.add_u64(q_lane_base, off)
                    };
                    let val = ctx.ld_global_f32(addr);
                    ctx.mov_f32_reg(q_regs[e], val);
                }

                // Causal limit: j goes from 0 to row_i (inclusive)
                let j_limit = ctx.add_u32(row_i, 1);

                // ===== Inner loop: for j = 0..row_i (inclusive, causal) =====
                let col_j = ctx.mov_u32_imm(0);
                ctx.label("col_start");
                let col_done = ctx.setp_ge_u32(col_j, j_limit);
                ctx.branch_if(col_done, "col_end");

                // K row base address
                let k_row_off = ctx.mul_wide_u32_reg(col_j, kv_stride);
                let k_row_off_bytes = ctx.mul_u64(k_row_off, 4);
                let k_row_base = ctx.add_u64(k_ptr, k_row_off_bytes);
                let k_row_base = ctx.add_u64(k_row_base, kv_head_byte_off);
                let k_lane_base = ctx.add_u64(k_row_base, lane_byte_off);

                // Dot product: Q[i] · K[j]
                let dot_partial = ctx.mov_f32_imm(0.0);
                for e in 0..epl as usize {
                    let k_addr = if e == 0 {
                        k_lane_base
                    } else {
                        let off = ctx.mov_u64_imm((e * 4) as u64);
                        ctx.add_u64(k_lane_base, off)
                    };
                    let k_val = ctx.ld_global_f32(k_addr);
                    ctx.fma_f32_inplace(dot_partial, q_regs[e], k_val);
                }

                // Warp reduce sum (shfl_down + add, 5 steps for 32 lanes)
                let s16 = ctx.shfl_down_f32(dot_partial, 16, 31);
                ctx.add_f32_inplace(dot_partial, s16);
                let s8 = ctx.shfl_down_f32(dot_partial, 8, 31);
                ctx.add_f32_inplace(dot_partial, s8);
                let s4 = ctx.shfl_down_f32(dot_partial, 4, 31);
                ctx.add_f32_inplace(dot_partial, s4);
                let s2 = ctx.shfl_down_f32(dot_partial, 2, 31);
                ctx.add_f32_inplace(dot_partial, s2);
                let s1 = ctx.shfl_down_f32(dot_partial, 1, 31);
                ctx.add_f32_inplace(dot_partial, s1);

                // Broadcast sum from lane 0 to all lanes
                let score = ctx.shfl_idx_f32(dot_partial, 0, 31);

                // Apply scale: s = score * (1/sqrt(d))
                let s_scaled = ctx.mul_f32(score, scale_reg);

                // ===== Online softmax update =====
                // m_new = max(m_i, s_scaled)
                let m_new = ctx.max_f32(m_i, s_scaled);

                // correction = exp2((m_i - m_new) * log2(e))
                let m_diff = ctx.sub_f32(m_i, m_new);
                let m_diff_log2 = ctx.mul_f32(m_diff, log2e_reg);
                let correction = ctx.ex2_f32(m_diff_log2);

                // p = exp2((s_scaled - m_new) * log2(e))
                let s_diff = ctx.sub_f32(s_scaled, m_new);
                let s_diff_log2 = ctx.mul_f32(s_diff, log2e_reg);
                let p_val = ctx.ex2_f32(s_diff_log2);

                // l_i = l_i * correction + p
                ctx.mul_f32_inplace(l_i, correction);
                ctx.add_f32_inplace(l_i, p_val);

                // Update max
                ctx.mov_f32_reg(m_i, m_new);

                // O[d] = O[d] * correction + p * V[j, d]
                let v_row_off = ctx.mul_wide_u32_reg(col_j, kv_stride);
                let v_row_off_bytes = ctx.mul_u64(v_row_off, 4);
                let v_row_base = ctx.add_u64(v_ptr, v_row_off_bytes);
                let v_row_base = ctx.add_u64(v_row_base, kv_head_byte_off);
                let v_lane_base = ctx.add_u64(v_row_base, lane_byte_off);

                for e in 0..epl as usize {
                    // O[e] *= correction
                    ctx.mul_f32_inplace(o_regs[e], correction);
                    // O[e] += p * V[j, lane_base + e]
                    let v_addr = if e == 0 {
                        v_lane_base
                    } else {
                        let off = ctx.mov_u64_imm((e * 4) as u64);
                        ctx.add_u64(v_lane_base, off)
                    };
                    let v_val = ctx.ld_global_f32(v_addr);
                    ctx.fma_f32_inplace(o_regs[e], p_val, v_val);
                }

                // j++
                ctx.add_u32_inplace(col_j, 1);
                ctx.branch("col_start");
                ctx.label("col_end");

                // ===== Normalize and store output =====
                let o_row_off = ctx.mul_wide_u32_reg(row_i, q_stride);
                let o_row_off_bytes = ctx.mul_u64(o_row_off, 4);
                let o_row_base = ctx.add_u64(o_ptr, o_row_off_bytes);
                let o_row_base = ctx.add_u64(o_row_base, q_head_byte_off);
                let o_lane_base = ctx.add_u64(o_row_base, lane_byte_off);

                for e in 0..epl as usize {
                    ctx.div_f32_inplace(o_regs[e], l_i);
                    let o_addr = if e == 0 {
                        o_lane_base
                    } else {
                        let off = ctx.mov_u64_imm((e * 4) as u64);
                        ctx.add_u64(o_lane_base, off)
                    };
                    ctx.st_global_f32(o_addr, o_regs[e]);
                }

                // row_i++
                ctx.add_u32_inplace(row_i, 1);
                ctx.branch("row_start");
                ctx.label("row_end");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefill_attention_kernel_name() {
        let kernel = PrefillAttentionKernel::new(64, 6);
        assert_eq!(kernel.name(), "fused_prefill_attention_causal");
    }

    #[test]
    fn test_prefill_attention_ptx_generation() {
        let kernel = PrefillAttentionKernel::new(64, 6);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("fused_prefill_attention_causal"));
        assert!(ptx.contains(".param .u64 q_ptr"));
        assert!(ptx.contains(".param .u32 m_param"));
        assert!(ptx.contains(".param .u32 num_q_heads"));
    }

    #[test]
    fn test_prefill_attention_head_dim_128() {
        let kernel = PrefillAttentionKernel::new(128, 6);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("fused_prefill_attention_causal"));
    }

    #[test]
    fn test_prefill_attention_mha() {
        // Multi-head attention (no GQA): heads_per_kv = 1
        let kernel = PrefillAttentionKernel::new(64, 1);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("fused_prefill_attention_causal"));
    }

    #[test]
    #[should_panic(expected = "head_dim must be multiple of 32")]
    fn test_prefill_attention_invalid_head_dim() {
        let _kernel = PrefillAttentionKernel::new(48, 6);
    }
}
