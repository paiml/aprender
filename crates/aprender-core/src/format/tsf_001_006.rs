// SHIP-TWO-001 — `tensor-shape-flow-v1` algorithm-level PARTIAL
// discharge for FALSIFY-TSF-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/tensor-shape-flow-v1.yaml`.
// Spec: Pipeline shape flow — tensor shape transformations through
// transformer layers (Vaswani 2017, Ainslie 2023 GQA, Shazeer 2020 SwiGLU).

// ===========================================================================
// TSF-001 — QKV shape: Q_dim == n_h*d_k, K_dim == V_dim == n_kv*d_k
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tsf001Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_qkv_shape(
    n_h: u64,
    n_kv: u64,
    d_k: u64,
    q_dim: u64,
    k_dim: u64,
    v_dim: u64,
) -> Tsf001Verdict {
    if n_h == 0 || n_kv == 0 || d_k == 0 { return Tsf001Verdict::Fail; }
    if q_dim != n_h * d_k { return Tsf001Verdict::Fail; }
    if k_dim != n_kv * d_k { return Tsf001Verdict::Fail; }
    if v_dim != n_kv * d_k { return Tsf001Verdict::Fail; }
    if k_dim != v_dim { return Tsf001Verdict::Fail; } // K and V share KV-head dim
    Tsf001Verdict::Pass
}

// ===========================================================================
// TSF-002 — GQA grouping: n_h % n_kv == 0 AND n_h >= n_kv
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tsf002Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_gqa_grouping(n_h: u64, n_kv: u64) -> Tsf002Verdict {
    if n_h == 0 || n_kv == 0 { return Tsf002Verdict::Fail; }
    if n_kv > n_h { return Tsf002Verdict::Fail; }
    if !n_h.is_multiple_of(n_kv) { return Tsf002Verdict::Fail; }
    Tsf002Verdict::Pass
}

// ===========================================================================
// TSF-003 — Residual: shape(x + sublayer(x)) == shape(x)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tsf003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_residual_shape(input_shape: &[u64], output_shape: &[u64]) -> Tsf003Verdict {
    if input_shape.is_empty() || output_shape.is_empty() { return Tsf003Verdict::Fail; }
    if input_shape == output_shape { Tsf003Verdict::Pass } else { Tsf003Verdict::Fail }
}

// ===========================================================================
// TSF-004 — SwiGLU shape: gate/up [h]→[d_ff], down [d_ff]→[h], d_ff > h
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tsf004Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_swiglu_shape(
    h: u64,
    d_ff: u64,
    gate_in_dim: u64,
    gate_out_dim: u64,
    down_in_dim: u64,
    down_out_dim: u64,
) -> Tsf004Verdict {
    if h == 0 || d_ff == 0 { return Tsf004Verdict::Fail; }
    if d_ff <= h { return Tsf004Verdict::Fail; } // intermediate must expand
    if gate_in_dim != h { return Tsf004Verdict::Fail; }
    if gate_out_dim != d_ff { return Tsf004Verdict::Fail; }
    if down_in_dim != d_ff { return Tsf004Verdict::Fail; }
    if down_out_dim != h { return Tsf004Verdict::Fail; }
    Tsf004Verdict::Pass
}

// ===========================================================================
// TSF-005 — LM head: output_dim == vocab_size
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tsf005Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_lm_head_shape(output_dim: u64, vocab_size: u64) -> Tsf005Verdict {
    if output_dim == 0 || vocab_size == 0 { return Tsf005Verdict::Fail; }
    if output_dim == vocab_size { Tsf005Verdict::Pass } else { Tsf005Verdict::Fail }
}

// ===========================================================================
// TSF-006 — SIMD shape parity: byte-exact (contract tolerance=0.0)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tsf006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_simd_shape_parity(scalar: &[u64], simd: &[u64]) -> Tsf006Verdict {
    if scalar.is_empty() || simd.is_empty() { return Tsf006Verdict::Fail; }
    if scalar.len() != simd.len() { return Tsf006Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if s != v { return Tsf006Verdict::Fail; }
    }
    Tsf006Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // TSF-001 (QKV shape)
    #[test] fn tsf001_pass_qwen2_7b() {
        // Qwen2-7B: n_h=28, n_kv=4, d_k=128 → Q=3584, K=V=512.
        assert_eq!(
            verdict_from_qkv_shape(28, 4, 128, 3584, 512, 512),
            Tsf001Verdict::Pass
        );
    }
    #[test] fn tsf001_pass_full_attention() {
        // Non-GQA: n_h == n_kv → Q == K == V.
        assert_eq!(
            verdict_from_qkv_shape(32, 32, 128, 4096, 4096, 4096),
            Tsf001Verdict::Pass
        );
    }
    #[test] fn tsf001_fail_wrong_q_dim() {
        // Q dim should be n_h * d_k = 3584; observed 1024 wrong.
        assert_eq!(
            verdict_from_qkv_shape(28, 4, 128, 1024, 512, 512),
            Tsf001Verdict::Fail
        );
    }
    #[test] fn tsf001_fail_k_v_dim_mismatch() {
        // K and V must share dim = n_kv * d_k.
        assert_eq!(
            verdict_from_qkv_shape(28, 4, 128, 3584, 512, 1024),
            Tsf001Verdict::Fail
        );
    }
    #[test] fn tsf001_fail_zero() {
        assert_eq!(
            verdict_from_qkv_shape(0, 4, 128, 0, 512, 512),
            Tsf001Verdict::Fail
        );
    }

    // TSF-002 (GQA grouping)
    #[test] fn tsf002_pass_canonical() {
        // 28 query heads / 4 KV heads = 7 (exact).
        assert_eq!(verdict_from_gqa_grouping(28, 4), Tsf002Verdict::Pass);
    }
    #[test] fn tsf002_pass_full_mha() {
        // n_h == n_kv (group size 1).
        assert_eq!(verdict_from_gqa_grouping(32, 32), Tsf002Verdict::Pass);
    }
    #[test] fn tsf002_pass_mqa() {
        // Multi-query: many queries, 1 KV head.
        assert_eq!(verdict_from_gqa_grouping(32, 1), Tsf002Verdict::Pass);
    }
    #[test] fn tsf002_fail_indivisible() {
        // The contract's stated falsifier: "num_kv_heads to prime not
        // dividing num_heads" — 28 % 5 != 0.
        assert_eq!(verdict_from_gqa_grouping(28, 5), Tsf002Verdict::Fail);
    }
    #[test] fn tsf002_fail_n_kv_above_n_h() {
        assert_eq!(verdict_from_gqa_grouping(4, 8), Tsf002Verdict::Fail);
    }
    #[test] fn tsf002_fail_zero() {
        assert_eq!(verdict_from_gqa_grouping(0, 4), Tsf002Verdict::Fail);
        assert_eq!(verdict_from_gqa_grouping(28, 0), Tsf002Verdict::Fail);
    }

    // TSF-003 (residual shape)
    #[test] fn tsf003_pass_match() {
        let s = vec![1_u64, 16, 4096];
        assert_eq!(verdict_from_residual_shape(&s, &s), Tsf003Verdict::Pass);
    }
    #[test] fn tsf003_fail_drift() {
        let input = vec![1_u64, 16, 4096];
        let output = vec![1_u64, 16, 4097];
        assert_eq!(verdict_from_residual_shape(&input, &output), Tsf003Verdict::Fail);
    }
    #[test] fn tsf003_fail_extra_dim() {
        let input = vec![1_u64, 16, 4096];
        let output = vec![1_u64, 16, 4096, 1];
        assert_eq!(verdict_from_residual_shape(&input, &output), Tsf003Verdict::Fail);
    }

    // TSF-004 (SwiGLU shape)
    #[test] fn tsf004_pass_canonical() {
        // h=4096, d_ff=14336 (Qwen2-7B-class).
        assert_eq!(
            verdict_from_swiglu_shape(4096, 14336, 4096, 14336, 14336, 4096),
            Tsf004Verdict::Pass
        );
    }
    #[test] fn tsf004_fail_d_ff_le_h() {
        // d_ff must strictly expand beyond h.
        assert_eq!(
            verdict_from_swiglu_shape(4096, 4096, 4096, 4096, 4096, 4096),
            Tsf004Verdict::Fail
        );
    }
    #[test] fn tsf004_fail_gate_in_wrong() {
        // gate must accept h, not something else.
        assert_eq!(
            verdict_from_swiglu_shape(4096, 14336, 2048, 14336, 14336, 4096),
            Tsf004Verdict::Fail
        );
    }
    #[test] fn tsf004_fail_down_out_wrong() {
        // down must contract to h, not something else.
        assert_eq!(
            verdict_from_swiglu_shape(4096, 14336, 4096, 14336, 14336, 8192),
            Tsf004Verdict::Fail
        );
    }
    #[test] fn tsf004_fail_zero() {
        assert_eq!(
            verdict_from_swiglu_shape(0, 14336, 0, 14336, 14336, 4096),
            Tsf004Verdict::Fail
        );
    }

    // TSF-005 (LM head)
    #[test] fn tsf005_pass_canonical() {
        // Qwen2 vocab_size = 152064.
        assert_eq!(verdict_from_lm_head_shape(152064, 152064), Tsf005Verdict::Pass);
    }
    #[test] fn tsf005_fail_drift() {
        assert_eq!(verdict_from_lm_head_shape(151936, 152064), Tsf005Verdict::Fail);
    }
    #[test] fn tsf005_fail_zero() {
        assert_eq!(verdict_from_lm_head_shape(0, 152064), Tsf005Verdict::Fail);
        assert_eq!(verdict_from_lm_head_shape(152064, 0), Tsf005Verdict::Fail);
    }

    // TSF-006 (SIMD shape parity)
    #[test] fn tsf006_pass_identical() {
        let s = vec![1_u64, 16, 4096];
        assert_eq!(verdict_from_simd_shape_parity(&s, &s), Tsf006Verdict::Pass);
    }
    #[test] fn tsf006_fail_drift() {
        let scalar = vec![1_u64, 16, 4096];
        let simd = vec![1_u64, 16, 4097];
        assert_eq!(verdict_from_simd_shape_parity(&scalar, &simd), Tsf006Verdict::Fail);
    }
    #[test] fn tsf006_fail_length() {
        let scalar = vec![1_u64];
        let simd = vec![1_u64, 2];
        assert_eq!(verdict_from_simd_shape_parity(&scalar, &simd), Tsf006Verdict::Fail);
    }
    #[test] fn tsf006_fail_empty() {
        assert_eq!(verdict_from_simd_shape_parity(&[], &[]), Tsf006Verdict::Fail);
    }
}
