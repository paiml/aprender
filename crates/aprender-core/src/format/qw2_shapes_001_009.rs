// SHIP-TWO-001 — `qwen2-shapes-v1` algorithm-level PARTIAL discharge
// for FALSIFY-QW2-001..009 (closes 9/9 sweep).
//
// Contract: `contracts/qwen2-shapes-v1.yaml`.
// Spec: SHIP-TWO-001 §4 (MODEL-1 Qwen2.5-Coder-7B teacher).
//
// All shape constants pinned to the canonical Qwen2.5-Coder-7B
// configuration. Each verdict is a strict-equality decision rule.

// ===========================================================================
// Canonical Qwen2.5-Coder-7B shape constants (HuggingFace config.json).
// ===========================================================================

pub const AC_QW2_HIDDEN_DIM: u64 = 3584;
pub const AC_QW2_NUM_HEADS: u64 = 28;
pub const AC_QW2_NUM_KV_HEADS: u64 = 4;
pub const AC_QW2_HEAD_DIM: u64 = 128;
pub const AC_QW2_INTERMEDIATE_SIZE: u64 = 18_944;
pub const AC_QW2_ROPE_BASE: f32 = 1_000_000.0;

// ===========================================================================
// QW2-001 — Q projection: n_h * d_k = hidden_dim
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2Shape001Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_q_projection(n_h: u64, d_k: u64) -> Qw2Shape001Verdict {
    if n_h == 0 || d_k == 0 { return Qw2Shape001Verdict::Fail; }
    if n_h * d_k == AC_QW2_HIDDEN_DIM { Qw2Shape001Verdict::Pass } else { Qw2Shape001Verdict::Fail }
}

// ===========================================================================
// QW2-002 — KV projection: n_kv * d_k = 512
// ===========================================================================

pub const AC_QW2_KV_DIM: u64 = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2Shape002Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_kv_projection(n_kv: u64, d_k: u64) -> Qw2Shape002Verdict {
    if n_kv == 0 || d_k == 0 { return Qw2Shape002Verdict::Fail; }
    if n_kv * d_k == AC_QW2_KV_DIM { Qw2Shape002Verdict::Pass } else { Qw2Shape002Verdict::Fail }
}

// ===========================================================================
// QW2-003 — GQA divisibility: n_h.is_multiple_of(n_kv)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2Shape003Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_gqa_divisibility(n_h: u64, n_kv: u64) -> Qw2Shape003Verdict {
    if n_h == 0 || n_kv == 0 { return Qw2Shape003Verdict::Fail; }
    if n_h.is_multiple_of(n_kv) { Qw2Shape003Verdict::Pass } else { Qw2Shape003Verdict::Fail }
}

// ===========================================================================
// QW2-004 — SwiGLU gate/up shape: [intermediate_size, hidden_dim]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2Shape004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_swiglu_shape(gate_shape: [u64; 2], up_shape: [u64; 2]) -> Qw2Shape004Verdict {
    let expected = [AC_QW2_INTERMEDIATE_SIZE, AC_QW2_HIDDEN_DIM];
    if gate_shape != expected { return Qw2Shape004Verdict::Fail; }
    if up_shape != expected { return Qw2Shape004Verdict::Fail; }
    Qw2Shape004Verdict::Pass
}

// ===========================================================================
// QW2-005 — O projection is square (transpose of Q): [hidden, hidden]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2Shape005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_o_projection_square(o_shape: [u64; 2]) -> Qw2Shape005Verdict {
    if o_shape != [AC_QW2_HIDDEN_DIM, AC_QW2_HIDDEN_DIM] { return Qw2Shape005Verdict::Fail; }
    // Transpose check: square matrix is its own transpose-shape.
    if o_shape != [o_shape[1], o_shape[0]] { return Qw2Shape005Verdict::Fail; }
    Qw2Shape005Verdict::Pass
}

// ===========================================================================
// QW2-006 — RoPE frequency vector length == d_k / 2
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2Shape006Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_rope_freq_len(d_k: u64, observed_freq_len: u64) -> Qw2Shape006Verdict {
    if d_k == 0 || !d_k.is_multiple_of(2) { return Qw2Shape006Verdict::Fail; }
    if observed_freq_len == d_k / 2 { Qw2Shape006Verdict::Pass } else { Qw2Shape006Verdict::Fail }
}

// ===========================================================================
// QW2-007 — RoPE frequencies strictly decreasing
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2Shape007Verdict { Pass, Fail }

/// Compute canonical RoPE freqs: `freq_i = base^(-2*i / d_k)`.
#[must_use]
pub fn rope_freqs(d_k: u64, base: f32) -> Vec<f32> {
    if d_k == 0 || !d_k.is_multiple_of(2) || base <= 1.0 { return vec![]; }
    let half = d_k / 2;
    (0..half).map(|i| {
        let p = -2.0_f32 * (i as f32) / (d_k as f32);
        base.powf(p)
    }).collect()
}

#[must_use]
pub fn verdict_from_rope_decreasing(d_k: u64, base: f32) -> Qw2Shape007Verdict {
    let freqs = rope_freqs(d_k, base);
    if freqs.len() < 2 { return Qw2Shape007Verdict::Fail; }
    for w in freqs.windows(2) {
        if w[0] <= w[1] { return Qw2Shape007Verdict::Fail; }
    }
    Qw2Shape007Verdict::Pass
}

// ===========================================================================
// QW2-008 — Head dim consistency: hidden_dim % num_heads == 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2Shape008Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_head_dim_consistency(hidden: u64, num_heads: u64) -> Qw2Shape008Verdict {
    if hidden == 0 || num_heads == 0 { return Qw2Shape008Verdict::Fail; }
    if !hidden.is_multiple_of(num_heads) { return Qw2Shape008Verdict::Fail; }
    if hidden / num_heads == AC_QW2_HEAD_DIM { Qw2Shape008Verdict::Pass } else { Qw2Shape008Verdict::Fail }
}

// ===========================================================================
// QW2-009 — SIMD vs scalar shape equivalence
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2Shape009Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_simd_shape_match(scalar_shape: &[u64], simd_shape: &[u64]) -> Qw2Shape009Verdict {
    if scalar_shape.is_empty() || simd_shape.is_empty() { return Qw2Shape009Verdict::Fail; }
    if scalar_shape == simd_shape { Qw2Shape009Verdict::Pass } else { Qw2Shape009Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- QW2-001 ----------------------------------------------------------

    #[test] fn qw2_001_pass_canonical() { assert_eq!(verdict_from_q_projection(28, 128), Qw2Shape001Verdict::Pass); }
    #[test] fn qw2_001_fail_off_by_n_h() { assert_eq!(verdict_from_q_projection(27, 128), Qw2Shape001Verdict::Fail); }
    #[test] fn qw2_001_fail_off_by_d_k() { assert_eq!(verdict_from_q_projection(28, 64), Qw2Shape001Verdict::Fail); }
    #[test] fn qw2_001_fail_zero() { assert_eq!(verdict_from_q_projection(0, 128), Qw2Shape001Verdict::Fail); }

    // ----- QW2-002 ----------------------------------------------------------

    #[test] fn qw2_002_pass_canonical() { assert_eq!(verdict_from_kv_projection(4, 128), Qw2Shape002Verdict::Pass); }
    #[test] fn qw2_002_fail_off() { assert_eq!(verdict_from_kv_projection(3, 128), Qw2Shape002Verdict::Fail); }

    // ----- QW2-003 ----------------------------------------------------------

    #[test] fn qw2_003_pass_canonical() { assert_eq!(verdict_from_gqa_divisibility(28, 4), Qw2Shape003Verdict::Pass); }
    #[test] fn qw2_003_fail_indivisible() { assert_eq!(verdict_from_gqa_divisibility(28, 5), Qw2Shape003Verdict::Fail); }

    // ----- QW2-004 ----------------------------------------------------------

    #[test] fn qw2_004_pass_canonical() {
        assert_eq!(
            verdict_from_swiglu_shape([18944, 3584], [18944, 3584]),
            Qw2Shape004Verdict::Pass
        );
    }
    #[test] fn qw2_004_fail_swapped() {
        assert_eq!(
            verdict_from_swiglu_shape([3584, 18944], [18944, 3584]),
            Qw2Shape004Verdict::Fail
        );
    }
    #[test] fn qw2_004_fail_wrong_intermediate() {
        assert_eq!(
            verdict_from_swiglu_shape([18432, 3584], [18432, 3584]),
            Qw2Shape004Verdict::Fail
        );
    }

    // ----- QW2-005 ----------------------------------------------------------

    #[test] fn qw2_005_pass_canonical() { assert_eq!(verdict_from_o_projection_square([3584, 3584]), Qw2Shape005Verdict::Pass); }
    #[test] fn qw2_005_fail_non_square() { assert_eq!(verdict_from_o_projection_square([3584, 1024]), Qw2Shape005Verdict::Fail); }

    // ----- QW2-006 ----------------------------------------------------------

    #[test] fn qw2_006_pass_canonical() { assert_eq!(verdict_from_rope_freq_len(128, 64), Qw2Shape006Verdict::Pass); }
    #[test] fn qw2_006_fail_off_by_one() { assert_eq!(verdict_from_rope_freq_len(128, 63), Qw2Shape006Verdict::Fail); }
    #[test] fn qw2_006_fail_odd_d_k() { assert_eq!(verdict_from_rope_freq_len(127, 63), Qw2Shape006Verdict::Fail); }

    // ----- QW2-007 ----------------------------------------------------------

    #[test] fn qw2_007_pass_canonical() {
        assert_eq!(verdict_from_rope_decreasing(128, AC_QW2_ROPE_BASE), Qw2Shape007Verdict::Pass);
    }
    #[test] fn qw2_007_pass_smaller_d_k() {
        assert_eq!(verdict_from_rope_decreasing(64, 10000.0), Qw2Shape007Verdict::Pass);
    }
    #[test] fn qw2_007_fail_too_small() {
        assert_eq!(verdict_from_rope_decreasing(0, 10000.0), Qw2Shape007Verdict::Fail);
    }

    // ----- QW2-008 ----------------------------------------------------------

    #[test] fn qw2_008_pass_canonical() { assert_eq!(verdict_from_head_dim_consistency(3584, 28), Qw2Shape008Verdict::Pass); }
    #[test] fn qw2_008_fail_indivisible() { assert_eq!(verdict_from_head_dim_consistency(3584, 27), Qw2Shape008Verdict::Fail); }
    #[test] fn qw2_008_fail_wrong_head_dim() { assert_eq!(verdict_from_head_dim_consistency(3584, 56), Qw2Shape008Verdict::Fail); }

    // ----- QW2-009 ----------------------------------------------------------

    #[test] fn qw2_009_pass_match() {
        assert_eq!(
            verdict_from_simd_shape_match(&[3584, 3584], &[3584, 3584]),
            Qw2Shape009Verdict::Pass
        );
    }
    #[test] fn qw2_009_fail_drift() {
        assert_eq!(
            verdict_from_simd_shape_match(&[3584, 3584], &[3584, 1024]),
            Qw2Shape009Verdict::Fail
        );
    }

    // Provenance pins
    #[test] fn provenance_constants() {
        assert_eq!(AC_QW2_HIDDEN_DIM, 3584);
        assert_eq!(AC_QW2_NUM_HEADS, 28);
        assert_eq!(AC_QW2_NUM_KV_HEADS, 4);
        assert_eq!(AC_QW2_HEAD_DIM, 128);
        assert_eq!(AC_QW2_INTERMEDIATE_SIZE, 18_944);
        assert_eq!(AC_QW2_KV_DIM, 512);
        assert!((AC_QW2_ROPE_BASE - 1_000_000.0).abs() < 1e-3);
    }

    #[test] fn provenance_consistency_self_check() {
        assert_eq!(AC_QW2_NUM_HEADS * AC_QW2_HEAD_DIM, AC_QW2_HIDDEN_DIM);
        assert_eq!(AC_QW2_NUM_KV_HEADS * AC_QW2_HEAD_DIM, AC_QW2_KV_DIM);
        assert_eq!(AC_QW2_HIDDEN_DIM % AC_QW2_NUM_HEADS, 0);
        assert_eq!(AC_QW2_NUM_HEADS % AC_QW2_NUM_KV_HEADS, 0);
    }
}
