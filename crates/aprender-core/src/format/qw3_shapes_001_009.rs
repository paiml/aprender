// SHIP-TWO-001 — `qwen3-shapes-v1` algorithm-level PARTIAL discharge
// for FALSIFY-QW3-001..009 (closes 9/9 sweep).
//
// Contract: `contracts/qwen3-shapes-v1.yaml`.

// ===========================================================================
// Canonical Qwen3-8B shape constants
// ===========================================================================

pub const AC_QW3_HIDDEN_DIM: u64 = 4096;
pub const AC_QW3_NUM_HEADS: u64 = 32;
pub const AC_QW3_NUM_KV_HEADS: u64 = 8;
pub const AC_QW3_HEAD_DIM: u64 = 128;
pub const AC_QW3_INTERMEDIATE_SIZE: u64 = 12_288;
pub const AC_QW3_KV_DIM: u64 = 1024;
pub const AC_QW3_SWIGLU_RATIO: f32 = 3.0;

// ===========================================================================
// QW3-001 — Q proj: n_h * d_k = 4096
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3Shape001Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_q_projection(n_h: u64, d_k: u64) -> Qw3Shape001Verdict {
    if n_h == 0 || d_k == 0 { return Qw3Shape001Verdict::Fail; }
    if n_h * d_k == AC_QW3_HIDDEN_DIM { Qw3Shape001Verdict::Pass } else { Qw3Shape001Verdict::Fail }
}

// ===========================================================================
// QW3-002 — KV proj: n_kv * d_k = 1024
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3Shape002Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_kv_projection(n_kv: u64, d_k: u64) -> Qw3Shape002Verdict {
    if n_kv == 0 || d_k == 0 { return Qw3Shape002Verdict::Fail; }
    if n_kv * d_k == AC_QW3_KV_DIM { Qw3Shape002Verdict::Pass } else { Qw3Shape002Verdict::Fail }
}

// ===========================================================================
// QW3-003 — GQA divisibility: n_h.is_multiple_of(n_kv)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3Shape003Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_gqa_divisibility(n_h: u64, n_kv: u64) -> Qw3Shape003Verdict {
    if n_h == 0 || n_kv == 0 { return Qw3Shape003Verdict::Fail; }
    if n_h.is_multiple_of(n_kv) { Qw3Shape003Verdict::Pass } else { Qw3Shape003Verdict::Fail }
}

// ===========================================================================
// QW3-004 — SwiGLU expansion ratio: intermediate / hidden == 3.0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3Shape004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_swiglu_ratio(hidden: u64, intermediate: u64) -> Qw3Shape004Verdict {
    if hidden == 0 || intermediate == 0 { return Qw3Shape004Verdict::Fail; }
    let ratio = intermediate as f32 / hidden as f32;
    if (ratio - AC_QW3_SWIGLU_RATIO).abs() < 1e-6 { Qw3Shape004Verdict::Pass } else { Qw3Shape004Verdict::Fail }
}

// ===========================================================================
// QW3-005 — O proj is square [4096, 4096]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3Shape005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_o_projection_square(o_shape: [u64; 2]) -> Qw3Shape005Verdict {
    if o_shape != [AC_QW3_HIDDEN_DIM, AC_QW3_HIDDEN_DIM] { return Qw3Shape005Verdict::Fail; }
    if o_shape != [o_shape[1], o_shape[0]] { return Qw3Shape005Verdict::Fail; }
    Qw3Shape005Verdict::Pass
}

// ===========================================================================
// QW3-006 — RoPE freq vector len == d_k / 2
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3Shape006Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_rope_freq_len(d_k: u64, observed_freq_len: u64) -> Qw3Shape006Verdict {
    if d_k == 0 || !d_k.is_multiple_of(2) { return Qw3Shape006Verdict::Fail; }
    if observed_freq_len == d_k / 2 { Qw3Shape006Verdict::Pass } else { Qw3Shape006Verdict::Fail }
}

// ===========================================================================
// QW3-007 — RoPE freqs strictly decreasing
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3Shape007Verdict { Pass, Fail }

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
pub fn verdict_from_rope_decreasing(d_k: u64, base: f32) -> Qw3Shape007Verdict {
    let freqs = rope_freqs(d_k, base);
    if freqs.len() < 2 { return Qw3Shape007Verdict::Fail; }
    for w in freqs.windows(2) {
        if w[0] <= w[1] { return Qw3Shape007Verdict::Fail; }
    }
    Qw3Shape007Verdict::Pass
}

// ===========================================================================
// QW3-008 — Head dim consistency: 4096 / 32 == 128
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3Shape008Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_head_dim_consistency(hidden: u64, num_heads: u64) -> Qw3Shape008Verdict {
    if hidden == 0 || num_heads == 0 { return Qw3Shape008Verdict::Fail; }
    if !hidden.is_multiple_of(num_heads) { return Qw3Shape008Verdict::Fail; }
    if hidden / num_heads == AC_QW3_HEAD_DIM { Qw3Shape008Verdict::Pass } else { Qw3Shape008Verdict::Fail }
}

// ===========================================================================
// QW3-009 — SIMD vs scalar shape equivalence
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3Shape009Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_simd_shape_match(scalar_shape: &[u64], simd_shape: &[u64]) -> Qw3Shape009Verdict {
    if scalar_shape.is_empty() || simd_shape.is_empty() { return Qw3Shape009Verdict::Fail; }
    if scalar_shape == simd_shape { Qw3Shape009Verdict::Pass } else { Qw3Shape009Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn qw3_001_pass() { assert_eq!(verdict_from_q_projection(32, 128), Qw3Shape001Verdict::Pass); }
    #[test] fn qw3_001_fail_n_h() { assert_eq!(verdict_from_q_projection(31, 128), Qw3Shape001Verdict::Fail); }
    #[test] fn qw3_001_fail_zero() { assert_eq!(verdict_from_q_projection(0, 128), Qw3Shape001Verdict::Fail); }

    #[test] fn qw3_002_pass() { assert_eq!(verdict_from_kv_projection(8, 128), Qw3Shape002Verdict::Pass); }
    #[test] fn qw3_002_fail() { assert_eq!(verdict_from_kv_projection(7, 128), Qw3Shape002Verdict::Fail); }

    #[test] fn qw3_003_pass() { assert_eq!(verdict_from_gqa_divisibility(32, 8), Qw3Shape003Verdict::Pass); }
    #[test] fn qw3_003_fail() { assert_eq!(verdict_from_gqa_divisibility(32, 7), Qw3Shape003Verdict::Fail); }

    #[test] fn qw3_004_pass_canonical() {
        assert_eq!(verdict_from_swiglu_ratio(4096, 12288), Qw3Shape004Verdict::Pass);
    }
    #[test] fn qw3_004_fail_2x() {
        assert_eq!(verdict_from_swiglu_ratio(4096, 8192), Qw3Shape004Verdict::Fail);
    }
    #[test] fn qw3_004_fail_4x() {
        assert_eq!(verdict_from_swiglu_ratio(4096, 16384), Qw3Shape004Verdict::Fail);
    }

    #[test] fn qw3_005_pass() { assert_eq!(verdict_from_o_projection_square([4096, 4096]), Qw3Shape005Verdict::Pass); }
    #[test] fn qw3_005_fail() { assert_eq!(verdict_from_o_projection_square([4096, 2048]), Qw3Shape005Verdict::Fail); }

    #[test] fn qw3_006_pass() { assert_eq!(verdict_from_rope_freq_len(128, 64), Qw3Shape006Verdict::Pass); }
    #[test] fn qw3_006_fail() { assert_eq!(verdict_from_rope_freq_len(128, 65), Qw3Shape006Verdict::Fail); }
    #[test] fn qw3_006_fail_odd_d_k() { assert_eq!(verdict_from_rope_freq_len(127, 63), Qw3Shape006Verdict::Fail); }

    #[test] fn qw3_007_pass() { assert_eq!(verdict_from_rope_decreasing(128, 1_000_000.0), Qw3Shape007Verdict::Pass); }
    #[test] fn qw3_007_fail_zero_d_k() { assert_eq!(verdict_from_rope_decreasing(0, 10000.0), Qw3Shape007Verdict::Fail); }

    #[test] fn qw3_008_pass() { assert_eq!(verdict_from_head_dim_consistency(4096, 32), Qw3Shape008Verdict::Pass); }
    #[test] fn qw3_008_fail_indivisible() { assert_eq!(verdict_from_head_dim_consistency(4096, 31), Qw3Shape008Verdict::Fail); }

    #[test] fn qw3_009_pass() {
        assert_eq!(verdict_from_simd_shape_match(&[4096, 4096], &[4096, 4096]), Qw3Shape009Verdict::Pass);
    }
    #[test] fn qw3_009_fail() {
        assert_eq!(verdict_from_simd_shape_match(&[4096, 4096], &[4096, 2048]), Qw3Shape009Verdict::Fail);
    }

    #[test] fn provenance_constants() {
        assert_eq!(AC_QW3_HIDDEN_DIM, 4096);
        assert_eq!(AC_QW3_NUM_HEADS, 32);
        assert_eq!(AC_QW3_NUM_KV_HEADS, 8);
        assert_eq!(AC_QW3_HEAD_DIM, 128);
        assert_eq!(AC_QW3_INTERMEDIATE_SIZE, 12_288);
        assert_eq!(AC_QW3_KV_DIM, 1024);
        assert!((AC_QW3_SWIGLU_RATIO - 3.0).abs() < 1e-9);
    }

    #[test] fn provenance_self_consistency() {
        assert_eq!(AC_QW3_NUM_HEADS * AC_QW3_HEAD_DIM, AC_QW3_HIDDEN_DIM);
        assert_eq!(AC_QW3_NUM_KV_HEADS * AC_QW3_HEAD_DIM, AC_QW3_KV_DIM);
        assert_eq!(AC_QW3_HIDDEN_DIM % AC_QW3_NUM_HEADS, 0);
        assert_eq!(AC_QW3_NUM_HEADS % AC_QW3_NUM_KV_HEADS, 0);
        assert_eq!(AC_QW3_INTERMEDIATE_SIZE, AC_QW3_HIDDEN_DIM * 3);
    }
}
