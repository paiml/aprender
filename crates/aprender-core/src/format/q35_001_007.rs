// SHIP-TWO-001 — `qwen35-shapes-v1` algorithm-level PARTIAL discharge
// for FALSIFY-Q35-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/qwen35-shapes-v1.yaml`.

// ===========================================================================
// Canonical Qwen3.5-9B shape constants (note larger d_k=256)
// ===========================================================================

pub const AC_Q35_HIDDEN_DIM: u64 = 4096;
pub const AC_Q35_NUM_HEADS: u64 = 16;
pub const AC_Q35_NUM_KV_HEADS: u64 = 4;
pub const AC_Q35_HEAD_DIM: u64 = 256;
pub const AC_Q35_INTERMEDIATE: u64 = 12_288;
pub const AC_Q35_KV_DIM: u64 = 1024;
pub const AC_Q35_SWIGLU_RATIO: f32 = 3.0;

// ===========================================================================
// Q35-001 — Q proj: n_h * d_k = 4096
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35001Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_q_projection(n_h: u64, d_k: u64) -> Q35001Verdict {
    if n_h == 0 || d_k == 0 { return Q35001Verdict::Fail; }
    if n_h * d_k == AC_Q35_HIDDEN_DIM { Q35001Verdict::Pass } else { Q35001Verdict::Fail }
}

// ===========================================================================
// Q35-002 — KV proj: n_kv * d_k = 1024
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35002Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_kv_projection(n_kv: u64, d_k: u64) -> Q35002Verdict {
    if n_kv == 0 || d_k == 0 { return Q35002Verdict::Fail; }
    if n_kv * d_k == AC_Q35_KV_DIM { Q35002Verdict::Pass } else { Q35002Verdict::Fail }
}

// ===========================================================================
// Q35-003 — SwiGLU expansion ratio == 3.0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_swiglu_ratio(hidden: u64, intermediate: u64) -> Q35003Verdict {
    if hidden == 0 || intermediate == 0 { return Q35003Verdict::Fail; }
    let ratio = intermediate as f32 / hidden as f32;
    if (ratio - AC_Q35_SWIGLU_RATIO).abs() < 1e-6 { Q35003Verdict::Pass } else { Q35003Verdict::Fail }
}

// ===========================================================================
// Q35-004 — O proj is square [hidden, hidden]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_o_projection_square(o_shape: [u64; 2]) -> Q35004Verdict {
    if o_shape != [AC_Q35_HIDDEN_DIM, AC_Q35_HIDDEN_DIM] { return Q35004Verdict::Fail; }
    if o_shape[0] != o_shape[1] { return Q35004Verdict::Fail; }
    Q35004Verdict::Pass
}

// ===========================================================================
// Q35-005 — RoPE freq vector len == d_k / 2 = 128
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35005Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_rope_freq_len(d_k: u64, observed: u64) -> Q35005Verdict {
    if d_k == 0 || !d_k.is_multiple_of(2) { return Q35005Verdict::Fail; }
    if observed == d_k / 2 { Q35005Verdict::Pass } else { Q35005Verdict::Fail }
}

// ===========================================================================
// Q35-006 — RoPE freqs strictly decreasing
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35006Verdict { Pass, Fail }

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
pub fn verdict_from_rope_decreasing(d_k: u64, base: f32) -> Q35006Verdict {
    let freqs = rope_freqs(d_k, base);
    if freqs.len() < 2 { return Q35006Verdict::Fail; }
    for w in freqs.windows(2) {
        if w[0] <= w[1] { return Q35006Verdict::Fail; }
    }
    Q35006Verdict::Pass
}

// ===========================================================================
// Q35-007 — SIMD vs scalar shape match
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_simd_shape_match(scalar_shape: &[u64], simd_shape: &[u64]) -> Q35007Verdict {
    if scalar_shape.is_empty() || simd_shape.is_empty() { return Q35007Verdict::Fail; }
    if scalar_shape == simd_shape { Q35007Verdict::Pass } else { Q35007Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn q35_001_pass() { assert_eq!(verdict_from_q_projection(16, 256), Q35001Verdict::Pass); }
    #[test] fn q35_001_fail_n_h() { assert_eq!(verdict_from_q_projection(15, 256), Q35001Verdict::Fail); }
    #[test] fn q35_001_fail_d_k() { assert_eq!(verdict_from_q_projection(16, 128), Q35001Verdict::Fail); }
    #[test] fn q35_001_fail_zero() { assert_eq!(verdict_from_q_projection(0, 256), Q35001Verdict::Fail); }

    #[test] fn q35_002_pass() { assert_eq!(verdict_from_kv_projection(4, 256), Q35002Verdict::Pass); }
    #[test] fn q35_002_fail() { assert_eq!(verdict_from_kv_projection(3, 256), Q35002Verdict::Fail); }

    #[test] fn q35_003_pass_3x() { assert_eq!(verdict_from_swiglu_ratio(4096, 12288), Q35003Verdict::Pass); }
    #[test] fn q35_003_fail_2x() { assert_eq!(verdict_from_swiglu_ratio(4096, 8192), Q35003Verdict::Fail); }

    #[test] fn q35_004_pass() { assert_eq!(verdict_from_o_projection_square([4096, 4096]), Q35004Verdict::Pass); }
    #[test] fn q35_004_fail() { assert_eq!(verdict_from_o_projection_square([4096, 2048]), Q35004Verdict::Fail); }

    #[test] fn q35_005_pass() { assert_eq!(verdict_from_rope_freq_len(256, 128), Q35005Verdict::Pass); }
    #[test] fn q35_005_fail_off_one() { assert_eq!(verdict_from_rope_freq_len(256, 127), Q35005Verdict::Fail); }
    #[test] fn q35_005_fail_odd() { assert_eq!(verdict_from_rope_freq_len(255, 127), Q35005Verdict::Fail); }

    #[test] fn q35_006_pass_canonical() { assert_eq!(verdict_from_rope_decreasing(256, 1_000_000.0), Q35006Verdict::Pass); }
    #[test] fn q35_006_fail_zero() { assert_eq!(verdict_from_rope_decreasing(0, 10000.0), Q35006Verdict::Fail); }

    #[test] fn q35_007_pass() {
        assert_eq!(verdict_from_simd_shape_match(&[4096, 4096], &[4096, 4096]), Q35007Verdict::Pass);
    }
    #[test] fn q35_007_fail() {
        assert_eq!(verdict_from_simd_shape_match(&[4096, 4096], &[4096, 2048]), Q35007Verdict::Fail);
    }

    // Provenance pins
    #[test] fn provenance_constants() {
        assert_eq!(AC_Q35_HIDDEN_DIM, 4096);
        assert_eq!(AC_Q35_NUM_HEADS, 16);
        assert_eq!(AC_Q35_NUM_KV_HEADS, 4);
        assert_eq!(AC_Q35_HEAD_DIM, 256);
        assert_eq!(AC_Q35_INTERMEDIATE, 12_288);
        assert_eq!(AC_Q35_KV_DIM, 1024);
        assert!((AC_Q35_SWIGLU_RATIO - 3.0).abs() < 1e-9);
    }

    #[test] fn provenance_self_consistency() {
        assert_eq!(AC_Q35_NUM_HEADS * AC_Q35_HEAD_DIM, AC_Q35_HIDDEN_DIM);
        assert_eq!(AC_Q35_NUM_KV_HEADS * AC_Q35_HEAD_DIM, AC_Q35_KV_DIM);
        assert_eq!(AC_Q35_INTERMEDIATE, AC_Q35_HIDDEN_DIM * 3);
    }
}
