// SHIP-TWO-001 — `alibi-kernel-v1` algorithm-level PARTIAL
// discharge for FALSIFY-AL-001..005 (closes 5/5 sweep).
//
// Contract: `contracts/alibi-kernel-v1.yaml`.
// Spec: ALiBi positional encoding (Press et al. 2022 Train Short, Test Long).

// ===========================================================================
// Helpers — alibi slope and bias (in-module reference impl)
// ===========================================================================

/// m_h = 2^(-8h/H)
#[must_use]
pub fn alibi_slope(h: u64, total_heads: u64) -> Option<f32> {
    if total_heads == 0 || h >= total_heads { return None; }
    let exponent = -8.0_f32 * (h as f32) / (total_heads as f32);
    let m = (2.0_f32).powf(exponent);
    if !m.is_finite() || m <= 0.0 { return None; }
    Some(m)
}

/// bias[i, j] = -m_h * |i - j|
#[must_use]
pub fn alibi_bias(i: u64, j: u64, slope: f32) -> Option<f32> {
    if !slope.is_finite() || slope <= 0.0 { return None; }
    let dist = if i > j { (i - j) as f32 } else { (j - i) as f32 };
    let b = -slope * dist;
    if !b.is_finite() { return None; }
    Some(b)
}

// ===========================================================================
// AL-001 — Negative bias: -m_h * |i - j| ≤ 0 ∀ i, j, h
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Al001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_negative_bias(probes: &[(u64, u64, u64, u64)]) -> Al001Verdict {
    if probes.is_empty() { return Al001Verdict::Fail; }
    for &(i, j, h, total_heads) in probes {
        let m = match alibi_slope(h, total_heads) {
            Some(v) => v,
            None => return Al001Verdict::Fail,
        };
        let b = match alibi_bias(i, j, m) {
            Some(v) => v,
            None => return Al001Verdict::Fail,
        };
        if b > 0.0 { return Al001Verdict::Fail; }
    }
    Al001Verdict::Pass
}

// ===========================================================================
// AL-002 — Slope positivity: m_h > 0 ∀ h ∈ {0, ..., H-1}
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Al002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_slope_positivity(total_heads: u64) -> Al002Verdict {
    if total_heads == 0 { return Al002Verdict::Fail; }
    for h in 0..total_heads {
        match alibi_slope(h, total_heads) {
            Some(m) if m > 0.0 && m.is_finite() => {}
            _ => return Al002Verdict::Fail,
        }
    }
    Al002Verdict::Pass
}

// ===========================================================================
// AL-003 — Causal consistency: j > i ⟹ scores[i, j] == -inf in causal mode
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Al003Verdict { Pass, Fail }

/// Caller passes scores after causal mask is applied. Pass iff every
/// score where j > i is -inf, AND every score where j ≤ i is finite.
#[must_use]
pub fn verdict_from_causal_consistency(scores: &[Vec<f32>]) -> Al003Verdict {
    if scores.is_empty() { return Al003Verdict::Fail; }
    let n = scores.len();
    for (i, row) in scores.iter().enumerate() {
        if row.len() != n { return Al003Verdict::Fail; }
        for (j, &s) in row.iter().enumerate() {
            if j > i {
                // Future positions must be -inf.
                if !s.is_infinite() || s > 0.0 { return Al003Verdict::Fail; }
            } else {
                // Past + current positions must be finite.
                if !s.is_finite() { return Al003Verdict::Fail; }
            }
        }
    }
    Al003Verdict::Pass
}

// ===========================================================================
// AL-004 — Head-monotonic slopes: m_{h} > m_{h+1}
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Al004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_head_monotonic(total_heads: u64) -> Al004Verdict {
    if total_heads < 2 { return Al004Verdict::Fail; } // need ≥2 heads to compare
    let mut prev = match alibi_slope(0, total_heads) {
        Some(m) => m,
        None => return Al004Verdict::Fail,
    };
    for h in 1..total_heads {
        let cur = match alibi_slope(h, total_heads) {
            Some(m) => m,
            None => return Al004Verdict::Fail,
        };
        if cur >= prev { return Al004Verdict::Fail; }
        prev = cur;
    }
    Al004Verdict::Pass
}

// ===========================================================================
// AL-005 — SIMD parity within 8 ULP
// ===========================================================================

pub const AC_AL_005_ULP_TOLERANCE: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Al005Verdict { Pass, Fail }

#[must_use]
pub fn ulp_distance(a: f32, b: f32) -> u32 {
    if !a.is_finite() || !b.is_finite() { return u32::MAX; }
    if a == b { return 0; }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    let ord_a = if ai < 0 { i32::MIN.wrapping_sub(ai).wrapping_add(1) } else { ai };
    let ord_b = if bi < 0 { i32::MIN.wrapping_sub(bi).wrapping_add(1) } else { bi };
    ord_a.wrapping_sub(ord_b).unsigned_abs()
}

#[must_use]
pub fn verdict_from_simd_parity(scalar: &[f32], simd: &[f32]) -> Al005Verdict {
    if scalar.is_empty() || simd.is_empty() { return Al005Verdict::Fail; }
    if scalar.len() != simd.len() { return Al005Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if !s.is_finite() || !v.is_finite() { return Al005Verdict::Fail; }
        if ulp_distance(s, v) > AC_AL_005_ULP_TOLERANCE { return Al005Verdict::Fail; }
    }
    Al005Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // AL-001 (negative bias)
    #[test] fn al001_pass_canonical() {
        // 8 heads, various positions.
        let probes = vec![(0_u64, 0, 0, 8), (10, 5, 1, 8), (5, 10, 3, 8), (100, 50, 7, 8)];
        assert_eq!(verdict_from_negative_bias(&probes), Al001Verdict::Pass);
    }
    #[test] fn al001_pass_self_position_zero() {
        // i == j → bias = 0.
        let probes = vec![(0_u64, 0, 0, 8), (5, 5, 1, 8)];
        assert_eq!(verdict_from_negative_bias(&probes), Al001Verdict::Pass);
    }
    #[test] fn al001_fail_invalid_head() {
        // h ≥ total_heads is invalid.
        let probes = vec![(0_u64, 0, 8, 8)];
        assert_eq!(verdict_from_negative_bias(&probes), Al001Verdict::Fail);
    }
    #[test] fn al001_fail_empty() {
        assert_eq!(verdict_from_negative_bias(&[]), Al001Verdict::Fail);
    }

    // AL-002 (slope positivity)
    #[test] fn al002_pass_8_heads() {
        assert_eq!(verdict_from_slope_positivity(8), Al002Verdict::Pass);
    }
    #[test] fn al002_pass_32_heads() {
        // Qwen2-7B has 28 heads; 32 is similar scale.
        assert_eq!(verdict_from_slope_positivity(32), Al002Verdict::Pass);
    }
    #[test] fn al002_pass_one_head() {
        assert_eq!(verdict_from_slope_positivity(1), Al002Verdict::Pass);
    }
    #[test] fn al002_pass_max_typical() {
        // 128 heads (Llama-3-70B class).
        assert_eq!(verdict_from_slope_positivity(128), Al002Verdict::Pass);
    }
    #[test] fn al002_fail_zero_heads() {
        assert_eq!(verdict_from_slope_positivity(0), Al002Verdict::Fail);
    }

    // AL-003 (causal consistency)
    #[test] fn al003_pass_canonical_4x4() {
        let neg_inf = f32::NEG_INFINITY;
        let scores = vec![
            vec![0.0_f32, neg_inf, neg_inf, neg_inf],
            vec![-0.5,    0.0,     neg_inf, neg_inf],
            vec![-1.0,    -0.5,    0.0,     neg_inf],
            vec![-1.5,    -1.0,    -0.5,    0.0],
        ];
        assert_eq!(verdict_from_causal_consistency(&scores), Al003Verdict::Pass);
    }
    #[test] fn al003_fail_no_mask() {
        // Future position should be -inf but is finite.
        let scores = vec![vec![0.0_f32, -0.5], vec![-0.5, 0.0]];
        assert_eq!(verdict_from_causal_consistency(&scores), Al003Verdict::Fail);
    }
    #[test] fn al003_fail_finite_past() {
        // Past position is -inf (wrong direction of mask).
        let neg_inf = f32::NEG_INFINITY;
        let scores = vec![
            vec![0.0_f32, neg_inf],
            vec![neg_inf, 0.0], // [1, 0] should be finite
        ];
        assert_eq!(verdict_from_causal_consistency(&scores), Al003Verdict::Fail);
    }
    #[test] fn al003_fail_non_square() {
        let scores = vec![vec![0.0_f32, -0.5, -1.0]];
        assert_eq!(verdict_from_causal_consistency(&scores), Al003Verdict::Fail);
    }
    #[test] fn al003_fail_empty() {
        let empty: Vec<Vec<f32>> = vec![];
        assert_eq!(verdict_from_causal_consistency(&empty), Al003Verdict::Fail);
    }

    // AL-004 (head monotonic)
    #[test] fn al004_pass_8_heads() {
        // m_0 > m_1 > ... > m_7 since exponent grows more negative.
        assert_eq!(verdict_from_head_monotonic(8), Al004Verdict::Pass);
    }
    #[test] fn al004_pass_32_heads() {
        assert_eq!(verdict_from_head_monotonic(32), Al004Verdict::Pass);
    }
    #[test] fn al004_pass_2_heads() {
        assert_eq!(verdict_from_head_monotonic(2), Al004Verdict::Pass);
    }
    #[test] fn al004_fail_one_head() {
        // Need ≥2 heads to compare.
        assert_eq!(verdict_from_head_monotonic(1), Al004Verdict::Fail);
    }
    #[test] fn al004_fail_zero_heads() {
        assert_eq!(verdict_from_head_monotonic(0), Al004Verdict::Fail);
    }

    // AL-005 (SIMD parity)
    #[test] fn al005_pass_identical() {
        let a = vec![-0.5_f32, -1.0, -1.5];
        assert_eq!(verdict_from_simd_parity(&a, &a), Al005Verdict::Pass);
    }
    #[test] fn al005_pass_within_ulp() {
        let a = vec![-0.5_f32];
        let b = vec![f32::from_bits((-0.5_f32).to_bits() + 4)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Al005Verdict::Pass);
    }
    #[test] fn al005_fail_above_ulp() {
        let a = vec![-0.5_f32];
        let b = vec![f32::from_bits((-0.5_f32).to_bits() + 100)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Al005Verdict::Fail);
    }

    // Helper sanity
    #[test] fn slope_first_head() {
        // m_0 = 2^(-8*0/H) = 2^0 = 1.0
        assert!((alibi_slope(0, 8).unwrap() - 1.0).abs() < 1e-7);
    }
    #[test] fn slope_last_head_8h() {
        // For H=8, h=7: m_7 = 2^(-7) = 1/128 = 0.0078125.
        assert!((alibi_slope(7, 8).unwrap() - 0.0078125).abs() < 1e-7);
    }
    #[test] fn bias_self_position_zero() {
        let m = 0.5_f32;
        assert!(alibi_bias(5, 5, m).unwrap() == 0.0);
    }
    #[test] fn bias_distance_negative() {
        let m = 0.5_f32;
        assert!((alibi_bias(0, 5, m).unwrap() - (-2.5)).abs() < 1e-7);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert_eq!(AC_AL_005_ULP_TOLERANCE, 8);
    }
}
