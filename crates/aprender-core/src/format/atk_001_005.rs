// SHIP-TWO-001 — `attention-kernel-v1` algorithm-level PARTIAL
// discharge for FALSIFY-ATT-001..005 (closes 5/5 sweep).
//
// Contract: `contracts/attention-kernel-v1.yaml`.
// Spec: Scaled dot-product attention kernel (Vaswani et al. 2017).

// ===========================================================================
// ATT-001 — Row normalization: Σ_j softmax(QK^T/√d_k)_{ij} == 1 (tol 1e-5)
// ===========================================================================

pub const AC_ATT_001_TOLERANCE: f32 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Att001Verdict { Pass, Fail }

/// Pass iff every row of `weights` (shape n × m, row-major) sums to 1.0
/// within `AC_ATT_001_TOLERANCE`.
#[must_use]
pub fn verdict_from_row_normalization(weights: &[f32], n: usize, m: usize) -> Att001Verdict {
    if weights.is_empty() || n == 0 || m == 0 { return Att001Verdict::Fail; }
    if weights.len() != n * m { return Att001Verdict::Fail; }
    for row in 0..n {
        let mut sum = 0.0_f32;
        for col in 0..m {
            let v = weights[row * m + col];
            if !v.is_finite() { return Att001Verdict::Fail; }
            sum += v;
        }
        if (sum - 1.0).abs() > AC_ATT_001_TOLERANCE { return Att001Verdict::Fail; }
    }
    Att001Verdict::Pass
}

// ===========================================================================
// ATT-002 — Output convexity: min(V) ≤ output_{ij} ≤ max(V) per column j
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Att002Verdict { Pass, Fail }

/// V is shape (m × d_v) row-major; output is shape (n × d_v) row-major.
/// Per column j of V, the output's column j must lie in
/// `[min(V[:, j]), max(V[:, j])]` (convex combination of V rows).
#[must_use]
pub fn verdict_from_output_convexity(
    v: &[f32],
    output: &[f32],
    m: usize,
    n: usize,
    d_v: usize,
) -> Att002Verdict {
    if v.is_empty() || output.is_empty() { return Att002Verdict::Fail; }
    if v.len() != m * d_v || output.len() != n * d_v { return Att002Verdict::Fail; }
    if m == 0 || n == 0 || d_v == 0 { return Att002Verdict::Fail; }
    for col in 0..d_v {
        let mut col_min = f32::INFINITY;
        let mut col_max = f32::NEG_INFINITY;
        for row in 0..m {
            let val = v[row * d_v + col];
            if !val.is_finite() { return Att002Verdict::Fail; }
            if val < col_min { col_min = val; }
            if val > col_max { col_max = val; }
        }
        for row in 0..n {
            let val = output[row * d_v + col];
            if !val.is_finite() { return Att002Verdict::Fail; }
            // Use a small slack for rounding (1 ULP at f32 magnitudes).
            let slack = (col_max - col_min).abs() * 1.0e-5 + 1.0e-7;
            if val < col_min - slack || val > col_max + slack {
                return Att002Verdict::Fail;
            }
        }
    }
    Att002Verdict::Pass
}

// ===========================================================================
// ATT-003 — Scaling factor: uses 1/√d_k (not 1/d_k)
// ===========================================================================

pub const AC_ATT_003_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Att003Verdict { Pass, Fail }

/// Pass iff `observed_scale ≈ 1.0 / sqrt(d_k)` AND clearly distinguishable
/// from `1.0 / d_k` (which is the canonical regression class).
#[must_use]
pub fn verdict_from_scaling_factor(d_k: u64, observed_scale: f32) -> Att003Verdict {
    if d_k == 0 || !observed_scale.is_finite() { return Att003Verdict::Fail; }
    let expected_sqrt = 1.0 / (d_k as f32).sqrt();
    let wrong_dk = 1.0 / (d_k as f32);
    if (observed_scale - expected_sqrt).abs() > AC_ATT_003_TOLERANCE { return Att003Verdict::Fail; }
    // Sanity: when d_k > 1, sqrt scale is strictly larger than 1/d_k —
    // the verdict should reject the wrong-form scale outright.
    if d_k > 1 && (observed_scale - wrong_dk).abs() < AC_ATT_003_TOLERANCE {
        return Att003Verdict::Fail;
    }
    Att003Verdict::Pass
}

// ===========================================================================
// ATT-004 — SIMD parity within 8 ULP
// ===========================================================================

pub const AC_ATT_004_ULP_TOLERANCE: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Att004Verdict { Pass, Fail }

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
pub fn verdict_from_simd_parity(scalar: &[f32], simd: &[f32]) -> Att004Verdict {
    if scalar.is_empty() || simd.is_empty() { return Att004Verdict::Fail; }
    if scalar.len() != simd.len() { return Att004Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if !s.is_finite() || !v.is_finite() { return Att004Verdict::Fail; }
        if ulp_distance(s, v) > AC_ATT_004_ULP_TOLERANCE { return Att004Verdict::Fail; }
    }
    Att004Verdict::Pass
}

// ===========================================================================
// ATT-005 — Attention weights strictly in (0, 1)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Att005Verdict { Pass, Fail }

/// Pass iff every weight is strictly > 0 AND strictly < 1 (open interval).
/// In f32, softmax saturates at extreme inputs; the verdict catches
/// saturation (any 0.0 or 1.0 = Fail) and any out-of-range (≤ 0 or ≥ 1).
#[must_use]
pub fn verdict_from_weights_bounded(weights: &[f32]) -> Att005Verdict {
    if weights.is_empty() { return Att005Verdict::Fail; }
    for &w in weights {
        if !w.is_finite() { return Att005Verdict::Fail; }
        if w <= 0.0 || w >= 1.0 { return Att005Verdict::Fail; }
    }
    Att005Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // ATT-001 (row normalization)
    #[test] fn att001_pass_canonical_2x3() {
        // Two rows that each sum to 1.0
        let w = vec![
            0.2, 0.3, 0.5,
            0.1, 0.6, 0.3,
        ];
        assert_eq!(verdict_from_row_normalization(&w, 2, 3), Att001Verdict::Pass);
    }
    #[test] fn att001_pass_within_tolerance() {
        let w = vec![0.4, 0.4, 0.2 + 1e-7]; // 1.0000001 — within tol
        assert_eq!(verdict_from_row_normalization(&w, 1, 3), Att001Verdict::Pass);
    }
    #[test] fn att001_fail_row_undersum() {
        // The contract's stated falsifier: "softmax not applied row-wise".
        let w = vec![0.2, 0.3, 0.4]; // sums to 0.9
        assert_eq!(verdict_from_row_normalization(&w, 1, 3), Att001Verdict::Fail);
    }
    #[test] fn att001_fail_row_oversum() {
        let w = vec![0.5, 0.5, 0.5]; // sums to 1.5
        assert_eq!(verdict_from_row_normalization(&w, 1, 3), Att001Verdict::Fail);
    }
    #[test] fn att001_fail_one_bad_row() {
        let w = vec![
            0.2, 0.3, 0.5,
            0.5, 0.6, 0.3, // sums to 1.4
        ];
        assert_eq!(verdict_from_row_normalization(&w, 2, 3), Att001Verdict::Fail);
    }
    #[test] fn att001_fail_nan() {
        let w = vec![0.5, f32::NAN, 0.5];
        assert_eq!(verdict_from_row_normalization(&w, 1, 3), Att001Verdict::Fail);
    }

    // ATT-002 (output convexity)
    #[test] fn att002_pass_within_v_range() {
        // V has rows [1.0, 0.0], [3.0, 4.0]. Output is convex combination.
        let v = vec![1.0_f32, 0.0, 3.0, 4.0];
        let output = vec![2.0_f32, 2.0]; // 0.5 * V[0] + 0.5 * V[1] = [2, 2]
        assert_eq!(verdict_from_output_convexity(&v, &output, 2, 1, 2), Att002Verdict::Pass);
    }
    #[test] fn att002_fail_out_of_range() {
        // Output [10, 10] is way above max(V).
        let v = vec![1.0_f32, 0.0, 3.0, 4.0];
        let output = vec![10.0_f32, 10.0];
        assert_eq!(verdict_from_output_convexity(&v, &output, 2, 1, 2), Att002Verdict::Fail);
    }
    #[test] fn att002_fail_dim_mismatch() {
        let v = vec![1.0_f32, 2.0];
        let output = vec![1.0_f32, 2.0, 3.0]; // wrong size
        assert_eq!(verdict_from_output_convexity(&v, &output, 2, 1, 2), Att002Verdict::Fail);
    }

    // ATT-003 (scaling factor √d_k)
    #[test] fn att003_pass_d_k_64() {
        // 1/√64 = 0.125
        assert_eq!(verdict_from_scaling_factor(64, 0.125), Att003Verdict::Pass);
    }
    #[test] fn att003_pass_d_k_128() {
        // 1/√128 ≈ 0.08838834
        let scale = 1.0 / (128.0_f32).sqrt();
        assert_eq!(verdict_from_scaling_factor(128, scale), Att003Verdict::Pass);
    }
    #[test] fn att003_fail_uses_1_over_d_k() {
        // The contract's stated falsifier: "Use 1/d_k instead of 1/√d_k".
        // For d_k=64, wrong = 1/64 = 0.015625 (not 0.125).
        assert_eq!(verdict_from_scaling_factor(64, 0.015625), Att003Verdict::Fail);
    }
    #[test] fn att003_fail_zero_d_k() {
        assert_eq!(verdict_from_scaling_factor(0, 1.0), Att003Verdict::Fail);
    }
    #[test] fn att003_fail_nan_scale() {
        assert_eq!(verdict_from_scaling_factor(64, f32::NAN), Att003Verdict::Fail);
    }
    #[test] fn att003_pass_d_k_1_edge() {
        // d_k=1: 1/√1 == 1/1 == 1.0, the two forms coincide. Verdict accepts.
        assert_eq!(verdict_from_scaling_factor(1, 1.0), Att003Verdict::Pass);
    }

    // ATT-004 (SIMD parity)
    #[test] fn att004_pass_identical() {
        let a = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_simd_parity(&a, &a), Att004Verdict::Pass);
    }
    #[test] fn att004_pass_within_ulp() {
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 4)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Att004Verdict::Pass);
    }
    #[test] fn att004_fail_above_8_ulp() {
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 100)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Att004Verdict::Fail);
    }
    #[test] fn att004_fail_length() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &b), Att004Verdict::Fail);
    }

    // ATT-005 (weights strictly in (0, 1))
    #[test] fn att005_pass_canonical() {
        let w = vec![0.1_f32, 0.5, 0.4, 0.99];
        assert_eq!(verdict_from_weights_bounded(&w), Att005Verdict::Pass);
    }
    #[test] fn att005_fail_zero_weight() {
        // Saturated softmax (one mass=0).
        let w = vec![0.0_f32, 1.0]; // 0.0 violates strict >0
        assert_eq!(verdict_from_weights_bounded(&w), Att005Verdict::Fail);
    }
    #[test] fn att005_fail_one_weight() {
        let w = vec![1.0_f32, 0.0]; // 1.0 violates strict <1
        assert_eq!(verdict_from_weights_bounded(&w), Att005Verdict::Fail);
    }
    #[test] fn att005_fail_negative() {
        let w = vec![-0.1_f32, 1.1];
        assert_eq!(verdict_from_weights_bounded(&w), Att005Verdict::Fail);
    }
    #[test] fn att005_fail_nan() {
        let w = vec![f32::NAN];
        assert_eq!(verdict_from_weights_bounded(&w), Att005Verdict::Fail);
    }
    #[test] fn att005_fail_empty() {
        assert_eq!(verdict_from_weights_bounded(&[]), Att005Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_ATT_001_TOLERANCE - 1e-5).abs() < 1e-12);
        assert!((AC_ATT_003_TOLERANCE - 1e-6).abs() < 1e-12);
        assert_eq!(AC_ATT_004_ULP_TOLERANCE, 8);
    }
}
