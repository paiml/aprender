// SHIP-TWO-001 — `linear-projection-v1` algorithm-level PARTIAL
// discharge for FALSIFY-LP-001..005 (closes 5/5 sweep).
//
// Contract: `contracts/linear-projection-v1.yaml`.
// Spec: Linear projection — y = x @ W^T + b (Bishop 2006 PRML).

// ===========================================================================
// Helper — reference linear forward (in-module, no external deps)
// ===========================================================================

/// Reference linear-no-bias: y = x @ W^T where x is (batch, d_in) row-major
/// and W is (d_out, d_in) row-major. Output is (batch, d_out) row-major.
#[must_use]
pub fn linear_no_bias(x: &[f32], w: &[f32], batch: usize, d_in: usize, d_out: usize) -> Vec<f32> {
    if x.len() != batch * d_in || w.len() != d_out * d_in { return vec![]; }
    let mut y = vec![0.0_f32; batch * d_out];
    for i in 0..batch {
        for k in 0..d_out {
            let mut acc = 0.0_f32;
            for j in 0..d_in {
                acc += x[i * d_in + j] * w[k * d_in + j];
            }
            y[i * d_out + k] = acc;
        }
    }
    y
}

#[must_use]
pub fn linear_forward(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    if b.len() != d_out { return vec![]; }
    let mut y = linear_no_bias(x, w, batch, d_in, d_out);
    if y.is_empty() { return y; }
    for i in 0..batch {
        for k in 0..d_out {
            y[i * d_out + k] += b[k];
        }
    }
    y
}

// ===========================================================================
// LP-001 — Output shape: y.shape == (batch, d_out)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lp001Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_output_shape(
    batch: u64,
    d_in: u64,
    d_out: u64,
    observed_len: u64,
) -> Lp001Verdict {
    if batch == 0 || d_in == 0 || d_out == 0 { return Lp001Verdict::Fail; }
    if observed_len == batch * d_out { Lp001Verdict::Pass } else { Lp001Verdict::Fail }
}

// ===========================================================================
// LP-002 — Homogeneity without bias: f(α·x) ≈ α·f(x) within 4 ULP
// ===========================================================================

pub const AC_LP_002_ULP_TOLERANCE: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lp002Verdict { Pass, Fail }

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

/// Pass iff `linear_no_bias(α·x, W) == α·linear_no_bias(x, W)` within 4 ULP.
#[must_use]
pub fn verdict_from_homogeneity(
    x: &[f32],
    w: &[f32],
    alpha: f32,
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Lp002Verdict {
    if !alpha.is_finite() { return Lp002Verdict::Fail; }
    if x.is_empty() || w.is_empty() { return Lp002Verdict::Fail; }
    if !x.iter().all(|v| v.is_finite()) { return Lp002Verdict::Fail; }
    if !w.iter().all(|v| v.is_finite()) { return Lp002Verdict::Fail; }
    let scaled: Vec<f32> = x.iter().map(|&v| alpha * v).collect();
    let lhs = linear_no_bias(&scaled, w, batch, d_in, d_out);
    let base = linear_no_bias(x, w, batch, d_in, d_out);
    if lhs.is_empty() || base.is_empty() || lhs.len() != base.len() { return Lp002Verdict::Fail; }
    for (l, b) in lhs.iter().zip(base.iter()) {
        let rhs = alpha * b;
        if !rhs.is_finite() || !l.is_finite() { return Lp002Verdict::Fail; }
        if ulp_distance(*l, rhs) > AC_LP_002_ULP_TOLERANCE { return Lp002Verdict::Fail; }
    }
    Lp002Verdict::Pass
}

// ===========================================================================
// LP-003 — Bias additivity: linear_forward(x, W, b) - linear_no_bias(x, W) == b
// ===========================================================================

pub const AC_LP_003_TOLERANCE: f32 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lp003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_bias_additivity(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Lp003Verdict {
    if x.is_empty() || w.is_empty() || b.is_empty() { return Lp003Verdict::Fail; }
    if b.len() != d_out { return Lp003Verdict::Fail; }
    let with = linear_forward(x, w, b, batch, d_in, d_out);
    let without = linear_no_bias(x, w, batch, d_in, d_out);
    if with.is_empty() || without.is_empty() || with.len() != without.len() {
        return Lp003Verdict::Fail;
    }
    for i in 0..batch {
        for k in 0..d_out {
            let delta = with[i * d_out + k] - without[i * d_out + k];
            if !delta.is_finite() { return Lp003Verdict::Fail; }
            if (delta - b[k]).abs() > AC_LP_003_TOLERANCE { return Lp003Verdict::Fail; }
        }
    }
    Lp003Verdict::Pass
}

// ===========================================================================
// LP-004 — Zero input produces bias: linear_forward(0, W, b) == b broadcast
// ===========================================================================

pub const AC_LP_004_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lp004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_zero_input_bias(
    w: &[f32],
    b: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Lp004Verdict {
    if w.is_empty() || b.is_empty() { return Lp004Verdict::Fail; }
    if b.len() != d_out { return Lp004Verdict::Fail; }
    let zero_x = vec![0.0_f32; batch * d_in];
    let y = linear_forward(&zero_x, w, b, batch, d_in, d_out);
    if y.is_empty() || y.len() != batch * d_out { return Lp004Verdict::Fail; }
    for i in 0..batch {
        for k in 0..d_out {
            if (y[i * d_out + k] - b[k]).abs() > AC_LP_004_TOLERANCE {
                return Lp004Verdict::Fail;
            }
        }
    }
    Lp004Verdict::Pass
}

// ===========================================================================
// LP-005 — SIMD parity within 4 ULP (matches MM-003's stricter band)
// ===========================================================================

pub const AC_LP_005_ULP_TOLERANCE: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lp005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_simd_parity(scalar: &[f32], simd: &[f32]) -> Lp005Verdict {
    if scalar.is_empty() || simd.is_empty() { return Lp005Verdict::Fail; }
    if scalar.len() != simd.len() { return Lp005Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if !s.is_finite() || !v.is_finite() { return Lp005Verdict::Fail; }
        if ulp_distance(s, v) > AC_LP_005_ULP_TOLERANCE { return Lp005Verdict::Fail; }
    }
    Lp005Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // LP-001 (output shape)
    #[test] fn lp001_pass_canonical() {
        // batch=4, d_in=8, d_out=16 → output = 64 elements.
        assert_eq!(verdict_from_output_shape(4, 8, 16, 64), Lp001Verdict::Pass);
    }
    #[test] fn lp001_fail_off_by_one() {
        assert_eq!(verdict_from_output_shape(4, 8, 16, 63), Lp001Verdict::Fail);
    }
    #[test] fn lp001_fail_zero() {
        assert_eq!(verdict_from_output_shape(0, 8, 16, 0), Lp001Verdict::Fail);
        assert_eq!(verdict_from_output_shape(4, 0, 16, 0), Lp001Verdict::Fail);
    }

    // LP-002 (homogeneity)
    #[test] fn lp002_pass_canonical() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0]; // batch=2, d_in=2
        let w = vec![0.5_f32, 0.5, 1.0, 1.0]; // d_out=2, d_in=2
        assert_eq!(verdict_from_homogeneity(&x, &w, 2.0, 2, 2, 2), Lp002Verdict::Pass);
    }
    #[test] fn lp002_pass_negative_alpha() {
        let x = vec![1.0_f32, 2.0];
        let w = vec![0.5_f32, 0.5];
        assert_eq!(verdict_from_homogeneity(&x, &w, -3.0, 1, 2, 1), Lp002Verdict::Pass);
    }
    #[test] fn lp002_pass_zero_alpha() {
        // f(0·x) = 0·f(x) = 0
        let x = vec![1.0_f32, 2.0];
        let w = vec![0.5_f32, 0.5];
        assert_eq!(verdict_from_homogeneity(&x, &w, 0.0, 1, 2, 1), Lp002Verdict::Pass);
    }
    #[test] fn lp002_fail_nan_alpha() {
        let x = vec![1.0_f32];
        let w = vec![1.0_f32];
        assert_eq!(verdict_from_homogeneity(&x, &w, f32::NAN, 1, 1, 1), Lp002Verdict::Fail);
    }
    #[test] fn lp002_fail_nan_x() {
        let x = vec![f32::NAN];
        let w = vec![1.0_f32];
        assert_eq!(verdict_from_homogeneity(&x, &w, 2.0, 1, 1, 1), Lp002Verdict::Fail);
    }

    // LP-003 (bias additivity)
    #[test] fn lp003_pass_canonical() {
        let x = vec![1.0_f32, 2.0]; // batch=1, d_in=2
        let w = vec![0.5_f32, 0.5, 1.0, 1.0]; // d_out=2
        let b = vec![0.1_f32, -0.2];
        assert_eq!(verdict_from_bias_additivity(&x, &w, &b, 1, 2, 2), Lp003Verdict::Pass);
    }
    #[test] fn lp003_pass_zero_bias() {
        let x = vec![1.0_f32];
        let w = vec![1.0_f32];
        let b = vec![0.0_f32];
        assert_eq!(verdict_from_bias_additivity(&x, &w, &b, 1, 1, 1), Lp003Verdict::Pass);
    }
    #[test] fn lp003_fail_wrong_b_size() {
        let x = vec![1.0_f32];
        let w = vec![1.0_f32];
        let b = vec![0.1_f32, 0.2]; // d_out=1 but b has 2 entries
        assert_eq!(verdict_from_bias_additivity(&x, &w, &b, 1, 1, 1), Lp003Verdict::Fail);
    }

    // LP-004 (zero input bias)
    #[test] fn lp004_pass_canonical() {
        let w = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![0.5_f32, -0.5];
        assert_eq!(verdict_from_zero_input_bias(&w, &b, 2, 2, 2), Lp004Verdict::Pass);
    }
    #[test] fn lp004_pass_zero_bias() {
        let w = vec![1.0_f32];
        let b = vec![0.0_f32];
        assert_eq!(verdict_from_zero_input_bias(&w, &b, 1, 1, 1), Lp004Verdict::Pass);
    }
    #[test] fn lp004_fail_b_dim_mismatch() {
        let w = vec![1.0_f32];
        let b = vec![0.0_f32, 0.0];
        assert_eq!(verdict_from_zero_input_bias(&w, &b, 1, 1, 1), Lp004Verdict::Fail);
    }
    #[test] fn lp004_fail_empty() {
        let w: Vec<f32> = vec![];
        let b = vec![0.0_f32];
        assert_eq!(verdict_from_zero_input_bias(&w, &b, 1, 1, 1), Lp004Verdict::Fail);
    }

    // LP-005 (SIMD parity, 4 ULP)
    #[test] fn lp005_pass_identical() {
        let a = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &a), Lp005Verdict::Pass);
    }
    #[test] fn lp005_pass_within_4_ulp() {
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 3)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Lp005Verdict::Pass);
    }
    #[test] fn lp005_fail_above_4_ulp() {
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 5)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Lp005Verdict::Fail);
    }
    #[test] fn lp005_fail_length() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &b), Lp005Verdict::Fail);
    }

    // Reference impl sanity
    #[test] fn linear_forward_2x2_canonical() {
        // x = [[1, 2]], W = [[1, 2], [3, 4]], b = [0.5, -0.5]
        // y = x @ W^T + b = [1*1+2*2 + 0.5, 1*3+2*4 - 0.5] = [5.5, 10.5]
        let x = vec![1.0_f32, 2.0];
        let w = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![0.5_f32, -0.5];
        assert_eq!(linear_forward(&x, &w, &b, 1, 2, 2), vec![5.5_f32, 10.5]);
    }
    #[test] fn linear_no_bias_2x2_canonical() {
        let x = vec![1.0_f32, 2.0];
        let w = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(linear_no_bias(&x, &w, 1, 2, 2), vec![5.0_f32, 11.0]);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert_eq!(AC_LP_002_ULP_TOLERANCE, 4);
        assert!((AC_LP_003_TOLERANCE - 1e-5).abs() < 1e-12);
        assert!((AC_LP_004_TOLERANCE - 1e-6).abs() < 1e-12);
        assert_eq!(AC_LP_005_ULP_TOLERANCE, 4);
    }
}
