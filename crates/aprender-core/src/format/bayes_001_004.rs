// SHIP-TWO-001 — `bias-add-v1` algorithm-level PARTIAL discharge for
// FALSIFY-BA-001..004 (closes 4/4 sweep).
//
// Contract: `contracts/bias-add-v1.yaml`.
// Spec: Bias addition kernel — broadcast bias vector over batch.

// ===========================================================================
// Helper — bias_add reference (in-module)
// ===========================================================================

/// y[b, i] = x[b, i] + bias[i] for x ∈ R^{B × D}, bias ∈ R^D.
/// Row-major flattened input: x.len() == B * D, bias.len() == D.
#[must_use]
pub fn bias_add(x: &[f32], bias: &[f32], batch: usize, d: usize) -> Option<Vec<f32>> {
    if x.len() != batch * d { return None; }
    if bias.len() != d { return None; }
    if !x.iter().all(|v| v.is_finite()) { return None; }
    if !bias.iter().all(|v| v.is_finite()) { return None; }
    let mut y = vec![0.0_f32; batch * d];
    for b in 0..batch {
        for i in 0..d {
            y[b * d + i] = x[b * d + i] + bias[i];
        }
    }
    Some(y)
}

// ===========================================================================
// BA-001 — Shape preservation: shape(y) == shape(x)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ba001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_shape_preservation(
    x: &[f32],
    bias: &[f32],
    batch: usize,
    d: usize,
) -> Ba001Verdict {
    match bias_add(x, bias, batch, d) {
        Some(y) if y.len() == x.len() => Ba001Verdict::Pass,
        _ => Ba001Verdict::Fail,
    }
}

// ===========================================================================
// BA-002 — Zero-bias identity: bias_add(x, 0) == x byte-exactly
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ba002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_zero_bias_identity(x: &[f32], batch: usize, d: usize) -> Ba002Verdict {
    if x.is_empty() { return Ba002Verdict::Fail; }
    let zero_bias = vec![0.0_f32; d];
    let y = match bias_add(x, &zero_bias, batch, d) {
        Some(v) => v,
        None => return Ba002Verdict::Fail,
    };
    if y.len() != x.len() { return Ba002Verdict::Fail; }
    for (a, b) in x.iter().zip(y.iter()) {
        if a.to_bits() != b.to_bits() { return Ba002Verdict::Fail; }
    }
    Ba002Verdict::Pass
}

// ===========================================================================
// BA-003 — Additivity: bias_add(bias_add(x, b1), b2) ≈ bias_add(x, b1 + b2)
// ===========================================================================

pub const AC_BA_003_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ba003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_additivity(
    x: &[f32],
    b1: &[f32],
    b2: &[f32],
    batch: usize,
    d: usize,
) -> Ba003Verdict {
    if b1.len() != d || b2.len() != d { return Ba003Verdict::Fail; }
    // Path 1: x → +b1 → +b2.
    let after_b1 = match bias_add(x, b1, batch, d) {
        Some(v) => v,
        None => return Ba003Verdict::Fail,
    };
    let path1 = match bias_add(&after_b1, b2, batch, d) {
        Some(v) => v,
        None => return Ba003Verdict::Fail,
    };
    // Path 2: x → +(b1 + b2).
    let combined: Vec<f32> = b1.iter().zip(b2.iter()).map(|(&a, &c)| a + c).collect();
    let path2 = match bias_add(x, &combined, batch, d) {
        Some(v) => v,
        None => return Ba003Verdict::Fail,
    };
    if path1.len() != path2.len() { return Ba003Verdict::Fail; }
    for (a, b) in path1.iter().zip(path2.iter()) {
        if !a.is_finite() || !b.is_finite() { return Ba003Verdict::Fail; }
        if (a - b).abs() > AC_BA_003_TOLERANCE { return Ba003Verdict::Fail; }
    }
    Ba003Verdict::Pass
}

// ===========================================================================
// BA-004 — SIMD parity: scalar == SIMD byte-exactly (contract tolerance=0.0)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ba004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_simd_parity(scalar: &[f32], simd: &[f32]) -> Ba004Verdict {
    if scalar.is_empty() || simd.is_empty() { return Ba004Verdict::Fail; }
    if scalar.len() != simd.len() { return Ba004Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if !s.is_finite() || !v.is_finite() { return Ba004Verdict::Fail; }
        if s.to_bits() != v.to_bits() { return Ba004Verdict::Fail; }
    }
    Ba004Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // BA-001 (shape preservation)
    #[test] fn ba001_pass_canonical() {
        // batch=2, d=3.
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = vec![0.1_f32, 0.2, 0.3];
        assert_eq!(verdict_from_shape_preservation(&x, &bias, 2, 3), Ba001Verdict::Pass);
    }
    #[test] fn ba001_fail_dim_mismatch() {
        let x = vec![1.0_f32, 2.0, 3.0]; // batch * d = 6 ≠ 3
        let bias = vec![0.1_f32, 0.2, 0.3];
        assert_eq!(verdict_from_shape_preservation(&x, &bias, 2, 3), Ba001Verdict::Fail);
    }
    #[test] fn ba001_fail_bias_dim_mismatch() {
        let x = vec![1.0_f32; 6];
        let bias = vec![0.1_f32, 0.2]; // wrong size
        assert_eq!(verdict_from_shape_preservation(&x, &bias, 2, 3), Ba001Verdict::Fail);
    }
    #[test] fn ba001_fail_nan() {
        let x = vec![1.0_f32, f32::NAN, 3.0];
        let bias = vec![0.1_f32, 0.2, 0.3];
        assert_eq!(verdict_from_shape_preservation(&x, &bias, 1, 3), Ba001Verdict::Fail);
    }

    // BA-002 (zero-bias identity)
    #[test] fn ba002_pass_canonical() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(verdict_from_zero_bias_identity(&x, 2, 3), Ba002Verdict::Pass);
    }
    #[test] fn ba002_pass_random_x() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.0).collect();
        // batch=4, d=64.
        assert_eq!(verdict_from_zero_bias_identity(&x, 4, 64), Ba002Verdict::Pass);
    }
    #[test] fn ba002_fail_dim_mismatch() {
        let x = vec![1.0_f32, 2.0]; // batch * d = 6 ≠ 2
        assert_eq!(verdict_from_zero_bias_identity(&x, 2, 3), Ba002Verdict::Fail);
    }
    #[test] fn ba002_fail_empty() {
        assert_eq!(verdict_from_zero_bias_identity(&[], 0, 0), Ba002Verdict::Fail);
    }

    // BA-003 (additivity)
    #[test] fn ba003_pass_canonical() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b1 = vec![0.1_f32, 0.2];
        let b2 = vec![0.3_f32, 0.4];
        assert_eq!(verdict_from_additivity(&x, &b1, &b2, 2, 2), Ba003Verdict::Pass);
    }
    #[test] fn ba003_pass_zero_biases() {
        let x = vec![1.0_f32, 2.0];
        let b1 = vec![0.0_f32, 0.0];
        let b2 = vec![0.0_f32, 0.0];
        assert_eq!(verdict_from_additivity(&x, &b1, &b2, 1, 2), Ba003Verdict::Pass);
    }
    #[test] fn ba003_fail_b1_wrong_size() {
        let x = vec![1.0_f32, 2.0];
        let b1 = vec![0.1_f32]; // d=2 but b1 has 1
        let b2 = vec![0.3_f32, 0.4];
        assert_eq!(verdict_from_additivity(&x, &b1, &b2, 1, 2), Ba003Verdict::Fail);
    }
    #[test] fn ba003_fail_b2_wrong_size() {
        let x = vec![1.0_f32, 2.0];
        let b1 = vec![0.1_f32, 0.2];
        let b2 = vec![0.3_f32]; // d=2 but b2 has 1
        assert_eq!(verdict_from_additivity(&x, &b1, &b2, 1, 2), Ba003Verdict::Fail);
    }

    // BA-004 (SIMD parity)
    #[test] fn ba004_pass_identical() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_simd_parity(&v, &v), Ba004Verdict::Pass);
    }
    #[test] fn ba004_fail_one_ulp() {
        // Contract tolerance=0.0 — even 1 ULP fails.
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 1)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Ba004Verdict::Fail);
    }
    #[test] fn ba004_fail_length() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &b), Ba004Verdict::Fail);
    }
    #[test] fn ba004_fail_nan() {
        let a = vec![f32::NAN];
        let b = vec![f32::NAN];
        assert_eq!(verdict_from_simd_parity(&a, &b), Ba004Verdict::Fail);
    }

    // Helper sanity
    #[test] fn bias_add_canonical() {
        // [[1, 2], [3, 4]] + [10, 20] = [[11, 22], [13, 24]]
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let bias = vec![10.0_f32, 20.0];
        let y = bias_add(&x, &bias, 2, 2).unwrap();
        assert_eq!(y, vec![11.0_f32, 22.0, 13.0, 24.0]);
    }
    #[test] fn bias_add_broadcasts() {
        // Bias [b0, b1] applied to ALL rows.
        let x = vec![1.0_f32; 6];
        let bias = vec![10.0_f32, 20.0, 30.0];
        let y = bias_add(&x, &bias, 2, 3).unwrap();
        // Row 0: [11, 21, 31]; row 1: [11, 21, 31].
        assert_eq!(y, vec![11.0_f32, 21.0, 31.0, 11.0, 21.0, 31.0]);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_BA_003_TOLERANCE - 1e-6).abs() < 1e-12);
    }
}
