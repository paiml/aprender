// SHIP-TWO-001 — `matmul-kernel-v1` algorithm-level PARTIAL discharge
// for FALSIFY-MM-001..005 (closes 5/5 sweep).
//
// Contract: `contracts/matmul-kernel-v1.yaml`.
// Spec: Matrix multiplication kernel — general and quantized variants
// (Goto & van de Geijn 2008; Dettmers et al. 2022 LLM.int8).

// ===========================================================================
// MM-001 — Output shape correctness: matmul(A[m,p], B[p,n]) has shape [m,n]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mm001Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_output_shape(
    m: u64,
    p: u64,
    n: u64,
    observed_rows: u64,
    observed_cols: u64,
) -> Mm001Verdict {
    if m == 0 || p == 0 || n == 0 { return Mm001Verdict::Fail; }
    if observed_rows == m && observed_cols == n { Mm001Verdict::Pass } else { Mm001Verdict::Fail }
}

// ===========================================================================
// MM-002 — Numerical accuracy: |matmul(A,B) - reference| < 1e-4
// ===========================================================================

pub const AC_MM_002_TOLERANCE: f32 = 1.0e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mm002Verdict { Pass, Fail }

/// Pure scalar reference matmul, row-major, no fancy ordering.
/// `a` is m×p, `b` is p×n, `c` is m×n; all row-major.
#[must_use]
pub fn matmul_reference(a: &[f32], b: &[f32], m: usize, p: usize, n: usize) -> Vec<f32> {
    if a.len() != m * p || b.len() != p * n { return vec![]; }
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for k in 0..p {
                acc += a[i * p + k] * b[k * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[must_use]
pub fn verdict_from_numerical_accuracy(observed: &[f32], reference: &[f32]) -> Mm002Verdict {
    if observed.is_empty() || reference.is_empty() { return Mm002Verdict::Fail; }
    if observed.len() != reference.len() { return Mm002Verdict::Fail; }
    for (&a, &b) in observed.iter().zip(reference.iter()) {
        if !a.is_finite() || !b.is_finite() { return Mm002Verdict::Fail; }
        if (a - b).abs() > AC_MM_002_TOLERANCE { return Mm002Verdict::Fail; }
    }
    Mm002Verdict::Pass
}

// ===========================================================================
// MM-003 — SIMD parity within 4 ULP (stricter than 8-ULP general kernel band)
// ===========================================================================

pub const AC_MM_003_ULP_TOLERANCE: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mm003Verdict { Pass, Fail }

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
pub fn verdict_from_simd_parity(scalar: &[f32], simd: &[f32]) -> Mm003Verdict {
    if scalar.is_empty() || simd.is_empty() { return Mm003Verdict::Fail; }
    if scalar.len() != simd.len() { return Mm003Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if !s.is_finite() || !v.is_finite() { return Mm003Verdict::Fail; }
        if ulp_distance(s, v) > AC_MM_003_ULP_TOLERANCE { return Mm003Verdict::Fail; }
    }
    Mm003Verdict::Pass
}

// ===========================================================================
// MM-004 — Quantized accuracy: |q_dot - float_dot| ≤ quant_bound
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mm004Verdict { Pass, Fail }

/// Reference quantized dot product: q_dot(a, b, s_a, s_b) = s_a * s_b * Σ a_k * b_k
#[must_use]
pub fn quantized_dot(a: &[i8], b: &[i8], scale_a: f32, scale_b: f32) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() { return 0.0; }
    if !scale_a.is_finite() || !scale_b.is_finite() { return f32::NAN; }
    let mut acc: i64 = 0;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        acc += (ai as i64) * (bi as i64);
    }
    scale_a * scale_b * (acc as f32)
}

/// Float dot of dequantized vectors.
#[must_use]
pub fn float_dot(a: &[i8], b: &[i8], scale_a: f32, scale_b: f32) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() { return 0.0; }
    let mut acc = 0.0_f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        acc += (ai as f32 * scale_a) * (bi as f32 * scale_b);
    }
    acc
}

/// Pass iff `|q_dot - float_dot| ≤ user-supplied bound`.
#[must_use]
pub fn verdict_from_quantized_accuracy(
    a: &[i8],
    b: &[i8],
    scale_a: f32,
    scale_b: f32,
    bound: f32,
) -> Mm004Verdict {
    if a.is_empty() || b.is_empty() || a.len() != b.len() { return Mm004Verdict::Fail; }
    if !scale_a.is_finite() || scale_a <= 0.0 { return Mm004Verdict::Fail; }
    if !scale_b.is_finite() || scale_b <= 0.0 { return Mm004Verdict::Fail; }
    if !bound.is_finite() || bound < 0.0 { return Mm004Verdict::Fail; }
    let q = quantized_dot(a, b, scale_a, scale_b);
    let f = float_dot(a, b, scale_a, scale_b);
    if !q.is_finite() || !f.is_finite() { return Mm004Verdict::Fail; }
    if (q - f).abs() <= bound { Mm004Verdict::Pass } else { Mm004Verdict::Fail }
}

// ===========================================================================
// MM-005 — Identity matrix: matmul(A, I) = A AND matmul(I, B) = B
// ===========================================================================

pub const AC_MM_005_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mm005Verdict { Pass, Fail }

#[must_use]
pub fn make_identity(n: usize) -> Vec<f32> {
    let mut i = vec![0.0_f32; n * n];
    for k in 0..n {
        i[k * n + k] = 1.0;
    }
    i
}

/// Pass iff matmul(A, I_n) == A AND matmul(I_m, A) == A within tolerance.
/// `a` is m×n row-major.
#[must_use]
pub fn verdict_from_identity_preservation(a: &[f32], m: usize, n: usize) -> Mm005Verdict {
    if a.is_empty() || m == 0 || n == 0 { return Mm005Verdict::Fail; }
    if a.len() != m * n { return Mm005Verdict::Fail; }
    if !a.iter().all(|v| v.is_finite()) { return Mm005Verdict::Fail; }
    // Right multiply: A * I_n → should equal A.
    let i_n = make_identity(n);
    let ai = matmul_reference(a, &i_n, m, n, n);
    if ai.len() != a.len() { return Mm005Verdict::Fail; }
    for (&x, &y) in a.iter().zip(ai.iter()) {
        if (x - y).abs() > AC_MM_005_TOLERANCE { return Mm005Verdict::Fail; }
    }
    // Left multiply: I_m * A → should equal A.
    let i_m = make_identity(m);
    let ia = matmul_reference(&i_m, a, m, m, n);
    if ia.len() != a.len() { return Mm005Verdict::Fail; }
    for (&x, &y) in a.iter().zip(ia.iter()) {
        if (x - y).abs() > AC_MM_005_TOLERANCE { return Mm005Verdict::Fail; }
    }
    Mm005Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // MM-001 (shape)
    #[test] fn mm001_pass_canonical() {
        // A[3, 4] × B[4, 5] = C[3, 5]
        assert_eq!(verdict_from_output_shape(3, 4, 5, 3, 5), Mm001Verdict::Pass);
    }
    #[test] fn mm001_fail_swapped() {
        // The contract's stated falsifier: "Swap row/column indices".
        assert_eq!(verdict_from_output_shape(3, 4, 5, 5, 3), Mm001Verdict::Fail);
    }
    #[test] fn mm001_fail_zero() {
        assert_eq!(verdict_from_output_shape(0, 4, 5, 0, 5), Mm001Verdict::Fail);
        assert_eq!(verdict_from_output_shape(3, 0, 5, 3, 5), Mm001Verdict::Fail);
        assert_eq!(verdict_from_output_shape(3, 4, 0, 3, 0), Mm001Verdict::Fail);
    }

    // MM-002 (numerical accuracy)
    #[test] fn mm002_pass_canonical() {
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        let c_ref = matmul_reference(&a, &b, 2, 2, 2);
        assert_eq!(c_ref, vec![19.0_f32, 22.0, 43.0, 50.0]);
        let observed = vec![19.0_f32, 22.0, 43.0, 50.0];
        assert_eq!(verdict_from_numerical_accuracy(&observed, &c_ref), Mm002Verdict::Pass);
    }
    #[test] fn mm002_pass_within_tolerance() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32 + 1e-6]; // < 1e-4
        assert_eq!(verdict_from_numerical_accuracy(&a, &b), Mm002Verdict::Pass);
    }
    #[test] fn mm002_fail_above_tolerance() {
        let a = vec![1.0_f32];
        let b = vec![1.001_f32]; // > 1e-4
        assert_eq!(verdict_from_numerical_accuracy(&a, &b), Mm002Verdict::Fail);
    }
    #[test] fn mm002_fail_length_mismatch() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_numerical_accuracy(&a, &b), Mm002Verdict::Fail);
    }

    // MM-003 (SIMD parity, 4 ULP)
    #[test] fn mm003_pass_identical() {
        let a = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &a), Mm003Verdict::Pass);
    }
    #[test] fn mm003_pass_within_4_ulp() {
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 3)]; // 3 ULP < 4
        assert_eq!(verdict_from_simd_parity(&a, &b), Mm003Verdict::Pass);
    }
    #[test] fn mm003_fail_above_4_ulp() {
        // 5 ULP fails (matmul band is stricter than the kernel-default 8 ULP).
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 5)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Mm003Verdict::Fail);
    }
    #[test] fn mm003_fail_length() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &b), Mm003Verdict::Fail);
    }

    // MM-004 (quantized accuracy)
    #[test] fn mm004_pass_canonical() {
        // a=[1,2], b=[3,4], s_a=0.1, s_b=0.05.
        // q_dot = 0.1*0.05*(1*3+2*4) = 0.005*11 = 0.055
        // float_dot = (1*0.1)*(3*0.05) + (2*0.1)*(4*0.05) = 0.015 + 0.04 = 0.055
        // Both should be exactly equal here (no rounding error at this scale).
        let a = vec![1_i8, 2];
        let b = vec![3_i8, 4];
        assert_eq!(verdict_from_quantized_accuracy(&a, &b, 0.1, 0.05, 1e-3), Mm004Verdict::Pass);
    }
    #[test] fn mm004_fail_zero_scale() {
        let a = vec![1_i8];
        let b = vec![1_i8];
        assert_eq!(verdict_from_quantized_accuracy(&a, &b, 0.0, 0.05, 1e-3), Mm004Verdict::Fail);
    }
    #[test] fn mm004_fail_negative_bound() {
        let a = vec![1_i8];
        let b = vec![1_i8];
        assert_eq!(verdict_from_quantized_accuracy(&a, &b, 0.1, 0.05, -1e-3), Mm004Verdict::Fail);
    }
    #[test] fn mm004_fail_length_mismatch() {
        let a = vec![1_i8, 2];
        let b = vec![1_i8];
        assert_eq!(verdict_from_quantized_accuracy(&a, &b, 0.1, 0.05, 1e-3), Mm004Verdict::Fail);
    }

    // MM-005 (identity)
    #[test] fn mm005_pass_2x3() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(verdict_from_identity_preservation(&a, 2, 3), Mm005Verdict::Pass);
    }
    #[test] fn mm005_pass_random() {
        // 4x4 matrix.
        let a: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.5).collect();
        assert_eq!(verdict_from_identity_preservation(&a, 4, 4), Mm005Verdict::Pass);
    }
    #[test] fn mm005_fail_dim_mismatch() {
        let a = vec![1.0_f32; 5]; // m*n = 6 but a.len() = 5
        assert_eq!(verdict_from_identity_preservation(&a, 2, 3), Mm005Verdict::Fail);
    }
    #[test] fn mm005_fail_zero_dim() {
        let a = vec![1.0_f32];
        assert_eq!(verdict_from_identity_preservation(&a, 0, 1), Mm005Verdict::Fail);
    }
    #[test] fn mm005_fail_nan() {
        let a = vec![1.0_f32, f32::NAN];
        assert_eq!(verdict_from_identity_preservation(&a, 1, 2), Mm005Verdict::Fail);
    }

    // matmul_reference helper sanity
    #[test] fn matmul_2x2_canonical() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        assert_eq!(matmul_reference(&a, &b, 2, 2, 2), vec![19.0_f32, 22.0, 43.0, 50.0]);
    }

    // make_identity sanity
    #[test] fn identity_3() {
        assert_eq!(make_identity(3), vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_MM_002_TOLERANCE - 1e-4).abs() < 1e-9);
        assert_eq!(AC_MM_003_ULP_TOLERANCE, 4); // matmul has stricter SIMD band
        assert!((AC_MM_005_TOLERANCE - 1e-6).abs() < 1e-12);
    }
}
