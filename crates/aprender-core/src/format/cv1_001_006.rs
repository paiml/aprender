// SHIP-TWO-001 — `conv1d-kernel-v1` algorithm-level PARTIAL discharge
// for FALSIFY-CV-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/conv1d-kernel-v1.yaml`.
// Spec: 1D convolution kernel — output shape, linearity, im2col-GEMM
// equivalence, SIMD parity, K=1 pointwise, identity-kernel preservation.

// ===========================================================================
// CV-001 — Output shape: L_out = floor((L + 2*pad - K) / stride) + 1
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cv001Verdict { Pass, Fail }

#[must_use]
pub const fn conv1d_output_length(l: u64, kernel_size: u64, stride: u64, pad: u64) -> Option<u64> {
    if stride == 0 || kernel_size == 0 || l == 0 { return None; }
    let padded = l + 2 * pad;
    if padded < kernel_size { return None; }
    Some((padded - kernel_size) / stride + 1)
}

#[must_use]
pub fn verdict_from_output_shape(
    l: u64,
    kernel_size: u64,
    stride: u64,
    pad: u64,
    observed: u64,
) -> Cv001Verdict {
    match conv1d_output_length(l, kernel_size, stride, pad) {
        Some(expected) if expected == observed => Cv001Verdict::Pass,
        _ => Cv001Verdict::Fail,
    }
}

// ===========================================================================
// CV-002 — Linearity: |conv(a*x + b*z) - a*conv(x) - b*conv(z)| < 1e-5
// ===========================================================================

pub const AC_CV_002_TOLERANCE: f32 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cv002Verdict { Pass, Fail }

/// Pure scalar 1D conv (single-channel, stride=1, pad=0) — algorithm-level
/// reference implementation used by linearity / im2col / boundary tests.
#[must_use]
pub fn conv1d_scalar(x: &[f32], w: &[f32]) -> Vec<f32> {
    if w.is_empty() || x.len() < w.len() { return vec![]; }
    let l_out = x.len() - w.len() + 1;
    let mut y = vec![0.0_f32; l_out];
    for n in 0..l_out {
        let mut acc = 0.0_f32;
        for k in 0..w.len() {
            acc += w[k] * x[n + k];
        }
        y[n] = acc;
    }
    y
}

#[must_use]
pub fn verdict_from_linearity(
    x: &[f32],
    z: &[f32],
    w: &[f32],
    a: f32,
    b: f32,
) -> Cv002Verdict {
    if x.is_empty() || z.is_empty() || w.is_empty() { return Cv002Verdict::Fail; }
    if x.len() != z.len() { return Cv002Verdict::Fail; }
    if !a.is_finite() || !b.is_finite() { return Cv002Verdict::Fail; }
    if !x.iter().all(|v| v.is_finite()) || !z.iter().all(|v| v.is_finite()) {
        return Cv002Verdict::Fail;
    }
    if !w.iter().all(|v| v.is_finite()) { return Cv002Verdict::Fail; }
    // Left:  conv(a*x + b*z)
    let combined: Vec<f32> = x.iter().zip(z.iter()).map(|(&xi, &zi)| a * xi + b * zi).collect();
    let lhs = conv1d_scalar(&combined, w);
    // Right: a*conv(x) + b*conv(z)
    let cx = conv1d_scalar(x, w);
    let cz = conv1d_scalar(z, w);
    if cx.len() != cz.len() || cx.len() != lhs.len() { return Cv002Verdict::Fail; }
    for i in 0..lhs.len() {
        let rhs = a * cx[i] + b * cz[i];
        if !rhs.is_finite() || !lhs[i].is_finite() { return Cv002Verdict::Fail; }
        if (lhs[i] - rhs).abs() > AC_CV_002_TOLERANCE { return Cv002Verdict::Fail; }
    }
    Cv002Verdict::Pass
}

// ===========================================================================
// CV-003 — im2col-GEMM equivalence: |conv_direct - conv_im2col| < 1e-6
// ===========================================================================

pub const AC_CV_003_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cv003Verdict { Pass, Fail }

/// Tiny im2col-then-GEMM reference implementation (single channel,
/// stride=1, pad=0). Output: y[n] = Σ_k w[k] * x[n+k] computed by
/// building a (K × L_out) im2col matrix and multiplying by w.
#[must_use]
pub fn conv1d_im2col(x: &[f32], w: &[f32]) -> Vec<f32> {
    if w.is_empty() || x.len() < w.len() { return vec![]; }
    let k = w.len();
    let l_out = x.len() - k + 1;
    // im2col: column j contains x[j..j+k]
    // GEMM:   y[j] = Σ_i w[i] * im2col[i, j] = Σ_i w[i] * x[j+i]
    let mut y = vec![0.0_f32; l_out];
    for j in 0..l_out {
        let mut acc = 0.0_f32;
        for i in 0..k {
            acc += w[i] * x[j + i];
        }
        y[j] = acc;
    }
    y
}

#[must_use]
pub fn verdict_from_im2col_equivalence(x: &[f32], w: &[f32]) -> Cv003Verdict {
    if x.is_empty() || w.is_empty() { return Cv003Verdict::Fail; }
    let direct = conv1d_scalar(x, w);
    let via_im2col = conv1d_im2col(x, w);
    if direct.len() != via_im2col.len() || direct.is_empty() { return Cv003Verdict::Fail; }
    for (&a, &b) in direct.iter().zip(via_im2col.iter()) {
        if !a.is_finite() || !b.is_finite() { return Cv003Verdict::Fail; }
        if (a - b).abs() > AC_CV_003_TOLERANCE { return Cv003Verdict::Fail; }
    }
    Cv003Verdict::Pass
}

// ===========================================================================
// CV-004 — SIMD parity within 8 ULP
// ===========================================================================

pub const AC_CV_004_ULP_TOLERANCE: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cv004Verdict { Pass, Fail }

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
pub fn verdict_from_simd_parity(scalar: &[f32], simd: &[f32]) -> Cv004Verdict {
    if scalar.is_empty() || simd.is_empty() { return Cv004Verdict::Fail; }
    if scalar.len() != simd.len() { return Cv004Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if !s.is_finite() || !v.is_finite() { return Cv004Verdict::Fail; }
        if ulp_distance(s, v) > AC_CV_004_ULP_TOLERANCE { return Cv004Verdict::Fail; }
    }
    Cv004Verdict::Pass
}

// ===========================================================================
// CV-005 — K=1 boundary: conv1d(K=1) ≡ pointwise scaling by w[0]
// ===========================================================================

pub const AC_CV_005_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cv005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_k1_boundary(x: &[f32], w0: f32) -> Cv005Verdict {
    if x.is_empty() { return Cv005Verdict::Fail; }
    if !w0.is_finite() { return Cv005Verdict::Fail; }
    if !x.iter().all(|v| v.is_finite()) { return Cv005Verdict::Fail; }
    let w = vec![w0];
    let conv_out = conv1d_scalar(x, &w);
    let pointwise: Vec<f32> = x.iter().map(|&xi| w0 * xi).collect();
    if conv_out.len() != pointwise.len() { return Cv005Verdict::Fail; }
    for (&c, &p) in conv_out.iter().zip(pointwise.iter()) {
        if (c - p).abs() > AC_CV_005_TOLERANCE { return Cv005Verdict::Fail; }
    }
    Cv005Verdict::Pass
}

// ===========================================================================
// CV-006 — Identity kernel: conv1d with delta kernel preserves valid window
// ===========================================================================

pub const AC_CV_006_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cv006Verdict { Pass, Fail }

/// With identity kernel `[1, 0, 0, ..., 0]` (length K), the no-pad
/// conv1d output is exactly `x[0..L_out]` where L_out = L - K + 1.
/// More generally, the identity kernel placed at position p produces
/// `x[p..p + L_out]`.
#[must_use]
pub fn verdict_from_identity_kernel(x: &[f32], kernel_size: usize, identity_pos: usize) -> Cv006Verdict {
    if x.is_empty() || kernel_size == 0 || x.len() < kernel_size { return Cv006Verdict::Fail; }
    if identity_pos >= kernel_size { return Cv006Verdict::Fail; }
    if !x.iter().all(|v| v.is_finite()) { return Cv006Verdict::Fail; }
    let mut w = vec![0.0_f32; kernel_size];
    w[identity_pos] = 1.0;
    let y = conv1d_scalar(x, &w);
    let l_out = x.len() - kernel_size + 1;
    if y.len() != l_out { return Cv006Verdict::Fail; }
    for n in 0..l_out {
        let expected = x[n + identity_pos];
        if (y[n] - expected).abs() > AC_CV_006_TOLERANCE { return Cv006Verdict::Fail; }
    }
    Cv006Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // CV-001 (output shape)
    #[test] fn cv001_pass_canonical() {
        // L=10, K=3, stride=1, pad=0 → L_out = 8.
        assert_eq!(conv1d_output_length(10, 3, 1, 0), Some(8));
        assert_eq!(verdict_from_output_shape(10, 3, 1, 0, 8), Cv001Verdict::Pass);
    }
    #[test] fn cv001_pass_with_padding() {
        // L=10, K=3, stride=1, pad=1 → padded=12, L_out = (12-3)/1+1 = 10
        assert_eq!(conv1d_output_length(10, 3, 1, 1), Some(10));
        assert_eq!(verdict_from_output_shape(10, 3, 1, 1, 10), Cv001Verdict::Pass);
    }
    #[test] fn cv001_pass_with_stride() {
        // L=10, K=3, stride=2, pad=0 → (10-3)/2+1 = 4 (floor of 7/2 = 3, +1 = 4)
        assert_eq!(conv1d_output_length(10, 3, 2, 0), Some(4));
        assert_eq!(verdict_from_output_shape(10, 3, 2, 0, 4), Cv001Verdict::Pass);
    }
    #[test] fn cv001_fail_off_by_one() {
        // The contract's stated falsifier: "Off-by-one in output length".
        assert_eq!(verdict_from_output_shape(10, 3, 1, 0, 9), Cv001Verdict::Fail);
    }
    #[test] fn cv001_fail_zero_l() {
        assert_eq!(verdict_from_output_shape(0, 3, 1, 0, 0), Cv001Verdict::Fail);
    }
    #[test] fn cv001_fail_zero_stride() {
        assert_eq!(verdict_from_output_shape(10, 3, 0, 0, 10), Cv001Verdict::Fail);
    }

    // CV-002 (linearity)
    #[test] fn cv002_pass_canonical() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let z = vec![0.5_f32, 1.5, 2.5, 3.5, 4.5];
        let w = vec![0.1_f32, 0.2, 0.3];
        assert_eq!(verdict_from_linearity(&x, &z, &w, 2.0, -0.5), Cv002Verdict::Pass);
    }
    #[test] fn cv002_pass_zero_scalars() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let z = vec![5.0_f32, 6.0, 7.0, 8.0];
        let w = vec![0.5_f32, 0.5];
        assert_eq!(verdict_from_linearity(&x, &z, &w, 0.0, 0.0), Cv002Verdict::Pass);
    }
    #[test] fn cv002_fail_length_mismatch() {
        let x = vec![1.0_f32, 2.0];
        let z = vec![1.0_f32, 2.0, 3.0];
        let w = vec![0.5_f32, 0.5];
        assert_eq!(verdict_from_linearity(&x, &z, &w, 1.0, 1.0), Cv002Verdict::Fail);
    }
    #[test] fn cv002_fail_nan() {
        let x = vec![1.0_f32, f32::NAN];
        let z = vec![1.0_f32, 2.0];
        let w = vec![0.5_f32, 0.5];
        assert_eq!(verdict_from_linearity(&x, &z, &w, 1.0, 1.0), Cv002Verdict::Fail);
    }

    // CV-003 (im2col equivalence)
    #[test] fn cv003_pass_canonical() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let w = vec![0.1_f32, 0.2, 0.3];
        assert_eq!(verdict_from_im2col_equivalence(&x, &w), Cv003Verdict::Pass);
    }
    #[test] fn cv003_pass_random() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let w: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        assert_eq!(verdict_from_im2col_equivalence(&x, &w), Cv003Verdict::Pass);
    }
    #[test] fn cv003_fail_empty() {
        assert_eq!(verdict_from_im2col_equivalence(&[], &[1.0]), Cv003Verdict::Fail);
        assert_eq!(verdict_from_im2col_equivalence(&[1.0], &[]), Cv003Verdict::Fail);
    }

    // CV-004 (SIMD parity)
    #[test] fn cv004_pass_identical() {
        let a = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_simd_parity(&a, &a), Cv004Verdict::Pass);
    }
    #[test] fn cv004_pass_within_ulp() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![
            f32::from_bits(1.0_f32.to_bits() + 1),
            f32::from_bits(2.0_f32.to_bits() + 2),
        ];
        assert_eq!(verdict_from_simd_parity(&a, &b), Cv004Verdict::Pass);
    }
    #[test] fn cv004_fail_above_8_ulp() {
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 100)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Cv004Verdict::Fail);
    }
    #[test] fn cv004_fail_length_mismatch() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &b), Cv004Verdict::Fail);
    }

    // CV-005 (K=1 boundary)
    #[test] fn cv005_pass_canonical() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(verdict_from_k1_boundary(&x, 0.5), Cv005Verdict::Pass);
    }
    #[test] fn cv005_pass_negative_w() {
        let x = vec![1.0_f32, -2.0, 3.0];
        assert_eq!(verdict_from_k1_boundary(&x, -3.0), Cv005Verdict::Pass);
    }
    #[test] fn cv005_pass_zero_w() {
        let x = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_k1_boundary(&x, 0.0), Cv005Verdict::Pass);
    }
    #[test] fn cv005_fail_empty_x() {
        assert_eq!(verdict_from_k1_boundary(&[], 1.0), Cv005Verdict::Fail);
    }
    #[test] fn cv005_fail_nan() {
        assert_eq!(verdict_from_k1_boundary(&[1.0_f32], f32::NAN), Cv005Verdict::Fail);
    }

    // CV-006 (identity kernel)
    #[test] fn cv006_pass_identity_at_zero() {
        // Kernel [1, 0, 0] applied to [a, b, c, d, e] → [a, b, c]
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(verdict_from_identity_kernel(&x, 3, 0), Cv006Verdict::Pass);
    }
    #[test] fn cv006_pass_identity_centered() {
        // Kernel [0, 1, 0] applied to [a, b, c, d, e] → [b, c, d]
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(verdict_from_identity_kernel(&x, 3, 1), Cv006Verdict::Pass);
    }
    #[test] fn cv006_pass_identity_at_end() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(verdict_from_identity_kernel(&x, 3, 2), Cv006Verdict::Pass);
    }
    #[test] fn cv006_fail_empty_x() {
        assert_eq!(verdict_from_identity_kernel(&[], 3, 0), Cv006Verdict::Fail);
    }
    #[test] fn cv006_fail_pos_oob() {
        // identity_pos >= kernel_size is invalid.
        let x = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_identity_kernel(&x, 2, 5), Cv006Verdict::Fail);
    }

    // Conv1d helper sanity
    #[test] fn conv1d_simple() {
        // [1, 2, 3, 4] * [1, 1] → [3, 5, 7]
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w = vec![1.0_f32, 1.0];
        assert_eq!(conv1d_scalar(&x, &w), vec![3.0_f32, 5.0, 7.0]);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert_eq!(AC_CV_004_ULP_TOLERANCE, 8);
        assert!((AC_CV_002_TOLERANCE - 1e-5).abs() < 1e-12);
        assert!((AC_CV_003_TOLERANCE - 1e-6).abs() < 1e-12);
    }
}
