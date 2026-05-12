// SHIP-TWO-001 — `lora-algebra-v1` algorithm-level PARTIAL discharge
// for FALSIFY-LA-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/lora-algebra-v1.yaml`.
// Spec: SVD LoRA extraction + merge strategy algebra (Hu 2021 LoRA;
// Eckart-Young-Mirsky 1936; Yadav 2023 TIES; Yu 2023 DARE).

// ===========================================================================
// LA-001 — Task vector roundtrip: base + (fine - base) == fine within ULP
// ===========================================================================

pub const AC_LA_001_TOLERANCE: f32 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum La001Verdict { Pass, Fail }

/// delta = fine - base, then verify base + delta ≈ fine element-wise.
#[must_use]
pub fn verdict_from_task_vector_roundtrip(base: &[f32], fine: &[f32]) -> La001Verdict {
    if base.is_empty() || fine.is_empty() { return La001Verdict::Fail; }
    if base.len() != fine.len() { return La001Verdict::Fail; }
    for (&b, &f) in base.iter().zip(fine.iter()) {
        if !b.is_finite() || !f.is_finite() { return La001Verdict::Fail; }
        let delta = f - b;
        let recovered = b + delta;
        if (recovered - f).abs() > AC_LA_001_TOLERANCE { return La001Verdict::Fail; }
    }
    La001Verdict::Pass
}

// ===========================================================================
// LA-002 — Eckart-Young bound: ||M - M_r||_F <= sigma_{r+1}
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum La002Verdict { Pass, Fail }

/// Pass iff truncation_error_frobenius ≤ sigma_{r+1} (the (r+1)-th
/// singular value). Caller computes both quantities; verdict checks
/// the inequality.
#[must_use]
pub fn verdict_from_eckart_young(
    truncation_error_frobenius: f32,
    sigma_r_plus_1: f32,
) -> La002Verdict {
    if !truncation_error_frobenius.is_finite() { return La002Verdict::Fail; }
    if !sigma_r_plus_1.is_finite() { return La002Verdict::Fail; }
    if truncation_error_frobenius < 0.0 { return La002Verdict::Fail; }
    if sigma_r_plus_1 < 0.0 { return La002Verdict::Fail; }
    // Allow a small slack for f32 rounding in the Frobenius norm computation.
    let slack = 1.0e-5_f32 * sigma_r_plus_1.abs() + 1.0e-7;
    if truncation_error_frobenius > sigma_r_plus_1 + slack { return La002Verdict::Fail; }
    La002Verdict::Pass
}

// ===========================================================================
// LA-003 — LoRA shape: A=[m,r], B=[r,n] => A@B=[m,n], r << min(m,n)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum La003Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_lora_shape(m: u64, r: u64, n: u64) -> La003Verdict {
    if m == 0 || r == 0 || n == 0 { return La003Verdict::Fail; }
    // Rank must be ≤ min(m, n) (strict equality of A@B shape requires
    // r ≤ min(m,n) for the SVD interpretation; r > min(m,n) breaks it).
    let min_mn = if m < n { m } else { n };
    if r > min_mn { return La003Verdict::Fail; }
    La003Verdict::Pass
}

// ===========================================================================
// LA-004 — DARE unbiased: E[DARE(delta, p)] = delta
// ===========================================================================

pub const AC_LA_004_TOLERANCE: f32 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum La004Verdict { Pass, Fail }

/// Sample mean of N DARE-applied delta vectors should approximate the
/// original delta within statistical tolerance. Verdict checks element-wise.
#[must_use]
pub fn verdict_from_dare_unbiased(
    delta: &[f32],
    sample_mean: &[f32],
    p: f32,
) -> La004Verdict {
    if delta.is_empty() || sample_mean.is_empty() { return La004Verdict::Fail; }
    if delta.len() != sample_mean.len() { return La004Verdict::Fail; }
    if !p.is_finite() || p <= 0.0 || p >= 1.0 { return La004Verdict::Fail; }
    for (&d, &s) in delta.iter().zip(sample_mean.iter()) {
        if !d.is_finite() || !s.is_finite() { return La004Verdict::Fail; }
        // Statistical tolerance scales with magnitude of delta.
        let tol = AC_LA_004_TOLERANCE * d.abs().max(0.1);
        if (s - d).abs() > tol { return La004Verdict::Fail; }
    }
    La004Verdict::Pass
}

// ===========================================================================
// LA-005 — Shape preservation: shape(W + B@A) == shape(W)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum La005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_shape_preservation(w_shape: &[u64], merged_shape: &[u64]) -> La005Verdict {
    if w_shape.is_empty() || merged_shape.is_empty() { return La005Verdict::Fail; }
    if w_shape == merged_shape { La005Verdict::Pass } else { La005Verdict::Fail }
}

// ===========================================================================
// LA-006 — SIMD LoRA parity (contract tolerance=0.0 → byte-exact)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum La006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_simd_parity(scalar: &[f32], simd: &[f32]) -> La006Verdict {
    if scalar.is_empty() || simd.is_empty() { return La006Verdict::Fail; }
    if scalar.len() != simd.len() { return La006Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if !s.is_finite() || !v.is_finite() { return La006Verdict::Fail; }
        if s.to_bits() != v.to_bits() { return La006Verdict::Fail; }
    }
    La006Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // LA-001 (task vector roundtrip)
    #[test] fn la001_pass_canonical() {
        let base = vec![1.0_f32, 2.0, 3.0];
        let fine = vec![1.5_f32, 2.3, 2.9];
        assert_eq!(verdict_from_task_vector_roundtrip(&base, &fine), La001Verdict::Pass);
    }
    #[test] fn la001_pass_zero_delta() {
        let base = vec![1.0_f32, 2.0, 3.0];
        let fine = base.clone();
        assert_eq!(verdict_from_task_vector_roundtrip(&base, &fine), La001Verdict::Pass);
    }
    #[test] fn la001_fail_length_mismatch() {
        let base = vec![1.0_f32];
        let fine = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_task_vector_roundtrip(&base, &fine), La001Verdict::Fail);
    }
    #[test] fn la001_fail_nan() {
        let base = vec![f32::NAN];
        let fine = vec![1.0_f32];
        assert_eq!(verdict_from_task_vector_roundtrip(&base, &fine), La001Verdict::Fail);
    }
    #[test] fn la001_fail_empty() {
        assert_eq!(verdict_from_task_vector_roundtrip(&[], &[]), La001Verdict::Fail);
    }

    // LA-002 (Eckart-Young)
    #[test] fn la002_pass_below_bound() {
        // Truncation error 0.5, sigma_{r+1} = 1.0 — bound holds.
        assert_eq!(verdict_from_eckart_young(0.5, 1.0), La002Verdict::Pass);
    }
    #[test] fn la002_pass_at_bound() {
        // Equality is allowed (with rounding slack).
        assert_eq!(verdict_from_eckart_young(1.0, 1.0), La002Verdict::Pass);
    }
    #[test] fn la002_fail_above_bound() {
        // Truncation error exceeds sigma_{r+1} — SVD impl is wrong.
        assert_eq!(verdict_from_eckart_young(2.0, 1.0), La002Verdict::Fail);
    }
    #[test] fn la002_fail_negative_error() {
        assert_eq!(verdict_from_eckart_young(-0.5, 1.0), La002Verdict::Fail);
    }
    #[test] fn la002_fail_nan() {
        assert_eq!(verdict_from_eckart_young(f32::NAN, 1.0), La002Verdict::Fail);
    }

    // LA-003 (LoRA shape)
    #[test] fn la003_pass_canonical() {
        // m=4096, r=8, n=4096 — typical LoRA rank-8 on 4096-dim layer.
        assert_eq!(verdict_from_lora_shape(4096, 8, 4096), La003Verdict::Pass);
    }
    #[test] fn la003_pass_high_rank() {
        // r at the boundary: r == min(m, n).
        assert_eq!(verdict_from_lora_shape(4, 4, 8), La003Verdict::Pass);
    }
    #[test] fn la003_fail_rank_too_high() {
        // The contract's stated falsifier: "Use rank > min(m,n)".
        assert_eq!(verdict_from_lora_shape(4, 5, 8), La003Verdict::Fail);
    }
    #[test] fn la003_fail_zero() {
        assert_eq!(verdict_from_lora_shape(0, 8, 4096), La003Verdict::Fail);
        assert_eq!(verdict_from_lora_shape(4096, 0, 4096), La003Verdict::Fail);
    }

    // LA-004 (DARE unbiased)
    #[test] fn la004_pass_within_tolerance() {
        let delta = vec![1.0_f32, -0.5, 0.3];
        // Statistical sample mean within tolerance.
        let mean = vec![1.02_f32, -0.49, 0.31];
        assert_eq!(verdict_from_dare_unbiased(&delta, &mean, 0.5), La004Verdict::Pass);
    }
    #[test] fn la004_fail_above_tolerance() {
        let delta = vec![1.0_f32];
        // 50% off — exceeds 5% tolerance.
        let mean = vec![1.5_f32];
        assert_eq!(verdict_from_dare_unbiased(&delta, &mean, 0.5), La004Verdict::Fail);
    }
    #[test] fn la004_fail_p_zero() {
        let delta = vec![1.0_f32];
        let mean = vec![1.0_f32];
        assert_eq!(verdict_from_dare_unbiased(&delta, &mean, 0.0), La004Verdict::Fail);
    }
    #[test] fn la004_fail_p_one() {
        let delta = vec![1.0_f32];
        let mean = vec![1.0_f32];
        assert_eq!(verdict_from_dare_unbiased(&delta, &mean, 1.0), La004Verdict::Fail);
    }

    // LA-005 (shape preservation)
    #[test] fn la005_pass_match() {
        let s = vec![4096_u64, 4096];
        assert_eq!(verdict_from_shape_preservation(&s, &s), La005Verdict::Pass);
    }
    #[test] fn la005_fail_drift() {
        let w = vec![4096_u64, 4096];
        let merged = vec![4096_u64, 4097];
        assert_eq!(verdict_from_shape_preservation(&w, &merged), La005Verdict::Fail);
    }
    #[test] fn la005_fail_extra_dim() {
        let w = vec![4096_u64, 4096];
        let merged = vec![4096_u64, 4096, 1];
        assert_eq!(verdict_from_shape_preservation(&w, &merged), La005Verdict::Fail);
    }

    // LA-006 (SIMD parity)
    #[test] fn la006_pass_identical() {
        let a = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &a), La006Verdict::Pass);
    }
    #[test] fn la006_fail_one_ulp() {
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 1)];
        // tolerance=0.0 — even 1 ULP fails.
        assert_eq!(verdict_from_simd_parity(&a, &b), La006Verdict::Fail);
    }
    #[test] fn la006_fail_length() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &b), La006Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_LA_001_TOLERANCE - 1e-5).abs() < 1e-12);
        assert!((AC_LA_004_TOLERANCE - 0.05).abs() < 1e-9);
    }
}
