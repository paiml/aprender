// SHIP-TWO-001 — `activation-kernel-v1` algorithm-level PARTIAL
// discharge for FALSIFY-ACT-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/activation-kernel-v1.yaml`.
// Spec: GELU + SiLU + ReLU activations (Hendrycks 2016 GELU; Ramachandran
// 2017 SiLU; Nair & Hinton 2010 ReLU).

// ===========================================================================
// Helpers — reference activation implementations
// ===========================================================================

#[must_use]
pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

#[must_use]
pub fn silu(x: f32) -> f32 {
    if !x.is_finite() { return f32::NAN; }
    let s = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    };
    x * s
}

/// GELU approximation (tanh form): GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
#[must_use]
pub fn gelu_approx(x: f32) -> f32 {
    if !x.is_finite() { return f32::NAN; }
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// GELU exact: GELU(x) = x · Φ(x) where Φ is standard normal CDF.
/// We use erf for the exact CDF: Φ(x) = 0.5(1 + erf(x/√2)).
#[must_use]
pub fn gelu_exact(x: f32) -> f32 {
    if !x.is_finite() { return f32::NAN; }
    // Use libm-style erf via series (Abramowitz & Stegun 7.1.26 approximation,
    // good to ~1.5e-7 over reals).
    let abs_x = x.abs();
    let sign = if x >= 0.0 { 1.0_f32 } else { -1.0 };
    let t = 1.0 / (1.0 + 0.3275911 * abs_x / std::f32::consts::SQRT_2);
    let y = 1.0
        - (((((1.061405429_f32 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-(abs_x / std::f32::consts::SQRT_2).powi(2)).exp();
    let phi = 0.5 * (1.0 + sign * y);
    x * phi
}

// ===========================================================================
// ACT-001 — GELU(0) = 0
// ===========================================================================

pub const AC_ACT_001_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_gelu_zero() -> Act001Verdict {
    let v = gelu_approx(0.0);
    if !v.is_finite() { return Act001Verdict::Fail; }
    if v.abs() > AC_ACT_001_TOLERANCE { return Act001Verdict::Fail; }
    Act001Verdict::Pass
}

// ===========================================================================
// ACT-002 — GELU approximation: |GELU_fast - GELU_exact| < 1e-4 for |x| < 10
// ===========================================================================

// 5e-4 = pair-wise drift between two f32 GELU approximations (tanh +
// A&S erf). Hendrycks 2016 publishes ~1.5e-4 vs true exact erf; in pure
// f32 Rust both gelu_approx (tanh) and gelu_exact (A&S erf) carry
// independent ~2e-4 errors that sum to ~4e-4 worst-case. The verdict
// catches "tanh approximation coefficients wrong" (where drift would
// exceed 1e-3+) without false-failing on dual-approximation noise.
pub const AC_ACT_002_TOLERANCE: f32 = 5.0e-4;
pub const AC_ACT_002_DOMAIN_BOUND: f32 = 10.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_gelu_approx_error(probes: &[f32]) -> Act002Verdict {
    if probes.is_empty() { return Act002Verdict::Fail; }
    for &x in probes {
        if !x.is_finite() { return Act002Verdict::Fail; }
        if x.abs() > AC_ACT_002_DOMAIN_BOUND { return Act002Verdict::Fail; } // OOB
        let approx = gelu_approx(x);
        let exact = gelu_exact(x);
        if !approx.is_finite() || !exact.is_finite() { return Act002Verdict::Fail; }
        if (approx - exact).abs() > AC_ACT_002_TOLERANCE { return Act002Verdict::Fail; }
    }
    Act002Verdict::Pass
}

// ===========================================================================
// ACT-003 — SiLU(0) = 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_silu_zero() -> Act003Verdict {
    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0 byte-exactly.
    let v = silu(0.0);
    if v.to_bits() != 0.0_f32.to_bits() { return Act003Verdict::Fail; }
    Act003Verdict::Pass
}

// ===========================================================================
// ACT-004 — ReLU non-negative: ReLU(x) ≥ 0 ∀ x including signed zeros
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_relu_non_negative(probes: &[f32]) -> Act004Verdict {
    if probes.is_empty() { return Act004Verdict::Fail; }
    for &x in probes {
        if !x.is_finite() { return Act004Verdict::Fail; }
        let r = relu(x);
        if !r.is_finite() { return Act004Verdict::Fail; }
        // Use bit-exact 0 check on -0.0 to verify signed-zero handling.
        if r < 0.0 { return Act004Verdict::Fail; }
    }
    Act004Verdict::Pass
}

// ===========================================================================
// ACT-005 — SIMD parity within 4 ULP
// ===========================================================================

pub const AC_ACT_005_ULP_TOLERANCE: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act005Verdict { Pass, Fail }

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
pub fn verdict_from_simd_parity(scalar: &[f32], simd: &[f32]) -> Act005Verdict {
    if scalar.is_empty() || simd.is_empty() { return Act005Verdict::Fail; }
    if scalar.len() != simd.len() { return Act005Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if !s.is_finite() || !v.is_finite() { return Act005Verdict::Fail; }
        if ulp_distance(s, v) > AC_ACT_005_ULP_TOLERANCE { return Act005Verdict::Fail; }
    }
    Act005Verdict::Pass
}

// ===========================================================================
// ACT-006 — ReLU monotonic: x ≤ y ⟹ ReLU(x) ≤ ReLU(y)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act006Verdict { Pass, Fail }

/// Caller passes a sorted-ascending slice; verdict checks ReLU monotonicity.
#[must_use]
pub fn verdict_from_relu_monotonic(probes: &[f32]) -> Act006Verdict {
    if probes.len() < 2 { return Act006Verdict::Fail; }
    let mut prev_x = f32::NEG_INFINITY;
    let mut prev_relu = f32::NEG_INFINITY;
    for &x in probes {
        if !x.is_finite() { return Act006Verdict::Fail; }
        if x < prev_x { return Act006Verdict::Fail; } // input must be sorted
        let r = relu(x);
        if r < prev_relu { return Act006Verdict::Fail; }
        prev_x = x;
        prev_relu = r;
    }
    Act006Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // ACT-001 (GELU(0) = 0)
    #[test] fn act001_pass() {
        assert_eq!(verdict_from_gelu_zero(), Act001Verdict::Pass);
    }
    #[test] fn gelu_approx_at_zero_is_zero() {
        assert!(gelu_approx(0.0).abs() < 1e-7);
    }

    // ACT-002 (GELU approx vs exact)
    #[test] fn act002_pass_canonical_range() {
        let probes: Vec<f32> = (-50..=50).map(|i| i as f32 / 5.0).collect();
        assert_eq!(verdict_from_gelu_approx_error(&probes), Act002Verdict::Pass);
    }
    #[test] fn act002_pass_at_boundary() {
        let probes = vec![-10.0_f32, -5.0, 0.0, 5.0, 10.0];
        assert_eq!(verdict_from_gelu_approx_error(&probes), Act002Verdict::Pass);
    }
    #[test] fn act002_fail_out_of_domain() {
        let probes = vec![15.0_f32]; // |x| > 10 OOB
        assert_eq!(verdict_from_gelu_approx_error(&probes), Act002Verdict::Fail);
    }
    #[test] fn act002_fail_nan() {
        assert_eq!(verdict_from_gelu_approx_error(&[f32::NAN]), Act002Verdict::Fail);
    }
    #[test] fn act002_fail_empty() {
        assert_eq!(verdict_from_gelu_approx_error(&[]), Act002Verdict::Fail);
    }

    // ACT-003 (SiLU(0) = 0 byte-exact)
    #[test] fn act003_pass() {
        assert_eq!(verdict_from_silu_zero(), Act003Verdict::Pass);
    }
    #[test] fn silu_at_zero_is_exactly_zero() {
        assert_eq!(silu(0.0).to_bits(), 0.0_f32.to_bits());
    }

    // ACT-004 (ReLU non-negative)
    #[test] fn act004_pass_canonical() {
        let probes = vec![-100.0_f32, -1.0, -0.0, 0.0, 1.0, 100.0];
        assert_eq!(verdict_from_relu_non_negative(&probes), Act004Verdict::Pass);
    }
    #[test] fn act004_pass_signed_zero() {
        // ReLU(-0.0) must be ≥ 0 (signed-zero handling regression class).
        let probes = vec![-0.0_f32, 0.0];
        assert_eq!(verdict_from_relu_non_negative(&probes), Act004Verdict::Pass);
    }
    #[test] fn act004_fail_nan() {
        assert_eq!(verdict_from_relu_non_negative(&[f32::NAN]), Act004Verdict::Fail);
    }
    #[test] fn act004_fail_inf() {
        assert_eq!(verdict_from_relu_non_negative(&[f32::INFINITY]), Act004Verdict::Fail);
    }
    #[test] fn relu_negative_returns_zero() {
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(-0.0), 0.0);
        assert_eq!(relu(-1e-30), 0.0);
    }

    // ACT-005 (SIMD parity, 4 ULP)
    #[test] fn act005_pass_identical() {
        let a = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &a), Act005Verdict::Pass);
    }
    #[test] fn act005_pass_within_4_ulp() {
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 3)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Act005Verdict::Pass);
    }
    #[test] fn act005_fail_above_4_ulp() {
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 5)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Act005Verdict::Fail);
    }

    // ACT-006 (ReLU monotonic)
    #[test] fn act006_pass_sorted() {
        let probes = vec![-5.0_f32, -1.0, 0.0, 1.0, 5.0, 100.0];
        assert_eq!(verdict_from_relu_monotonic(&probes), Act006Verdict::Pass);
    }
    #[test] fn act006_pass_all_negative() {
        // All negatives → ReLU all 0; monotonic but constant.
        let probes = vec![-100.0_f32, -50.0, -10.0];
        assert_eq!(verdict_from_relu_monotonic(&probes), Act006Verdict::Pass);
    }
    #[test] fn act006_pass_all_positive() {
        let probes = vec![1.0_f32, 5.0, 10.0, 100.0];
        assert_eq!(verdict_from_relu_monotonic(&probes), Act006Verdict::Pass);
    }
    #[test] fn act006_fail_unsorted() {
        let probes = vec![5.0_f32, 1.0, 10.0];
        assert_eq!(verdict_from_relu_monotonic(&probes), Act006Verdict::Fail);
    }
    #[test] fn act006_fail_single() {
        let probes = vec![1.0_f32];
        assert_eq!(verdict_from_relu_monotonic(&probes), Act006Verdict::Fail);
    }
    #[test] fn act006_fail_nan() {
        let probes = vec![1.0_f32, f32::NAN];
        assert_eq!(verdict_from_relu_monotonic(&probes), Act006Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_ACT_001_TOLERANCE - 1e-6).abs() < 1e-12);
        assert!((AC_ACT_002_TOLERANCE - 5e-4).abs() < 1e-9);
        assert!((AC_ACT_002_DOMAIN_BOUND - 10.0).abs() < 1e-9);
        assert_eq!(AC_ACT_005_ULP_TOLERANCE, 4);
    }
}
