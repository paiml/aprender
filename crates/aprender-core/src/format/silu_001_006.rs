// SHIP-TWO-001 — `silu-kernel-v1` algorithm-level PARTIAL discharge
// for FALSIFY-SI-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/silu-kernel-v1.yaml`.
// Spec: SiLU activation function (Ramachandran 2017; Elfwing 2018):
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

// ===========================================================================
// Helpers — sigmoid + silu (in-module reference impls)
// ===========================================================================

#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[must_use]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

// ===========================================================================
// SI-001 — Zero preservation: SiLU(0) == 0 exactly
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Si001Verdict { Pass, Fail }

/// Pass iff `silu(0.0) == 0.0` byte-exactly. SiLU(0) = 0 * sigmoid(0)
/// = 0 * 0.5 = 0.0; this should hold without any tolerance.
#[must_use]
pub fn verdict_from_zero_preservation() -> Si001Verdict {
    let v = silu(0.0);
    if v.to_bits() == 0.0_f32.to_bits() { Si001Verdict::Pass } else { Si001Verdict::Fail }
}

// ===========================================================================
// SI-002 — Global lower bound: SiLU(x) > -0.279 for x in [-1000, 1000]
// ===========================================================================

pub const AC_SI_002_LOWER_BOUND: f32 = -0.279;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Si002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_lower_bound(probes: &[f32]) -> Si002Verdict {
    if probes.is_empty() { return Si002Verdict::Fail; }
    for &x in probes {
        if !x.is_finite() { return Si002Verdict::Fail; }
        let s = silu(x);
        if !s.is_finite() { return Si002Verdict::Fail; }
        if s <= AC_SI_002_LOWER_BOUND { return Si002Verdict::Fail; }
    }
    Si002Verdict::Pass
}

// ===========================================================================
// SI-003 — Positive monotonicity: x > y > 0 ⇒ SiLU(x) > SiLU(y)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Si003Verdict { Pass, Fail }

/// Caller passes a sorted-ascending slice of strictly positive probes.
/// Pass iff silu produces a strictly increasing sequence.
#[must_use]
pub fn verdict_from_positive_monotonicity(probes: &[f32]) -> Si003Verdict {
    if probes.len() < 2 { return Si003Verdict::Fail; }
    let mut prev_x = f32::NEG_INFINITY;
    let mut prev_silu = f32::NEG_INFINITY;
    for &x in probes {
        if !x.is_finite() || x <= 0.0 { return Si003Verdict::Fail; }
        if x <= prev_x { return Si003Verdict::Fail; } // strict ascending input
        let s = silu(x);
        if !s.is_finite() { return Si003Verdict::Fail; }
        if s <= prev_silu { return Si003Verdict::Fail; } // strict ascending output
        prev_x = x;
        prev_silu = s;
    }
    Si003Verdict::Pass
}

// ===========================================================================
// SI-004 — SIMD parity within 8 ULP
// ===========================================================================

pub const AC_SI_004_ULP_TOLERANCE: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Si004Verdict { Pass, Fail }

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
pub fn verdict_from_simd_parity(scalar: &[f32], simd: &[f32]) -> Si004Verdict {
    if scalar.is_empty() || simd.is_empty() { return Si004Verdict::Fail; }
    if scalar.len() != simd.len() { return Si004Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if !s.is_finite() || !v.is_finite() { return Si004Verdict::Fail; }
        if ulp_distance(s, v) > AC_SI_004_ULP_TOLERANCE { return Si004Verdict::Fail; }
    }
    Si004Verdict::Pass
}

// ===========================================================================
// SI-005 — Asymptotic linearity: |SiLU(x) - x| < 0.01 for x > 10
// ===========================================================================

pub const AC_SI_005_TOLERANCE: f32 = 0.01;
pub const AC_SI_005_THRESHOLD: f32 = 10.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Si005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_asymptotic_linearity(probes: &[f32]) -> Si005Verdict {
    if probes.is_empty() { return Si005Verdict::Fail; }
    for &x in probes {
        if !x.is_finite() { return Si005Verdict::Fail; }
        if x <= AC_SI_005_THRESHOLD { return Si005Verdict::Fail; } // out of contract domain
        let s = silu(x);
        if !s.is_finite() { return Si005Verdict::Fail; }
        if (s - x).abs() >= AC_SI_005_TOLERANCE { return Si005Verdict::Fail; }
    }
    Si005Verdict::Pass
}

// ===========================================================================
// SI-006 — Large negative boundary: |SiLU(x)| < 0.01 for x < -10
// ===========================================================================

pub const AC_SI_006_TOLERANCE: f32 = 0.01;
pub const AC_SI_006_THRESHOLD: f32 = -10.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Si006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_large_negative_boundary(probes: &[f32]) -> Si006Verdict {
    if probes.is_empty() { return Si006Verdict::Fail; }
    for &x in probes {
        if !x.is_finite() { return Si006Verdict::Fail; }
        if x >= AC_SI_006_THRESHOLD { return Si006Verdict::Fail; } // out of domain
        let s = silu(x);
        if !s.is_finite() { return Si006Verdict::Fail; }
        if s.abs() >= AC_SI_006_TOLERANCE { return Si006Verdict::Fail; }
    }
    Si006Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // SI-001 (zero preservation)
    #[test] fn si001_pass() {
        assert_eq!(verdict_from_zero_preservation(), Si001Verdict::Pass);
    }
    #[test] fn silu_at_zero_is_exactly_zero() {
        // The bit-pattern check is meaningful: 0.0 in f32 is 0x00000000.
        assert_eq!(silu(0.0).to_bits(), 0.0_f32.to_bits());
    }

    // SI-002 (lower bound)
    #[test] fn si002_pass_canonical_range() {
        // Wide sweep including the global minimum at z ≈ -1.278.
        let probes: Vec<f32> = (-100..100)
            .step_by(5)
            .map(|i| i as f32 / 10.0)
            .collect();
        assert_eq!(verdict_from_lower_bound(&probes), Si002Verdict::Pass);
    }
    #[test] fn si002_pass_at_global_minimum() {
        let probes = [-1.2784_f32, -1.0, -1.5, -2.0];
        assert_eq!(verdict_from_lower_bound(&probes), Si002Verdict::Pass);
    }
    #[test] fn si002_pass_at_extremes_within_finite_range() {
        // SiLU saturates to ~0 for very negative x and ~x for very positive
        // x — both well above -0.279.
        let probes = [-100.0_f32, -50.0, -10.0, 50.0, 100.0];
        assert_eq!(verdict_from_lower_bound(&probes), Si002Verdict::Pass);
    }
    #[test] fn si002_fail_nan() {
        assert_eq!(verdict_from_lower_bound(&[f32::NAN]), Si002Verdict::Fail);
    }
    #[test] fn si002_fail_inf() {
        assert_eq!(verdict_from_lower_bound(&[f32::INFINITY]), Si002Verdict::Fail);
    }
    #[test] fn si002_fail_empty() {
        assert_eq!(verdict_from_lower_bound(&[]), Si002Verdict::Fail);
    }

    // SI-003 (positive monotonicity)
    #[test] fn si003_pass_sorted_positive() {
        let probes = [0.1_f32, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0];
        assert_eq!(verdict_from_positive_monotonicity(&probes), Si003Verdict::Pass);
    }
    #[test] fn si003_fail_unsorted() {
        let probes = [1.0_f32, 0.5, 2.0]; // not strictly ascending
        assert_eq!(verdict_from_positive_monotonicity(&probes), Si003Verdict::Fail);
    }
    #[test] fn si003_fail_zero() {
        // Zero is not in the strictly-positive domain.
        let probes = [0.0_f32, 1.0];
        assert_eq!(verdict_from_positive_monotonicity(&probes), Si003Verdict::Fail);
    }
    #[test] fn si003_fail_negative() {
        let probes = [-1.0_f32, 1.0];
        assert_eq!(verdict_from_positive_monotonicity(&probes), Si003Verdict::Fail);
    }
    #[test] fn si003_fail_single_probe() {
        // Need at least two probes to verify monotonicity.
        let probes = [1.0_f32];
        assert_eq!(verdict_from_positive_monotonicity(&probes), Si003Verdict::Fail);
    }

    // SI-004 (SIMD parity)
    #[test] fn si004_pass_identical() {
        let a = vec![0.5_f32, 1.0, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &a), Si004Verdict::Pass);
    }
    #[test] fn si004_pass_within_ulp() {
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 4)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Si004Verdict::Pass);
    }
    #[test] fn si004_fail_above_8_ulp() {
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 100)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Si004Verdict::Fail);
    }

    // SI-005 (asymptotic linearity)
    #[test] fn si005_pass_above_threshold() {
        // For x > 10, SiLU(x) ≈ x (sigmoid(x) ≈ 1).
        let probes = [11.0_f32, 15.0, 20.0, 50.0, 100.0];
        assert_eq!(verdict_from_asymptotic_linearity(&probes), Si005Verdict::Pass);
    }
    #[test] fn si005_fail_at_threshold() {
        // x = 10.0 is NOT strictly > 10 — outside the contract domain.
        let probes = [10.0_f32];
        assert_eq!(verdict_from_asymptotic_linearity(&probes), Si005Verdict::Fail);
    }
    #[test] fn si005_fail_below_threshold() {
        let probes = [5.0_f32];
        assert_eq!(verdict_from_asymptotic_linearity(&probes), Si005Verdict::Fail);
    }
    #[test] fn si005_fail_nan() {
        assert_eq!(verdict_from_asymptotic_linearity(&[f32::NAN]), Si005Verdict::Fail);
    }

    // SI-006 (large negative boundary)
    #[test] fn si006_pass_below_threshold() {
        // For x < -10, SiLU(x) ≈ 0 (sigmoid(x) ≈ 0, so x * 0 ≈ 0).
        let probes = [-11.0_f32, -15.0, -20.0, -50.0, -100.0];
        assert_eq!(verdict_from_large_negative_boundary(&probes), Si006Verdict::Pass);
    }
    #[test] fn si006_fail_at_threshold() {
        // x = -10.0 is NOT strictly < -10 — outside the contract domain.
        let probes = [-10.0_f32];
        assert_eq!(verdict_from_large_negative_boundary(&probes), Si006Verdict::Fail);
    }
    #[test] fn si006_fail_above_threshold() {
        let probes = [-5.0_f32];
        assert_eq!(verdict_from_large_negative_boundary(&probes), Si006Verdict::Fail);
    }

    // SiLU helper sanity
    #[test] fn silu_at_one_is_canonical() {
        // SiLU(1) = 1 * sigmoid(1) = 1 / (1 + 1/e) ≈ 0.7311
        let s = silu(1.0);
        assert!((s - 0.7310586).abs() < 1e-5);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_SI_002_LOWER_BOUND - (-0.279)).abs() < 1e-9);
        assert_eq!(AC_SI_004_ULP_TOLERANCE, 8);
        assert!((AC_SI_005_TOLERANCE - 0.01).abs() < 1e-9);
        assert!((AC_SI_005_THRESHOLD - 10.0).abs() < 1e-9);
        assert!((AC_SI_006_TOLERANCE - 0.01).abs() < 1e-9);
        assert!((AC_SI_006_THRESHOLD - (-10.0)).abs() < 1e-9);
    }
}
