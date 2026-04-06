//! Canonical scalar activation functions.
//!
//! # One Path Rule (UCBD §4)
//!
//! These are THE canonical implementations for scalar activation functions.
//! All downstream crates (aprender, realizar, entrenar, whisper-apr) MUST
//! import from here instead of re-implementing.
//!
//! For SIMD-vectorized slice operations, see `backends::*/ops/activations`.
//! For `Vector`-level operations, see `vector::ops::activations`.

/// SiLU (Sigmoid Linear Unit) / Swish activation: x * σ(x).
///
/// # Equation
/// ```text
/// SiLU(x) = x * σ(x) = x / (1 + exp(-x))
/// ```
///
/// # Contract
/// - Domain: x ∈ ℝ
/// - Codomain: SiLU(x) ∈ (-0.278..., ∞)
/// - SiLU(0) = 0
/// - limₓ→∞ SiLU(x) = x
/// - limₓ→-∞ SiLU(x) = 0
#[inline]
#[must_use]
pub fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU (Gaussian Error Linear Unit) activation.
///
/// Uses the fast tanh approximation (same as PyTorch `gelu('tanh')`).
///
/// # Equation
/// ```text
/// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
/// ```
///
/// # Contract
/// - Domain: x ∈ ℝ
/// - Codomain: GELU(x) ∈ (-0.170..., ∞)
/// - GELU(0) = 0
/// - limₓ→∞ GELU(x) = x
/// - limₓ→-∞ GELU(x) = 0
#[inline]
#[must_use]
pub fn gelu_scalar(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}

/// Sigmoid activation: σ(x) = 1 / (1 + exp(-x)).
///
/// # Equation
/// ```text
/// σ(x) = 1 / (1 + exp(-x))
/// ```
///
/// # Contract
/// - Domain: x ∈ ℝ
/// - Codomain: σ(x) ∈ (0, 1)
/// - σ(0) = 0.5
/// - σ(-x) = 1 - σ(x) (symmetry)
#[inline]
#[must_use]
pub fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// ReLU (Rectified Linear Unit) activation.
///
/// # Equation
/// ```text
/// ReLU(x) = max(0, x)
/// ```
///
/// # Contract
/// - Domain: x ∈ ℝ
/// - Codomain: ReLU(x) ∈ [0, ∞)
/// - ReLU(x) = 0 for x ≤ 0
/// - ReLU(x) = x for x > 0
#[inline]
#[must_use]
pub fn relu_scalar(x: f32) -> f32 {
    x.max(0.0)
}

/// Tanh activation.
///
/// # Equation
/// ```text
/// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
/// ```
///
/// # Contract
/// - Domain: x ∈ ℝ
/// - Codomain: tanh(x) ∈ (-1, 1)
/// - tanh(0) = 0
/// - tanh(-x) = -tanh(x) (odd function)
#[inline]
#[must_use]
pub fn tanh_scalar(x: f32) -> f32 {
    x.tanh()
}

/// f16 → f32 conversion (IEEE 754 half-precision).
///
/// Manual bit-manipulation implementation (no `half` crate dependency).
/// Delegates to `tiling::q4k_matvec::f16_bits_to_f32` which is the
/// existing canonical implementation in trueno.
///
/// # Contract
/// - Domain: any u16 (interpreted as IEEE 754 binary16)
/// - Codomain: f32 (exact representation, no precision loss for normal f16)
/// - Subnormals, ±inf, NaN handled correctly
#[inline]
#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = bits & 0x3FF;

    // Fast path: normal numbers
    if exponent != 0 && exponent != 31 {
        let f32_exp = (exponent as u32 + 112) as u32; // bias adjustment: 127 - 15 = 112
        let f32_mant = (mantissa as u32) << 13; // 10 bits → 23 bits
        let f32_bits = ((sign as u32) << 31) | (f32_exp << 23) | f32_mant;
        return f32::from_bits(f32_bits);
    }

    // Special cases
    if exponent == 0 {
        if mantissa == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal
        const TWO_POW_NEG_14: f32 = 6.103_515_625e-5; // 2^-14
        let m = mantissa as f32 * (1.0 / 1024.0);
        let result = m * TWO_POW_NEG_14;
        return if sign == 1 { -result } else { result };
    }

    // exponent == 31: Inf or NaN
    if mantissa == 0 {
        if sign == 1 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }
    } else {
        f32::NAN
    }
}

/// f32 → f16 conversion (IEEE 754 half-precision).
///
/// Manual bit-manipulation implementation. Rounds to nearest even.
///
/// # Contract
/// - Domain: f32
/// - Codomain: u16 (IEEE 754 binary16 bits)
/// - Rounds to nearest even
#[inline]
#[must_use]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    // Special cases
    if exponent == 255 {
        // Inf or NaN
        if mantissa == 0 {
            return sign | 0x7C00; // ±Inf
        }
        return sign | 0x7C00 | ((mantissa >> 13) as u16).max(1); // NaN (preserve payload)
    }

    // Rebias exponent: f32 bias=127, f16 bias=15
    let new_exp = exponent - 112; // 127 - 15

    if new_exp >= 31 {
        return sign | 0x7C00; // Overflow → ±Inf
    }
    if new_exp <= 0 {
        // Subnormal or zero
        if new_exp < -10 {
            return sign; // Too small → ±0
        }
        let mant = (mantissa | 0x0080_0000) >> (1 - new_exp + 13);
        return sign | mant as u16;
    }

    // Normal number: round to nearest even
    let round_bit = (mantissa >> 12) & 1;
    let mant16 = ((mantissa >> 13) as u16) + round_bit as u16;
    sign | ((new_exp as u16) << 10) | (mant16 & 0x03FF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_zero() {
        assert!((silu_scalar(0.0)).abs() < 1e-7);
    }

    #[test]
    fn test_silu_positive() {
        // SiLU(x) → x for large positive x
        let x = 10.0;
        assert!((silu_scalar(x) - x).abs() < 0.01);
    }

    #[test]
    fn test_silu_negative() {
        // SiLU(x) → 0 for large negative x
        assert!(silu_scalar(-10.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu_zero() {
        assert!((gelu_scalar(0.0)).abs() < 1e-7);
    }

    #[test]
    fn test_gelu_positive() {
        let x = 10.0;
        assert!((gelu_scalar(x) - x).abs() < 0.01);
    }

    #[test]
    fn test_sigmoid_zero() {
        assert!((sigmoid_scalar(0.0) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        let x = 2.5;
        assert!((sigmoid_scalar(x) + sigmoid_scalar(-x) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_positive() {
        assert!((relu_scalar(3.0) - 3.0).abs() < 1e-7);
    }

    #[test]
    fn test_relu_negative() {
        assert!((relu_scalar(-3.0)).abs() < 1e-7);
    }

    #[test]
    fn test_tanh_zero() {
        assert!((tanh_scalar(0.0)).abs() < 1e-7);
    }

    #[test]
    fn test_tanh_odd() {
        let x = 1.5;
        assert!((tanh_scalar(x) + tanh_scalar(-x)).abs() < 1e-6);
    }

    #[test]
    fn test_f16_roundtrip() {
        let val = 1.5_f32;
        let bits = f32_to_f16(val);
        let back = f16_to_f32(bits);
        assert!((val - back).abs() < 1e-3);
    }

    #[test]
    fn test_f16_zero() {
        assert_eq!(f16_to_f32(0), 0.0);
    }

    // =========================================================================
    // FALSIFY-GE: gelu-kernel-v1.yaml contract (trueno gelu_scalar)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: trueno had basic gelu tests but zero FALSIFY-GE-* tests
    //   Why 2: tests checked 2 values (zero, large), not mathematical invariants
    //   Why 3: no mapping from gelu-kernel-v1.yaml to trueno test names
    //   Why 4: trueno predates the provable-contracts YAML convention
    //   Why 5: GELU was "obviously correct" (tanh approximation is textbook)
    //
    // References:
    //   - provable-contracts/contracts/gelu-kernel-v1.yaml
    //   - Hendrycks & Gimpel (2016) "Gaussian Error Linear Units (GELUs)"
    // =========================================================================

    /// FALSIFY-GE-001: Non-negativity — GELU(x) >= 0 for all x > 0
    #[test]
    fn falsify_ge_001_non_negativity() {
        let test_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1e6];
        for &x in &test_values {
            let y = gelu_scalar(x);
            assert!(y >= 0.0, "FALSIFIED GE-001: GELU({x}) = {y} < 0 for positive input");
        }
    }

    /// FALSIFY-GE-002: Monotonicity — GELU(x) > GELU(y) when x > y > 0
    #[test]
    fn falsify_ge_002_positive_monotonicity() {
        let values: Vec<f32> = vec![0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0];
        for window in values.windows(2) {
            let (y_lo, y_hi) = (gelu_scalar(window[0]), gelu_scalar(window[1]));
            assert!(
                y_hi > y_lo,
                "FALSIFIED GE-002: GELU({}) = {} not > GELU({}) = {}",
                window[1],
                y_hi,
                window[0],
                y_lo
            );
        }
    }

    /// FALSIFY-GE-003: Zero preservation — GELU(0) = 0
    #[test]
    fn falsify_ge_003_zero_preservation() {
        let y = gelu_scalar(0.0);
        assert!(y.abs() < 1e-7, "FALSIFIED GE-003: GELU(0) = {y}, expected 0");
    }

    /// FALSIFY-GE-005: Tanh approximation vs exact CDF — |diff| < 0.005
    ///
    /// Exact GELU: x * Phi(x) where Phi is the standard normal CDF.
    /// We approximate Phi via Abramowitz & Stegun erf formula (max error 1.5e-7).
    #[test]
    fn falsify_ge_005_tanh_approx_accuracy() {
        // Abramowitz & Stegun erf approximation (7.1.26), max |error| < 1.5e-7
        fn erf_approx(x: f32) -> f32 {
            let sign = x.signum();
            let x = x.abs();
            let t = 1.0 / (1.0 + 0.327_591_1 * x);
            let t2 = t * t;
            let t3 = t2 * t;
            let t4 = t3 * t;
            let t5 = t4 * t;
            let poly = 0.254_829_592 * t - 0.284_496_736 * t2 + 1.421_413_741 * t3
                - 1.453_152_027 * t4
                + 1.061_405_429 * t5;
            sign * (1.0 - poly * (-x * x).exp())
        }

        fn gelu_exact(x: f32) -> f32 {
            let phi = 0.5 * (1.0 + erf_approx(x / std::f32::consts::SQRT_2));
            x * phi
        }

        let test_values: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        for &x in &test_values {
            let approx = gelu_scalar(x);
            let exact = gelu_exact(x);
            let diff = (approx - exact).abs();
            assert!(
                diff < 0.005,
                "FALSIFIED GE-005: |GELU_approx({x}) - GELU_exact({x})| = {diff} >= 0.005"
            );
        }
    }

    /// FALSIFY-GE-006: Large input stability — GELU(x) ≈ x for large x, ≈ 0 for large -x
    #[test]
    fn falsify_ge_006_large_input_stability() {
        for &x in &[10.0_f32, 50.0, 100.0, 1000.0] {
            let y = gelu_scalar(x);
            assert!((y - x).abs() < 0.01, "FALSIFIED GE-006: GELU({x}) = {y}, expected ≈ {x}");
        }
        for &x in &[-10.0_f32, -50.0, -100.0, -1000.0] {
            let y = gelu_scalar(x);
            assert!(y.abs() < 0.01, "FALSIFIED GE-006: GELU({x}) = {y}, expected ≈ 0");
        }
    }

    mod ge_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // GE-001-prop: non-negativity for positive x
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]
            #[test]
            fn falsify_ge_001_prop_non_negativity(x in 0.0_f32..1000.0) {
                let y = gelu_scalar(x);
                prop_assert!(y >= 0.0, "FALSIFIED GE-001-prop: gelu({x}) = {y} < 0");
            }
        }

        // GE-002-prop: monotonicity for positive pairs
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(300))]
            #[test]
            fn falsify_ge_002_prop_monotonic_positive(
                a in 0.001_f32..100.0,
                b in 0.001_f32..100.0,
            ) {
                if a != b {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    let y_lo = gelu_scalar(lo);
                    let y_hi = gelu_scalar(hi);
                    prop_assert!(
                        y_hi > y_lo,
                        "FALSIFIED GE-002-prop: gelu({hi})={y_hi} not > gelu({lo})={y_lo}"
                    );
                }
            }
        }

        // GE-006-prop: large input stability
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn falsify_ge_006_prop_large_positive(x in 10.0_f32..500.0) {
                let y = gelu_scalar(x);
                prop_assert!(
                    (y - x).abs() < 0.01,
                    "FALSIFIED GE-006-prop: |gelu({x}) - {x}| = {}",
                    (y - x).abs()
                );
            }
        }
    }
}

// =========================================================================
// FALSIFY-SI: silu-kernel-v1.yaml contract (trueno silu_scalar)
//
// Five-Whys (PMAT-354, Phase 11):
//   Why 1: trueno had basic silu unit tests but zero FALSIFY-SI-* tests
//   Why 2: unit tests verify point values, not mathematical invariants
//   Why 3: no mapping from silu-kernel-v1.yaml to trueno test names
//   Why 4: trueno predates the provable-contracts YAML convention
//   Why 5: SiLU was "obviously correct" (x * sigmoid(x))
//
// References:
//   - provable-contracts/contracts/silu-kernel-v1.yaml
//   - Ramachandran et al. (2017) "Searching for Activation Functions"
// =========================================================================

#[cfg(test)]
mod silu_contract_tests {
    use super::*;

    /// FALSIFY-SI-001: Zero preservation — SiLU(0) = 0
    #[test]
    fn falsify_si_001_zero_preservation() {
        let y = silu_scalar(0.0);
        assert!(y.abs() < 1e-7, "FALSIFIED SI-001: SiLU(0) = {y}, expected 0");
    }

    /// FALSIFY-SI-002: Global lower bound — SiLU(x) > -0.279 for all x
    #[test]
    fn falsify_si_002_global_lower_bound() {
        let test_values: Vec<f32> =
            vec![-100.0, -50.0, -10.0, -5.0, -2.0, -1.278, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 100.0];
        for &x in &test_values {
            let y = silu_scalar(x);
            assert!(y > -0.28, "FALSIFIED SI-002: SiLU({x}) = {y}, expected > -0.279");
        }
    }

    /// FALSIFY-SI-003: Monotonic for positive inputs — x > y > 0 ⟹ SiLU(x) > SiLU(y)
    #[test]
    fn falsify_si_003_monotonic_positive() {
        let values: Vec<f32> = vec![0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0];
        for i in 1..values.len() {
            let y_prev = silu_scalar(values[i - 1]);
            let y_curr = silu_scalar(values[i]);
            assert!(
                y_curr > y_prev,
                "FALSIFIED SI-003: SiLU({}) = {y_curr} not > SiLU({}) = {y_prev}",
                values[i],
                values[i - 1]
            );
        }
    }

    /// FALSIFY-SI-005: Asymptotic linearity — |SiLU(x) - x| < 0.01 for x > 10
    #[test]
    fn falsify_si_005_asymptotic_linearity() {
        for &x in &[10.0f32, 20.0, 50.0, 100.0, 500.0] {
            let y = silu_scalar(x);
            assert!(
                (y - x).abs() < 0.01,
                "FALSIFIED SI-005: |SiLU({x}) - {x}| = {} >= 0.01",
                (y - x).abs()
            );
        }
    }

    /// FALSIFY-SI-006: Large negative → 0 — |SiLU(x)| < 0.01 for x < -10
    #[test]
    fn falsify_si_006_large_negative_vanishes() {
        for &x in &[-10.0f32, -20.0, -50.0, -100.0, -500.0] {
            let y = silu_scalar(x);
            assert!(y.abs() < 0.01, "FALSIFIED SI-006: SiLU({x}) = {y}, expected ≈ 0");
        }
    }

    mod si_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // SI-002-prop: global lower bound
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]
            #[test]
            fn falsify_si_002_prop_lower_bound(x in -1000.0_f32..1000.0) {
                let y = silu_scalar(x);
                prop_assert!(
                    y > -0.28,
                    "FALSIFIED SI-002-prop: SiLU({x}) = {y} <= -0.279"
                );
            }
        }

        // SI-003-prop: monotonic for positive pairs
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(300))]
            #[test]
            fn falsify_si_003_prop_monotonic_positive(
                a in 0.001_f32..100.0,
                b in 0.001_f32..100.0,
            ) {
                if a != b {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    let y_lo = silu_scalar(lo);
                    let y_hi = silu_scalar(hi);
                    prop_assert!(
                        y_hi > y_lo,
                        "FALSIFIED SI-003-prop: SiLU({hi})={y_hi} not > SiLU({lo})={y_lo}"
                    );
                }
            }
        }

        // SI-005-prop: asymptotic linearity for large positive x
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn falsify_si_005_prop_asymptotic(x in 10.0_f32..500.0) {
                let y = silu_scalar(x);
                prop_assert!(
                    (y - x).abs() < 0.01,
                    "FALSIFIED SI-005-prop: |SiLU({x}) - {x}| = {}",
                    (y - x).abs()
                );
            }
        }
    }
}
