// Bundles two sister contracts in one verdict module:
//
//   `f16-conversion-v1` (FALSIFY-F16-001..004)
//   `dpo-loss-v1` (FALSIFY-DPO-001..005)
//
// F16-001: bit-trick f16→f32 matches Rust's f32::from(f16)
// F16-002: f16→f32→f16 roundtrip is identity for normal values
// F16-003: sign preservation under conversion
// F16-004: SIMD f16 conversion bit-exact equal to scalar
// DPO-001: L_DPO ≥ 0 for all valid log-ratio pairs and beta > 0
// DPO-002: L_DPO == log(2) when log_ratio_w == log_ratio_l == 0
// DPO-003: monotonicity in preferred log-ratio (raising r_w lowers loss)
// DPO-004: finite output for log-ratios in [-100, 100]
// DPO-005: symmetry under preference swap

/// DPO-002 reference value: log(2).
pub const AC_DPO_LOSS_AT_ZERO: f32 = std::f32::consts::LN_2;
/// DPO-002 tolerance for log(2) approximation.
pub const AC_DPO_LOG_2_TOLERANCE: f32 = 1e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F16DpoVerdict {
    Pass,
    Fail,
}

// ----------------------------------------------------------------
// F16-001..004
// ----------------------------------------------------------------

/// F16-001: bit-trick output bit-exact equal to Rust's f32::from(f16).
#[must_use]
pub fn verdict_from_f16_bit_trick(
    bittrick_output: f32,
    rust_canonical: f32,
) -> F16DpoVerdict {
    if !bittrick_output.is_finite() && !rust_canonical.is_finite() {
        // Both NaN/Inf — accept if bit pattern matches.
        if bittrick_output.to_bits() == rust_canonical.to_bits() {
            return F16DpoVerdict::Pass;
        }
        return F16DpoVerdict::Fail;
    }
    if bittrick_output.to_bits() == rust_canonical.to_bits() {
        F16DpoVerdict::Pass
    } else {
        F16DpoVerdict::Fail
    }
}

/// F16-002: roundtrip identity. `original_f16_bits` and
/// `roundtrip_f16_bits` are u16 representations.
#[must_use]
pub fn verdict_from_f16_roundtrip(
    original_f16_bits: u16,
    roundtrip_f16_bits: u16,
) -> F16DpoVerdict {
    if original_f16_bits == roundtrip_f16_bits {
        F16DpoVerdict::Pass
    } else {
        F16DpoVerdict::Fail
    }
}

/// F16-003: sign preservation.
#[must_use]
pub fn verdict_from_f16_sign(input_sign: bool, output_sign: bool) -> F16DpoVerdict {
    if input_sign == output_sign {
        F16DpoVerdict::Pass
    } else {
        F16DpoVerdict::Fail
    }
}

/// F16-004: SIMD f16 conversion bit-exact equal to scalar.
#[must_use]
pub fn verdict_from_f16_simd_parity(
    simd_output: &[f32],
    scalar_output: &[f32],
) -> F16DpoVerdict {
    if simd_output.is_empty() || simd_output.len() != scalar_output.len() {
        return F16DpoVerdict::Fail;
    }
    for (s, sc) in simd_output.iter().zip(scalar_output.iter()) {
        if s.to_bits() != sc.to_bits() {
            return F16DpoVerdict::Fail;
        }
    }
    F16DpoVerdict::Pass
}

// ----------------------------------------------------------------
// DPO-001..005
// ----------------------------------------------------------------

/// Reference DPO loss: -log(sigmoid(beta * (r_w - r_l))).
///
/// Uses log1p(exp(-x)) trick for numerical stability.
#[must_use]
pub fn dpo_loss(log_ratio_w: f32, log_ratio_l: f32, beta: f32) -> f32 {
    if !log_ratio_w.is_finite() || !log_ratio_l.is_finite() || !beta.is_finite() {
        return f32::NAN;
    }
    if beta <= 0.0 {
        return f32::NAN;
    }
    let z = beta * (log_ratio_w - log_ratio_l);
    // -log(sigma(z)) = log(1 + exp(-z))
    // Numerically stable via softplus(-z) = max(0, -z) + ln(1 + exp(-|z|))
    let abs_neg_z = (-z).abs();
    (-z).max(0.0) + (1.0 + (-abs_neg_z).exp()).ln()
}

/// DPO-001: L_DPO ≥ 0 for valid inputs.
#[must_use]
pub fn verdict_from_dpo_nonneg(loss: f32) -> F16DpoVerdict {
    if !loss.is_finite() {
        return F16DpoVerdict::Fail;
    }
    if loss >= 0.0 {
        F16DpoVerdict::Pass
    } else {
        F16DpoVerdict::Fail
    }
}

/// DPO-002: when r_w == r_l == 0, L_DPO == log(2) within tolerance.
#[must_use]
pub fn verdict_from_dpo_at_reference(observed_loss: f32) -> F16DpoVerdict {
    if !observed_loss.is_finite() {
        return F16DpoVerdict::Fail;
    }
    if (observed_loss - AC_DPO_LOSS_AT_ZERO).abs() <= AC_DPO_LOG_2_TOLERANCE {
        F16DpoVerdict::Pass
    } else {
        F16DpoVerdict::Fail
    }
}

/// DPO-003: monotonicity — raising r_w lowers loss.
///
/// `loss_low` corresponds to lower r_w; `loss_high` to higher r_w.
/// Pass iff `loss_high < loss_low` AND both finite.
#[must_use]
pub fn verdict_from_dpo_monotone(loss_low_rw: f32, loss_high_rw: f32) -> F16DpoVerdict {
    if !loss_low_rw.is_finite() || !loss_high_rw.is_finite() {
        return F16DpoVerdict::Fail;
    }
    if loss_high_rw < loss_low_rw {
        F16DpoVerdict::Pass
    } else {
        F16DpoVerdict::Fail
    }
}

/// DPO-004: loss finite for extreme log-ratios.
#[must_use]
pub fn verdict_from_dpo_stability(observed_loss: f32) -> F16DpoVerdict {
    if observed_loss.is_finite() {
        F16DpoVerdict::Pass
    } else {
        F16DpoVerdict::Fail
    }
}

/// DPO-005: symmetry — L(r_w, r_l) + L(r_l, r_w) ≈ |z| + 2*log(1+exp(-|z|))
/// for z = beta*(r_w - r_l). We use a simpler check: the sum equals
/// `|z| + 2 * dpo_loss(0, 0, 1)` adjusted for z, which simplifies to
/// `softplus(z) + softplus(-z)` — the `log(1+exp(-|z|))` is included.
/// Per the YAML, the analytical form is:
///   L(r_w,r_l) + L(r_l,r_w) = z + 2*log(1+exp(-|z|))
#[must_use]
pub fn verdict_from_dpo_symmetry(
    sum_observed: f32,
    z: f32,
    tolerance: f32,
) -> F16DpoVerdict {
    if !sum_observed.is_finite() || !z.is_finite() || !tolerance.is_finite() {
        return F16DpoVerdict::Fail;
    }
    let abs_z = z.abs();
    let expected = abs_z + 2.0 * (1.0 + (-abs_z).exp()).ln();
    if (sum_observed - expected).abs() <= tolerance {
        F16DpoVerdict::Pass
    } else {
        F16DpoVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_constants() {
        assert!((AC_DPO_LOSS_AT_ZERO - 0.6931472_f32).abs() < 1e-6);
        assert_eq!(AC_DPO_LOG_2_TOLERANCE, 1e-3);
    }

    // -----------------------------------------------------------------
    // Section 2: F16-001..004.
    // -----------------------------------------------------------------
    #[test]
    fn ff16_001_pass_bit_exact() {
        let v = verdict_from_f16_bit_trick(1.5, 1.5);
        assert_eq!(v, F16DpoVerdict::Pass);
    }

    #[test]
    fn ff16_001_fail_one_ulp_drift() {
        let bumped = f32::from_bits(1.5_f32.to_bits() + 1);
        let v = verdict_from_f16_bit_trick(bumped, 1.5);
        assert_eq!(v, F16DpoVerdict::Fail);
    }

    #[test]
    fn ff16_002_pass_identity() {
        let v = verdict_from_f16_roundtrip(0x3C00, 0x3C00); // 1.0 in f16
        assert_eq!(v, F16DpoVerdict::Pass);
    }

    #[test]
    fn ff16_002_fail_drift() {
        let v = verdict_from_f16_roundtrip(0x3C00, 0x3C01);
        assert_eq!(v, F16DpoVerdict::Fail);
    }

    #[test]
    fn ff16_003_pass_negative() {
        let v = verdict_from_f16_sign(true, true);
        assert_eq!(v, F16DpoVerdict::Pass);
    }

    #[test]
    fn ff16_003_fail_sign_flip() {
        let v = verdict_from_f16_sign(true, false);
        assert_eq!(v, F16DpoVerdict::Fail);
    }

    #[test]
    fn ff16_004_pass_bit_identical() {
        let simd = vec![1.0_f32, 2.0, 3.0];
        let scalar = simd.clone();
        let v = verdict_from_f16_simd_parity(&simd, &scalar);
        assert_eq!(v, F16DpoVerdict::Pass);
    }

    #[test]
    fn ff16_004_fail_one_ulp() {
        let simd = vec![1.0_f32];
        let bumped = f32::from_bits(1.0_f32.to_bits() + 1);
        let scalar = vec![bumped];
        let v = verdict_from_f16_simd_parity(&simd, &scalar);
        assert_eq!(v, F16DpoVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: dpo_loss reference.
    // -----------------------------------------------------------------
    #[test]
    fn dpo_loss_at_zero_is_log2() {
        let l = dpo_loss(0.0, 0.0, 1.0);
        assert!((l - std::f32::consts::LN_2).abs() < 1e-3);
    }

    #[test]
    fn dpo_loss_extreme_inputs_finite() {
        let l = dpo_loss(100.0, -100.0, 1.0);
        assert!(l.is_finite());
        let l = dpo_loss(-100.0, 100.0, 1.0);
        assert!(l.is_finite());
    }

    #[test]
    fn dpo_loss_invalid_beta_returns_nan() {
        assert!(dpo_loss(1.0, 0.0, 0.0).is_nan());
        assert!(dpo_loss(1.0, 0.0, -1.0).is_nan());
    }

    // -----------------------------------------------------------------
    // Section 4: DPO-001..005.
    // -----------------------------------------------------------------
    #[test]
    fn fdpo_001_pass_typical() {
        let l = dpo_loss(2.0, 0.0, 1.0);
        let v = verdict_from_dpo_nonneg(l);
        assert_eq!(v, F16DpoVerdict::Pass);
    }

    #[test]
    fn fdpo_001_fail_negative() {
        let v = verdict_from_dpo_nonneg(-0.001);
        assert_eq!(v, F16DpoVerdict::Fail);
    }

    #[test]
    fn fdpo_001_fail_nan() {
        let v = verdict_from_dpo_nonneg(f32::NAN);
        assert_eq!(v, F16DpoVerdict::Fail);
    }

    #[test]
    fn fdpo_002_pass_log2_at_zero() {
        let l = dpo_loss(0.0, 0.0, 1.0);
        let v = verdict_from_dpo_at_reference(l);
        assert_eq!(v, F16DpoVerdict::Pass);
    }

    #[test]
    fn fdpo_002_fail_at_zero_drift() {
        let v = verdict_from_dpo_at_reference(2.0);
        assert_eq!(v, F16DpoVerdict::Fail);
    }

    #[test]
    fn fdpo_003_pass_monotone() {
        let lo = dpo_loss(0.0, 0.0, 1.0);
        let hi = dpo_loss(2.0, 0.0, 1.0);
        let v = verdict_from_dpo_monotone(lo, hi);
        assert_eq!(v, F16DpoVerdict::Pass);
    }

    #[test]
    fn fdpo_003_fail_inverted() {
        let lo = dpo_loss(2.0, 0.0, 1.0);
        let hi = dpo_loss(0.0, 0.0, 1.0); // higher loss at lower rw
        let v = verdict_from_dpo_monotone(lo, hi);
        assert_eq!(v, F16DpoVerdict::Fail);
    }

    #[test]
    fn fdpo_004_pass_extreme_finite() {
        let l = dpo_loss(100.0, -100.0, 1.0);
        let v = verdict_from_dpo_stability(l);
        assert_eq!(v, F16DpoVerdict::Pass);
    }

    #[test]
    fn fdpo_004_fail_overflow() {
        let v = verdict_from_dpo_stability(f32::INFINITY);
        assert_eq!(v, F16DpoVerdict::Fail);
    }

    #[test]
    fn fdpo_005_pass_symmetry() {
        let r_w = 1.5_f32;
        let r_l = 0.5;
        let beta = 1.0;
        let z = beta * (r_w - r_l);
        let l_wl = dpo_loss(r_w, r_l, beta);
        let l_lw = dpo_loss(r_l, r_w, beta);
        let v = verdict_from_dpo_symmetry(l_wl + l_lw, z, 1e-3);
        assert_eq!(v, F16DpoVerdict::Pass);
    }

    #[test]
    fn fdpo_005_fail_asymmetric() {
        let v = verdict_from_dpo_symmetry(99.0, 1.0, 1e-3);
        assert_eq!(v, F16DpoVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_dpo_monotonicity_sweep() {
        // Sweep r_w from 0 to 5; loss should be strictly decreasing
        let r_l = 0.0_f32;
        let beta = 1.0_f32;
        let mut prev_loss = f32::INFINITY;
        for i in 0..=10 {
            let r_w = i as f32 * 0.5;
            let loss = dpo_loss(r_w, r_l, beta);
            assert!(loss.is_finite(), "r_w={r_w} produced non-finite");
            assert!(loss < prev_loss, "monotonicity violated: {loss} >= {prev_loss}");
            prev_loss = loss;
        }
    }

    // -----------------------------------------------------------------
    // Section 6: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_9() {
        let v1 = verdict_from_f16_bit_trick(1.5, 1.5);
        let v2 = verdict_from_f16_roundtrip(0x3C00, 0x3C00);
        let v3 = verdict_from_f16_sign(false, false);
        let v4 = verdict_from_f16_simd_parity(&[1.0_f32], &[1.0_f32]);
        let v5 = verdict_from_dpo_nonneg(dpo_loss(2.0, 0.0, 1.0));
        let v6 = verdict_from_dpo_at_reference(dpo_loss(0.0, 0.0, 1.0));
        let v7 = verdict_from_dpo_monotone(dpo_loss(0.0, 0.0, 1.0), dpo_loss(2.0, 0.0, 1.0));
        let v8 = verdict_from_dpo_stability(dpo_loss(50.0, -50.0, 1.0));
        let z = 1.0;
        let r_w = 0.5;
        let r_l = -0.5;
        let v9 = verdict_from_dpo_symmetry(dpo_loss(r_w, r_l, 1.0) + dpo_loss(r_l, r_w, 1.0), z, 1e-3);
        for v in [v1, v2, v3, v4, v5, v6, v7, v8, v9] {
            assert_eq!(v, F16DpoVerdict::Pass);
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Pre-fix regressions.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_pre_fix_all_9_failures() {
        let bumped = f32::from_bits(1.5_f32.to_bits() + 1);
        let v1 = verdict_from_f16_bit_trick(bumped, 1.5);
        let v2 = verdict_from_f16_roundtrip(0x3C00, 0x3C01);
        let v3 = verdict_from_f16_sign(true, false);
        let v4 = verdict_from_f16_simd_parity(&[1.0_f32], &[bumped]);
        let v5 = verdict_from_dpo_nonneg(-0.5); // sign error
        let v6 = verdict_from_dpo_at_reference(2.0); // wrong reference value
        let v7 = verdict_from_dpo_monotone(0.5, 1.0); // gradient sign inverted
        let v8 = verdict_from_dpo_stability(f32::NAN);
        let v9 = verdict_from_dpo_symmetry(99.0, 1.0, 1e-3);
        for v in [v1, v2, v3, v4, v5, v6, v7, v8, v9] {
            assert_eq!(v, F16DpoVerdict::Fail);
        }
    }
}
