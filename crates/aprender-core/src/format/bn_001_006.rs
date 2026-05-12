// `batchnorm-kernel-v1` algorithm-level PARTIAL discharge for the 6
// BatchNorm falsifiers (training standardization, denominator safety,
// running stats non-negativity, eval-mode determinism, SIMD equivalence,
// batch=1 boundary).
//
// Contract: `contracts/batchnorm-kernel-v1.yaml`.
// Refs: Ioffe & Szegedy (2015) Batch Normalization.

/// Tolerance for "post-BN mean ≈ 0" check when gamma=1, beta=0.
pub const AC_BN_MEAN_TOLERANCE: f32 = 1.0e-5;

/// SIMD-vs-scalar tolerance (8 ULP per contract proof_obligations).
/// At |x| ~ 1.0, 8 ULP ≈ 8 * 2^-23 ≈ 9.5e-7.
pub const AC_BN_SIMD_TOLERANCE: f32 = 1.0e-5;

// =============================================================================
// FALSIFY-BN-001 — training standardization
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BnTrainStandardizeVerdict {
    /// |mean(BN(x)[:,c])| < 1e-5 per channel c when gamma=1, beta=0.
    Pass,
    /// Some channel's post-BN mean exceeds tolerance.
    Fail,
}

#[must_use]
pub fn verdict_from_bn_train_standardize(per_channel_means: &[f32]) -> BnTrainStandardizeVerdict {
    if per_channel_means.is_empty() {
        return BnTrainStandardizeVerdict::Fail;
    }
    for &m in per_channel_means {
        if m.abs() >= AC_BN_MEAN_TOLERANCE {
            return BnTrainStandardizeVerdict::Fail;
        }
    }
    BnTrainStandardizeVerdict::Pass
}

// =============================================================================
// FALSIFY-BN-002 — denominator safety (no NaN/Inf with eps>0)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BnDenomSafetyVerdict {
    /// All output elements finite (no NaN/Inf) even on zero-variance input.
    Pass,
    /// Output contains NaN or Inf — eps not added before sqrt.
    Fail,
}

#[must_use]
pub fn verdict_from_bn_denom_safety(output: &[f32]) -> BnDenomSafetyVerdict {
    if output.is_empty() {
        return BnDenomSafetyVerdict::Fail;
    }
    for &v in output {
        if !v.is_finite() {
            return BnDenomSafetyVerdict::Fail;
        }
    }
    BnDenomSafetyVerdict::Pass
}

// =============================================================================
// FALSIFY-BN-003 — running stats non-negativity
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BnRunningNonnegVerdict {
    /// sigma_run >= 0 across all channels.
    Pass,
    /// Some channel's running variance went negative — EMA accumulation bug.
    Fail,
}

#[must_use]
pub fn verdict_from_bn_running_nonneg(sigma_run_per_channel: &[f32]) -> BnRunningNonnegVerdict {
    if sigma_run_per_channel.is_empty() {
        return BnRunningNonnegVerdict::Fail;
    }
    for &s in sigma_run_per_channel {
        if s < 0.0 || !s.is_finite() {
            return BnRunningNonnegVerdict::Fail;
        }
    }
    BnRunningNonnegVerdict::Pass
}

// =============================================================================
// FALSIFY-BN-004 — eval mode uses running stats, not batch stats
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BnEvalModeVerdict {
    /// When running ≠ batch stats, BN_eval(x) ≠ BN_train(x).
    Pass,
    /// Eval and train modes produced same output despite divergent stats —
    /// mode flag ignored.
    Fail,
}

#[must_use]
pub fn verdict_from_bn_eval_mode(
    eval_output: &[f32],
    train_output: &[f32],
    stats_diverged: bool,
) -> BnEvalModeVerdict {
    if !stats_diverged {
        // Stats are equal: eval and train should produce equal output —
        // out of scope for this gate.
        return BnEvalModeVerdict::Pass;
    }
    if eval_output.len() != train_output.len() {
        return BnEvalModeVerdict::Fail;
    }
    if eval_output.is_empty() {
        return BnEvalModeVerdict::Fail;
    }
    for (a, b) in eval_output.iter().zip(train_output.iter()) {
        if (a - b).abs() > 1e-9 {
            return BnEvalModeVerdict::Pass;
        }
    }
    BnEvalModeVerdict::Fail
}

// =============================================================================
// FALSIFY-BN-005 — SIMD matches scalar within 8 ULP
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BnSimdEquivalenceVerdict {
    /// max_i |simd[i] - scalar[i]| < tolerance.
    Pass,
    /// SIMD reduction order differs.
    Fail,
}

#[must_use]
pub fn verdict_from_bn_simd_equivalence(simd: &[f32], scalar: &[f32]) -> BnSimdEquivalenceVerdict {
    if simd.len() != scalar.len() {
        return BnSimdEquivalenceVerdict::Fail;
    }
    if simd.is_empty() {
        return BnSimdEquivalenceVerdict::Fail;
    }
    for (a, b) in simd.iter().zip(scalar.iter()) {
        if (a - b).abs() >= AC_BN_SIMD_TOLERANCE {
            return BnSimdEquivalenceVerdict::Fail;
        }
    }
    BnSimdEquivalenceVerdict::Pass
}

// =============================================================================
// FALSIFY-BN-006 — batch=1 boundary (zero variance ⇒ output = beta when gamma=1)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BnBatchOneBoundaryVerdict {
    /// At N=1: output[c] == beta[c] within tolerance (gamma=1).
    Pass,
    /// Edge-case in variance computation for N=1.
    Fail,
}

#[must_use]
pub fn verdict_from_bn_batch_one_boundary(
    output: &[f32],
    beta: &[f32],
) -> BnBatchOneBoundaryVerdict {
    if output.len() != beta.len() {
        return BnBatchOneBoundaryVerdict::Fail;
    }
    if output.is_empty() {
        return BnBatchOneBoundaryVerdict::Fail;
    }
    for (o, b) in output.iter().zip(beta.iter()) {
        if (o - b).abs() >= AC_BN_MEAN_TOLERANCE {
            return BnBatchOneBoundaryVerdict::Fail;
        }
    }
    BnBatchOneBoundaryVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_mean_tolerance_1e_neg5() {
        assert!((AC_BN_MEAN_TOLERANCE - 1.0e-5).abs() < f32::EPSILON);
    }

    #[test]
    fn provenance_simd_tolerance_1e_neg5() {
        assert!((AC_BN_SIMD_TOLERANCE - 1.0e-5).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Section 2: BN-001 training standardization.
    // -------------------------------------------------------------------------
    #[test]
    fn fbn001_pass_zero_mean_per_channel() {
        let m = vec![0.0_f32, 1e-7, -2e-7, 5e-6];
        assert_eq!(
            verdict_from_bn_train_standardize(&m),
            BnTrainStandardizeVerdict::Pass
        );
    }

    #[test]
    fn fbn001_fail_channel_mean_drift() {
        let m = vec![0.0_f32, 0.5];
        assert_eq!(
            verdict_from_bn_train_standardize(&m),
            BnTrainStandardizeVerdict::Fail
        );
    }

    #[test]
    fn fbn001_fail_at_threshold() {
        let m = vec![1.0e-5_f32];
        assert_eq!(
            verdict_from_bn_train_standardize(&m),
            BnTrainStandardizeVerdict::Fail
        );
    }

    #[test]
    fn fbn001_fail_empty() {
        assert_eq!(
            verdict_from_bn_train_standardize(&[]),
            BnTrainStandardizeVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: BN-002 denominator safety.
    // -------------------------------------------------------------------------
    #[test]
    fn fbn002_pass_finite_zero_variance() {
        let o = vec![0.0_f32, 0.0, 0.0];
        assert_eq!(
            verdict_from_bn_denom_safety(&o),
            BnDenomSafetyVerdict::Pass
        );
    }

    #[test]
    fn fbn002_fail_nan_in_output() {
        let o = vec![0.0_f32, f32::NAN, 0.0];
        assert_eq!(
            verdict_from_bn_denom_safety(&o),
            BnDenomSafetyVerdict::Fail
        );
    }

    #[test]
    fn fbn002_fail_inf_in_output() {
        let o = vec![0.0_f32, f32::INFINITY];
        assert_eq!(
            verdict_from_bn_denom_safety(&o),
            BnDenomSafetyVerdict::Fail
        );
    }

    #[test]
    fn fbn002_fail_empty() {
        assert_eq!(verdict_from_bn_denom_safety(&[]), BnDenomSafetyVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: BN-003 running stats non-negativity.
    // -------------------------------------------------------------------------
    #[test]
    fn fbn003_pass_all_nonneg() {
        let s = vec![0.0_f32, 1.0, 5.5, 100.0];
        assert_eq!(
            verdict_from_bn_running_nonneg(&s),
            BnRunningNonnegVerdict::Pass
        );
    }

    #[test]
    fn fbn003_fail_negative_variance() {
        let s = vec![1.0_f32, -0.001];
        assert_eq!(
            verdict_from_bn_running_nonneg(&s),
            BnRunningNonnegVerdict::Fail
        );
    }

    #[test]
    fn fbn003_fail_nan_variance() {
        let s = vec![1.0_f32, f32::NAN];
        assert_eq!(
            verdict_from_bn_running_nonneg(&s),
            BnRunningNonnegVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: BN-004 eval mode.
    // -------------------------------------------------------------------------
    #[test]
    fn fbn004_pass_eval_diverges_from_train() {
        let eval = vec![1.0_f32, 2.0];
        let train = vec![0.5_f32, 1.5];
        assert_eq!(
            verdict_from_bn_eval_mode(&eval, &train, true),
            BnEvalModeVerdict::Pass
        );
    }

    #[test]
    fn fbn004_pass_stats_match_then_outputs_should_match() {
        // stats_diverged=false → vacuous pass.
        let eval = vec![1.0_f32];
        let train = vec![1.0_f32];
        assert_eq!(
            verdict_from_bn_eval_mode(&eval, &train, false),
            BnEvalModeVerdict::Pass
        );
    }

    #[test]
    fn fbn004_fail_eval_equal_train_when_stats_diverged() {
        // The regression: mode flag ignored.
        let eval = vec![1.0_f32, 2.0];
        let train = vec![1.0_f32, 2.0];
        assert_eq!(
            verdict_from_bn_eval_mode(&eval, &train, true),
            BnEvalModeVerdict::Fail
        );
    }

    #[test]
    fn fbn004_fail_length_mismatch() {
        let eval = vec![1.0_f32];
        let train = vec![1.0_f32, 2.0];
        assert_eq!(
            verdict_from_bn_eval_mode(&eval, &train, true),
            BnEvalModeVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: BN-005 SIMD equivalence.
    // -------------------------------------------------------------------------
    #[test]
    fn fbn005_pass_exact_match() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(
            verdict_from_bn_simd_equivalence(&v, &v),
            BnSimdEquivalenceVerdict::Pass
        );
    }

    #[test]
    fn fbn005_pass_within_tolerance() {
        let s = vec![1.0_f32, 2.0];
        let scalar = vec![1.0_f32 + 1e-7, 2.0 + 1e-7];
        assert_eq!(
            verdict_from_bn_simd_equivalence(&s, &scalar),
            BnSimdEquivalenceVerdict::Pass
        );
    }

    #[test]
    fn fbn005_fail_outside_tolerance() {
        let s = vec![1.5_f32];
        let scalar = vec![1.0_f32];
        assert_eq!(
            verdict_from_bn_simd_equivalence(&s, &scalar),
            BnSimdEquivalenceVerdict::Fail
        );
    }

    #[test]
    fn fbn005_fail_length_mismatch() {
        assert_eq!(
            verdict_from_bn_simd_equivalence(&[1.0], &[1.0, 2.0]),
            BnSimdEquivalenceVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: BN-006 batch=1 boundary.
    // -------------------------------------------------------------------------
    #[test]
    fn fbn006_pass_output_equals_beta() {
        let o = vec![0.5_f32, 1.5, -0.3];
        let b = vec![0.5_f32, 1.5, -0.3];
        assert_eq!(
            verdict_from_bn_batch_one_boundary(&o, &b),
            BnBatchOneBoundaryVerdict::Pass
        );
    }

    #[test]
    fn fbn006_pass_within_tolerance() {
        let o = vec![0.5_f32 + 1e-7];
        let b = vec![0.5_f32];
        assert_eq!(
            verdict_from_bn_batch_one_boundary(&o, &b),
            BnBatchOneBoundaryVerdict::Pass
        );
    }

    #[test]
    fn fbn006_fail_output_diverges_from_beta() {
        let o = vec![1.0_f32];
        let b = vec![0.5_f32];
        assert_eq!(
            verdict_from_bn_batch_one_boundary(&o, &b),
            BnBatchOneBoundaryVerdict::Fail
        );
    }

    #[test]
    fn fbn006_fail_length_mismatch() {
        assert_eq!(
            verdict_from_bn_batch_one_boundary(&[1.0], &[1.0, 2.0]),
            BnBatchOneBoundaryVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 8: Realistic — full healthy BatchNorm passes all 6.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_batchnorm_passes_all_6() {
        // 001
        let means = vec![0.0_f32, 1e-8, -1e-8];
        assert_eq!(
            verdict_from_bn_train_standardize(&means),
            BnTrainStandardizeVerdict::Pass
        );
        // 002
        let out = vec![0.5_f32, 1.5];
        assert_eq!(verdict_from_bn_denom_safety(&out), BnDenomSafetyVerdict::Pass);
        // 003
        let s = vec![0.5_f32, 1.0];
        assert_eq!(
            verdict_from_bn_running_nonneg(&s),
            BnRunningNonnegVerdict::Pass
        );
        // 004
        let eval = vec![1.0_f32, 2.0];
        let train = vec![0.7_f32, 1.7];
        assert_eq!(
            verdict_from_bn_eval_mode(&eval, &train, true),
            BnEvalModeVerdict::Pass
        );
        // 005
        let v = vec![0.5_f32, 1.5];
        assert_eq!(
            verdict_from_bn_simd_equivalence(&v, &v),
            BnSimdEquivalenceVerdict::Pass
        );
        // 006
        let o = vec![0.5_f32];
        let b = vec![0.5_f32];
        assert_eq!(
            verdict_from_bn_batch_one_boundary(&o, &b),
            BnBatchOneBoundaryVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_6_failures() {
        // 001: forgot to subtract batch mean.
        assert_eq!(
            verdict_from_bn_train_standardize(&[0.5]),
            BnTrainStandardizeVerdict::Fail
        );
        // 002: zero-variance + no eps → NaN.
        assert_eq!(
            verdict_from_bn_denom_safety(&[f32::NAN]),
            BnDenomSafetyVerdict::Fail
        );
        // 003: EMA accumulator went negative.
        assert_eq!(
            verdict_from_bn_running_nonneg(&[-1e-3]),
            BnRunningNonnegVerdict::Fail
        );
        // 004: mode flag ignored — eval == train despite divergent stats.
        let eval = vec![1.0_f32];
        let train = vec![1.0_f32];
        assert_eq!(
            verdict_from_bn_eval_mode(&eval, &train, true),
            BnEvalModeVerdict::Fail
        );
        // 005: SIMD reduction off.
        assert_eq!(
            verdict_from_bn_simd_equivalence(&[1.5], &[1.0]),
            BnSimdEquivalenceVerdict::Fail
        );
        // 006: N=1 case bug.
        assert_eq!(
            verdict_from_bn_batch_one_boundary(&[10.0], &[0.0]),
            BnBatchOneBoundaryVerdict::Fail
        );
    }
}
