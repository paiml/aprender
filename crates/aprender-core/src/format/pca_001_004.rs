// SHIP-TWO-001 — `pca-v1` algorithm-level PARTIAL discharge
// for FALSIFY-PCA-001..004.
//
// Contract: `contracts/pca-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Four PCA gates from Jolliffe (2002) / Bishop (2006) §12.1:
//
// - PCA-001 (dimensionality reduction): output shape == (n, k).
// - PCA-002 (explained variance bounded): each ratio ∈ [0, 1] AND
//   sum ≈ 1.
// - PCA-003 (variance ordering): explained_variance_ratio non-increasing.
// - PCA-004 (deterministic transform): transform(X) ≡ transform(X).
//
// All four are pure properties of (Z, eigenvalues, ratios). No
// SVD/eigendecomposition wired here.

/// Tolerance for Σ explained_variance_ratio == 1.
pub const AC_PCA_002_RATIO_SUM_EPS: f32 = 1e-5;

/// Lower bound for each ratio (inclusive).
pub const AC_PCA_002_RATIO_MIN: f32 = 0.0;

/// Upper bound for each ratio (inclusive).
pub const AC_PCA_002_RATIO_MAX: f32 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcaVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// Verdict 1: PCA-001 — dimensionality reduction.
// -----------------------------------------------------------------------------

/// Pass iff `transformed_shape == (n_samples, n_components)`.
#[must_use]
pub fn verdict_from_dimensionality(
    transformed_rows: usize,
    transformed_cols: usize,
    expected_n_samples: usize,
    expected_n_components: usize,
) -> PcaVerdict {
    if expected_n_samples == 0 || expected_n_components == 0 {
        return PcaVerdict::Fail;
    }
    if transformed_rows == expected_n_samples && transformed_cols == expected_n_components {
        PcaVerdict::Pass
    } else {
        PcaVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: PCA-002 — explained variance bounded.
// -----------------------------------------------------------------------------

/// Pass iff:
///   1. every ratio ∈ [0, 1],
///   2. sum ≈ 1 within `AC_PCA_002_RATIO_SUM_EPS`,
///   3. all entries finite.
#[must_use]
pub fn verdict_from_explained_variance_bounded(ratios: &[f32]) -> PcaVerdict {
    if ratios.is_empty() {
        return PcaVerdict::Fail;
    }
    let mut sum = 0.0_f32;
    for &r in ratios {
        if !r.is_finite() {
            return PcaVerdict::Fail;
        }
        if r < AC_PCA_002_RATIO_MIN || r > AC_PCA_002_RATIO_MAX {
            return PcaVerdict::Fail;
        }
        sum += r;
    }
    if (sum - 1.0).abs() < AC_PCA_002_RATIO_SUM_EPS {
        PcaVerdict::Pass
    } else {
        PcaVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: PCA-003 — variance ordering (non-increasing).
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_variance_ordering(ratios: &[f32]) -> PcaVerdict {
    if ratios.is_empty() {
        return PcaVerdict::Fail;
    }
    for w in ratios.windows(2) {
        if !w[0].is_finite() || !w[1].is_finite() {
            return PcaVerdict::Fail;
        }
        // Allow tiny numerical wobble (≤ eps).
        if w[1] > w[0] + AC_PCA_002_RATIO_SUM_EPS {
            return PcaVerdict::Fail;
        }
    }
    PcaVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 4: PCA-004 — deterministic transform.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_deterministic_transform(z1: &[f32], z2: &[f32]) -> PcaVerdict {
    if z1.len() != z2.len() {
        return PcaVerdict::Fail;
    }
    if z1.is_empty() {
        return PcaVerdict::Fail;
    }
    for (a, b) in z1.iter().zip(z2.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return PcaVerdict::Fail;
        }
        if a.to_bits() != b.to_bits() {
            return PcaVerdict::Fail;
        }
    }
    PcaVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_ratio_sum_eps_1e_5() {
        assert_eq!(AC_PCA_002_RATIO_SUM_EPS, 1e-5);
    }

    #[test]
    fn provenance_ratio_bounds_0_1() {
        assert_eq!(AC_PCA_002_RATIO_MIN, 0.0);
        assert_eq!(AC_PCA_002_RATIO_MAX, 1.0);
    }

    // -------------------------------------------------------------------------
    // Section 2: PCA-001 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn pca001_pass_typical_reduction() {
        // n=100 samples, d=10 features → reduce to k=3.
        assert_eq!(
            verdict_from_dimensionality(100, 3, 100, 3),
            PcaVerdict::Pass
        );
    }

    #[test]
    fn pca001_pass_full_rank() {
        // k = d ⇒ no actual reduction, but shape still (n, k).
        assert_eq!(
            verdict_from_dimensionality(50, 10, 50, 10),
            PcaVerdict::Pass
        );
    }

    #[test]
    fn pca001_pass_single_component() {
        assert_eq!(
            verdict_from_dimensionality(100, 1, 100, 1),
            PcaVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: PCA-001 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn pca001_fail_wrong_n_samples() {
        // Output truncated rows.
        assert_eq!(
            verdict_from_dimensionality(99, 3, 100, 3),
            PcaVerdict::Fail
        );
    }

    #[test]
    fn pca001_fail_wrong_n_components() {
        // Output kept too many components.
        assert_eq!(
            verdict_from_dimensionality(100, 5, 100, 3),
            PcaVerdict::Fail
        );
    }

    #[test]
    fn pca001_fail_zero_samples() {
        assert_eq!(verdict_from_dimensionality(0, 3, 0, 3), PcaVerdict::Fail);
    }

    #[test]
    fn pca001_fail_zero_components() {
        assert_eq!(verdict_from_dimensionality(100, 0, 100, 0), PcaVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: PCA-002 — explained variance bounded.
    // -------------------------------------------------------------------------
    #[test]
    fn pca002_pass_skewed_distribution() {
        // 80%, 15%, 5%.
        let ratios = vec![0.80_f32, 0.15, 0.05];
        assert_eq!(
            verdict_from_explained_variance_bounded(&ratios),
            PcaVerdict::Pass
        );
    }

    #[test]
    fn pca002_pass_uniform() {
        let ratios = vec![0.25_f32; 4];
        assert_eq!(
            verdict_from_explained_variance_bounded(&ratios),
            PcaVerdict::Pass
        );
    }

    #[test]
    fn pca002_pass_single_dominant_component() {
        let ratios = vec![1.0_f32, 0.0, 0.0];
        assert_eq!(
            verdict_from_explained_variance_bounded(&ratios),
            PcaVerdict::Pass
        );
    }

    #[test]
    fn pca002_fail_negative_ratio() {
        let ratios = vec![0.5_f32, -0.1, 0.6];
        assert_eq!(
            verdict_from_explained_variance_bounded(&ratios),
            PcaVerdict::Fail
        );
    }

    #[test]
    fn pca002_fail_ratio_above_one() {
        let ratios = vec![1.5_f32, -0.5];
        assert_eq!(
            verdict_from_explained_variance_bounded(&ratios),
            PcaVerdict::Fail
        );
    }

    #[test]
    fn pca002_fail_sum_below_one() {
        let ratios = vec![0.3_f32, 0.3, 0.3]; // sum 0.9
        assert_eq!(
            verdict_from_explained_variance_bounded(&ratios),
            PcaVerdict::Fail
        );
    }

    #[test]
    fn pca002_fail_sum_above_one() {
        let ratios = vec![0.5_f32, 0.5, 0.5]; // sum 1.5
        assert_eq!(
            verdict_from_explained_variance_bounded(&ratios),
            PcaVerdict::Fail
        );
    }

    #[test]
    fn pca002_fail_empty() {
        let ratios: Vec<f32> = vec![];
        assert_eq!(
            verdict_from_explained_variance_bounded(&ratios),
            PcaVerdict::Fail
        );
    }

    #[test]
    fn pca002_fail_nan() {
        let ratios = vec![0.5_f32, f32::NAN];
        assert_eq!(
            verdict_from_explained_variance_bounded(&ratios),
            PcaVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: PCA-003 — variance ordering.
    // -------------------------------------------------------------------------
    #[test]
    fn pca003_pass_strictly_decreasing() {
        let ratios = vec![0.5_f32, 0.3, 0.15, 0.05];
        assert_eq!(verdict_from_variance_ordering(&ratios), PcaVerdict::Pass);
    }

    #[test]
    fn pca003_pass_non_increasing_with_ties() {
        let ratios = vec![0.4_f32, 0.4, 0.2];
        assert_eq!(verdict_from_variance_ordering(&ratios), PcaVerdict::Pass);
    }

    #[test]
    fn pca003_pass_single_value() {
        let ratios = vec![1.0_f32];
        assert_eq!(verdict_from_variance_ordering(&ratios), PcaVerdict::Pass);
    }

    #[test]
    fn pca003_fail_ascending() {
        // Bug: eigenvalues sorted ascending instead of descending.
        let ratios = vec![0.05_f32, 0.15, 0.30, 0.50];
        assert_eq!(verdict_from_variance_ordering(&ratios), PcaVerdict::Fail);
    }

    #[test]
    fn pca003_fail_one_jump() {
        let ratios = vec![0.5_f32, 0.3, 0.4, 0.1]; // 0.3 → 0.4 ascent
        assert_eq!(verdict_from_variance_ordering(&ratios), PcaVerdict::Fail);
    }

    #[test]
    fn pca003_fail_empty() {
        let ratios: Vec<f32> = vec![];
        assert_eq!(verdict_from_variance_ordering(&ratios), PcaVerdict::Fail);
    }

    #[test]
    fn pca003_fail_nan() {
        let ratios = vec![0.5_f32, f32::NAN, 0.1];
        assert_eq!(verdict_from_variance_ordering(&ratios), PcaVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: PCA-004 — determinism.
    // -------------------------------------------------------------------------
    #[test]
    fn pca004_pass_byte_identical() {
        let z1 = vec![1.0_f32, 2.5, -0.3, 4.7];
        let z2 = z1.clone();
        assert_eq!(
            verdict_from_deterministic_transform(&z1, &z2),
            PcaVerdict::Pass
        );
    }

    #[test]
    fn pca004_pass_signed_zero_preserved() {
        // ±0 differ in bits — the contract specifies determinism.
        let z1 = vec![0.0_f32, -0.0];
        let z2 = vec![0.0_f32, -0.0];
        assert_eq!(
            verdict_from_deterministic_transform(&z1, &z2),
            PcaVerdict::Pass
        );
    }

    #[test]
    fn pca004_fail_one_bit_drift() {
        let z1 = vec![1.0_f32, 2.0];
        let z2 = vec![1.0_f32, 2.000001]; // drift
        assert_eq!(
            verdict_from_deterministic_transform(&z1, &z2),
            PcaVerdict::Fail
        );
    }

    #[test]
    fn pca004_fail_signed_zero_flip() {
        // +0 vs -0 differ in bits.
        let z1 = vec![0.0_f32];
        let z2 = vec![-0.0_f32];
        assert_eq!(
            verdict_from_deterministic_transform(&z1, &z2),
            PcaVerdict::Fail
        );
    }

    #[test]
    fn pca004_fail_length_mismatch() {
        let z1 = vec![1.0_f32, 2.0];
        let z2 = vec![1.0_f32];
        assert_eq!(
            verdict_from_deterministic_transform(&z1, &z2),
            PcaVerdict::Fail
        );
    }

    #[test]
    fn pca004_fail_empty() {
        let v: Vec<f32> = vec![];
        assert_eq!(
            verdict_from_deterministic_transform(&v, &v),
            PcaVerdict::Fail
        );
    }

    #[test]
    fn pca004_fail_nan() {
        let z1 = vec![f32::NAN];
        let z2 = vec![f32::NAN];
        // NaN bits-equal but not deterministic guarantees finiteness too.
        assert_eq!(
            verdict_from_deterministic_transform(&z1, &z2),
            PcaVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Sweep — n_components valid range.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_dimensionality_pass_band() {
        for k in 1..=10 {
            assert_eq!(
                verdict_from_dimensionality(50, k, 50, k),
                PcaVerdict::Pass,
                "k={k}"
            );
        }
    }

    #[test]
    fn sweep_dimensionality_fail_band() {
        // Off-by-one mismatches.
        let test_cases = [(50_usize, 3, 50, 4), (49, 3, 50, 3), (51, 3, 50, 3)];
        for (rows, cols, exp_n, exp_k) in test_cases {
            assert_eq!(
                verdict_from_dimensionality(rows, cols, exp_n, exp_k),
                PcaVerdict::Fail,
                "({rows}, {cols}) vs ({exp_n}, {exp_k})"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 8: Realistic — contract regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_eigenvalue_normalization_bug_caught() {
        // PCA-002 if_fails: "Eigenvalue normalization error".
        // Simulate: ratios sum to 1.5 (not normalized).
        let ratios = vec![0.6_f32, 0.5, 0.4]; // sum 1.5
        assert_eq!(
            verdict_from_explained_variance_bounded(&ratios),
            PcaVerdict::Fail
        );
    }

    #[test]
    fn realistic_eigenvalues_sorted_ascending_caught() {
        // PCA-003 if_fails: "Eigenvalues not sorted descending".
        // The contract failure example: smallest eigenvalue first.
        let ratios = vec![0.05_f32, 0.15, 0.30, 0.50];
        assert_eq!(verdict_from_variance_ordering(&ratios), PcaVerdict::Fail);
    }

    #[test]
    fn realistic_non_deterministic_random_state_caught() {
        // PCA-004 if_fails: "Non-deterministic random state".
        let z1 = vec![1.5_f32, 2.5, 3.5];
        let z2 = vec![1.5_f32, 2.5, 3.50001]; // tiny drift = nondeterministic
        assert_eq!(
            verdict_from_deterministic_transform(&z1, &z2),
            PcaVerdict::Fail
        );
    }

    #[test]
    fn realistic_full_pca_pipeline() {
        // Synthesize a typical PCA result on n=100, d=5, k=3.
        let n_samples = 100;
        let n_components = 3;
        let z = vec![1.0_f32; n_samples * n_components];
        let ratios = vec![0.6_f32, 0.3, 0.1]; // sums to 1
        // PCA-001:
        assert_eq!(
            verdict_from_dimensionality(n_samples, n_components, n_samples, n_components),
            PcaVerdict::Pass
        );
        // PCA-002:
        assert_eq!(
            verdict_from_explained_variance_bounded(&ratios),
            PcaVerdict::Pass
        );
        // PCA-003:
        assert_eq!(verdict_from_variance_ordering(&ratios), PcaVerdict::Pass);
        // PCA-004:
        let z2 = z.clone();
        assert_eq!(
            verdict_from_deterministic_transform(&z, &z2),
            PcaVerdict::Pass
        );
    }
}
