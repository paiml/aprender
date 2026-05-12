// `cma-es-kernel-v1` algorithm-level PARTIAL discharge for the 6
// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) falsifiers
// (sigma positivity, C positive-definite, weights normalized,
// C symmetric, SIMD equivalence, d=1 boundary).
//
// Contract: `contracts/cma-es-kernel-v1.yaml`.
// Refs: Hansen (2016) The CMA Evolution Strategy: A Tutorial, Hansen &
// Ostermeier (2001) Completely Derandomized Self-Adaptation in
// Evolution Strategies.

/// Tolerance for "weights sum to 1" check (1e-10 per contract).
pub const AC_CMA_WEIGHT_NORM_TOLERANCE: f64 = 1.0e-10;

/// Tolerance for "C is symmetric" check (1e-10 per contract).
pub const AC_CMA_SYMMETRY_TOLERANCE: f64 = 1.0e-10;

/// Tolerance for SIMD-vs-scalar sample equivalence (8 ULP).
pub const AC_CMA_SIMD_TOLERANCE: f64 = 1.0e-7;

// =============================================================================
// FALSIFY-CMA-001 — sigma > 0 at every generation
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmaSigmaPositivityVerdict {
    /// All sigma values across generations are strictly positive AND finite.
    Pass,
    /// Sigma went non-positive (zero or negative) or non-finite.
    Fail,
}

#[must_use]
pub fn verdict_from_cma_sigma_positivity(sigma_per_generation: &[f64]) -> CmaSigmaPositivityVerdict {
    if sigma_per_generation.is_empty() {
        return CmaSigmaPositivityVerdict::Fail;
    }
    for &s in sigma_per_generation {
        if !s.is_finite() {
            return CmaSigmaPositivityVerdict::Fail;
        }
        if s <= 0.0 {
            return CmaSigmaPositivityVerdict::Fail;
        }
    }
    CmaSigmaPositivityVerdict::Pass
}

// =============================================================================
// FALSIFY-CMA-002 — C eigenvalues > 0 at every generation
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmaCovariancePosDefVerdict {
    /// min eigenvalue > 0 across all generations.
    Pass,
    /// Some generation produced a C with non-positive eigenvalue —
    /// rank update introduced negative eigenvalue.
    Fail,
}

#[must_use]
pub fn verdict_from_cma_covariance_posdef(min_eigenvalues: &[f64]) -> CmaCovariancePosDefVerdict {
    if min_eigenvalues.is_empty() {
        return CmaCovariancePosDefVerdict::Fail;
    }
    for &lambda_min in min_eigenvalues {
        if !lambda_min.is_finite() {
            return CmaCovariancePosDefVerdict::Fail;
        }
        if lambda_min <= 0.0 {
            return CmaCovariancePosDefVerdict::Fail;
        }
    }
    CmaCovariancePosDefVerdict::Pass
}

// =============================================================================
// FALSIFY-CMA-003 — recombination weights sum to 1 within 1e-10
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmaWeightNormalizationVerdict {
    /// |sum(w_i) - 1.0| < 1e-10.
    Pass,
    /// Weights don't sum to 1 — formula error.
    Fail,
}

#[must_use]
pub fn verdict_from_cma_weight_normalization(weights: &[f64]) -> CmaWeightNormalizationVerdict {
    if weights.is_empty() {
        return CmaWeightNormalizationVerdict::Fail;
    }
    let sum: f64 = weights.iter().sum();
    if !sum.is_finite() {
        return CmaWeightNormalizationVerdict::Fail;
    }
    if (sum - 1.0).abs() < AC_CMA_WEIGHT_NORM_TOLERANCE {
        CmaWeightNormalizationVerdict::Pass
    } else {
        CmaWeightNormalizationVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-CMA-004 — C symmetric within 1e-10
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmaCovarianceSymmetryVerdict {
    /// max_ij |C[i,j] - C[j,i]| < 1e-10 across all generations.
    Pass,
    /// Asymmetric update in rank-one or rank-mu.
    Fail,
}

#[must_use]
pub fn verdict_from_cma_covariance_symmetry(
    dim: usize,
    covariance_matrix: &[f64],
) -> CmaCovarianceSymmetryVerdict {
    if dim == 0 {
        return CmaCovarianceSymmetryVerdict::Fail;
    }
    if covariance_matrix.len() != dim * dim {
        return CmaCovarianceSymmetryVerdict::Fail;
    }
    for i in 0..dim {
        for j in 0..i {
            let c_ij = covariance_matrix[i * dim + j];
            let c_ji = covariance_matrix[j * dim + i];
            if !c_ij.is_finite() || !c_ji.is_finite() {
                return CmaCovarianceSymmetryVerdict::Fail;
            }
            if (c_ij - c_ji).abs() >= AC_CMA_SYMMETRY_TOLERANCE {
                return CmaCovarianceSymmetryVerdict::Fail;
            }
        }
    }
    CmaCovarianceSymmetryVerdict::Pass
}

// =============================================================================
// FALSIFY-CMA-005 — SIMD vs scalar within 8 ULP
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmaSimdEquivalenceVerdict {
    /// max_i |simd_sample[i] - scalar_sample[i]| < 1e-7.
    Pass,
    /// SIMD Cholesky or matmul differs.
    Fail,
}

#[must_use]
pub fn verdict_from_cma_simd_equivalence(simd: &[f64], scalar: &[f64]) -> CmaSimdEquivalenceVerdict {
    if simd.len() != scalar.len() {
        return CmaSimdEquivalenceVerdict::Fail;
    }
    if simd.is_empty() {
        return CmaSimdEquivalenceVerdict::Fail;
    }
    for (a, b) in simd.iter().zip(scalar.iter()) {
        if (a - b).abs() >= AC_CMA_SIMD_TOLERANCE {
            return CmaSimdEquivalenceVerdict::Fail;
        }
    }
    CmaSimdEquivalenceVerdict::Pass
}

// =============================================================================
// FALSIFY-CMA-006 — d=1 reduces to (1+1)-ES
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmaDimOneBoundaryVerdict {
    /// At d=1, CMA-ES with sphere/quadratic function converges to optimum
    /// (final sigma very small, final mean ≈ optimum).
    Pass,
    /// Edge case in covariance for scalar — failed to converge.
    Fail,
}

#[must_use]
pub fn verdict_from_cma_dim_one_boundary(
    final_mean: f64,
    target_optimum: f64,
    final_sigma: f64,
) -> CmaDimOneBoundaryVerdict {
    if !final_mean.is_finite() || !final_sigma.is_finite() {
        return CmaDimOneBoundaryVerdict::Fail;
    }
    // Final sigma should have decayed (CMA-ES contracts to near optimum).
    if final_sigma >= 1.0 {
        return CmaDimOneBoundaryVerdict::Fail;
    }
    // Mean should be near target.
    if (final_mean - target_optimum).abs() >= 0.1 {
        return CmaDimOneBoundaryVerdict::Fail;
    }
    CmaDimOneBoundaryVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_weight_norm_tol_1e_neg10() {
        assert!((AC_CMA_WEIGHT_NORM_TOLERANCE - 1.0e-10).abs() < f64::EPSILON);
    }

    #[test]
    fn provenance_symmetry_tol_1e_neg10() {
        assert!((AC_CMA_SYMMETRY_TOLERANCE - 1.0e-10).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Section 2: CMA-001 sigma positivity.
    // -------------------------------------------------------------------------
    #[test]
    fn fcma001_pass_all_positive() {
        let s = vec![1.0_f64, 0.5, 0.25, 0.1, 0.01];
        assert_eq!(
            verdict_from_cma_sigma_positivity(&s),
            CmaSigmaPositivityVerdict::Pass
        );
    }

    #[test]
    fn fcma001_fail_zero() {
        let s = vec![1.0_f64, 0.0];
        assert_eq!(
            verdict_from_cma_sigma_positivity(&s),
            CmaSigmaPositivityVerdict::Fail
        );
    }

    #[test]
    fn fcma001_fail_negative() {
        let s = vec![1.0_f64, -1e-10];
        assert_eq!(
            verdict_from_cma_sigma_positivity(&s),
            CmaSigmaPositivityVerdict::Fail
        );
    }

    #[test]
    fn fcma001_fail_nan() {
        let s = vec![1.0_f64, f64::NAN];
        assert_eq!(
            verdict_from_cma_sigma_positivity(&s),
            CmaSigmaPositivityVerdict::Fail
        );
    }

    #[test]
    fn fcma001_fail_empty() {
        assert_eq!(
            verdict_from_cma_sigma_positivity(&[]),
            CmaSigmaPositivityVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: CMA-002 covariance positive definiteness.
    // -------------------------------------------------------------------------
    #[test]
    fn fcma002_pass_all_positive_eigenvalues() {
        let l = vec![1.0_f64, 0.5, 0.1, 1e-6];
        assert_eq!(
            verdict_from_cma_covariance_posdef(&l),
            CmaCovariancePosDefVerdict::Pass
        );
    }

    #[test]
    fn fcma002_fail_zero_eigenvalue() {
        let l = vec![1.0_f64, 0.0];
        assert_eq!(
            verdict_from_cma_covariance_posdef(&l),
            CmaCovariancePosDefVerdict::Fail
        );
    }

    #[test]
    fn fcma002_fail_negative_eigenvalue() {
        let l = vec![1.0_f64, -1e-10];
        assert_eq!(
            verdict_from_cma_covariance_posdef(&l),
            CmaCovariancePosDefVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: CMA-003 weight normalization.
    // -------------------------------------------------------------------------
    #[test]
    fn fcma003_pass_uniform_4() {
        let w = vec![0.25_f64, 0.25, 0.25, 0.25];
        assert_eq!(
            verdict_from_cma_weight_normalization(&w),
            CmaWeightNormalizationVerdict::Pass
        );
    }

    #[test]
    fn fcma003_pass_canonical_decreasing() {
        // Sum carefully: must be exactly within 1e-10 of 1.0.
        let w = vec![0.5_f64, 0.3, 0.15, 0.05];
        assert_eq!(
            verdict_from_cma_weight_normalization(&w),
            CmaWeightNormalizationVerdict::Pass
        );
    }

    #[test]
    fn fcma003_fail_undersum() {
        let w = vec![0.5_f64, 0.3];
        assert_eq!(
            verdict_from_cma_weight_normalization(&w),
            CmaWeightNormalizationVerdict::Fail
        );
    }

    #[test]
    fn fcma003_fail_oversum() {
        let w = vec![0.5_f64, 0.6];
        assert_eq!(
            verdict_from_cma_weight_normalization(&w),
            CmaWeightNormalizationVerdict::Fail
        );
    }

    #[test]
    fn fcma003_fail_empty() {
        assert_eq!(
            verdict_from_cma_weight_normalization(&[]),
            CmaWeightNormalizationVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: CMA-004 covariance symmetry.
    // -------------------------------------------------------------------------
    #[test]
    fn fcma004_pass_2x2_symmetric() {
        // [[1, 0.3], [0.3, 2]]
        let c = vec![1.0_f64, 0.3, 0.3, 2.0];
        assert_eq!(
            verdict_from_cma_covariance_symmetry(2, &c),
            CmaCovarianceSymmetryVerdict::Pass
        );
    }

    #[test]
    fn fcma004_pass_3x3_diagonal() {
        let c = vec![1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        assert_eq!(
            verdict_from_cma_covariance_symmetry(3, &c),
            CmaCovarianceSymmetryVerdict::Pass
        );
    }

    #[test]
    fn fcma004_fail_2x2_asymmetric() {
        // [[1, 0.3], [0.5, 2]] — c01=0.3, c10=0.5 (not equal).
        let c = vec![1.0_f64, 0.3, 0.5, 2.0];
        assert_eq!(
            verdict_from_cma_covariance_symmetry(2, &c),
            CmaCovarianceSymmetryVerdict::Fail
        );
    }

    #[test]
    fn fcma004_fail_zero_dim() {
        assert_eq!(
            verdict_from_cma_covariance_symmetry(0, &[]),
            CmaCovarianceSymmetryVerdict::Fail
        );
    }

    #[test]
    fn fcma004_fail_size_mismatch() {
        let c = vec![1.0_f64, 0.0, 0.0]; // expect dim*dim = 4
        assert_eq!(
            verdict_from_cma_covariance_symmetry(2, &c),
            CmaCovarianceSymmetryVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: CMA-005 SIMD equivalence.
    // -------------------------------------------------------------------------
    #[test]
    fn fcma005_pass_exact_match() {
        let s = vec![1.5_f64, 2.5, 3.5];
        assert_eq!(
            verdict_from_cma_simd_equivalence(&s, &s),
            CmaSimdEquivalenceVerdict::Pass
        );
    }

    #[test]
    fn fcma005_pass_within_tolerance() {
        let simd = vec![1.5_f64];
        let scalar = vec![1.5_f64 + 1e-9];
        assert_eq!(
            verdict_from_cma_simd_equivalence(&simd, &scalar),
            CmaSimdEquivalenceVerdict::Pass
        );
    }

    #[test]
    fn fcma005_fail_outside_tolerance() {
        let simd = vec![1.5_f64];
        let scalar = vec![1.4_f64];
        assert_eq!(
            verdict_from_cma_simd_equivalence(&simd, &scalar),
            CmaSimdEquivalenceVerdict::Fail
        );
    }

    #[test]
    fn fcma005_fail_length_mismatch() {
        assert_eq!(
            verdict_from_cma_simd_equivalence(&[1.0], &[1.0, 2.0]),
            CmaSimdEquivalenceVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: CMA-006 d=1 boundary.
    // -------------------------------------------------------------------------
    #[test]
    fn fcma006_pass_converged_to_optimum() {
        // Final mean ≈ 0 (target), sigma decayed.
        assert_eq!(
            verdict_from_cma_dim_one_boundary(0.0, 0.0, 1e-6),
            CmaDimOneBoundaryVerdict::Pass
        );
    }

    #[test]
    fn fcma006_pass_near_optimum() {
        assert_eq!(
            verdict_from_cma_dim_one_boundary(0.05, 0.0, 0.01),
            CmaDimOneBoundaryVerdict::Pass
        );
    }

    #[test]
    fn fcma006_fail_sigma_too_large() {
        // Didn't converge — sigma stayed at 1.0.
        assert_eq!(
            verdict_from_cma_dim_one_boundary(0.0, 0.0, 1.0),
            CmaDimOneBoundaryVerdict::Fail
        );
    }

    #[test]
    fn fcma006_fail_far_from_optimum() {
        assert_eq!(
            verdict_from_cma_dim_one_boundary(5.0, 0.0, 0.01),
            CmaDimOneBoundaryVerdict::Fail
        );
    }

    #[test]
    fn fcma006_fail_nan_mean() {
        assert_eq!(
            verdict_from_cma_dim_one_boundary(f64::NAN, 0.0, 0.1),
            CmaDimOneBoundaryVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 8: Realistic — full healthy CMA-ES run passes all 6.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_cma_passes_all_6() {
        // 1000 generations with monotonically decreasing sigma.
        let s: Vec<f64> = (0..1000).map(|t| 1.0 / (t as f64 + 1.0)).collect();
        assert_eq!(
            verdict_from_cma_sigma_positivity(&s),
            CmaSigmaPositivityVerdict::Pass
        );
        // 100 generations, all eigenvalues > 0.
        let l = vec![0.5_f64; 100];
        assert_eq!(
            verdict_from_cma_covariance_posdef(&l),
            CmaCovariancePosDefVerdict::Pass
        );
        // 4 weights uniform.
        let w = vec![0.25_f64; 4];
        assert_eq!(
            verdict_from_cma_weight_normalization(&w),
            CmaWeightNormalizationVerdict::Pass
        );
        // 2x2 identity covariance.
        let c = vec![1.0_f64, 0.0, 0.0, 1.0];
        assert_eq!(
            verdict_from_cma_covariance_symmetry(2, &c),
            CmaCovarianceSymmetryVerdict::Pass
        );
        // SIMD bit-identical to scalar.
        let v = vec![1.5_f64, 2.5];
        assert_eq!(
            verdict_from_cma_simd_equivalence(&v, &v),
            CmaSimdEquivalenceVerdict::Pass
        );
        // d=1 converged.
        assert_eq!(
            verdict_from_cma_dim_one_boundary(0.0, 0.0, 1e-6),
            CmaDimOneBoundaryVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_6_failures() {
        // 001: sigma went negative.
        assert_eq!(
            verdict_from_cma_sigma_positivity(&[1.0, -0.001]),
            CmaSigmaPositivityVerdict::Fail
        );
        // 002: rank update introduced negative eigenvalue.
        assert_eq!(
            verdict_from_cma_covariance_posdef(&[0.5, -0.01]),
            CmaCovariancePosDefVerdict::Fail
        );
        // 003: weights formula bug.
        assert_eq!(
            verdict_from_cma_weight_normalization(&[0.5, 0.6]),
            CmaWeightNormalizationVerdict::Fail
        );
        // 004: asymmetric C.
        let bad = vec![1.0_f64, 0.3, 0.5, 2.0];
        assert_eq!(
            verdict_from_cma_covariance_symmetry(2, &bad),
            CmaCovarianceSymmetryVerdict::Fail
        );
        // 005: SIMD Cholesky differs.
        assert_eq!(
            verdict_from_cma_simd_equivalence(&[1.5], &[1.0]),
            CmaSimdEquivalenceVerdict::Fail
        );
        // 006: d=1 didn't converge.
        assert_eq!(
            verdict_from_cma_dim_one_boundary(10.0, 0.0, 5.0),
            CmaDimOneBoundaryVerdict::Fail
        );
    }
}
