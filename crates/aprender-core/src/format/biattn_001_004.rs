// `bidirectional-attention-v1` algorithm-level PARTIAL discharge for the
// 4 BERT-style bidirectional-attention falsifiers (no causal mask, n=1
// causal parity, weight normalization, full density).
//
// Contract: `contracts/bidirectional-attention-v1.yaml`.
// Refs: Devlin et al. (2019) BERT.

/// Tolerance for "weights sum to 1" check (1e-5).
pub const AC_BIATTN_NORM_TOLERANCE: f32 = 1.0e-5;

/// Tolerance for n=1 causal-parity check (1e-6).
pub const AC_BIATTN_CAUSAL_PARITY_TOLERANCE: f32 = 1.0e-6;

// =============================================================================
// FALSIFY-BIATT-001 — no causal mask (upper triangle non-zero)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiattnNoCausalMaskVerdict {
    /// All entries attn_weights[i, j] > 0 (no zero from masking).
    Pass,
    /// Some upper-triangle entry is zero — causal mask leaked.
    Fail,
}

#[must_use]
pub fn verdict_from_biattn_no_causal_mask(
    n: usize,
    attn_weights: &[f32],
) -> BiattnNoCausalMaskVerdict {
    if n == 0 {
        return BiattnNoCausalMaskVerdict::Fail;
    }
    if attn_weights.len() != n * n {
        return BiattnNoCausalMaskVerdict::Fail;
    }
    for i in 0..n {
        for j in 0..n {
            if i < j {
                let w = attn_weights[i * n + j];
                if !w.is_finite() {
                    return BiattnNoCausalMaskVerdict::Fail;
                }
                if w <= 0.0 {
                    return BiattnNoCausalMaskVerdict::Fail;
                }
            }
        }
    }
    BiattnNoCausalMaskVerdict::Pass
}

// =============================================================================
// FALSIFY-BIATT-002 — n=1 causal parity
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiattnCausalParityVerdict {
    /// At n=1: |BiAttn(q,k,v) - CausalAttn(q,k,v)| < 1e-6.
    Pass,
    /// Mask application differs even when mask is trivial.
    Fail,
}

#[must_use]
pub fn verdict_from_biattn_causal_parity(
    bi_output: &[f32],
    causal_output: &[f32],
) -> BiattnCausalParityVerdict {
    if bi_output.len() != causal_output.len() {
        return BiattnCausalParityVerdict::Fail;
    }
    if bi_output.is_empty() {
        return BiattnCausalParityVerdict::Fail;
    }
    for (a, b) in bi_output.iter().zip(causal_output.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return BiattnCausalParityVerdict::Fail;
        }
        if (a - b).abs() >= AC_BIATTN_CAUSAL_PARITY_TOLERANCE {
            return BiattnCausalParityVerdict::Fail;
        }
    }
    BiattnCausalParityVerdict::Pass
}

// =============================================================================
// FALSIFY-BIATT-003 — weight normalization (each row sums to 1.0)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiattnWeightNormalizationVerdict {
    /// |sum(attn_weights[i, :]) - 1.0| < 1e-5 ∀ i.
    Pass,
    /// At least one row doesn't sum to 1 — softmax bug.
    Fail,
}

#[must_use]
pub fn verdict_from_biattn_weight_normalization(
    n: usize,
    attn_weights: &[f32],
) -> BiattnWeightNormalizationVerdict {
    if n == 0 {
        return BiattnWeightNormalizationVerdict::Fail;
    }
    if attn_weights.len() != n * n {
        return BiattnWeightNormalizationVerdict::Fail;
    }
    for i in 0..n {
        let row_sum: f32 = (0..n).map(|j| attn_weights[i * n + j]).sum();
        if !row_sum.is_finite() {
            return BiattnWeightNormalizationVerdict::Fail;
        }
        if (row_sum - 1.0).abs() >= AC_BIATTN_NORM_TOLERANCE {
            return BiattnWeightNormalizationVerdict::Fail;
        }
    }
    BiattnWeightNormalizationVerdict::Pass
}

// =============================================================================
// FALSIFY-BIATT-004 — full attention density (all entries > 0)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiattnFullDensityVerdict {
    /// All n×n attention weights are strictly positive.
    Pass,
    /// At least one entry is zero or non-positive — sparse attention bug.
    Fail,
}

#[must_use]
pub fn verdict_from_biattn_full_density(
    n: usize,
    attn_weights: &[f32],
) -> BiattnFullDensityVerdict {
    if n == 0 {
        return BiattnFullDensityVerdict::Fail;
    }
    if attn_weights.len() != n * n {
        return BiattnFullDensityVerdict::Fail;
    }
    for &w in attn_weights {
        if !w.is_finite() {
            return BiattnFullDensityVerdict::Fail;
        }
        if w <= 0.0 {
            return BiattnFullDensityVerdict::Fail;
        }
    }
    BiattnFullDensityVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_norm_tolerance_1e_neg5() {
        assert!((AC_BIATTN_NORM_TOLERANCE - 1.0e-5).abs() < f32::EPSILON);
    }

    #[test]
    fn provenance_causal_parity_tolerance_1e_neg6() {
        assert!((AC_BIATTN_CAUSAL_PARITY_TOLERANCE - 1.0e-6).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Section 2: BIATT-001 no causal mask.
    // -------------------------------------------------------------------------
    #[test]
    fn fbi001_pass_dense_3x3() {
        // All entries 1/3 (uniform attention).
        let w = vec![0.333_f32; 9];
        assert_eq!(
            verdict_from_biattn_no_causal_mask(3, &w),
            BiattnNoCausalMaskVerdict::Pass
        );
    }

    #[test]
    fn fbi001_fail_upper_triangle_zero() {
        // i=0, j=1 (upper) is zero — causal mask leaked.
        let w = vec![
            0.5_f32, 0.0, 0.0,
            0.3, 0.7, 0.0,
            0.1, 0.4, 0.5,
        ];
        assert_eq!(
            verdict_from_biattn_no_causal_mask(3, &w),
            BiattnNoCausalMaskVerdict::Fail
        );
    }

    #[test]
    fn fbi001_fail_zero_n() {
        assert_eq!(
            verdict_from_biattn_no_causal_mask(0, &[]),
            BiattnNoCausalMaskVerdict::Fail
        );
    }

    #[test]
    fn fbi001_fail_size_mismatch() {
        let w = vec![1.0_f32, 2.0]; // expect 9
        assert_eq!(
            verdict_from_biattn_no_causal_mask(3, &w),
            BiattnNoCausalMaskVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: BIATT-002 n=1 causal parity.
    // -------------------------------------------------------------------------
    #[test]
    fn fbi002_pass_n1_identical() {
        let bi = vec![1.5_f32, 2.5];
        let causal = vec![1.5_f32, 2.5];
        assert_eq!(
            verdict_from_biattn_causal_parity(&bi, &causal),
            BiattnCausalParityVerdict::Pass
        );
    }

    #[test]
    fn fbi002_pass_within_tolerance() {
        let bi = vec![1.5_f32 + 1e-7];
        let causal = vec![1.5_f32];
        assert_eq!(
            verdict_from_biattn_causal_parity(&bi, &causal),
            BiattnCausalParityVerdict::Pass
        );
    }

    #[test]
    fn fbi002_fail_outside_tolerance() {
        let bi = vec![2.0_f32];
        let causal = vec![1.0_f32];
        assert_eq!(
            verdict_from_biattn_causal_parity(&bi, &causal),
            BiattnCausalParityVerdict::Fail
        );
    }

    #[test]
    fn fbi002_fail_length_mismatch() {
        assert_eq!(
            verdict_from_biattn_causal_parity(&[1.0], &[1.0, 2.0]),
            BiattnCausalParityVerdict::Fail
        );
    }

    #[test]
    fn fbi002_fail_empty() {
        assert_eq!(
            verdict_from_biattn_causal_parity(&[], &[]),
            BiattnCausalParityVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: BIATT-003 weight normalization.
    // -------------------------------------------------------------------------
    #[test]
    fn fbi003_pass_uniform_3x3() {
        // 1/3 each, sum = 1.
        let w = vec![1.0_f32 / 3.0; 9];
        assert_eq!(
            verdict_from_biattn_weight_normalization(3, &w),
            BiattnWeightNormalizationVerdict::Pass
        );
    }

    #[test]
    fn fbi003_pass_softmax_distribution_2x2() {
        let w = vec![
            0.3_f32, 0.7,
            0.6, 0.4,
        ];
        assert_eq!(
            verdict_from_biattn_weight_normalization(2, &w),
            BiattnWeightNormalizationVerdict::Pass
        );
    }

    #[test]
    fn fbi003_fail_undersum() {
        let w = vec![
            0.3_f32, 0.3, // sum = 0.6
            0.5, 0.5,
        ];
        assert_eq!(
            verdict_from_biattn_weight_normalization(2, &w),
            BiattnWeightNormalizationVerdict::Fail
        );
    }

    #[test]
    fn fbi003_fail_oversum() {
        let w = vec![
            0.5_f32, 0.6, // sum = 1.1
            0.5, 0.5,
        ];
        assert_eq!(
            verdict_from_biattn_weight_normalization(2, &w),
            BiattnWeightNormalizationVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: BIATT-004 full density.
    // -------------------------------------------------------------------------
    #[test]
    fn fbi004_pass_all_positive() {
        let w = vec![0.25_f32; 16];
        assert_eq!(
            verdict_from_biattn_full_density(4, &w),
            BiattnFullDensityVerdict::Pass
        );
    }

    #[test]
    fn fbi004_fail_zero_entry() {
        let w = vec![
            0.5_f32, 0.5,
            0.0, 1.0, // top-left zero
        ];
        assert_eq!(
            verdict_from_biattn_full_density(2, &w),
            BiattnFullDensityVerdict::Fail
        );
    }

    #[test]
    fn fbi004_fail_negative_entry() {
        let w = vec![0.5_f32, 0.5, -0.1, 1.1];
        assert_eq!(
            verdict_from_biattn_full_density(2, &w),
            BiattnFullDensityVerdict::Fail
        );
    }

    #[test]
    fn fbi004_fail_nan_entry() {
        let w = vec![0.5_f32, f32::NAN, 0.5, 0.5];
        assert_eq!(
            verdict_from_biattn_full_density(2, &w),
            BiattnFullDensityVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic — full healthy bidirectional attention.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_biattn_passes_all_4() {
        // 4×4 nearly-uniform attention (typical BERT-style layer 0).
        let w = vec![0.25_f32; 16];
        assert_eq!(
            verdict_from_biattn_no_causal_mask(4, &w),
            BiattnNoCausalMaskVerdict::Pass
        );
        // n=1 causal parity (single token, mask trivial).
        let bi = vec![1.0_f32, 2.0];
        let causal = vec![1.0_f32, 2.0];
        assert_eq!(
            verdict_from_biattn_causal_parity(&bi, &causal),
            BiattnCausalParityVerdict::Pass
        );
        // Each row sums to 1.
        assert_eq!(
            verdict_from_biattn_weight_normalization(4, &w),
            BiattnWeightNormalizationVerdict::Pass
        );
        // All entries > 0.
        assert_eq!(
            verdict_from_biattn_full_density(4, &w),
            BiattnFullDensityVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_4_failures() {
        // 001: causal mask leaked into BERT path.
        let masked = vec![
            0.5_f32, 0.0, 0.0, 0.0,
            0.3, 0.7, 0.0, 0.0,
            0.1, 0.3, 0.6, 0.0,
            0.1, 0.2, 0.3, 0.4,
        ];
        assert_eq!(
            verdict_from_biattn_no_causal_mask(4, &masked),
            BiattnNoCausalMaskVerdict::Fail
        );
        // 002: BiAttn diverged from CausalAttn at n=1.
        assert_eq!(
            verdict_from_biattn_causal_parity(&[2.0], &[1.0]),
            BiattnCausalParityVerdict::Fail
        );
        // 003: rows don't sum to 1 (softmax skipped).
        let bad = vec![
            2.0_f32, 3.0,
            1.0, 1.0,
        ];
        assert_eq!(
            verdict_from_biattn_weight_normalization(2, &bad),
            BiattnWeightNormalizationVerdict::Fail
        );
        // 004: zero entries from sparse-attention bug.
        let sparse = vec![
            0.5_f32, 0.0,
            0.5, 0.5,
        ];
        assert_eq!(
            verdict_from_biattn_full_density(2, &sparse),
            BiattnFullDensityVerdict::Fail
        );
    }
}
