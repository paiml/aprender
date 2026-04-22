//! CRUX-C-22 — Typical-p (locally typical) sampling classifiers
//!
//! Discharges `contracts/crux-C-22-v1.yaml` FALSIFY gates at PARTIAL_ALGORITHM_LEVEL:
//! - FALSIFY-CRUX-C-22-001: p=1.0 is identity (no filtering)
//! - FALSIFY-CRUX-C-22-002: kept-token set matches HF `TypicalLogitsWarper`
//!
//! Algorithm (arXiv:2202.00666):
//!   p_i = softmax(logits)
//!   H   = -sum(p_i * log p_i)
//!   c_i = |-log p_i - H|
//!   sort tokens by c_i ASC; keep smallest-cumulative-mass set S s.t. sum >= p
//!   renormalize kept probs; discard rest

#![allow(dead_code)]

/// Minimum valid typical-p value (exclusive zero per arXiv:2202.00666).
pub(crate) const TYPICAL_P_MIN_EXCLUSIVE: f64 = 0.0;
/// Maximum valid typical-p value (inclusive).
pub(crate) const TYPICAL_P_MAX_INCLUSIVE: f64 = 1.0;
/// Renormalization sum tolerance (per contract v1.1.0).
pub(crate) const RENORM_TOLERANCE: f64 = 1e-6;

/// Parameter-range gate: p must be finite and in (0, 1].
#[derive(Debug, PartialEq)]
pub(crate) enum TypicalPRangeOutcome {
    Valid,
    NotFinite,
    BelowMinimum { p: f64 },
    AboveMaximum { p: f64 },
}

pub(crate) fn classify_typical_p_range(p: f64) -> TypicalPRangeOutcome {
    if !p.is_finite() {
        return TypicalPRangeOutcome::NotFinite;
    }
    if p <= TYPICAL_P_MIN_EXCLUSIVE {
        return TypicalPRangeOutcome::BelowMinimum { p };
    }
    if p > TYPICAL_P_MAX_INCLUSIVE {
        return TypicalPRangeOutcome::AboveMaximum { p };
    }
    TypicalPRangeOutcome::Valid
}

/// Identity gate (FALSIFY-CRUX-C-22-001): p=1.0 returns every token.
#[derive(Debug, PartialEq)]
pub(crate) enum IdentityOutcome {
    Ok {
        kept_count: usize,
        total_count: usize,
    },
    InvalidInput {
        reason: &'static str,
    },
    DroppedTokens {
        kept_count: usize,
        total_count: usize,
    },
}

pub(crate) fn classify_typical_p_identity(
    kept_indices: &[usize],
    total_tokens: usize,
    p: f64,
) -> IdentityOutcome {
    if total_tokens == 0 {
        return IdentityOutcome::InvalidInput {
            reason: "total_tokens == 0",
        };
    }
    if !p.is_finite() || (p - 1.0).abs() > f64::EPSILON {
        return IdentityOutcome::InvalidInput {
            reason: "p != 1.0",
        };
    }
    if kept_indices.len() != total_tokens {
        return IdentityOutcome::DroppedTokens {
            kept_count: kept_indices.len(),
            total_count: total_tokens,
        };
    }
    let mut seen = vec![false; total_tokens];
    for &idx in kept_indices {
        if idx >= total_tokens {
            return IdentityOutcome::InvalidInput {
                reason: "kept_index out of range",
            };
        }
        if seen[idx] {
            return IdentityOutcome::InvalidInput {
                reason: "duplicate kept_index",
            };
        }
        seen[idx] = true;
    }
    IdentityOutcome::Ok {
        kept_count: kept_indices.len(),
        total_count: total_tokens,
    }
}

/// Mass-coverage gate: kept probabilities must sum to ≥ p (minimal typical set).
#[derive(Debug, PartialEq)]
pub(crate) enum MassCoverageOutcome {
    Ok { kept_mass: f64 },
    InvalidInput { reason: &'static str },
    InsufficientMass { kept_mass: f64, required: f64 },
    TooLarge { kept_mass: f64, excess: f64 },
}

/// kept_probs are the ORIGINAL (pre-renormalization) probabilities
/// of the tokens kept by the typical-p filter.
pub(crate) fn classify_typical_p_mass_coverage(
    kept_probs: &[f64],
    p: f64,
) -> MassCoverageOutcome {
    if kept_probs.is_empty() {
        return MassCoverageOutcome::InvalidInput {
            reason: "kept_probs is empty",
        };
    }
    if !p.is_finite() || p <= 0.0 || p > 1.0 {
        return MassCoverageOutcome::InvalidInput {
            reason: "p out of (0, 1]",
        };
    }
    if !kept_probs.iter().all(|&x| x.is_finite() && (0.0..=1.0).contains(&x)) {
        return MassCoverageOutcome::InvalidInput {
            reason: "prob not in [0, 1]",
        };
    }
    let kept_mass: f64 = kept_probs.iter().sum();
    if kept_mass < p - RENORM_TOLERANCE {
        return MassCoverageOutcome::InsufficientMass {
            kept_mass,
            required: p,
        };
    }
    if kept_mass > 1.0 + RENORM_TOLERANCE {
        return MassCoverageOutcome::TooLarge {
            kept_mass,
            excess: kept_mass - 1.0,
        };
    }
    MassCoverageOutcome::Ok { kept_mass }
}

/// Renormalization gate: filtered distribution sums to 1.0 ± 1e-6.
#[derive(Debug, PartialEq)]
pub(crate) enum RenormOutcome {
    Ok { sum: f64 },
    InvalidInput { reason: &'static str },
    NotNormalized { sum: f64, deviation: f64 },
    ContainsNegative { first_bad_index: usize, value: f64 },
}

pub(crate) fn classify_typical_p_renormalization(filtered_probs: &[f64]) -> RenormOutcome {
    if filtered_probs.is_empty() {
        return RenormOutcome::InvalidInput {
            reason: "filtered_probs is empty",
        };
    }
    for (i, &x) in filtered_probs.iter().enumerate() {
        if !x.is_finite() {
            return RenormOutcome::InvalidInput {
                reason: "non-finite probability",
            };
        }
        if x < 0.0 {
            return RenormOutcome::ContainsNegative {
                first_bad_index: i,
                value: x,
            };
        }
    }
    let sum: f64 = filtered_probs.iter().sum();
    let deviation = (sum - 1.0).abs();
    if deviation > RENORM_TOLERANCE {
        return RenormOutcome::NotNormalized { sum, deviation };
    }
    RenormOutcome::Ok { sum }
}

/// Sort-order gate: kept tokens must be sorted by c_i = |−log p_i − H| ASC.
/// `kept_probs_in_sort_order` lists the original probabilities of kept tokens
/// in the order the filter emits them (should be c_i ASC).
#[derive(Debug, PartialEq)]
pub(crate) enum SortOrderOutcome {
    Ok,
    InvalidInput { reason: &'static str },
    OutOfOrder {
        at_index: usize,
        prev_c: f64,
        curr_c: f64,
    },
}

pub(crate) fn classify_typical_p_sort_order(
    all_probs: &[f64],
    kept_probs_in_sort_order: &[f64],
) -> SortOrderOutcome {
    if all_probs.is_empty() || kept_probs_in_sort_order.is_empty() {
        return SortOrderOutcome::InvalidInput {
            reason: "empty input",
        };
    }
    if !all_probs.iter().all(|&x| x.is_finite() && x > 0.0 && x <= 1.0) {
        return SortOrderOutcome::InvalidInput {
            reason: "all_probs must be strictly positive and finite",
        };
    }
    if !kept_probs_in_sort_order
        .iter()
        .all(|&x| x.is_finite() && x > 0.0 && x <= 1.0)
    {
        return SortOrderOutcome::InvalidInput {
            reason: "kept_probs must be strictly positive and finite",
        };
    }
    let entropy: f64 = -all_probs.iter().map(|&p| p * p.ln()).sum::<f64>();
    let c = |prob: f64| (-prob.ln() - entropy).abs();

    for i in 1..kept_probs_in_sort_order.len() {
        let prev_c = c(kept_probs_in_sort_order[i - 1]);
        let curr_c = c(kept_probs_in_sort_order[i]);
        if curr_c < prev_c - RENORM_TOLERANCE {
            return SortOrderOutcome::OutOfOrder {
                at_index: i,
                prev_c,
                curr_c,
            };
        }
    }
    SortOrderOutcome::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------- range --------

    #[test]
    fn range_valid_for_one() {
        assert_eq!(classify_typical_p_range(1.0), TypicalPRangeOutcome::Valid);
    }

    #[test]
    fn range_valid_for_half() {
        assert_eq!(classify_typical_p_range(0.5), TypicalPRangeOutcome::Valid);
    }

    #[test]
    fn range_valid_for_canonical_0_95() {
        assert_eq!(classify_typical_p_range(0.95), TypicalPRangeOutcome::Valid);
    }

    #[test]
    fn range_rejects_zero() {
        assert_eq!(
            classify_typical_p_range(0.0),
            TypicalPRangeOutcome::BelowMinimum { p: 0.0 }
        );
    }

    #[test]
    fn range_rejects_negative() {
        assert_eq!(
            classify_typical_p_range(-0.1),
            TypicalPRangeOutcome::BelowMinimum { p: -0.1 }
        );
    }

    #[test]
    fn range_rejects_above_one() {
        assert_eq!(
            classify_typical_p_range(1.5),
            TypicalPRangeOutcome::AboveMaximum { p: 1.5 }
        );
    }

    #[test]
    fn range_rejects_nan() {
        assert_eq!(
            classify_typical_p_range(f64::NAN),
            TypicalPRangeOutcome::NotFinite
        );
    }

    #[test]
    fn range_rejects_infinity() {
        assert_eq!(
            classify_typical_p_range(f64::INFINITY),
            TypicalPRangeOutcome::NotFinite
        );
    }

    // -------- identity (p = 1.0) --------

    #[test]
    fn identity_ok_when_all_kept() {
        let kept: Vec<usize> = (0..4).collect();
        assert_eq!(
            classify_typical_p_identity(&kept, 4, 1.0),
            IdentityOutcome::Ok {
                kept_count: 4,
                total_count: 4,
            }
        );
    }

    #[test]
    fn identity_ok_order_insensitive() {
        let kept = vec![3, 1, 0, 2];
        assert_eq!(
            classify_typical_p_identity(&kept, 4, 1.0),
            IdentityOutcome::Ok {
                kept_count: 4,
                total_count: 4,
            }
        );
    }

    #[test]
    fn identity_flags_dropped_tokens() {
        let kept = vec![0, 1, 2];
        assert_eq!(
            classify_typical_p_identity(&kept, 4, 1.0),
            IdentityOutcome::DroppedTokens {
                kept_count: 3,
                total_count: 4,
            }
        );
    }

    #[test]
    fn identity_rejects_non_unity_p() {
        let kept = vec![0, 1];
        assert_eq!(
            classify_typical_p_identity(&kept, 2, 0.95),
            IdentityOutcome::InvalidInput { reason: "p != 1.0" }
        );
    }

    #[test]
    fn identity_rejects_oob_index() {
        let kept = vec![0, 5];
        assert_eq!(
            classify_typical_p_identity(&kept, 2, 1.0),
            IdentityOutcome::InvalidInput {
                reason: "kept_index out of range"
            }
        );
    }

    #[test]
    fn identity_rejects_duplicate_index() {
        let kept = vec![0, 1, 1];
        assert_eq!(
            classify_typical_p_identity(&kept, 3, 1.0),
            IdentityOutcome::InvalidInput {
                reason: "duplicate kept_index"
            }
        );
    }

    #[test]
    fn identity_rejects_empty_total() {
        assert_eq!(
            classify_typical_p_identity(&[], 0, 1.0),
            IdentityOutcome::InvalidInput {
                reason: "total_tokens == 0"
            }
        );
    }

    // -------- mass coverage --------

    #[test]
    fn mass_ok_when_kept_meets_threshold() {
        let kept = vec![0.4, 0.3, 0.25];
        assert_eq!(
            classify_typical_p_mass_coverage(&kept, 0.95),
            MassCoverageOutcome::Ok { kept_mass: 0.95 }
        );
    }

    #[test]
    fn mass_rejects_under_threshold() {
        let kept = vec![0.3, 0.2];
        let outcome = classify_typical_p_mass_coverage(&kept, 0.95);
        match outcome {
            MassCoverageOutcome::InsufficientMass { kept_mass, required } => {
                assert!((kept_mass - 0.5).abs() < 1e-9);
                assert!((required - 0.95).abs() < 1e-9);
            }
            other => panic!("expected InsufficientMass, got {other:?}"),
        }
    }

    #[test]
    fn mass_rejects_above_one() {
        let kept = vec![0.6, 0.6];
        match classify_typical_p_mass_coverage(&kept, 0.95) {
            MassCoverageOutcome::TooLarge { kept_mass, excess } => {
                assert!((kept_mass - 1.2).abs() < 1e-9);
                assert!((excess - 0.2).abs() < 1e-9);
            }
            other => panic!("expected TooLarge, got {other:?}"),
        }
    }

    #[test]
    fn mass_rejects_empty() {
        assert_eq!(
            classify_typical_p_mass_coverage(&[], 0.95),
            MassCoverageOutcome::InvalidInput {
                reason: "kept_probs is empty"
            }
        );
    }

    #[test]
    fn mass_rejects_invalid_p_zero() {
        assert_eq!(
            classify_typical_p_mass_coverage(&[0.5], 0.0),
            MassCoverageOutcome::InvalidInput {
                reason: "p out of (0, 1]"
            }
        );
    }

    #[test]
    fn mass_rejects_prob_above_one() {
        assert_eq!(
            classify_typical_p_mass_coverage(&[1.5], 0.95),
            MassCoverageOutcome::InvalidInput {
                reason: "prob not in [0, 1]"
            }
        );
    }

    #[test]
    fn mass_rejects_negative_prob() {
        assert_eq!(
            classify_typical_p_mass_coverage(&[-0.1, 0.9], 0.5),
            MassCoverageOutcome::InvalidInput {
                reason: "prob not in [0, 1]"
            }
        );
    }

    #[test]
    fn mass_rejects_nan_prob() {
        assert_eq!(
            classify_typical_p_mass_coverage(&[f64::NAN], 0.5),
            MassCoverageOutcome::InvalidInput {
                reason: "prob not in [0, 1]"
            }
        );
    }

    // -------- renormalization --------

    #[test]
    fn renorm_ok_when_sums_to_one() {
        let probs = vec![0.4, 0.3, 0.3];
        assert_eq!(
            classify_typical_p_renormalization(&probs),
            RenormOutcome::Ok { sum: 1.0 }
        );
    }

    #[test]
    fn renorm_ok_within_tolerance() {
        let probs = vec![0.333_333, 0.333_333, 0.333_334];
        match classify_typical_p_renormalization(&probs) {
            RenormOutcome::Ok { sum } => assert!((sum - 1.0).abs() <= RENORM_TOLERANCE),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn renorm_flags_under_normalization() {
        let probs = vec![0.4, 0.3, 0.2];
        match classify_typical_p_renormalization(&probs) {
            RenormOutcome::NotNormalized { sum, deviation } => {
                assert!((sum - 0.9).abs() < 1e-9);
                assert!((deviation - 0.1).abs() < 1e-9);
            }
            other => panic!("expected NotNormalized, got {other:?}"),
        }
    }

    #[test]
    fn renorm_flags_negative_prob() {
        let probs = vec![0.5, -0.1, 0.6];
        assert_eq!(
            classify_typical_p_renormalization(&probs),
            RenormOutcome::ContainsNegative {
                first_bad_index: 1,
                value: -0.1,
            }
        );
    }

    #[test]
    fn renorm_rejects_nan() {
        assert_eq!(
            classify_typical_p_renormalization(&[f64::NAN, 0.5]),
            RenormOutcome::InvalidInput {
                reason: "non-finite probability"
            }
        );
    }

    #[test]
    fn renorm_rejects_infinity() {
        assert_eq!(
            classify_typical_p_renormalization(&[f64::INFINITY, 0.0]),
            RenormOutcome::InvalidInput {
                reason: "non-finite probability"
            }
        );
    }

    #[test]
    fn renorm_rejects_empty() {
        assert_eq!(
            classify_typical_p_renormalization(&[]),
            RenormOutcome::InvalidInput {
                reason: "filtered_probs is empty"
            }
        );
    }

    // -------- sort order by c_i --------

    #[test]
    fn sort_ok_when_kept_empty_is_rejected() {
        let all = vec![0.25, 0.25, 0.25, 0.25];
        assert_eq!(
            classify_typical_p_sort_order(&all, &[]),
            SortOrderOutcome::InvalidInput { reason: "empty input" }
        );
    }

    #[test]
    fn sort_ok_uniform_distribution_any_order() {
        // Uniform distribution: H = -ln(1/4) = ln 4; c_i = |-ln(0.25) - ln 4| = 0 for all.
        // Any order is equally valid because all c_i are equal.
        let all = vec![0.25; 4];
        let kept = vec![0.25, 0.25, 0.25, 0.25];
        assert_eq!(
            classify_typical_p_sort_order(&all, &kept),
            SortOrderOutcome::Ok
        );
    }

    #[test]
    fn sort_ok_when_kept_ordered_ascending_by_c() {
        // Skewed dist: one dominant token has lowest c_i.
        // all = [0.7, 0.15, 0.1, 0.05]
        // H = -sum(p ln p). Typical tokens are those with p closest to exp(-H).
        // Compute c_i for each:
        //   -ln(0.7)=0.357, -ln(0.15)=1.897, -ln(0.1)=2.303, -ln(0.05)=2.996
        //   H = 0.7*0.357 + 0.15*1.897 + 0.1*2.303 + 0.05*2.996
        //     = 0.2499 + 0.2846 + 0.2303 + 0.1498 ≈ 0.9146
        //   c_i: |0.357-0.915|=0.558, |1.897-0.915|=0.982,
        //        |2.303-0.915|=1.388, |2.996-0.915|=2.081
        // Sorted c_i ASC: 0.7, 0.15, 0.1, 0.05
        let all = vec![0.7, 0.15, 0.1, 0.05];
        let kept = vec![0.7, 0.15, 0.1, 0.05];
        assert_eq!(
            classify_typical_p_sort_order(&all, &kept),
            SortOrderOutcome::Ok
        );
    }

    #[test]
    fn sort_flags_descending_order() {
        // Reverse of the skewed dist above: c_i DESC which is wrong direction.
        let all = vec![0.7, 0.15, 0.1, 0.05];
        let kept = vec![0.05, 0.1, 0.15, 0.7];
        match classify_typical_p_sort_order(&all, &kept) {
            SortOrderOutcome::OutOfOrder {
                at_index,
                prev_c,
                curr_c,
            } => {
                assert_eq!(at_index, 1);
                assert!(prev_c > curr_c);
            }
            other => panic!("expected OutOfOrder, got {other:?}"),
        }
    }

    #[test]
    fn sort_rejects_zero_prob() {
        let all = vec![0.5, 0.5];
        let kept = vec![0.0];
        assert_eq!(
            classify_typical_p_sort_order(&all, &kept),
            SortOrderOutcome::InvalidInput {
                reason: "kept_probs must be strictly positive and finite"
            }
        );
    }

    #[test]
    fn sort_rejects_nan_kept() {
        let all = vec![0.5, 0.5];
        let kept = vec![f64::NAN];
        assert_eq!(
            classify_typical_p_sort_order(&all, &kept),
            SortOrderOutcome::InvalidInput {
                reason: "kept_probs must be strictly positive and finite"
            }
        );
    }

    #[test]
    fn sort_rejects_nan_all_probs() {
        let all = vec![f64::NAN, 0.5];
        let kept = vec![0.5];
        assert_eq!(
            classify_typical_p_sort_order(&all, &kept),
            SortOrderOutcome::InvalidInput {
                reason: "all_probs must be strictly positive and finite"
            }
        );
    }

    #[test]
    fn sort_rejects_empty_all_probs() {
        assert_eq!(
            classify_typical_p_sort_order(&[], &[0.5]),
            SortOrderOutcome::InvalidInput { reason: "empty input" }
        );
    }

    // -------- integration-level properties --------

    #[test]
    fn identity_p_one_and_mass_coverage_coincide() {
        // When p=1.0, mass coverage requires kept_mass >= 1.0, which means
        // ALL tokens must be kept → consistent with IdentityOutcome::Ok.
        let probs = vec![0.4, 0.3, 0.2, 0.1];
        let indices: Vec<usize> = (0..probs.len()).collect();

        assert_eq!(
            classify_typical_p_identity(&indices, probs.len(), 1.0),
            IdentityOutcome::Ok {
                kept_count: 4,
                total_count: 4
            }
        );
        match classify_typical_p_mass_coverage(&probs, 1.0) {
            MassCoverageOutcome::Ok { kept_mass } => {
                assert!((kept_mass - 1.0).abs() < 1e-9);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn renormalized_two_token_distribution_parity() {
        // Original kept probs: [0.4, 0.3]; renormalized: [4/7, 3/7]
        let renormalized = vec![4.0 / 7.0, 3.0 / 7.0];
        match classify_typical_p_renormalization(&renormalized) {
            RenormOutcome::Ok { sum } => assert!((sum - 1.0).abs() <= RENORM_TOLERANCE),
            other => panic!("expected Ok, got {other:?}"),
        }
    }
}
