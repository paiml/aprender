//! CRUX-C-23 — DRY (Don't Repeat Yourself) sampling classifiers
//!
//! Discharges `contracts/crux-C-23-v1.yaml` FALSIFY gates at PARTIAL_ALGORITHM_LEVEL:
//! - FALSIFY-CRUX-C-23-001: multiplier=0 is identity (penalty disabled)
//! - FALSIFY-CRUX-C-23-002: DRY reduces exact-phrase repetition
//!
//! Algorithm (llama.cpp DRY sampler):
//!   for candidate token t:
//!     match_len = longest suffix-match of (ctx + [t]) ending earlier in ctx
//!     if match_len >= allowed_length:
//!       penalty = multiplier * base^(match_len - allowed_length)
//!       logit[t] -= penalty

#![allow(dead_code)]

use std::collections::HashSet;

pub(crate) const DEFAULT_DRY_MULTIPLIER: f64 = 0.8;
pub(crate) const DEFAULT_DRY_BASE: f64 = 1.75;
pub(crate) const DEFAULT_DRY_ALLOWED_LENGTH: u32 = 2;

/// Parameter-range gate: multiplier ≥ 0, base ≥ 1, allowed_length ≥ 1.
#[derive(Debug, PartialEq)]
pub(crate) enum DryParamOutcome {
    Valid,
    NotFinite { field: &'static str },
    MultiplierNegative { multiplier: f64 },
    BaseBelowOne { base: f64 },
    AllowedLengthZero,
}

pub(crate) fn classify_dry_params(
    multiplier: f64,
    base: f64,
    allowed_length: u32,
) -> DryParamOutcome {
    if !multiplier.is_finite() {
        return DryParamOutcome::NotFinite {
            field: "multiplier",
        };
    }
    if !base.is_finite() {
        return DryParamOutcome::NotFinite { field: "base" };
    }
    if multiplier < 0.0 {
        return DryParamOutcome::MultiplierNegative { multiplier };
    }
    if base < 1.0 {
        return DryParamOutcome::BaseBelowOne { base };
    }
    if allowed_length == 0 {
        return DryParamOutcome::AllowedLengthZero;
    }
    DryParamOutcome::Valid
}

/// Identity gate (FALSIFY-CRUX-C-23-001): multiplier=0 → every token unchanged.
#[derive(Debug, PartialEq)]
pub(crate) enum IdentityOutcome {
    Ok,
    InvalidInput {
        reason: &'static str,
    },
    LogitsChanged {
        first_diff_index: usize,
        before: f64,
        after: f64,
    },
}

pub(crate) fn classify_dry_identity_zero_multiplier(
    logits_before: &[f64],
    logits_after: &[f64],
    multiplier: f64,
) -> IdentityOutcome {
    if logits_before.is_empty() {
        return IdentityOutcome::InvalidInput {
            reason: "logits_before is empty",
        };
    }
    if logits_before.len() != logits_after.len() {
        return IdentityOutcome::InvalidInput {
            reason: "logits length mismatch",
        };
    }
    if !multiplier.is_finite() || multiplier != 0.0 {
        return IdentityOutcome::InvalidInput {
            reason: "multiplier != 0.0",
        };
    }
    for (i, (&b, &a)) in logits_before.iter().zip(logits_after.iter()).enumerate() {
        if !b.is_finite() || !a.is_finite() {
            return IdentityOutcome::InvalidInput {
                reason: "non-finite logit",
            };
        }
        if (b - a).abs() > f64::EPSILON * b.abs().max(1.0) {
            return IdentityOutcome::LogitsChanged {
                first_diff_index: i,
                before: b,
                after: a,
            };
        }
    }
    IdentityOutcome::Ok
}

/// Computes the longest suffix of (ctx + [candidate]) that matches a substring
/// ending earlier in ctx. This is the classifier-level specification of
/// llama.cpp's DRY match_len.
///
/// Returns 0 if no match ≥ 1 token is found.
pub(crate) fn classify_dry_match_len(
    ctx: &[u32],
    candidate: u32,
    seq_breakers: &HashSet<u32>,
) -> u32 {
    // Build extended context with candidate appended.
    let mut ext: Vec<u32> = ctx.to_vec();
    ext.push(candidate);

    // Walk back looking for a suffix-of-ext match ending earlier in ctx.
    // max_len can be at most ctx.len() (match must end BEFORE position ctx.len()).
    let ctx_len = ctx.len();
    if ctx_len == 0 {
        return 0;
    }
    let mut best: u32 = 0;

    // For each possible earlier-ending position j < ctx_len:
    //   match_len = longest L s.t. ext[ext.len()-L ..] == ctx[j+1-L .. j+1]
    // Scan j from ctx_len-1 down; stop if we see a seq_breaker at ctx[j].
    for j in (0..ctx_len).rev() {
        if seq_breakers.contains(&ctx[j]) {
            // Seq breaker resets counter for THIS branch; skip to earlier j.
            continue;
        }
        // Compare ctx[..=j] tail with ext tail.
        let mut l: usize = 0;
        loop {
            let ext_idx = ext.len() - 1 - l;
            let ctx_idx_opt = j.checked_sub(l);
            match ctx_idx_opt {
                None => break,
                Some(ctx_idx) => {
                    if seq_breakers.contains(&ctx[ctx_idx]) {
                        break;
                    }
                    if ext[ext_idx] != ctx[ctx_idx] {
                        break;
                    }
                    l += 1;
                    if ext_idx == 0 {
                        break;
                    }
                }
            }
        }
        let l_u32 = u32::try_from(l).unwrap_or(u32::MAX);
        if l_u32 > best {
            best = l_u32;
        }
    }
    best
}

/// Penalty-formula gate: penalty = multiplier * base^(match_len - allowed_length)
/// when match_len >= allowed_length; 0 otherwise; penalty ≥ 0 always.
#[derive(Debug, PartialEq)]
pub(crate) enum PenaltyOutcome {
    Ok { penalty: f64 },
    InvalidInput { reason: &'static str },
    Negative { penalty: f64 },
}

pub(crate) fn classify_dry_penalty(
    match_len: u32,
    allowed_length: u32,
    multiplier: f64,
    base: f64,
) -> PenaltyOutcome {
    if !multiplier.is_finite() || !base.is_finite() {
        return PenaltyOutcome::InvalidInput {
            reason: "non-finite multiplier or base",
        };
    }
    if multiplier < 0.0 {
        return PenaltyOutcome::InvalidInput {
            reason: "multiplier negative",
        };
    }
    if base < 1.0 {
        return PenaltyOutcome::InvalidInput {
            reason: "base < 1.0",
        };
    }
    if allowed_length == 0 {
        return PenaltyOutcome::InvalidInput {
            reason: "allowed_length == 0",
        };
    }
    if match_len < allowed_length {
        // Below threshold — penalty is zero by spec.
        return PenaltyOutcome::Ok { penalty: 0.0 };
    }
    let exponent = f64::from(match_len - allowed_length);
    let penalty = multiplier * base.powf(exponent);
    if !penalty.is_finite() {
        return PenaltyOutcome::InvalidInput {
            reason: "penalty overflow",
        };
    }
    if penalty < 0.0 {
        return PenaltyOutcome::Negative { penalty };
    }
    PenaltyOutcome::Ok { penalty }
}

/// Monotonicity gate: penalty grows monotonically (non-strictly) with match_len.
/// This captures "DRY reduces exact-phrase repetition" algebraically.
#[derive(Debug, PartialEq)]
pub(crate) enum MonotonicityOutcome {
    Ok,
    InvalidInput {
        reason: &'static str,
    },
    Violation {
        match_len_a: u32,
        match_len_b: u32,
        penalty_a: f64,
        penalty_b: f64,
    },
}

pub(crate) fn classify_dry_penalty_monotone_in_match_len(
    match_len_a: u32,
    match_len_b: u32,
    allowed_length: u32,
    multiplier: f64,
    base: f64,
) -> MonotonicityOutcome {
    if match_len_a > match_len_b {
        return MonotonicityOutcome::InvalidInput {
            reason: "match_len_a must be <= match_len_b",
        };
    }
    let pa = match classify_dry_penalty(match_len_a, allowed_length, multiplier, base) {
        PenaltyOutcome::Ok { penalty } => penalty,
        PenaltyOutcome::InvalidInput { reason } => {
            return MonotonicityOutcome::InvalidInput { reason }
        }
        PenaltyOutcome::Negative { .. } => {
            return MonotonicityOutcome::InvalidInput {
                reason: "penalty_a negative",
            }
        }
    };
    let pb = match classify_dry_penalty(match_len_b, allowed_length, multiplier, base) {
        PenaltyOutcome::Ok { penalty } => penalty,
        PenaltyOutcome::InvalidInput { reason } => {
            return MonotonicityOutcome::InvalidInput { reason }
        }
        PenaltyOutcome::Negative { .. } => {
            return MonotonicityOutcome::InvalidInput {
                reason: "penalty_b negative",
            }
        }
    };
    if pb + f64::EPSILON < pa {
        return MonotonicityOutcome::Violation {
            match_len_a,
            match_len_b,
            penalty_a: pa,
            penalty_b: pb,
        };
    }
    MonotonicityOutcome::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    fn breakers(ids: &[u32]) -> HashSet<u32> {
        ids.iter().copied().collect()
    }

    // -------- parameter range --------

    #[test]
    fn params_valid_defaults() {
        assert_eq!(
            classify_dry_params(
                DEFAULT_DRY_MULTIPLIER,
                DEFAULT_DRY_BASE,
                DEFAULT_DRY_ALLOWED_LENGTH
            ),
            DryParamOutcome::Valid
        );
    }

    #[test]
    fn params_valid_zero_multiplier() {
        assert_eq!(classify_dry_params(0.0, 1.75, 2), DryParamOutcome::Valid);
    }

    #[test]
    fn params_rejects_negative_multiplier() {
        assert_eq!(
            classify_dry_params(-0.1, 1.75, 2),
            DryParamOutcome::MultiplierNegative { multiplier: -0.1 }
        );
    }

    #[test]
    fn params_rejects_base_below_one() {
        assert_eq!(
            classify_dry_params(0.8, 0.5, 2),
            DryParamOutcome::BaseBelowOne { base: 0.5 }
        );
    }

    #[test]
    fn params_rejects_allowed_length_zero() {
        assert_eq!(
            classify_dry_params(0.8, 1.75, 0),
            DryParamOutcome::AllowedLengthZero
        );
    }

    #[test]
    fn params_rejects_nan_multiplier() {
        assert_eq!(
            classify_dry_params(f64::NAN, 1.75, 2),
            DryParamOutcome::NotFinite {
                field: "multiplier"
            }
        );
    }

    #[test]
    fn params_rejects_nan_base() {
        assert_eq!(
            classify_dry_params(0.8, f64::NAN, 2),
            DryParamOutcome::NotFinite { field: "base" }
        );
    }

    #[test]
    fn params_rejects_infinity() {
        assert_eq!(
            classify_dry_params(f64::INFINITY, 1.75, 2),
            DryParamOutcome::NotFinite {
                field: "multiplier"
            }
        );
    }

    // -------- identity: multiplier = 0 --------

    #[test]
    fn identity_ok_when_logits_unchanged() {
        let before = vec![0.1, 0.5, -0.3];
        let after = before.clone();
        assert_eq!(
            classify_dry_identity_zero_multiplier(&before, &after, 0.0),
            IdentityOutcome::Ok
        );
    }

    #[test]
    fn identity_flags_changed_logit() {
        let before = vec![0.1, 0.5, -0.3];
        let after = vec![0.1, 0.3, -0.3];
        match classify_dry_identity_zero_multiplier(&before, &after, 0.0) {
            IdentityOutcome::LogitsChanged {
                first_diff_index,
                before,
                after,
            } => {
                assert_eq!(first_diff_index, 1);
                assert!((before - 0.5).abs() < 1e-9);
                assert!((after - 0.3).abs() < 1e-9);
            }
            other => panic!("expected LogitsChanged, got {other:?}"),
        }
    }

    #[test]
    fn identity_rejects_non_zero_multiplier() {
        let lg = vec![0.1, 0.5];
        assert_eq!(
            classify_dry_identity_zero_multiplier(&lg, &lg, 0.8),
            IdentityOutcome::InvalidInput {
                reason: "multiplier != 0.0"
            }
        );
    }

    #[test]
    fn identity_rejects_length_mismatch() {
        let before = vec![0.1, 0.5];
        let after = vec![0.1];
        assert_eq!(
            classify_dry_identity_zero_multiplier(&before, &after, 0.0),
            IdentityOutcome::InvalidInput {
                reason: "logits length mismatch"
            }
        );
    }

    #[test]
    fn identity_rejects_empty() {
        assert_eq!(
            classify_dry_identity_zero_multiplier(&[], &[], 0.0),
            IdentityOutcome::InvalidInput {
                reason: "logits_before is empty"
            }
        );
    }

    #[test]
    fn identity_rejects_nan() {
        let before = vec![f64::NAN];
        let after = vec![f64::NAN];
        assert_eq!(
            classify_dry_identity_zero_multiplier(&before, &after, 0.0),
            IdentityOutcome::InvalidInput {
                reason: "non-finite logit"
            }
        );
    }

    // -------- match length --------

    #[test]
    fn match_len_zero_when_ctx_empty() {
        let bl = HashSet::new();
        assert_eq!(classify_dry_match_len(&[], 1, &bl), 0);
    }

    #[test]
    fn match_len_zero_when_candidate_not_in_ctx() {
        let ctx = vec![1, 2, 3];
        let bl = HashSet::new();
        assert_eq!(classify_dry_match_len(&ctx, 99, &bl), 0);
    }

    #[test]
    fn match_len_one_when_candidate_matches_single_token() {
        // ctx = [5, 7, 3], candidate = 3. ext = [5, 7, 3, 3].
        // Suffix "3" matches ctx[2] — a one-token repetition. match_len = 1.
        // (Matches llama.cpp semantics: any suffix-of-ext ending anywhere in ctx.)
        let ctx = vec![5, 7, 3];
        let bl = HashSet::new();
        assert_eq!(classify_dry_match_len(&ctx, 3, &bl), 1);
    }

    #[test]
    fn match_len_detects_repeated_bigram() {
        // ctx = [A, B, A], candidate = B → suffix "A B" matches earlier "A B" at
        // positions 0,1. So match_len should be 2.
        let ctx = vec![1, 2, 1];
        let bl = HashSet::new();
        assert_eq!(classify_dry_match_len(&ctx, 2, &bl), 2);
    }

    #[test]
    fn match_len_detects_repeated_trigram() {
        // ctx = [1, 2, 3, 1, 2], candidate = 3 → suffix "1 2 3" matches "1 2 3" at
        // positions 0..=2 → match_len 3.
        let ctx = vec![1, 2, 3, 1, 2];
        let bl = HashSet::new();
        assert_eq!(classify_dry_match_len(&ctx, 3, &bl), 3);
    }

    #[test]
    fn match_len_seq_breaker_stops_extension() {
        // ctx = [1, 2, 9, 1, 2], candidate = 3, where 9 is a seq_breaker.
        // Suffix "1 2 3" would need to match at ctx[0..=2] but ctx[2]=9 is a breaker,
        // stopping the extension. No match_len >= 3 is possible; match_len = 0.
        let ctx = vec![1, 2, 9, 1, 2];
        let bl = breakers(&[9]);
        assert_eq!(classify_dry_match_len(&ctx, 3, &bl), 0);
    }

    #[test]
    fn match_len_repeated_trigram_twice() {
        // ctx = [1 2 3 1 2 3 1 2], candidate = 3 → ext = [1 2 3 1 2 3 1 2 3].
        // The suffix "1 2 3 1 2 3" (6 tokens) is also a substring of ctx ending at
        // position 5, so the longest suffix-match is 6 — the algorithm picks up the
        // DOUBLE repetition, not just the last trigram.
        let ctx = vec![1, 2, 3, 1, 2, 3, 1, 2];
        let bl = HashSet::new();
        assert_eq!(classify_dry_match_len(&ctx, 3, &bl), 6);
    }

    // -------- penalty formula --------

    #[test]
    fn penalty_zero_below_threshold() {
        assert_eq!(
            classify_dry_penalty(1, 2, 0.8, 1.75),
            PenaltyOutcome::Ok { penalty: 0.0 }
        );
    }

    #[test]
    fn penalty_equals_multiplier_at_threshold() {
        // match_len == allowed_length → exponent = 0 → base^0 = 1 → penalty = multiplier.
        match classify_dry_penalty(2, 2, 0.8, 1.75) {
            PenaltyOutcome::Ok { penalty } => assert!((penalty - 0.8).abs() < 1e-12),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn penalty_exponential_growth() {
        // match_len=5, allowed=2 → exponent=3 → 0.8 * 1.75^3 = 0.8 * 5.359375 = 4.2875.
        match classify_dry_penalty(5, 2, 0.8, 1.75) {
            PenaltyOutcome::Ok { penalty } => assert!((penalty - 4.287_5).abs() < 1e-9),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn penalty_zero_when_multiplier_zero() {
        assert_eq!(
            classify_dry_penalty(10, 2, 0.0, 1.75),
            PenaltyOutcome::Ok { penalty: 0.0 }
        );
    }

    #[test]
    fn penalty_rejects_negative_multiplier() {
        assert_eq!(
            classify_dry_penalty(5, 2, -0.1, 1.75),
            PenaltyOutcome::InvalidInput {
                reason: "multiplier negative"
            }
        );
    }

    #[test]
    fn penalty_rejects_base_below_one() {
        assert_eq!(
            classify_dry_penalty(5, 2, 0.8, 0.5),
            PenaltyOutcome::InvalidInput {
                reason: "base < 1.0"
            }
        );
    }

    #[test]
    fn penalty_rejects_allowed_zero() {
        assert_eq!(
            classify_dry_penalty(5, 0, 0.8, 1.75),
            PenaltyOutcome::InvalidInput {
                reason: "allowed_length == 0"
            }
        );
    }

    #[test]
    fn penalty_rejects_nan() {
        assert_eq!(
            classify_dry_penalty(5, 2, f64::NAN, 1.75),
            PenaltyOutcome::InvalidInput {
                reason: "non-finite multiplier or base"
            }
        );
    }

    // -------- monotonicity (non-decreasing in match_len) --------

    #[test]
    fn monotone_ok_below_threshold_both_zero() {
        assert_eq!(
            classify_dry_penalty_monotone_in_match_len(0, 1, 2, 0.8, 1.75),
            MonotonicityOutcome::Ok
        );
    }

    #[test]
    fn monotone_ok_below_to_at_threshold() {
        // 1→2 crosses threshold: 0 → 0.8.
        assert_eq!(
            classify_dry_penalty_monotone_in_match_len(1, 2, 2, 0.8, 1.75),
            MonotonicityOutcome::Ok
        );
    }

    #[test]
    fn monotone_ok_strict_growth_above_threshold() {
        // 3→5 above threshold; exponent grows 1→3.
        assert_eq!(
            classify_dry_penalty_monotone_in_match_len(3, 5, 2, 0.8, 1.75),
            MonotonicityOutcome::Ok
        );
    }

    #[test]
    fn monotone_rejects_decreasing_args() {
        assert_eq!(
            classify_dry_penalty_monotone_in_match_len(5, 3, 2, 0.8, 1.75),
            MonotonicityOutcome::InvalidInput {
                reason: "match_len_a must be <= match_len_b"
            }
        );
    }

    #[test]
    fn monotone_ok_equal_match_len() {
        assert_eq!(
            classify_dry_penalty_monotone_in_match_len(4, 4, 2, 0.8, 1.75),
            MonotonicityOutcome::Ok
        );
    }

    // -------- integration-level consistency --------

    #[test]
    fn identity_and_penalty_zero_multiplier_coincide() {
        // multiplier=0 → penalty=0 for any match_len.
        for m in 0..10 {
            match classify_dry_penalty(m, 2, 0.0, 1.75) {
                PenaltyOutcome::Ok { penalty } => assert_eq!(penalty, 0.0),
                other => panic!("expected Ok, got {other:?} at match_len={m}"),
            }
        }
    }

    #[test]
    fn match_len_above_allowed_triggers_positive_penalty() {
        let ctx = vec![1, 2, 3, 1, 2];
        let ml = classify_dry_match_len(&ctx, 3, &HashSet::new());
        // Verify bridge from match_len to penalty.
        assert_eq!(ml, 3);
        match classify_dry_penalty(ml, 2, 0.8, 1.75) {
            PenaltyOutcome::Ok { penalty } => assert!(penalty > 0.0),
            other => panic!("expected Ok, got {other:?}"),
        }
    }
}
