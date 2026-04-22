//! Unit tests for `dry_sampling_classifier` (extracted from `dry_sampling_classifier.rs` to keep file-size invariant).
//!
//! Included via `#[cfg(test)] #[path = "dry_sampling_classifier_tests.rs"] mod tests;` in the parent.

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
