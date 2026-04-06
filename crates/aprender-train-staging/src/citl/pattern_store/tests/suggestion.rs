//! FixSuggestion tests.

use super::*;

#[test]
fn test_fix_suggestion_new() {
    let pattern = FixPattern::new("E0308", "diff");
    let suggestion = FixSuggestion::new(pattern, 0.85, 0);

    assert_eq!(suggestion.score, 0.85);
    assert_eq!(suggestion.rank, 0);
}

#[test]
fn test_fix_suggestion_weighted_score() {
    let mut pattern = FixPattern::new("E0308", "diff");
    pattern.record_success();
    pattern.record_success();

    let suggestion = FixSuggestion::new(pattern, 0.8, 0);
    // weighted = 0.8 * (0.5 + 0.5 * 1.0) = 0.8 * 1.0 = 0.8
    assert!((suggestion.weighted_score() - 0.8).abs() < 0.01);
}

#[test]
fn test_fix_suggestion_weighted_score_partial_success() {
    let mut pattern = FixPattern::new("E0308", "diff");
    pattern.record_success();
    pattern.record_failure();

    let suggestion = FixSuggestion::new(pattern, 1.0, 0);
    // success_rate = 0.5, weighted = 1.0 * (0.5 + 0.5 * 0.5) = 1.0 * 0.75 = 0.75
    assert!((suggestion.weighted_score() - 0.75).abs() < 0.01);
}
