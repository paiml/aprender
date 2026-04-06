//! FixPattern tests.

use super::*;

#[test]
fn test_fix_pattern_new() {
    let pattern = FixPattern::new("E0308", "- old\n+ new");
    assert_eq!(pattern.error_code, "E0308");
    assert_eq!(pattern.fix_diff, "- old\n+ new");
    assert!(pattern.decision_sequence.is_empty());
    assert_eq!(pattern.success_count, 0);
    assert_eq!(pattern.attempt_count, 0);
}

#[test]
fn test_fix_pattern_with_decision() {
    let pattern = FixPattern::new("E0308", "diff")
        .with_decision("detect_mismatch")
        .with_decision("suggest_fix");

    assert_eq!(pattern.decision_sequence.len(), 2);
    assert_eq!(pattern.decision_sequence[0], "detect_mismatch");
    assert_eq!(pattern.decision_sequence[1], "suggest_fix");
}

#[test]
fn test_fix_pattern_with_decisions() {
    let decisions = vec!["step1".to_string(), "step2".to_string()];
    let pattern = FixPattern::new("E0308", "diff").with_decisions(decisions);

    assert_eq!(pattern.decision_sequence.len(), 2);
}

#[test]
fn test_fix_pattern_record_success() {
    let mut pattern = FixPattern::new("E0308", "diff");
    pattern.record_success();
    pattern.record_success();

    assert_eq!(pattern.success_count, 2);
    assert_eq!(pattern.attempt_count, 2);
}

#[test]
fn test_fix_pattern_record_failure() {
    let mut pattern = FixPattern::new("E0308", "diff");
    pattern.record_success();
    pattern.record_failure();

    assert_eq!(pattern.success_count, 1);
    assert_eq!(pattern.attempt_count, 2);
}

#[test]
fn test_fix_pattern_success_rate() {
    let mut pattern = FixPattern::new("E0308", "diff");
    assert_eq!(pattern.success_rate(), 0.0);

    pattern.record_success();
    pattern.record_success();
    pattern.record_failure();

    assert!((pattern.success_rate() - 0.666).abs() < 0.01);
}

#[test]
fn test_fix_pattern_to_searchable_text() {
    let pattern = FixPattern::new("E0308", "- i32\n+ &str").with_decision("type_mismatch");

    let text = pattern.to_searchable_text();
    assert!(text.contains("E0308"));
    assert!(text.contains("type_mismatch"));
    assert!(text.contains("- i32"));
}

#[test]
fn test_fix_pattern_serialization() {
    let pattern = FixPattern::new("E0308", "diff").with_decision("step1");

    let json = serde_json::to_string(&pattern).expect("JSON serialization should succeed");
    let deserialized: FixPattern =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert_eq!(pattern.error_code, deserialized.error_code);
    assert_eq!(pattern.fix_diff, deserialized.fix_diff);
    assert_eq!(pattern.decision_sequence, deserialized.decision_sequence);
}
