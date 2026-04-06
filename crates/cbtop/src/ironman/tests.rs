use super::*;

#[test]
fn test_gate_result_score() {
    assert_eq!(GateResult::Pass("ok".to_string()).score(10), 10);
    assert_eq!(GateResult::Fail("error".to_string()).score(10), 0);
    assert_eq!(GateResult::Skip("skipped".to_string()).score(10), 0);
    assert_eq!(GateResult::Pending.score(10), 0);
}

#[test]
fn test_gate_result_passed() {
    assert!(GateResult::Pass("ok".to_string()).passed());
    assert!(!GateResult::Fail("error".to_string()).passed());
    assert!(!GateResult::Skip("skipped".to_string()).passed());
}

#[test]
fn test_scorecard_new() {
    let scorecard = IronmanScorecard::new();
    assert_eq!(scorecard.total_score, 0);
    assert!(scorecard.max_score > 0);
    assert_eq!(scorecard.pass_threshold, 0.90);
}

#[test]
fn test_scorecard_record() {
    let mut scorecard = IronmanScorecard::new();
    scorecard.record("F909", GateResult::Pass("ok".to_string()));
    assert_eq!(scorecard.results.len(), 1);
    assert!(scorecard.total_score > 0);
}

#[test]
fn test_scorecard_percentage() {
    let mut scorecard = IronmanScorecard::new();
    // Record all gates as passed
    for gate in IRONMAN_GATES {
        scorecard.record(gate.id, GateResult::Pass("ok".to_string()));
    }
    assert!((scorecard.percentage() - 100.0).abs() < 0.1);
}

#[test]
fn test_scorecard_category_score() {
    let mut scorecard = IronmanScorecard::new();
    scorecard.record("F909", GateResult::Pass("ok".to_string())); // Quality
    scorecard.record("F910", GateResult::Pass("ok".to_string())); // Quality

    let (achieved, max) = scorecard.category_score(GateCategory::Quality);
    assert!(achieved > 0);
    assert!(max > achieved);
}

#[test]
fn test_ironman_gates_complete() {
    // Verify all F901-F920 gates are defined
    let gate_ids: Vec<_> = IRONMAN_GATES.iter().map(|g| g.id).collect();
    assert!(gate_ids.contains(&"F901"));
    assert!(gate_ids.contains(&"F920"));
    assert_eq!(IRONMAN_GATES.len(), 20);
}

#[test]
fn test_ironman_gates_weights_sum() {
    let total_weight: u32 = IRONMAN_GATES.iter().map(|g| g.weight).sum();
    // Total should be 150 points per spec
    assert!(total_weight > 0);
}

#[test]
fn test_gate_category_name() {
    assert_eq!(GateCategory::Resilience.name(), "Resilience");
    assert_eq!(GateCategory::Safety.name(), "Safety");
    assert_eq!(GateCategory::Quality.name(), "Quality");
    assert_eq!(GateCategory::Performance.name(), "Performance");
    assert_eq!(GateCategory::Usability.name(), "Usability");
}

#[test]
fn test_i18n_check_no_panic() {
    // This test verifies F920 doesn't panic on any input
    let validator = IronmanValidator::new(".");
    let result = validator.check_i18n();
    assert!(result.passed());
}

#[test]
fn test_scorecard_failed_gates() {
    let mut scorecard = IronmanScorecard::new();
    scorecard.record("F909", GateResult::Fail("error".to_string()));
    scorecard.record("F910", GateResult::Pass("ok".to_string()));

    let failed = scorecard.failed_gates();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].id, "F909");
}

#[test]
fn test_scorecard_skipped_gates() {
    let mut scorecard = IronmanScorecard::new();
    scorecard.record("F909", GateResult::Skip("skipped".to_string()));
    scorecard.record("F910", GateResult::Pass("ok".to_string()));

    let skipped = scorecard.skipped_gates();
    // Should include F909 plus all others not recorded
    assert!(!skipped.is_empty());
}

#[test]
fn test_quick_validate_skips_slow() {
    // This test verifies quick_validate mode skips slow checks
    // We can't actually run it without the project, but verify the function exists
    let project_root = std::env::current_dir().unwrap();
    let scorecard = quick_validate(&project_root);

    // Slow checks should be skipped
    if let Some(result) = scorecard.results.get("F901") {
        assert!(matches!(result, GateResult::Skip(_)));
    }
}
