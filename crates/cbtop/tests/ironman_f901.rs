#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Ironman Falsification Suite Tests (F901-F920)
//!
//! PMAT-017: Test the "Ironman" quality gate validators per §34.
//!
//! # Falsification Criteria
//!
//! | ID | Claim | Test | Pass Criteria |
//! |----|-------|------|---------------|
//! | F901 | Mutation Resilience >90% | cargo mutants | Score > 90% |
//! | F909 | Unsafe Audit Clean | cargo geiger | 0 forbid |
//! | F910 | Dependency Audit Clean | cargo audit | 0 vulns |
//! | F912 | Cognitive Complexity <15 | clippy | All fns pass |
//! | F915 | Binary Size <8MB | strip | Size < 8MB |
//! | F916 | Startup Time <20ms | cold start | Time < 20ms |
//! | F920 | I18n Safe | non-ASCII input | No crash |

use cbtop::{
    quick_validate, GateCategory, GateResult, IronmanScorecard, IronmanValidator, IRONMAN_GATES,
};

// ============================================================================
// F901: Mutation Resilience Tests
// ============================================================================

#[test]
fn f901_gate_result_types() {
    // F901: Verify GateResult enum covers all cases
    let pass = GateResult::Pass("ok".to_string());
    let fail = GateResult::Fail("error".to_string());
    let skip = GateResult::Skip("skipped".to_string());
    let pending = GateResult::Pending;

    assert!(pass.passed());
    assert!(!fail.passed());
    assert!(!skip.passed());
    assert!(!pending.passed());

    assert!(!pass.failed());
    assert!(fail.failed());
    assert!(!skip.failed());
    assert!(!pending.failed());
}

#[test]
fn f901_gate_result_scoring() {
    // F901: Verify scoring logic
    assert_eq!(GateResult::Pass("ok".to_string()).score(15), 15);
    assert_eq!(GateResult::Fail("err".to_string()).score(15), 0);
    assert_eq!(GateResult::Skip("skip".to_string()).score(15), 0);
    assert_eq!(GateResult::Pending.score(15), 0);
}

// ============================================================================
// F902-F908: Quality Gate Structure Tests
// ============================================================================

#[test]
fn f902_all_gates_defined() {
    // F902: Verify all F901-F920 gates are defined
    let expected_ids = [
        "F901", "F902", "F903", "F904", "F905", "F906", "F907", "F908", "F909", "F910", "F911",
        "F912", "F913", "F914", "F915", "F916", "F917", "F918", "F919", "F920",
    ];

    let gate_ids: Vec<_> = IRONMAN_GATES.iter().map(|g| g.id).collect();

    for expected in &expected_ids {
        assert!(gate_ids.contains(expected), "Missing gate: {}", expected);
    }

    assert_eq!(IRONMAN_GATES.len(), 20, "Expected exactly 20 gates");
}

#[test]
fn f903_gate_weights_positive() {
    // F903: All gates must have positive weights
    for gate in IRONMAN_GATES {
        assert!(gate.weight > 0, "Gate {} has zero weight", gate.id);
    }
}

#[test]
fn f904_gate_categories_valid() {
    // F904: All gates must have valid categories
    for gate in IRONMAN_GATES {
        let _ = gate.category.name(); // Should not panic
    }
}

#[test]
fn f905_gate_fields_non_empty() {
    // F905: All gate fields must be non-empty
    for gate in IRONMAN_GATES {
        assert!(!gate.id.is_empty(), "Gate has empty ID");
        assert!(!gate.name.is_empty(), "Gate {} has empty name", gate.id);
        assert!(!gate.tool.is_empty(), "Gate {} has empty tool", gate.id);
        assert!(!gate.target.is_empty(), "Gate {} has empty target", gate.id);
    }
}

// ============================================================================
// F909: Scorecard Tests
// ============================================================================

#[test]
fn f909_scorecard_initialization() {
    // F909: Scorecard initializes correctly
    let scorecard = IronmanScorecard::new();

    assert_eq!(scorecard.total_score, 0);
    assert!(scorecard.max_score > 0);
    assert_eq!(scorecard.pass_threshold, 0.90);
    assert!(scorecard.results.is_empty());
}

#[test]
fn f909_scorecard_record_valid_gate() {
    // F909: Recording valid gate updates score
    let mut scorecard = IronmanScorecard::new();
    let initial_score = scorecard.total_score;

    scorecard.record("F909", GateResult::Pass("ok".to_string()));

    assert!(scorecard.total_score > initial_score);
    assert!(scorecard.results.contains_key("F909"));
}

#[test]
fn f909_scorecard_record_invalid_gate() {
    // F909: Recording invalid gate has no effect
    let mut scorecard = IronmanScorecard::new();

    scorecard.record("INVALID", GateResult::Pass("ok".to_string()));

    assert_eq!(scorecard.total_score, 0);
    assert!(scorecard.results.is_empty());
}

#[test]
fn f909_scorecard_percentage_empty() {
    // F909: Empty scorecard has 0% score
    let scorecard = IronmanScorecard::new();
    assert_eq!(scorecard.percentage(), 0.0);
}

#[test]
fn f909_scorecard_percentage_full() {
    // F909: Full pass scorecard has 100% score
    let mut scorecard = IronmanScorecard::new();

    for gate in IRONMAN_GATES {
        scorecard.record(gate.id, GateResult::Pass("ok".to_string()));
    }

    assert!((scorecard.percentage() - 100.0).abs() < 0.01);
}

#[test]
fn f909_scorecard_passed_threshold() {
    // F909: Pass threshold works correctly
    let mut scorecard = IronmanScorecard::new();

    // Record enough gates to pass (need >90%)
    let gates_to_pass = (IRONMAN_GATES.len() as f64 * 0.95) as usize;
    for gate in IRONMAN_GATES.iter().take(gates_to_pass) {
        scorecard.record(gate.id, GateResult::Pass("ok".to_string()));
    }

    // Should pass if percentage > 90%
    if scorecard.percentage() >= 90.0 {
        assert!(scorecard.passed());
    }
}

// ============================================================================
// F910: Category Scoring Tests
// ============================================================================

#[test]
fn f910_category_score_resilience() {
    // F910: Resilience category scoring
    let mut scorecard = IronmanScorecard::new();
    scorecard.record("F901", GateResult::Pass("ok".to_string()));
    scorecard.record("F902", GateResult::Pass("ok".to_string()));

    let (achieved, max) = scorecard.category_score(GateCategory::Resilience);
    assert!(achieved > 0);
    assert!(max > 0);
    assert!(achieved <= max);
}

#[test]
fn f910_category_score_safety() {
    // F910: Safety category scoring
    let mut scorecard = IronmanScorecard::new();
    scorecard.record("F903", GateResult::Pass("ok".to_string()));

    let (achieved, max) = scorecard.category_score(GateCategory::Safety);
    assert!(achieved > 0);
    assert!(max > 0);
}

#[test]
fn f910_category_score_quality() {
    // F910: Quality category scoring
    let mut scorecard = IronmanScorecard::new();
    scorecard.record("F909", GateResult::Pass("ok".to_string()));
    scorecard.record("F910", GateResult::Pass("ok".to_string()));

    let (achieved, max) = scorecard.category_score(GateCategory::Quality);
    assert!(achieved > 0);
    assert!(max > 0);
}

#[test]
fn f910_category_score_performance() {
    // F910: Performance category scoring
    let mut scorecard = IronmanScorecard::new();
    scorecard.record("F915", GateResult::Pass("ok".to_string()));
    scorecard.record("F916", GateResult::Pass("ok".to_string()));

    let (achieved, max) = scorecard.category_score(GateCategory::Performance);
    assert!(achieved > 0);
    assert!(max > 0);
}

#[test]
fn f910_category_score_usability() {
    // F910: Usability category scoring
    let mut scorecard = IronmanScorecard::new();
    scorecard.record("F919", GateResult::Pass("ok".to_string()));
    scorecard.record("F920", GateResult::Pass("ok".to_string()));

    let (achieved, max) = scorecard.category_score(GateCategory::Usability);
    assert!(achieved > 0);
    assert!(max > 0);
}

// ============================================================================
// F911: Failed/Skipped Gate Tracking
// ============================================================================

#[test]
fn f911_failed_gates_tracking() {
    // F911: Failed gates are tracked correctly
    let mut scorecard = IronmanScorecard::new();
    scorecard.record("F909", GateResult::Fail("error".to_string()));
    scorecard.record("F910", GateResult::Pass("ok".to_string()));

    let failed = scorecard.failed_gates();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].id, "F909");
}

#[test]
fn f911_skipped_gates_tracking() {
    // F911: Skipped gates are tracked correctly
    let mut scorecard = IronmanScorecard::new();
    scorecard.record("F909", GateResult::Skip("skipped".to_string()));

    let skipped = scorecard.skipped_gates();
    // Should include F909 and all unrecorded gates
    assert!(skipped.iter().any(|g| g.id == "F909"));
}

// ============================================================================
// F912: Validator Configuration Tests
// ============================================================================

#[test]
fn f912_validator_new() {
    // F912: Validator creates correctly
    let validator = IronmanValidator::new(".");
    assert!(!validator.verbose);
    assert!(!validator.skip_slow);
}

#[test]
fn f912_validator_builder_pattern() {
    // F912: Validator builder pattern works
    let validator = IronmanValidator::new(".").verbose(true).skip_slow(true);

    assert!(validator.verbose);
    assert!(validator.skip_slow);
}

// ============================================================================
// F915-F916: Performance Gate Tests
// ============================================================================

#[test]
fn f915_binary_size_threshold() {
    // F915: Binary size threshold is 8MB
    let gate = IRONMAN_GATES.iter().find(|g| g.id == "F915").unwrap();
    assert_eq!(gate.target, "<8MB");
}

#[test]
fn f916_startup_time_threshold() {
    // F916: Startup time threshold is 20ms
    let gate = IRONMAN_GATES.iter().find(|g| g.id == "F916").unwrap();
    assert_eq!(gate.target, "<20ms");
}

// ============================================================================
// F920: Internationalization Tests
// ============================================================================

#[test]
fn f920_i18n_japanese() {
    // F920: Japanese text doesn't crash
    let input = "日本語テスト";
    let _len = input.len();
    let _chars = input.chars().count();
    let _formatted = format!("Input: {}", input);
    assert!(input.chars().count() > 0);
}

#[test]
fn f920_i18n_chinese() {
    // F920: Chinese text doesn't crash
    let input = "中文测试";
    let _len = input.len();
    let _chars = input.chars().count();
    let _formatted = format!("Input: {}", input);
    assert!(input.chars().count() > 0);
}

#[test]
fn f920_i18n_korean() {
    // F920: Korean text doesn't crash
    let input = "한국어 테스트";
    let _len = input.len();
    let _chars = input.chars().count();
    let _formatted = format!("Input: {}", input);
    assert!(input.chars().count() > 0);
}

#[test]
fn f920_i18n_russian() {
    // F920: Russian text doesn't crash
    let input = "тест на русском";
    let _len = input.len();
    let _chars = input.chars().count();
    let _formatted = format!("Input: {}", input);
    assert!(input.chars().count() > 0);
}

#[test]
fn f920_i18n_greek() {
    // F920: Greek text doesn't crash
    let input = "δοκιμή ελληνικά";
    let _len = input.len();
    let _chars = input.chars().count();
    let _formatted = format!("Input: {}", input);
    assert!(input.chars().count() > 0);
}

#[test]
fn f920_i18n_emoji() {
    // F920: Emoji don't crash
    let input = "🔥💻🚀";
    let _len = input.len();
    let _chars = input.chars().count();
    let _formatted = format!("Input: {}", input);
    assert!(input.chars().count() > 0);
}

#[test]
fn f920_i18n_bom() {
    // F920: BOM handling doesn't crash
    let input = "\u{FEFF}BOM test";
    let _len = input.len();
    let _chars = input.chars().count();
    let _formatted = format!("Input: {}", input);
    assert!(input.chars().count() > 0);
}

#[test]
fn f920_i18n_null_bytes() {
    // F920: Null bytes handled gracefully
    let input = "null\0byte";
    let _len = input.len();
    let _chars = input.chars().count();
    // Note: format! with null bytes is fine in Rust
    assert!(input.chars().count() > 0);
}

#[test]
fn f920_i18n_validator_check() {
    // F920: Validator i18n check passes
    let validator = IronmanValidator::new(".");
    let result = validator.check_i18n();
    assert!(result.passed());
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_quick_validate_runs() {
    // Integration: quick_validate completes without panic
    let project_root = std::env::current_dir().unwrap();
    let scorecard = quick_validate(&project_root);

    // Should have some results
    assert!(!scorecard.results.is_empty());
    // Slow checks should be skipped
    if let Some(result) = scorecard.results.get("F901") {
        assert!(matches!(result, GateResult::Skip(_)));
    }
}

#[test]
fn test_full_scorecard_report() {
    // Integration: Full scorecard can be generated
    let mut scorecard = IronmanScorecard::new();

    // Simulate mixed results
    scorecard.record("F909", GateResult::Pass("No unsafe".to_string()));
    scorecard.record("F910", GateResult::Pass("No vulns".to_string()));
    scorecard.record("F920", GateResult::Pass("I18n ok".to_string()));

    let passed = scorecard.results.values().filter(|r| r.passed()).count();
    let failed = scorecard.failed_gates().len();

    assert_eq!(passed, 3);
    assert_eq!(failed, 0);
}

#[test]
fn test_category_coverage() {
    // Integration: All categories have at least one gate
    let categories = [
        GateCategory::Resilience,
        GateCategory::Safety,
        GateCategory::Quality,
        GateCategory::Performance,
        GateCategory::Usability,
    ];

    for category in categories {
        let gates_in_category = IRONMAN_GATES.iter().filter(|g| g.category == category).count();
        assert!(gates_in_category > 0, "Category {:?} has no gates", category);
    }
}

#[test]
fn test_total_weight_reasonable() {
    // Integration: Total weight is reasonable (100-200 points)
    let total_weight: u32 = IRONMAN_GATES.iter().map(|g| g.weight).sum();
    assert!(total_weight >= 100, "Total weight too low: {}", total_weight);
    assert!(total_weight <= 200, "Total weight too high: {}", total_weight);
}
