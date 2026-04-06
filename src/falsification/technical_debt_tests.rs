use super::*;

// =========================================================================
// MTD-01: Entanglement Detection Tests
// =========================================================================

#[test]
fn test_mtd_01_entanglement_detection() {
    let result = check_entanglement_detection(Path::new("."));
    assert!(
        matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
        "Entanglement check failed: {:?}",
        result.rejection_reason
    );
}

// =========================================================================
// MTD-02: Correction Cascade Tests
// =========================================================================

#[test]
fn test_mtd_02_correction_cascade() {
    let result = check_correction_cascade_prevention(Path::new("."));
    assert!(
        !matches!(result.status, CheckStatus::Skipped),
        "Correction cascade check should complete"
    );
}

// =========================================================================
// MTD-03: Undeclared Consumer Tests
// =========================================================================

#[test]
fn test_mtd_03_undeclared_consumers() {
    let result = check_undeclared_consumer_detection(Path::new("."));
    assert!(
        matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
        "Consumer detection failed: {:?}",
        result.rejection_reason
    );
}

// =========================================================================
// MTD-04: Data Freshness Tests
// =========================================================================

#[test]
fn test_mtd_04_data_freshness() {
    let result = check_data_dependency_freshness(Path::new("."));
    assert!(!matches!(result.status, CheckStatus::Skipped), "Data freshness check should complete");
}

// =========================================================================
// MTD-05: Pipeline Glue Code Tests
// =========================================================================

#[test]
fn test_mtd_05_pipeline_glue() {
    let result = check_pipeline_glue_code(Path::new("."));
    // batuta has pipeline module
    assert!(
        matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
        "Pipeline check failed: {:?}",
        result.rejection_reason
    );
}

// =========================================================================
// MTD-06: Configuration Debt Tests
// =========================================================================

#[test]
fn test_mtd_06_configuration_debt() {
    let result = check_configuration_debt(Path::new("."));
    // batuta has config module
    assert!(
        matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
        "Config debt check failed: {:?}",
        result.rejection_reason
    );
}

// =========================================================================
// MTD-07: Dead Code Elimination Tests
// =========================================================================

#[test]
fn test_mtd_07_dead_code() {
    let result = check_dead_code_elimination(Path::new("."));
    assert!(!matches!(result.status, CheckStatus::Skipped), "Dead code check should complete");
}

// =========================================================================
// MTD-08: Abstraction Boundaries Tests
// =========================================================================

#[test]
fn test_mtd_08_abstraction_boundaries() {
    let result = check_abstraction_boundaries(Path::new("."));
    // batuta has multiple modules
    assert!(
        matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
        "Abstraction check failed: {:?}",
        result.rejection_reason
    );
}

// =========================================================================
// MTD-09: Feedback Loop Tests
// =========================================================================

#[test]
fn test_mtd_09_feedback_loops() {
    let result = check_feedback_loop_detection(Path::new("."));
    assert!(!matches!(result.status, CheckStatus::Skipped), "Feedback loop check should complete");
}

// =========================================================================
// MTD-10: Technical Debt Quantification Tests
// =========================================================================

#[test]
fn test_mtd_10_technical_debt() {
    let result = check_technical_debt_quantification(Path::new("."));
    assert!(!matches!(result.status, CheckStatus::Skipped), "TDG check should complete");
}

// =========================================================================
// Integration Tests
// =========================================================================

#[test]
fn test_evaluate_all_returns_10_items() {
    let results = evaluate_all(Path::new("."));
    assert_eq!(results.len(), 10, "Expected 10 technical debt checks");
}

#[test]
fn test_all_items_have_evidence() {
    let results = evaluate_all(Path::new("."));
    for item in &results {
        assert!(!item.evidence.is_empty(), "Item {} missing evidence", item.id);
    }
}

#[test]
fn test_all_items_have_tps_principle() {
    let results = evaluate_all(Path::new("."));
    for item in &results {
        assert!(!item.tps_principle.is_empty(), "Item {} missing TPS principle", item.id);
    }
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_mtd_id_and_severity_table() {
    let checks: Vec<(&str, fn(&Path) -> CheckItem)> = vec![
        ("MTD-01", check_entanglement_detection),
        ("MTD-02", check_correction_cascade_prevention),
        ("MTD-03", check_undeclared_consumer_detection),
        ("MTD-04", check_data_dependency_freshness),
        ("MTD-05", check_pipeline_glue_code),
        ("MTD-06", check_configuration_debt),
        ("MTD-07", check_dead_code_elimination),
        ("MTD-08", check_abstraction_boundaries),
        ("MTD-09", check_feedback_loop_detection),
        ("MTD-10", check_technical_debt_quantification),
    ];

    let path = Path::new(".");
    for (expected_id, check_fn) in &checks {
        let result = check_fn(path);
        assert_eq!(
            result.id, *expected_id,
            "Check function for {} returned wrong id: {}",
            expected_id, result.id
        );
        assert!(
            matches!(result.severity, Severity::Major | Severity::Critical),
            "Check {} has unexpected severity: {:?}",
            expected_id,
            result.severity
        );
    }
}

#[test]
fn test_nonexistent_path_handling() {
    let path = Path::new("/nonexistent/path/for/technical/debt");
    let results = evaluate_all(path);
    assert_eq!(results.len(), 10);
}

#[test]
fn test_all_items_have_reasonable_duration() {
    let results = evaluate_all(Path::new("."));
    for item in &results {
        // Duration should be reasonable (less than 1 minute per check)
        assert!(
            item.duration_ms < 60_000,
            "Item {} took unreasonably long: {}ms",
            item.id,
            item.duration_ms
        );
    }
}

// =========================================================================
// Helper Function Tests
// =========================================================================

#[test]
fn test_path_exists_any_found() {
    // At least one of these should exist in batuta
    assert!(path_exists_any(Path::new("."), &["Cargo.toml", "nonexistent.txt"]));
}

#[test]
fn test_path_exists_any_none() {
    assert!(!path_exists_any(Path::new("."), &["nonexistent1.txt", "nonexistent2.txt"]));
}

#[test]
fn test_file_contains_any_found() {
    // Cargo.toml should contain "[package]"
    assert!(file_contains_any(Path::new("./Cargo.toml"), &["[package]", "nonexistent_pattern"]));
}

#[test]
fn test_file_contains_any_none() {
    assert!(!file_contains_any(
        Path::new("./Cargo.toml"),
        &["nonexistent_pattern_1", "nonexistent_pattern_2"]
    ));
}

#[test]
fn test_file_contains_any_missing_file() {
    assert!(!file_contains_any(Path::new("./nonexistent_file.rs"), &["pattern"]));
}

#[test]
fn test_file_contains_all_success() {
    // Cargo.toml should contain both [package] and [dependencies]
    assert!(file_contains_all(Path::new("./Cargo.toml"), &[&["[package]"], &["[dependencies]"]]));
}

#[test]
fn test_file_contains_all_partial() {
    // Should fail when one group doesn't match
    assert!(!file_contains_all(
        Path::new("./Cargo.toml"),
        &[&["[package]"], &["nonexistent_unique_pattern_xyz"]]
    ));
}

#[test]
fn test_file_contains_all_missing_file() {
    assert!(!file_contains_all(Path::new("./nonexistent_file.rs"), &[&["pattern"]]));
}

#[test]
fn test_classify_isolation_patterns_feature_flags() {
    let content = r#"#[cfg(feature = "test")]"#;
    let patterns = classify_isolation_patterns(content);
    assert!(patterns.contains(&"feature_flags"));
}

#[test]
fn test_classify_isolation_patterns_generics() {
    let content = "impl<T> MyStruct where T: Clone {}";
    let patterns = classify_isolation_patterns(content);
    assert!(patterns.contains(&"generic_abstractions"));
}

#[test]
fn test_classify_isolation_patterns_traits() {
    let content = "trait MyTrait {} impl MyTrait for MyStruct {}";
    let patterns = classify_isolation_patterns(content);
    assert!(patterns.contains(&"trait_abstractions"));
}

#[test]
fn test_classify_isolation_patterns_visibility() {
    let content = "pub(crate) fn my_fn() {}";
    let patterns = classify_isolation_patterns(content);
    assert!(patterns.contains(&"visibility_control"));
}

#[test]
fn test_classify_isolation_patterns_empty() {
    let content = "fn simple() {}";
    let patterns = classify_isolation_patterns(content);
    assert!(patterns.is_empty());
}

#[test]
fn test_scan_isolation_indicators() {
    let indicators = scan_isolation_indicators(Path::new("."));
    // batuta should have at least some isolation patterns
    assert!(!indicators.is_empty(), "Should find isolation patterns in batuta");
}

#[test]
fn test_scan_cascade_indicators() {
    let indicators = scan_cascade_indicators(Path::new("."));
    // Just verify it runs without panic
    let _ = indicators;
}

#[test]
fn test_scan_standardization_indicators() {
    let indicators = scan_standardization_indicators(Path::new("."));
    // batuta should have some standardization patterns
    assert!(!indicators.is_empty(), "Should find standardization patterns in batuta");
}

#[test]
fn test_check_ci_for_content_found() {
    // Check if CI has clippy (it should in batuta)
    let has_clippy = check_ci_for_content(Path::new("."), "clippy");
    // This may or may not be true depending on CI config
    let _ = has_clippy;
}

#[test]
fn test_check_ci_for_content_not_found() {
    let has_unusual = check_ci_for_content(Path::new("."), "unusual_string_xyz_123");
    assert!(!has_unusual);
}

#[test]
fn test_check_ci_for_content_nonexistent_path() {
    let result = check_ci_for_content(Path::new("/nonexistent/path"), "test");
    assert!(!result);
}
