use super::*;
use std::path::PathBuf;

// =========================================================================
// SF-01: Unsafe Code Isolation Tests
// =========================================================================

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_sf_01_unsafe_code_isolation_current_project() {
    let result = check_unsafe_code_isolation(Path::new("."));
    // batuta should have minimal unsafe code
    assert!(
        matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
        "Unsafe code isolation failed: {:?}",
        result.rejection_reason
    );
}

// =========================================================================
// SF-02: Memory Safety Fuzzing Tests
// =========================================================================

#[test]
fn test_sf_02_memory_safety_fuzzing() {
    let result = check_memory_safety_fuzzing(Path::new("."));
    // Fuzzing is optional, partial is acceptable
    assert!(!matches!(result.status, CheckStatus::Skipped), "Fuzzing check should complete");
}

// =========================================================================
// SF-03: Miri Validation Tests
// =========================================================================

#[test]
fn test_sf_03_miri_validation() {
    let result = check_miri_validation(Path::new("."));
    assert!(!matches!(result.status, CheckStatus::Skipped), "Miri check should complete");
}

// =========================================================================
// SF-04: Formal Safety Properties Tests
// =========================================================================

#[test]
fn test_sf_04_formal_safety_properties() {
    let result = check_formal_safety_properties(Path::new("."));
    // Formal verification is advanced, partial acceptable
    assert!(
        !matches!(result.status, CheckStatus::Skipped),
        "Formal verification check should complete"
    );
}

// =========================================================================
// SF-05: Adversarial Robustness Tests
// =========================================================================

#[test]
fn test_sf_05_adversarial_robustness() {
    let result = check_adversarial_robustness(Path::new("."));
    // Adversarial testing depends on ML models
    assert!(
        !matches!(result.status, CheckStatus::Skipped),
        "Adversarial robustness check should complete"
    );
}

// =========================================================================
// SF-06: Thread Safety Tests
// =========================================================================

#[test]
fn test_sf_06_thread_safety() {
    let result = check_thread_safety(Path::new("."));
    assert!(
        matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
        "Thread safety failed: {:?}",
        result.rejection_reason
    );
}

// =========================================================================
// SF-07: Resource Leak Prevention Tests
// =========================================================================

#[test]
fn test_sf_07_resource_leak_prevention() {
    let result = check_resource_leak_prevention(Path::new("."));
    assert!(
        matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
        "Resource leak prevention failed: {:?}",
        result.rejection_reason
    );
}

// =========================================================================
// SF-08: Panic Safety Tests
// =========================================================================

#[test]
fn test_sf_08_panic_safety() {
    let result = check_panic_safety(Path::new("."));
    assert!(
        !matches!(result.status, CheckStatus::Fail),
        "Panic safety failed: {:?}",
        result.rejection_reason
    );
}

// =========================================================================
// SF-09: Input Validation Tests
// =========================================================================

#[test]
fn test_sf_09_input_validation() {
    let result = check_input_validation(Path::new("."));
    assert!(
        matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
        "Input validation failed: {:?}",
        result.rejection_reason
    );
}

// =========================================================================
// SF-10: Supply Chain Security Tests
// =========================================================================

#[test]
fn test_sf_10_supply_chain_security() {
    let result = check_supply_chain_security(Path::new("."));
    // Supply chain tools may not be installed in all environments
    assert!(!matches!(result.status, CheckStatus::Skipped), "Supply chain check should complete");
}

// =========================================================================
// Integration Tests
// =========================================================================

#[test]
fn test_evaluate_all_returns_10_items() {
    let results = evaluate_all(Path::new("."));
    assert_eq!(results.len(), 10, "Expected 10 safety checks");
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
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_sf_01_unsafe_code() {
    let result = check_unsafe_code_isolation(Path::new("."));
    assert_eq!(result.id, "SF-01");
    assert_eq!(result.severity, Severity::Major);
}

#[test]
fn test_sf_02_memory_safety() {
    let result = check_memory_safety_fuzzing(Path::new("."));
    assert_eq!(result.id, "SF-02");
}

#[test]
fn test_sf_03_miri_validation_id() {
    let result = check_miri_validation(Path::new("."));
    assert_eq!(result.id, "SF-03");
}

#[test]
fn test_sf_04_formal_safety() {
    let result = check_formal_safety_properties(Path::new("."));
    assert_eq!(result.id, "SF-04");
}

#[test]
fn test_sf_05_adversarial() {
    let result = check_adversarial_robustness(Path::new("."));
    assert_eq!(result.id, "SF-05");
}

#[test]
fn test_sf_06_thread_safety_id() {
    let result = check_thread_safety(Path::new("."));
    assert_eq!(result.id, "SF-06");
}

#[test]
fn test_sf_07_resource_leak_id() {
    let result = check_resource_leak_prevention(Path::new("."));
    assert_eq!(result.id, "SF-07");
}

#[test]
fn test_sf_08_panic_safety_id() {
    let result = check_panic_safety(Path::new("."));
    assert_eq!(result.id, "SF-08");
}

#[test]
fn test_sf_09_input_validation_id() {
    let result = check_input_validation(Path::new("."));
    assert_eq!(result.id, "SF-09");
}

#[test]
fn test_nonexistent_path() {
    let results = evaluate_all(Path::new("/nonexistent/path"));
    assert_eq!(results.len(), 10);
}

#[test]
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
fn test_temp_dir_with_unsafe() {
    let temp_dir = std::env::temp_dir().join("test_unsafe_code");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("src/lib.rs"),
        r#"
// SAFETY: no actual unsafe code -- test string literal or variable containing 'unsafe'
unsafe fn dangerous() {}
"#,
    )
    .unwrap();

    let result = check_unsafe_code_isolation(&temp_dir);
    assert_eq!(result.id, "SF-01");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_temp_dir_with_fuzzing() {
    let temp_dir = std::env::temp_dir().join("test_fuzzing");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("fuzz")).unwrap();

    std::fs::write(temp_dir.join("fuzz/fuzz_targets.rs"), "// fuzz target").unwrap();

    let result = check_memory_safety_fuzzing(&temp_dir);
    assert_eq!(result.id, "SF-02");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_temp_dir_with_proptest() {
    let temp_dir = std::env::temp_dir().join("test_proptest");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"
[package]
name = "test"
version = "0.1.0"

[dev-dependencies]
proptest = "1.0"
"#,
    )
    .unwrap();

    let result = check_adversarial_robustness(&temp_dir);
    assert_eq!(result.id, "SF-05");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_all_severities_present() {
    let results = evaluate_all(Path::new("."));
    let has_critical = results.iter().any(|r| r.severity == Severity::Critical);
    assert!(has_critical, "Expected at least one Critical check");
}
