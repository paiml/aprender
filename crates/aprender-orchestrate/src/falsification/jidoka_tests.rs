use super::*;

// =========================================================================
// JA-01: Pre-Commit Hook Tests
// =========================================================================

#[test]
fn test_ja_01_precommit_hooks() {
    let result = check_precommit_hooks(Path::new("."));
    // batuta should have some form of pre-commit
    assert!(!matches!(result.status, CheckStatus::Skipped), "Pre-commit check should complete");
}

// =========================================================================
// JA-02: Sovereignty Linting Tests
// =========================================================================

#[test]
fn test_ja_02_sovereignty_linting() {
    let result = check_automated_sovereignty_linting(Path::new("."));
    assert!(!matches!(result.status, CheckStatus::Skipped), "Linting check should complete");
}

// =========================================================================
// JA-03: Data Drift Circuit Breaker Tests
// =========================================================================

#[test]
fn test_ja_03_data_drift_circuit_breaker() {
    let result = check_data_drift_circuit_breaker(Path::new("."));
    assert!(!matches!(result.status, CheckStatus::Skipped), "Data drift check should complete");
}

// =========================================================================
// JA-04: Performance Regression Tests
// =========================================================================

#[test]
fn test_ja_04_performance_regression() {
    let result = check_performance_regression_gate(Path::new("."));
    assert!(!matches!(result.status, CheckStatus::Skipped), "Performance check should complete");
}

// =========================================================================
// JA-05: Fairness Metric Circuit Breaker Tests
// =========================================================================

#[test]
fn test_ja_05_fairness_circuit_breaker() {
    let result = check_fairness_metric_circuit_breaker(Path::new("."));
    assert!(!matches!(result.status, CheckStatus::Skipped), "Fairness check should complete");
}

// =========================================================================
// JA-06: Latency SLA Circuit Breaker Tests
// =========================================================================

#[test]
fn test_ja_06_latency_sla_circuit_breaker() {
    let result = check_latency_sla_circuit_breaker(Path::new("."));
    assert!(!matches!(result.status, CheckStatus::Skipped), "Latency check should complete");
}

// =========================================================================
// JA-07: Memory Footprint Gate Tests
// =========================================================================

#[test]
fn test_ja_07_memory_footprint_gate() {
    let result = check_memory_footprint_gate(Path::new("."));
    assert!(!matches!(result.status, CheckStatus::Skipped), "Memory check should complete");
}

// =========================================================================
// JA-08: Security Scan Tests
// =========================================================================

#[test]
fn test_ja_08_security_scan() {
    let result = check_security_scan_gate(Path::new("."));
    assert!(!matches!(result.status, CheckStatus::Skipped), "Security check should complete");
}

// =========================================================================
// JA-09: License Compliance Tests
// =========================================================================

#[test]
fn test_ja_09_license_compliance() {
    let result = check_license_compliance_gate(Path::new("."));
    // batuta should have LICENSE
    assert!(
        matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
        "License check failed: {:?}",
        result.rejection_reason
    );
}

// =========================================================================
// JA-10: Documentation Gate Tests
// =========================================================================

#[test]
fn test_ja_10_documentation_gate() {
    let result = check_documentation_gate(Path::new("."));
    // batuta should have README
    assert!(
        matches!(result.status, CheckStatus::Pass | CheckStatus::Partial),
        "Documentation check failed: {:?}",
        result.rejection_reason
    );
}

// =========================================================================
// Integration Tests
// =========================================================================

#[test]
fn test_evaluate_all_returns_10_items() {
    let results = evaluate_all(Path::new("."));
    assert_eq!(results.len(), 10, "Expected 10 Jidoka checks");
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
fn test_nonexistent_path() {
    let results = evaluate_all(Path::new("/nonexistent/path/that/does/not/exist"));
    // Should still return 10 items, likely all failed
    assert_eq!(results.len(), 10);
}

#[test]
fn test_check_ci_for_content_nonexistent() {
    let result = check_ci_for_content(Path::new("/nonexistent/path"), "cargo test");
    assert!(!result);
}

#[test]
fn test_check_ci_for_content_exists() {
    // batuta has CI configured
    let result = check_ci_for_content(Path::new("."), "rust");
    // Should check all CI files and find some content
    // Result depends on actual CI config, so just test it doesn't panic
    let _ = result;
}

#[test]
fn test_all_items_have_valid_ids() {
    let results = evaluate_all(Path::new("."));
    for item in &results {
        assert!(item.id.starts_with("JA-"), "Item ID {} should start with JA-", item.id);
    }
}

#[test]
fn test_all_items_have_durations() {
    let results = evaluate_all(Path::new("."));
    for item in &results {
        // duration_ms is set by with_duration() call
        // All checks in this module set duration
        // duration_ms is u64 — verify evaluation actually ran (not stuck at max)
        assert!(
            item.duration_ms < 60_000,
            "Item {} took too long: {}ms",
            item.id,
            item.duration_ms
        );
    }
}

#[test]
fn test_all_items_have_valid_severities() {
    let results = evaluate_all(Path::new("."));
    for item in &results {
        assert!(
            matches!(
                item.severity,
                Severity::Critical | Severity::Major | Severity::Minor | Severity::Info
            ),
            "Item {} has unexpected severity",
            item.id
        );
    }
}

#[test]
fn test_all_items_have_claims() {
    let results = evaluate_all(Path::new("."));
    for item in &results {
        assert!(!item.claim.is_empty(), "Item {} missing claim", item.id);
    }
}

#[test]
fn test_all_items_have_names() {
    let results = evaluate_all(Path::new("."));
    for item in &results {
        assert!(!item.name.is_empty(), "Item {} missing name", item.id);
    }
}

#[test]
fn test_ja_01_has_correct_severity() {
    let result = check_precommit_hooks(Path::new("."));
    assert_eq!(result.severity, Severity::Major);
}

#[test]
fn test_ja_08_has_critical_severity() {
    let result = check_security_scan_gate(Path::new("."));
    assert_eq!(result.severity, Severity::Critical);
}

#[test]
fn test_ja_10_has_minor_severity() {
    let result = check_documentation_gate(Path::new("."));
    assert_eq!(result.severity, Severity::Minor);
}

#[test]
fn test_check_items_order() {
    let results = evaluate_all(Path::new("."));
    let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(
        ids,
        vec![
            "JA-01", "JA-02", "JA-03", "JA-04", "JA-05", "JA-06", "JA-07", "JA-08", "JA-09",
            "JA-10"
        ]
    );
}

#[test]
fn test_evidence_type_is_static_analysis() {
    let results = evaluate_all(Path::new("."));
    for item in &results {
        for evidence in &item.evidence {
            assert!(
                matches!(evidence.evidence_type, EvidenceType::StaticAnalysis),
                "Item {} has non-static-analysis evidence",
                item.id
            );
        }
    }
}

#[test]
fn test_evidence_descriptions_not_empty() {
    let results = evaluate_all(Path::new("."));
    for item in &results {
        for evidence in &item.evidence {
            assert!(
                !evidence.description.is_empty(),
                "Item {} has empty evidence description",
                item.id
            );
        }
    }
}
