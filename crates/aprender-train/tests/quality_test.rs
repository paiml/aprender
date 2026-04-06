//! Integration tests for Quality Gates module (ENT-005, ENT-006, ENT-007)

use entrenar::quality::{
    Advisory, AuditStatus, CodeQualityMetrics, DependencyAudit, FailureCategory, FailureContext,
    PmatGrade, Severity,
};

// =============================================================================
// ENT-005: CodeQualityMetrics Integration Tests
// =============================================================================

#[test]
fn test_code_quality_metrics_grade_a_threshold() {
    let metrics = CodeQualityMetrics::new(95.0, 85.0, 0);

    assert_eq!(metrics.pmat_grade, PmatGrade::A);
    assert!(metrics.meets_threshold(90.0, 80.0));
    assert!(metrics.meets_grade(PmatGrade::A));
    assert!(metrics.is_clippy_clean());
}

#[test]
fn test_code_quality_metrics_from_cargo_output() {
    // Simulate cargo llvm-cov JSON output
    let coverage_json = r#"{"data":[{"totals":{"lines":{"percent":92.5}}}]}"#;

    // Simulate cargo mutants JSON output
    let mutants_json = r#"{"total_mutants":200,"caught":170,"missed":25,"timeout":5}"#;

    let metrics = CodeQualityMetrics::from_cargo_output(coverage_json, mutants_json, 2)
        .expect("operation should succeed");

    assert!((metrics.coverage_percent - 92.5).abs() < 0.01);
    assert!((metrics.mutation_score - 85.0).abs() < 0.01);
    assert_eq!(metrics.clippy_warnings, 2);
    assert_eq!(metrics.pmat_grade, PmatGrade::B); // 92.5 >= 85, 85 >= 75 but not both A threshold
}

#[test]
fn test_code_quality_metrics_serialization_roundtrip() {
    let original = CodeQualityMetrics::new(90.0, 80.0, 5);
    let json = serde_json::to_string(&original).expect("JSON serialization should succeed");
    let restored: CodeQualityMetrics =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert!((restored.coverage_percent - original.coverage_percent).abs() < f64::EPSILON);
    assert!((restored.mutation_score - original.mutation_score).abs() < f64::EPSILON);
    assert_eq!(restored.clippy_warnings, original.clippy_warnings);
    assert_eq!(restored.pmat_grade, original.pmat_grade);
}

// =============================================================================
// ENT-006: DependencyAudit Integration Tests
// =============================================================================

#[test]
fn test_dependency_audit_clean_crate() {
    let audit = DependencyAudit::clean("serde", "1.0.200", "MIT OR Apache-2.0");

    assert_eq!(audit.crate_name, "serde");
    assert_eq!(audit.version, "1.0.200");
    assert_eq!(audit.license, "MIT OR Apache-2.0");
    assert_eq!(audit.audit_status, AuditStatus::Clean);
    assert!(!audit.is_vulnerable());
    assert_eq!(audit.max_severity(), Severity::None);
}

#[test]
fn test_dependency_audit_vulnerable_crate() {
    let advisory = Advisory::new(
        "RUSTSEC-2024-0001",
        Severity::Critical,
        "Remote code execution vulnerability",
    );

    let audit = DependencyAudit::vulnerable("unsafe-crate", "0.1.0", "MIT", vec![advisory]);

    assert!(audit.is_vulnerable());
    assert_eq!(audit.audit_status, AuditStatus::Vulnerable);
    assert_eq!(audit.max_severity(), Severity::Critical);
    assert_eq!(audit.advisories.len(), 1);
}

#[test]
fn test_dependency_audit_from_cargo_deny_output() {
    let cargo_deny_json = r#"{"type":"diagnostic","fields":{"graphs":[],"severity":"error","code":"A001","message":"Crate has known vulnerability RUSTSEC-2024-0001","labels":[{"span":{"crate":{"name":"vuln-crate","version":"1.2.3"}}}]}}"#;

    let audits =
        DependencyAudit::from_cargo_deny_output(cargo_deny_json).expect("operation should succeed");

    assert_eq!(audits.len(), 1);
    assert_eq!(audits[0].crate_name, "vuln-crate");
    assert_eq!(audits[0].version, "1.2.3");
    assert!(audits[0].is_vulnerable());
}

#[test]
fn test_dependency_audit_multiple_lines() {
    let cargo_deny_json = r#"{"type":"diagnostic","fields":{"graphs":[],"severity":"error","code":"A001","message":"Vuln 1","labels":[{"span":{"crate":{"name":"crate1","version":"1.0.0"}}}]}}
{"type":"diagnostic","fields":{"graphs":[],"severity":"error","code":"A002","message":"Vuln 2","labels":[{"span":{"crate":{"name":"crate2","version":"2.0.0"}}}]}}
{"type":"summary","fields":{"total":100}}"#;

    let audits =
        DependencyAudit::from_cargo_deny_output(cargo_deny_json).expect("operation should succeed");

    assert_eq!(audits.len(), 2);
    assert_eq!(audits[0].crate_name, "crate1");
    assert_eq!(audits[1].crate_name, "crate2");
}

#[test]
fn test_severity_ordering() {
    assert!(Severity::Critical > Severity::High);
    assert!(Severity::High > Severity::Medium);
    assert!(Severity::Medium > Severity::Low);
    assert!(Severity::Low > Severity::None);
}

// =============================================================================
// ENT-007: FailureContext Integration Tests
// =============================================================================

#[test]
fn test_failure_context_auto_categorization() {
    // Model convergence failures
    let ctx = FailureContext::new("E001", "Training failed: loss is NaN at step 500");
    assert_eq!(ctx.category, FailureCategory::ModelConvergence);

    // Resource exhaustion
    let ctx = FailureContext::new("E002", "CUDA out of memory");
    assert_eq!(ctx.category, FailureCategory::ResourceExhaustion);

    // Data quality
    let ctx = FailureContext::new("E003", "Corrupt data file detected");
    assert_eq!(ctx.category, FailureCategory::DataQuality);

    // Configuration error
    let ctx = FailureContext::new("E004", "Missing required field 'learning_rate'");
    assert_eq!(ctx.category, FailureCategory::ConfigurationError);

    // Dependency failure
    let ctx = FailureContext::new("E005", "Failed to compile crate dependency");
    assert_eq!(ctx.category, FailureCategory::DependencyFailure);
}

#[test]
fn test_failure_context_with_metadata() {
    let ctx = FailureContext::new("NAN_LOSS", "Loss became NaN at step 1000")
        .with_stack_trace("at training_loop:125\nat step:50")
        .with_suggested_fix("Try reducing learning rate to 1e-5")
        .with_related_runs(vec!["run-001".to_string(), "run-002".to_string()]);

    assert_eq!(ctx.error_code, "NAN_LOSS");
    assert!(ctx.stack_trace.is_some());
    assert!(ctx.suggested_fix.is_some());
    assert_eq!(ctx.related_runs.len(), 2);
}

#[test]
fn test_failure_context_pareto_analysis() {
    use entrenar::quality::failure::ParetoAnalysis;

    // Create a typical failure distribution (80/20 rule)
    let mut failures = Vec::new();

    // 60% convergence failures
    for i in 0..60 {
        failures.push(FailureContext::with_category(
            format!("E{i:03}"),
            "NaN loss",
            FailureCategory::ModelConvergence,
        ));
    }

    // 20% resource failures
    for i in 60..80 {
        failures.push(FailureContext::with_category(
            format!("E{i:03}"),
            "OOM",
            FailureCategory::ResourceExhaustion,
        ));
    }

    // 10% config errors
    for i in 80..90 {
        failures.push(FailureContext::with_category(
            format!("E{i:03}"),
            "Config",
            FailureCategory::ConfigurationError,
        ));
    }

    // 10% other
    for i in 90..100 {
        failures.push(FailureContext::with_category(
            format!("E{i:03}"),
            "Unknown",
            FailureCategory::Unknown,
        ));
    }

    let analysis = ParetoAnalysis::from_failures(&failures);

    // Verify total
    assert_eq!(analysis.total_failures, 100);

    // Verify top category is ModelConvergence
    assert_eq!(analysis.categories[0].0, FailureCategory::ModelConvergence);
    assert_eq!(analysis.categories[0].1, 60);

    // Verify vital few (categories accounting for ~80%)
    let vital = analysis.vital_few();
    assert!(!vital.is_empty());

    // Verify percentages
    let percentages = analysis.percentages();
    assert!((percentages[0].1 - 60.0).abs() < f64::EPSILON);
}

#[test]
fn test_failure_context_generate_suggested_fix() {
    let categories = [
        (FailureCategory::ModelConvergence, "learning rate"),
        (FailureCategory::ResourceExhaustion, "batch size"),
        (FailureCategory::DataQuality, "data format"),
        (FailureCategory::ConfigurationError, "configuration"),
        (FailureCategory::DependencyFailure, "cargo update"),
        (FailureCategory::Unknown, "debug"),
    ];

    for (category, expected_keyword) in categories {
        let ctx = FailureContext::with_category("E001", "Error", category);
        let fix = ctx.generate_suggested_fix();
        assert!(
            fix.to_lowercase().contains(expected_keyword),
            "Category {category:?} fix should contain '{expected_keyword}'"
        );
    }
}

#[test]
fn test_failure_context_from_std_error() {
    use std::io;

    let error = io::Error::new(io::ErrorKind::OutOfMemory, "System out of memory");
    let ctx = FailureContext::from(&error);

    assert_eq!(ctx.category, FailureCategory::ResourceExhaustion);
    assert!(ctx.message.contains("out of memory"));
}

// =============================================================================
// Cross-Module Integration Tests
// =============================================================================

#[test]
fn test_quality_gate_workflow() {
    // Simulate a quality gate check workflow
    let coverage_json = r#"{"data":[{"totals":{"lines":{"percent":96.0}}}]}"#;
    let mutants_json = r#"{"total_mutants":100,"caught":88,"missed":10,"timeout":2}"#;

    // Step 1: Parse code quality metrics
    let metrics = CodeQualityMetrics::from_cargo_output(coverage_json, mutants_json, 0)
        .expect("operation should succeed");

    // Step 2: Check against thresholds (95% coverage, 85% mutation required for A)
    assert!(metrics.meets_threshold(95.0, 85.0));
    assert_eq!(metrics.pmat_grade, PmatGrade::A);

    // Step 3: Simulate cargo-deny check (no vulnerabilities)
    let deny_json = r#"{"type":"summary","fields":{"total":150,"clean":150}}"#;
    let audits =
        DependencyAudit::from_cargo_deny_output(deny_json).expect("operation should succeed");
    assert!(audits.is_empty());

    // All gates pass!
}

#[test]
fn test_quality_gate_failure_workflow() {
    // Simulate a failing quality gate
    let coverage_json = r#"{"data":[{"totals":{"lines":{"percent":70.0}}}]}"#;
    let mutants_json = r#"{"total_mutants":100,"caught":50,"missed":45,"timeout":5}"#;

    let metrics = CodeQualityMetrics::from_cargo_output(coverage_json, mutants_json, 10)
        .expect("operation should succeed");

    // Check fails (70% coverage >= 60%, 50% mutation >= 50% = Grade D)
    assert!(!metrics.meets_threshold(90.0, 80.0));
    assert_eq!(metrics.pmat_grade, PmatGrade::D);
    assert!(!metrics.is_clippy_clean());

    // Create failure context for the quality gate failure
    let ctx = FailureContext::with_category(
        "QUALITY_GATE_FAILED",
        format!(
            "Quality gate failed: coverage {}%, mutation {}%, grade {}",
            metrics.coverage_percent, metrics.mutation_score, metrics.pmat_grade
        ),
        FailureCategory::ConfigurationError,
    )
    .with_suggested_fix("Increase test coverage and mutation kill rate before merging");

    assert!(ctx.suggested_fix.is_some());
}
