use super::*;

fn contracts_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

#[test]
fn lint_passes_on_real_contracts() {
    let dir = contracts_dir();
    let config = LintConfig::new(&dir, None, 0.0);
    let report = run_lint(&config);
    assert!(report.passed, "lint should pass: {report:?}");
    // 9 gates: validate, audit, score, verify, enforce, enforcement-level,
    // reverse-coverage, duplicate-stems (PV-DUP-001), composition.
    assert_eq!(report.gates.len(), 9);
}

#[test]
fn lint_score_gate_fails_with_high_threshold() {
    let dir = contracts_dir();
    let config = LintConfig::new(&dir, None, 0.99);
    let report = run_lint(&config);
    assert!(!report.passed);
    assert!(!report.findings.is_empty());
}

#[test]
fn lint_empty_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let config = LintConfig::new(tmp.path(), None, 0.0);
    let report = run_lint(&config);
    assert!(report.passed);
}

#[test]
fn lint_report_serializes_to_json() {
    let dir = contracts_dir();
    let config = LintConfig::new(&dir, None, 0.0);
    let report = run_lint(&config);
    let json = serde_json::to_string_pretty(&report).unwrap();
    assert!(json.contains("\"passed\""));
}

#[test]
fn gate_detail_variants() {
    let skipped = GateDetail::Skipped {
        reason: "test".into(),
    };
    let json = serde_json::to_string(&skipped).unwrap();
    assert!(json.contains("skipped"));
}

#[test]
fn lint_findings_on_failure() {
    let dir = contracts_dir();
    let config = LintConfig::new(&dir, None, 0.99);
    let report = run_lint(&config);
    assert!(report.findings.iter().any(|f| f.rule_id == "PV-SCR-001"));
}

#[test]
fn lint_severity_filter() {
    let dir = contracts_dir();
    let mut config = LintConfig::new(&dir, None, 0.99);
    config.severity_filter = Some(RuleSeverity::Error);
    let report = run_lint(&config);
    assert!(report
        .findings
        .iter()
        .all(|f| f.severity >= RuleSeverity::Error));
}

#[test]
fn lint_suppression_by_rule() {
    let dir = contracts_dir();
    let mut config = LintConfig::new(&dir, None, 0.99);
    config.suppressed_rules = vec!["PV-SCR-001".into()];
    let report = run_lint(&config);
    for f in &report.findings {
        if f.rule_id == "PV-SCR-001" {
            assert!(f.suppressed);
        }
    }
}

#[test]
fn lint_strict_mode() {
    let dir = contracts_dir();
    let mut config = LintConfig::new(&dir, None, 0.0);
    config.strict = true;
    let report = run_lint(&config);
    for f in &report.findings {
        assert_ne!(f.severity, RuleSeverity::Warning);
    }
}

#[test]
fn lint_sarif_output() {
    let dir = contracts_dir();
    let config = LintConfig::new(&dir, None, 0.99);
    let report = run_lint(&config);
    let sarif_log = sarif::findings_to_sarif(&report.findings, "0.1.0");
    let json = sarif::sarif_to_json(&sarif_log, true);
    assert!(json.contains("sarif-schema-2.1.0"));
    assert!(json.contains("PV-SCR-001"));
}

#[test]
fn skipped_gate_creates_correct_result() {
    let g = skipped_gate("test", "reason");
    assert_eq!(g.name, "test");
    assert!(!g.passed);
    assert!(g.skipped);
}

#[test]
fn lint_cache_populates_stats() {
    let dir = contracts_dir();
    let config = LintConfig::new(&dir, None, 0.0);
    let report = run_lint(&config);
    // Default config has cache enabled, so stats should be populated
    assert!(report.cache_stats.total > 0);
    assert_eq!(
        report.cache_stats.total,
        report.cache_stats.hits + report.cache_stats.misses
    );
}

#[test]
fn lint_no_cache_skips_stats() {
    let dir = contracts_dir();
    let mut config = LintConfig::new(&dir, None, 0.0);
    config.no_cache = true;
    let report = run_lint(&config);
    assert_eq!(report.cache_stats.total, 0);
}

#[test]
fn lint_cache_second_run_hits() {
    let tmp = tempfile::tempdir().unwrap();
    let tmp_dir = tmp.path().join("contracts");
    std::fs::create_dir_all(&tmp_dir).unwrap();
    // Copy one contract for a small test
    let src = contracts_dir().join("softmax-kernel-v1.yaml");
    std::fs::copy(&src, tmp_dir.join("softmax-kernel-v1.yaml")).unwrap();

    let config = LintConfig::new(&tmp_dir, None, 0.0);
    let r1 = run_lint(&config);
    assert!(r1.cache_stats.misses > 0);

    let r2 = run_lint(&config);
    assert!(r2.cache_stats.hits > 0);
}

#[test]
fn lint_validation_failure_skips_audit_and_score() {
    let tmp = tempfile::tempdir().unwrap();
    // Write a malformed YAML that will parse into a Contract with validation errors
    // Actually: write something that fails to parse entirely
    std::fs::write(tmp.path().join("bad.yaml"), "not: valid: yaml: {{{{").unwrap();
    let config = LintConfig::new(tmp.path(), None, 0.0);
    let report = run_lint(&config);
    assert!(!report.passed);
    // validate should fail, all subsequent gates should be skipped
    assert_eq!(report.gates.len(), 9);
    assert!(!report.gates[0].passed); // validate failed
    assert!(report.gates[1].skipped); // audit skipped
    assert!(report.gates[2].skipped); // score skipped
    assert!(report.gates[3].skipped); // verify skipped
    assert!(report.gates[4].skipped); // enforce skipped
    assert!(report.gates[5].skipped); // enforcement-level skipped
    assert!(report.gates[6].skipped); // reverse-coverage skipped
    assert!(report.gates[7].skipped); // duplicate-stems skipped
    assert!(report.gates[8].skipped); // composition skipped
}

#[test]
fn lint_suppression_by_stem() {
    let dir = contracts_dir();
    let mut config = LintConfig::new(&dir, None, 0.99);
    // Suppress by contract stem (--suppress)
    config.suppressed_findings = vec!["special-tokens-registry-v1".into()];
    let report = run_lint(&config);
    for f in &report.findings {
        if f.contract_stem.as_deref() == Some("special-tokens-registry-v1") {
            assert!(f.suppressed);
        }
    }
}

#[test]
fn lint_suppression_by_file_pattern() {
    let dir = contracts_dir();
    let mut config = LintConfig::new(&dir, None, 0.99);
    config.suppressed_files = vec!["arch-constraints".into()];
    let report = run_lint(&config);
    for f in &report.findings {
        if f.file.contains("arch-constraints") {
            assert!(f.suppressed);
        }
    }
}

#[test]
fn lint_severity_override() {
    let dir = contracts_dir();
    let mut config = LintConfig::new(&dir, None, 0.99);
    let mut overrides = HashMap::new();
    overrides.insert("PV-SCR-001".into(), RuleSeverity::Warning);
    config.severity_overrides = overrides;
    let report = run_lint(&config);
    for f in &report.findings {
        if f.rule_id == "PV-SCR-001" {
            assert_eq!(f.severity, RuleSeverity::Warning);
        }
    }
}

#[test]
fn lifecycle_first_run_all_new() {
    let tmp = tempfile::tempdir().unwrap();
    let contract_dir = tmp.path().join("contracts");
    std::fs::create_dir_all(&contract_dir).unwrap();
    let src = contracts_dir().join("softmax-kernel-v1.yaml");
    std::fs::copy(&src, contract_dir.join("softmax-kernel-v1.yaml")).unwrap();

    let config = LintConfig::new(&contract_dir, None, 0.99);
    let report = run_lint(&config);
    // First run: every finding should be marked new
    let active: Vec<_> = report.findings.iter().filter(|f| !f.suppressed).collect();
    assert!(!active.is_empty(), "should have findings");
    assert!(
        active.iter().all(|f| f.is_new),
        "first run: all findings should be new"
    );
}

#[test]
fn lifecycle_second_run_pre_existing() {
    let tmp = tempfile::tempdir().unwrap();
    let contract_dir = tmp.path().join("contracts");
    std::fs::create_dir_all(&contract_dir).unwrap();
    let src = contracts_dir().join("softmax-kernel-v1.yaml");
    std::fs::copy(&src, contract_dir.join("softmax-kernel-v1.yaml")).unwrap();

    // First run — seeds .pv/lint-previous.json
    let config = LintConfig::new(&contract_dir, None, 0.99);
    let _ = run_lint(&config);

    // Second run — same contract, same findings
    let report2 = run_lint(&config);
    let active: Vec<_> = report2.findings.iter().filter(|f| !f.suppressed).collect();
    assert!(!active.is_empty(), "should have findings");
    assert!(
        active.iter().all(|f| !f.is_new),
        "second identical run: no finding should be new"
    );
}

#[test]
fn lifecycle_persists_fingerprints() {
    let tmp = tempfile::tempdir().unwrap();
    let contract_dir = tmp.path().join("contracts");
    std::fs::create_dir_all(&contract_dir).unwrap();
    let src = contracts_dir().join("softmax-kernel-v1.yaml");
    std::fs::copy(&src, contract_dir.join("softmax-kernel-v1.yaml")).unwrap();

    let config = LintConfig::new(&contract_dir, None, 0.99);
    let _ = run_lint(&config);

    let previous_path = tmp.path().join(".pv").join("lint-previous.json");
    assert!(
        previous_path.exists(),
        ".pv/lint-previous.json should be created"
    );
    let content = std::fs::read_to_string(&previous_path).unwrap();
    let fps: std::collections::HashSet<String> = serde_json::from_str(&content).unwrap();
    assert!(!fps.is_empty(), "fingerprint set should not be empty");
}

#[test]
fn lifecycle_mark_new_findings_unit() {
    use super::mark_new_findings;

    let tmp = tempfile::tempdir().unwrap();
    let contract_dir = tmp.path().join("contracts");
    std::fs::create_dir_all(&contract_dir).unwrap();

    let mut findings = vec![
        finding::LintFinding::new("PV-VAL-001", RuleSeverity::Error, "msg1", "a.yaml"),
        finding::LintFinding::new("PV-VAL-002", RuleSeverity::Warning, "msg2", "b.yaml"),
    ];

    // First call: all new (no previous file)
    mark_new_findings(&mut findings, &contract_dir);
    assert!(findings[0].is_new);
    assert!(findings[1].is_new);

    // Reset is_new for second pass
    for f in &mut findings {
        f.is_new = false;
    }

    // Second call: same findings, none new
    mark_new_findings(&mut findings, &contract_dir);
    assert!(!findings[0].is_new);
    assert!(!findings[1].is_new);

    // Third call: add a new finding
    findings.push(finding::LintFinding::new(
        "PV-VAL-003",
        RuleSeverity::Info,
        "msg3",
        "c.yaml",
    ));
    for f in &mut findings {
        f.is_new = false;
    }
    mark_new_findings(&mut findings, &contract_dir);
    assert!(
        !findings[0].is_new,
        "pre-existing finding should not be new"
    );
    assert!(
        !findings[1].is_new,
        "pre-existing finding should not be new"
    );
    assert!(findings[2].is_new, "newly added finding should be new");
}
