use super::*;

fn contracts_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

// Linting the real corpus is the cost of this file: each run lints every contract in the repo,
// and nextest runs every #[test] in its own process, so nothing can be shared between tests. So
// the corpus is linted by exactly two tests, one per score threshold, and each read-only check
// on it is a plain fn those two call. Tests that turn one config knob lint `knob_corpus`.

/// A three-contract corpus for the tests that turn one config knob. It holds each finding they
/// look at (PV-SCR-001 below 0.99, the special-tokens stem, an arch-constraints file) without
/// re-linting the whole repo per test. Nested, not a bare tempdir (#4173).
fn knob_corpus() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("contracts");
    std::fs::create_dir_all(&dir).unwrap();
    for stem in [
        "softmax-kernel-v1",
        "special-tokens-registry-v1",
        "arch-constraints-v1",
    ] {
        let f = format!("{stem}.yaml");
        std::fs::copy(contracts_dir().join(&f), dir.join(&f)).unwrap();
    }
    (tmp, dir)
}

#[test]
fn lint_passes_on_real_contracts() {
    let report = run_lint(&LintConfig::new(&contracts_dir(), None, 0.0));
    lint_report_serializes_to_json(&report);
    lint_cache_populates_stats(&report);
    every_gate_verdict_agrees_with_passed_and_skipped_on_the_real_corpus(&report);
    valid_under_is_computed_on_the_real_corpus(&report);
    assert!(report.passed, "lint should pass: {report:?}");
    // 15 gates: validate, audit, score, verify, enforce, enforcement-level, reverse-coverage,
    // duplicate-stems (PV-DUP-001), composition, sigma (ONT-2b), relations (ONT-4), shapes (ONT-4b),
    // valid-under (ONT-7), theorem-pairing and depends-on-present (PVL-001 EV-11).
    assert_eq!(report.gates.len(), 15);
}

fn lint_score_gate_fails_with_high_threshold(report: &LintReport) {
    assert!(!report.passed);
    assert!(!report.findings.is_empty());
}

#[test]
fn lint_empty_dir() {
    // The lint takes the contract dir's PARENT as the project root and reads
    // `scripts/contract_duplicate_stem_baseline.txt` from it. A bare tempdir's parent is
    // the shared `/tmp`, so a stray `/tmp/scripts/` failed this test (#4207). Nest it.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("contracts");
    std::fs::create_dir_all(&dir).unwrap();
    let config = LintConfig::new(&dir, None, 0.0);
    let report = run_lint(&config);
    assert!(report.passed, "empty dir should pass: {report:?}");
    // The new-finding state is the fixture's own, not `$TMPDIR/.pv` shared with every other run (#4173).
    assert_eq!(pv_state_dir(&dir), tmp.path().join(".pv"));
    assert!(tmp.path().join(".pv/lint-previous.json").is_file());
}

/// #4173: `pv_state_dir` is the contract dir's PARENT, so a test that lints a bare `tempdir()` reads
/// and writes `$TMPDIR/.pv/lint-previous.json`, a file every other run on the host shares. Its
/// verdict then depends on state it does not own. Every lint test must nest its corpus.
#[test]
fn no_lint_test_points_the_lint_at_a_bare_tempdir() {
    let bare = |line: &str| {
        let l: String = line.chars().filter(|c| !c.is_whitespace()).collect();
        ["(tmp.path()", "(&tmp.path()"].iter().any(|arg| {
            let pat = format!("{}{arg}", concat!("LintConfig::", "new"));
            l.match_indices(&pat)
                .any(|(k, _)| matches!(l[k + pat.len()..].chars().next(), Some(',' | ')')))
        })
    };
    for (line, want) in [
        (
            concat!("run_lint(&LintConfig::new", "(tmp.path(), None, 0.0));"),
            true,
        ),
        (
            concat!("let c = LintConfig::new", "( &tmp.path(), None, 0.0);"),
            true,
        ),
        ("let c = LintConfig::new(&corpus, None, 0.0);", false),
        (
            "let c = LintConfig::new(&tmp.path().join(\"contracts\"), None, 0.0);",
            false,
        ),
    ] {
        assert_eq!(bare(line), want, "case table: {line}");
    }
    let lint_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lint");
    let mut scanned = 0;
    for entry in std::fs::read_dir(&lint_src).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "rs") {
            scanned += 1;
            let src = std::fs::read_to_string(&path).unwrap();
            for (i, line) in src.lines().enumerate() {
                assert!(
                    !bare(line),
                    "{}:{}: lints a bare tempdir: {line}",
                    path.display(),
                    i + 1
                );
            }
        }
    }
    assert!(
        scanned > 5,
        "scanned only {scanned} files under {}",
        lint_src.display()
    );
}

fn lint_report_serializes_to_json(report: &LintReport) {
    let json = serde_json::to_string_pretty(report).unwrap();
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
    let report = run_lint(&LintConfig::new(&contracts_dir(), None, 0.99));
    lint_score_gate_fails_with_high_threshold(&report);
    lint_sarif_output(&report);
    assert!(report.findings.iter().any(|f| f.rule_id == "PV-SCR-001"));
}

#[test]
fn lint_severity_filter() {
    let (_tmp, dir) = knob_corpus();
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
    let (_tmp, dir) = knob_corpus();
    let mut config = LintConfig::new(&dir, None, 0.99);
    config.suppressed_rules = vec!["PV-SCR-001".into()];
    let report = run_lint(&config);
    assert!(report.findings.iter().any(|f| f.rule_id == "PV-SCR-001"));
    for f in &report.findings {
        if f.rule_id == "PV-SCR-001" {
            assert!(f.suppressed);
        }
    }
}

#[test]
fn lint_strict_mode() {
    let (_tmp, dir) = knob_corpus();
    let mut config = LintConfig::new(&dir, None, 0.0);
    config.strict = true;
    let report = run_lint(&config);
    for f in &report.findings {
        assert_ne!(f.severity, RuleSeverity::Warning);
    }
}

fn lint_sarif_output(report: &LintReport) {
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

fn lint_cache_populates_stats(report: &LintReport) {
    // Default config has cache enabled, so stats should be populated
    assert!(report.cache_stats.total > 0);
    assert_eq!(
        report.cache_stats.total,
        report.cache_stats.hits + report.cache_stats.misses
    );
}

#[test]
fn lint_no_cache_skips_stats() {
    let (_tmp, dir) = knob_corpus();
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
    let dir = tmp.path().join("contracts");
    std::fs::create_dir_all(&dir).unwrap();
    // Write a malformed YAML that will parse into a Contract with validation errors
    // Actually: write something that fails to parse entirely
    std::fs::write(dir.join("bad.yaml"), "not: valid: yaml: {{{{").unwrap();
    let config = LintConfig::new(&dir, None, 0.0);
    let report = run_lint(&config);
    assert!(!report.passed);
    // validate should fail, all subsequent gates should be skipped
    assert_eq!(report.gates.len(), 15);
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
    let (_tmp, dir) = knob_corpus();
    let mut config = LintConfig::new(&dir, None, 0.99);
    // Suppress by contract stem (--suppress)
    config.suppressed_findings = vec!["special-tokens-registry-v1".into()];
    let report = run_lint(&config);
    assert!(report
        .findings
        .iter()
        .any(|f| f.contract_stem.as_deref() == Some("special-tokens-registry-v1")));
    for f in &report.findings {
        if f.contract_stem.as_deref() == Some("special-tokens-registry-v1") {
            assert!(f.suppressed);
        }
    }
}

#[test]
fn lint_suppression_by_file_pattern() {
    let (_tmp, dir) = knob_corpus();
    let mut config = LintConfig::new(&dir, None, 0.99);
    config.suppressed_files = vec!["arch-constraints".into()];
    let report = run_lint(&config);
    assert!(report
        .findings
        .iter()
        .any(|f| f.file.contains("arch-constraints")));
    for f in &report.findings {
        if f.file.contains("arch-constraints") {
            assert!(f.suppressed);
        }
    }
}

#[test]
fn lint_severity_override() {
    let (_tmp, dir) = knob_corpus();
    let mut config = LintConfig::new(&dir, None, 0.99);
    let mut overrides = HashMap::new();
    overrides.insert("PV-SCR-001".into(), RuleSeverity::Warning);
    config.severity_overrides = overrides;
    let report = run_lint(&config);
    assert!(report.findings.iter().any(|f| f.rule_id == "PV-SCR-001"));
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

/// ONT-6 (PMAT-3451): every gate of a real run carries the lattice element its own `passed`/`skipped`
/// pair maps to, and the report's verdict is the meet over the default armed set — so no constructor can
/// set a verdict that disagrees with the booleans it sits beside.
fn every_gate_verdict_agrees_with_passed_and_skipped_on_the_real_corpus(report: &LintReport) {
    for g in &report.gates {
        assert_eq!(
            g.verdict,
            Verdict::from_gate(g.passed, g.skipped),
            "gate {}",
            g.name
        );
    }
    let armed = ArmedGates::default_set();
    let expected = armed
        .names()
        .iter()
        .map(|n| {
            report.gates.iter().find(|g| &g.name == n).map_or(
                Verdict::Unknown(crate::ontology::verdict::Reason::NotRun),
                |g| g.verdict,
            )
        })
        .fold(Verdict::Pass, Verdict::meet);
    assert_eq!(report.verdict, expected);
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "the repo corpus passes its armed meet"
    );
    // `run_lint` arms the DEFAULT set (the 8), so the three gates outside it are reported and excluded. The repo's
    // own `lint-baseline.json` arms `sigma` and `relations` as well — per-repo declarations, not the default.
    assert_eq!(
        report.not_armed,
        vec![
            "reverse-coverage".to_string(),
            "sigma".to_string(),
            "relations".to_string(),
            "shapes".to_string(),
            // ONT-7, R-8: computed in every run, armed only when the baseline names it.
            "valid-under".to_string(),
            // PVL-001 EV-11: computed in every run (R-8), armed per repo.
            "theorem-pairing".to_string(),
            "depends-on-present".to_string(),
        ]
    );
}

/// ONT-001 §3.9: arming is the corpus's declaration, not the command line. A gate a flag ran is computed and
/// reported, and stays outside the meet unless `armed_gates` names it.
#[test]
fn a_gate_a_flag_ran_is_reported_but_not_armed() {
    let tmp = tempfile::tempdir().unwrap();
    let corpus = tmp.path().join("contracts");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::copy(
        contracts_dir().join("softmax-kernel-v1.yaml"),
        corpus.join("softmax-kernel-v1.yaml"),
    )
    .unwrap();
    let mut config = LintConfig::new(&corpus, None, 0.0);
    config.strict_test_binding = true;
    let report = run_lint(&config);
    assert!(
        report.gates.iter().any(|g| g.name == "strict-test-binding"),
        "the flag ran the gate"
    );
    assert!(
        report.not_armed.iter().any(|n| n == "strict-test-binding"),
        "not declared, so not armed: {:?}",
        report.not_armed
    );
    assert!(!report
        .armed_gates
        .iter()
        .any(|g| g.name == "strict-test-binding"));
    assert_eq!(report.armed_gates.len(), 8);
}

/// ONT-7, R-8, the computed half: gate 13 runs and passes on the repo corpus.
fn valid_under_is_computed_on_the_real_corpus(report: &LintReport) {
    let g = report
        .gates
        .iter()
        .find(|g| g.name == "valid-under")
        .expect("gate 13 is in every run");
    assert!(
        !g.skipped && g.passed,
        "computed and Pass on the repo corpus: {g:?}"
    );
}

/// ONT-7, R-8: gate 13 is COMPUTED when validation passes and SKIPPED, naming why, when it fails. Both halves
/// at the lib level, because the CI mutation lane runs `--lib` only (#4076 round-2 review): a mutant that
/// inverts `validation_passed` in `valid_under_result` must fail here, not only in the CLI integration test.
#[test]
fn valid_under_is_computed_when_validation_passes_and_skipped_when_it_fails() {
    // The computed half runs on the real corpus in `lint_passes_on_real_contracts`.
    // Σ and a kernel contract are present, so the ONLY reason to skip is the failed validation.
    // Nested, not the bare tempdir: the lint writes its state into the contract dir's PARENT (#4173).
    let tmp = tempfile::tempdir().unwrap();
    let corpus = tmp.path().join("contracts");
    std::fs::create_dir_all(&corpus).unwrap();
    let fixture = contracts_dir().join("../tests/fixtures/ont/valid-under-ok");
    for f in ["ontology.yaml", "fixture-vu-v1.yaml"] {
        std::fs::copy(fixture.join(f), corpus.join(f)).unwrap();
    }
    std::fs::write(corpus.join("bad.yaml"), "not: valid: yaml: {{{{").unwrap();
    let report = run_lint(&LintConfig::new(&corpus, None, 0.0));
    let g = report
        .gates
        .iter()
        .find(|g| g.name == "valid-under")
        .expect("gate 13 is in every run");
    assert!(
        g.skipped,
        "validation failed, so the gate is skipped: {g:?}"
    );
    assert!(
        matches!(&g.detail, GateDetail::Skipped { reason } if reason == "validation failed"),
        "{g:?}"
    );
}
