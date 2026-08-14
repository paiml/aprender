#![allow(clippy::all)]
use super::*;
use crate::schema::{
    Contract, EnforcementLevel, Equation, FalsificationTest, KaniHarness, Metadata,
    VerificationSummary,
};

/// Build a minimal contract with no equations, no tests, no harnesses.
fn minimal_contract() -> Contract {
    Contract {
        metadata: Metadata {
            version: "1".into(),
            description: "test contract".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn make_falsification_test(id: &str, test_name: &str) -> FalsificationTest {
    FalsificationTest {
        id: id.into(),
        rule: "test rule".into(),
        prediction: "test prediction".into(),
        test: Some(test_name.into()),
        if_fails: "investigate".into(),
        ..Default::default()
    }
}

fn make_kani_harness(id: &str) -> KaniHarness {
    KaniHarness {
        id: id.into(),
        obligation: "test-obl".into(),
        property: None,
        bound: None,
        strategy: None,
        solver: None,
        harness: None,
        actually_verified: None,
    }
}

fn make_equation(
    preconditions: Vec<&str>,
    postconditions: Vec<&str>,
    lean: Option<&str>,
) -> Equation {
    Equation {
        formula: "x + y".into(),
        domain: None,
        codomain: None,
        invariants: vec![],
        preconditions: preconditions.into_iter().map(String::from).collect(),
        postconditions: postconditions.into_iter().map(String::from).collect(),
        lean_theorem: lean.map(String::from),
        float_tolerance: None,
        assumes: None,
        guarantees: None,
    }
}
// ---------------------------------------------------------------------------
// collect_test_fns
// ---------------------------------------------------------------------------

#[test]
fn test_collect_test_fns_finds_test_functions() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("example.rs"),
        "fn test_basic_add() {}\nfn test_edge_case() {}\nfn prop_monotonic() {}\nfn helper_function() {}\n",
    )
    .unwrap();

    let mut tests = HashSet::new();
    collect_test_fns(dir.path(), &mut tests);

    assert!(tests.contains("test_basic_add"));
    assert!(tests.contains("test_edge_case"));
    assert!(tests.contains("prop_monotonic"));
    assert!(
        !tests.contains("helper_function"),
        "should not collect non-test functions"
    );
}

#[test]
fn test_collect_test_fns_empty_dir() {
    let dir = tempfile::tempdir().unwrap();
    let mut tests = HashSet::new();
    collect_test_fns(dir.path(), &mut tests);
    assert!(tests.is_empty());
}

#[test]
fn test_collect_test_fns_nested_dirs() {
    let dir = tempfile::tempdir().unwrap();
    let sub = dir.path().join("submodule");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(sub.join("tests.rs"), "fn test_nested() {}\n").unwrap();

    let mut tests = HashSet::new();
    collect_test_fns(dir.path(), &mut tests);
    assert!(tests.contains("test_nested"));
}

#[test]
fn test_collect_test_fns_ignores_non_rs_files() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("test.txt"), "fn test_fake() {}\n").unwrap();
    std::fs::write(dir.path().join("test.py"), "def test_python(): pass\n").unwrap();

    let mut tests = HashSet::new();
    collect_test_fns(dir.path(), &mut tests);
    assert!(tests.is_empty());
}

#[test]
fn test_collect_test_fns_extracts_name_correctly() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lib.rs"),
        "    fn test_with_spaces() {\n        fn test_inside_body() {}\n    }\n",
    )
    .unwrap();

    let mut tests = HashSet::new();
    collect_test_fns(dir.path(), &mut tests);
    // Both should be found since we do line-by-line scan
    assert!(tests.contains("test_with_spaces"));
    assert!(tests.contains("test_inside_body"));
}

// ---------------------------------------------------------------------------
// run_verify_gate
// ---------------------------------------------------------------------------

#[test]
fn test_verify_gate_all_tests_present() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(
        src.join("lib.rs"),
        "fn test_alpha() {}\nfn test_beta() {}\n",
    )
    .unwrap();

    let mut c = minimal_contract();
    c.falsification_tests = vec![
        make_falsification_test("F-01", "test_alpha"),
        make_falsification_test("F-02", "test_beta"),
    ];

    let contracts = vec![("test-contract-v1".to_string(), c)];
    let (result, findings) = run_verify_gate(&contracts, dir.path());

    assert!(result.passed);
    assert!(findings.is_empty());
    if let GateDetail::Verify {
        total_refs,
        existing,
        missing,
    } = result.detail
    {
        assert_eq!(total_refs, 2);
        assert_eq!(existing, 2);
        assert_eq!(missing, 0);
    } else {
        panic!("expected Verify detail");
    }
}

#[test]
fn test_verify_gate_missing_test() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), "fn test_alpha() {}\n").unwrap();

    let mut c = minimal_contract();
    c.falsification_tests = vec![
        make_falsification_test("F-01", "test_alpha"),
        make_falsification_test("F-02", "test_missing_fn"),
    ];

    let contracts = vec![("test-contract-v1".to_string(), c)];
    let (result, findings) = run_verify_gate(&contracts, dir.path());

    assert!(!result.passed, "gate should fail when test is missing");
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].rule_id, "PV-VER-001");
    assert!(findings[0].message.contains("test_missing_fn"));
}

#[test]
fn test_verify_gate_gpu_test_is_warning_not_error() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), "").unwrap();

    let mut c = minimal_contract();
    c.falsification_tests = vec![make_falsification_test("F-01", "test_gpu_dispatch")];

    let contracts = vec![("gpu-kernel-v1".to_string(), c)];
    let (result, findings) = run_verify_gate(&contracts, dir.path());

    // GPU tests get Warning severity, so gate should still pass
    assert!(result.passed, "gate should pass for GPU-only warnings");
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].severity, RuleSeverity::Warning);
    assert!(findings[0].message.contains("may require --features gpu"));
}

#[test]
fn test_verify_gate_skips_non_test_names() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), "").unwrap();

    let mut c = minimal_contract();
    // test name that does not start with test_ or prop_
    c.falsification_tests = vec![make_falsification_test("F-01", "verify_something")];

    let contracts = vec![("test-v1".to_string(), c)];
    let (result, findings) = run_verify_gate(&contracts, dir.path());

    assert!(result.passed);
    assert!(findings.is_empty(), "non-test/prop names should be skipped");
}

#[test]
fn test_verify_gate_no_falsification_tests() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), "").unwrap();

    let c = minimal_contract();
    let contracts = vec![("test-v1".to_string(), c)];
    let (result, findings) = run_verify_gate(&contracts, dir.path());

    assert!(result.passed);
    assert!(findings.is_empty());
}

#[test]
fn test_verify_gate_test_with_module_path() {
    // Test names like "module::test_foo" should be resolved to just "test_foo"
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), "fn test_foo() {}\n").unwrap();

    let mut c = minimal_contract();
    c.falsification_tests = vec![make_falsification_test(
        "F-01",
        "my_module::submod::test_foo",
    )];

    let contracts = vec![("test-v1".to_string(), c)];
    let (result, findings) = run_verify_gate(&contracts, dir.path());

    assert!(result.passed);
    assert!(findings.is_empty());
}

#[test]
fn test_verify_gate_test_name_with_quotes() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), "fn test_quoted() {}\n").unwrap();

    let mut c = minimal_contract();
    c.falsification_tests = vec![make_falsification_test("F-01", "\"test_quoted\"")];

    let contracts = vec![("test-v1".to_string(), c)];
    let (result, findings) = run_verify_gate(&contracts, dir.path());

    assert!(result.passed);
    assert!(findings.is_empty());
}

// ---------------------------------------------------------------------------
// run_enforce_gate
// ---------------------------------------------------------------------------

#[test]
fn test_enforce_gate_no_equations() {
    let c = minimal_contract();
    let contracts = vec![("test-v1".to_string(), c)];
    let (result, findings) = run_enforce_gate(&contracts);

    assert!(result.passed);
    assert!(findings.is_empty());
    if let GateDetail::Enforce {
        equations_total, ..
    } = result.detail
    {
        assert_eq!(equations_total, 0);
    } else {
        panic!("expected Enforce detail");
    }
}

#[test]
fn test_enforce_gate_equation_with_all_fields() {
    let mut c = minimal_contract();
    c.equations.insert(
        "softmax".to_string(),
        make_equation(
            vec!["input.len() > 0"],
            vec!["result.sum() == 1.0"],
            Some("Theorems.Softmax"),
        ),
    );

    let contracts = vec![("test-v1".to_string(), c)];
    let (result, findings) = run_enforce_gate(&contracts);

    assert!(result.passed);
    assert!(findings.is_empty());
    if let GateDetail::Enforce {
        equations_total,
        equations_with_pre,
        equations_with_post,
        equations_with_lean,
    } = result.detail
    {
        assert_eq!(equations_total, 1);
        assert_eq!(equations_with_pre, 1);
        assert_eq!(equations_with_post, 1);
        assert_eq!(equations_with_lean, 1);
    } else {
        panic!("expected Enforce detail");
    }
}

#[test]
fn test_enforce_gate_equation_missing_preconditions() {
    let mut c = minimal_contract();
    c.equations.insert(
        "silu".to_string(),
        make_equation(vec![], vec!["result > 0"], Some("Theorems.Silu")),
    );

    let contracts = vec![("test-v1".to_string(), c)];
    let (result, findings) = run_enforce_gate(&contracts);

    assert!(
        result.passed,
        "missing preconditions are warnings, not errors"
    );
    let pre_findings: Vec<_> = findings
        .iter()
        .filter(|f| f.rule_id == "PV-ENF-001")
        .collect();
    assert_eq!(pre_findings.len(), 1);
    assert!(pre_findings[0].message.contains("no preconditions"));
    assert!(pre_findings[0].suggestion.is_some());
}

#[test]
fn test_enforce_gate_equation_missing_lean_theorem() {
    let mut c = minimal_contract();
    c.equations.insert(
        "gelu".to_string(),
        make_equation(vec!["x > 0"], vec!["y > 0"], None),
    );

    let contracts = vec![("test-v1".to_string(), c)];
    let (result, findings) = run_enforce_gate(&contracts);

    assert!(result.passed, "missing lean is a warning, not error");
    let lean_findings: Vec<_> = findings
        .iter()
        .filter(|f| f.rule_id == "PV-ENF-002")
        .collect();
    assert_eq!(lean_findings.len(), 1);
    assert!(lean_findings[0].message.contains("no lean_theorem"));
    // Suggestion should capitalize the equation name
    let suggestion = lean_findings[0].suggestion.as_deref().unwrap();
    assert!(
        suggestion.contains("Gelu"),
        "expected capitalized 'Gelu' in suggestion: {suggestion}"
    );
}

#[test]
fn test_enforce_gate_skips_registries() {
    let mut c = minimal_contract();
    c.metadata.registry = true;
    c.equations.insert(
        "registry_eq".to_string(),
        make_equation(vec![], vec![], None),
    );

    let contracts = vec![("registry-v1".to_string(), c)];
    let (result, findings) = run_enforce_gate(&contracts);

    assert!(result.passed);
    assert!(findings.is_empty(), "registry contracts should be skipped");
}

#[test]
fn test_enforce_gate_multiple_equations_mixed() {
    let mut c = minimal_contract();
    c.equations.insert(
        "eq_good".to_string(),
        make_equation(vec!["x > 0"], vec!["y > 0"], Some("Theorems.Good")),
    );
    c.equations
        .insert("eq_bad".to_string(), make_equation(vec![], vec![], None));

    let contracts = vec![("mixed-v1".to_string(), c)];
    let (result, findings) = run_enforce_gate(&contracts);

    assert!(result.passed);
    // eq_bad is missing preconditions (PV-ENF-001), postconditions (PV-ENF-001), and lean (PV-ENF-002)
    assert_eq!(findings.len(), 3);
}

// ---------------------------------------------------------------------------
// capitalize_first
// ---------------------------------------------------------------------------

#[test]
fn test_capitalize_first_normal() {
    assert_eq!(capitalize_first("softmax"), "Softmax");
}

#[test]
fn test_capitalize_first_empty() {
    assert_eq!(capitalize_first(""), "");
}

#[test]
fn test_capitalize_first_single_char() {
    assert_eq!(capitalize_first("x"), "X");
}

#[test]
fn test_capitalize_first_already_capitalized() {
    assert_eq!(capitalize_first("Foo"), "Foo");
}

// ---------------------------------------------------------------------------
// run_enforcement_level_gate
// ---------------------------------------------------------------------------

#[test]
fn test_enforcement_level_gate_basic_contract_meets_basic() {
    let mut c = minimal_contract(); // no tests, no kani = Basic
                                    // Explicitly declare Basic so declared == actual
    c.metadata.enforcement_level = Some(EnforcementLevel::Basic);
    let contracts = vec![("test-v1".to_string(), c)];
    let (result, findings) = run_enforcement_level_gate(&contracts, EnforcementLevel::Basic);

    assert!(result.passed);
    assert!(findings.is_empty());
}

#[test]
fn test_enforcement_level_gate_basic_below_standard() {
    let c = minimal_contract(); // no tests, no kani = Basic
    let contracts = vec![("test-v1".to_string(), c)];
    let (result, findings) = run_enforcement_level_gate(&contracts, EnforcementLevel::Standard);

    // Basic < Standard, so this should generate a finding
    let level_findings: Vec<_> = findings
        .iter()
        .filter(|f| f.message.contains("below required"))
        .collect();
    assert!(!level_findings.is_empty());
    assert_eq!(result.name, "enforcement-level");
}

#[test]
fn test_enforcement_level_gate_standard_contract() {
    let mut c = minimal_contract();
    c.falsification_tests = vec![make_falsification_test("F-01", "test_x")];
    c.kani_harnesses = vec![make_kani_harness("K-01")];

    let contracts = vec![("test-v1".to_string(), c)];
    let (result, findings) = run_enforcement_level_gate(&contracts, EnforcementLevel::Standard);

    // Has falsification + kani = Standard, meets Standard
    assert!(result.passed);
    assert!(findings.is_empty());
}

#[test]
fn test_enforcement_level_gate_declared_vs_actual_mismatch() {
    let mut c = minimal_contract();
    // Declare Strict but actual is Basic (no tests/kani)
    c.metadata.enforcement_level = Some(EnforcementLevel::Strict);

    let contracts = vec![("test-v1".to_string(), c)];
    let (result, findings) = run_enforcement_level_gate(&contracts, EnforcementLevel::Basic);

    // Actual (Basic) < Declared (Strict) — generates a finding
    let mismatch_findings: Vec<_> = findings
        .iter()
        .filter(|f| f.message.contains("declares enforcement_level"))
        .collect();
    assert_eq!(mismatch_findings.len(), 1);
    assert!(mismatch_findings[0].message.contains("Strict"));
    assert!(mismatch_findings[0].message.contains("Basic"));
    assert_eq!(result.name, "enforcement-level");
}

#[test]
fn test_enforcement_level_gate_skips_registries() {
    let mut c = minimal_contract();
    c.metadata.registry = true;

    let contracts = vec![("registry-v1".to_string(), c)];
    let (result, findings) = run_enforcement_level_gate(&contracts, EnforcementLevel::Proven);

    assert!(result.passed);
    assert!(findings.is_empty());
}

// ---------------------------------------------------------------------------
// compute_actual_level
// ---------------------------------------------------------------------------

#[test]
fn test_compute_actual_level_basic() {
    let c = minimal_contract();
    assert_eq!(compute_actual_level(&c), EnforcementLevel::Basic);
}

#[test]
fn test_compute_actual_level_standard() {
    let mut c = minimal_contract();
    c.falsification_tests = vec![make_falsification_test("F-01", "test_x")];
    c.kani_harnesses = vec![make_kani_harness("K-01")];
    assert_eq!(compute_actual_level(&c), EnforcementLevel::Standard);
}

#[test]
fn test_compute_actual_level_proven() {
    let mut c = minimal_contract();
    c.falsification_tests = vec![make_falsification_test("F-01", "test_x")];
    c.kani_harnesses = vec![make_kani_harness("K-01")];
    c.verification_summary = Some(VerificationSummary {
        total_obligations: 1,
        l2_property_tested: 1,
        l3_kani_proved: 1,
        l4_lean_proved: 1,
        l4_sorry_count: 0,
        l4_not_applicable: 0,
    });
    assert_eq!(compute_actual_level(&c), EnforcementLevel::Proven);
}

#[test]
fn test_compute_actual_level_with_sorry_not_proven() {
    let mut c = minimal_contract();
    c.falsification_tests = vec![make_falsification_test("F-01", "test_x")];
    c.kani_harnesses = vec![make_kani_harness("K-01")];
    c.verification_summary = Some(VerificationSummary {
        total_obligations: 1,
        l2_property_tested: 1,
        l3_kani_proved: 1,
        l4_lean_proved: 1,
        l4_sorry_count: 1, // has sorry!
        l4_not_applicable: 0,
    });
    // sorry_count > 0 means not fully proven
    assert_eq!(compute_actual_level(&c), EnforcementLevel::Standard);
}

#[test]
fn test_compute_actual_level_only_falsification() {
    let mut c = minimal_contract();
    c.falsification_tests = vec![make_falsification_test("F-01", "test_x")];
    // No kani harnesses => not Standard
    assert_eq!(compute_actual_level(&c), EnforcementLevel::Basic);
}

#[test]
fn test_compute_actual_level_only_kani() {
    let mut c = minimal_contract();
    c.kani_harnesses = vec![make_kani_harness("K-01")];
    // No falsification tests => not Standard
    assert_eq!(compute_actual_level(&c), EnforcementLevel::Basic);
}

// ---------------------------------------------------------------------------
// Level lock (integrated in run_enforcement_level_gate)
// ---------------------------------------------------------------------------

#[test]
fn test_level_lock_regression_detected() {
    let mut c = minimal_contract();
    // Lock at Standard but contract is actually Basic
    c.metadata.locked_level = Some("standard".to_string());

    let contracts = vec![("locked-v1".to_string(), c)];
    let (result, findings) = run_enforcement_level_gate(&contracts, EnforcementLevel::Basic);

    assert!(!result.passed, "should fail: level lock regression");
    let lock_findings: Vec<_> = findings
        .iter()
        .filter(|f| f.rule_id == "PV-LCK-001")
        .collect();
    assert_eq!(lock_findings.len(), 1);
    assert!(lock_findings[0].message.contains("locked"));
    assert!(lock_findings[0].message.contains("regressed"));
    assert_eq!(lock_findings[0].severity, RuleSeverity::Error);
}

#[test]
fn test_level_lock_no_regression() {
    let mut c = minimal_contract();
    c.falsification_tests = vec![make_falsification_test("F-01", "test_x")];
    c.kani_harnesses = vec![make_kani_harness("K-01")];
    // Lock at Standard, actual is Standard => no regression
    c.metadata.locked_level = Some("standard".to_string());

    let contracts = vec![("locked-v1".to_string(), c)];
    let (result, findings) = run_enforcement_level_gate(&contracts, EnforcementLevel::Basic);

    assert!(result.passed);
    let lock_findings: Vec<_> = findings
        .iter()
        .filter(|f| f.rule_id == "PV-LCK-001")
        .collect();
    assert!(lock_findings.is_empty());
}

#[test]
fn test_level_lock_aliases() {
    // Test the alias parsing: "l1" => Basic, "l4" => Strict, "l5" => Proven
    let mut c = minimal_contract();
    c.metadata.locked_level = Some("l1".to_string()); // Basic
                                                      // Basic contract locked at Basic => no regression
    let contracts = vec![("alias-v1".to_string(), c.clone())];
    let (result, _) = run_enforcement_level_gate(&contracts, EnforcementLevel::Basic);
    assert!(result.passed);

    // Lock at l5 (Proven) but actual is Basic => regression
    let mut c2 = minimal_contract();
    c2.metadata.locked_level = Some("l5".to_string());
    let contracts2 = vec![("alias-v1".to_string(), c2)];
    let (result2, findings2) = run_enforcement_level_gate(&contracts2, EnforcementLevel::Basic);
    assert!(!result2.passed);
    assert!(findings2.iter().any(|f| f.rule_id == "PV-LCK-001"));
}

// ---------------------------------------------------------------------------
// run_reverse_coverage_gate
// ---------------------------------------------------------------------------

#[test]
fn test_reverse_coverage_gate_high_coverage() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), "pub fn alpha() {}\npub fn beta() {}\n").unwrap();

    let binding = dir.path().join("binding.yaml");
    std::fs::write(
        &binding,
        "bindings:\n  - function: alpha\n  - function: beta\n",
    )
    .unwrap();

    let (result, findings) = run_reverse_coverage_gate(&binding, dir.path());
    assert!(result.passed);
    // No unbound findings
    let unbound: Vec<_> = findings
        .iter()
        .filter(|f| f.rule_id == "PV-RCV-001")
        .collect();
    assert!(unbound.is_empty());
}

#[test]
fn test_reverse_coverage_gate_below_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    // Many unbound functions that are not auto-exempt
    // Use long compound names that won't match prefix/suffix patterns
    let mut code = String::new();
    for i in 0..10 {
        code.push_str(&format!("pub fn zzzunbound_qqq_{i}() {{}}\n"));
    }
    std::fs::write(src.join("lib.rs"), &code).unwrap();

    let binding = dir.path().join("binding.yaml");
    std::fs::write(&binding, "bindings: []\n").unwrap();

    let (result, findings) = run_reverse_coverage_gate(&binding, dir.path());
    assert!(!result.passed, "should fail: coverage below 50%");
    // Should have PV-RCV-002 for threshold failure
    assert!(findings.iter().any(|f| f.rule_id == "PV-RCV-002"));
}

// ---------------------------------------------------------------------------
// check_stale_suppressions
// ---------------------------------------------------------------------------

#[test]
fn test_stale_suppressions_rule_still_active() {
    let findings = vec![LintFinding::new(
        "PV-VAL-001",
        RuleSeverity::Error,
        "msg",
        "a.yaml",
    )];
    let stale = check_stale_suppressions(&findings, &["PV-VAL-001".into()], &[]);
    assert!(stale.is_empty(), "active rule should not be stale");
}

#[test]
fn test_stale_suppressions_rule_no_longer_fires() {
    let findings = vec![LintFinding::new(
        "PV-VAL-001",
        RuleSeverity::Error,
        "msg",
        "a.yaml",
    )];
    let stale = check_stale_suppressions(&findings, &["PV-GONE-001".into()], &[]);
    assert_eq!(stale.len(), 1);
    assert_eq!(stale[0].rule_id, "PV-SUP-001");
    assert!(stale[0].message.contains("PV-GONE-001"));
    assert!(stale[0].message.contains("stale"));
}

#[test]
fn test_stale_suppressions_stem_still_has_findings() {
    let mut f = LintFinding::new("PV-VAL-001", RuleSeverity::Error, "msg", "a.yaml");
    f.contract_stem = Some("softmax-v1".into());
    let findings = vec![f];
    let stale = check_stale_suppressions(&findings, &[], &["softmax-v1".into()]);
    assert!(
        stale.is_empty(),
        "stem with active findings should not be stale"
    );
}

#[test]
fn test_stale_suppressions_stem_no_findings() {
    let findings: Vec<LintFinding> = vec![];
    let stale = check_stale_suppressions(&findings, &[], &["old-contract-v1".into()]);
    assert_eq!(stale.len(), 1);
    assert!(stale[0].message.contains("old-contract-v1"));
}

#[test]
fn test_stale_suppressions_empty_everything() {
    let stale = check_stale_suppressions(&[], &[], &[]);
    assert!(stale.is_empty());
}

#[test]
fn test_stale_suppressions_multiple_stale_rules() {
    let findings: Vec<LintFinding> = vec![];
    let stale = check_stale_suppressions(&findings, &["PV-A-001".into(), "PV-B-001".into()], &[]);
    assert_eq!(stale.len(), 2);
}
