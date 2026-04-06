use super::*;

#[test]
fn probes_populated_for_complete_contract() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    assert!(!score.probes.is_empty(), "probes should be populated");

    // Should have spec_depth probes
    let spec_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "spec_depth")
        .collect();
    assert!(
        spec_probes.len() >= 5,
        "spec_depth should have at least 5 probes, got {}",
        spec_probes.len()
    );

    // has_equations should pass
    let eq_probe = spec_probes
        .iter()
        .find(|p| p.probe == "has_equations")
        .expect("should have has_equations probe");
    assert!(eq_probe.outcome, "has_equations should pass");
    assert!(eq_probe.detail.contains("1 equation"), "detail: {}", eq_probe.detail);
}

#[test]
fn probes_kani_shows_harness_id() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let score = score_contract(&contract, None, "test-v1");

    let kani_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "kani")
        .collect();
    assert!(!kani_probes.is_empty(), "should have kani probes");

    // The obligation "finite" should match harness KANI-001
    let finite_probe = kani_probes
        .iter()
        .find(|p| p.probe == "finite")
        .expect("should have probe for 'finite' obligation");
    assert!(finite_probe.outcome, "finite should have a harness");
    assert!(
        finite_probe.detail.contains("KANI-001"),
        "detail should mention KANI-001, got: {}",
        finite_probe.detail
    );
}

#[test]
fn probes_falsification_shows_test_id() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let score = score_contract(&contract, None, "test-v1");

    let fals_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "falsification")
        .collect();
    assert!(!fals_probes.is_empty(), "should have falsification probes");

    // The obligation "finite" should match test FALSIFY-001 (rule="finite")
    let finite_probe = fals_probes
        .iter()
        .find(|p| p.probe == "finite")
        .expect("should have probe for 'finite' obligation");
    assert!(finite_probe.outcome, "finite should have a test");
    assert!(
        finite_probe.detail.contains("FALSIFY-001"),
        "detail should mention FALSIFY-001, got: {}",
        finite_probe.detail
    );
}

#[test]
fn probes_lean_no_obligations() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let score = score_contract(&contract, None, "test-v1");

    let lean_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "lean")
        .collect();
    // The minimal YAML has one obligation but no lean metadata
    assert!(!lean_probes.is_empty(), "should have lean probes");
    let probe = &lean_probes[0];
    assert!(!probe.outcome, "should fail (no lean_theorem)");
    assert!(
        probe.detail.contains("no lean_theorem"),
        "detail: {}",
        probe.detail
    );
}

#[test]
fn probes_binding_no_registry() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let score = score_contract(&contract, None, "test-v1");

    let bind_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "binding")
        .collect();
    assert!(!bind_probes.is_empty(), "should have binding probes");
    let probe = &bind_probes[0];
    assert!(!probe.outcome, "should fail (no binding registry)");
    assert!(
        probe.detail.contains("no binding registry"),
        "detail: {}",
        probe.detail
    );
}

#[test]
fn probes_binding_with_partial_status() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
  g:
    formula: "g(x) = x^2"
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let binding = BindingRegistry {
        version: "1.0.0".into(),
        target_crate: "test".into(),
        critical_path: vec![],
            bindings: vec![
            crate::binding::KernelBinding {
                contract: "test-v1".into(),
                equation: "f".into(),
                module_path: Some("test::f".into()),
                function: Some("f".into()),
                signature: None,
                status: ImplStatus::Implemented,
                notes: None,
            },
            crate::binding::KernelBinding {
                contract: "test-v1".into(),
                equation: "g".into(),
                module_path: Some("test::g".into()),
                function: Some("g".into()),
                signature: None,
                status: ImplStatus::Partial,
                notes: None,
            },
        ],
    };
    let score = score_contract(&contract, Some(&binding), "test-v1");

    let bind_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "binding")
        .collect();
    assert_eq!(bind_probes.len(), 2, "should have 2 binding probes");

    let f_probe = bind_probes
        .iter()
        .find(|p| p.probe == "f")
        .expect("should have probe for equation 'f'");
    assert!(f_probe.outcome, "f should be implemented");
    assert!(f_probe.detail.contains("implemented"), "detail: {}", f_probe.detail);

    let g_probe = bind_probes
        .iter()
        .find(|p| p.probe == "g")
        .expect("should have probe for equation 'g'");
    assert!(!g_probe.outcome, "g is partial, not fully implemented");
    assert!(g_probe.detail.contains("partial"), "detail: {}", g_probe.detail);
}

#[test]
fn probes_serialization_skip_empty() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    // When probes are present, they should serialize
    let json = serde_json::to_string(&score).unwrap();
    assert!(json.contains("probes"), "JSON should include probes when non-empty");

    // Manually construct a ContractScore with empty probes to verify skip_serializing_if
    let empty_score = ContractScore {
        stem: "empty".into(),
        spec_depth: 0.0,
        falsification_coverage: 0.0,
        kani_coverage: 0.0,
        lean_coverage: 0.0,
        binding_coverage: 0.0,
        composite: 0.0,
        grade: Grade::F,
        probes: vec![],
    };
    let json = serde_json::to_string(&empty_score).unwrap();
    assert!(!json.contains("probes"), "JSON should omit probes when empty");
}

#[test]
fn probes_spec_depth_per_equation() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
    domain: "R"
    invariants: ["finite"]
  g:
    formula: "g(x) = x^2"
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "test-v1");

    let spec_probes: Vec<_> = score
        .probes
        .iter()
        .filter(|p| p.dimension == "spec_depth")
        .collect();

    // f should have domain=true and invariants=true
    let f_domain = spec_probes
        .iter()
        .find(|p| p.probe == "f: domain")
        .expect("should have f: domain probe");
    assert!(f_domain.outcome, "f has domain");

    let f_inv = spec_probes
        .iter()
        .find(|p| p.probe == "f: invariants")
        .expect("should have f: invariants probe");
    assert!(f_inv.outcome, "f has invariants");

    // g should have domain=false and invariants=false
    let g_domain = spec_probes
        .iter()
        .find(|p| p.probe == "g: domain")
        .expect("should have g: domain probe");
    assert!(!g_domain.outcome, "g has no domain");

    let g_inv = spec_probes
        .iter()
        .find(|p| p.probe == "g: invariants")
        .expect("should have g: invariants probe");
    assert!(!g_inv.outcome, "g has no invariants");
}
