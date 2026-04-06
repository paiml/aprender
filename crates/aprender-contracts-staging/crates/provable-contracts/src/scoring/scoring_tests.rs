// Coverage-targeted tests for contract scoring.

use super::*;
use crate::schema::parse_contract_str;

fn minimal_kernel_yaml() -> &'static str {
    r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
  depends_on: ["other-v1"]
equations:
  f:
    formula: "f(x) = x"
    domain: "R"
    codomain: "R"
    invariants: ["output finite"]
proof_obligations:
  - type: invariant
    property: "finite"
    tolerance: 1e-6
falsification_tests:
  - id: FALSIFY-001
    rule: "finite"
    prediction: "finite"
    if_fails: "overflow"
kani_harnesses:
  - id: KANI-001
    obligation: "finite"
    bound: 8
    strategy: stub_float
    solver: cadical
"#
}

#[test]
fn score_complete_contract() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    assert!(score.spec_depth >= 0.70, "spec_depth={}", score.spec_depth);
    assert!((score.falsification_coverage - 1.0).abs() < f64::EPSILON);
    assert!(score.kani_coverage > 0.0);
    assert!(score.lean_coverage.abs() < f64::EPSILON);
    assert!(score.binding_coverage.abs() < f64::EPSILON);
    assert!(score.composite > 0.0);
}

#[test]
fn grade_thresholds() {
    assert_eq!(Grade::from_score(0.95), Grade::A);
    assert_eq!(Grade::from_score(0.80), Grade::B);
    assert_eq!(Grade::from_score(0.65), Grade::C);
    assert_eq!(Grade::from_score(0.45), Grade::D);
    assert_eq!(Grade::from_score(0.30), Grade::F);
}

#[test]
fn custom_weights_change_composite() {
    let contract = parse_contract_str(minimal_kernel_yaml()).unwrap();
    let default = score_contract(&contract, None, "test-v1");
    let kani_heavy = score_contract_weighted(
        &contract,
        None,
        "test-v1",
        &ScoringWeights {
            spec_depth: 0.05,
            falsification: 0.05,
            kani: 0.70,
            lean: 0.10,
            binding: 0.10,
        },
    );
    // Different weights should produce different composites
    assert!(
        (default.composite - kani_heavy.composite).abs() > 0.01,
        "default={} kani_heavy={}",
        default.composite,
        kani_heavy.composite
    );
    // Individual dimensions should be the same
    assert!((default.spec_depth - kani_heavy.spec_depth).abs() < f64::EPSILON);
    assert!((default.kani_coverage - kani_heavy.kani_coverage).abs() < f64::EPSILON);
}

#[test]
fn weights_normalization() {
    let w = ScoringWeights {
        spec_depth: 2.0,
        falsification: 2.0,
        kani: 2.0,
        lean: 2.0,
        binding: 2.0,
    };
    let n = w.normalized();
    let total = n.spec_depth + n.falsification + n.kani + n.lean + n.binding;
    assert!((total - 1.0).abs() < 1e-9, "total={total}");
    assert!((n.spec_depth - 0.2).abs() < 1e-9);
}

#[test]
fn empty_contract_scores_low() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty"
  registry: true
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "empty-v1");
    assert!(score.composite < 0.40, "composite={}", score.composite);
    assert_eq!(score.grade, Grade::F);
}

#[test]
fn score_real_softmax_contract() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/softmax-kernel-v1.yaml");
    let contract = crate::schema::parse_contract(&path).unwrap();
    let score = score_contract(&contract, None, "softmax-kernel-v1");
    assert!(
        score.composite > 0.40,
        "softmax should score well: {}",
        score.composite
    );
    assert!(score.spec_depth > 0.50);
    assert!(score.falsification_coverage > 0.0);
    assert!(score.kani_coverage > 0.0);
}

#[test]
fn grade_display_all_variants() {
    assert_eq!(format!("{}", Grade::A), "A");
    assert_eq!(format!("{}", Grade::B), "B");
    assert_eq!(format!("{}", Grade::C), "C");
    assert_eq!(format!("{}", Grade::D), "D");
    assert_eq!(format!("{}", Grade::F), "F");
}

#[test]
fn contract_score_display() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/softmax-kernel-v1.yaml");
    let contract = crate::schema::parse_contract(&path).unwrap();
    let score = score_contract(&contract, None, "softmax-kernel-v1");
    let text = format!("{score}");
    assert!(
        text.contains("softmax-kernel-v1"),
        "Display should include stem"
    );
    assert!(text.contains("Grade"), "Display should include Grade");
    assert!(
        text.contains("Spec:"),
        "Display should include dimension breakdown"
    );
}

#[test]
fn kani_compositional_strategy_weight() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test compositional"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "finite"
kani_harnesses:
  - id: KANI-001
    obligation: "finite"
    bound: 8
    strategy: compositional
    solver: cadical
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    assert!(
        (score.kani_coverage - 0.7).abs() < 1e-9,
        "got {}",
        score.kani_coverage
    );
}

#[test]
fn kani_no_strategy_weight() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test no strategy"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "finite"
kani_harnesses:
  - id: KANI-001
    obligation: "finite"
    bound: 8
    solver: cadical
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    assert!(
        (score.kani_coverage - 0.5).abs() < 1e-9,
        "got {}",
        score.kani_coverage
    );
}

#[test]
fn binding_partial_status_coverage() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let binding = BindingRegistry {
        version: "1.0.0".into(),
        target_crate: "test".into(),
        critical_path: vec![],
        bindings: vec![crate::binding::KernelBinding {
            contract: "test-v1".into(),
            equation: "f".into(),
            module_path: Some("test::f".into()),
            function: Some("f".into()),
            signature: None,
            status: ImplStatus::Partial,
            notes: None,
        }],
    };
    let score = score_contract(&contract, Some(&binding), "test-v1");
    assert!(
        (score.binding_coverage - 0.5).abs() < 1e-9,
        "Partial = 0.5, got {}",
        score.binding_coverage
    );
}

#[test]
fn binding_no_relevant_entries() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let binding = BindingRegistry {
        version: "1.0.0".into(),
        target_crate: "test".into(),
        critical_path: vec![],
        bindings: vec![crate::binding::KernelBinding {
            contract: "other-v1".into(),
            equation: "g".into(),
            module_path: None,
            function: None,
            signature: None,
            status: ImplStatus::Implemented,
            notes: None,
        }],
    };
    let score = score_contract(&contract, Some(&binding), "test-v1");
    assert!(score.binding_coverage.abs() < f64::EPSILON);
}

#[test]
fn kani_coverage_with_no_obligations() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-001
    obligation: "finite"
    bound: 8
    strategy: exhaustive
    solver: cadical
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    assert!((score.kani_coverage - 1.0).abs() < f64::EPSILON);
}

#[test]
fn kani_strategy_weights_all_variants() {
    // Exercises all KaniStrategy match arms: Exhaustive, BoundedInt, StubFloat, Compositional, None
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "p1"
  - type: invariant
    property: "p2"
  - type: invariant
    property: "p3"
  - type: invariant
    property: "p4"
  - type: invariant
    property: "p5"
kani_harnesses:
  - id: K1
    obligation: "p1"
    bound: 8
    strategy: exhaustive
  - id: K2
    obligation: "p2"
    bound: 8
    strategy: bounded_int
  - id: K3
    obligation: "p3"
    bound: 8
    strategy: stub_float
  - id: K4
    obligation: "p4"
    bound: 8
    strategy: compositional
  - id: K5
    obligation: "p5"
    bound: 8
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    // Weighted: (1.0 + 0.9 + 0.8 + 0.7 + 0.5) / 5 = 0.78
    assert!(score.kani_coverage > 0.7);
    assert!(score.kani_coverage < 0.85);
}

#[test]
fn registry_scoring_full_binding_credit() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test registry"
  registry: true
equations:
  lookup:
    formula: "lookup(key) = value"
proof_obligations:
  - type: invariant
    property: "key exists"
falsification_tests:
  - id: F1
    rule: "key exists"
    prediction: "all keys resolve"
    test: "proptest"
    if_fails: "missing key"
kani_harnesses:
  - id: K1
    obligation: "key exists"
    bound: 8
    strategy: exhaustive
"#;
    let contract = parse_contract_str(yaml).unwrap();
    let score = score_contract(&contract, None, "test-v1");
    // Registry gets 1.0 binding credit even without bindings
    assert!((score.binding_coverage - 1.0).abs() < f64::EPSILON);
    // Lean gets at least 0.5 for registries
    assert!(score.lean_coverage >= 0.5);
}

#[path = "scoring_probe_tests.rs"]
mod probe_tests;
