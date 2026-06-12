    use super::*;
    use crate::schema::parse_contract_str;

    #[test]
    fn valid_contract_has_no_errors() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Valid"
  references:
    - "Paper (2024)"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output is finite"
falsification_tests:
  - id: FALSIFY-001
    rule: "finiteness"
    prediction: "output is always finite"
    if_fails: "overflow in computation"
kani_harnesses:
  - id: KANI-001
    obligation: "output is finite"
    bound: 8
    strategy: stub_float
    solver: cadical
    harness: verify_finiteness
qa_gate:
  id: F-001
  name: "Test Gate"
  checks:
    - "finiteness"
  pass_criteria: "All tests pass"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty(), "Expected no errors, got: {errors:?}");
    }

    #[test]
    fn missing_references_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No refs"
  references: []
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-001"));
    }

    #[test]
    fn empty_formula_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty formula"
  references:
    - "Paper"
equations:
  bad:
    formula: ""
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-004"));
    }

    #[test]
    fn duplicate_falsification_id_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Dup IDs"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "works"
    if_fails: "broken"
  - id: FALSIFY-001
    rule: "test2"
    prediction: "works2"
    if_fails: "broken2"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-007"));
    }

    #[test]
    fn kani_harness_without_bound_is_warning() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No bound"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-012"));
    }

    #[test]
    fn no_equations_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No equations"
  references:
    - "Paper"
equations: {}
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-003"));
    }

    #[test]
    fn empty_version_is_error() {
        let yaml = r#"
metadata:
  version: ""
  description: "Empty version"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-002"));
    }

    #[test]
    fn empty_property_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty prop"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: ""
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-005"));
    }

    #[test]
    fn duplicate_formal_is_warning() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Dup formal"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "prop1"
    formal: "same_formal"
  - type: bound
    property: "prop2"
    formal: "same_formal"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-006"));
    }

    #[test]
    fn empty_prediction_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty pred"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: ""
    if_fails: "broken"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-008"));
    }

    #[test]
    fn empty_if_fails_is_warning() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty if_fails"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "works"
    if_fails: ""
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-009"));
    }

    #[test]
    fn duplicate_kani_id_is_error() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Dup kani"
  references:
    - "Paper"
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 8
  - id: KANI-001
    obligation: OBL-002
    bound: 16
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "SCHEMA-010"));
    }

    #[path = "validator_tests_extra.rs"]
    mod extra;

    // ── PMAT-741 BeatBenchmark validator (BEAT-001..007) ──────────────────────

    /// Wrap a `beat:` block body in a valid beat-benchmark metadata envelope so
    /// each test isolates a single broken `beat.*` field.
    fn beat_contract(beat_block: &str) -> String {
        format!(
            r#"
metadata:
  kind: beat-benchmark
  version: "1.0.0"
  description: "BEAT validator test"
  references:
    - "docs/specifications/campaign-ev-reprioritization-2026-06-12.md"
beat:
{beat_block}"#
        )
    }

    /// A well-formed `beat:` block — each negative test mutates exactly one line.
    const VALID_BEAT_BLOCK: &str = r#"  incumbent: scikit-learn
  metric: accuracy
  direction: higher_is_better
  beat_threshold: 0.92
  ci_gate_name: beat_sklearn_iris
  approved_compute: CPU
"#;

    #[test]
    fn beat_001_missing_beat_block_errors() {
        // kind: beat-benchmark but no `beat:` block at all.
        let yaml = r#"
metadata:
  kind: beat-benchmark
  version: "1.0.0"
  description: "no beat block"
  references:
    - "ref"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "BEAT-001"), "{violations:?}");
    }

    #[test]
    fn beat_002_incumbent_not_a_pillar_errors() {
        let block = VALID_BEAT_BLOCK.replace("scikit-learn", "tensorflow");
        let contract = parse_contract_str(&beat_contract(&block)).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "BEAT-002"), "{violations:?}");
    }

    #[test]
    fn beat_003_missing_metric_errors() {
        let block = VALID_BEAT_BLOCK.replace("  metric: accuracy\n", "");
        let contract = parse_contract_str(&beat_contract(&block)).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "BEAT-003"), "{violations:?}");
    }

    #[test]
    fn beat_004_bad_direction_errors() {
        let block = VALID_BEAT_BLOCK.replace("higher_is_better", "bigger_number");
        let contract = parse_contract_str(&beat_contract(&block)).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "BEAT-004"), "{violations:?}");
    }

    #[test]
    fn beat_005_missing_threshold_errors() {
        let block = VALID_BEAT_BLOCK.replace("  beat_threshold: 0.92\n", "");
        let contract = parse_contract_str(&beat_contract(&block)).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "BEAT-005"), "{violations:?}");
    }

    #[test]
    fn beat_006_missing_ci_gate_errors() {
        let block = VALID_BEAT_BLOCK.replace("  ci_gate_name: beat_sklearn_iris\n", "");
        let contract = parse_contract_str(&beat_contract(&block)).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "BEAT-006"), "{violations:?}");
    }

    #[test]
    fn beat_007_bad_compute_errors() {
        let block = VALID_BEAT_BLOCK.replace("approved_compute: CPU", "approved_compute: TPU");
        let contract = parse_contract_str(&beat_contract(&block)).unwrap();
        let violations = validate_contract(&contract);
        assert!(violations.iter().any(|v| v.rule == "BEAT-007"), "{violations:?}");
    }

    #[test]
    fn beat_valid_block_has_no_beat_errors() {
        let contract = parse_contract_str(&beat_contract(VALID_BEAT_BLOCK)).unwrap();
        let violations = validate_contract(&contract);
        assert!(
            !violations.iter().any(|v| v.rule.starts_with("BEAT-")),
            "a valid beat block must raise no BEAT-* violations: {violations:?}"
        );
    }
