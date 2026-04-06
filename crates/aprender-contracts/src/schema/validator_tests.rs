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
