    use super::*;
    use crate::schema::parse_contract_str;

    #[test]
    fn audit_minimal_contract() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Minimal"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let report = audit_contract(&contract);
        assert_eq!(report.equations, 1);
        assert_eq!(report.falsification_tests, 0);
        // Should warn about no falsification tests
        assert!(report.violations.iter().any(|v| v.rule == "AUDIT-001"));
    }

    #[test]
    fn audit_complete_contract() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Complete"
  references: ["Paper"]
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
    if_fails: "overflow"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 16
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let report = audit_contract(&contract);
        assert_eq!(report.equations, 1);
        assert_eq!(report.obligations, 1);
        assert_eq!(report.falsification_tests, 1);
        assert_eq!(report.kani_harnesses, 1);
        // No errors expected
        let errors: Vec<_> = report
            .violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty(), "Expected no errors, got: {errors:?}");
    }

    #[test]
    fn binding_audit_all_implemented() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output finite"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let binding = crate::binding::parse_binding_str(
            r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#,
        )
        .unwrap();

        let report = audit_binding(&[("test.yaml", &contract)], &binding);
        assert_eq!(report.total_equations, 1);
        assert_eq!(report.bound_equations, 1);
        assert_eq!(report.implemented, 1);
        assert!(
            report
                .violations
                .iter()
                .all(|v| v.severity != Severity::Error)
        );
    }

    #[test]
    fn binding_audit_missing_equation() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
  g:
    formula: "g(x) = x^2"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let binding = crate::binding::parse_binding_str(
            r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#,
        )
        .unwrap();

        let report = audit_binding(&[("test.yaml", &contract)], &binding);
        assert_eq!(report.total_equations, 2);
        assert_eq!(report.bound_equations, 1);
        assert!(report.violations.iter().any(|v| v.rule == "BIND-001"));
    }

    #[test]
    fn binding_audit_not_implemented() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let binding = crate::binding::parse_binding_str(
            r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    status: not_implemented
"#,
        )
        .unwrap();

        let report = audit_binding(&[("test.yaml", &contract)], &binding);
        assert_eq!(report.not_implemented, 1);
        assert!(report.violations.iter().any(|v| v.rule == "BIND-003"));
    }

    #[test]
    fn audit_warns_on_no_obligations() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No obligations"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "works"
    if_fails: "broken"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let report = audit_contract(&contract);
        assert!(report.violations.iter().any(|v| v.rule == "AUDIT-003"));
    }

    #[test]
    fn audit_empty_harness_obligation() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty harness obl"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output finite"
kani_harnesses:
  - id: KANI-001
    obligation: ""
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "works"
    if_fails: "broken"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let report = audit_contract(&contract);
        assert!(report.violations.iter().any(|v| v.rule == "AUDIT-002"));
    }

    #[test]
    fn binding_audit_partial_status() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Partial"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output finite"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let binding = crate::binding::parse_binding_str(
            r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: partial
"#,
        )
        .unwrap();

        let report = audit_binding(&[("test.yaml", &contract)], &binding);
        assert_eq!(report.partial, 1);
        assert!(report.violations.iter().any(|v| v.rule == "BIND-002"));
    }

    #[test]
    fn binding_audit_multiple_contracts() {
        let yaml1 = r#"
metadata:
  version: "1.0.0"
  description: "Contract 1"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let yaml2 = r#"
metadata:
  version: "1.0.0"
  description: "Contract 2"
  references: ["Paper"]
equations:
  g:
    formula: "g(x) = x^2"
falsification_tests: []
"#;
        let c1 = parse_contract_str(yaml1).unwrap();
        let c2 = parse_contract_str(yaml2).unwrap();
        let binding = crate::binding::parse_binding_str(
            r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: a.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#,
        )
        .unwrap();

        let report = audit_binding(&[("a.yaml", &c1), ("b.yaml", &c2)], &binding);
        assert_eq!(report.total_equations, 2);
        assert_eq!(report.bound_equations, 1);
        // g in b.yaml has no binding
        assert!(report.violations.iter().any(|v| v.rule == "BIND-001"));
    }
