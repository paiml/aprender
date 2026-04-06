    use super::*;
    use crate::schema::parse_contract_str;

    fn minimal_contract(version: &str) -> Contract {
        parse_contract_str(&format!(
            r#"
metadata:
  version: "{version}"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#
        ))
        .unwrap()
    }

    #[test]
    fn identical_contracts() {
        let c = minimal_contract("1.0.0");
        let diff = diff_contracts(&c, &c);
        assert!(is_identical(&diff));
        assert_eq!(diff.suggested_bump, SemverBump::None);
    }

    #[test]
    fn added_equation() {
        let old = minimal_contract("1.0.0");
        let new = parse_contract_str(
            r#"
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
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let eq_section = &diff.sections[0];
        assert_eq!(eq_section.added, vec!["g"]);
        assert!(eq_section.removed.is_empty());
        assert_eq!(diff.suggested_bump, SemverBump::Minor);
    }

    #[test]
    fn removed_equation() {
        let old = parse_contract_str(
            r#"
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
"#,
        )
        .unwrap();
        let new = minimal_contract("1.0.0");
        let diff = diff_contracts(&old, &new);
        let eq_section = &diff.sections[0];
        assert_eq!(eq_section.removed, vec!["g"]);
        assert_eq!(diff.suggested_bump, SemverBump::Major);
    }

    #[test]
    fn changed_formula() {
        let old = minimal_contract("1.0.0");
        let new = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x + 1"
falsification_tests: []
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let eq_section = &diff.sections[0];
        assert!(!eq_section.changed.is_empty());
        assert_eq!(diff.suggested_bump, SemverBump::Major);
    }

    #[test]
    fn added_obligation() {
        let old = minimal_contract("1.0.0");
        let new = parse_contract_str(
            r#"
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
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let ob_section = &diff.sections[1];
        assert_eq!(ob_section.added.len(), 1);
        assert_eq!(diff.suggested_bump, SemverBump::Minor);
    }

    #[test]
    fn removed_obligation_is_major() {
        let old = parse_contract_str(
            r#"
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
"#,
        )
        .unwrap();
        let new = minimal_contract("1.0.0");
        let diff = diff_contracts(&old, &new);
        let ob_section = &diff.sections[1];
        assert_eq!(ob_section.removed.len(), 1);
        assert_eq!(diff.suggested_bump, SemverBump::Major);
    }

    #[test]
    fn version_change_is_patch() {
        let old = minimal_contract("1.0.0");
        let new = minimal_contract("1.0.1");
        let diff = diff_contracts(&old, &new);
        assert_eq!(diff.old_version, "1.0.0");
        assert_eq!(diff.new_version, "1.0.1");
        assert_eq!(diff.suggested_bump, SemverBump::Patch);
    }

    #[test]
    fn added_falsification_test() {
        let old = minimal_contract("1.0.0");
        let new = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "works"
    if_fails: "broken"
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let ft_section = &diff.sections[2];
        assert_eq!(ft_section.added, vec!["FALSIFY-001"]);
        assert_eq!(diff.suggested_bump, SemverBump::Minor);
    }

    #[test]
    fn semver_bump_display() {
        assert_eq!(SemverBump::None.to_string(), "none");
        assert_eq!(SemverBump::Patch.to_string(), "patch");
        assert_eq!(SemverBump::Minor.to_string(), "minor");
        assert_eq!(SemverBump::Major.to_string(), "major");
    }

    #[test]
    fn section_diff_is_empty() {
        let s = SectionDiff {
            section: "test".to_string(),
            added: vec![],
            removed: vec![],
            changed: vec![],
        };
        assert!(s.is_empty());

        let s2 = SectionDiff {
            section: "test".to_string(),
            added: vec!["x".to_string()],
            removed: vec![],
            changed: vec![],
        };
        assert!(!s2.is_empty());
    }

    #[test]
    fn added_kani_harness() {
        let old = minimal_contract("1.0.0");
        let new = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 16
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let kh_section = &diff.sections[3];
        assert_eq!(kh_section.added, vec!["KANI-001"]);
        assert_eq!(diff.suggested_bump, SemverBump::Minor);
    }

    #[test]
    fn enforcement_added() {
        let old = minimal_contract("1.0.0");
        let new = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
enforcement:
  rule1:
    description: "must hold"
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let enf_section = &diff.sections[4];
        assert_eq!(enf_section.added, vec!["rule1"]);
        assert_eq!(diff.suggested_bump, SemverBump::Patch);
    }

    #[test]
    fn domain_change_detected() {
        let old = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
    domain: "R"
falsification_tests: []
"#,
        )
        .unwrap();
        let new = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
    domain: "R^n"
falsification_tests: []
"#,
        )
        .unwrap();
        let diff = diff_contracts(&old, &new);
        let eq_section = &diff.sections[0];
        assert!(eq_section.changed.iter().any(|c| c.contains("domain")));
    }
