//! PMAT-3091: an obligation that is NOT a property of code is declared
//! `applies_to: not_applicable` and must say why (`na_reason`) and where the
//! claim IS verified (`na_owner`). A justification without the declaration is
//! decoration.

use super::*;
use crate::schema::parse_contract_str;

fn contract_with_obligation(extra: &str) -> Contract {
    let yaml = format!(
        r#"
metadata:
  version: "1.0.0"
  description: "N/A fixture"
  references: ["Paper (2024)"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "a checkpoint fact"
{extra}
falsification_tests:
  - id: FALSIFY-001
    rule: "finiteness"
    prediction: "output is always finite"
    if_fails: "overflow in computation"
kani_harnesses:
  - id: KANI-001
    obligation: "a checkpoint fact"
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
"#
    );
    parse_contract_str(&yaml).expect("fixture must parse")
}

fn na_rules(contract: &Contract) -> Vec<String> {
    validate_contract(contract)
        .into_iter()
        .filter(|v| v.severity == Severity::Error)
        .filter(|v| matches!(v.rule.as_str(), "SCHEMA-021" | "SCHEMA-022" | "SCHEMA-023"))
        .map(|v| v.rule)
        .collect()
}

#[test]
fn na_valid_obligation_with_reason_and_owner_has_no_na_errors() {
    let c = contract_with_obligation(
        "    applies_to: not_applicable\n    na_reason: \"a checkpoint fact\"\n    na_owner: \"apr inspect evidence\"",
    );
    assert!(na_rules(&c).is_empty(), "{:?}", na_rules(&c));
    let errors: Vec<_> = validate_contract(&c)
        .into_iter()
        .filter(|v| v.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "a valid N/A obligation must have 0 errors: {errors:?}");
}

#[test]
fn na_missing_reason_is_schema_021() {
    let c = contract_with_obligation("    applies_to: not_applicable\n    na_owner: \"bench\"");
    assert_eq!(na_rules(&c), vec!["SCHEMA-021".to_string()]);
}

#[test]
fn na_empty_reason_is_schema_021() {
    let c = contract_with_obligation(
        "    applies_to: not_applicable\n    na_reason: \"  \"\n    na_owner: \"bench\"",
    );
    assert_eq!(na_rules(&c), vec!["SCHEMA-021".to_string()]);
}

#[test]
fn na_missing_owner_is_schema_022() {
    let c = contract_with_obligation("    applies_to: not_applicable\n    na_reason: \"why\"");
    assert_eq!(na_rules(&c), vec!["SCHEMA-022".to_string()]);
}

#[test]
fn na_empty_owner_is_schema_022() {
    let c = contract_with_obligation(
        "    applies_to: not_applicable\n    na_reason: \"why\"\n    na_owner: \"\"",
    );
    assert_eq!(na_rules(&c), vec!["SCHEMA-022".to_string()]);
}

#[test]
fn na_reason_without_not_applicable_is_schema_023() {
    let c = contract_with_obligation("    applies_to: all\n    na_reason: \"why\"");
    assert_eq!(na_rules(&c), vec!["SCHEMA-023".to_string()]);
}

#[test]
fn na_owner_without_applies_to_is_schema_023() {
    let c = contract_with_obligation("    na_owner: \"bench\"");
    assert_eq!(na_rules(&c), vec!["SCHEMA-023".to_string()]);
}

#[test]
fn na_plain_obligation_has_no_na_errors() {
    let c = contract_with_obligation("    applies_to: all");
    assert!(na_rules(&c).is_empty());
}
