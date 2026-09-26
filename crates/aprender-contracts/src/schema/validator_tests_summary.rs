//! SCHEMA-024 (aprender#2648): `verification_summary.total_obligations` is a
//! copy of `|proof_obligations|`, and a copy that disagrees is an error.
//!
//! Paired controls, as in `validator_tests_top_level.rs`: each case asserts
//! both directions on the same document, so the rule is seen firing AND seen
//! quiet — a rule only ever seen green proves nothing.

use super::*;
use crate::schema::parse_contract_str;

/// A contract of `kind` listing `n` obligations whose summary states `stated`.
fn summary(kind: &str, n: usize, stated: u32) -> String {
    let obligations: String = (0..n)
        .map(|i| format!("  - type: invariant\n    property: \"p{i}\"\n    formal: \"x{i} = x{i}\"\n"))
        .collect();
    let list = if n == 0 {
        "proof_obligations: []\n".to_string()
    } else {
        format!("proof_obligations:\n{obligations}")
    };
    format!(
        r#"metadata:
  version: "1.0.0"
  kind: {kind}
  description: "summary-count fixture"
  references: ["Paper (2024)"]
{list}verification_summary:
  total_obligations: {stated}
"#
    )
}

fn fires(yaml: &str) -> bool {
    let contract = parse_contract_str(yaml).expect("fixture must parse");
    validate_contract(&contract)
        .iter()
        .any(|v| v.rule == "SCHEMA-024" && v.severity == Severity::Error)
}

#[test]
fn schema_024_stated_count_must_equal_the_list() {
    for kind in ["registry", "schema", "model-family"] {
        assert!(!fires(&summary(kind, 3, 3)), "{kind}: 3 of 3 is not drift");
        assert!(fires(&summary(kind, 3, 4)), "{kind}: overstated");
        assert!(fires(&summary(kind, 3, 2)), "{kind}: understated");
    }
}

#[test]
fn schema_024_a_count_of_nothing_is_drift_outside_work_contracts() {
    // A non-schema file with no obligations that claims some has drifted
    // (ci-infra-v1 claimed 6 of 0).
    assert!(fires(&summary("registry", 0, 6)));
    assert!(!fires(&summary("registry", 0, 0)));
    // The generated pmat work-contract shape counts a list it does not carry.
    assert!(!fires(&summary("schema", 0, 47)));
    // …but a schema file that DOES list obligations is held to them.
    assert!(fires(&summary("schema", 1, 47)));
}

#[test]
fn schema_024_no_summary_no_finding() {
    let yaml = summary("registry", 2, 9);
    let cut = yaml.find("verification_summary").expect("fixture has one");
    assert!(!fires(&yaml[..cut]));
    assert!(fires(&yaml));
}

#[test]
fn schema_024_names_both_numbers() {
    let contract = parse_contract_str(&summary("registry", 2, 5)).expect("parses");
    let v = validate_contract(&contract)
        .into_iter()
        .find(|v| v.rule == "SCHEMA-024")
        .expect("fires");
    assert!(v.message.contains("is 5") && v.message.contains("lists 2"), "{}", v.message);
    assert_eq!(v.location.as_deref(), Some("verification_summary.total_obligations"));
}
