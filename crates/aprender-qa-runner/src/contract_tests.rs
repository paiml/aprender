use super::*;

use crate::evidence::Outcome;

/// Verify InvariantDef deserializes bool `implemented` field from YAML bool value
#[test]
fn test_deserialize_invariant_bool_value() {
    let yaml = r#"
        id: "I-1"
        name: "Test"
        description: "Test invariant"
        catches: []
        gate_id: "F-CONTRACT-I1-001"
        implemented: true
    "#;
    let inv: InvariantDef = serde_yaml::from_str(yaml).expect("should parse bool");
    assert!(inv.implemented);
}

/// Verify InvariantDef deserializes string "true"/"yes"/"on" as true
#[test]
fn test_deserialize_invariant_string_truthy() {
    for value in ["\"true\"", "\"yes\"", "\"on\"", "\"TRUE\"", "\"Yes\"", "\"ON\""] {
        let yaml = format!(
            "id: I-1\nname: T\ndescription: D\ncatches: []\ngate_id: G\nimplemented: {value}"
        );
        let inv: InvariantDef =
            serde_yaml::from_str(&yaml).unwrap_or_else(|e| panic!("failed for {value}: {e}"));
        assert!(inv.implemented, "expected true for {value}");
    }
}

/// Verify InvariantDef deserializes string "false"/"no"/"off" as false
#[test]
fn test_deserialize_invariant_string_falsy() {
    for value in ["\"false\"", "\"no\"", "\"off\"", "\"FALSE\"", "\"No\""] {
        let yaml = format!(
            "id: I-1\nname: T\ndescription: D\ncatches: []\ngate_id: G\nimplemented: {value}"
        );
        let inv: InvariantDef =
            serde_yaml::from_str(&yaml).unwrap_or_else(|e| panic!("failed for {value}: {e}"));
        assert!(!inv.implemented, "expected false for {value}");
    }
}

/// Verify InvariantDef rejects invalid string values for implemented
#[test]
fn test_deserialize_invariant_string_invalid() {
    let yaml = "id: I-1\nname: T\ndescription: D\ncatches: []\ngate_id: G\nimplemented: \"maybe\"";
    let result: Result<InvariantDef, _> = serde_yaml::from_str(yaml);
    assert!(result.is_err());
}

include!("contract_tests_part_a.rs");
include!("contract_tests_part_b.rs");
