//! Serialization/deserialization tests for provenance types.

use crate::monitor::inference::provenance::*;

#[test]
fn test_serde_provenance_node() {
    let node =
        ProvenanceNode::Input { source: "test".to_string(), timestamp_ns: 12345, hash: 0xdeadbeef };
    let json = serde_json::to_string(&node).expect("JSON serialization should succeed");
    let deserialized: ProvenanceNode =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(deserialized.type_name(), "Input");
}

#[test]
fn test_serde_provenance_edge() {
    let edge =
        ProvenanceEdge { from: 1, to: 2, relation: CausalRelation::Triggered, timestamp_ns: 5000 };
    let json = serde_json::to_string(&edge).expect("JSON serialization should succeed");
    let deserialized: ProvenanceEdge =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(deserialized.from, 1);
    assert_eq!(deserialized.to, 2);
    assert_eq!(deserialized.relation, CausalRelation::Triggered);
}

#[test]
fn test_serde_causal_relation() {
    let relation = CausalRelation::Influenced;
    let json = serde_json::to_string(&relation).expect("JSON serialization should succeed");
    let deserialized: CausalRelation =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(deserialized, CausalRelation::Influenced);
}
