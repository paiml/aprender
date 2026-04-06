//! Tests for ProvenanceEdge and CausalRelation.

use crate::monitor::inference::provenance::*;

#[test]
fn test_causal_relation_equality() {
    assert_eq!(CausalRelation::DataFlow, CausalRelation::DataFlow);
    assert_ne!(CausalRelation::DataFlow, CausalRelation::Triggered);
}

#[test]
fn test_causal_relation_all_variants() {
    assert_eq!(CausalRelation::DataFlow, CausalRelation::DataFlow);
    assert_eq!(CausalRelation::Triggered, CausalRelation::Triggered);
    assert_eq!(CausalRelation::Influenced, CausalRelation::Influenced);
    assert_eq!(CausalRelation::Caused, CausalRelation::Caused);
    assert_ne!(CausalRelation::Influenced, CausalRelation::Caused);
}

#[test]
fn test_provenance_edge_clone() {
    let edge =
        ProvenanceEdge { from: 1, to: 2, relation: CausalRelation::DataFlow, timestamp_ns: 1000 };
    let cloned = edge.clone();
    assert_eq!(edge.from, cloned.from);
    assert_eq!(edge.to, cloned.to);
    assert_eq!(edge.relation, cloned.relation);
    assert_eq!(edge.timestamp_ns, cloned.timestamp_ns);
}
