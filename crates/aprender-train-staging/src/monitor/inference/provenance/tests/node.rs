//! Tests for ProvenanceNode.

use crate::monitor::inference::provenance::*;

#[test]
fn test_node_type_name() {
    let node = ProvenanceNode::Input { source: "test".to_string(), timestamp_ns: 0, hash: 0 };
    assert_eq!(node.type_name(), "Input");

    let node = ProvenanceNode::Inference {
        model_id: "m".to_string(),
        model_version: "v".to_string(),
        confidence: 0.9,
        output: 1.0,
    };
    assert_eq!(node.type_name(), "Inference");
}

#[test]
fn test_provenance_node_timestamp_non_input() {
    let node = ProvenanceNode::Transform { operation: "op".to_string(), input_refs: vec![] };
    assert!(node.timestamp_ns().is_none());

    let node = ProvenanceNode::Inference {
        model_id: "m".to_string(),
        model_version: "v".to_string(),
        confidence: 0.9,
        output: 1.0,
    };
    assert!(node.timestamp_ns().is_none());

    let node = ProvenanceNode::Fusion { strategy: "avg".to_string(), input_refs: vec![] };
    assert!(node.timestamp_ns().is_none());

    let node = ProvenanceNode::Action {
        action_type: "stop".to_string(),
        confidence: 0.8,
        alternatives: vec![],
    };
    assert!(node.timestamp_ns().is_none());
}

#[test]
fn test_provenance_node_all_type_names() {
    assert_eq!(
        ProvenanceNode::Transform { operation: "x".to_string(), input_refs: vec![] }.type_name(),
        "Transform"
    );
    assert_eq!(
        ProvenanceNode::Fusion { strategy: "x".to_string(), input_refs: vec![] }.type_name(),
        "Fusion"
    );
    assert_eq!(
        ProvenanceNode::Action {
            action_type: "x".to_string(),
            confidence: 0.5,
            alternatives: vec![]
        }
        .type_name(),
        "Action"
    );
}
