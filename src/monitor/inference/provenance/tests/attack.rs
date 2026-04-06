//! Tests for AttackPath and Anomaly.

use crate::monitor::inference::provenance::*;

#[test]
fn test_attack_path() {
    let path = AttackPath { nodes: vec![], edges: vec![], duration_ns: 0, anomaly_indices: vec![] };

    assert!(path.is_empty());
    assert!(!path.has_anomalies());

    let path_with_anomalies =
        AttackPath { nodes: vec![], edges: vec![], duration_ns: 0, anomaly_indices: vec![0, 1] };

    assert!(path_with_anomalies.has_anomalies());
}

#[test]
fn test_attack_path_len() {
    let path = AttackPath {
        nodes: vec![(
            0,
            ProvenanceNode::Input { source: "a".to_string(), timestamp_ns: 0, hash: 0 },
        )],
        edges: vec![],
        duration_ns: 0,
        anomaly_indices: vec![],
    };
    assert_eq!(path.len(), 1);
    assert!(!path.is_empty());
}

#[test]
fn test_attack_path_clone() {
    let path = AttackPath {
        nodes: vec![(
            0,
            ProvenanceNode::Input { source: "a".to_string(), timestamp_ns: 0, hash: 0 },
        )],
        edges: vec![],
        duration_ns: 1000,
        anomaly_indices: vec![0],
    };
    let cloned = path.clone();
    assert_eq!(path.len(), cloned.len());
    assert_eq!(path.duration_ns, cloned.duration_ns);
    assert_eq!(path.anomaly_indices.len(), cloned.anomaly_indices.len());
}

#[test]
fn test_anomaly_struct() {
    let anomaly = Anomaly { node_id: 42, description: "test anomaly".to_string(), severity: 0.75 };
    assert_eq!(anomaly.node_id, 42);
    assert_eq!(anomaly.description, "test anomaly");
    assert!((anomaly.severity - 0.75).abs() < 1e-6);
}

#[test]
fn test_anomaly_clone() {
    let anomaly = Anomaly { node_id: 1, description: "test".to_string(), severity: 0.5 };
    let cloned = anomaly.clone();
    assert_eq!(anomaly.node_id, cloned.node_id);
    assert_eq!(anomaly.description, cloned.description);
}
