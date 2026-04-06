//! Tests for IncidentReconstructor.

use crate::monitor::inference::provenance::*;

#[test]
fn test_reconstruct_path() {
    let mut graph = ProvenanceGraph::new();

    // Build a chain: Input -> Transform -> Inference -> Action
    let id1 = graph.add_node(ProvenanceNode::Input {
        source: "sensor".to_string(),
        timestamp_ns: 1000,
        hash: 0,
    });
    let id2 = graph.add_node(ProvenanceNode::Transform {
        operation: "normalize".to_string(),
        input_refs: vec![id1],
    });
    let id3 = graph.add_node(ProvenanceNode::Inference {
        model_id: "detector".to_string(),
        model_version: "1.0".to_string(),
        confidence: 0.9,
        output: 1.0,
    });
    let id4 = graph.add_node(ProvenanceNode::Action {
        action_type: "brake".to_string(),
        confidence: 0.85,
        alternatives: vec![("accelerate".to_string(), 0.1)],
    });

    graph.add_edge(ProvenanceEdge {
        from: id1,
        to: id2,
        relation: CausalRelation::DataFlow,
        timestamp_ns: 1100,
    });
    graph.add_edge(ProvenanceEdge {
        from: id2,
        to: id3,
        relation: CausalRelation::Triggered,
        timestamp_ns: 1200,
    });
    graph.add_edge(ProvenanceEdge {
        from: id3,
        to: id4,
        relation: CausalRelation::Caused,
        timestamp_ns: 1300,
    });

    let reconstructor = IncidentReconstructor::new(&graph);
    let path = reconstructor.reconstruct_path(id4, 10);

    assert_eq!(path.len(), 4);
    assert_eq!(path.edges.len(), 3);
}

#[test]
fn test_identify_anomalies() {
    let mut graph = ProvenanceGraph::new();

    let id1 = graph.add_node(ProvenanceNode::Input {
        source: "sensor".to_string(),
        timestamp_ns: 1000,
        hash: 0,
    });
    let id2 = graph.add_node(ProvenanceNode::Inference {
        model_id: "detector".to_string(),
        model_version: "1.0".to_string(),
        confidence: 0.3, // Low confidence
        output: 1.0,
    });

    graph.add_edge(ProvenanceEdge {
        from: id1,
        to: id2,
        relation: CausalRelation::Triggered,
        timestamp_ns: 1100,
    });

    let reconstructor = IncidentReconstructor::new(&graph);
    let path = reconstructor.reconstruct_path(id2, 10);
    let anomalies = reconstructor.identify_anomalies(&path, 0.7);

    assert!(!anomalies.is_empty());
    assert!(anomalies[0].description.contains("Low confidence"));
}

#[test]
fn test_calculate_duration_with_timestamps() {
    let mut graph = ProvenanceGraph::new();

    let id1 = graph.add_node(ProvenanceNode::Input {
        source: "sensor".to_string(),
        timestamp_ns: 1000,
        hash: 0,
    });
    let id2 = graph.add_node(ProvenanceNode::Input {
        source: "sensor2".to_string(),
        timestamp_ns: 5000,
        hash: 0,
    });

    graph.add_edge(ProvenanceEdge {
        from: id1,
        to: id2,
        relation: CausalRelation::DataFlow,
        timestamp_ns: 2000,
    });

    let reconstructor = IncidentReconstructor::new(&graph);
    let path = reconstructor.reconstruct_path(id2, 10);
    assert!(path.duration_ns >= 4000);
}

#[test]
fn test_reconstruct_path_max_depth_exceeded() {
    let mut graph = ProvenanceGraph::new();

    let id1 =
        graph.add_node(ProvenanceNode::Input { source: "a".to_string(), timestamp_ns: 0, hash: 0 });
    let id2 = graph.add_node(ProvenanceNode::Transform {
        operation: "op1".to_string(),
        input_refs: vec![id1],
    });
    let id3 = graph.add_node(ProvenanceNode::Transform {
        operation: "op2".to_string(),
        input_refs: vec![id2],
    });

    graph.add_edge(ProvenanceEdge {
        from: id1,
        to: id2,
        relation: CausalRelation::DataFlow,
        timestamp_ns: 100,
    });
    graph.add_edge(ProvenanceEdge {
        from: id2,
        to: id3,
        relation: CausalRelation::DataFlow,
        timestamp_ns: 200,
    });

    let reconstructor = IncidentReconstructor::new(&graph);
    // Max depth 1 should not include id1
    let path = reconstructor.reconstruct_path(id3, 1);
    assert!(path.len() <= 3);
}

#[test]
fn test_identify_anomalies_fusion_many_inputs() {
    let mut graph = ProvenanceGraph::new();

    // Create many input nodes
    let input_ids: Vec<NodeId> = (0..15)
        .map(|i| {
            graph.add_node(ProvenanceNode::Input {
                source: format!("sensor_{i}"),
                timestamp_ns: 0,
                hash: 0,
            })
        })
        .collect();

    // Create fusion with many inputs
    let fusion_id = graph.add_node(ProvenanceNode::Fusion {
        strategy: "merge".to_string(),
        input_refs: input_ids.clone(),
    });

    // Add edges
    for id in &input_ids {
        graph.add_edge(ProvenanceEdge {
            from: *id,
            to: fusion_id,
            relation: CausalRelation::DataFlow,
            timestamp_ns: 100,
        });
    }

    let reconstructor = IncidentReconstructor::new(&graph);
    let path = reconstructor.reconstruct_path(fusion_id, 10);
    let anomalies = reconstructor.identify_anomalies(&path, 0.7);

    assert!(anomalies.iter().any(|a| a.description.contains("many fusion inputs")));
}

#[test]
fn test_identify_anomalies_no_predecessors() {
    let mut graph = ProvenanceGraph::new();

    // Create a transform node with no predecessors (orphan)
    let id = graph.add_node(ProvenanceNode::Transform {
        operation: "orphan".to_string(),
        input_refs: vec![],
    });

    let reconstructor = IncidentReconstructor::new(&graph);
    let path = reconstructor.reconstruct_path(id, 10);
    let anomalies = reconstructor.identify_anomalies(&path, 0.7);

    assert!(anomalies.iter().any(|a| a.description.contains("no predecessors")));
}
