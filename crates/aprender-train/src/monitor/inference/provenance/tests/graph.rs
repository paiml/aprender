//! Tests for ProvenanceGraph operations.

use crate::monitor::inference::provenance::*;

#[test]
fn test_provenance_graph_new() {
    let graph = ProvenanceGraph::new();
    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn test_add_node() {
    let mut graph = ProvenanceGraph::new();

    let id = graph.add_node(ProvenanceNode::Input {
        source: "camera".to_string(),
        timestamp_ns: 1000,
        hash: 0xdeadbeef,
    });

    assert_eq!(id, 0);
    assert_eq!(graph.node_count(), 1);
    assert!(graph.get_node(0).is_some());
}

#[test]
fn test_add_edge() {
    let mut graph = ProvenanceGraph::new();

    let id1 = graph.add_node(ProvenanceNode::Input {
        source: "camera".to_string(),
        timestamp_ns: 1000,
        hash: 0xdeadbeef,
    });

    let id2 = graph.add_node(ProvenanceNode::Inference {
        model_id: "detector".to_string(),
        model_version: "1.0".to_string(),
        confidence: 0.9,
        output: 1.0,
    });

    graph.add_edge(ProvenanceEdge {
        from: id1,
        to: id2,
        relation: CausalRelation::Triggered,
        timestamp_ns: 1100,
    });

    assert_eq!(graph.edge_count(), 1);
    assert_eq!(graph.predecessors(id2), vec![id1]);
    assert_eq!(graph.successors(id1), vec![id2]);
}

#[test]
fn test_incoming_outgoing_edges() {
    let mut graph = ProvenanceGraph::new();

    let id1 =
        graph.add_node(ProvenanceNode::Input { source: "a".to_string(), timestamp_ns: 0, hash: 0 });
    let id2 =
        graph.add_node(ProvenanceNode::Input { source: "b".to_string(), timestamp_ns: 0, hash: 0 });
    let id3 = graph.add_node(ProvenanceNode::Fusion {
        strategy: "merge".to_string(),
        input_refs: vec![id1, id2],
    });

    graph.add_edge(ProvenanceEdge {
        from: id1,
        to: id3,
        relation: CausalRelation::DataFlow,
        timestamp_ns: 100,
    });
    graph.add_edge(ProvenanceEdge {
        from: id2,
        to: id3,
        relation: CausalRelation::DataFlow,
        timestamp_ns: 100,
    });

    assert_eq!(graph.incoming_edges(id3).len(), 2);
    assert_eq!(graph.outgoing_edges(id1).len(), 1);
}

#[test]
fn test_provenance_graph_default() {
    let graph = ProvenanceGraph::default();
    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn test_edges_accessor() {
    let mut graph = ProvenanceGraph::new();
    let id1 =
        graph.add_node(ProvenanceNode::Input { source: "a".to_string(), timestamp_ns: 0, hash: 0 });
    let id2 =
        graph.add_node(ProvenanceNode::Input { source: "b".to_string(), timestamp_ns: 0, hash: 0 });

    graph.add_edge(ProvenanceEdge {
        from: id1,
        to: id2,
        relation: CausalRelation::DataFlow,
        timestamp_ns: 100,
    });

    assert_eq!(graph.edges().len(), 1);
}

#[test]
fn test_nodes_accessor() {
    let mut graph = ProvenanceGraph::new();
    graph.add_node(ProvenanceNode::Input { source: "a".to_string(), timestamp_ns: 0, hash: 0 });
    graph.add_node(ProvenanceNode::Input { source: "b".to_string(), timestamp_ns: 0, hash: 0 });

    assert_eq!(graph.nodes().len(), 2);
}

#[test]
fn test_incoming_edges_no_edges() {
    let mut graph = ProvenanceGraph::new();
    let id =
        graph.add_node(ProvenanceNode::Input { source: "a".to_string(), timestamp_ns: 0, hash: 0 });
    assert!(graph.incoming_edges(id).is_empty());
}

#[test]
fn test_outgoing_edges_no_edges() {
    let mut graph = ProvenanceGraph::new();
    let id =
        graph.add_node(ProvenanceNode::Input { source: "a".to_string(), timestamp_ns: 0, hash: 0 });
    assert!(graph.outgoing_edges(id).is_empty());
}

#[test]
fn test_predecessors_empty() {
    let mut graph = ProvenanceGraph::new();
    let id =
        graph.add_node(ProvenanceNode::Input { source: "a".to_string(), timestamp_ns: 0, hash: 0 });
    assert!(graph.predecessors(id).is_empty());
}

#[test]
fn test_successors_empty() {
    let mut graph = ProvenanceGraph::new();
    let id =
        graph.add_node(ProvenanceNode::Input { source: "a".to_string(), timestamp_ns: 0, hash: 0 });
    assert!(graph.successors(id).is_empty());
}

#[test]
fn test_get_node_nonexistent() {
    let graph = ProvenanceGraph::new();
    assert!(graph.get_node(999).is_none());
}

#[test]
fn test_add_inference_from_trace() {
    use crate::monitor::inference::path::LinearPath;
    use crate::monitor::inference::trace::DecisionTrace;

    // LinearPath::new(contributions, intercept, logit, prediction)
    let path = LinearPath::new(vec![0.5, 0.3, 0.2], 0.1, 0.7, 0.65);
    // DecisionTrace::new(timestamp_ns, sequence, input_hash, path, output, latency_ns)
    let trace = DecisionTrace::new(1000, 1, 0xdeadbeef, path, 0.7, 100);

    let mut graph = ProvenanceGraph::new();
    let id = graph.add_inference(&trace, "model-1", "v1.0");

    let node = graph.get_node(id).expect("operation should succeed");
    assert_eq!(node.type_name(), "Inference");
}
