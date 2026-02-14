use super::*;

fn create_test_graph() -> CodeGraph {
    let mut graph = CodeGraph::new();

    // Add nodes: x (variable), y (variable), add (function)
    graph.add_node(CodeGraphNode::new(0, vec![1.0, 0.0, 0.0], "variable"));
    graph.add_node(CodeGraphNode::new(1, vec![0.0, 1.0, 0.0], "variable"));
    graph.add_node(CodeGraphNode::new(2, vec![0.0, 0.0, 1.0], "function"));

    // Add edges: x -> add (data flow), y -> add (data flow)
    graph.add_edge(CodeGraphEdge::new(0, 2, CodeEdgeType::DataFlow));
    graph.add_edge(CodeGraphEdge::new(1, 2, CodeEdgeType::DataFlow));

    graph
}

#[test]
fn test_code_graph_creation() {
    let graph = create_test_graph();
    assert_eq!(graph.num_nodes(), 3);
    assert_eq!(graph.num_edges(), 2);
}

#[test]
fn test_code_graph_neighbors() {
    let graph = create_test_graph();

    // Node 0 (x) should have one neighbor (add)
    assert_eq!(graph.neighbors(0).len(), 1);
    assert_eq!(graph.neighbors(0)[0].0, 2); // Neighbor is node 2

    // Node 2 (add) should have two neighbors (x and y)
    assert_eq!(graph.neighbors(2).len(), 2);
}

#[test]
fn test_mpnn_layer_creation() {
    let layer = CodeMPNNLayer::new(3, 4);
    assert_eq!(layer.in_dim(), 3);
    assert_eq!(layer.out_dim(), 4);
}

#[test]
fn test_mpnn_layer_forward() {
    let graph = create_test_graph();
    let layer = CodeMPNNLayer::new(3, 4);

    let output = layer.forward(&graph);

    assert_eq!(output.len(), 3);
    for node_output in &output {
        assert_eq!(node_output.len(), 4);
    }
}

#[test]
fn test_mpnn_stack() {
    let graph = create_test_graph();
    let mpnn = CodeMPNN::new(&[3, 8, 4]);

    let output = mpnn.forward(&graph);

    assert_eq!(output.len(), 3);
    assert_eq!(mpnn.num_layers(), 2);
    assert_eq!(mpnn.out_dim(), 4);
}

#[test]
fn test_mean_pooling() {
    let embeddings = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    let pooled = pooling::mean_pool(&embeddings);
    assert_eq!(pooled.len(), 3);
    assert!((pooled[0] - 4.0).abs() < 1e-6);
    assert!((pooled[1] - 5.0).abs() < 1e-6);
    assert!((pooled[2] - 6.0).abs() < 1e-6);
}

#[test]
fn test_max_pooling() {
    let embeddings = vec![
        vec![1.0, 8.0, 3.0],
        vec![4.0, 5.0, 9.0],
        vec![7.0, 2.0, 6.0],
    ];

    let pooled = pooling::max_pool(&embeddings);
    assert_eq!(pooled.len(), 3);
    assert!((pooled[0] - 7.0).abs() < 1e-6);
    assert!((pooled[1] - 8.0).abs() < 1e-6);
    assert!((pooled[2] - 9.0).abs() < 1e-6);
}

#[test]
fn test_sum_pooling() {
    let embeddings = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

    let pooled = pooling::sum_pool(&embeddings);
    assert_eq!(pooled.len(), 3);
    assert!((pooled[0] - 5.0).abs() < 1e-6);
    assert!((pooled[1] - 7.0).abs() < 1e-6);
    assert!((pooled[2] - 9.0).abs() < 1e-6);
}

#[test]
fn test_empty_graph() {
    let graph = CodeGraph::new();
    let layer = CodeMPNNLayer::new(3, 4);

    let output = layer.forward(&graph);
    assert!(output.is_empty());
}

#[test]
fn test_deterministic_weights() {
    let layer1 = CodeMPNNLayer::new(3, 4).with_seed(123);
    let layer2 = CodeMPNNLayer::new(3, 4).with_seed(123);

    // Same seed should produce same weights
    assert_eq!(layer1.message_weights.len(), layer2.message_weights.len());
    for (w1, w2) in layer1
        .message_weights
        .iter()
        .zip(layer2.message_weights.iter())
    {
        assert!((w1 - w2).abs() < 1e-10);
    }
}

#[test]
fn test_edge_with_features() {
    let mut graph = CodeGraph::new();
    graph.add_node(CodeGraphNode::new(0, vec![1.0], "a"));
    graph.add_node(CodeGraphNode::new(1, vec![2.0], "b"));

    let edge = CodeGraphEdge::new(0, 1, CodeEdgeType::DataFlow).with_features(vec![0.5, 0.5]);
    graph.add_edge(edge);

    assert_eq!(graph.edges()[0].features, Some(vec![0.5, 0.5]));
}

#[test]
fn test_code_graph_default() {
    let graph = CodeGraph::default();
    assert_eq!(graph.num_nodes(), 0);
    assert_eq!(graph.num_edges(), 0);
}
