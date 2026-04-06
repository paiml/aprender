//! Integration tests for trueno-graph
//!
//! Tests real-world usage scenarios (call graphs, dependency graphs)

use trueno_graph::{CsrGraph, NodeId};

#[test]
fn test_simple_call_graph() {
    // Build a simple call graph:
    // main() → parse_args(), validate(), execute()
    // parse_args() → validate()

    let mut graph = CsrGraph::new();

    // Add edges
    graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap(); // main → parse_args
    graph.add_edge(NodeId(0), NodeId(2), 1.0).unwrap(); // main → validate
    graph.add_edge(NodeId(0), NodeId(3), 1.0).unwrap(); // main → execute
    graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap(); // parse_args → validate

    // Add node names
    graph.set_node_name(NodeId(0), "main".to_string());
    graph.set_node_name(NodeId(1), "parse_args".to_string());
    graph.set_node_name(NodeId(2), "validate".to_string());
    graph.set_node_name(NodeId(3), "execute".to_string());

    // Test outgoing edges (callees)
    let main_callees = graph.outgoing_neighbors(NodeId(0)).unwrap();
    assert_eq!(main_callees, &[1, 2, 3]);

    let parse_args_callees = graph.outgoing_neighbors(NodeId(1)).unwrap();
    assert_eq!(parse_args_callees, &[2]);

    // Test incoming edges (callers)
    let validate_callers = graph.incoming_neighbors(NodeId(2)).unwrap();
    assert_eq!(validate_callers, vec![0, 1]);

    // Test graph metrics
    assert_eq!(graph.num_nodes(), 4);
    assert_eq!(graph.num_edges(), 4);
}

#[test]
fn test_from_edge_list_performance() {
    // Test building graph from edge list (batch mode)
    let edges: Vec<_> = (0..1000).map(|i| (NodeId(i), NodeId(i + 1), 1.0)).collect();

    let graph = CsrGraph::from_edge_list(&edges).unwrap();

    assert_eq!(graph.num_nodes(), 1001);
    assert_eq!(graph.num_edges(), 1000);

    // Verify first and last edges
    assert_eq!(graph.outgoing_neighbors(NodeId(0)).unwrap(), &[1]);
    assert_eq!(graph.outgoing_neighbors(NodeId(999)).unwrap(), &[1000]);
}

#[tokio::test]
async fn test_parquet_persistence() {
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let path = dir.path().join("call_graph");

    // Build graph
    let mut graph = CsrGraph::new();
    graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
    graph.add_edge(NodeId(0), NodeId(2), 2.0).unwrap();
    graph.set_node_name(NodeId(0), "main".to_string());
    graph.set_node_name(NodeId(1), "foo".to_string());
    graph.set_node_name(NodeId(2), "bar".to_string());

    // Save
    graph.write_parquet(&path).await.unwrap();

    // Verify files exist
    assert!(std::path::Path::new(&format!("{}_edges.parquet", path.display())).exists());
    assert!(std::path::Path::new(&format!("{}_nodes.parquet", path.display())).exists());

    // Load
    let loaded = CsrGraph::read_parquet(&path).await.unwrap();

    // Verify
    assert_eq!(loaded.num_nodes(), 3);
    assert_eq!(loaded.num_edges(), 2);
    assert_eq!(loaded.get_node_name(NodeId(0)), Some("main"));
}

#[test]
fn test_csr_components_for_aprender() {
    // Test CSR component extraction (for aprender integration)
    let edges =
        vec![(NodeId(0), NodeId(1), 1.0), (NodeId(0), NodeId(2), 2.0), (NodeId(1), NodeId(2), 3.0)];

    let graph = CsrGraph::from_edge_list(&edges).unwrap();
    let (row_offsets, col_indices, weights) = graph.csr_components();

    // Verify CSR format
    assert_eq!(row_offsets, &[0, 2, 3, 3]);
    assert_eq!(col_indices, &[1, 2, 2]);
    assert_eq!(weights, &[1.0, 2.0, 3.0]);
}

#[test]
fn test_adjacency_iteration() {
    // Test iter_adjacency for complete graph processing
    let edges = vec![
        (NodeId(0), NodeId(1), 1.5),
        (NodeId(0), NodeId(2), 2.5),
        (NodeId(1), NodeId(2), 3.5),
        (NodeId(1), NodeId(3), 4.5),
    ];

    let graph = CsrGraph::from_edge_list(&edges).unwrap();

    // Test adjacency method
    let (targets, weights) = graph.adjacency(NodeId(0));
    assert_eq!(targets, &[1, 2]);
    assert_eq!(weights, &[1.5, 2.5]);

    let (targets, weights) = graph.adjacency(NodeId(1));
    assert_eq!(targets, &[2, 3]);
    assert_eq!(weights, &[3.5, 4.5]);

    // Test iterator
    let adjacencies: Vec<_> = graph.iter_adjacency().collect();
    assert_eq!(adjacencies.len(), 4);

    // Verify slices
    assert_eq!(graph.row_offsets_slice(), &[0, 2, 4, 4, 4]);
    assert_eq!(graph.col_indices_slice(), &[1, 2, 2, 3]);
    assert_eq!(graph.edge_weights_slice(), &[1.5, 2.5, 3.5, 4.5]);
}
