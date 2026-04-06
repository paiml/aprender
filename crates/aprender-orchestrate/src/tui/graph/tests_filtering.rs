//! Filtering tests for graph TUI visualization
//!
//! Tests for filtering operations and edge features.

use super::*;

// -------------------------------------------------------------------------
// Filtering Tests (Neo4j/Gephi patterns)
// -------------------------------------------------------------------------

#[test]
fn test_filter_nodes() {
    let mut graph: Graph<&str, ()> = Graph::new();
    graph.add_node(Node::new("A", "data").with_label("Alpha"));
    graph.add_node(Node::new("B", "data").with_label("Beta"));
    graph.add_node(Node::new("C", "data").with_label("Gamma"));
    graph.add_edge(Edge::new("A", "B", ()));
    graph.add_edge(Edge::new("B", "C", ()));

    let filtered = graph.filter_nodes(|n| n.id != "B");
    assert_eq!(filtered.node_count(), 2);
    assert_eq!(filtered.edge_count(), 0); // Edges to B should be removed
}

#[test]
fn test_filter_by_label() {
    let mut graph: Graph<&str, ()> = Graph::new();
    graph.add_node(Node::new("A", "data").with_label("Alpha"));
    graph.add_node(Node::new("B", "data").with_label("Beta"));
    graph.add_node(Node::new("C", "data").with_label("Gamma"));

    let filtered = graph.filter_by_label("alpha"); // Case insensitive
    assert_eq!(filtered.node_count(), 1);
    assert!(filtered.get_node("A").is_some());
}

#[test]
fn test_filter_top_n() {
    let mut graph: Graph<(), ()> = Graph::new();
    for i in 0..5 {
        let mut node = Node::new(format!("n{}", i), ());
        node.importance = i as f32;
        graph.add_node(node);
    }

    let filtered = graph.filter_top_n(3);
    assert_eq!(filtered.node_count(), 3);

    // Should contain n2, n3, n4 (highest importance)
    assert!(filtered.get_node("n4").is_some());
    assert!(filtered.get_node("n3").is_some());
    assert!(filtered.get_node("n2").is_some());
}

#[test]
fn test_filter_path() {
    let mut graph: Graph<(), ()> = Graph::new();
    for c in ['A', 'B', 'C', 'D', 'E'] {
        graph.add_node(Node::new(c.to_string(), ()));
    }
    graph.add_edge(Edge::new("A", "B", ()));
    graph.add_edge(Edge::new("B", "C", ()));
    graph.add_edge(Edge::new("C", "D", ()));
    graph.add_edge(Edge::new("A", "E", ())); // Side branch

    let path_graph = graph.filter_path("A", "D");
    assert_eq!(path_graph.node_count(), 4); // A, B, C, D
    assert!(path_graph.get_node("E").is_none()); // E not on path
}

// -------------------------------------------------------------------------
// Edge Features Tests
// -------------------------------------------------------------------------

#[test]
fn test_edge_weight() {
    let edge: Edge<()> = Edge::new("A", "B", ()).with_weight(2.5);
    assert_eq!(edge.weight, 2.5);
}

#[test]
fn test_edge_label() {
    let edge: Edge<()> = Edge::new("A", "B", ()).with_label("depends_on");
    assert_eq!(edge.label.as_deref(), Some("depends_on"));
}
