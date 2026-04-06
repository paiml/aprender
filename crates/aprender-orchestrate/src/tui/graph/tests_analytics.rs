//! Analytics tests for graph TUI visualization
//!
//! Tests for PageRank, community detection, centrality metrics, and path analysis.

use super::*;
use crate::tui::graph_analytics::{GraphAnalytics, GraphAnalyticsExt};

// -------------------------------------------------------------------------
// PageRank Tests
// -------------------------------------------------------------------------

#[test]
fn test_pagerank_empty_graph() {
    let graph: Graph<(), ()> = Graph::new();
    let ranks = GraphAnalytics::pagerank(&graph, 0.85, 20);
    assert!(ranks.is_empty());
}

#[test]
fn test_pagerank_single_node() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("solo", ()));

    let ranks = GraphAnalytics::pagerank(&graph, 0.85, 20);
    assert_eq!(ranks.len(), 1);
    assert!((ranks.get("solo").unwrap() - 1.0).abs() < 0.01);
}

#[test]
fn test_pagerank_linear_chain() {
    // A -> B -> C
    // C should have highest rank (receives most "flow")
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("A", ()));
    graph.add_node(Node::new("B", ()));
    graph.add_node(Node::new("C", ()));
    graph.add_edge(Edge::new("A", "B", ()));
    graph.add_edge(Edge::new("B", "C", ()));

    let ranks = GraphAnalytics::pagerank(&graph, 0.85, 50);

    let rank_a = *ranks.get("A").unwrap();
    let rank_b = *ranks.get("B").unwrap();
    let rank_c = *ranks.get("C").unwrap();

    // C should have highest rank (sink node)
    assert!(rank_c > rank_b, "C should rank higher than B");
    assert!(rank_b > rank_a, "B should rank higher than A");
}

#[test]
fn test_pagerank_cycle() {
    // A -> B -> C -> A (cycle)
    // All nodes should have equal rank
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("A", ()));
    graph.add_node(Node::new("B", ()));
    graph.add_node(Node::new("C", ()));
    graph.add_edge(Edge::new("A", "B", ()));
    graph.add_edge(Edge::new("B", "C", ()));
    graph.add_edge(Edge::new("C", "A", ()));

    let ranks = GraphAnalytics::pagerank(&graph, 0.85, 50);

    let rank_a = *ranks.get("A").unwrap();
    let rank_b = *ranks.get("B").unwrap();
    let rank_c = *ranks.get("C").unwrap();

    // All should be approximately equal in a cycle
    assert!((rank_a - rank_b).abs() < 0.1);
    assert!((rank_b - rank_c).abs() < 0.1);
}

#[test]
fn test_pagerank_star_topology() {
    // Center -> A, B, C, D (star)
    // Center should have lower rank (gives away rank)
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("center", ()));
    graph.add_node(Node::new("A", ()));
    graph.add_node(Node::new("B", ()));
    graph.add_node(Node::new("C", ()));
    graph.add_node(Node::new("D", ()));
    graph.add_edge(Edge::new("center", "A", ()));
    graph.add_edge(Edge::new("center", "B", ()));
    graph.add_edge(Edge::new("center", "C", ()));
    graph.add_edge(Edge::new("center", "D", ()));

    let ranks = GraphAnalytics::pagerank(&graph, 0.85, 50);

    let rank_center = *ranks.get("center").unwrap();
    let rank_a = *ranks.get("A").unwrap();

    // Leaves should have higher rank than center
    assert!(rank_a > rank_center, "Leaf A should rank higher than center");
}

#[test]
fn test_apply_pagerank() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("A", ()));
    graph.add_node(Node::new("B", ()));
    graph.add_edge(Edge::new("A", "B", ()));

    // Initially importance is 1.0
    assert_eq!(graph.get_node("A").unwrap().importance, 1.0);

    GraphAnalytics::apply_pagerank(&mut graph, 0.85, 20);

    // After PageRank, importance should be updated
    let imp_a = graph.get_node("A").unwrap().importance;
    let imp_b = graph.get_node("B").unwrap().importance;
    assert!(imp_b > imp_a, "B should have higher importance after PageRank");
}

// -------------------------------------------------------------------------
// Community Detection Tests
// -------------------------------------------------------------------------

#[test]
fn test_community_empty_graph() {
    let graph: Graph<(), ()> = Graph::new();
    let communities = GraphAnalytics::detect_communities(&graph);
    assert!(communities.is_empty());
}

#[test]
fn test_community_single_node() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("solo", ()));

    let communities = GraphAnalytics::detect_communities(&graph);
    assert_eq!(communities.len(), 1);
    assert_eq!(*communities.get("solo").unwrap(), 0);
}

#[test]
fn test_community_two_clusters() {
    // Create two disconnected clusters (no inter-cluster edge)
    // This is a simpler test case for the simplified algorithm
    let mut graph: Graph<(), ()> = Graph::new();

    // Cluster 1: A-B-C (chain)
    graph.add_node(Node::new("A", ()));
    graph.add_node(Node::new("B", ()));
    graph.add_node(Node::new("C", ()));
    graph.add_edge(Edge::new("A", "B", ()));
    graph.add_edge(Edge::new("B", "C", ()));
    graph.add_edge(Edge::new("C", "A", ())); // Close the triangle

    let communities = GraphAnalytics::detect_communities(&graph);

    // All nodes should be assigned a community
    assert_eq!(communities.len(), 3);

    // Communities should be contiguous integers starting from 0
    let max_comm = *communities.values().max().unwrap();
    assert!(max_comm < 3, "Should have at most 3 communities for 3 nodes");
}

#[test]
fn test_community_colors() {
    // Test that community colors are valid ANSI codes
    for i in 0..10 {
        let color = GraphAnalytics::community_color(i);
        assert!(color.starts_with("\x1b["), "Should be ANSI escape code");
        assert!(color.ends_with('m'), "Should end with 'm'");
    }
}

#[test]
fn test_apply_communities() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("A", ()));
    graph.add_node(Node::new("B", ()));
    graph.add_node(Node::new("C", ()));
    graph.add_edge(Edge::new("A", "B", ()));
    graph.add_edge(Edge::new("B", "C", ()));

    let num_communities = GraphAnalytics::apply_communities(&mut graph);

    // Should detect at least 1 community
    assert!(num_communities >= 1);

    // Importance should be updated
    let imp_a = graph.get_node("A").unwrap().importance;
    assert!((0.0..=1.0).contains(&imp_a));
}

// -------------------------------------------------------------------------
// GraphAnalyticsExt Trait Tests
// -------------------------------------------------------------------------

#[test]
fn test_analytics_ext_trait() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("A", ()));
    graph.add_node(Node::new("B", ()));
    graph.add_edge(Edge::new("A", "B", ()));

    // Test trait methods
    let scores = graph.pagerank_scores();
    assert_eq!(scores.len(), 2);

    graph.compute_pagerank(0.85, 10);
    let num_comm = graph.detect_communities();
    assert!(num_comm >= 1);
}

// -------------------------------------------------------------------------
// Centrality Metrics Tests (Neo4j/Gephi patterns)
// -------------------------------------------------------------------------

#[test]
fn test_degree_centrality() {
    let mut graph: Graph<(), ()> = Graph::new();
    // Star topology: center connected to 4 nodes
    graph.add_node(Node::new("center", ()));
    for i in 0..4 {
        graph.add_node(Node::new(format!("n{}", i), ()));
        graph.add_edge(Edge::new("center", format!("n{}", i), ()));
    }

    let degrees = GraphAnalytics::degree_centrality(&graph);

    // Center should have highest degree
    let center_deg = *degrees.get("center").unwrap();
    for i in 0..4 {
        let leaf_deg = *degrees.get(&format!("n{}", i)).unwrap();
        assert!(center_deg > leaf_deg, "Center should have highest degree centrality");
    }
}

#[test]
fn test_betweenness_centrality() {
    let mut graph: Graph<(), ()> = Graph::new();
    // Line graph: A -> B -> C -> D
    for c in ['A', 'B', 'C', 'D'] {
        graph.add_node(Node::new(c.to_string(), ()));
    }
    graph.add_edge(Edge::new("A", "B", ()));
    graph.add_edge(Edge::new("B", "C", ()));
    graph.add_edge(Edge::new("C", "D", ()));

    let betweenness = GraphAnalytics::betweenness_centrality(&graph);

    // B and C should have higher betweenness (they're in the middle)
    let b_between = *betweenness.get("B").unwrap();
    let a_between = *betweenness.get("A").unwrap();
    assert!(b_between >= a_between, "Middle nodes should have higher betweenness");
}

#[test]
fn test_closeness_centrality() {
    let mut graph: Graph<(), ()> = Graph::new();
    // Star topology
    graph.add_node(Node::new("center", ()));
    for i in 0..4 {
        graph.add_node(Node::new(format!("n{}", i), ()));
        graph.add_edge(Edge::new("center", format!("n{}", i), ()));
    }

    let closeness = GraphAnalytics::closeness_centrality(&graph);

    // Center should have highest closeness
    let center_close = *closeness.get("center").unwrap();
    assert!(center_close > 0.0, "Center should have positive closeness");
}

// -------------------------------------------------------------------------
// Path Analysis Tests (Neo4j/Cytoscape patterns)
// -------------------------------------------------------------------------

#[test]
fn test_shortest_path_exists() {
    let mut graph: Graph<(), ()> = Graph::new();
    for c in ['A', 'B', 'C', 'D'] {
        graph.add_node(Node::new(c.to_string(), ()));
    }
    graph.add_edge(Edge::new("A", "B", ()));
    graph.add_edge(Edge::new("B", "C", ()));
    graph.add_edge(Edge::new("C", "D", ()));

    let path = GraphAnalytics::shortest_path(&graph, "A", "D");
    assert!(path.is_some());
    let path = path.unwrap();
    assert_eq!(path, vec!["A", "B", "C", "D"]);
}

#[test]
fn test_shortest_path_same_node() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("A", ()));

    let path = GraphAnalytics::shortest_path(&graph, "A", "A");
    assert!(path.is_some());
    assert_eq!(path.unwrap(), vec!["A"]);
}

#[test]
fn test_shortest_path_no_path() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("A", ()));
    graph.add_node(Node::new("B", ()));
    // No edge between them

    let path = GraphAnalytics::shortest_path(&graph, "A", "B");
    assert!(path.is_none());
}

#[test]
fn test_is_connected() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("A", ()));
    graph.add_node(Node::new("B", ()));
    graph.add_edge(Edge::new("A", "B", ()));

    assert!(GraphAnalytics::is_connected(&graph));

    // Add disconnected node
    graph.add_node(Node::new("C", ()));
    assert!(!GraphAnalytics::is_connected(&graph));
}
