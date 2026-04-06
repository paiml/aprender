//! Core Graph data structure
//!
//! Contains the `Graph` struct and its core methods for node/edge management.

use std::collections::HashMap;

use super::types::{Edge, Node, MAX_TUI_NODES};

// ============================================================================
// GRAPH-002: Graph Data Structure
// ============================================================================

/// Graph for TUI visualization
#[derive(Debug, Clone)]
pub struct Graph<N, E> {
    /// Nodes indexed by ID
    pub(crate) nodes: HashMap<String, Node<N>>,
    /// Edges
    pub(crate) edges: Vec<Edge<E>>,
    /// Adjacency list for fast neighbor lookup
    adjacency: HashMap<String, Vec<String>>,
}

impl<N, E> Default for Graph<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> Graph<N, E> {
    /// Create empty graph
    #[must_use]
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), edges: Vec::new(), adjacency: HashMap::new() }
    }

    /// Add a node
    pub fn add_node(&mut self, node: Node<N>) {
        let id = node.id.clone();
        self.nodes.insert(id.clone(), node);
        self.adjacency.entry(id).or_default();
    }

    /// Add an edge
    pub fn add_edge(&mut self, edge: Edge<E>) {
        self.adjacency.entry(edge.from.clone()).or_default().push(edge.to.clone());
        self.edges.push(edge);
    }

    /// Get node count
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get node by ID
    #[must_use]
    pub fn get_node(&self, id: &str) -> Option<&Node<N>> {
        self.nodes.get(id)
    }

    /// Get mutable node by ID
    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut Node<N>> {
        self.nodes.get_mut(id)
    }

    /// Iterate over nodes
    pub fn nodes(&self) -> impl Iterator<Item = &Node<N>> {
        self.nodes.values()
    }

    /// Iterate over nodes mutably
    pub fn nodes_mut(&mut self) -> impl Iterator<Item = &mut Node<N>> {
        self.nodes.values_mut()
    }

    /// Iterate over edges
    pub fn edges(&self) -> impl Iterator<Item = &Edge<E>> {
        self.edges.iter()
    }

    /// Get neighbors of a node
    #[must_use]
    pub fn neighbors(&self, id: &str) -> Vec<&str> {
        self.adjacency.get(id).map(|v| v.iter().map(String::as_str).collect()).unwrap_or_default()
    }

    /// Check if graph exceeds TUI limit (Muri prevention)
    #[must_use]
    pub fn exceeds_tui_limit(&self) -> bool {
        self.nodes.len() > MAX_TUI_NODES
    }

    /// Get top N nodes by importance (Mieruka)
    #[must_use]
    pub fn top_nodes_by_importance(&self, n: usize) -> Vec<&Node<N>> {
        let mut nodes: Vec<_> = self.nodes.values().collect();
        nodes.sort_by(|a, b| {
            b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal)
        });
        nodes.into_iter().take(n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_new() {
        let graph: Graph<(), ()> = Graph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_graph_default() {
        let graph: Graph<i32, &str> = Graph::default();
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_graph_add_node() {
        let mut graph: Graph<i32, ()> = Graph::new();
        graph.add_node(Node::new("A", 42));
        assert_eq!(graph.node_count(), 1);
        assert!(graph.get_node("A").is_some());
    }

    #[test]
    fn test_graph_add_edge() {
        let mut graph: Graph<(), &str> = Graph::new();
        graph.add_node(Node::new("A", ()));
        graph.add_node(Node::new("B", ()));
        graph.add_edge(Edge::new("A", "B", "connects"));
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_graph_get_node_not_found() {
        let graph: Graph<(), ()> = Graph::new();
        assert!(graph.get_node("X").is_none());
    }

    #[test]
    fn test_graph_get_node_mut() {
        let mut graph: Graph<i32, ()> = Graph::new();
        graph.add_node(Node::new("A", 10));
        if let Some(node) = graph.get_node_mut("A") {
            node.importance = 5.0;
        }
        assert_eq!(graph.get_node("A").expect("unexpected failure").importance, 5.0);
    }

    #[test]
    fn test_graph_nodes_iterator() {
        let mut graph: Graph<i32, ()> = Graph::new();
        graph.add_node(Node::new("A", 1));
        graph.add_node(Node::new("B", 2));
        let count = graph.nodes().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_graph_nodes_mut_iterator() {
        let mut graph: Graph<i32, ()> = Graph::new();
        graph.add_node(Node::new("A", 1));
        for node in graph.nodes_mut() {
            node.importance = 0.5;
        }
        assert_eq!(graph.get_node("A").expect("unexpected failure").importance, 0.5);
    }

    #[test]
    fn test_graph_edges_iterator() {
        let mut graph: Graph<(), i32> = Graph::new();
        graph.add_node(Node::new("A", ()));
        graph.add_node(Node::new("B", ()));
        graph.add_edge(Edge::new("A", "B", 100));
        let edges: Vec<_> = graph.edges().collect();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].data, 100);
    }

    #[test]
    fn test_graph_neighbors() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ()));
        graph.add_node(Node::new("B", ()));
        graph.add_node(Node::new("C", ()));
        graph.add_edge(Edge::new("A", "B", ()));
        graph.add_edge(Edge::new("A", "C", ()));
        let neighbors = graph.neighbors("A");
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_graph_neighbors_empty() {
        let graph: Graph<(), ()> = Graph::new();
        let neighbors = graph.neighbors("X");
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_graph_exceeds_tui_limit() {
        let graph: Graph<(), ()> = Graph::new();
        assert!(!graph.exceeds_tui_limit());
    }

    #[test]
    fn test_graph_top_nodes_by_importance() {
        let mut graph: Graph<(), ()> = Graph::new();
        for i in 0..5 {
            let mut node = Node::new(format!("n{}", i), ());
            node.importance = i as f32;
            graph.add_node(node);
        }
        let top = graph.top_nodes_by_importance(3);
        assert_eq!(top.len(), 3);
        // Highest importance first
        assert!(top[0].importance >= top[1].importance);
        assert!(top[1].importance >= top[2].importance);
    }

    #[test]
    fn test_graph_top_nodes_fewer_than_n() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ()));
        let top = graph.top_nodes_by_importance(10);
        assert_eq!(top.len(), 1);
    }
}
