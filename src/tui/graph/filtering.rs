//! Graph filtering operations
//!
//! Contains filtering methods for `Graph<N: Clone, E: Clone>`.

use super::graph_core::Graph;
use super::types::{Edge, Node};
use crate::tui::graph_analytics::GraphAnalytics;

// ============================================================================
// GRAPH-005b: Filtering (Neo4j/Gephi pattern)
// ============================================================================

impl<N: Clone, E: Clone> Graph<N, E> {
    /// Filter graph to nodes matching predicate
    ///
    /// Returns a new graph containing only matching nodes and their edges.
    #[must_use]
    pub fn filter_nodes<F>(&self, predicate: F) -> Self
    where
        F: Fn(&Node<N>) -> bool,
    {
        let mut filtered = Self::new();

        // Add matching nodes
        for node in self.nodes() {
            if predicate(node) {
                filtered.add_node(node.clone());
            }
        }

        // Add edges where both endpoints exist
        for edge in &self.edges {
            if filtered.nodes.contains_key(&edge.from) && filtered.nodes.contains_key(&edge.to) {
                filtered.add_edge(edge.clone());
            }
        }

        filtered
    }

    /// Filter to nodes with minimum degree
    #[must_use]
    pub fn filter_by_min_degree(&self, min_degree: usize) -> Self {
        let degrees = GraphAnalytics::degree_centrality(self);
        let n = self.node_count();
        let threshold = if n > 1 { min_degree as f32 / (n - 1) as f32 } else { 0.0 };

        self.filter_nodes(|node| degrees.get(&node.id).unwrap_or(&0.0) >= &threshold)
    }

    /// Filter to top N nodes by importance
    #[must_use]
    pub fn filter_top_n(&self, n: usize) -> Self {
        let mut nodes_by_importance: Vec<_> = self.nodes().collect();
        nodes_by_importance.sort_by(|a, b| {
            b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal)
        });

        let top_ids: std::collections::HashSet<_> =
            nodes_by_importance.iter().take(n).map(|n| &n.id).collect();

        self.filter_nodes(|node| top_ids.contains(&node.id))
    }

    /// Filter to nodes matching label pattern
    #[must_use]
    pub fn filter_by_label(&self, pattern: &str) -> Self {
        let pattern_lower = pattern.to_lowercase();
        self.filter_nodes(|node| {
            node.label.as_ref().map(|l| l.to_lowercase().contains(&pattern_lower)).unwrap_or(false)
        })
    }

    /// Filter to subgraph containing path between two nodes
    #[must_use]
    pub fn filter_path(&self, from: &str, to: &str) -> Self {
        if let Some(path) = GraphAnalytics::shortest_path(self, from, to) {
            let path_set: std::collections::HashSet<_> = path.into_iter().collect();
            self.filter_nodes(|node| path_set.contains(&node.id))
        } else {
            Self::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_by_min_degree_single_node() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ()));
        // Single node has threshold 0.0
        let filtered = graph.filter_by_min_degree(1);
        assert_eq!(filtered.node_count(), 1);
    }

    #[test]
    fn test_filter_by_min_degree_empty_graph() {
        let graph: Graph<(), ()> = Graph::new();
        let filtered = graph.filter_by_min_degree(0);
        assert_eq!(filtered.node_count(), 0);
    }

    #[test]
    fn test_filter_path_no_path() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ()));
        graph.add_node(Node::new("B", ()));
        // No edges, so no path
        let filtered = graph.filter_path("A", "B");
        assert_eq!(filtered.node_count(), 0);
    }

    #[test]
    fn test_filter_by_label_no_match() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ()).with_label("Hello"));
        let filtered = graph.filter_by_label("xyz");
        assert_eq!(filtered.node_count(), 0);
    }

    #[test]
    fn test_filter_by_label_no_labels() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ())); // No label
        let filtered = graph.filter_by_label("test");
        assert_eq!(filtered.node_count(), 0);
    }

    #[test]
    fn test_filter_nodes_with_edges() {
        let mut graph: Graph<&str, i32> = Graph::new();
        graph.add_node(Node::new("A", "a"));
        graph.add_node(Node::new("B", "b"));
        graph.add_node(Node::new("C", "c"));
        graph.add_edge(Edge::new("A", "B", 1));
        graph.add_edge(Edge::new("B", "C", 2));

        // Filter to keep A and C (not B)
        let filtered = graph.filter_nodes(|n| n.id != "B");
        assert_eq!(filtered.node_count(), 2);
        assert_eq!(filtered.edge_count(), 0); // Edges to B are removed
    }

    #[test]
    fn test_filter_top_n_with_equal_importance() {
        let mut graph: Graph<(), ()> = Graph::new();
        for i in 0..5 {
            let node = Node::new(format!("n{}", i), ());
            // All same importance
            graph.add_node(node);
        }
        let filtered = graph.filter_top_n(3);
        assert_eq!(filtered.node_count(), 3);
    }
}
