//! Citation Graph with upstream aggregation (ENT-025)
//!
//! Provides citation graph construction and traversal for
//! aggregating citations from upstream dependencies.

use crate::research::citation::CitationMetadata;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A node in the citation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationNode {
    /// The citation metadata
    pub metadata: CitationMetadata,
    /// Whether this is an upstream dependency
    pub is_upstream: bool,
    /// Depth from the root artifact
    pub depth: usize,
}

impl CitationNode {
    /// Create a new citation node
    pub fn new(metadata: CitationMetadata, is_upstream: bool) -> Self {
        Self { metadata, is_upstream, depth: 0 }
    }

    /// Set the depth
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }
}

/// An edge in the citation graph
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CitationEdge {
    /// Source artifact ID (the one doing the citing)
    pub from: String,
    /// Target artifact ID (the one being cited)
    pub to: String,
    /// Edge type
    pub edge_type: EdgeType,
}

impl CitationEdge {
    /// Create a new citation edge
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self { from: from.into(), to: to.into(), edge_type: EdgeType::Cites }
    }

    /// Set the edge type
    pub fn with_type(mut self, edge_type: EdgeType) -> Self {
        self.edge_type = edge_type;
        self
    }
}

/// Type of citation relationship
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Standard citation
    Cites,
    /// Builds upon or extends
    Extends,
    /// Uses as a dependency
    DependsOn,
    /// Derived from
    DerivedFrom,
}

/// Citation graph for tracking and aggregating citations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CitationGraph {
    /// Nodes indexed by artifact ID
    pub nodes: HashMap<String, CitationNode>,
    /// Edges in the graph
    pub edges: Vec<CitationEdge>,
}

impl CitationGraph {
    /// Create a new empty citation graph
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), edges: Vec::new() }
    }

    /// Add a citation node
    pub fn add_node(&mut self, id: impl Into<String>, node: CitationNode) {
        self.nodes.insert(id.into(), node);
    }

    /// Add a citation (creates an edge)
    pub fn add_citation(&mut self, from: impl Into<String>, to: impl Into<String>) {
        let edge = CitationEdge::new(from, to);
        if !self.edges.contains(&edge) {
            self.edges.push(edge);
        }
    }

    /// Add a citation with a specific type
    pub fn add_citation_typed(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        edge_type: EdgeType,
    ) {
        let edge = CitationEdge::new(from, to).with_type(edge_type);
        if !self.edges.contains(&edge) {
            self.edges.push(edge);
        }
    }

    /// Get all citations from a specific artifact
    pub fn citations_from(&self, artifact_id: &str) -> Vec<&CitationEdge> {
        self.edges.iter().filter(|e| e.from == artifact_id).collect()
    }

    /// Get all citations to a specific artifact
    pub fn citations_to(&self, artifact_id: &str) -> Vec<&CitationEdge> {
        self.edges.iter().filter(|e| e.to == artifact_id).collect()
    }

    /// Get upstream citations for an artifact (what it cites)
    pub fn cite_upstream(&self, artifact_id: &str) -> Vec<&CitationMetadata> {
        self.citations_from(artifact_id)
            .iter()
            .filter_map(|edge| self.nodes.get(&edge.to))
            .map(|node| &node.metadata)
            .collect()
    }

    /// Aggregate all citations transitively (including transitive dependencies)
    pub fn aggregate_all_citations(&self, root_id: &str) -> Vec<&CitationMetadata> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();

        self.aggregate_recursive(root_id, &mut visited, &mut result);

        result
    }

    /// Recursive helper for citation aggregation
    fn aggregate_recursive<'a>(
        &'a self,
        current_id: &str,
        visited: &mut HashSet<String>,
        result: &mut Vec<&'a CitationMetadata>,
    ) {
        if visited.contains(current_id) {
            return;
        }
        visited.insert(current_id.to_string());

        for edge in self.citations_from(current_id) {
            if let Some(node) = self.nodes.get(&edge.to) {
                if !visited.contains(&edge.to) {
                    result.push(&node.metadata);
                    // Recursively get transitive citations
                    self.aggregate_recursive(&edge.to, visited, result);
                }
            }
        }
    }

    /// Check for transitive citations (A cites B, B cites C => A transitively cites C)
    pub fn has_transitive_citation(&self, from: &str, to: &str) -> bool {
        let mut visited = HashSet::new();
        self.has_path(from, to, &mut visited)
    }

    /// Check if there's a path from source to target
    fn has_path(&self, current: &str, target: &str, visited: &mut HashSet<String>) -> bool {
        if current == target {
            return true;
        }
        if visited.contains(current) {
            return false;
        }
        visited.insert(current.to_string());

        for edge in self.citations_from(current) {
            if self.has_path(&edge.to, target, visited) {
                return true;
            }
        }

        false
    }

    /// Export all citations to BibTeX
    pub fn to_bibtex_all(&self) -> String {
        self.nodes.values().map(|node| node.metadata.to_bibtex()).collect::<Vec<_>>().join("\n\n")
    }

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get all upstream nodes (is_upstream = true)
    pub fn upstream_nodes(&self) -> Vec<&CitationNode> {
        self.nodes.values().filter(|n| n.is_upstream).collect()
    }

    /// Remove duplicate citations (same from-to pair)
    pub fn deduplicate(&mut self) {
        let mut seen = HashSet::new();
        self.edges.retain(|edge| {
            let key = (edge.from.clone(), edge.to.clone());
            seen.insert(key)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::artifact::{ArtifactType, Author, License, ResearchArtifact};

    fn create_test_citation(id: &str, title: &str, year: u16) -> CitationMetadata {
        let artifact = ResearchArtifact::new(id, title, ArtifactType::Paper, License::CcBy4)
            .with_author(Author::new("Test Author"));
        CitationMetadata::new(artifact, year)
    }

    #[test]
    fn test_add_citation() {
        let mut graph = CitationGraph::new();

        graph.add_citation("paper-a", "paper-b");

        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.edges[0].from, "paper-a");
        assert_eq!(graph.edges[0].to, "paper-b");
    }

    #[test]
    fn test_cite_upstream_aggregation() {
        let mut graph = CitationGraph::new();

        // Add nodes
        let citation_b = create_test_citation("paper-b", "Paper B", 2023);
        let citation_c = create_test_citation("paper-c", "Paper C", 2022);

        graph.add_node("paper-b", CitationNode::new(citation_b, true));
        graph.add_node("paper-c", CitationNode::new(citation_c, true));

        // Paper A cites B and C
        graph.add_citation("paper-a", "paper-b");
        graph.add_citation("paper-a", "paper-c");

        let upstream = graph.cite_upstream("paper-a");

        assert_eq!(upstream.len(), 2);
    }

    #[test]
    fn test_transitive_citations() {
        let mut graph = CitationGraph::new();

        // Add nodes
        let citation_b = create_test_citation("paper-b", "Paper B", 2023);
        let citation_c = create_test_citation("paper-c", "Paper C", 2022);
        let citation_d = create_test_citation("paper-d", "Paper D", 2021);

        graph.add_node("paper-b", CitationNode::new(citation_b, true));
        graph.add_node("paper-c", CitationNode::new(citation_c, true));
        graph.add_node("paper-d", CitationNode::new(citation_d, true));

        // A -> B -> C -> D
        graph.add_citation("paper-a", "paper-b");
        graph.add_citation("paper-b", "paper-c");
        graph.add_citation("paper-c", "paper-d");

        // Direct path exists
        assert!(graph.has_transitive_citation("paper-a", "paper-b"));
        assert!(graph.has_transitive_citation("paper-b", "paper-c"));

        // Transitive path exists
        assert!(graph.has_transitive_citation("paper-a", "paper-c"));
        assert!(graph.has_transitive_citation("paper-a", "paper-d"));

        // No reverse path
        assert!(!graph.has_transitive_citation("paper-d", "paper-a"));
    }

    #[test]
    fn test_no_duplicate_citations() {
        let mut graph = CitationGraph::new();

        graph.add_citation("paper-a", "paper-b");
        graph.add_citation("paper-a", "paper-b"); // Duplicate

        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_graph_to_bibtex_all() {
        let mut graph = CitationGraph::new();

        let citation_a = create_test_citation("paper-a", "Paper A", 2024);
        let citation_b = create_test_citation("paper-b", "Paper B", 2023);

        graph.add_node("paper-a", CitationNode::new(citation_a, false));
        graph.add_node("paper-b", CitationNode::new(citation_b, true));

        let bibtex = graph.to_bibtex_all();

        assert!(bibtex.contains("Paper A"));
        assert!(bibtex.contains("Paper B"));
        assert!(bibtex.contains("@article{"));
    }

    #[test]
    fn test_aggregate_all_citations() {
        let mut graph = CitationGraph::new();

        // Build a citation chain: A -> B -> C
        let citation_b = create_test_citation("paper-b", "Paper B", 2023);
        let citation_c = create_test_citation("paper-c", "Paper C", 2022);

        graph.add_node("paper-b", CitationNode::new(citation_b, true));
        graph.add_node("paper-c", CitationNode::new(citation_c, true));

        graph.add_citation("paper-a", "paper-b");
        graph.add_citation("paper-b", "paper-c");

        let all_citations = graph.aggregate_all_citations("paper-a");

        // Should get both B and C (transitively)
        assert_eq!(all_citations.len(), 2);
    }

    #[test]
    fn test_edge_types() {
        let mut graph = CitationGraph::new();

        graph.add_citation_typed("paper-a", "paper-b", EdgeType::Extends);
        graph.add_citation_typed("paper-a", "library-x", EdgeType::DependsOn);

        assert_eq!(graph.edges[0].edge_type, EdgeType::Extends);
        assert_eq!(graph.edges[1].edge_type, EdgeType::DependsOn);
    }

    #[test]
    fn test_citations_to() {
        let mut graph = CitationGraph::new();

        graph.add_citation("paper-a", "paper-x");
        graph.add_citation("paper-b", "paper-x");
        graph.add_citation("paper-c", "paper-x");

        let incoming = graph.citations_to("paper-x");
        assert_eq!(incoming.len(), 3);
    }

    #[test]
    fn test_upstream_nodes() {
        let mut graph = CitationGraph::new();

        let citation_a = create_test_citation("paper-a", "Paper A", 2024);
        let citation_b = create_test_citation("paper-b", "Paper B", 2023);
        let citation_c = create_test_citation("paper-c", "Paper C", 2022);

        graph.add_node("paper-a", CitationNode::new(citation_a, false)); // Not upstream
        graph.add_node("paper-b", CitationNode::new(citation_b, true)); // Upstream
        graph.add_node("paper-c", CitationNode::new(citation_c, true)); // Upstream

        let upstream = graph.upstream_nodes();
        assert_eq!(upstream.len(), 2);
    }

    #[test]
    fn test_deduplicate() {
        let mut graph = CitationGraph::new();

        // Manually add duplicate edges
        graph.edges.push(CitationEdge::new("a", "b"));
        graph.edges.push(CitationEdge::new("a", "b"));
        graph.edges.push(CitationEdge::new("a", "c"));

        assert_eq!(graph.edge_count(), 3);

        graph.deduplicate();

        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_node_with_depth() {
        let citation = create_test_citation("paper-a", "Paper A", 2024);
        let node = CitationNode::new(citation, true).with_depth(3);

        assert_eq!(node.depth, 3);
        assert!(node.is_upstream);
    }

    #[test]
    fn test_cycle_handling() {
        let mut graph = CitationGraph::new();

        let citation_a = create_test_citation("paper-a", "Paper A", 2024);
        let citation_b = create_test_citation("paper-b", "Paper B", 2023);

        graph.add_node("paper-a", CitationNode::new(citation_a, false));
        graph.add_node("paper-b", CitationNode::new(citation_b, true));

        // Create a cycle: A -> B -> A
        graph.add_citation("paper-a", "paper-b");
        graph.add_citation("paper-b", "paper-a");

        // Should not infinite loop
        let all = graph.aggregate_all_citations("paper-a");
        assert_eq!(all.len(), 1); // Only B, not A again
    }
}
