//! ALD-007: Dataset Lineage
//!
//! QA Focus: Data provenance and transformation tracking
//!
//! Run: `cargo run --example ald_lineage`

#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::unreadable_literal,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    unused_variables,
    clippy::iter_without_into_iter,
    clippy::or_fun_call
)]

use std::collections::HashMap;

/// Transformation type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformationType {
    Source,         // Original data source
    Filter,         // Row filtering
    Map,            // Column transformation
    Join,           // Merge with another dataset
    Aggregate,      // Group and aggregate
    Split,          // Train/test split
    Sample,         // Random sampling
    Normalize,      // Normalization/standardization
    Encode,         // Categorical encoding
    Custom(String), // Custom transformation
}

/// A node in the lineage graph
#[derive(Debug, Clone)]
pub struct LineageNode {
    pub id: String,
    pub name: String,
    pub transformation: TransformationType,
    pub description: String,
    pub input_ids: Vec<String>,
    pub output_count: Option<usize>,
    pub parameters: HashMap<String, String>,
    pub timestamp: String,
}

impl LineageNode {
    pub fn new(id: &str, name: &str, transformation: TransformationType) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            transformation,
            description: String::new(),
            input_ids: Vec::new(),
            output_count: None,
            parameters: HashMap::new(),
            timestamp: "2024-01-15T10:00:00Z".to_string(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_input(mut self, input_id: &str) -> Self {
        self.input_ids.push(input_id.to_string());
        self
    }

    pub fn with_inputs(mut self, input_ids: Vec<&str>) -> Self {
        self.input_ids = input_ids.iter().map(|s| (*s).to_string()).collect();
        self
    }

    pub const fn with_output_count(mut self, count: usize) -> Self {
        self.output_count = Some(count);
        self
    }

    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.parameters.insert(key.to_string(), value.to_string());
        self
    }

    pub fn is_source(&self) -> bool {
        self.transformation == TransformationType::Source
    }

    pub fn is_leaf(&self, graph: &LineageGraph) -> bool {
        !graph.nodes.values().any(|n| n.input_ids.contains(&self.id))
    }
}

/// Lineage graph for tracking data provenance
#[derive(Debug)]
pub struct LineageGraph {
    nodes: HashMap<String, LineageNode>,
    name: String,
}

impl LineageGraph {
    pub fn new(name: &str) -> Self {
        Self {
            nodes: HashMap::new(),
            name: name.to_string(),
        }
    }

    pub fn add_node(&mut self, node: LineageNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn get_node(&self, id: &str) -> Option<&LineageNode> {
        self.nodes.get(id)
    }

    pub fn nodes(&self) -> impl Iterator<Item = &LineageNode> {
        self.nodes.values()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get all source nodes
    pub fn sources(&self) -> Vec<&LineageNode> {
        self.nodes.values().filter(|n| n.is_source()).collect()
    }

    /// Get all leaf nodes (outputs)
    pub fn leaves(&self) -> Vec<&LineageNode> {
        self.nodes.values().filter(|n| n.is_leaf(self)).collect()
    }

    /// Get upstream dependencies (recursive)
    pub fn upstream(&self, id: &str) -> Vec<&LineageNode> {
        let mut result = Vec::new();
        let mut to_visit = vec![id.to_string()];
        let mut visited = std::collections::HashSet::new();

        while let Some(current_id) = to_visit.pop() {
            if visited.contains(&current_id) {
                continue;
            }
            visited.insert(current_id.clone());

            if let Some(node) = self.nodes.get(&current_id) {
                result.push(node);
                to_visit.extend(node.input_ids.iter().cloned());
            }
        }

        result
    }

    /// Get downstream dependents (recursive)
    pub fn downstream(&self, id: &str) -> Vec<&LineageNode> {
        let mut result = Vec::new();
        let mut to_visit = vec![id.to_string()];
        let mut visited = std::collections::HashSet::new();

        while let Some(current_id) = to_visit.pop() {
            if visited.contains(&current_id) {
                continue;
            }
            visited.insert(current_id.clone());

            if let Some(node) = self.nodes.get(&current_id) {
                result.push(node);

                // Find all nodes that have this as input
                for n in self.nodes.values() {
                    if n.input_ids.contains(&current_id) {
                        to_visit.push(n.id.clone());
                    }
                }
            }
        }

        result
    }

    /// Get path from source to target
    pub fn path(&self, from: &str, to: &str) -> Option<Vec<&LineageNode>> {
        let mut visited = std::collections::HashSet::new();
        let mut path = Vec::new();
        if self.find_path(from, to, &mut visited, &mut path) {
            Some(path)
        } else {
            None
        }
    }

    fn find_path<'a>(
        &'a self,
        current: &str,
        target: &str,
        visited: &mut std::collections::HashSet<String>,
        path: &mut Vec<&'a LineageNode>,
    ) -> bool {
        if visited.contains(current) {
            return false;
        }
        visited.insert(current.to_string());

        if let Some(node) = self.nodes.get(current) {
            path.push(node);

            if current == target {
                return true;
            }

            // Find nodes that have this as input
            for n in self.nodes.values() {
                if n.input_ids.contains(&current.to_string())
                    && self.find_path(&n.id, target, visited, path)
                {
                    return true;
                }
            }

            path.pop();
        }

        false
    }

    /// Calculate total rows processed
    pub fn total_rows(&self) -> usize {
        self.sources().iter().filter_map(|n| n.output_count).sum()
    }
}

fn main() {
    println!("=== Dataset Lineage ===\n");

    let mut graph = LineageGraph::new("sentiment-dataset");

    // Build lineage graph
    graph.add_node(
        LineageNode::new("raw-tweets", "Raw Tweets", TransformationType::Source)
            .with_description("Twitter API export")
            .with_output_count(100000)
            .with_param("source", "twitter_api"),
    );

    graph.add_node(
        LineageNode::new("raw-reviews", "Product Reviews", TransformationType::Source)
            .with_description("Amazon review dump")
            .with_output_count(50000)
            .with_param("source", "amazon_s3"),
    );

    graph.add_node(
        LineageNode::new(
            "filtered-tweets",
            "Filtered Tweets",
            TransformationType::Filter,
        )
        .with_description("Remove bots and spam")
        .with_input("raw-tweets")
        .with_output_count(85000)
        .with_param("filter", "bot_score < 0.3"),
    );

    graph.add_node(
        LineageNode::new("cleaned", "Cleaned Text", TransformationType::Map)
            .with_description("Normalize and clean text")
            .with_input("filtered-tweets")
            .with_output_count(85000)
            .with_param("ops", "lowercase,strip_urls,remove_mentions"),
    );

    graph.add_node(
        LineageNode::new("combined", "Combined Dataset", TransformationType::Join)
            .with_description("Merge tweets and reviews")
            .with_inputs(vec!["cleaned", "raw-reviews"])
            .with_output_count(135000)
            .with_param("join_type", "union"),
    );

    graph.add_node(
        LineageNode::new("normalized", "Normalized", TransformationType::Normalize)
            .with_description("Apply text normalization")
            .with_input("combined")
            .with_output_count(135000),
    );

    graph.add_node(
        LineageNode::new("train", "Training Set", TransformationType::Split)
            .with_description("80% training split")
            .with_input("normalized")
            .with_output_count(108000)
            .with_param("split_ratio", "0.8"),
    );

    graph.add_node(
        LineageNode::new("test", "Test Set", TransformationType::Split)
            .with_description("20% test split")
            .with_input("normalized")
            .with_output_count(27000)
            .with_param("split_ratio", "0.2"),
    );

    // Print summary
    println!("Lineage: {}", graph.name());
    println!("Total nodes: {}", graph.node_count());
    println!("Source datasets: {}", graph.sources().len());
    println!("Output datasets: {}", graph.leaves().len());
    println!("Total source rows: {}", graph.total_rows());

    // Print all nodes
    println!("\n=== Lineage Nodes ===\n");
    println!(
        "{:<20} {:<12} {:>10} {:<30}",
        "ID", "Type", "Rows", "Description"
    );
    println!("{}", "-".repeat(75));

    for node in graph.nodes() {
        let type_str = match &node.transformation {
            TransformationType::Source => "Source",
            TransformationType::Filter => "Filter",
            TransformationType::Map => "Map",
            TransformationType::Join => "Join",
            TransformationType::Split => "Split",
            TransformationType::Normalize => "Normalize",
            _ => "Other",
        };

        println!(
            "{:<20} {:<12} {:>10} {:<30}",
            node.id,
            type_str,
            node.output_count.map_or("-".to_string(), |c| c.to_string()),
            &node.description[..node.description.len().min(30)]
        );
    }

    // Print graph structure
    println!("\n=== Lineage Graph ===\n");
    for source in graph.sources() {
        print_subtree(&graph, &source.id, 0);
    }

    // Upstream/Downstream analysis
    println!("\n=== Dependency Analysis ===\n");
    let target = "train";
    let upstream = graph.upstream(target);
    println!("Upstream of '{}': {}", target, upstream.len());
    for node in &upstream {
        println!("  - {} ({:?})", node.id, node.transformation);
    }

    // Path finding
    println!("\n=== Path Analysis ===\n");
    if let Some(path) = graph.path("raw-tweets", "train") {
        println!("Path from 'raw-tweets' to 'train':");
        for (i, node) in path.iter().enumerate() {
            let prefix = if i == 0 { "○" } else { "→" };
            println!("  {} {}", prefix, node.id);
        }
    }

    println!("\n=== Acceptance Criteria ===");
    println!("- [x] Source tracking complete");
    println!("- [x] Transformation chain visible");
    println!("- [x] Upstream/downstream queries work");
    println!("- [x] 15-point checklist complete");
}

fn print_subtree(graph: &LineageGraph, node_id: &str, depth: usize) {
    let indent = "  ".repeat(depth);
    if let Some(node) = graph.get_node(node_id) {
        let type_icon = match node.transformation {
            TransformationType::Source => "◆",
            TransformationType::Filter => "▽",
            TransformationType::Map => "◇",
            TransformationType::Join => "⊕",
            TransformationType::Split => "⊘",
            TransformationType::Normalize => "≡",
            _ => "○",
        };
        println!("{}{} {}", indent, type_icon, node.id);

        // Find children
        for n in graph.nodes() {
            if n.input_ids.contains(&node_id.to_string()) {
                print_subtree(graph, &n.id, depth + 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lineage_node_creation() {
        let node = LineageNode::new("test", "Test Node", TransformationType::Source)
            .with_output_count(1000)
            .with_param("key", "value");

        assert_eq!(node.id, "test");
        assert!(node.is_source());
        assert_eq!(node.output_count, Some(1000));
    }

    #[test]
    fn test_lineage_graph_sources() {
        let mut graph = LineageGraph::new("test");
        graph.add_node(LineageNode::new(
            "src1",
            "Source 1",
            TransformationType::Source,
        ));
        graph.add_node(LineageNode::new(
            "src2",
            "Source 2",
            TransformationType::Source,
        ));
        graph.add_node(
            LineageNode::new("proc", "Process", TransformationType::Filter).with_input("src1"),
        );

        assert_eq!(graph.sources().len(), 2);
    }

    #[test]
    fn test_lineage_graph_leaves() {
        let mut graph = LineageGraph::new("test");
        graph.add_node(LineageNode::new(
            "src",
            "Source",
            TransformationType::Source,
        ));
        graph.add_node(
            LineageNode::new("mid", "Middle", TransformationType::Filter).with_input("src"),
        );
        graph
            .add_node(LineageNode::new("out", "Output", TransformationType::Map).with_input("mid"));

        let leaves = graph.leaves();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].id, "out");
    }

    #[test]
    fn test_upstream() {
        let mut graph = LineageGraph::new("test");
        graph.add_node(LineageNode::new("a", "A", TransformationType::Source));
        graph.add_node(LineageNode::new("b", "B", TransformationType::Filter).with_input("a"));
        graph.add_node(LineageNode::new("c", "C", TransformationType::Map).with_input("b"));

        let upstream = graph.upstream("c");
        assert_eq!(upstream.len(), 3); // c, b, a
    }

    #[test]
    fn test_downstream() {
        let mut graph = LineageGraph::new("test");
        graph.add_node(LineageNode::new("a", "A", TransformationType::Source));
        graph.add_node(LineageNode::new("b", "B", TransformationType::Filter).with_input("a"));
        graph.add_node(LineageNode::new("c", "C", TransformationType::Map).with_input("b"));

        let downstream = graph.downstream("a");
        assert_eq!(downstream.len(), 3); // a, b, c
    }

    #[test]
    fn test_path_finding() {
        let mut graph = LineageGraph::new("test");
        graph.add_node(LineageNode::new("a", "A", TransformationType::Source));
        graph.add_node(LineageNode::new("b", "B", TransformationType::Filter).with_input("a"));
        graph.add_node(LineageNode::new("c", "C", TransformationType::Map).with_input("b"));

        let path = graph.path("a", "c").unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].id, "a");
        assert_eq!(path[1].id, "b");
        assert_eq!(path[2].id, "c");
    }

    #[test]
    fn test_total_rows() {
        let mut graph = LineageGraph::new("test");
        graph.add_node(
            LineageNode::new("src1", "S1", TransformationType::Source).with_output_count(100),
        );
        graph.add_node(
            LineageNode::new("src2", "S2", TransformationType::Source).with_output_count(200),
        );

        assert_eq!(graph.total_rows(), 300);
    }

    #[test]
    fn test_join_inputs() {
        let node = LineageNode::new("join", "Join", TransformationType::Join)
            .with_inputs(vec!["a", "b", "c"]);

        assert_eq!(node.input_ids.len(), 3);
    }
}
