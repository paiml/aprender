//! Subgraph pattern matching for anti-pattern detection
//!
//! Find specific graph patterns (e.g., code smells, anti-patterns).
//! Based on subgraph isomorphism and motif detection algorithms.
//!
//! # References
//! - Kim et al. (2023): "Efficient Subgraph Matching on Large Graphs"
//! - Cordella et al. (2004): "VF2 algorithm for subgraph isomorphism"

use crate::{CsrGraph, NodeId};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Pattern matching result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Nodes involved in the pattern (pattern node → graph node mapping)
    pub node_mapping: HashMap<u32, NodeId>,

    /// Pattern name (e.g., "God Class", "Circular Dependency")
    pub pattern_name: String,

    /// Severity: Low, Medium, High, Critical
    pub severity: Severity,
}

/// Severity level for anti-patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Low severity (minor code smell)
    Low,
    /// Medium severity (should be refactored)
    Medium,
    /// High severity (significant anti-pattern)
    High,
    /// Critical severity (architectural flaw)
    Critical,
}

/// Pattern specification for matching
#[derive(Debug, Clone)]
pub struct Pattern {
    /// Pattern name
    pub name: String,

    /// Pattern edges (source, target) - node IDs are local to the pattern
    pub edges: Vec<(u32, u32)>,

    /// Minimum number of nodes in the pattern
    pub min_nodes: usize,

    /// Maximum number of nodes (None = unbounded)
    pub max_nodes: Option<usize>,

    /// Severity if found
    pub severity: Severity,
}

impl Pattern {
    /// Create a new pattern
    #[must_use]
    pub fn new(name: String, edges: Vec<(u32, u32)>, severity: Severity) -> Self {
        let num_nodes =
            edges.iter().flat_map(|(s, t)| [*s, *t]).max().map_or(0, |max| max as usize + 1);

        Self { name, edges, min_nodes: num_nodes, max_nodes: None, severity }
    }

    /// Create a "God Class" pattern (node with many outgoing edges)
    ///
    /// Detects functions that call too many other functions (high fan-out)
    #[must_use]
    pub fn god_class(min_callees: usize) -> Self {
        Self {
            name: "God Class".to_string(),
            edges: Vec::new(), // Filled dynamically based on min_callees
            min_nodes: min_callees + 1,
            max_nodes: None,
            severity: Severity::High,
        }
    }

    /// Create a "Circular Dependency" pattern (cycle of length N)
    ///
    /// Detects function call cycles
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // Pattern length >4B nodes not realistic
    pub fn circular_dependency(cycle_length: usize) -> Self {
        let mut edges = Vec::new();
        for i in 0..cycle_length {
            let next = (i + 1) % cycle_length;
            edges.push((i as u32, next as u32));
        }

        Self {
            name: "Circular Dependency".to_string(),
            edges,
            min_nodes: cycle_length,
            max_nodes: Some(cycle_length),
            severity: Severity::Critical,
        }
    }

    /// Create a "Dead Code" pattern (node with no incoming edges)
    ///
    /// Detects uncalled functions
    #[must_use]
    pub fn dead_code() -> Self {
        Self {
            name: "Dead Code".to_string(),
            edges: Vec::new(),
            min_nodes: 1,
            max_nodes: Some(1),
            severity: Severity::Medium,
        }
    }
}

/// Find all pattern matches in the graph
///
/// # Arguments
///
/// * `graph` - The graph to search
/// * `pattern` - The pattern to find
///
/// # Returns
///
/// Vector of pattern matches
///
/// # Errors
///
/// Returns error if pattern is malformed
///
/// # Example
///
/// ```ignore
/// # use trueno_graph::{CsrGraph, NodeId};
/// # use trueno_graph::algorithms::pattern::{Pattern, find_patterns, Severity};
/// let mut graph = CsrGraph::new();
/// // Create a 3-node cycle: 0 -> 1 -> 2 -> 0
/// graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
/// graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();
/// graph.add_edge(NodeId(2), NodeId(0), 1.0).unwrap();
///
/// // Find circular dependencies of length 3
/// let pattern = Pattern::circular_dependency(3);
/// let matches = find_patterns(&graph, &pattern).unwrap();
///
/// assert_eq!(matches.len(), 1);
/// ```
pub fn find_patterns(graph: &CsrGraph, pattern: &Pattern) -> Result<Vec<PatternMatch>> {
    match pattern.name.as_str() {
        "God Class" => Ok(find_god_class_patterns(graph, pattern)),
        "Circular Dependency" => Ok(find_cycle_patterns(graph, pattern)),
        "Dead Code" => Ok(find_dead_code_patterns(graph, pattern)),
        _ => find_generic_patterns(graph, pattern),
    }
}

/// Find "God Class" patterns (high fan-out nodes)
fn find_god_class_patterns(graph: &CsrGraph, pattern: &Pattern) -> Vec<PatternMatch> {
    let mut matches = Vec::new();

    for node_id in 0..graph.num_nodes() {
        #[allow(clippy::cast_possible_truncation)]
        let node = NodeId(node_id as u32);

        if let Ok(neighbors) = graph.outgoing_neighbors(node) {
            if neighbors.len() >= pattern.min_nodes - 1 {
                // High fan-out detected
                let mut node_mapping = HashMap::new();
                node_mapping.insert(0, node); // Pattern node 0 = god class

                matches.push(PatternMatch {
                    node_mapping,
                    pattern_name: pattern.name.clone(),
                    severity: pattern.severity,
                });
            }
        }
    }

    matches
}

/// Find "Circular Dependency" patterns (cycles)
fn find_cycle_patterns(graph: &CsrGraph, pattern: &Pattern) -> Vec<PatternMatch> {
    let cycle_length = pattern.min_nodes;
    let mut matches = Vec::new();
    let mut visited_cycles = HashSet::new();

    // DFS to find cycles of specific length
    for start_node in 0..graph.num_nodes() {
        #[allow(clippy::cast_possible_truncation)]
        let start = NodeId(start_node as u32);

        let cycles = find_cycles_from_node(graph, start, cycle_length);

        for cycle in cycles {
            // Normalize cycle to avoid duplicates (start from smallest node)
            let mut normalized = cycle.clone();
            normalized.sort_unstable();

            if visited_cycles.insert(normalized) {
                // Create node mapping
                let mut node_mapping = HashMap::new();
                for (pattern_id, &graph_node) in cycle.iter().enumerate() {
                    #[allow(clippy::cast_possible_truncation)]
                    node_mapping.insert(pattern_id as u32, graph_node);
                }

                matches.push(PatternMatch {
                    node_mapping,
                    pattern_name: pattern.name.clone(),
                    severity: pattern.severity,
                });
            }
        }
    }

    matches
}

/// Find "Dead Code" patterns (nodes with no incoming edges)
fn find_dead_code_patterns(graph: &CsrGraph, pattern: &Pattern) -> Vec<PatternMatch> {
    let mut matches = Vec::new();

    for node_id in 0..graph.num_nodes() {
        #[allow(clippy::cast_possible_truncation)]
        let node = NodeId(node_id as u32);

        if let Ok(incoming) = graph.incoming_neighbors(node) {
            if incoming.is_empty() {
                // No callers = dead code
                let mut node_mapping = HashMap::new();
                node_mapping.insert(0, node);

                matches.push(PatternMatch {
                    node_mapping,
                    pattern_name: pattern.name.clone(),
                    severity: pattern.severity,
                });
            }
        }
    }

    matches
}

/// Find generic patterns using simplified subgraph matching
///
/// Uses backtracking to find all subgraph isomorphisms. Suitable for small patterns
/// (typical in code smell detection). For larger patterns, consider specialized algorithms.
#[allow(clippy::unnecessary_wraps, clippy::cast_possible_truncation)]
fn find_generic_patterns(graph: &CsrGraph, pattern: &Pattern) -> Result<Vec<PatternMatch>> {
    let mut matches = Vec::new();

    // Build pattern adjacency for quick lookup
    let mut pattern_adj: HashMap<u32, HashSet<u32>> = HashMap::new();
    for (src, dst) in &pattern.edges {
        pattern_adj.entry(*src).or_default().insert(*dst);
    }

    // Try matching pattern starting from each node
    for start_node in 0..graph.num_nodes() {
        let mut mapping: HashMap<u32, NodeId> = HashMap::new();
        let mut used_nodes: HashSet<NodeId> = HashSet::new();

        if try_match_pattern(
            graph,
            &pattern_adj,
            pattern.min_nodes,
            0,
            NodeId(start_node as u32),
            &mut mapping,
            &mut used_nodes,
        ) {
            // Found a match
            matches.push(PatternMatch {
                node_mapping: mapping.clone(),
                pattern_name: pattern.name.clone(),
                severity: pattern.severity,
            });
        }
    }

    Ok(matches)
}

/// Verify that all mapped edges in the pattern exist in the graph
fn edges_consistent(
    graph: &CsrGraph,
    pattern_adj: &HashMap<u32, HashSet<u32>>,
    mapping: &HashMap<u32, NodeId>,
) -> bool {
    for (&p_src, &g_src) in mapping.iter() {
        let Some(p_targets) = pattern_adj.get(&p_src) else {
            continue;
        };
        let g_targets: HashSet<u32> =
            graph.outgoing_neighbors(g_src).unwrap_or_default().iter().copied().collect();
        for &p_target in p_targets {
            if let Some(&g_target) = mapping.get(&p_target) {
                if !g_targets.contains(&g_target.0) {
                    return false;
                }
            }
        }
    }
    true
}

/// Remove a node from the mapping and used set (backtrack step)
fn backtrack(
    pattern_node: u32,
    graph_node: NodeId,
    mapping: &mut HashMap<u32, NodeId>,
    used_nodes: &mut HashSet<NodeId>,
) {
    mapping.remove(&pattern_node);
    used_nodes.remove(&graph_node);
}

/// Recursive helper for pattern matching (backtracking)
#[allow(clippy::cast_possible_truncation)]
fn try_match_pattern(
    graph: &CsrGraph,
    pattern_adj: &HashMap<u32, HashSet<u32>>,
    pattern_size: usize,
    pattern_node: u32,
    graph_node: NodeId,
    mapping: &mut HashMap<u32, NodeId>,
    used_nodes: &mut HashSet<NodeId>,
) -> bool {
    if used_nodes.contains(&graph_node) {
        return false;
    }

    mapping.insert(pattern_node, graph_node);
    used_nodes.insert(graph_node);

    if mapping.len() == pattern_size {
        return true;
    }

    if !edges_consistent(graph, pattern_adj, mapping) {
        backtrack(pattern_node, graph_node, mapping, used_nodes);
        return false;
    }

    // Try extending mapping to unmapped pattern nodes
    for next_pattern_node in 0..pattern_size as u32 {
        if mapping.contains_key(&next_pattern_node) {
            continue;
        }
        let found = (0..graph.num_nodes()).any(|next_graph_node| {
            try_match_pattern(
                graph,
                pattern_adj,
                pattern_size,
                next_pattern_node,
                NodeId(next_graph_node as u32),
                mapping,
                used_nodes,
            )
        });
        if !found {
            backtrack(pattern_node, graph_node, mapping, used_nodes);
        }
        return found;
    }

    backtrack(pattern_node, graph_node, mapping, used_nodes);
    false
}

/// Helper: Find cycles of specific length starting from a node
fn find_cycles_from_node(
    graph: &CsrGraph,
    start: NodeId,
    target_length: usize,
) -> Vec<Vec<NodeId>> {
    let mut cycles = Vec::new();
    let mut path = vec![start];
    let mut visited = HashSet::new();
    visited.insert(start.0);

    dfs_find_cycles(graph, start, start, &mut path, &mut visited, target_length, &mut cycles);

    cycles
}

/// DFS helper for cycle detection
#[allow(clippy::too_many_arguments)]
fn dfs_find_cycles(
    graph: &CsrGraph,
    current: NodeId,
    start: NodeId,
    path: &mut Vec<NodeId>,
    visited: &mut HashSet<u32>,
    target_length: usize,
    cycles: &mut Vec<Vec<NodeId>>,
) {
    if path.len() == target_length {
        // Check if current node connects back to start
        if let Ok(neighbors) = graph.outgoing_neighbors(current) {
            if neighbors.contains(&start.0) {
                cycles.push(path.clone());
            }
        }
        return;
    }

    // Explore neighbors
    if let Ok(neighbors) = graph.outgoing_neighbors(current) {
        for &neighbor_id in neighbors {
            let neighbor = NodeId(neighbor_id);

            if !visited.contains(&neighbor_id)
                || (neighbor == start && path.len() == target_length - 1)
            {
                visited.insert(neighbor_id);
                path.push(neighbor);

                dfs_find_cycles(graph, neighbor, start, path, visited, target_length, cycles);

                path.pop();
                visited.remove(&neighbor_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_god_class_pattern() {
        // Node 0 calls 5 other functions (high fan-out)
        let mut graph = CsrGraph::new();
        for i in 1..=5 {
            graph.add_edge(NodeId(0), NodeId(i), 1.0).unwrap();
        }

        let pattern = Pattern::god_class(5);
        let matches = find_patterns(&graph, &pattern).unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_name, "God Class");
        assert_eq!(matches[0].severity, Severity::High);
    }

    #[test]
    fn test_circular_dependency_pattern() {
        // Create 3-node cycle: 0 -> 1 -> 2 -> 0
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(0), 1.0).unwrap();

        let pattern = Pattern::circular_dependency(3);
        let matches = find_patterns(&graph, &pattern).unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_name, "Circular Dependency");
        assert_eq!(matches[0].severity, Severity::Critical);
    }

    #[test]
    fn test_dead_code_pattern() {
        // Node 0 -> Node 1, but Node 2 has no incoming edges
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(1), 1.0).unwrap(); // Node 2 exists but not called

        // Manually expand graph to include node 3 with no callers
        graph.add_edge(NodeId(3), NodeId(4), 1.0).unwrap();

        let pattern = Pattern::dead_code();
        let matches = find_patterns(&graph, &pattern).unwrap();

        // Nodes with no incoming edges (except if they're source nodes in edge list)
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_pattern_new() {
        let edges = vec![(0, 1), (1, 2)];
        let pattern = Pattern::new("Test Pattern".to_string(), edges, Severity::Medium);

        assert_eq!(pattern.name, "Test Pattern");
        assert_eq!(pattern.min_nodes, 3); // Nodes 0, 1, 2
        assert_eq!(pattern.severity, Severity::Medium);
    }

    #[test]
    fn test_no_circular_dependency() {
        // Linear graph: 0 -> 1 -> 2 (no cycle)
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();

        let pattern = Pattern::circular_dependency(3);
        let matches = find_patterns(&graph, &pattern).unwrap();

        assert_eq!(matches.len(), 0); // No cycles found
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Low < Severity::Medium);
        assert!(Severity::Medium < Severity::High);
        assert!(Severity::High < Severity::Critical);
    }

    #[test]
    fn test_generic_pattern_triangle() {
        // Create a triangle pattern: 0 -> 1, 1 -> 2, 2 -> 0
        let pattern =
            Pattern::new("Triangle".to_string(), vec![(0, 1), (1, 2), (2, 0)], Severity::Low);

        // Graph with triangle
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(0), 1.0).unwrap();

        let matches = find_patterns(&graph, &pattern).unwrap();
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_generic_pattern_line() {
        // Create a line pattern: 0 -> 1 -> 2
        let pattern = Pattern::new("Line".to_string(), vec![(0, 1), (1, 2)], Severity::Low);

        // Graph with matching line
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();

        let matches = find_patterns(&graph, &pattern).unwrap();
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_generic_pattern_no_match() {
        // Create a pattern that won't match
        let pattern = Pattern::new("V-shape".to_string(), vec![(0, 1), (0, 2)], Severity::Low);

        // Graph with different structure (linear)
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();

        let matches = find_patterns(&graph, &pattern).unwrap();
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_pattern_with_max_nodes() {
        let mut pattern = Pattern::circular_dependency(3);
        pattern.max_nodes = Some(3);

        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(0), 1.0).unwrap();

        let matches = find_patterns(&graph, &pattern).unwrap();
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_multiple_god_classes() {
        // Two nodes with high fan-out
        let mut graph = CsrGraph::new();
        // Node 0 calls 5 functions
        for i in 1..=5 {
            graph.add_edge(NodeId(0), NodeId(i), 1.0).unwrap();
        }
        // Node 6 calls 5 functions
        for i in 7..=11 {
            graph.add_edge(NodeId(6), NodeId(i), 1.0).unwrap();
        }

        let pattern = Pattern::god_class(5);
        let matches = find_patterns(&graph, &pattern).unwrap();

        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_multiple_cycles() {
        // Create two separate 3-node cycles
        let mut graph = CsrGraph::new();
        // Cycle 1: 0 -> 1 -> 2 -> 0
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(0), 1.0).unwrap();
        // Cycle 2: 3 -> 4 -> 5 -> 3
        graph.add_edge(NodeId(3), NodeId(4), 1.0).unwrap();
        graph.add_edge(NodeId(4), NodeId(5), 1.0).unwrap();
        graph.add_edge(NodeId(5), NodeId(3), 1.0).unwrap();

        let pattern = Pattern::circular_dependency(3);
        let matches = find_patterns(&graph, &pattern).unwrap();

        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_dead_code_with_callers() {
        // Node 0 has callers, node 1 has no callers
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(2), NodeId(0), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(3), 1.0).unwrap();

        let pattern = Pattern::dead_code();
        let matches = find_patterns(&graph, &pattern).unwrap();

        // Should find nodes 1 and 2 (no incoming edges)
        assert!(matches.len() >= 2);
    }
}
