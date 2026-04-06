//! Contract dependency graph — composition via `depends_on`.
//!
//! Builds a directed acyclic graph (DAG) of contract dependencies
//! and provides cycle detection and topological ordering.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// A dependency graph of contracts.
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Map from contract stem to its direct dependencies.
    pub edges: BTreeMap<String, Vec<String>>,
    /// All unique contract stems in the graph.
    pub nodes: BTreeSet<String>,
    /// Topological ordering (empty if cycle detected).
    pub topo_order: Vec<String>,
    /// Detected cycles (empty if DAG is valid).
    pub cycles: Vec<Vec<String>>,
}

/// A single node in the graph with metadata.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Contract stem (e.g. "softmax-kernel-v1").
    pub stem: String,
    /// Number of contracts that depend on this one.
    pub dependents: usize,
    /// Number of contracts this one depends on.
    pub dependencies: usize,
}

/// Build a dependency graph from a set of contracts.
///
/// Each contract is identified by its stem (filename without `.yaml`).
/// Dependencies come from `metadata.depends_on`.
pub fn dependency_graph(contracts: &[(String, &crate::schema::Contract)]) -> DependencyGraph {
    let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut nodes = BTreeSet::new();

    for (stem, contract) in contracts {
        nodes.insert(stem.clone());
        let deps = &contract.metadata.depends_on;
        edges.insert(stem.clone(), deps.clone());
        for dep in deps {
            nodes.insert(dep.clone());
        }
    }

    // Fill in missing nodes (dependencies that aren't in the input set)
    for node in &nodes {
        edges.entry(node.clone()).or_default();
    }

    let cycles = detect_cycles(&edges);
    let topo_order = if cycles.is_empty() {
        topological_sort(&edges, &nodes)
    } else {
        Vec::new()
    };

    DependencyGraph {
        edges,
        nodes,
        topo_order,
        cycles,
    }
}

/// Get metadata about each node in the graph.
pub fn graph_nodes(graph: &DependencyGraph) -> Vec<GraphNode> {
    let mut dependents_count: BTreeMap<&str, usize> = BTreeMap::new();
    for deps in graph.edges.values() {
        for dep in deps {
            *dependents_count.entry(dep.as_str()).or_default() += 1;
        }
    }

    graph
        .nodes
        .iter()
        .map(|stem| GraphNode {
            stem: stem.clone(),
            dependents: dependents_count.get(stem.as_str()).copied().unwrap_or(0),
            dependencies: graph.edges.get(stem).map_or(0, Vec::len),
        })
        .collect()
}

/// Kahn's algorithm for topological sort (build order: dependencies first).
fn topological_sort(
    edges: &BTreeMap<String, Vec<String>>,
    nodes: &BTreeSet<String>,
) -> Vec<String> {
    // Build reverse adjacency: for each dependency, track who depends on it
    let mut reverse_edges: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut in_degree: BTreeMap<&str, usize> = BTreeMap::new();
    for node in nodes {
        in_degree.insert(node.as_str(), 0);
        reverse_edges.entry(node.as_str()).or_default();
    }
    // in_degree[node] = number of things node depends on
    for (node, deps) in edges {
        *in_degree.entry(node.as_str()).or_default() = deps.len();
        for dep in deps {
            reverse_edges
                .entry(dep.as_str())
                .or_default()
                .push(node.as_str());
        }
    }

    // Start with nodes that depend on nothing (foundations)
    let mut queue: VecDeque<String> = nodes
        .iter()
        .filter(|n| in_degree.get(n.as_str()) == Some(&0))
        .cloned()
        .collect();

    let mut result = Vec::new();
    while let Some(node) = queue.pop_front() {
        result.push(node.clone());
        // For each node that depends on this one, decrement its in-degree
        if let Some(dependents) = reverse_edges.get(node.as_str()) {
            for &dependent in dependents {
                if let Some(deg) = in_degree.get_mut(dependent) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(dependent.to_string());
                    }
                }
            }
        }
    }

    result
}

#[derive(Clone, Copy, PartialEq)]
enum DfsColor {
    White,
    Gray,
    Black,
}

fn dfs_visit<'a>(
    node: &'a str,
    edges: &'a BTreeMap<String, Vec<String>>,
    color: &mut BTreeMap<&'a str, DfsColor>,
    path: &mut Vec<String>,
    cycles: &mut Vec<Vec<String>>,
) {
    color.insert(node, DfsColor::Gray);
    path.push(node.to_string());

    if let Some(deps) = edges.get(node) {
        for dep in deps {
            match color.get(dep.as_str()) {
                Some(DfsColor::Gray) => {
                    if let Some(pos) = path.iter().position(|n| n == dep) {
                        let cycle: Vec<String> = path[pos..].to_vec();
                        cycles.push(cycle);
                    }
                }
                Some(DfsColor::White) | None => {
                    dfs_visit(dep.as_str(), edges, color, path, cycles);
                }
                Some(DfsColor::Black) => {}
            }
        }
    }

    path.pop();
    color.insert(node, DfsColor::Black);
}

/// Detect cycles using DFS coloring.
fn detect_cycles(edges: &BTreeMap<String, Vec<String>>) -> Vec<Vec<String>> {
    let mut color: BTreeMap<&str, DfsColor> = BTreeMap::new();
    for key in edges.keys() {
        color.insert(key.as_str(), DfsColor::White);
    }

    let mut cycles = Vec::new();
    let keys: Vec<String> = edges.keys().cloned().collect();
    let mut path = Vec::new();
    for node in &keys {
        if color.get(node.as_str()) == Some(&DfsColor::White) {
            dfs_visit(node.as_str(), edges, &mut color, &mut path, &mut cycles);
        }
    }

    cycles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    fn contract_with_deps(deps: &[&str]) -> crate::schema::Contract {
        let deps_yaml = if deps.is_empty() {
            String::new()
        } else {
            let items: Vec<String> = deps.iter().map(|d| format!("    - \"{d}\"")).collect();
            format!("  depends_on:\n{}", items.join("\n"))
        };
        parse_contract_str(&format!(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
{deps_yaml}
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#
        ))
        .unwrap()
    }

    #[test]
    fn empty_graph() {
        let graph = dependency_graph(&[]);
        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());
        assert!(graph.cycles.is_empty());
    }

    #[test]
    fn single_node_no_deps() {
        let c = contract_with_deps(&[]);
        let graph = dependency_graph(&[("softmax".to_string(), &c)]);
        assert_eq!(graph.nodes.len(), 1);
        assert!(graph.cycles.is_empty());
        assert_eq!(graph.topo_order.len(), 1);
    }

    #[test]
    fn linear_chain() {
        let a = contract_with_deps(&["b"]);
        let b = contract_with_deps(&["c"]);
        let c = contract_with_deps(&[]);
        let graph = dependency_graph(&[
            ("a".to_string(), &a),
            ("b".to_string(), &b),
            ("c".to_string(), &c),
        ]);
        assert_eq!(graph.nodes.len(), 3);
        assert!(graph.cycles.is_empty());
        assert_eq!(graph.topo_order.len(), 3);
    }

    #[test]
    fn cycle_detected() {
        let a = contract_with_deps(&["b"]);
        let b = contract_with_deps(&["a"]);
        let graph = dependency_graph(&[("a".to_string(), &a), ("b".to_string(), &b)]);
        assert!(!graph.cycles.is_empty());
        assert!(graph.topo_order.is_empty());
    }

    #[test]
    fn diamond_dependency() {
        let a = contract_with_deps(&["b", "c"]);
        let b = contract_with_deps(&["d"]);
        let c = contract_with_deps(&["d"]);
        let d = contract_with_deps(&[]);
        let graph = dependency_graph(&[
            ("a".to_string(), &a),
            ("b".to_string(), &b),
            ("c".to_string(), &c),
            ("d".to_string(), &d),
        ]);
        assert_eq!(graph.nodes.len(), 4);
        assert!(graph.cycles.is_empty());
        assert_eq!(graph.topo_order.len(), 4);
    }

    #[test]
    fn graph_nodes_metadata() {
        let a = contract_with_deps(&["b"]);
        let b = contract_with_deps(&[]);
        let graph = dependency_graph(&[("a".to_string(), &a), ("b".to_string(), &b)]);
        let nodes = graph_nodes(&graph);
        assert_eq!(nodes.len(), 2);

        let a_node = nodes.iter().find(|n| n.stem == "a").unwrap();
        assert_eq!(a_node.dependencies, 1);
        assert_eq!(a_node.dependents, 0);

        let b_node = nodes.iter().find(|n| n.stem == "b").unwrap();
        assert_eq!(b_node.dependencies, 0);
        assert_eq!(b_node.dependents, 1);
    }

    #[test]
    fn external_dependency_added_to_nodes() {
        let a = contract_with_deps(&["external-crate"]);
        let graph = dependency_graph(&[("a".to_string(), &a)]);
        assert!(graph.nodes.contains("external-crate"));
        assert_eq!(graph.nodes.len(), 2);
    }

    #[test]
    fn topo_order_respects_deps() {
        let a = contract_with_deps(&["b"]);
        let b = contract_with_deps(&[]);
        let graph = dependency_graph(&[("a".to_string(), &a), ("b".to_string(), &b)]);
        let a_pos = graph.topo_order.iter().position(|n| n == "a").unwrap();
        let b_pos = graph.topo_order.iter().position(|n| n == "b").unwrap();
        // Build order: dependencies come before dependents.
        // "a" depends on "b", so "b" (foundation) should come before "a".
        assert!(b_pos < a_pos);
    }
}
