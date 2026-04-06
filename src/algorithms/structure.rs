//! Graph structure algorithms: connected components and strongly connected components
//!
//! Provides structural analysis algorithms:
//! - `connected_components`: Count weakly connected components
//! - `kosaraju_scc`: Find strongly connected components using Kosaraju's algorithm
//!
//! # Example
//!
//! ```
//! use trueno_graph::{CsrGraph, NodeId, connected_components, kosaraju_scc};
//!
//! // Build a graph with two components: 0 → 1, 2 → 3
//! let edges = vec![
//!     (NodeId(0), NodeId(1), 1.0),
//!     (NodeId(2), NodeId(3), 1.0),
//! ];
//! let graph = CsrGraph::from_edge_list(&edges).unwrap();
//!
//! // Two weakly connected components
//! assert_eq!(connected_components(&graph), 2);
//!
//! // Four SCCs (each node is its own SCC in a DAG)
//! let sccs = kosaraju_scc(&graph);
//! assert_eq!(sccs.len(), 4);
//! ```

use crate::storage::CsrGraph;
use crate::NodeId;

/// Count the number of weakly connected components in the graph
///
/// Treats the graph as undirected for connectivity purposes.
/// Two nodes are in the same component if there's a path between them
/// (ignoring edge direction).
///
/// # Arguments
///
/// * `graph` - The CSR graph to analyze
///
/// # Returns
///
/// The number of weakly connected components
///
/// # Example
///
/// ```
/// use trueno_graph::{CsrGraph, NodeId, connected_components};
///
/// // Two disconnected edges: 0 → 1, 2 → 3
/// let edges = vec![
///     (NodeId(0), NodeId(1), 1.0),
///     (NodeId(2), NodeId(3), 1.0),
/// ];
/// let graph = CsrGraph::from_edge_list(&edges).unwrap();
///
/// assert_eq!(connected_components(&graph), 2);
/// ```
#[must_use]
pub fn connected_components(graph: &CsrGraph) -> usize {
    let n = graph.num_nodes();
    if n == 0 {
        return 0;
    }

    let mut visited = vec![false; n];
    let mut count = 0;

    for start in 0..n {
        if !visited[start] {
            // BFS/DFS to mark all reachable nodes (treating as undirected)
            undirected_dfs(graph, start, &mut visited);
            count += 1;
        }
    }

    count
}

/// DFS treating graph as undirected (follows both incoming and outgoing edges)
fn undirected_dfs(graph: &CsrGraph, node: usize, visited: &mut [bool]) {
    if visited[node] {
        return;
    }
    visited[node] = true;

    #[allow(clippy::cast_possible_truncation)]
    let node_id = NodeId(node as u32);

    // Follow outgoing edges
    if let Ok(neighbors) = graph.outgoing_neighbors(node_id) {
        for &neighbor in neighbors {
            let neighbor_idx = neighbor as usize;
            if !visited[neighbor_idx] {
                undirected_dfs(graph, neighbor_idx, visited);
            }
        }
    }

    // Follow incoming edges (treat as undirected)
    if let Ok(neighbors) = graph.incoming_neighbors(node_id) {
        for &neighbor in neighbors {
            let neighbor_idx = neighbor as usize;
            if !visited[neighbor_idx] {
                undirected_dfs(graph, neighbor_idx, visited);
            }
        }
    }
}

/// Find strongly connected components using Kosaraju's algorithm
///
/// A strongly connected component (SCC) is a maximal set of nodes where
/// every node is reachable from every other node following edge directions.
///
/// # Arguments
///
/// * `graph` - The CSR graph to analyze
///
/// # Returns
///
/// A vector of SCCs, where each SCC is a vector of `NodeId`s.
/// SCCs are returned in reverse topological order (sink SCCs first).
///
/// # Algorithm
///
/// Kosaraju's algorithm runs in O(V + E):
/// 1. DFS on original graph to get finish order
/// 2. DFS on transpose graph in reverse finish order
/// 3. Each DFS tree in step 2 is an SCC
///
/// # Example
///
/// ```
/// use trueno_graph::{CsrGraph, NodeId, kosaraju_scc};
///
/// // Cycle: 0 → 1 → 2 → 0
/// let edges = vec![
///     (NodeId(0), NodeId(1), 1.0),
///     (NodeId(1), NodeId(2), 1.0),
///     (NodeId(2), NodeId(0), 1.0),
/// ];
/// let graph = CsrGraph::from_edge_list(&edges).unwrap();
///
/// let sccs = kosaraju_scc(&graph);
/// // All three nodes form one SCC
/// assert_eq!(sccs.len(), 1);
/// assert_eq!(sccs[0].len(), 3);
/// ```
#[must_use]
pub fn kosaraju_scc(graph: &CsrGraph) -> Vec<Vec<NodeId>> {
    let n = graph.num_nodes();
    if n == 0 {
        return Vec::new();
    }

    // Step 1: DFS to get finish order
    let mut visited = vec![false; n];
    let mut finish_order = Vec::with_capacity(n);

    for start in 0..n {
        if !visited[start] {
            dfs_finish_order(graph, start, &mut visited, &mut finish_order);
        }
    }

    // Step 2: DFS on transpose in reverse finish order
    let mut visited = vec![false; n];
    let mut sccs = Vec::new();

    for &node in finish_order.iter().rev() {
        if !visited[node] {
            let mut component = Vec::new();
            dfs_transpose(graph, node, &mut visited, &mut component);
            sccs.push(component);
        }
    }

    sccs
}

/// DFS to compute finish order (post-order)
fn dfs_finish_order(
    graph: &CsrGraph,
    node: usize,
    visited: &mut [bool],
    finish_order: &mut Vec<usize>,
) {
    visited[node] = true;

    #[allow(clippy::cast_possible_truncation)]
    if let Ok(neighbors) = graph.outgoing_neighbors(NodeId(node as u32)) {
        for &neighbor in neighbors {
            let neighbor_idx = neighbor as usize;
            if !visited[neighbor_idx] {
                dfs_finish_order(graph, neighbor_idx, visited, finish_order);
            }
        }
    }

    finish_order.push(node);
}

/// DFS on transpose graph (follow incoming edges instead of outgoing)
fn dfs_transpose(graph: &CsrGraph, node: usize, visited: &mut [bool], component: &mut Vec<NodeId>) {
    visited[node] = true;

    #[allow(clippy::cast_possible_truncation)]
    component.push(NodeId(node as u32));

    // Follow incoming edges (transpose of outgoing)
    #[allow(clippy::cast_possible_truncation)]
    if let Ok(neighbors) = graph.incoming_neighbors(NodeId(node as u32)) {
        for &neighbor in neighbors {
            let neighbor_idx = neighbor as usize;
            if !visited[neighbor_idx] {
                dfs_transpose(graph, neighbor_idx, visited, component);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph_components() {
        let graph = CsrGraph::new();
        assert_eq!(connected_components(&graph), 0);
    }

    #[test]
    fn test_empty_graph_scc() {
        let graph = CsrGraph::new();
        let sccs = kosaraju_scc(&graph);
        assert!(sccs.is_empty());
    }

    #[test]
    fn test_single_node_component() {
        let edges = vec![(NodeId(0), NodeId(1), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        // 0 and 1 are connected
        assert_eq!(connected_components(&graph), 1);
    }

    #[test]
    fn test_two_disconnected_edges() {
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(2), NodeId(3), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        assert_eq!(connected_components(&graph), 2);
    }

    #[test]
    fn test_chain_single_component() {
        // 0 → 1 → 2 → 3
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(2), 1.0),
            (NodeId(2), NodeId(3), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        assert_eq!(connected_components(&graph), 1);
    }

    #[test]
    fn test_diamond_single_component() {
        // Diamond: 0 → 1 → 3, 0 → 2 → 3
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(0), NodeId(2), 1.0),
            (NodeId(1), NodeId(3), 1.0),
            (NodeId(2), NodeId(3), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        assert_eq!(connected_components(&graph), 1);
    }

    #[test]
    fn test_scc_dag_each_node_separate() {
        // DAG: 0 → 1 → 2 (no cycles, each node is its own SCC)
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(1), NodeId(2), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        let sccs = kosaraju_scc(&graph);
        assert_eq!(sccs.len(), 3);
    }

    #[test]
    fn test_scc_simple_cycle() {
        // Cycle: 0 → 1 → 2 → 0
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(2), 1.0),
            (NodeId(2), NodeId(0), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        let sccs = kosaraju_scc(&graph);
        // All three form one SCC
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);
    }

    #[test]
    fn test_scc_two_node_cycle() {
        // 0 ↔ 1
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(1), NodeId(0), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        let sccs = kosaraju_scc(&graph);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 2);
    }

    #[test]
    fn test_scc_self_loop() {
        // Self-loop: 0 → 0
        let edges = vec![(NodeId(0), NodeId(0), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        let sccs = kosaraju_scc(&graph);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 1);
    }

    #[test]
    fn test_scc_two_separate_cycles() {
        // Two separate cycles: 0 ↔ 1, 2 ↔ 3
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(0), 1.0),
            (NodeId(2), NodeId(3), 1.0),
            (NodeId(3), NodeId(2), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        let sccs = kosaraju_scc(&graph);
        assert_eq!(sccs.len(), 2);
        assert!(sccs.iter().all(|scc| scc.len() == 2));
    }

    #[test]
    fn test_scc_complex_graph() {
        // Complex: Two SCCs connected by a bridge
        // SCC1: 0 ↔ 1, SCC2: 2 ↔ 3, Bridge: 1 → 2
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(0), 1.0),
            (NodeId(1), NodeId(2), 1.0), // Bridge
            (NodeId(2), NodeId(3), 1.0),
            (NodeId(3), NodeId(2), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        let sccs = kosaraju_scc(&graph);
        assert_eq!(sccs.len(), 2);
    }

    #[test]
    fn test_scc_disconnected_with_cycles() {
        // Two disconnected cycles
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(2), 1.0),
            (NodeId(2), NodeId(0), 1.0),
            (NodeId(3), NodeId(4), 1.0),
            (NodeId(4), NodeId(3), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        let sccs = kosaraju_scc(&graph);
        assert_eq!(sccs.len(), 2);
        // One SCC has 3 nodes, one has 2
        let sizes: Vec<_> = sccs.iter().map(std::vec::Vec::len).collect();
        assert!(sizes.contains(&3));
        assert!(sizes.contains(&2));
    }

    #[test]
    fn test_connected_components_with_cycle() {
        // Cycle forms one component
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(2), 1.0),
            (NodeId(2), NodeId(0), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        assert_eq!(connected_components(&graph), 1);
    }

    #[test]
    fn test_three_separate_components() {
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(2), NodeId(3), 1.0),
            (NodeId(4), NodeId(5), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        assert_eq!(connected_components(&graph), 3);
    }
}
