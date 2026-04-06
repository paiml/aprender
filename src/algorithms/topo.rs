//! Topological algorithms: cycle detection and topological sort
//!
//! Provides graph ordering algorithms needed for dependency analysis:
//! - `is_cyclic`: Detect if a directed graph contains cycles
//! - `toposort`: Topological ordering of a DAG (Directed Acyclic Graph)
//!
//! # Example
//!
//! ```
//! use trueno_graph::{CsrGraph, NodeId, is_cyclic, toposort};
//!
//! // Build a DAG: 0 → 1 → 2
//! let edges = vec![
//!     (NodeId(0), NodeId(1), 1.0),
//!     (NodeId(1), NodeId(2), 1.0),
//! ];
//! let graph = CsrGraph::from_edge_list(&edges).unwrap();
//!
//! assert!(!is_cyclic(&graph));
//!
//! let order = toposort(&graph).unwrap();
//! assert_eq!(order, vec![NodeId(0), NodeId(1), NodeId(2)]);
//! ```

use crate::storage::CsrGraph;
use crate::NodeId;
use anyhow::{anyhow, Result};

/// Node state during DFS traversal
#[derive(Clone, Copy, PartialEq, Eq)]
enum NodeState {
    /// Not yet visited
    Unvisited,
    /// Currently in DFS stack (part of current path)
    InStack,
    /// Fully processed (all descendants visited)
    Finished,
}

/// Check if the graph contains any cycles
///
/// Uses depth-first search with three-color marking:
/// - Unvisited: not yet seen
/// - `InStack`: currently being explored (back edge = cycle)
/// - Finished: fully explored
///
/// # Arguments
///
/// * `graph` - The CSR graph to check
///
/// # Returns
///
/// `true` if the graph contains at least one cycle, `false` otherwise
///
/// # Example
///
/// ```
/// use trueno_graph::{CsrGraph, NodeId, is_cyclic};
///
/// // Acyclic: 0 → 1 → 2
/// let dag = CsrGraph::from_edge_list(&[
///     (NodeId(0), NodeId(1), 1.0),
///     (NodeId(1), NodeId(2), 1.0),
/// ]).unwrap();
/// assert!(!is_cyclic(&dag));
///
/// // Cyclic: 0 → 1 → 2 → 0
/// let cyclic = CsrGraph::from_edge_list(&[
///     (NodeId(0), NodeId(1), 1.0),
///     (NodeId(1), NodeId(2), 1.0),
///     (NodeId(2), NodeId(0), 1.0),
/// ]).unwrap();
/// assert!(is_cyclic(&cyclic));
/// ```
#[must_use]
pub fn is_cyclic(graph: &CsrGraph) -> bool {
    let n = graph.num_nodes();
    if n == 0 {
        return false;
    }

    let mut state = vec![NodeState::Unvisited; n];

    // Check all nodes (handles disconnected components)
    for start in 0..n {
        if state[start] == NodeState::Unvisited && has_cycle_from(graph, start, &mut state) {
            return true;
        }
    }

    false
}

/// DFS helper to detect cycle starting from a node
fn has_cycle_from(graph: &CsrGraph, node: usize, state: &mut [NodeState]) -> bool {
    state[node] = NodeState::InStack;

    // Check all outgoing neighbors
    #[allow(clippy::cast_possible_truncation)]
    let neighbors = graph.outgoing_neighbors(NodeId(node as u32));

    if let Ok(neighbors) = neighbors {
        for &neighbor in neighbors {
            let neighbor_idx = neighbor as usize;

            match state[neighbor_idx] {
                NodeState::InStack => {
                    // Back edge found - cycle detected!
                    return true;
                }
                NodeState::Unvisited => {
                    if has_cycle_from(graph, neighbor_idx, state) {
                        return true;
                    }
                }
                NodeState::Finished => {
                    // Already fully explored, skip
                }
            }
        }
    }

    state[node] = NodeState::Finished;
    false
}

/// Compute topological ordering of a directed acyclic graph (DAG)
///
/// Returns nodes in an order where for every edge u → v, u appears before v.
/// This is the order needed for dependency resolution (dependencies first).
///
/// # Arguments
///
/// * `graph` - The CSR graph to sort
///
/// # Returns
///
/// * `Ok(Vec<NodeId>)` - Nodes in topological order
/// * `Err` - If the graph contains a cycle
///
/// # Errors
///
/// Returns an error if the graph contains a cycle, making topological ordering impossible.
///
/// # Example
///
/// ```
/// use trueno_graph::{CsrGraph, NodeId, toposort};
///
/// // DAG: 0 → 1 → 2, 0 → 2
/// let graph = CsrGraph::from_edge_list(&[
///     (NodeId(0), NodeId(1), 1.0),
///     (NodeId(1), NodeId(2), 1.0),
///     (NodeId(0), NodeId(2), 1.0),
/// ]).unwrap();
///
/// let order = toposort(&graph).unwrap();
///
/// // Node 0 must come before 1 and 2
/// let pos_0 = order.iter().position(|&n| n == NodeId(0)).unwrap();
/// let pos_1 = order.iter().position(|&n| n == NodeId(1)).unwrap();
/// let pos_2 = order.iter().position(|&n| n == NodeId(2)).unwrap();
///
/// assert!(pos_0 < pos_1);
/// assert!(pos_0 < pos_2);
/// assert!(pos_1 < pos_2);
/// ```
pub fn toposort(graph: &CsrGraph) -> Result<Vec<NodeId>> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut state = vec![NodeState::Unvisited; n];
    let mut result = Vec::with_capacity(n);

    // Visit all nodes (handles disconnected components)
    for start in 0..n {
        if state[start] == NodeState::Unvisited {
            toposort_dfs(graph, start, &mut state, &mut result)?;
        }
    }

    // Reverse to get correct order (DFS gives reverse topological order)
    result.reverse();
    Ok(result)
}

/// DFS helper for topological sort
fn toposort_dfs(
    graph: &CsrGraph,
    node: usize,
    state: &mut [NodeState],
    result: &mut Vec<NodeId>,
) -> Result<()> {
    state[node] = NodeState::InStack;

    // Visit all outgoing neighbors
    #[allow(clippy::cast_possible_truncation)]
    let neighbors = graph.outgoing_neighbors(NodeId(node as u32))?;

    for &neighbor in neighbors {
        let neighbor_idx = neighbor as usize;

        match state[neighbor_idx] {
            NodeState::InStack => {
                // Back edge found - cycle detected!
                return Err(anyhow!("Cycle detected: cannot compute topological order"));
            }
            NodeState::Unvisited => {
                toposort_dfs(graph, neighbor_idx, state, result)?;
            }
            NodeState::Finished => {
                // Already processed, skip
            }
        }
    }

    state[node] = NodeState::Finished;

    #[allow(clippy::cast_possible_truncation)]
    result.push(NodeId(node as u32));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph_not_cyclic() {
        let graph = CsrGraph::new();
        assert!(!is_cyclic(&graph));
    }

    #[test]
    fn test_empty_graph_toposort() {
        let graph = CsrGraph::new();
        let order = toposort(&graph).unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn test_single_node_not_cyclic() {
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        // Node 0 exists with edge to node 1
        assert!(!is_cyclic(&graph));
    }

    #[test]
    fn test_self_loop_is_cyclic() {
        let edges = vec![(NodeId(0), NodeId(0), 1.0)]; // Self-loop
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        assert!(is_cyclic(&graph));
    }

    #[test]
    fn test_simple_dag_not_cyclic() {
        // 0 → 1 → 2
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(1), NodeId(2), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        assert!(!is_cyclic(&graph));
    }

    #[test]
    fn test_simple_cycle() {
        // 0 → 1 → 2 → 0 (cycle)
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(2), 1.0),
            (NodeId(2), NodeId(0), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        assert!(is_cyclic(&graph));
    }

    #[test]
    fn test_diamond_dag_not_cyclic() {
        // Diamond: 0 → 1 → 3, 0 → 2 → 3
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(0), NodeId(2), 1.0),
            (NodeId(1), NodeId(3), 1.0),
            (NodeId(2), NodeId(3), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        assert!(!is_cyclic(&graph));
    }

    #[test]
    fn test_toposort_simple_chain() {
        // 0 → 1 → 2
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(1), NodeId(2), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let order = toposort(&graph).unwrap();
        assert_eq!(order, vec![NodeId(0), NodeId(1), NodeId(2)]);
    }

    #[test]
    fn test_toposort_diamond() {
        // Diamond: 0 → 1 → 3, 0 → 2 → 3
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(0), NodeId(2), 1.0),
            (NodeId(1), NodeId(3), 1.0),
            (NodeId(2), NodeId(3), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let order = toposort(&graph).unwrap();

        // Verify constraints
        let pos = |n: u32| order.iter().position(|&x| x == NodeId(n)).unwrap();

        assert!(pos(0) < pos(1), "0 must come before 1");
        assert!(pos(0) < pos(2), "0 must come before 2");
        assert!(pos(1) < pos(3), "1 must come before 3");
        assert!(pos(2) < pos(3), "2 must come before 3");
    }

    #[test]
    fn test_toposort_fails_on_cycle() {
        // 0 → 1 → 2 → 0 (cycle)
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(2), 1.0),
            (NodeId(2), NodeId(0), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let result = toposort(&graph);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cycle"));
    }

    #[test]
    fn test_disconnected_components() {
        // Two disconnected chains: 0 → 1, 2 → 3
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(2), NodeId(3), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        assert!(!is_cyclic(&graph));

        let order = toposort(&graph).unwrap();
        assert_eq!(order.len(), 4);

        // Verify ordering constraints
        let pos = |n: u32| order.iter().position(|&x| x == NodeId(n)).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(2) < pos(3));
    }

    #[test]
    fn test_complex_dag() {
        // Complex DAG with multiple paths
        //     0
        //    /|\
        //   1 2 3
        //    \|/
        //     4
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(0), NodeId(2), 1.0),
            (NodeId(0), NodeId(3), 1.0),
            (NodeId(1), NodeId(4), 1.0),
            (NodeId(2), NodeId(4), 1.0),
            (NodeId(3), NodeId(4), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        assert!(!is_cyclic(&graph));

        let order = toposort(&graph).unwrap();

        // 0 must be first, 4 must be last
        assert_eq!(order[0], NodeId(0));
        assert_eq!(order[4], NodeId(4));
    }

    #[test]
    fn test_two_node_cycle() {
        // Simple 2-node cycle: 0 ↔ 1
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(1), NodeId(0), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        assert!(is_cyclic(&graph));
    }

    #[test]
    fn test_cycle_in_subgraph() {
        // DAG with cycle in subgraph: 0 → 1 → 2 → 1 (cycle), 3 → 4
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(2), 1.0),
            (NodeId(2), NodeId(1), 1.0), // Creates cycle
            (NodeId(3), NodeId(4), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        assert!(is_cyclic(&graph));
    }
}
