//! Graph traversal algorithms (BFS, DFS, `find_callers`)
//!
//! Based on Ligra (Shun & Blelloch, `PPoPP` 2013) frontier-based traversal patterns.

use crate::storage::CsrGraph;
use crate::NodeId;
use anyhow::Result;
use std::collections::{HashSet, VecDeque};

/// Find all functions that transitively call the target function
///
/// Uses reverse BFS from target node to find all callers within specified depth.
///
/// # Arguments
///
/// * `graph` - CSR graph representation
/// * `target` - Target node to find callers for
/// * `max_depth` - Maximum traversal depth (0 = direct callers only)
///
/// # Returns
///
/// Set of all node IDs that call the target (directly or transitively)
///
/// # Errors
///
/// Returns an error if graph operations fail (e.g., invalid node access).
///
/// # Example
///
/// ```
/// use trueno_graph::{CsrGraph, NodeId, find_callers};
///
/// let mut graph = CsrGraph::new();
/// graph.add_edge(NodeId(0), NodeId(2), 1.0).unwrap(); // main → validate
/// graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap(); // parse → validate
///
/// let callers = find_callers(&graph, NodeId(2), 10).unwrap();
/// assert!(callers.contains(&0)); // main calls validate
/// assert!(callers.contains(&1)); // parse calls validate
/// ```
pub fn find_callers(graph: &CsrGraph, target: NodeId, max_depth: usize) -> Result<Vec<u32>> {
    let mut visited = HashSet::new();
    let mut frontier = VecDeque::new();
    let mut depth = 0;

    // Start BFS from target node
    frontier.push_back(target.0);
    visited.insert(target.0);

    while !frontier.is_empty() && depth < max_depth {
        let level_size = frontier.len();

        for _ in 0..level_size {
            if let Some(current) = frontier.pop_front() {
                // Find all incoming neighbors (callers)
                let callers = graph.incoming_neighbors(NodeId(current))?;

                for &caller in callers {
                    if !visited.contains(&caller) {
                        visited.insert(caller);
                        frontier.push_back(caller);
                    }
                }
            }
        }

        depth += 1;
    }

    // Remove target node from result (we want callers, not the target itself)
    visited.remove(&target.0);

    Ok(visited.into_iter().collect())
}

/// Breadth-First Search from source node
///
/// Standard BFS traversal returning all reachable nodes.
///
/// # Arguments
///
/// * `graph` - CSR graph representation
/// * `source` - Starting node for BFS
///
/// # Returns
///
/// Vec of all node IDs reachable from source
///
/// # Errors
///
/// Returns an error if graph operations fail (e.g., invalid node access).
///
/// # Example
///
/// ```
/// use trueno_graph::{CsrGraph, NodeId, bfs};
///
/// let mut graph = CsrGraph::new();
/// graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
/// graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();
///
/// let reachable = bfs(&graph, NodeId(0)).unwrap();
/// assert_eq!(reachable.len(), 3); // All 3 nodes reachable
/// ```
pub fn bfs(graph: &CsrGraph, source: NodeId) -> Result<Vec<u32>> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    queue.push_back(source.0);
    visited.insert(source.0);

    while let Some(current) = queue.pop_front() {
        // Get outgoing neighbors
        let neighbors = graph.outgoing_neighbors(NodeId(current))?;

        for &neighbor in neighbors {
            if !visited.contains(&neighbor) {
                visited.insert(neighbor);
                queue.push_back(neighbor);
            }
        }
    }

    Ok(visited.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_callers_direct() {
        // Graph: 0 → 2, 1 → 2 (both call 2)
        let edges = vec![(NodeId(0), NodeId(2), 1.0), (NodeId(1), NodeId(2), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let callers = find_callers(&graph, NodeId(2), 1).unwrap();
        assert_eq!(callers.len(), 2);
        assert!(callers.contains(&0));
        assert!(callers.contains(&1));
    }

    #[test]
    fn test_find_callers_transitive() {
        // Graph: 0 → 1 → 2 (0 transitively calls 2)
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(1), NodeId(2), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let callers = find_callers(&graph, NodeId(2), 10).unwrap();
        assert_eq!(callers.len(), 2);
        assert!(callers.contains(&0));
        assert!(callers.contains(&1));
    }

    #[test]
    fn test_find_callers_depth_limit() {
        // Graph: 0 → 1 → 2 → 3
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(2), 1.0),
            (NodeId(2), NodeId(3), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        // Depth 1: only node 2 calls node 3
        let callers = find_callers(&graph, NodeId(3), 1).unwrap();
        assert_eq!(callers.len(), 1);
        assert!(callers.contains(&2));

        // Depth 2: nodes 1 and 2 call node 3
        let callers = find_callers(&graph, NodeId(3), 2).unwrap();
        assert_eq!(callers.len(), 2);
        assert!(callers.contains(&1));
        assert!(callers.contains(&2));

        // Depth 10: all nodes call node 3
        let callers = find_callers(&graph, NodeId(3), 10).unwrap();
        assert_eq!(callers.len(), 3);
        assert!(callers.contains(&0));
        assert!(callers.contains(&1));
        assert!(callers.contains(&2));
    }

    #[test]
    fn test_bfs_simple() {
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(1), NodeId(2), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let reachable = bfs(&graph, NodeId(0)).unwrap();
        assert_eq!(reachable.len(), 3);
        assert!(reachable.contains(&0));
        assert!(reachable.contains(&1));
        assert!(reachable.contains(&2));
    }

    #[test]
    fn test_bfs_disconnected() {
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(2), NodeId(3), 1.0), // Disconnected component
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let reachable = bfs(&graph, NodeId(0)).unwrap();
        assert_eq!(reachable.len(), 2); // Only nodes 0 and 1
        assert!(reachable.contains(&0));
        assert!(reachable.contains(&1));
        assert!(!reachable.contains(&2)); // Not reachable
    }
}
