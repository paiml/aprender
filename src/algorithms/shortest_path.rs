//! Shortest path algorithms: Dijkstra's algorithm
//!
//! Provides shortest path computation for weighted graphs:
//! - `dijkstra`: Single-source shortest paths with non-negative weights
//! - `dijkstra_path`: Shortest path between two specific nodes
//!
//! # Example
//!
//! ```
//! use trueno_graph::{CsrGraph, NodeId, dijkstra};
//!
//! // Build a weighted graph
//! let edges = vec![
//!     (NodeId(0), NodeId(1), 1.0),
//!     (NodeId(1), NodeId(2), 2.0),
//!     (NodeId(0), NodeId(2), 5.0),
//! ];
//! let graph = CsrGraph::from_edge_list(&edges).unwrap();
//!
//! // Find shortest paths from node 0
//! let distances = dijkstra(&graph, NodeId(0));
//! assert_eq!(distances.get(&NodeId(0)), Some(&0.0));
//! assert_eq!(distances.get(&NodeId(1)), Some(&1.0));
//! assert_eq!(distances.get(&NodeId(2)), Some(&3.0)); // 0→1→2 = 3.0, not 0→2 = 5.0
//! ```

use crate::storage::CsrGraph;
use crate::NodeId;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// State for Dijkstra's priority queue
#[derive(Clone, Copy)]
struct State {
    cost: f32,
    node: usize,
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (BinaryHeap is max-heap by default)
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Compute single-source shortest paths using Dijkstra's algorithm
///
/// Finds the shortest path from the source node to all reachable nodes.
/// Edge weights are used as distances (must be non-negative).
///
/// # Arguments
///
/// * `graph` - The CSR graph with edge weights as distances
/// * `source` - The starting node
///
/// # Returns
///
/// A `HashMap` mapping each reachable `NodeId` to its shortest distance from source.
/// Unreachable nodes are not included in the map.
///
/// # Complexity
///
/// O((V + E) log V) using a binary heap
///
/// # Example
///
/// ```
/// use trueno_graph::{CsrGraph, NodeId, dijkstra};
///
/// let edges = vec![
///     (NodeId(0), NodeId(1), 4.0),
///     (NodeId(0), NodeId(2), 1.0),
///     (NodeId(2), NodeId(1), 2.0),
/// ];
/// let graph = CsrGraph::from_edge_list(&edges).unwrap();
///
/// let distances = dijkstra(&graph, NodeId(0));
/// // Shortest to node 1: 0→2→1 = 3.0 (not 0→1 = 4.0)
/// assert_eq!(distances.get(&NodeId(1)), Some(&3.0));
/// ```
#[must_use]
pub fn dijkstra(graph: &CsrGraph, source: NodeId) -> HashMap<NodeId, f32> {
    let n = graph.num_nodes();
    if n == 0 {
        return HashMap::new();
    }

    let source_idx = source.0 as usize;
    if source_idx >= n {
        return HashMap::new();
    }

    let mut distances: HashMap<NodeId, f32> = HashMap::new();
    let mut heap = BinaryHeap::new();

    // Start with source
    distances.insert(source, 0.0);
    heap.push(State { cost: 0.0, node: source_idx });

    while let Some(State { cost, node }) = heap.pop() {
        #[allow(clippy::cast_possible_truncation)]
        let node_id = NodeId(node as u32);

        // Skip if we've found a better path
        if let Some(&d) = distances.get(&node_id) {
            if cost > d {
                continue;
            }
        }

        // Explore neighbors using adjacency (targets, weights)
        let (neighbors, weights) = graph.adjacency(node_id);
        for (i, &neighbor) in neighbors.iter().enumerate() {
            let neighbor_id = NodeId(neighbor);
            let weight = weights.get(i).copied().unwrap_or(1.0);

            let next_cost = cost + weight;

            // Update if this path is shorter
            let is_shorter = distances.get(&neighbor_id).map_or(true, |&d| next_cost < d);

            if is_shorter {
                distances.insert(neighbor_id, next_cost);
                heap.push(State { cost: next_cost, node: neighbor as usize });
            }
        }
    }

    distances
}

/// Find the shortest path between two nodes
///
/// Returns the distance and the path (sequence of nodes) from source to target.
///
/// # Arguments
///
/// * `graph` - The CSR graph with edge weights as distances
/// * `source` - The starting node
/// * `target` - The destination node
///
/// # Returns
///
/// * `Some((distance, path))` if a path exists
/// * `None` if target is unreachable from source
///
/// # Example
///
/// ```
/// use trueno_graph::{CsrGraph, NodeId, dijkstra_path};
///
/// let edges = vec![
///     (NodeId(0), NodeId(1), 1.0),
///     (NodeId(1), NodeId(2), 2.0),
/// ];
/// let graph = CsrGraph::from_edge_list(&edges).unwrap();
///
/// let result = dijkstra_path(&graph, NodeId(0), NodeId(2));
/// assert!(result.is_some());
/// let (dist, path) = result.unwrap();
/// assert_eq!(dist, 3.0);
/// assert_eq!(path, vec![NodeId(0), NodeId(1), NodeId(2)]);
/// ```
#[must_use]
/// Reconstruct the shortest path from predecessors map
fn reconstruct_path(
    predecessors: &HashMap<NodeId, NodeId>,
    target: NodeId,
) -> Vec<NodeId> {
    let mut path = vec![target];
    let mut current = target;
    while let Some(&pred) = predecessors.get(&current) {
        path.push(pred);
        current = pred;
    }
    path.reverse();
    path
}

pub fn dijkstra_path(
    graph: &CsrGraph,
    source: NodeId,
    target: NodeId,
) -> Option<(f32, Vec<NodeId>)> {
    let n = graph.num_nodes();
    if n == 0 || source.0 as usize >= n || target.0 as usize >= n {
        return None;
    }

    if source == target {
        return Some((0.0, vec![source]));
    }

    let mut distances: HashMap<NodeId, f32> = HashMap::new();
    let mut predecessors: HashMap<NodeId, NodeId> = HashMap::new();
    let mut heap = BinaryHeap::new();

    distances.insert(source, 0.0);
    heap.push(State { cost: 0.0, node: source.0 as usize });

    while let Some(State { cost, node }) = heap.pop() {
        #[allow(clippy::cast_possible_truncation)]
        let node_id = NodeId(node as u32);

        if node_id == target {
            let path = reconstruct_path(&predecessors, target);
            return Some((cost, path));
        }

        if distances.get(&node_id).map_or(false, |&d| cost > d) {
            continue;
        }

        let (neighbors, weights) = graph.adjacency(node_id);
        for (i, &neighbor) in neighbors.iter().enumerate() {
            let neighbor_id = NodeId(neighbor);
            let weight = weights.get(i).copied().unwrap_or(1.0);
            let next_cost = cost + weight;

            if distances.get(&neighbor_id).map_or(true, |&d| next_cost < d) {
                distances.insert(neighbor_id, next_cost);
                predecessors.insert(neighbor_id, node_id);
                heap.push(State { cost: next_cost, node: neighbor as usize });
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let graph = CsrGraph::new();
        let distances = dijkstra(&graph, NodeId(0));
        assert!(distances.is_empty());
    }

    #[test]
    fn test_single_edge() {
        let edges = vec![(NodeId(0), NodeId(1), 5.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let distances = dijkstra(&graph, NodeId(0));
        assert_eq!(distances.get(&NodeId(0)), Some(&0.0));
        assert_eq!(distances.get(&NodeId(1)), Some(&5.0));
    }

    #[test]
    fn test_chain() {
        // 0 --1.0--> 1 --2.0--> 2
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(1), NodeId(2), 2.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let distances = dijkstra(&graph, NodeId(0));
        assert_eq!(distances.get(&NodeId(0)), Some(&0.0));
        assert_eq!(distances.get(&NodeId(1)), Some(&1.0));
        assert_eq!(distances.get(&NodeId(2)), Some(&3.0));
    }

    #[test]
    fn test_shorter_path_via_intermediate() {
        // Direct: 0 --5.0--> 2
        // Via 1:  0 --1.0--> 1 --2.0--> 2 (total: 3.0)
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(2), 2.0),
            (NodeId(0), NodeId(2), 5.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let distances = dijkstra(&graph, NodeId(0));
        assert_eq!(distances.get(&NodeId(2)), Some(&3.0)); // Not 5.0
    }

    #[test]
    fn test_unreachable_node() {
        // 0 → 1, 2 → 3 (disconnected)
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(2), NodeId(3), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let distances = dijkstra(&graph, NodeId(0));
        assert_eq!(distances.get(&NodeId(0)), Some(&0.0));
        assert_eq!(distances.get(&NodeId(1)), Some(&1.0));
        assert!(distances.get(&NodeId(2)).is_none());
        assert!(distances.get(&NodeId(3)).is_none());
    }

    #[test]
    fn test_dijkstra_path_simple() {
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(1), NodeId(2), 2.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let result = dijkstra_path(&graph, NodeId(0), NodeId(2));
        assert!(result.is_some());
        let (dist, path) = result.unwrap();
        assert_eq!(dist, 3.0);
        assert_eq!(path, vec![NodeId(0), NodeId(1), NodeId(2)]);
    }

    #[test]
    fn test_dijkstra_path_same_node() {
        let edges = vec![(NodeId(0), NodeId(1), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let result = dijkstra_path(&graph, NodeId(0), NodeId(0));
        assert!(result.is_some());
        let (dist, path) = result.unwrap();
        assert_eq!(dist, 0.0);
        assert_eq!(path, vec![NodeId(0)]);
    }

    #[test]
    fn test_dijkstra_path_unreachable() {
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(2), NodeId(3), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let result = dijkstra_path(&graph, NodeId(0), NodeId(3));
        assert!(result.is_none());
    }

    #[test]
    fn test_dijkstra_path_chooses_shorter() {
        // 0 --5.0--> 2 (direct)
        // 0 --1.0--> 1 --2.0--> 2 (via 1, total 3.0)
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(2), 2.0),
            (NodeId(0), NodeId(2), 5.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let result = dijkstra_path(&graph, NodeId(0), NodeId(2));
        assert!(result.is_some());
        let (dist, path) = result.unwrap();
        assert_eq!(dist, 3.0);
        assert_eq!(path, vec![NodeId(0), NodeId(1), NodeId(2)]);
    }

    #[test]
    fn test_diamond_shortest_path() {
        // Diamond with different weights
        //     1 (weight 1)
        //    / \
        //   0   3  (1→3: 1, 2→3: 5)
        //    \ /
        //     2 (weight 2)
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(0), NodeId(2), 2.0),
            (NodeId(1), NodeId(3), 1.0),
            (NodeId(2), NodeId(3), 5.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let result = dijkstra_path(&graph, NodeId(0), NodeId(3));
        assert!(result.is_some());
        let (dist, path) = result.unwrap();
        assert_eq!(dist, 2.0); // 0→1→3
        assert_eq!(path, vec![NodeId(0), NodeId(1), NodeId(3)]);
    }

    #[test]
    fn test_cycle_in_graph() {
        // Cycle: 0 → 1 → 2 → 0, with 0 → 3
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(2), 1.0),
            (NodeId(2), NodeId(0), 1.0),
            (NodeId(0), NodeId(3), 10.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let distances = dijkstra(&graph, NodeId(0));
        assert_eq!(distances.get(&NodeId(0)), Some(&0.0));
        assert_eq!(distances.get(&NodeId(1)), Some(&1.0));
        assert_eq!(distances.get(&NodeId(2)), Some(&2.0));
        assert_eq!(distances.get(&NodeId(3)), Some(&10.0));
    }

    #[test]
    fn test_source_out_of_bounds() {
        let edges = vec![(NodeId(0), NodeId(1), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let distances = dijkstra(&graph, NodeId(100));
        assert!(distances.is_empty());
    }

    #[test]
    fn test_zero_weight_edge() {
        let edges = vec![(NodeId(0), NodeId(1), 0.0), (NodeId(1), NodeId(2), 0.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let distances = dijkstra(&graph, NodeId(0));
        assert_eq!(distances.get(&NodeId(2)), Some(&0.0));
    }
}
