//! `PageRank` algorithm via aprender sparse matrix operations
//!
//! Based on Page et al. (1999) "The `PageRank` Citation Ranking: Bringing Order to the Web"
//! Implementation uses power iteration via aprender's sparse matrix multiplication.

use crate::storage::CsrGraph;
use anyhow::Result;

/// Damping factor for `PageRank` (Google standard)
const DAMPING_FACTOR: f32 = 0.85;

/// Compute `PageRank` scores for all nodes in the graph
///
/// Uses power iteration algorithm with damping factor 0.85 (Google standard).
///
/// # Arguments
///
/// * `graph` - CSR graph representation
/// * `max_iterations` - Maximum number of power iterations (typically 20-50)
/// * `tolerance` - Convergence threshold (default: 1e-6)
///
/// # Returns
///
/// Vector of `PageRank` scores (one per node, sum = 1.0)
///
/// # Algorithm
///
/// `PageRank` formula:
/// ```text
/// PR(u) = (1-d)/N + d * Σ(PR(v) / outdegree(v))
/// ```
///
/// Where:
/// - d = 0.85 (damping factor)
/// - N = total number of nodes
/// - v = nodes with edges to u
///
/// # Errors
///
/// This function does not return errors in the current implementation.
/// The Result type is used for future extensibility (e.g., invalid graph states).
///
/// # Example
///
/// ```
/// use trueno_graph::{CsrGraph, NodeId, pagerank};
///
/// let mut graph = CsrGraph::new();
/// graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
/// graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();
/// graph.add_edge(NodeId(2), NodeId(0), 1.0).unwrap(); // Cycle
///
/// let scores = pagerank(&graph, 20, 1e-6).unwrap();
/// assert_eq!(scores.len(), 3);
/// assert!((scores.iter().sum::<f32>() - 1.0).abs() < 1e-5); // Sum = 1.0
/// ```
#[allow(clippy::cast_precision_loss)] // Graphs >16M nodes unlikely
#[allow(clippy::cast_possible_truncation)] // Graphs with >4B edges unlikely
pub fn pagerank(graph: &CsrGraph, max_iterations: usize, tolerance: f32) -> Result<Vec<f32>> {
    let n = graph.num_nodes();

    if n == 0 {
        return Ok(Vec::new());
    }

    let teleport = (1.0 - DAMPING_FACTOR) / n as f32;

    // Initialize: uniform distribution
    let mut ranks = vec![1.0 / n as f32; n];
    let mut new_ranks = vec![0.0; n];

    // Get CSR components for iteration
    let (row_offsets, col_indices, _edge_weights) = graph.csr_components();

    // Compute out-degrees for normalization
    let mut out_degrees = vec![0_u32; n];
    for node in 0..n {
        let start = row_offsets[node] as usize;
        let end = row_offsets[node + 1] as usize;
        out_degrees[node] = (end - start) as u32;
    }

    // Power iteration
    #[allow(unused_variables)] // iteration only used in test builds
    for iteration in 0..max_iterations {
        // Reset new ranks to teleport value
        new_ranks.fill(teleport);

        // Distribute rank from each node to its neighbors
        for node in 0..n {
            let start = row_offsets[node] as usize;
            let end = row_offsets[node + 1] as usize;

            if out_degrees[node] > 0 {
                let rank_contribution = DAMPING_FACTOR * ranks[node] / out_degrees[node] as f32;

                for &target in &col_indices[start..end] {
                    new_ranks[target as usize] += rank_contribution;
                }
            } else {
                // Dangling node: distribute rank equally to all nodes
                let dangling_contribution = DAMPING_FACTOR * ranks[node] / n as f32;
                for r in &mut new_ranks {
                    *r += dangling_contribution;
                }
            }
        }

        // Check convergence (L1 norm)
        let mut diff = 0.0;
        for i in 0..n {
            diff += (new_ranks[i] - ranks[i]).abs();
        }

        // Swap buffers
        std::mem::swap(&mut ranks, &mut new_ranks);

        if diff < tolerance {
            #[cfg(test)]
            eprintln!("PageRank converged after {} iterations (diff={:.2e})", iteration + 1, diff);
            break;
        }
    }

    Ok(ranks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeId;

    #[test]
    fn test_pagerank_simple_chain() {
        // Linear chain: 0 → 1 → 2
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(1), NodeId(2), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let scores = pagerank(&graph, 20, 1e-6).unwrap();

        // Verify properties
        assert_eq!(scores.len(), 3);

        // Sum should be 1.0
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Sum = {sum}");

        // In a chain, last node gets highest score (sink node)
        assert!(scores[2] > scores[1], "Node 2 should have higher score than 1");
        assert!(scores[1] > scores[0], "Node 1 should have higher score than 0");
    }

    #[test]
    fn test_pagerank_cycle() {
        // Cycle: 0 → 1 → 2 → 0
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(2), 1.0),
            (NodeId(2), NodeId(0), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let scores = pagerank(&graph, 50, 1e-6).unwrap();

        // In a symmetric cycle, all nodes should have equal rank
        assert_eq!(scores.len(), 3);

        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // All scores should be approximately 1/3
        for score in &scores {
            assert!((*score - 1.0 / 3.0).abs() < 0.01, "Score = {score}");
        }
    }

    #[test]
    fn test_pagerank_star() {
        // Star: 0 ← 1, 0 ← 2, 0 ← 3 (all point to center)
        let edges = vec![
            (NodeId(1), NodeId(0), 1.0),
            (NodeId(2), NodeId(0), 1.0),
            (NodeId(3), NodeId(0), 1.0),
        ];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let scores = pagerank(&graph, 20, 1e-6).unwrap();

        // Center node (0) should have highest score
        assert!(scores[0] > scores[1]);
        assert!(scores[0] > scores[2]);
        assert!(scores[0] > scores[3]);

        // Peripheral nodes should have similar scores
        assert!((scores[1] - scores[2]).abs() < 0.01);
        assert!((scores[2] - scores[3]).abs() < 0.01);
    }

    #[test]
    fn test_pagerank_empty_graph() {
        let graph = CsrGraph::new();
        let scores = pagerank(&graph, 20, 1e-6).unwrap();
        assert_eq!(scores.len(), 0);
    }

    #[test]
    fn test_pagerank_single_node() {
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(0), 1.0).unwrap(); // Self-loop

        let scores = pagerank(&graph, 20, 1e-6).unwrap();
        assert_eq!(scores.len(), 1);
        assert!((scores[0] - 1.0).abs() < 1e-5); // Single node gets all rank
    }

    #[test]
    fn test_pagerank_convergence() {
        // Large cycle to test convergence
        let mut edges = Vec::new();
        for i in 0..10 {
            edges.push((NodeId(i), NodeId((i + 1) % 10), 1.0));
        }
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let scores = pagerank(&graph, 100, 1e-6).unwrap();

        // Should converge to uniform distribution
        for score in &scores {
            assert!((*score - 0.1).abs() < 0.01, "Score = {score}");
        }
    }
}
