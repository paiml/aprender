//! Property-based tests for trueno-graph
//!
//! Verifies CSR invariants hold for arbitrary graphs

use proptest::prelude::*;
use trueno_graph::{CsrGraph, NodeId};

// Property: from_edge_list should produce valid CSR structure
proptest! {
    #[test]
    fn prop_from_edge_list_valid_csr(edges in prop_edge_list(0usize..100usize, 0u32..50u32)) {
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        // Invariant 1: row_offsets is monotonically increasing
        for i in 0..graph.csr_components().0.len() - 1 {
            let (row_offsets, _, _) = graph.csr_components();
            prop_assert!(row_offsets[i] <= row_offsets[i + 1]);
        }

        // Invariant 2: last row_offset == num_edges
        let (row_offsets, col_indices, _) = graph.csr_components();
        prop_assert_eq!(*row_offsets.last().unwrap() as usize, col_indices.len());

        // Invariant 3: col_indices and edge_weights have same length
        let (_, col_indices, edge_weights) = graph.csr_components();
        prop_assert_eq!(col_indices.len(), edge_weights.len());

        // Invariant 4: num_edges matches edge count
        prop_assert_eq!(graph.num_edges(), edges.len());
    }
}

// Property: outgoing_neighbors should return correct edges
proptest! {
    #[test]
    fn prop_outgoing_neighbors_correct(edges in prop_edge_list(0usize..100usize, 0u32..20u32)) {
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        for node_id in 0..graph.num_nodes() {
            let neighbors = graph.outgoing_neighbors(NodeId(node_id as u32)).unwrap();

            // Count expected outgoing edges
            let expected: Vec<_> = edges.iter()
                .filter(|(src, _, _)| src.0 == node_id as u32)
                .map(|(_, dst, _)| dst.0)
                .collect();

            prop_assert_eq!(neighbors.len(), expected.len());

            // All neighbors should be in expected set
            for neighbor in neighbors {
                prop_assert!(expected.contains(neighbor));
            }
        }
    }
}

// Property: incoming_neighbors should return correct callers
proptest! {
    #[test]
    fn prop_incoming_neighbors_correct(edges in prop_edge_list(0usize..100usize, 0u32..20u32)) {
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        for node_id in 0..graph.num_nodes() {
            let callers = graph.incoming_neighbors(NodeId(node_id as u32)).unwrap();

            // Count expected incoming edges (including duplicates/multi-edges)
            let mut expected: Vec<_> = edges.iter()
                .filter(|(_, dst, _)| dst.0 == node_id as u32)
                .map(|(src, _, _)| src.0)
                .collect();
            expected.sort_unstable();

            prop_assert_eq!(callers.len(), expected.len(),
                "Node {}: callers.len()={}, expected.len()={}",
                node_id, callers.len(), expected.len());

            // Convert callers to sorted vec for comparison
            let mut callers_sorted = callers.to_vec();
            callers_sorted.sort_unstable();

            prop_assert_eq!(callers_sorted, expected);
        }
    }
}

// Property: Parquet roundtrip preserves graph structure
proptest! {
    #[test]
    fn prop_parquet_roundtrip(edges in prop_edge_list(0usize..100usize, 0u32..20u32)) {
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime.block_on(async {
            let graph = CsrGraph::from_edge_list(&edges).unwrap();

            // Write to temp file
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("prop_test_graph");

            graph.write_parquet(&path).await.unwrap();

            // Read back
            let loaded = CsrGraph::read_parquet(&path).await.unwrap();

            // Verify structure preserved
            prop_assert_eq!(loaded.num_nodes(), graph.num_nodes());
            prop_assert_eq!(loaded.num_edges(), graph.num_edges());

            // Verify CSR components match
            let (row1, col1, weights1) = graph.csr_components();
            let (row2, col2, weights2) = loaded.csr_components();

            prop_assert_eq!(row1, row2);
            prop_assert_eq!(col1, col2);
            prop_assert_eq!(weights1, weights2);

            Ok(())
        })?;
    }
}

// Property: add_edge increases edge count by 1
proptest! {
    #[test]
    fn prop_add_edge_increases_count(
        src in 0u32..50,
        dst in 0u32..50,
        weight in 0.0f32..100.0
    ) {
        let mut graph = CsrGraph::new();
        let initial_count = graph.num_edges();

        graph.add_edge(NodeId(src), NodeId(dst), weight).unwrap();

        prop_assert_eq!(graph.num_edges(), initial_count + 1);
    }
}

// Property: node_count is never negative and grows with edges
proptest! {
    #[test]
    fn prop_node_count_grows(edges in prop_edge_list(0usize..100usize, 0u32..50u32)) {
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        // Node count should be >= max node ID + 1
        if let Some(max_node) = edges.iter()
            .flat_map(|(src, dst, _)| [src.0, dst.0])
            .max()
        {
            prop_assert!(graph.num_nodes() >= (max_node + 1) as usize);
        }

        // Node count is always non-negative (usize), just verify it's a valid value
        let _ = graph.num_nodes(); // No assertion needed - type guarantees >= 0
    }
}

// Helper: Generate arbitrary edge list
fn prop_edge_list(
    num_edges: impl Strategy<Value = usize>,
    max_node: impl Strategy<Value = u32>,
) -> impl Strategy<Value = Vec<(NodeId, NodeId, f32)>> {
    (num_edges, max_node).prop_flat_map(|(n, max_node)| {
        // Ensure max_node is at least 1 to avoid empty range
        let max_node = max_node.max(1);
        prop::collection::vec(
            (0..max_node, 0..max_node, 0.0..100.0f32)
                .prop_map(|(src, dst, weight)| (NodeId(src), NodeId(dst), weight)),
            0..=n,
        )
    })
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_empty_graph_invariants() {
        let graph = CsrGraph::new();

        // Empty graph should have valid CSR structure
        let (row_offsets, col_indices, edge_weights) = graph.csr_components();

        assert_eq!(row_offsets, &[0]); // Single offset for empty graph
        assert_eq!(col_indices.len(), 0);
        assert_eq!(edge_weights.len(), 0);
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_single_edge_invariants() {
        let edges = vec![(NodeId(0), NodeId(1), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let (row_offsets, col_indices, edge_weights) = graph.csr_components();

        // Should have 2 nodes (0 and 1)
        assert_eq!(graph.num_nodes(), 2);

        // row_offsets: [0, 1, 1] (node 0 has 1 edge, node 1 has 0 edges)
        assert_eq!(row_offsets, &[0, 1, 1]);

        // col_indices: [1] (edge 0→1)
        assert_eq!(col_indices, &[1]);

        // edge_weights: [1.0]
        assert_eq!(edge_weights, &[1.0]);
    }
}
