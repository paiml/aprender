//! Provenance Graph Property Tests

use crate::monitor::inference::{CausalRelation, ProvenanceEdge, ProvenanceGraph, ProvenanceNode};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_provenance_graph_node_ids_unique(n_nodes in 1..50usize) {
        let mut graph = ProvenanceGraph::new();
        let mut ids = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            let id = graph.add_node(ProvenanceNode::Input {
                source: format!("source_{i}"),
                timestamp_ns: i as u64 * 1000,
                hash: i as u64,
            });
            ids.push(id);
        }

        // All IDs should be unique
        let unique_ids: std::collections::HashSet<_> = ids.iter().collect();
        prop_assert_eq!(unique_ids.len(), ids.len(), "Node IDs not unique");
    }

    #[test]
    fn prop_provenance_graph_edge_consistency(n_nodes in 2..20usize) {
        let mut graph = ProvenanceGraph::new();
        let mut ids = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            let id = graph.add_node(ProvenanceNode::Input {
                source: format!("source_{i}"),
                timestamp_ns: i as u64 * 1000,
                hash: i as u64,
            });
            ids.push(id);
        }

        // Add edges in a chain
        for i in 1..n_nodes {
            graph.add_edge(ProvenanceEdge {
                from: ids[i-1],
                to: ids[i],
                relation: CausalRelation::DataFlow,
                timestamp_ns: i as u64 * 1000,
            });
        }

        // Check adjacency consistency
        for i in 1..n_nodes {
            let preds = graph.predecessors(ids[i]);
            prop_assert!(preds.contains(&ids[i-1]), "Missing predecessor at {}", i);

            let succs = graph.successors(ids[i-1]);
            prop_assert!(succs.contains(&ids[i]), "Missing successor at {}", i-1);
        }
    }
}
