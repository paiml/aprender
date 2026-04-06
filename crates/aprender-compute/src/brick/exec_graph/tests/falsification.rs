//! BrickStats cycles and falsification tests.

use super::super::*;

#[test]
fn test_brick_stats_cycles() {
    let mut stats = BrickStats::new("test");
    stats.add_sample_with_cycles(1000, 100, 3000);

    assert_eq!(stats.total_cycles, 3000);
    assert!((stats.cycles_per_element() - 30.0).abs() < 0.001);
}

// FALSIFICATION TESTS

/// FALSIFICATION TEST: BrickId round-trip via name
#[test]
fn test_falsify_brick_id_round_trip() {
    for brick_id in [
        BrickId::RmsNorm,
        BrickId::LayerNorm,
        BrickId::QkvProjection,
        BrickId::RopeEmbedding,
        BrickId::AttentionScore,
        BrickId::AttentionSoftmax,
        BrickId::AttentionOutput,
        BrickId::OutputProjection,
        BrickId::GateProjection,
        BrickId::UpProjection,
        BrickId::Activation,
        BrickId::DownProjection,
        BrickId::Embedding,
        BrickId::LmHead,
        BrickId::Sampling,
    ] {
        let name = brick_id.name();
        let parsed = BrickId::from_str(name);
        assert_eq!(
            parsed,
            Some(brick_id),
            "FALSIFICATION FAILED: BrickId::{:?}.name() = {:?} does not round-trip",
            brick_id,
            name
        );
    }
}

/// FALSIFICATION TEST: ExecutionGraph maintains node/edge count consistency
#[test]
fn test_falsify_graph_consistency() {
    let mut graph = ExecutionGraph::new();

    // Add nodes and edges
    let n1 = graph.add_node(ExecutionNode::Layer { index: 0 });
    let n2 = graph.add_node(ExecutionNode::Layer { index: 1 });
    graph.add_edge(n1, n2, EdgeType::Sequence);

    assert_eq!(graph.num_nodes(), 2, "FALSIFICATION FAILED: node count mismatch");
    assert_eq!(graph.num_edges(), 1, "FALSIFICATION FAILED: edge count mismatch");

    // Clear and verify
    graph.clear();
    assert_eq!(graph.num_nodes(), 0, "FALSIFICATION FAILED: clear did not reset nodes");
    assert_eq!(graph.num_edges(), 0, "FALSIFICATION FAILED: clear did not reset edges");
}

/// FALSIFICATION TEST: BrickStats min/max tracking
#[test]
fn test_falsify_brick_stats_minmax() {
    let mut stats = BrickStats::new("test");

    for ns in [1000u64, 500, 2000, 750, 1500] {
        stats.add_sample(ns, 100);
    }

    assert_eq!(
        stats.min_ns, 500,
        "FALSIFICATION FAILED: min_ns should be 500, got {}",
        stats.min_ns
    );
    assert_eq!(
        stats.max_ns, 2000,
        "FALSIFICATION FAILED: max_ns should be 2000, got {}",
        stats.max_ns
    );
}
