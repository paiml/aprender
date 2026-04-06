//! BrickId, ExecutionGraph basics, BrickStats, CategoryStats, PtxRegistry, and Transfer tests.

use super::super::*;

#[test]
fn test_brick_id_category() {
    assert_eq!(BrickId::RmsNorm.category(), BrickCategory::Norm);
    assert_eq!(BrickId::LayerNorm.category(), BrickCategory::Norm);
    assert_eq!(BrickId::QkvProjection.category(), BrickCategory::Attention);
    assert_eq!(BrickId::GateProjection.category(), BrickCategory::Ffn);
    assert_eq!(BrickId::Embedding.category(), BrickCategory::Other);
}

#[test]
fn test_brick_id_name() {
    assert_eq!(BrickId::RmsNorm.name(), "RmsNorm");
    assert_eq!(BrickId::QkvProjection.name(), "QkvProjection");
}

#[test]
fn test_brick_id_from_str() {
    assert_eq!(BrickId::from_str("RmsNorm"), Some(BrickId::RmsNorm));
    assert_eq!(BrickId::from_str("Qkv"), Some(BrickId::QkvProjection));
    assert_eq!(BrickId::from_str("RoPE"), Some(BrickId::RopeEmbedding));
    assert_eq!(BrickId::from_str("Unknown"), None);
}

#[test]
fn test_brick_id_display() {
    assert_eq!(format!("{}", BrickId::RmsNorm), "RmsNorm");
}

#[test]
fn test_brick_category_name() {
    assert_eq!(BrickCategory::Norm.name(), "Norm");
    assert_eq!(BrickCategory::Ffn.name(), "FFN");
}

#[test]
fn test_brick_bottleneck_display() {
    assert_eq!(format!("{}", BrickBottleneck::Memory), "memory");
    assert_eq!(format!("{}", BrickBottleneck::Compute), "compute");
}

#[test]
fn test_execution_graph_basic() {
    let mut graph = ExecutionGraph::new();
    let layer = graph.add_node(ExecutionNode::Layer { index: 0 });
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 4096,
    });
    graph.add_edge(layer, brick, EdgeType::Contains);

    assert_eq!(graph.num_nodes(), 2);
    assert_eq!(graph.num_edges(), 1);
}

#[test]
fn test_execution_node_name() {
    let brick = ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 1000, elements: 4096 };
    assert_eq!(brick.name(), "RmsNorm");

    let layer = ExecutionNode::Layer { index: 5 };
    assert_eq!(layer.name(), "Layer5");
}

#[test]
fn test_execution_graph_scopes() {
    let mut graph = ExecutionGraph::new();
    let layer = graph.push_scope(ExecutionNode::Layer { index: 0 });
    let brick = graph.add_node_in_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 4096,
    });
    graph.pop_scope();

    assert_eq!(graph.num_nodes(), 2);
    // Should have a Contains edge from layer to brick
    let edges: Vec<_> = graph.outgoing_edges(layer).collect();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].dst, brick);
}

#[test]
fn test_brick_stats_basic() {
    let mut stats = BrickStats::new("test_brick");
    stats.add_sample(1000, 100);
    stats.add_sample(2000, 200);

    assert_eq!(stats.count, 2);
    assert_eq!(stats.total_ns, 3000);
    assert_eq!(stats.total_elements, 300);
    assert_eq!(stats.min_ns, 1000);
    assert_eq!(stats.max_ns, 2000);
}

#[test]
fn test_category_stats_percentage() {
    let stats = CategoryStats { total_ns: 250, total_elements: 1000, count: 10 };
    assert!((stats.percentage(1000) - 25.0).abs() < 0.001);
}

#[test]
fn test_ptx_registry() {
    let mut registry = PtxRegistry::new();
    registry.register("test_kernel", ".version 8.0\n.entry test {}", None);

    assert_eq!(registry.len(), 1);
    assert!(!registry.is_empty());

    let hash = PtxRegistry::hash_ptx(".version 8.0\n.entry test {}");
    assert_eq!(registry.lookup_name(hash), Some("test_kernel"));
}

#[test]
fn test_transfer_direction() {
    let node = ExecutionNode::Transfer {
        src: "host".to_string(),
        dst: "device".to_string(),
        bytes: 1024,
        direction: TransferDirection::H2D,
        timing_ns: Some(100),
    };
    assert!(node.is_transfer());
    assert_eq!(node.transfer_bytes(), Some(1024));
}

#[test]
fn test_execution_graph_to_dot() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Layer { index: 0 });
    let dot = graph.to_dot();
    assert!(dot.contains("digraph ExecutionGraph"));
    assert!(dot.contains("Layer 0"));
}

#[test]
fn test_execution_graph_to_ascii_tree() {
    let mut graph = ExecutionGraph::new();
    graph.push_scope(ExecutionNode::Layer { index: 0 });
    graph.add_node_in_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 4096,
    });
    graph.pop_scope();

    let tree = graph.to_ascii_tree();
    assert!(tree.contains("Layer 0"));
    assert!(tree.contains("RmsNorm"));
}
