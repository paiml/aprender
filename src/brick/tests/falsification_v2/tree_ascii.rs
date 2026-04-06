use super::super::super::*;

// ========================================================================
// Tree node and ASCII tree tests (F121-F127)
// ========================================================================

/// F121: to_tree_node conversion produces correct hierarchy
#[test]
#[cfg(feature = "presentar-tui")]
fn test_f121_to_tree_node_hierarchy() {
    let mut graph = ExecutionGraph::new();

    // Build: Layer -> Brick -> Kernel
    let layer_id = graph.push_scope(ExecutionNode::Layer { index: 0 });
    graph.push_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 50_000,
        elements: 4096,
    });
    graph.record_kernel_launch("rmsnorm_kernel", 0x1234, (16, 1, 1), (256, 1, 1), 1024);
    graph.pop_scope(); // pop brick
    graph.pop_scope(); // pop layer

    let tree = graph.to_tree_node();

    // Root should be Layer 0 (single root)
    assert_eq!(tree.label, "Layer 0", "F121: Root is Layer");
    assert_eq!(tree.children.len(), 1, "F121: Layer has 1 child (brick)");

    let brick = &tree.children[0];
    assert_eq!(brick.label, "RmsNorm", "F121: Brick label");
    assert!(brick.info.as_ref().is_some_and(|i| i.contains("50.0µs")), "F121: Brick has timing");
    assert_eq!(brick.children.len(), 1, "F121: Brick has 1 child (kernel)");

    let kernel = &brick.children[0];
    assert_eq!(kernel.label, "rmsnorm_kernel", "F121: Kernel label");
    assert!(
        kernel.info.as_ref().is_some_and(|i| i.contains("smem=1024B")),
        "F121: Kernel has shared mem"
    );

    // Verify depth
    assert_eq!(tree.depth(), 3, "F121: Tree depth is 3 (layer->brick->kernel)");
    assert_eq!(tree.count_nodes(), 3, "F121: Tree has 3 nodes");

    let _ = layer_id;
}

/// F122: to_tree_node with multiple roots wraps in synthetic root
#[test]
#[cfg(feature = "presentar-tui")]
fn test_f122_to_tree_node_multiple_roots() {
    let mut graph = ExecutionGraph::new();

    // Two disjoint layers (no parent)
    graph.add_node(ExecutionNode::Layer { index: 0 });
    graph.add_node(ExecutionNode::Layer { index: 1 });

    let tree = graph.to_tree_node();

    // Should have synthetic "Execution Graph" root
    assert_eq!(tree.label, "Execution Graph", "F122: Synthetic root label");
    assert_eq!(tree.children.len(), 2, "F122: Two children (two layers)");
    assert_eq!(tree.children[0].label, "Layer 0", "F122: First child");
    assert_eq!(tree.children[1].label, "Layer 1", "F122: Second child");
}

/// F123: to_tree_node with empty graph
#[test]
#[cfg(feature = "presentar-tui")]
fn test_f123_to_tree_node_empty() {
    let graph = ExecutionGraph::new();
    let tree = graph.to_tree_node();

    assert_eq!(tree.label, "Empty Graph", "F123: Empty graph label");
    assert!(tree.children.is_empty(), "F123: No children");
}

/// F124: to_ascii_tree produces correct hierarchy (headless mode)
#[test]
fn test_f124_to_ascii_tree_hierarchy() {
    let mut graph = ExecutionGraph::new();

    // Build: Layer -> Brick -> Kernel
    graph.push_scope(ExecutionNode::Layer { index: 0 });
    graph.push_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 50_000,
        elements: 4096,
    });
    graph.record_kernel_launch("rmsnorm_kernel", 0x1234, (16, 1, 1), (256, 1, 1), 1024);
    graph.pop_scope(); // pop brick
    graph.pop_scope(); // pop layer

    let tree = graph.to_ascii_tree();

    // Verify structure
    assert!(tree.contains("Layer 0"), "F124: Contains Layer 0");
    assert!(tree.contains("RmsNorm"), "F124: Contains RmsNorm");
    assert!(tree.contains("50.0µs"), "F124: Contains timing");
    assert!(tree.contains("rmsnorm_kernel"), "F124: Contains kernel");
    assert!(tree.contains("smem=1024B"), "F124: Contains shared mem");

    // Verify tree structure characters
    assert!(tree.contains("├──") || tree.contains("└──"), "F124: Has tree connectors");
}

/// F125: to_ascii_tree with multiple roots
#[test]
fn test_f125_to_ascii_tree_multiple_roots() {
    let mut graph = ExecutionGraph::new();

    // Two disjoint layers (no parent)
    graph.add_node(ExecutionNode::Layer { index: 0 });
    graph.add_node(ExecutionNode::Layer { index: 1 });

    let tree = graph.to_ascii_tree();

    // Should have synthetic "Execution Graph" root
    assert!(tree.starts_with("Execution Graph"), "F125: Synthetic root");
    assert!(tree.contains("Layer 0"), "F125: Contains Layer 0");
    assert!(tree.contains("Layer 1"), "F125: Contains Layer 1");
}

/// F126: to_ascii_tree with empty graph
#[test]
fn test_f126_to_ascii_tree_empty() {
    let graph = ExecutionGraph::new();
    let tree = graph.to_ascii_tree();

    assert_eq!(tree, "(empty graph)", "F126: Empty graph output");
}

/// F127: to_ascii_tree snapshot stability (deterministic)
#[test]
fn test_f127_to_ascii_tree_snapshot() {
    let mut graph = ExecutionGraph::new();

    // Build a specific structure
    graph.push_scope(ExecutionNode::Layer { index: 0 });
    graph.push_scope(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 200_000,
        elements: 4096,
    });
    graph.record_kernel_launch("batched_gemv", 0xABCD, (32, 1, 1), (256, 1, 1), 4096);
    graph.pop_scope();
    graph.pop_scope();

    let tree = graph.to_ascii_tree();

    // Verify exact output (for snapshot testing)
    let expected = "\
Layer 0
└── QkvProjection  200.0µs (4096 elem)
    └── batched_gemv  <<<32,256,1>>> smem=4096B";

    assert_eq!(tree, expected, "F127: Snapshot matches expected output");
}
