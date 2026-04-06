//! Comprehensive to_dot() and to_ascii_tree() coverage tests.

use super::super::*;

// ================================================================
// Comprehensive to_dot() coverage - all node types and edge types
// ================================================================

#[test]
fn test_to_dot_all_node_types() {
    let mut graph = ExecutionGraph::new();

    // Layer node
    graph.add_node(ExecutionNode::Layer { index: 0 });

    // Brick node
    graph.add_node(ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 5000, elements: 2048 });

    // Kernel node
    graph.add_node(ExecutionNode::Kernel {
        name: "batched_q4k_gemv".into(),
        ptx_hash: 0xDEADBEEF,
        grid: (32, 1, 1),
        block: (256, 1, 1),
        shared_mem: 4096,
        timing_ns: Some(1500),
        arithmetic_intensity: None,
        achieved_tflops: None,
    });

    // Function node with file/line
    graph.add_node(ExecutionNode::Function {
        name: "forward_pass".into(),
        file: Some("model.rs".into()),
        line: Some(42),
    });

    // Function node without file/line
    graph.add_node(ExecutionNode::Function { name: "dispatch".into(), file: None, line: None });

    // Transfer H2D
    graph.add_node(ExecutionNode::Transfer {
        src: "host".into(),
        dst: "gpu0".into(),
        bytes: 1_000_000,
        direction: TransferDirection::H2D,
        timing_ns: Some(500),
    });

    // Transfer D2H
    graph.add_node(ExecutionNode::Transfer {
        src: "gpu0".into(),
        dst: "host".into(),
        bytes: 2_000_000,
        direction: TransferDirection::D2H,
        timing_ns: None,
    });

    // Transfer D2D
    graph.add_node(ExecutionNode::Transfer {
        src: "gpu0".into(),
        dst: "gpu1".into(),
        bytes: 500_000,
        direction: TransferDirection::D2D,
        timing_ns: Some(200),
    });

    // AsyncTask
    graph.add_node(ExecutionNode::AsyncTask {
        name: "prefetch_weights".into(),
        poll_count: 5,
        yield_count: 3,
        total_poll_ns: 10_000,
    });

    // AsyncTask with zero polls
    graph.add_node(ExecutionNode::AsyncTask {
        name: "idle_task".into(),
        poll_count: 0,
        yield_count: 0,
        total_poll_ns: 0,
    });

    let dot = graph.to_dot();

    // Structure
    assert!(dot.starts_with("digraph ExecutionGraph {\n"));
    assert!(dot.ends_with("}\n"));
    assert!(dot.contains("rankdir=TB"));

    // Layer
    assert!(dot.contains("Layer 0"));
    assert!(dot.contains("fillcolor=lightblue"));

    // Brick
    assert!(dot.contains("RmsNorm"));
    assert!(dot.contains("fillcolor=lightgreen"));

    // Kernel
    assert!(dot.contains("batched_q4k_gemv"));
    assert!(dot.contains("<<<32,256,1>>>"));
    assert!(dot.contains("fillcolor=lightyellow"));

    // Function with location
    assert!(dot.contains("forward_pass"));
    assert!(dot.contains("model.rs:42"));
    assert!(dot.contains("fillcolor=lightgray"));

    // Function without location
    assert!(dot.contains("dispatch"));

    // Transfers
    assert!(dot.contains("H2D"));
    assert!(dot.contains("D2H"));
    assert!(dot.contains("D2D"));
    assert!(dot.contains("fillcolor=lightsalmon"));

    // AsyncTask
    assert!(dot.contains("prefetch_weights"));
    assert!(dot.contains("polls:5"));
    assert!(dot.contains("yields:3"));
    assert!(dot.contains("fillcolor=lightcyan"));
}

#[test]
fn test_to_dot_all_edge_types() {
    let mut graph = ExecutionGraph::new();

    let n0 = graph.add_node(ExecutionNode::Layer { index: 0 });
    let n1 = graph.add_node(ExecutionNode::Layer { index: 1 });
    let n2 = graph.add_node(ExecutionNode::Layer { index: 2 });
    let n3 = graph.add_node(ExecutionNode::Layer { index: 3 });
    let n4 = graph.add_node(ExecutionNode::Layer { index: 4 });
    let n5 = graph.add_node(ExecutionNode::Layer { index: 5 });

    graph.add_edge(n0, n1, EdgeType::Calls);
    graph.add_edge(n0, n2, EdgeType::Contains);
    graph.add_edge(n2, n3, EdgeType::Launches);
    graph.add_edge(n3, n4, EdgeType::Sequence);
    graph.add_edge(n4, n5, EdgeType::DependsOn);
    graph.add_edge(n1, n5, EdgeType::Transfer { bytes: 1024, direction: TransferDirection::H2D });

    let dot = graph.to_dot();

    assert!(dot.contains("style=solid"));
    assert!(dot.contains("style=dashed"));
    assert!(dot.contains("style=bold,color=red"));
    assert!(dot.contains("style=dotted"));
    assert!(dot.contains("style=solid,color=blue"));
    assert!(dot.contains("style=bold,color=orange"));
}

#[test]
fn test_to_dot_empty_graph() {
    let graph = ExecutionGraph::new();
    let dot = graph.to_dot();
    assert!(dot.contains("digraph ExecutionGraph"));
    assert!(dot.contains("}\n"));
    // No nodes or edges
    assert!(!dot.contains("n0"));
}

// ================================================================
// Comprehensive to_ascii_tree() coverage
// ================================================================

#[test]
fn test_to_ascii_tree_empty_graph() {
    let graph = ExecutionGraph::new();
    let tree = graph.to_ascii_tree();
    assert_eq!(tree, "(empty graph)");
}

#[test]
fn test_to_ascii_tree_single_node() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Layer { index: 7 });
    let tree = graph.to_ascii_tree();
    assert!(tree.contains("Layer 7"));
}

#[test]
fn test_to_ascii_tree_multiple_roots() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Layer { index: 0 });
    graph.add_node(ExecutionNode::Layer { index: 1 });
    graph.add_node(ExecutionNode::Layer { index: 2 });

    let tree = graph.to_ascii_tree();
    assert!(tree.contains("Execution Graph"));
    assert!(tree.contains("Layer 0"));
    assert!(tree.contains("Layer 1"));
    assert!(tree.contains("Layer 2"));
    // Should have tree connectors
    assert!(tree.contains("\u{251c}\u{2500}\u{2500}") || tree.contains("\u{2514}\u{2500}\u{2500}"));
}

#[test]
fn test_to_ascii_tree_all_node_types() {
    let mut graph = ExecutionGraph::new();

    let root = graph.add_node(ExecutionNode::Layer { index: 0 });

    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 3000,
        elements: 1024,
    });
    graph.add_edge(root, brick, EdgeType::Contains);

    let kernel = graph.add_node(ExecutionNode::Kernel {
        name: "gemv_kernel".into(),
        ptx_hash: 0x1234,
        grid: (4, 1, 1),
        block: (128, 1, 1),
        shared_mem: 2048,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(brick, kernel, EdgeType::Launches);

    let func = graph.add_node(ExecutionNode::Function {
        name: "compute".into(),
        file: Some("ops.rs".into()),
        line: Some(100),
    });
    graph.add_edge(root, func, EdgeType::Contains);

    let func_no_loc =
        graph.add_node(ExecutionNode::Function { name: "init".into(), file: None, line: None });
    graph.add_edge(root, func_no_loc, EdgeType::Contains);

    let transfer = graph.add_node(ExecutionNode::Transfer {
        src: "cpu".into(),
        dst: "gpu".into(),
        bytes: 8192,
        direction: TransferDirection::H2D,
        timing_ns: Some(250),
    });
    graph.add_edge(root, transfer, EdgeType::Contains);

    let transfer_no_timing = graph.add_node(ExecutionNode::Transfer {
        src: "gpu".into(),
        dst: "cpu".into(),
        bytes: 4096,
        direction: TransferDirection::D2H,
        timing_ns: None,
    });
    graph.add_edge(root, transfer_no_timing, EdgeType::Contains);

    let async_task = graph.add_node(ExecutionNode::AsyncTask {
        name: "prefetch".into(),
        poll_count: 3,
        yield_count: 1,
        total_poll_ns: 5000,
    });
    graph.add_edge(root, async_task, EdgeType::Contains);

    let async_zero = graph.add_node(ExecutionNode::AsyncTask {
        name: "idle".into(),
        poll_count: 0,
        yield_count: 0,
        total_poll_ns: 0,
    });
    graph.add_edge(root, async_zero, EdgeType::Contains);

    let tree = graph.to_ascii_tree();

    assert!(tree.contains("Layer 0"));
    assert!(tree.contains("QkvProjection"));
    assert!(tree.contains("gemv_kernel"));
    assert!(tree.contains("compute (ops.rs:100)"));
    assert!(tree.contains("init"));
    assert!(tree.contains("H2D: cpu"));
    assert!(tree.contains("D2H: gpu"));
    assert!(tree.contains("prefetch"));
    assert!(tree.contains("idle"));
    // Brick should show timing info
    assert!(tree.contains("\u{00b5}s"));
    // Kernel should show launch config
    assert!(tree.contains("<<<"));
}

#[test]
fn test_to_ascii_tree_deep_hierarchy() {
    let mut graph = ExecutionGraph::new();
    let _l0 = graph.push_scope(ExecutionNode::Layer { index: 0 });
    let _b1 = graph.add_node_in_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100,
        elements: 64,
    });
    let _b2 = graph.add_node_in_scope(ExecutionNode::Brick {
        id: BrickId::AttentionScore,
        timing_ns: 200,
        elements: 128,
    });
    graph.pop_scope();

    let tree = graph.to_ascii_tree();
    assert!(tree.contains("Layer 0"));
    assert!(tree.contains("RmsNorm"));
    assert!(tree.contains("AttentionScore"));
    // Both children should have tree connectors
    assert!(tree.contains("\u{251c}\u{2500}\u{2500}") || tree.contains("\u{2514}\u{2500}\u{2500}"));
}

#[test]
fn test_to_ascii_tree_nested_launches() {
    let mut graph = ExecutionGraph::new();

    let layer = graph.add_node(ExecutionNode::Layer { index: 0 });
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::GateProjection,
        timing_ns: 5000,
        elements: 2048,
    });
    graph.add_edge(layer, brick, EdgeType::Contains);

    let k1 = graph.add_node(ExecutionNode::Kernel {
        name: "kernel_a".into(),
        ptx_hash: 0,
        grid: (1, 1, 1),
        block: (64, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(brick, k1, EdgeType::Launches);

    let k2 = graph.add_node(ExecutionNode::Kernel {
        name: "kernel_b".into(),
        ptx_hash: 0,
        grid: (2, 1, 1),
        block: (32, 1, 1),
        shared_mem: 512,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(brick, k2, EdgeType::Launches);

    let tree = graph.to_ascii_tree();
    assert!(tree.contains("Layer 0"));
    assert!(tree.contains("GateProjection"));
    assert!(tree.contains("kernel_a"));
    assert!(tree.contains("kernel_b"));
}

// ================================================================
// node_to_dot_label: Function partial location coverage
// ================================================================

/// Function with file but no line — (Some, None) branch in node_to_dot_label.
#[test]
fn test_to_dot_function_file_no_line() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Function {
        name: "partial_func".into(),
        file: Some("module.rs".into()),
        line: None,
    });

    let dot = graph.to_dot();
    // (Some, None) should produce empty loc string — no file:line in label
    assert!(dot.contains("partial_func"));
    assert!(
        !dot.contains("module.rs"),
        "Function with file but no line should not show file in DOT label"
    );
}

/// Function with no file but has line — (None, Some) branch in node_to_dot_label.
#[test]
fn test_to_dot_function_no_file_has_line() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Function {
        name: "orphan_func".into(),
        file: None,
        line: Some(99),
    });

    let dot = graph.to_dot();
    assert!(dot.contains("orphan_func"));
    // (None, Some(99)) should produce empty loc string
    assert!(!dot.contains(":99"), "Function with no file should not show :99 in DOT label");
}

// ================================================================
// to_ascii_tree: Function partial location coverage in ASCII tree
// ================================================================

/// ASCII tree: Function with file but no line.
#[test]
fn test_to_ascii_tree_function_file_no_line() {
    let mut graph = ExecutionGraph::new();
    let root = graph.add_node(ExecutionNode::Layer { index: 0 });
    let func = graph.add_node(ExecutionNode::Function {
        name: "setup".into(),
        file: Some("config.rs".into()),
        line: None,
    });
    graph.add_edge(root, func, EdgeType::Contains);

    let tree = graph.to_ascii_tree();
    assert!(tree.contains("setup"));
    // (Some, None) branch: empty location string
    assert!(
        !tree.contains("config.rs"),
        "Function with file but no line should not show file: {}",
        tree
    );
}

/// ASCII tree: Function with no file but has line.
#[test]
fn test_to_ascii_tree_function_no_file_has_line() {
    let mut graph = ExecutionGraph::new();
    let root = graph.add_node(ExecutionNode::Layer { index: 0 });
    let func = graph.add_node(ExecutionNode::Function {
        name: "teardown".into(),
        file: None,
        line: Some(55),
    });
    graph.add_edge(root, func, EdgeType::Contains);

    let tree = graph.to_ascii_tree();
    assert!(tree.contains("teardown"));
    assert!(!tree.contains(":55"), "Function with no file should not show line number: {}", tree);
}

// ================================================================
// node_to_dot_label: AsyncTask zero polls (0% efficiency) in DOT
// ================================================================

/// Verify DOT output for AsyncTask with zero polls shows 0% efficiency.
#[test]
fn test_to_dot_async_task_zero_polls_efficiency() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::AsyncTask {
        name: "zero_poll_task".into(),
        poll_count: 0,
        yield_count: 0,
        total_poll_ns: 0,
    });

    let dot = graph.to_dot();
    assert!(dot.contains("zero_poll_task"));
    assert!(dot.contains("polls:0"));
    assert!(dot.contains("0%"), "Zero polls should show 0% efficiency in DOT: {}", dot);
}

/// Verify DOT output for AsyncTask with nonzero polls shows correct efficiency.
#[test]
fn test_to_dot_async_task_nonzero_polls_efficiency() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::AsyncTask {
        name: "busy_task".into(),
        poll_count: 4,
        yield_count: 2,
        total_poll_ns: 8000,
    });

    let dot = graph.to_dot();
    assert!(dot.contains("busy_task"));
    assert!(dot.contains("polls:4"));
    assert!(dot.contains("yields:2"));
    // 100.0 / 4 = 25%
    assert!(dot.contains("25%"), "4 polls should show 25% efficiency: {}", dot);
}

// ================================================================
// to_tree_node: empty graph (Empty Graph branch)
// ================================================================

#[cfg(feature = "presentar-tui")]
#[test]
fn test_to_tree_node_empty_graph() {
    let graph = ExecutionGraph::new();
    let tree = graph.to_tree_node();
    assert_eq!(tree.label, "Empty Graph");
    assert!(tree.children.is_empty());
}
