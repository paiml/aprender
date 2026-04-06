use super::super::super::super::*;

// ========================
// Coverage Tests (C021-C031): ExecutionGraph operations
// ========================

/// C021: MatmulOp::tokens() method
#[test]
fn test_c021_matmul_op_tokens() {
    let op = MatmulOp::new(4, 8, 16);
    let a = vec![0.0f32; 4 * 8];
    let b = vec![0.0f32; 8 * 16];
    // tokens = m * n = 4 * 16 = 64
    assert_eq!(op.tokens(&(a, b)), 64);
}

/// C022: ExecutionGraph::add_weighted_edge()
#[test]
fn test_c022_add_weighted_edge() {
    let mut graph = ExecutionGraph::new();
    let n1 = graph.add_node(ExecutionNode::Layer { index: 0 });
    let n2 = graph.add_node(ExecutionNode::Layer { index: 1 });

    graph.add_weighted_edge(n1, n2, EdgeType::Sequence, 2.5);

    assert_eq!(graph.num_edges(), 1);
    let edges = graph.edges();
    assert!((edges[0].weight - 2.5).abs() < 0.001);
}

/// C023: ExecutionGraph::node() lookup by ID
#[test]
fn test_c023_node_by_id() {
    let mut graph = ExecutionGraph::new();
    let id = graph.add_node(ExecutionNode::Layer { index: 42 });

    let node = graph.node(id);
    assert!(node.is_some());
    if let Some(ExecutionNode::Layer { index }) = node {
        assert_eq!(*index, 42);
    } else {
        panic!("Expected Layer node");
    }

    // Non-existent ID
    let bad_id = ExecutionNodeId(999);
    assert!(graph.node(bad_id).is_none());
}

/// C024: ExecutionGraph::node_by_name() lookup
#[test]
fn test_c024_node_by_name() {
    let mut graph = ExecutionGraph::new();

    // Add a function node with a name
    let _id = graph.add_node(ExecutionNode::Function {
        name: "test_function".into(),
        file: Some("test.rs".into()),
        line: Some(100),
    });

    let result = graph.node_by_name("test_function");
    assert!(result.is_some());

    let result = graph.node_by_name("nonexistent");
    assert!(result.is_none());
}

/// C025: record_kernel_launch_with_metrics within scope
#[test]
fn test_c025_record_kernel_with_parent() {
    let mut graph = ExecutionGraph::new();

    // Create a parent scope
    let _brick = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 1000,
        elements: 100,
    });

    // Record kernel within scope
    let kernel_id = graph.record_kernel_launch_with_metrics(
        "child_kernel",
        0x1234,
        (1, 1, 1),
        (32, 1, 1),
        1024,
        500,
        10.0,
        5.0,
    );

    graph.pop_scope();

    // Should have Launches edge from brick to kernel
    let edges: Vec<_> = graph
        .edges()
        .iter()
        .filter(|e| e.dst == kernel_id && matches!(e.edge_type, EdgeType::Launches))
        .collect();
    assert_eq!(edges.len(), 1, "Should have Launches edge");
}

/// C026: record_transfer within scope
#[test]
fn test_c026_record_transfer_with_parent() {
    let mut graph = ExecutionGraph::new();

    // Create a parent scope
    let _layer = graph.push_scope(ExecutionNode::Layer { index: 0 });

    // Record transfer within scope
    let transfer_id =
        graph.record_transfer("host", "device", 1024, TransferDirection::H2D, Some(100));

    graph.pop_scope();

    // Should have Contains edge from layer to transfer
    let edges: Vec<_> = graph
        .edges()
        .iter()
        .filter(|e| e.dst == transfer_id && matches!(e.edge_type, EdgeType::Contains))
        .collect();
    assert_eq!(edges.len(), 1, "Should have Contains edge");
}

/// C027: DotOp::tokens() method
#[test]
fn test_c027_dot_op_tokens() {
    let op = DotOp::new(5);
    let input = (vec![1.0; 5], vec![1.0; 5]);
    assert_eq!(op.tokens(&input), 5);
}

/// C028: SoftmaxOp::tokens() method
#[test]
fn test_c028_softmax_op_tokens() {
    let op = SoftmaxOp::new(10);
    let input = vec![1.0f32; 10];
    assert_eq!(op.tokens(&input), 10);
}

/// C029: ExecutionGraph::current_scope()
#[test]
fn test_c029_current_scope() {
    let mut graph = ExecutionGraph::new();

    // No scope initially
    assert!(graph.current_scope().is_none());

    // Push scope
    let layer_id = graph.push_scope(ExecutionNode::Layer { index: 0 });
    assert_eq!(graph.current_scope(), Some(layer_id));

    // Push another scope
    let brick_id = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100,
        elements: 10,
    });
    assert_eq!(graph.current_scope(), Some(brick_id));

    // Pop back
    graph.pop_scope();
    assert_eq!(graph.current_scope(), Some(layer_id));

    graph.pop_scope();
    assert!(graph.current_scope().is_none());
}

/// C030: to_dot() with Function and Transfer nodes
#[test]
fn test_c030_to_dot_function_and_transfer() {
    let mut graph = ExecutionGraph::new();

    // Add a function node
    graph.add_node(ExecutionNode::Function {
        name: "my_function".into(),
        file: Some("src/main.rs".into()),
        line: Some(42),
    });

    // Add function without file/line
    graph.add_node(ExecutionNode::Function { name: "anonymous".into(), file: None, line: None });

    // Add transfer nodes
    graph.add_node(ExecutionNode::Transfer {
        src: "host".into(),
        dst: "device".into(),
        bytes: 1024 * 1024,
        direction: TransferDirection::H2D,
        timing_ns: Some(100),
    });

    graph.add_node(ExecutionNode::Transfer {
        src: "dev0".into(),
        dst: "dev1".into(),
        bytes: 2 * 1024 * 1024,
        direction: TransferDirection::D2D,
        timing_ns: None,
    });

    let dot = graph.to_dot();

    // Verify DOT output contains expected elements
    assert!(dot.contains("digraph"), "Should be valid digraph");
    assert!(dot.contains("my_function"), "Should contain function name");
    assert!(dot.contains("src/main.rs:42"), "Should contain file:line");
    assert!(dot.contains("anonymous"), "Should contain anonymous function");
    assert!(dot.contains("H2D"), "Should contain H2D transfer");
    assert!(dot.contains("D2D"), "Should contain D2D transfer");
    assert!(dot.contains("lightsalmon"), "Transfer should have color");
    assert!(dot.contains("lightgray"), "Function should have color");
}

/// C031: to_tree_node with Function node (presentar-tui feature)
#[cfg(feature = "presentar-tui")]
#[test]
fn test_c031_to_tree_node_function() {
    let mut graph = ExecutionGraph::new();

    graph.add_node(ExecutionNode::Function {
        name: "test_func".into(),
        file: Some("test.rs".into()),
        line: Some(10),
    });

    let tree = graph.to_tree_node();
    // Just verify it doesn't panic
    assert!(!format!("{:?}", tree).is_empty());
}

// ========================
// Slowest Kernel Tests
// ========================

#[test]
fn test_slowest_kernel_empty_graph() {
    let graph = ExecutionGraph::new();
    assert!(graph.slowest_kernel().is_none());
}

#[test]
fn test_slowest_kernel_no_bricks() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Layer { index: 0 });
    graph.add_node(ExecutionNode::Function { name: "func".into(), file: None, line: None });
    assert!(graph.slowest_kernel().is_none());
}

#[test]
fn test_slowest_kernel_brick_without_launches() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 5000, elements: 100 });
    // Brick exists but has no Launches edges
    assert!(graph.slowest_kernel().is_none());
}

#[test]
fn test_slowest_kernel_single_brick_with_kernel() {
    let mut graph = ExecutionGraph::new();
    let brick_id = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 3000,
        elements: 100,
    });
    let kernel_id = graph.add_node(ExecutionNode::Kernel {
        name: "matmul".into(),
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 1024,
        timing_ns: Some(2000),
        arithmetic_intensity: None,
        achieved_tflops: None,
        ptx_hash: 0,
    });
    graph.add_edge(brick_id, kernel_id, EdgeType::Launches);

    let result = graph.slowest_kernel();
    assert!(result.is_some());
    let (id, _node, timing) = result.unwrap();
    assert_eq!(id, brick_id);
    assert_eq!(timing, 3000);
}

#[test]
fn test_slowest_kernel_multiple_bricks() {
    let mut graph = ExecutionGraph::new();

    let brick1 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 50,
    });
    let kernel1 = graph.add_node(ExecutionNode::Kernel {
        name: "norm".into(),
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 512,
        timing_ns: Some(800),
        arithmetic_intensity: None,
        achieved_tflops: None,
        ptx_hash: 0,
    });
    graph.add_edge(brick1, kernel1, EdgeType::Launches);

    let brick2 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 5000,
        elements: 200,
    });
    let kernel2 = graph.add_node(ExecutionNode::Kernel {
        name: "matmul".into(),
        grid: (4, 1, 1),
        block: (256, 1, 1),
        shared_mem: 4096,
        timing_ns: Some(4000),
        arithmetic_intensity: None,
        achieved_tflops: None,
        ptx_hash: 0,
    });
    graph.add_edge(brick2, kernel2, EdgeType::Launches);

    let result = graph.slowest_kernel();
    assert!(result.is_some());
    let (id, _node, timing) = result.unwrap();
    assert_eq!(id, brick2); // brick2 is slower
    assert_eq!(timing, 5000);
}

// ========================
// Tree Node All-Variant Tests
// ========================

#[test]
#[cfg(feature = "presentar-tui")]
fn test_to_tree_node_empty_graph() {
    let graph = ExecutionGraph::new();
    let tree = graph.to_tree_node();
    assert!(!format!("{:?}", tree).is_empty());
}

#[test]
#[cfg(feature = "presentar-tui")]
fn test_to_tree_node_multiple_roots() {
    let mut graph = ExecutionGraph::new();
    // Two disconnected nodes = two roots
    graph.add_node(ExecutionNode::Layer { index: 0 });
    graph.add_node(ExecutionNode::Layer { index: 1 });

    let tree = graph.to_tree_node();
    assert!(!format!("{:?}", tree).is_empty());
}

#[test]
#[cfg(feature = "presentar-tui")]
fn test_to_tree_node_all_node_types() {
    let mut graph = ExecutionGraph::new();

    let layer = graph.add_node(ExecutionNode::Layer { index: 0 });
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 100,
    });
    let kernel = graph.add_node(ExecutionNode::Kernel {
        name: "matmul_kernel".into(),
        grid: (4, 1, 1),
        block: (256, 1, 1),
        shared_mem: 2048,
        timing_ns: Some(500),
        arithmetic_intensity: Some(2.5),
        achieved_tflops: Some(1.2),
        ptx_hash: 0,
    });
    let func = graph.add_node(ExecutionNode::Function {
        name: "forward".into(),
        file: Some("model.rs".into()),
        line: Some(42),
    });
    let func_no_loc =
        graph.add_node(ExecutionNode::Function { name: "helper".into(), file: None, line: None });
    let transfer = graph.add_node(ExecutionNode::Transfer {
        src: "host".into(),
        dst: "device".into(),
        bytes: 4096,
        direction: TransferDirection::H2D,
        timing_ns: Some(200),
    });
    let transfer_no_timing = graph.add_node(ExecutionNode::Transfer {
        src: "device".into(),
        dst: "host".into(),
        bytes: 2048,
        direction: TransferDirection::D2H,
        timing_ns: None,
    });
    let async_task = graph.add_node(ExecutionNode::AsyncTask {
        name: "prefetch".into(),
        poll_count: 5,
        yield_count: 2,
        total_poll_ns: 1500,
    });
    let async_zero = graph.add_node(ExecutionNode::AsyncTask {
        name: "idle_task".into(),
        poll_count: 0,
        yield_count: 0,
        total_poll_ns: 0,
    });

    // Connect: layer -> brick -> kernel
    graph.add_edge(layer, brick, EdgeType::Sequence);
    graph.add_edge(brick, kernel, EdgeType::Launches);
    graph.add_edge(layer, func, EdgeType::Sequence);
    graph.add_edge(layer, func_no_loc, EdgeType::Sequence);
    graph.add_edge(layer, transfer, EdgeType::Sequence);
    graph.add_edge(layer, transfer_no_timing, EdgeType::Sequence);
    graph.add_edge(layer, async_task, EdgeType::Sequence);
    graph.add_edge(layer, async_zero, EdgeType::Sequence);

    let tree = graph.to_tree_node();
    let debug = format!("{:?}", tree);
    assert!(!debug.is_empty());
}

// ========================
// Critical Path Coverage
// ========================

#[test]
fn test_critical_path_empty_graph() {
    let graph = ExecutionGraph::new();
    let (path, total) = graph.critical_path();
    assert!(path.is_empty());
    assert_eq!(total, 0);
}

#[test]
fn test_critical_path_linear_chain() {
    let mut graph = ExecutionGraph::new();
    let n1 =
        graph.add_node(ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 100, elements: 10 });
    let n2 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 200,
        elements: 20,
    });
    let n3 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::Activation,
        timing_ns: 50,
        elements: 10,
    });

    graph.add_edge(n1, n2, EdgeType::Sequence);
    graph.add_edge(n2, n3, EdgeType::Sequence);

    let (path, total) = graph.critical_path();
    assert!(!path.is_empty());
    assert!(total > 0);
}

// ========================
// ASCII Tree Coverage
// ========================

#[test]
fn test_to_ascii_tree_with_all_node_types() {
    let mut graph = ExecutionGraph::new();
    let layer = graph.add_node(ExecutionNode::Layer { index: 0 });
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 100,
    });
    let kernel = graph.add_node(ExecutionNode::Kernel {
        name: "matmul".into(),
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 512,
        timing_ns: Some(800),
        arithmetic_intensity: None,
        achieved_tflops: None,
        ptx_hash: 0,
    });
    graph.add_edge(layer, brick, EdgeType::Sequence);
    graph.add_edge(brick, kernel, EdgeType::Launches);

    let ascii = graph.to_ascii_tree();
    assert!(!ascii.is_empty());
}
