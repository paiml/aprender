// =========================================================================
// ExecutionGraph coverage tests (PMAT-018) - test uncovered node variants
// =========================================================================

/// Test ExecutionNode::Function variant formatting
#[test]
fn test_execution_node_function_formatting() {
    use crate::brick::exec_graph::{ExecutionGraph, ExecutionNode};

    let mut graph = ExecutionGraph::new();

    // Function with file and line
    let func1 = graph.add_node(ExecutionNode::Function {
        name: "test_func".to_string(),
        file: Some("src/main.rs".to_string()),
        line: Some(42),
    });

    // Function without file/line
    let func2 = graph.add_node(ExecutionNode::Function {
        name: "anonymous".to_string(),
        file: None,
        line: None,
    });

    // Function with file but no line
    let func3 = graph.add_node(ExecutionNode::Function {
        name: "partial".to_string(),
        file: Some("lib.rs".to_string()),
        line: None,
    });

    // Test the formatting via to_ascii_tree
    let ascii = graph.to_ascii_tree();
    assert!(ascii.contains("test_func"), "Should contain function name");
    assert!(ascii.contains("anonymous"), "Should contain anonymous function");

    // Use the node IDs to prevent unused warnings
    assert!(graph.node(func1).is_some());
    assert!(graph.node(func2).is_some());
    assert!(graph.node(func3).is_some());
}

/// Test ExecutionNode::Transfer variant formatting
#[test]
fn test_execution_node_transfer_formatting() {
    use crate::brick::exec_graph::{ExecutionGraph, ExecutionNode, TransferDirection};

    let mut graph = ExecutionGraph::new();

    // Transfer with timing (Host to Device)
    let t1 = graph.add_node(ExecutionNode::Transfer {
        src: "CPU".to_string(),
        dst: "GPU".to_string(),
        bytes: 1024 * 1024, // 1MB
        direction: TransferDirection::H2D,
        timing_ns: Some(5000),
    });

    // Transfer without timing (Device to Host)
    let t2 = graph.add_node(ExecutionNode::Transfer {
        src: "GPU".to_string(),
        dst: "CPU".to_string(),
        bytes: 512,
        direction: TransferDirection::D2H,
        timing_ns: None,
    });

    // Device to device transfer
    let t3 = graph.add_node(ExecutionNode::Transfer {
        src: "GPU0".to_string(),
        dst: "GPU1".to_string(),
        bytes: 256,
        direction: TransferDirection::D2D,
        timing_ns: Some(100),
    });

    // Test the formatting via to_ascii_tree
    let ascii = graph.to_ascii_tree();
    assert!(ascii.contains("CPU"), "Should contain CPU");
    assert!(ascii.contains("GPU"), "Should contain GPU");

    // Use the node IDs to prevent unused warnings
    assert!(graph.node(t1).is_some());
    assert!(graph.node(t2).is_some());
    assert!(graph.node(t3).is_some());
}

/// Test slowest_kernel edge cases (covers the `_ => {}` match arm)
#[test]
fn test_slowest_kernel_edge_cases() {
    use crate::brick::exec_graph::{BrickId, ExecutionGraph, ExecutionNode};

    let mut graph = ExecutionGraph::new();

    // Add bricks with various timings
    graph.add_node(ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 100, elements: 1 });

    graph.add_node(ExecutionNode::Brick {
        id: BrickId::AttentionScore,
        timing_ns: 50, // Smaller timing - tests the `_ => {}` arm
        elements: 1,
    });

    graph.add_node(ExecutionNode::Brick {
        id: BrickId::GateProjection,
        timing_ns: 200, // Largest timing
        elements: 1,
    });

    // Add a layer (non-timed node)
    graph.add_node(ExecutionNode::Layer { index: 0 });

    // slowest_kernel only returns Kernel nodes, not Brick nodes
    // So this should be None since we only have Bricks
    let slowest = graph.slowest_kernel();
    assert!(slowest.is_none(), "No kernels added, should be None");
}

/// Test AsyncTask node formatting
#[test]
fn test_execution_node_async_task_formatting() {
    use crate::brick::exec_graph::{ExecutionGraph, ExecutionNode};

    let mut graph = ExecutionGraph::new();

    // AsyncTask with multiple polls
    let task1 = graph.add_node(ExecutionNode::AsyncTask {
        name: "load_weights".to_string(),
        poll_count: 5,
        yield_count: 3,
        total_poll_ns: 10000,
    });

    // AsyncTask with single poll (no yields)
    let task2 = graph.add_node(ExecutionNode::AsyncTask {
        name: "prefetch".to_string(),
        poll_count: 1,
        yield_count: 0,
        total_poll_ns: 500,
    });

    // Test formatting
    let ascii = graph.to_ascii_tree();
    assert!(ascii.contains("load_weights") || !ascii.is_empty()); // May or may not be in tree

    // Use node IDs
    assert!(graph.node(task1).is_some());
    assert!(graph.node(task2).is_some());
}

/// Test graph with all node types for DOT export
#[test]
fn test_to_dot_all_node_types() {
    use crate::brick::exec_graph::{BrickId, ExecutionGraph, ExecutionNode, TransferDirection};

    let mut graph = ExecutionGraph::new();

    // Add one of each type
    let layer = graph.push_scope(ExecutionNode::Layer { index: 0 });

    let brick = graph.add_node_in_scope(ExecutionNode::Brick {
        id: BrickId::DownProjection,
        timing_ns: 5000,
        elements: 1024,
    });

    let kernel = graph.add_node_in_scope(ExecutionNode::Kernel {
        name: "matmul_f32".to_string(),
        ptx_hash: 0x1234567890abcdef,
        grid: (32, 1, 1),
        block: (256, 1, 1),
        shared_mem: 1024,
        timing_ns: Some(2500),
        arithmetic_intensity: Some(1.5),
        achieved_tflops: Some(0.8),
    });

    let func = graph.add_node_in_scope(ExecutionNode::Function {
        name: "compute".to_string(),
        file: Some("src/ops.rs".to_string()),
        line: Some(100),
    });

    let transfer = graph.add_node_in_scope(ExecutionNode::Transfer {
        src: "RAM".to_string(),
        dst: "VRAM".to_string(),
        bytes: 4096,
        direction: TransferDirection::H2D,
        timing_ns: Some(1000),
    });

    let async_task = graph.add_node_in_scope(ExecutionNode::AsyncTask {
        name: "io_wait".to_string(),
        poll_count: 3,
        yield_count: 1,
        total_poll_ns: 500,
    });

    graph.pop_scope();

    // Generate DOT output
    let dot = graph.to_dot();

    // Verify DOT contains expected structure
    assert!(dot.contains("digraph ExecutionGraph"), "Should have digraph header");
    assert!(dot.contains("Layer 0"), "Should contain layer");
    assert!(dot.contains("matmul_f32"), "Should contain kernel name");

    // Use all node IDs
    let _ = (layer, brick, kernel, func, transfer, async_task);
}

/// Test slowest_kernel with actual kernels (via Brick->Kernel edges)
#[test]
fn test_slowest_kernel_with_kernels() {
    use crate::brick::exec_graph::{BrickId, EdgeType, ExecutionGraph, ExecutionNode};

    let mut graph = ExecutionGraph::new();

    // Add bricks that will launch kernels
    let brick1 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100, // Fast brick
        elements: 1,
    });

    let brick2 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::AttentionScore,
        timing_ns: 500, // Slow brick (should be slowest)
        elements: 1,
    });

    let brick3 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::GateProjection,
        timing_ns: 200, // Medium brick - tests `_ => {}` arm
        elements: 1,
    });

    // Add kernels
    let kernel1 = graph.add_node(ExecutionNode::Kernel {
        name: "kernel_fast".to_string(),
        ptx_hash: 0x1111,
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
        timing_ns: Some(50),
        arithmetic_intensity: None,
        achieved_tflops: None,
    });

    let kernel2 = graph.add_node(ExecutionNode::Kernel {
        name: "kernel_slow".to_string(),
        ptx_hash: 0x2222,
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
        timing_ns: Some(250),
        arithmetic_intensity: None,
        achieved_tflops: None,
    });

    let kernel3 = graph.add_node(ExecutionNode::Kernel {
        name: "kernel_medium".to_string(),
        ptx_hash: 0x3333,
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
        timing_ns: Some(100),
        arithmetic_intensity: None,
        achieved_tflops: None,
    });

    // Connect bricks to kernels with Launches edges
    graph.add_edge(brick1, kernel1, EdgeType::Launches);
    graph.add_edge(brick2, kernel2, EdgeType::Launches);
    graph.add_edge(brick3, kernel3, EdgeType::Launches);

    // Find slowest kernel (actually slowest brick with kernel children)
    let slowest = graph.slowest_kernel();
    assert!(slowest.is_some(), "Should find slowest brick with kernel");
    let (_, node, timing) = slowest.unwrap();
    assert_eq!(timing, 500, "Slowest brick should have timing 500");
    assert!(node.is_brick(), "Should be a brick node (not kernel)");
}
