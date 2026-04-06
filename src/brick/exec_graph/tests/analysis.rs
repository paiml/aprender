//! Coverage tests for analysis.rs: critical_path, critical_path_summary, format_node_name.
//!
//! Target uncovered functions:
//!   - critical_path (line 39, 84.4% cov) — Transfer edge handling, single-node path
//!   - critical_path_summary (line 273, 72.1% cov) — parallelization opportunities section
//!   - format_node_name (line 329, 30.8% cov) — Transfer, AsyncTask branches

use super::super::*;

// ========================================================================
// critical_path tests — cover Transfer edges, Sequence edges, single-node
// ========================================================================

/// Empty graph produces empty critical path.
#[test]
fn test_critical_path_empty_graph() {
    let graph = ExecutionGraph::new();
    let (path, total) = graph.critical_path();
    assert!(path.is_empty());
    assert_eq!(total, 0);
}

/// Single node graph — critical path is just that node.
#[test]
fn test_critical_path_single_node() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 5000, elements: 1024 });
    let (path, total) = graph.critical_path();
    assert_eq!(path.len(), 1);
    assert_eq!(total, 5000);
}

/// Linear chain with DependsOn edges — critical path follows the chain.
#[test]
fn test_critical_path_linear_depends_on() {
    let mut graph = ExecutionGraph::new();
    let a = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 512,
    });
    let b = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 3000,
        elements: 2048,
    });
    let c = graph.add_node(ExecutionNode::Brick {
        id: BrickId::AttentionScore,
        timing_ns: 2000,
        elements: 1024,
    });
    graph.add_dependency(a, b);
    graph.add_dependency(b, c);

    let (path, total) = graph.critical_path();
    assert!(path.len() >= 2, "Path should include at least 2 nodes");
    assert!(total >= 5000, "Total should be at least 5000ns: got {}", total);
}

/// Sequence edges are included in critical path analysis.
#[test]
fn test_critical_path_sequence_edges() {
    let mut graph = ExecutionGraph::new();
    let a = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 256,
    });
    let b = graph.add_node(ExecutionNode::Brick {
        id: BrickId::LayerNorm,
        timing_ns: 2000,
        elements: 256,
    });
    graph.add_edge(a, b, EdgeType::Sequence);

    let (path, total) = graph.critical_path();
    assert!(!path.is_empty());
    assert!(total >= 2000, "Total should include sequence weight");
}

/// Transfer edges (EdgeType::Transfer) are handled in critical path.
#[test]
fn test_critical_path_transfer_edge() {
    let mut graph = ExecutionGraph::new();
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 2000,
        elements: 1024,
    });
    let transfer = graph.add_node(ExecutionNode::Transfer {
        src: "host".into(),
        dst: "gpu0".into(),
        bytes: 1_000_000,
        direction: TransferDirection::H2D,
        timing_ns: Some(500),
    });
    graph.add_edge(
        brick,
        transfer,
        EdgeType::Transfer { bytes: 1_000_000, direction: TransferDirection::H2D },
    );

    let (path, total) = graph.critical_path();
    assert!(!path.is_empty());
    // The total should include the transfer node timing
    assert!(total >= 500, "Total should include transfer timing: got {}", total);
}

/// Contains edges contribute to critical path (hierarchical).
#[test]
fn test_critical_path_contains_edges() {
    let mut graph = ExecutionGraph::new();
    let layer = graph.add_node(ExecutionNode::Layer { index: 0 });
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 5000,
        elements: 2048,
    });
    graph.add_edge(layer, brick, EdgeType::Contains);

    let (path, total) = graph.critical_path();
    assert!(path.len() >= 2, "Path should contain layer and brick");
    assert!(total >= 5000, "Total should include brick timing");
}

/// Calls edges contribute to critical path.
#[test]
fn test_critical_path_calls_edges() {
    let mut graph = ExecutionGraph::new();
    let func1 =
        graph.add_node(ExecutionNode::Function { name: "outer".into(), file: None, line: None });
    let func2 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::GateProjection,
        timing_ns: 7000,
        elements: 4096,
    });
    graph.add_edge(func1, func2, EdgeType::Calls);

    let (path, total) = graph.critical_path();
    assert!(!path.is_empty());
    assert!(total >= 7000, "Total should include brick timing via Calls edge");
}

/// Launches edges contribute to critical path.
#[test]
fn test_critical_path_launches_edges() {
    let mut graph = ExecutionGraph::new();
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 3000,
        elements: 1024,
    });
    let kernel = graph.add_node(ExecutionNode::Kernel {
        name: "gemv".into(),
        ptx_hash: 0xDEAD,
        grid: (4, 1, 1),
        block: (128, 1, 1),
        shared_mem: 2048,
        timing_ns: Some(1500),
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(brick, kernel, EdgeType::Launches);

    let (path, total) = graph.critical_path();
    assert!(path.len() >= 2);
    assert!(total >= 1500, "Total should include kernel timing via Launches edge");
}

/// Diamond graph — critical path picks the longer branch.
#[test]
fn test_critical_path_diamond() {
    let mut graph = ExecutionGraph::new();
    let start = graph.add_node(ExecutionNode::Brick {
        id: BrickId::Embedding,
        timing_ns: 100,
        elements: 64,
    });
    // Fast path
    let fast = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 64,
    });
    // Slow path
    let slow = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 9000,
        elements: 64,
    });
    let end =
        graph.add_node(ExecutionNode::Brick { id: BrickId::LmHead, timing_ns: 200, elements: 64 });

    graph.add_dependency(start, fast);
    graph.add_dependency(start, slow);
    graph.add_dependency(fast, end);
    graph.add_dependency(slow, end);

    let (path, total) = graph.critical_path();
    // Should pick start -> slow -> end
    assert!(path.len() >= 2);
    assert!(total >= 9000, "Critical path should follow slow branch: got {}", total);
}

/// Kernel with no timing (timing_ns = None) contributes 0 to critical path.
#[test]
fn test_critical_path_kernel_no_timing() {
    let mut graph = ExecutionGraph::new();
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 256,
    });
    let kernel = graph.add_node(ExecutionNode::Kernel {
        name: "untimed_kernel".into(),
        ptx_hash: 0,
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(brick, kernel, EdgeType::Launches);

    let (path, total) = graph.critical_path();
    assert!(!path.is_empty());
    // Kernel with None timing contributes 0
    assert_eq!(total, 1000, "Kernel with no timing should contribute 0");
}

/// Transfer with no timing (timing_ns = None) contributes 0.
#[test]
fn test_critical_path_transfer_no_timing() {
    let mut graph = ExecutionGraph::new();
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 2000,
        elements: 128,
    });
    let transfer = graph.add_node(ExecutionNode::Transfer {
        src: "host".into(),
        dst: "gpu0".into(),
        bytes: 4096,
        direction: TransferDirection::H2D,
        timing_ns: None,
    });
    graph.add_edge(brick, transfer, EdgeType::Contains);

    let (path, _total) = graph.critical_path();
    assert!(!path.is_empty());
}

/// Layer and Function nodes contribute 0 timing to critical path.
#[test]
fn test_critical_path_zero_timing_nodes() {
    let mut graph = ExecutionGraph::new();
    let layer = graph.add_node(ExecutionNode::Layer { index: 0 });
    let func =
        graph.add_node(ExecutionNode::Function { name: "setup".into(), file: None, line: None });
    graph.add_edge(layer, func, EdgeType::Contains);

    let (path, total) = graph.critical_path();
    assert!(!path.is_empty());
    assert_eq!(total, 0, "Layer and Function contribute 0 timing");
}

/// AsyncTask nodes contribute 0 timing to critical path.
#[test]
fn test_critical_path_async_task() {
    let mut graph = ExecutionGraph::new();
    let async_node = graph.add_node(ExecutionNode::AsyncTask {
        name: "prefetch".into(),
        poll_count: 5,
        yield_count: 2,
        total_poll_ns: 10_000,
    });
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::Activation,
        timing_ns: 3000,
        elements: 512,
    });
    graph.add_dependency(async_node, brick);

    let (path, total) = graph.critical_path();
    assert!(!path.is_empty());
    // AsyncTask contributes 0 timing, brick contributes 3000
    assert!(total >= 3000);
}

// ========================================================================
// critical_path_summary tests — cover parallelization opportunities
// ========================================================================

/// Summary for empty graph.
#[test]
fn test_critical_path_summary_empty() {
    let graph = ExecutionGraph::new();
    let summary = graph.critical_path_summary();
    assert!(summary.contains("Critical Path:"));
    assert!(summary.contains("0 nodes"));
}

/// Summary for single-node graph (no parallelization opportunities).
#[test]
fn test_critical_path_summary_single_node() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 5000, elements: 1024 });

    let summary = graph.critical_path_summary();
    assert!(summary.contains("Critical Path:"));
    assert!(summary.contains("1 nodes"));
    // Single node on critical path means no parallelization opportunities
}

/// Summary for linear chain — shows node names and prefix characters.
#[test]
fn test_critical_path_summary_chain() {
    let mut graph = ExecutionGraph::new();
    let a = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 256,
    });
    let b = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 3000,
        elements: 1024,
    });
    let c = graph.add_node(ExecutionNode::Brick {
        id: BrickId::AttentionScore,
        timing_ns: 2000,
        elements: 512,
    });
    graph.add_dependency(a, b);
    graph.add_dependency(b, c);

    let summary = graph.critical_path_summary();
    assert!(summary.contains("Critical Path:"));
    // Should show Unicode box-drawing characters for path rendering
    assert!(
        summary.contains("\u{250c}")
            || summary.contains("\u{2502}")
            || summary.contains("\u{2514}"),
        "Summary should contain box-drawing characters for path prefix"
    );
}

/// Summary shows parallelization opportunities (nodes with slack > 0).
#[test]
fn test_critical_path_summary_with_slack() {
    let mut graph = ExecutionGraph::new();

    // Create a diamond so the off-path branch has slack
    let start = graph.add_node(ExecutionNode::Brick {
        id: BrickId::Embedding,
        timing_ns: 100,
        elements: 32,
    });
    let fast = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 64,
    });
    let slow = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 10_000,
        elements: 2048,
    });
    let end =
        graph.add_node(ExecutionNode::Brick { id: BrickId::LmHead, timing_ns: 200, elements: 32 });

    graph.add_dependency(start, fast);
    graph.add_dependency(start, slow);
    graph.add_dependency(fast, end);
    graph.add_dependency(slow, end);

    let summary = graph.critical_path_summary();
    assert!(summary.contains("Critical Path:"));
    // The fast path node should have slack -> parallelization opportunities
    assert!(
        summary.contains("Parallelization Opportunities"),
        "Should show parallelization opportunities when slack exists: {}",
        summary
    );
    assert!(summary.contains("slack="), "Should show slack values: {}", summary);
}

/// Summary with Transfer node in the path exercises format_node_name for Transfer.
#[test]
fn test_critical_path_summary_with_transfer() {
    let mut graph = ExecutionGraph::new();
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 2000,
        elements: 1024,
    });
    let transfer = graph.add_node(ExecutionNode::Transfer {
        src: "host".into(),
        dst: "gpu0".into(),
        bytes: 1_000_000,
        direction: TransferDirection::H2D,
        timing_ns: Some(5000),
    });
    graph.add_edge(brick, transfer, EdgeType::Contains);

    let summary = graph.critical_path_summary();
    assert!(summary.contains("Critical Path:"));
    // Transfer format_node_name should show "H2D host -> gpu0"
    assert!(
        summary.contains("H2D") || summary.contains("QkvProjection"),
        "Summary should mention transfer or brick name: {}",
        summary
    );
}

/// Summary with AsyncTask node exercises format_node_name for AsyncTask.
#[test]
fn test_critical_path_summary_with_async_task() {
    let mut graph = ExecutionGraph::new();
    let async_node = graph.add_node(ExecutionNode::AsyncTask {
        name: "prefetch_weights".into(),
        poll_count: 5,
        yield_count: 3,
        total_poll_ns: 10_000,
    });
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 3000,
        elements: 512,
    });
    graph.add_dependency(async_node, brick);

    let summary = graph.critical_path_summary();
    assert!(summary.contains("Critical Path:"));
}

/// Summary with Kernel and Function nodes exercises those format_node_name branches.
#[test]
fn test_critical_path_summary_kernel_and_function() {
    let mut graph = ExecutionGraph::new();
    let func = graph.add_node(ExecutionNode::Function {
        name: "forward".into(),
        file: Some("model.rs".into()),
        line: Some(42),
    });
    let kernel = graph.add_node(ExecutionNode::Kernel {
        name: "matmul_kernel".into(),
        ptx_hash: 0xABCD,
        grid: (8, 1, 1),
        block: (256, 1, 1),
        shared_mem: 4096,
        timing_ns: Some(5000),
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(func, kernel, EdgeType::Launches);

    let summary = graph.critical_path_summary();
    assert!(summary.contains("matmul_kernel") || summary.contains("forward"));
}

/// Summary with Layer node exercises the Layer branch of format_node_name.
#[test]
fn test_critical_path_summary_layer() {
    let mut graph = ExecutionGraph::new();
    let layer = graph.add_node(ExecutionNode::Layer { index: 3 });
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 64,
    });
    graph.add_edge(layer, brick, EdgeType::Contains);

    let summary = graph.critical_path_summary();
    assert!(summary.contains("Layer 3") || summary.contains("RmsNorm"));
}

// ========================================================================
// format_node_name tests — exercise all branches via critical_path_summary
// ========================================================================

/// format_node_name: Transfer node with D2H direction.
#[test]
fn test_format_node_name_via_summary_transfer_d2h() {
    let mut graph = ExecutionGraph::new();
    // Only node — will be on critical path, exercising format_node_name
    graph.add_node(ExecutionNode::Transfer {
        src: "gpu0".into(),
        dst: "host".into(),
        bytes: 2048,
        direction: TransferDirection::D2H,
        timing_ns: Some(1000),
    });

    let summary = graph.critical_path_summary();
    assert!(
        summary.contains("D2H"),
        "Summary should contain D2H from format_node_name: {}",
        summary
    );
    assert!(summary.contains("gpu0"), "Summary should contain src: {}", summary);
}

/// format_node_name: Transfer node with D2D direction.
#[test]
fn test_format_node_name_via_summary_transfer_d2d() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Transfer {
        src: "gpu0".into(),
        dst: "gpu1".into(),
        bytes: 4096,
        direction: TransferDirection::D2D,
        timing_ns: Some(500),
    });

    let summary = graph.critical_path_summary();
    assert!(summary.contains("D2D"), "Summary should contain D2D: {}", summary);
}

/// format_node_name: AsyncTask node with poll_count.
#[test]
fn test_format_node_name_via_summary_async_task() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::AsyncTask {
        name: "fetch_data".into(),
        poll_count: 7,
        yield_count: 4,
        total_poll_ns: 15_000,
    });

    let summary = graph.critical_path_summary();
    assert!(summary.contains("fetch_data"), "Summary should contain async task name: {}", summary);
    assert!(
        summary.contains("7polls"),
        "Summary should contain poll count from format_node_name: {}",
        summary
    );
}

/// format_node_name: Layer node.
#[test]
fn test_format_node_name_via_summary_layer() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Layer { index: 42 });

    let summary = graph.critical_path_summary();
    assert!(summary.contains("Layer 42"), "Summary should contain Layer name: {}", summary);
}

/// format_node_name: Brick node (exercises id.name()).
#[test]
fn test_format_node_name_via_summary_brick() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Brick {
        id: BrickId::DownProjection,
        timing_ns: 6000,
        elements: 4096,
    });

    let summary = graph.critical_path_summary();
    assert!(summary.contains("DownProjection"), "Summary should contain brick name: {}", summary);
}

/// format_node_name: Kernel node (exercises name clone).
#[test]
fn test_format_node_name_via_summary_kernel() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Kernel {
        name: "softmax_warp".into(),
        ptx_hash: 0xFF,
        grid: (4, 1, 1),
        block: (128, 1, 1),
        shared_mem: 1024,
        timing_ns: Some(3000),
        arithmetic_intensity: None,
        achieved_tflops: None,
    });

    let summary = graph.critical_path_summary();
    assert!(summary.contains("softmax_warp"), "Summary should contain kernel name: {}", summary);
}

/// format_node_name: Function node.
#[test]
fn test_format_node_name_via_summary_function() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Function {
        name: "inference".into(),
        file: Some("main.rs".into()),
        line: Some(10),
    });

    let summary = graph.critical_path_summary();
    assert!(summary.contains("inference"), "Summary should contain function name: {}", summary);
}

// ========================================================================
// compute_slack tests — exercise the slack computation paths
// ========================================================================

/// Slack for empty graph.
#[test]
fn test_compute_slack_empty() {
    let graph = ExecutionGraph::new();
    let slack = graph.compute_slack();
    assert!(slack.is_empty());
}

/// Single node on critical path has slack = 0.
#[test]
fn test_compute_slack_single_node() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 1000, elements: 64 });

    let slack = graph.compute_slack();
    assert_eq!(slack.len(), 1);
    assert_eq!(slack[&ExecutionNodeId(0)], 0);
}

/// Diamond graph — off-critical-path node has slack > 0.
#[test]
fn test_compute_slack_diamond() {
    let mut graph = ExecutionGraph::new();
    let start = graph.add_node(ExecutionNode::Brick {
        id: BrickId::Embedding,
        timing_ns: 100,
        elements: 32,
    });
    let fast = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 64,
    });
    let slow = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 10_000,
        elements: 2048,
    });
    let end =
        graph.add_node(ExecutionNode::Brick { id: BrickId::LmHead, timing_ns: 200, elements: 32 });

    graph.add_dependency(start, fast);
    graph.add_dependency(start, slow);
    graph.add_dependency(fast, end);
    graph.add_dependency(slow, end);

    let slack = graph.compute_slack();
    // The "fast" path should have non-zero slack
    let fast_slack = slack[&fast];
    assert!(fast_slack > 0, "Fast branch should have positive slack: got {}", fast_slack);
}

// ========================================================================
// critical_path_summary prefix rendering — two-node path has start/end
// ========================================================================

/// Two-node path should render with start (box-drawing top) and end (box-drawing bottom).
#[test]
fn test_critical_path_summary_two_node_prefix() {
    let mut graph = ExecutionGraph::new();
    let a = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 64,
    });
    let b = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 2000,
        elements: 128,
    });
    graph.add_dependency(a, b);

    let summary = graph.critical_path_summary();
    // Two-node path: first gets "top-left corner", last gets "bottom-left corner"
    assert!(
        summary.contains("\u{250c}"),
        "First node should have top-left corner prefix: {}",
        summary
    );
    assert!(
        summary.contains("\u{2514}"),
        "Last node should have bottom-left corner prefix: {}",
        summary
    );
}

/// Three-node path has middle prefix character.
#[test]
fn test_critical_path_summary_three_node_prefix() {
    let mut graph = ExecutionGraph::new();
    let a = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 64,
    });
    let b = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 3000,
        elements: 128,
    });
    let c = graph.add_node(ExecutionNode::Brick {
        id: BrickId::AttentionScore,
        timing_ns: 2000,
        elements: 256,
    });
    graph.add_dependency(a, b);
    graph.add_dependency(b, c);

    let summary = graph.critical_path_summary();
    // Three-node path: middle gets pipe character
    assert!(
        summary.contains("\u{2502}"),
        "Middle node should have pipe prefix in 3-node path: {}",
        summary
    );
}
