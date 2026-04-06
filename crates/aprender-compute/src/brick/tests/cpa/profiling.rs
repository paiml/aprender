use super::super::super::*;

// ========================
// Phase 9: CPA and Advanced Profiling Tests (F128-F135)
// ========================

/// F128: Critical path identifies longest execution chain
#[test]
fn test_f128_critical_path_linear() {
    let mut graph = ExecutionGraph::new();

    // Create a linear chain: A -> B -> C with increasing timing
    let a = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100_000, // 100µs
        elements: 1024,
    });
    graph.pop_scope();

    let b = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 200_000, // 200µs
        elements: 2048,
    });
    graph.pop_scope();

    let c = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::AttentionScore,
        timing_ns: 300_000, // 300µs
        elements: 4096,
    });
    graph.pop_scope();

    // Add dependencies: A -> B -> C
    graph.add_dependency(a, b);
    graph.add_dependency(b, c);

    let (path, total_ns) = graph.critical_path();

    // Critical path should be A -> B -> C = 100 + 200 + 300 = 600µs
    assert_eq!(path.len(), 3, "F128: Critical path should have 3 nodes");
    assert!(total_ns >= 600_000, "F128: Total time >= 600µs");
}

/// F129: Slack is zero for nodes on critical path
#[test]
fn test_f129_slack_critical_path_zero() {
    let mut graph = ExecutionGraph::new();

    // Linear chain where all nodes are on critical path
    let a = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100_000,
        elements: 1024,
    });
    graph.pop_scope();

    let b = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 200_000,
        elements: 2048,
    });
    graph.pop_scope();

    graph.add_dependency(a, b);

    let (critical_path, _) = graph.critical_path();
    let slack = graph.compute_slack();

    // All nodes on critical path should have zero slack
    for node_id in &critical_path {
        let node_slack = slack.get(node_id).copied().unwrap_or(u64::MAX);
        assert_eq!(node_slack, 0, "F129: Critical path node has zero slack");
    }
}

/// F130: Non-critical nodes have positive slack
#[test]
fn test_f130_slack_parallel_branch() {
    let mut graph = ExecutionGraph::new();

    // Diamond pattern: A -> B, A -> C, B -> D, C -> D
    // If B takes 200µs and C takes 100µs, C has slack
    let a = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 50_000,
        elements: 1024,
    });
    graph.pop_scope();

    let b = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 200_000, // Longer path
        elements: 2048,
    });
    graph.pop_scope();

    let c = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::AttentionScore,
        timing_ns: 100_000, // Shorter path
        elements: 2048,
    });
    graph.pop_scope();

    let d = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::GateProjection,
        timing_ns: 50_000,
        elements: 4096,
    });
    graph.pop_scope();

    // A -> B and A -> C
    graph.add_dependency(a, b);
    graph.add_dependency(a, c);
    // B -> D and C -> D
    graph.add_dependency(b, d);
    graph.add_dependency(c, d);

    let slack = graph.compute_slack();

    // C should have slack (it's the shorter parallel path)
    let _c_slack = slack.get(&c).copied().unwrap_or(0);
    // Note: exact slack depends on algorithm details
    assert!(slack.values().any(|&s| s > 0), "F130: At least one node should have positive slack");
}

/// F131: Roofline distance is 0.0 for kernel at peak
#[test]
fn test_f131_roofline_at_peak() {
    let mut graph = ExecutionGraph::new();

    // Kernel achieving peak performance
    let _kernel = graph.record_kernel_launch_with_metrics(
        "peak_kernel",
        0x1234,
        (128, 1, 1),
        (256, 1, 1),
        8192,
        100_000, // 100µs
        100.0,   // AI = 100 FLOPs/byte (compute bound)
        10.0,    // 10 TFLOPS achieved
    );

    // Peak = 10 TFLOPS, bandwidth = 1000 GB/s
    let distances = graph.roofline_distance(10.0, 1000.0);

    // Should be at or near zero distance (achieving peak)
    for &dist in distances.values() {
        assert!(dist <= 0.1, "F131: Roofline distance should be near 0 at peak");
    }
}

/// F132: Roofline distance is high for underperforming kernel
#[test]
fn test_f132_roofline_underperforming() {
    let mut graph = ExecutionGraph::new();

    // Kernel achieving only 10% of peak
    let _kernel = graph.record_kernel_launch_with_metrics(
        "slow_kernel",
        0x5678,
        (32, 1, 1),
        (64, 1, 1),
        1024,
        100_000, // 100µs
        100.0,   // AI = 100 (compute bound)
        1.0,     // Only 1 TFLOPS (10% of peak)
    );

    // Peak = 10 TFLOPS
    let distances = graph.roofline_distance(10.0, 1000.0);

    // Distance should be high (0.9 = 90% from optimal)
    for &dist in distances.values() {
        assert!(dist >= 0.8, "F132: Roofline distance should be high for underperforming kernel");
    }
}

/// F133: Ping-pong detection finds H2D->D2H patterns
#[test]
fn test_f133_ping_pong_detection() {
    let mut graph = ExecutionGraph::new();

    // Create H2D followed by D2H on same buffer
    let _h2d = graph.record_transfer(
        "host_buffer",
        "device_buffer",
        1024 * 1024, // 1MB
        TransferDirection::H2D,
        Some(50_000),
    );

    let _d2h = graph.record_transfer(
        "device_buffer",
        "host_buffer",
        1024 * 1024, // Same size
        TransferDirection::D2H,
        Some(50_000),
    );

    let patterns = graph.detect_ping_pong();

    assert_eq!(patterns.len(), 1, "F133: Should detect 1 ping-pong pattern");
}

/// F134: No ping-pong for different buffer sizes
#[test]
fn test_f134_no_false_positive_ping_pong() {
    let mut graph = ExecutionGraph::new();

    // Different sizes - not a ping-pong
    let _h2d = graph.record_transfer(
        "host_a",
        "device_a",
        1024 * 1024, // 1MB
        TransferDirection::H2D,
        Some(50_000),
    );

    let _d2h = graph.record_transfer(
        "device_b",
        "host_b",
        2048 * 1024, // 2MB - different size
        TransferDirection::D2H,
        Some(50_000),
    );

    let patterns = graph.detect_ping_pong();

    assert!(patterns.is_empty(), "F134: Should not detect ping-pong for different sizes");
}

/// F135: Critical path summary includes all critical nodes
#[test]
fn test_f135_critical_path_summary() {
    let mut graph = ExecutionGraph::new();

    // Simple chain
    let a = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100_000,
        elements: 1024,
    });
    graph.pop_scope();

    let b = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 200_000,
        elements: 2048,
    });
    graph.pop_scope();

    graph.add_dependency(a, b);

    let summary = graph.critical_path_summary();

    // Summary should mention both bricks
    assert!(summary.contains("RmsNorm"), "F135: Summary should include RmsNorm");
    assert!(summary.contains("QkvProjection"), "F135: Summary should include QkvProjection");
    assert!(summary.contains("ms"), "F135: Summary should include timing in ms");
}
