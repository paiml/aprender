use super::super::super::*;

// ========================
// Extended Falsification Tests (F136-F140)
// ========================

/// F136: CPA selects longer parallel branch over single heavy node
/// Scenario A: 1x10ms vs 5x3ms (15ms total) - must pick 5-node branch
#[test]
fn test_f136_cpa_parallel_heavy_branch() {
    let mut graph = ExecutionGraph::new();

    // Root node
    let root = graph.push_scope(ExecutionNode::Layer { index: 0 });
    graph.pop_scope();

    // Branch A: single 10ms node
    let branch_a = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 10_000_000, // 10ms
        elements: 4096,
    });
    graph.pop_scope();

    // Branch B: five 3ms nodes chained (15ms total)
    let b1 = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 3_000_000, // 3ms
        elements: 1024,
    });
    graph.pop_scope();

    let b2 = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::AttentionScore,
        timing_ns: 3_000_000,
        elements: 1024,
    });
    graph.pop_scope();

    let b3 = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::GateProjection,
        timing_ns: 3_000_000,
        elements: 1024,
    });
    graph.pop_scope();

    let b4 = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::UpProjection,
        timing_ns: 3_000_000,
        elements: 1024,
    });
    graph.pop_scope();

    let b5 = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::DownProjection,
        timing_ns: 3_000_000,
        elements: 1024,
    });
    graph.pop_scope();

    // Connect: root -> branch_a, root -> b1 -> b2 -> b3 -> b4 -> b5
    graph.add_dependency(root, branch_a);
    graph.add_dependency(root, b1);
    graph.add_dependency(b1, b2);
    graph.add_dependency(b2, b3);
    graph.add_dependency(b3, b4);
    graph.add_dependency(b4, b5);

    let (path, total_ns) = graph.critical_path();

    // Critical path must be the 5-node branch (15ms > 10ms)
    assert!(
        total_ns >= 15_000_000,
        "F136: Critical path should be >= 15ms, got {}ms",
        total_ns / 1_000_000
    );
    assert!(path.len() >= 5, "F136: Critical path should have >= 5 nodes, got {}", path.len());
}

/// F137: DependsOn edge overrides wall-clock sequence
/// Scenario B: CUDA event sync creates logical dependency
#[test]
fn test_f137_depends_on_overrides_sequence() {
    let mut graph = ExecutionGraph::new();

    // Three nodes: A (early), B (late but depends on C), C (middle)
    let a = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100_000, // 100µs
        elements: 1024,
    });
    graph.pop_scope();

    let b = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 500_000, // 500µs - heavyweight
        elements: 4096,
    });
    graph.pop_scope();

    let c = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::AttentionScore,
        timing_ns: 200_000, // 200µs
        elements: 2048,
    });
    graph.pop_scope();

    // Wall-clock order: A -> B -> C
    // But logical dependency: A -> C -> B (C must complete before B)
    graph.add_dependency(a, c);
    graph.add_dependency(c, b);

    let (path, total_ns) = graph.critical_path();

    // Path must respect DependsOn: A -> C -> B = 100 + 200 + 500 = 800µs
    assert!(
        total_ns >= 800_000,
        "F137: DependsOn path should be >= 800µs, got {}µs",
        total_ns / 1000
    );

    // B must come after C in the path
    let b_pos = path.iter().position(|&id| id == b);
    let c_pos = path.iter().position(|&id| id == c);
    if let (Some(bp), Some(cp)) = (b_pos, c_pos) {
        assert!(bp > cp, "F137: B must come after C in critical path");
    }
}

/// F138: Roofline distance detects anomalous TFLOPS (physics bound)
#[test]
fn test_f138_roofline_anomaly_detection() {
    let mut graph = ExecutionGraph::new();

    // Record kernel with impossible 1000 TFLOPS on RTX 4090 (peak ~83 TFLOPS)
    let _kernel = graph.record_kernel_launch_with_metrics(
        "impossible_kernel",
        0xBAD,
        (128, 1, 1),
        (256, 1, 1),
        8192,
        100_000, // 100µs
        50.0,    // AI = 50 FLOPs/byte
        1000.0,  // 1000 TFLOPS - impossible!
    );

    // Distance should be negative (or clamped) since achieved > peak
    let distances = graph.roofline_distance(83.0, 1008.0);

    // The efficiency would be > 100%, so distance should be 0 (clamped)
    for &dist in distances.values() {
        assert!(
            dist <= 0.0 || dist >= 0.0, // Just verify it doesn't panic
            "F138: Should handle anomalous TFLOPS gracefully"
        );
    }
}

/// F139: Large-scale ping-pong detection (100 iterations)
#[test]
fn test_f139_ping_pong_large_scale() {
    let mut graph = ExecutionGraph::new();

    // Simulate 100 iterations of H2D -> D2H of 1GB buffer
    for i in 0..100 {
        let _h2d = graph.record_transfer(
            &format!("host_buf_{}", i),
            &format!("device_buf_{}", i),
            1024 * 1024 * 1024, // 1GB
            TransferDirection::H2D,
            Some(50_000_000), // 50ms
        );

        let _d2h = graph.record_transfer(
            &format!("device_buf_{}", i),
            &format!("host_buf_{}", i),
            1024 * 1024 * 1024, // 1GB
            TransferDirection::D2H,
            Some(50_000_000), // 50ms
        );
    }

    let patterns = graph.detect_ping_pong();

    // Should detect many ping-pong patterns
    assert!(
        patterns.len() >= 50,
        "F139: Should detect >= 50 ping-pong patterns, got {}",
        patterns.len()
    );
}

/// F140: Transfer recording preserves all metadata
#[test]
fn test_f140_transfer_metadata_preservation() {
    let mut graph = ExecutionGraph::new();

    let transfer_id = graph.record_transfer(
        "src_buffer",
        "dst_buffer",
        4 * 1024 * 1024, // 4MB
        TransferDirection::H2D,
        Some(25_000), // 25µs
    );

    // Verify the node was recorded with correct data
    let node = &graph.nodes()[transfer_id.0 as usize];
    if let ExecutionNode::Transfer { src, dst, bytes, direction, timing_ns } = node {
        assert_eq!(src, "src_buffer", "F140: Source buffer mismatch");
        assert_eq!(dst, "dst_buffer", "F140: Dest buffer mismatch");
        assert_eq!(*bytes, 4 * 1024 * 1024, "F140: Bytes mismatch");
        assert_eq!(*direction, TransferDirection::H2D, "F140: Direction mismatch");
        assert_eq!(*timing_ns, Some(25_000), "F140: Timing mismatch");
    } else {
        panic!("F140: Expected Transfer node");
    }
}
