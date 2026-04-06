use super::super::super::super::*;

// =========================================================================
// F279-F281: WatermarkedBuffer, ExecutionGraph, AttentionTraceConfig
// =========================================================================

/// F279: WatermarkedBuffer full API coverage
#[test]
fn test_f279_watermarked_buffer_api() {
    let wm = BufferWatermarks { low: 100, high: 1000 };
    let mut buf = WatermarkedBuffer::new(wm);

    // Test len and is_empty
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());

    // Test write
    buf.write(&[1, 2, 3, 4, 5]);
    assert_eq!(buf.len(), 5);
    assert!(!buf.is_empty());

    // Test watermarks accessor
    let retrieved = buf.watermarks();
    assert_eq!(retrieved.low, 100);
    assert_eq!(retrieved.high, 1000);

    // Test drain
    let drained = buf.drain(3);
    assert_eq!(drained, vec![1, 2, 3]);
    assert_eq!(buf.len(), 2);

    // Test drain more than available
    let drained2 = buf.drain(100);
    assert_eq!(drained2.len(), 2);
    assert!(buf.is_empty());

    // Test clear
    buf.write(&[10, 20, 30]);
    assert_eq!(buf.len(), 3);
    buf.clear();
    assert!(buf.is_empty());

    // Test pressure_level
    buf.write(&vec![0u8; 600]);
    let pressure = buf.pressure_level();
    assert!(pressure > 0.0 && pressure < 1.0);
}

/// F280: ExecutionGraph node and edge coverage
#[test]
fn test_f280_execution_graph_node_types() {
    let mut graph = ExecutionGraph::new();

    // Add various node types
    let root = graph.add_node(ExecutionNode::Layer { index: 0 });
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 1000,
        elements: 1024,
    });
    let kernel = graph.add_node(ExecutionNode::Kernel {
        name: "matmul".to_string(),
        ptx_hash: 12345,
        grid: (1, 1, 1),
        block: (256, 1, 1),
        shared_mem: 4096,
        timing_ns: Some(500),
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    let func = graph.add_node(ExecutionNode::Function {
        name: "forward".to_string(),
        file: Some("model.rs".to_string()),
        line: Some(100),
    });
    let transfer = graph.add_node(ExecutionNode::Transfer {
        src: "CPU".to_string(),
        dst: "GPU".to_string(),
        bytes: 4096,
        direction: TransferDirection::H2D,
        timing_ns: Some(200),
    });

    // Add edges of different types
    graph.add_edge(root, brick, EdgeType::Contains);
    graph.add_edge(brick, kernel, EdgeType::Launches);
    graph.add_edge(root, func, EdgeType::Calls);
    graph.add_edge(
        func,
        transfer,
        EdgeType::Transfer { bytes: 4096, direction: TransferDirection::H2D },
    );
    graph.add_edge(kernel, transfer, EdgeType::DependsOn);

    // Verify node IDs are sequential
    assert_eq!(root.0, 0);
    assert_eq!(brick.0, 1);
    assert_eq!(kernel.0, 2);
    assert_eq!(func.0, 3);
    assert_eq!(transfer.0, 4);
}

/// F281: AttentionTraceConfig filtering
#[test]
fn test_f281_attention_trace_config_filtering() {
    // Test with specific layers/heads
    let config = AttentionTraceConfig {
        top_k: 10,
        layers: Some(vec![0, 5, 10, 15]),
        heads: Some(vec![0, 1]),
        weight_threshold: 0.01,
    };

    assert!(config.should_trace_layer(0));
    assert!(config.should_trace_layer(5));
    assert!(!config.should_trace_layer(3));
    assert!(config.should_trace_head(0));
    assert!(config.should_trace_head(1));
    assert!(!config.should_trace_head(2));

    // Test with None (trace all)
    let config_all =
        AttentionTraceConfig { top_k: 5, layers: None, heads: None, weight_threshold: 0.05 };

    assert!(config_all.should_trace_layer(99));
    assert!(config_all.should_trace_head(31));
}
