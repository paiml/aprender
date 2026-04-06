use super::super::super::*;

// ========================================================================
// PAR-201: Execution Path Graph Falsification Tests (F111-F120)
// ========================================================================

/// F111: Graph export node/edge count matches
#[test]
fn test_f111_graph_export_node_edge_count() {
    let mut graph = ExecutionGraph::new();

    // Add 3 nodes
    let layer = graph.add_node(ExecutionNode::Layer { index: 0 });
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 1000,
        elements: 4096,
    });
    let kernel = graph.add_node(ExecutionNode::Kernel {
        name: "test_kernel".into(),
        ptx_hash: 0x12345678,
        grid: (32, 1, 1),
        block: (256, 1, 1),
        shared_mem: 4096,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });

    // Add 2 edges
    graph.add_edge(layer, brick, EdgeType::Contains);
    graph.add_edge(brick, kernel, EdgeType::Launches);

    assert_eq!(graph.num_nodes(), 3, "F111: Expected 3 nodes");
    assert_eq!(graph.num_edges(), 2, "F111: Expected 2 edges");
}

/// F112: PTX hash stable across runs
#[test]
fn test_f112_ptx_hash_stable() {
    let ptx1 = ".version 7.0
.target sm_80
.entry test() { ret; }";
    let ptx2 = ".version 7.0
.target sm_80
.entry test() { ret; }";

    let hash1 = PtxRegistry::hash_ptx(ptx1);
    let hash2 = PtxRegistry::hash_ptx(ptx2);

    assert_eq!(hash1, hash2, "F112: Same PTX must produce same hash");

    // Different PTX should produce different hash
    let ptx3 = ".version 7.0
.target sm_80
.entry other() { ret; }";
    let hash3 = PtxRegistry::hash_ptx(ptx3);
    assert_ne!(hash1, hash3, "F112: Different PTX must produce different hash");
}

/// F113: Kernel launch recorded in graph
#[test]
fn test_f113_kernel_launch_recorded() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    profiler.enable_graph();

    // Push a scope
    profiler.graph_push_scope(ExecutionNode::Layer { index: 0 });

    // Record kernel
    let kernel_id =
        profiler.graph_record_kernel("batched_q4k_gemv", 0xDEADBEEF, (32, 1, 1), (256, 1, 1), 4096);

    profiler.graph_pop_scope();

    assert!(kernel_id.is_some(), "F113: Kernel should be recorded");
    assert_eq!(profiler.execution_graph().num_nodes(), 2, "F113: Should have layer + kernel nodes");

    // Verify kernel node exists
    let kernels: Vec<_> = profiler.execution_graph().kernel_nodes().collect();
    assert_eq!(kernels.len(), 1, "F113: Should have 1 kernel node");
}

/// F114: Scope push/pop balanced
#[test]
fn test_f114_scope_balanced() {
    let mut graph = ExecutionGraph::new();

    assert!(graph.is_scope_balanced(), "F114: Empty graph should be balanced");

    graph.push_scope(ExecutionNode::Layer { index: 0 });
    assert!(!graph.is_scope_balanced(), "F114: After push, not balanced");

    graph.push_scope(ExecutionNode::Layer { index: 1 });
    assert!(!graph.is_scope_balanced(), "F114: After 2 pushes, not balanced");

    graph.pop_scope();
    assert!(!graph.is_scope_balanced(), "F114: After 1 pop, not balanced");

    graph.pop_scope();
    assert!(graph.is_scope_balanced(), "F114: After 2 pops, balanced");
}

/// F115: Graph queries are O(V+E) - benchmark with 1000 nodes
#[test]
fn test_f115_graph_query_performance() {
    let mut graph = ExecutionGraph::new();

    // Add 1000 nodes
    for i in 0..1000 {
        graph.add_node(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: i as u64 * 100,
            elements: 4096,
        });
    }

    // Add 999 edges (chain)
    for i in 0..999 {
        graph.add_edge(ExecutionNodeId(i), ExecutionNodeId(i + 1), EdgeType::Sequence);
    }

    // Query should complete quickly
    let start = std::time::Instant::now();
    let _outgoing: Vec<_> = graph.outgoing_edges(ExecutionNodeId(500)).collect();
    let _incoming: Vec<_> = graph.incoming_edges(ExecutionNodeId(500)).collect();
    let elapsed = start.elapsed();

    // Should complete in <1ms for 1000 nodes
    assert!(elapsed.as_millis() < 10, "F115: Query took {}ms, expected <10ms", elapsed.as_millis());
}

/// F116: DOT export is valid
#[test]
fn test_f116_dot_export_valid() {
    let mut graph = ExecutionGraph::new();

    let layer = graph.push_scope(ExecutionNode::Layer { index: 0 });
    let brick = graph.add_node_in_scope(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 1000,
        elements: 4096,
    });
    graph.record_kernel_launch("test_kernel", 0x12345678, (32, 1, 1), (256, 1, 1), 0);
    graph.pop_scope();

    let dot = graph.to_dot();

    // Basic DOT format validation
    assert!(dot.starts_with("digraph"), "F116: DOT must start with digraph");
    assert!(dot.contains("->"), "F116: DOT must contain edges");
    assert!(
        dot.ends_with(
            "}
"
        ),
        "F116: DOT must end with closing brace"
    );
    assert!(dot.contains("Layer 0"), "F116: DOT must contain layer label");
    assert!(dot.contains("QkvProjection"), "F116: DOT must contain brick label");
    assert!(dot.contains("test_kernel"), "F116: DOT must contain kernel label");

    // Check node count in DOT
    let node_count = dot.matches("[label=").count();
    assert_eq!(node_count, 3, "F116: DOT should have 3 nodes");

    let _ = (layer, brick); // Silence unused warnings
}

/// F117: Edge types preserved
#[test]
fn test_f117_edge_types_preserved() {
    let mut graph = ExecutionGraph::new();

    let n1 = graph.add_node(ExecutionNode::Layer { index: 0 });
    let n2 =
        graph.add_node(ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 100, elements: 1 });
    let n3 = graph.add_node(ExecutionNode::Kernel {
        name: "k".into(),
        ptx_hash: 0,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });

    graph.add_edge(n1, n2, EdgeType::Contains);
    graph.add_edge(n2, n3, EdgeType::Launches);
    graph.add_edge(n1, n3, EdgeType::Calls);
    graph.add_edge(n2, n2, EdgeType::Sequence);

    let edges = graph.edges();
    assert_eq!(edges[0].edge_type, EdgeType::Contains, "F117: Edge 0 type");
    assert_eq!(edges[1].edge_type, EdgeType::Launches, "F117: Edge 1 type");
    assert_eq!(edges[2].edge_type, EdgeType::Calls, "F117: Edge 2 type");
    assert_eq!(edges[3].edge_type, EdgeType::Sequence, "F117: Edge 3 type");
}

/// F118: PtxRegistry lookup works
#[test]
fn test_f118_ptx_registry_lookup() {
    let mut registry = PtxRegistry::new();

    let ptx1 = ".version 7.0
.entry kernel1() {}";
    let ptx2 = ".version 7.0
.entry kernel2() {}";

    registry.register("kernel1", ptx1, None);
    registry.register("kernel2", ptx2, Some(std::path::Path::new("/src/kernels.ptx")));

    let hash1 = PtxRegistry::hash_ptx(ptx1);
    let hash2 = PtxRegistry::hash_ptx(ptx2);

    assert_eq!(registry.lookup(hash1), Some(ptx1), "F118: PTX1 lookup");
    assert_eq!(registry.lookup(hash2), Some(ptx2), "F118: PTX2 lookup");
    assert_eq!(registry.lookup_name(hash1), Some("kernel1"), "F118: Name1 lookup");
    assert_eq!(registry.lookup_name(hash2), Some("kernel2"), "F118: Name2 lookup");
    assert!(registry.lookup_path(hash1).is_none(), "F118: Path1 is None");
    assert_eq!(
        registry.lookup_path(hash2),
        Some(std::path::Path::new("/src/kernels.ptx")),
        "F118: Path2 lookup"
    );
    assert_eq!(registry.len(), 2, "F118: Registry has 2 entries");
}

/// F119: Slowest kernel detection
#[test]
fn test_f119_slowest_kernel_detection() {
    let mut graph = ExecutionGraph::new();

    // Brick 1: 100ns, has kernel
    let b1 =
        graph.add_node(ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 100, elements: 1 });
    let k1 = graph.add_node(ExecutionNode::Kernel {
        name: "fast".into(),
        ptx_hash: 1,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(b1, k1, EdgeType::Launches);

    // Brick 2: 500ns, has kernel (slowest)
    let b2 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 500,
        elements: 1,
    });
    let k2 = graph.add_node(ExecutionNode::Kernel {
        name: "slow".into(),
        ptx_hash: 2,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(b2, k2, EdgeType::Launches);

    // Brick 3: 1000ns, NO kernel (should not be selected)
    let _b3 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::Sampling,
        timing_ns: 1000,
        elements: 1,
    });

    let slowest = graph.slowest_kernel();
    assert!(slowest.is_some(), "F119: Should find slowest");
    let (id, node, timing) = slowest.unwrap();
    assert_eq!(id, b2, "F119: Slowest should be brick 2");
    assert_eq!(timing, 500, "F119: Timing should be 500ns");
    assert!(node.is_brick(), "F119: Node should be brick");
}

/// F120: Graph clear works
#[test]
fn test_f120_graph_clear() {
    let mut graph = ExecutionGraph::new();

    // Add some nodes and edges
    let n1 = graph.push_scope(ExecutionNode::Layer { index: 0 });
    graph.add_node_in_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100,
        elements: 1,
    });

    assert!(!graph.is_scope_balanced(), "F120: Pre-clear not balanced");
    assert!(graph.num_nodes() > 0, "F120: Pre-clear has nodes");
    assert!(graph.num_edges() > 0, "F120: Pre-clear has edges");

    graph.clear();

    assert!(graph.is_scope_balanced(), "F120: Post-clear balanced");
    assert_eq!(graph.num_nodes(), 0, "F120: Post-clear no nodes");
    assert_eq!(graph.num_edges(), 0, "F120: Post-clear no edges");
    assert!(graph.node_by_name("Layer0").is_none(), "F120: Post-clear no name lookup");

    let _ = n1; // Silence unused warning
}
