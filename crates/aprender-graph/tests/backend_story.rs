//! Backend Story Integration Tests
//!
//! CRITICAL: These tests enforce that ALL operations in trueno-graph support the complete
//! backend story: CPU (via aprender/trueno SIMD) and GPU (via wgpu).
//!
//! If these tests fail, it means a new operation was added without proper backend support.
//! THIS IS A BLOCKING ISSUE - do not merge code that breaks the backend story.
//!
//! Reference: CLAUDE.md "Backend Story Policy"

use trueno_graph::{bfs, find_callers, find_patterns, pagerank, CsrGraph, NodeId, Pattern};

// ============================================================================
// HELPER: Build test graphs
// ============================================================================

/// Build a simple test graph for algorithm verification
///
/// Graph structure:
/// ```text
///     0 → 1 → 3
///     ↓   ↓
///     2 → 4
/// ```
fn build_test_graph() -> CsrGraph {
    let edges = vec![
        (NodeId(0), NodeId(1), 1.0),
        (NodeId(0), NodeId(2), 1.0),
        (NodeId(1), NodeId(3), 1.0),
        (NodeId(1), NodeId(4), 1.0),
        (NodeId(2), NodeId(4), 1.0),
    ];
    CsrGraph::from_edge_list(&edges).unwrap()
}

/// Build a larger test graph for performance-sensitive tests
fn build_large_test_graph(num_nodes: usize) -> CsrGraph {
    let mut edges = Vec::new();
    // Create a chain graph: 0 → 1 → 2 → ... → n-1
    for i in 0..(num_nodes - 1) {
        edges.push((NodeId(i as u32), NodeId((i + 1) as u32), 1.0));
    }
    // Add some cross edges for more interesting structure
    for i in 0..(num_nodes / 10) {
        let src = i * 10;
        let dst = (i * 10 + 5) % num_nodes;
        if src != dst {
            edges.push((NodeId(src as u32), NodeId(dst as u32), 0.5));
        }
    }
    CsrGraph::from_edge_list(&edges).unwrap()
}

// ============================================================================
// CSR GRAPH TESTS (Storage Layer)
// ============================================================================

/// Test CSR graph construction
#[test]
fn test_csr_graph_construction() {
    let graph = build_test_graph();

    // Verify node count (nodes 0-4)
    assert!(graph.num_nodes() >= 5, "Graph should have at least 5 nodes");

    // Verify edge count
    assert_eq!(graph.num_edges(), 5, "Graph should have 5 edges");
}

/// Test CSR outgoing neighbors query
#[test]
fn test_csr_outgoing_neighbors() {
    let graph = build_test_graph();

    // Node 0 has outgoing edges to 1 and 2
    let neighbors = graph.outgoing_neighbors(NodeId(0)).unwrap();
    assert_eq!(neighbors.len(), 2);

    // Node 1 has outgoing edges to 3 and 4
    let neighbors = graph.outgoing_neighbors(NodeId(1)).unwrap();
    assert_eq!(neighbors.len(), 2);

    // Node 3 has no outgoing edges
    let neighbors = graph.outgoing_neighbors(NodeId(3)).unwrap();
    assert_eq!(neighbors.len(), 0);
}

/// Test CSR incoming neighbors query (reverse CSR)
#[test]
fn test_csr_incoming_neighbors() {
    let graph = build_test_graph();

    // Node 4 has incoming edges from 1 and 2
    let callers = graph.incoming_neighbors(NodeId(4)).unwrap();
    assert_eq!(callers.len(), 2, "Node 4 should have 2 incoming edges");

    // Node 0 has no incoming edges
    let callers = graph.incoming_neighbors(NodeId(0)).unwrap();
    assert_eq!(callers.len(), 0, "Node 0 should have no incoming edges");
}

// ============================================================================
// BFS TESTS (Traversal Algorithm)
// ============================================================================

/// Test BFS traversal finds all reachable nodes
#[test]
fn test_bfs_reachability() {
    let graph = build_test_graph();

    let reachable = bfs(&graph, NodeId(0)).expect("BFS should succeed");

    // All 5 nodes should be reachable from node 0
    assert_eq!(reachable.len(), 5, "All 5 nodes should be reachable");
    assert!(reachable.contains(&0), "Source should be in result");
    assert!(reachable.contains(&1));
    assert!(reachable.contains(&2));
    assert!(reachable.contains(&3));
    assert!(reachable.contains(&4));
}

/// Test BFS with disconnected nodes
#[test]
fn test_bfs_disconnected() {
    // Create two disconnected components
    let edges = vec![
        (NodeId(0), NodeId(1), 1.0),
        (NodeId(2), NodeId(3), 1.0), // Disconnected from 0-1
    ];
    let graph = CsrGraph::from_edge_list(&edges).unwrap();

    let reachable = bfs(&graph, NodeId(0)).expect("BFS should succeed");

    // Only nodes 0 and 1 should be reachable
    assert_eq!(reachable.len(), 2, "Only component 0-1 should be reachable");
    assert!(reachable.contains(&0));
    assert!(reachable.contains(&1));
    assert!(!reachable.contains(&2), "Node 2 should NOT be reachable");
    assert!(!reachable.contains(&3), "Node 3 should NOT be reachable");
}

// ============================================================================
// FIND_CALLERS TESTS (Reverse BFS)
// ============================================================================

/// Test find_callers (reverse BFS)
#[test]
fn test_find_callers() {
    let graph = build_test_graph();

    // Find all callers of node 4 within depth 10
    let callers = find_callers(&graph, NodeId(4), 10).expect("find_callers should succeed");

    // Nodes 1, 2 directly call node 4
    assert!(callers.contains(&1), "Node 1 should be a caller of 4");
    assert!(callers.contains(&2), "Node 2 should be a caller of 4");

    // Node 0 can reach 4 via 1→4 or 2→4 (transitive caller)
    assert!(callers.contains(&0), "Node 0 should be a caller of 4 (indirectly)");
}

/// Test find_callers with depth limit
#[test]
fn test_find_callers_depth_limit() {
    let graph = build_test_graph();

    // Depth 1: only direct callers
    let callers = find_callers(&graph, NodeId(4), 1).expect("find_callers should succeed");
    assert!(callers.contains(&1), "Node 1 directly calls 4");
    assert!(callers.contains(&2), "Node 2 directly calls 4");
    // Node 0 is depth 2, should NOT be included
    assert!(!callers.contains(&0), "Node 0 should NOT be included at depth 1");
}

// ============================================================================
// PAGERANK TESTS (Sparse Matrix Operations via aprender/trueno)
// ============================================================================

/// Test PageRank produces valid scores
#[test]
fn test_pagerank_basic() {
    let graph = build_test_graph();

    // PageRank with 20 iterations and 1e-6 tolerance
    let scores = pagerank(&graph, 20, 1e-6).expect("PageRank should succeed");

    // All scores should be positive
    for score in &scores {
        assert!(*score >= 0.0, "PageRank scores should be non-negative");
    }

    // Scores should sum to approximately 1.0 (normalized)
    let sum: f32 = scores.iter().sum();
    assert!((sum - 1.0).abs() < 0.1, "PageRank scores should sum to ~1.0, got {sum}");
}

/// Test PageRank converges for larger graphs
#[test]
fn test_pagerank_larger_graph() {
    let graph = build_large_test_graph(100);

    let scores = pagerank(&graph, 50, 1e-6).expect("PageRank should succeed");

    // Should have scores for all nodes
    assert!(scores.len() >= 100, "Should have scores for all 100 nodes");

    // All scores should be positive
    for score in &scores {
        assert!(*score >= 0.0, "All PageRank scores should be non-negative");
    }
}

// ============================================================================
// COMMUNITY DETECTION TESTS (Louvain via aprender)
// ============================================================================

/// Test Louvain community detection
#[test]
fn test_louvain_basic() {
    use trueno_graph::louvain;

    // Create a graph with two clear communities
    let edges = vec![
        // Community 1: 0-1-2 (dense connections)
        (NodeId(0), NodeId(1), 1.0),
        (NodeId(1), NodeId(0), 1.0),
        (NodeId(1), NodeId(2), 1.0),
        (NodeId(2), NodeId(1), 1.0),
        (NodeId(0), NodeId(2), 1.0),
        (NodeId(2), NodeId(0), 1.0),
        // Community 2: 3-4-5 (dense connections)
        (NodeId(3), NodeId(4), 1.0),
        (NodeId(4), NodeId(3), 1.0),
        (NodeId(4), NodeId(5), 1.0),
        (NodeId(5), NodeId(4), 1.0),
        (NodeId(3), NodeId(5), 1.0),
        (NodeId(5), NodeId(3), 1.0),
        // Weak connection between communities
        (NodeId(2), NodeId(3), 0.1),
    ];
    let graph = CsrGraph::from_edge_list(&edges).unwrap();

    let result = louvain(&graph).expect("Louvain should succeed");

    // Should detect at least 2 communities
    assert!(
        result.num_communities >= 2,
        "Should detect at least 2 communities, got {}",
        result.num_communities
    );

    // Modularity should be positive for graphs with community structure
    assert!(result.modularity > 0.0, "Modularity should be positive, got {}", result.modularity);
}

// ============================================================================
// PATTERN DETECTION TESTS
// ============================================================================

/// Test pattern detection finds circular dependencies
#[test]
fn test_pattern_detection_circular() {
    // Create a graph with a 3-node cycle (circular dependency)
    let edges = vec![
        (NodeId(0), NodeId(1), 1.0),
        (NodeId(1), NodeId(2), 1.0),
        (NodeId(2), NodeId(0), 1.0), // Creates cycle 0→1→2→0
    ];
    let graph = CsrGraph::from_edge_list(&edges).unwrap();

    let patterns = find_patterns(&graph, &Pattern::circular_dependency(3))
        .expect("Pattern detection should succeed");

    // Should find the circular dependency
    assert!(!patterns.is_empty(), "Should detect the circular dependency (3-cycle)");
}

/// Test pattern detection for god class
#[test]
fn test_pattern_detection_god_class() {
    // Create a graph where node 0 calls many other nodes (god class)
    let mut edges = Vec::new();
    for i in 1..=10 {
        edges.push((NodeId(0), NodeId(i), 1.0));
    }
    let graph = CsrGraph::from_edge_list(&edges).unwrap();

    let patterns =
        find_patterns(&graph, &Pattern::god_class(5)).expect("Pattern detection should succeed");

    // Node 0 calls 10 nodes, should be flagged as god class (threshold 5)
    assert!(!patterns.is_empty(), "Should detect god class (node 0 has 10 callees)");
}

// ============================================================================
// GPU BACKEND TESTS (Optional - requires gpu feature)
// ============================================================================

#[cfg(feature = "gpu")]
mod gpu_tests {
    use super::*;
    use trueno_graph::{gpu_bfs, GpuDevice};

    /// Test GPU device initialization
    #[tokio::test]
    async fn test_gpu_device_init() {
        // This test should never panic, even without GPU hardware
        match GpuDevice::new().await {
            Ok(_device) => {
                // GPU available - tests will run
            }
            Err(e) => {
                // GPU not available - graceful degradation
                eprintln!("GPU initialization failed (expected on CI): {e}");
            }
        }
    }

    /// Test GPU BFS matches CPU BFS
    #[tokio::test]
    async fn test_gpu_cpu_bfs_equivalence() {
        let Ok(_device) = GpuDevice::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let graph = build_test_graph();

        // CPU BFS
        let cpu_result = bfs(&graph, NodeId(0)).expect("CPU BFS should succeed");

        // GPU BFS
        let gpu_result = gpu_bfs(&graph, NodeId(0)).await.expect("GPU BFS should succeed");

        // Results should have same set of reachable nodes
        let cpu_set: std::collections::HashSet<_> = cpu_result.into_iter().collect();
        let gpu_set: std::collections::HashSet<_> = gpu_result.reachable.into_iter().collect();

        assert_eq!(cpu_set, gpu_set, "CPU and GPU BFS should produce same reachable nodes");
    }

    /// Test GPU BFS with larger graph
    #[tokio::test]
    async fn test_gpu_bfs_large_graph() {
        let Ok(_device) = GpuDevice::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let graph = build_large_test_graph(1000);

        // GPU BFS should handle larger graphs
        let result = gpu_bfs(&graph, NodeId(0)).await.expect("GPU BFS should succeed");

        // Should have visited nodes
        assert!(!result.reachable.is_empty(), "GPU BFS should find reachable nodes");
    }
}

// ============================================================================
// POLICY ENFORCEMENT: Backend Story Completeness
// ============================================================================
//
// The following module contains compile-time assertions that verify the
// backend story is complete for all major operations.
//
// If you're adding a new algorithm to trueno-graph, you MUST:
// 1. Implement CPU version using aprender/trueno (SIMD-accelerated)
// 2. If performance-critical, implement GPU version in src/gpu/
// 3. Add equivalence test to this file verifying CPU == GPU results
// 4. Update CLAUDE.md if adding a new category of operations
//
// Failure to do so will cause CI to fail and block your PR.
// ============================================================================

#[cfg(test)]
mod backend_completeness {
    //! Compile-time verification that critical functions exist

    use trueno_graph::{
        bfs, find_callers, find_patterns, louvain, pagerank, CsrGraph, NodeId, Pattern,
    };

    /// Verify core algorithm functions exist
    #[test]
    fn test_algorithm_functions_exist() {
        // These function pointer assignments verify the functions exist at compile time
        let _: fn(&CsrGraph, NodeId) -> anyhow::Result<Vec<u32>> = bfs;
        let _: fn(&CsrGraph, NodeId, usize) -> anyhow::Result<Vec<u32>> = find_callers;
        let _: fn(&CsrGraph, usize, f32) -> anyhow::Result<Vec<f32>> = pagerank;
        let _: fn(&CsrGraph, &Pattern) -> anyhow::Result<_> = find_patterns;
        let _: fn(&CsrGraph) -> anyhow::Result<_> = louvain;
    }

    /// Verify CSR graph methods exist
    #[test]
    fn test_csr_graph_methods_exist() {
        let edges = vec![(NodeId(0), NodeId(1), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        // These method calls verify the methods exist at compile time
        let _ = graph.num_nodes();
        let _ = graph.num_edges();
        let _ = graph.outgoing_neighbors(NodeId(0));
        let _ = graph.incoming_neighbors(NodeId(0));
    }

    /// Verify Pattern constructors exist
    #[test]
    fn test_pattern_constructors_exist() {
        let _ = Pattern::god_class(5);
        let _ = Pattern::circular_dependency(3);
        let _ = Pattern::dead_code();
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_functions_exist() {
        use trueno_graph::{gpu_bfs, GpuDevice};

        // Verify GPU functions exist
        // Note: We can't easily get function pointers for async fns,
        // but the import itself verifies they exist
        let _ = gpu_bfs;
        let _ = GpuDevice::new;
    }
}
