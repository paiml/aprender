
#[test]
fn test_density_directed() {
    // Directed: 0->1, 1->2 (2 edges, 3 nodes)
    let g = Graph::from_edges(&[(0, 1), (1, 2)], true);

    // Directed: density = m / (n*(n-1)) = 2 / (3*2) = 1/3
    assert!((g.density() - 1.0 / 3.0).abs() < 1e-6);
}

// Diameter Tests

#[test]
fn test_diameter_empty() {
    let g = Graph::new(false);
    assert_eq!(g.diameter(), None);
}

#[test]
fn test_diameter_single_node() {
    let g = Graph::from_edges(&[(0, 0)], false);
    assert_eq!(g.diameter(), Some(0));
}

#[test]
fn test_diameter_path_graph() {
    // Path: 0--1--2--3 has diameter 3
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
    assert_eq!(g.diameter(), Some(3));
}

#[test]
fn test_diameter_star_graph() {
    // Star graph: center to any leaf is 1, leaf to leaf is 2
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    assert_eq!(g.diameter(), Some(2));
}

#[test]
fn test_diameter_disconnected() {
    // Disconnected: (0--1) and (2--3)
    let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
    assert_eq!(g.diameter(), None); // Disconnected
}

#[test]
fn test_diameter_complete_graph() {
    // Complete graph K4: diameter is 1 (all nodes adjacent)
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
    assert_eq!(g.diameter(), Some(1));
}

// Clustering Coefficient Tests

#[test]
fn test_clustering_coefficient_empty() {
    let g = Graph::new(false);
    assert_eq!(g.clustering_coefficient(), 0.0);
}

#[test]
fn test_clustering_coefficient_triangle() {
    // Triangle: perfect clustering (coefficient = 1.0)
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
    assert!((g.clustering_coefficient() - 1.0).abs() < 1e-6);
}

#[test]
fn test_clustering_coefficient_star_graph() {
    // Star graph: no triangles (coefficient = 0.0)
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    assert_eq!(g.clustering_coefficient(), 0.0);
}

#[test]
fn test_clustering_coefficient_partial() {
    // Graph with one triangle among 4 nodes
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (0, 3)], false);

    // Node 0: 3 neighbors, 1 triangle (0-1-2)
    // Node 1: 2 neighbors, 1 triangle
    // Node 2: 2 neighbors, 1 triangle
    // Node 3: 1 neighbor, 0 triangles
    let cc = g.clustering_coefficient();
    assert!(cc > 0.0);
    assert!(cc < 1.0);
}

// Assortativity Tests

#[test]
fn test_assortativity_empty() {
    let g = Graph::new(false);
    assert_eq!(g.assortativity(), 0.0);
}

#[test]
fn test_assortativity_star_graph() {
    // Star graph: hub (deg 3) connects to leaves (deg 1)
    // Negative assortativity (high-degree connects to low-degree)
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    assert!(g.assortativity() < 0.0);
}

#[test]
fn test_assortativity_complete_graph() {
    // Complete graph K4: all nodes have same degree
    // Should have assortativity close to 0 (or NaN due to zero variance)
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
    let assort = g.assortativity();

    // All nodes have degree 3, so variance is 0
    // Assortativity is undefined but we return 0.0
    assert_eq!(assort, 0.0);
}

#[test]
fn test_assortativity_path_graph() {
    // Path: 0--1--2--3
    // Endpoints (deg 1) connect to middle (deg 2)
    // Middle nodes (deg 2) connect to mixed
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
    let assort = g.assortativity();

    // Should have negative assortativity
    assert!(assort < 0.0);
}

// ========================================================================
// Pathfinding Algorithm Tests
// ========================================================================

#[test]
fn test_shortest_path_direct_edge() {
    // Simplest case: direct edge between source and target
    let g = Graph::from_edges(&[(0, 1)], false);
    let path = g.shortest_path(0, 1).expect("path should exist");
    assert_eq!(path, vec![0, 1]);
    assert_eq!(path.len(), 2);
}

#[test]
fn test_shortest_path_same_node() {
    // Source == target should return single-node path
    let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
    let path = g.shortest_path(1, 1).expect("path should exist");
    assert_eq!(path, vec![1]);
}

#[test]
fn test_shortest_path_disconnected() {
    // No path between disconnected components
    let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
    assert!(g.shortest_path(0, 3).is_none());
    assert!(g.shortest_path(1, 2).is_none());
}

#[test]
fn test_shortest_path_invalid_nodes() {
    // Out-of-bounds node IDs should return None
    let g = Graph::from_edges(&[(0, 1)], false);
    assert!(g.shortest_path(0, 10).is_none());
    assert!(g.shortest_path(10, 0).is_none());
}

#[test]
fn test_shortest_path_multiple_paths() {
    // Graph with multiple paths of same length
    // 0 -- 1
    // |    |
    // 2 -- 3
    let g = Graph::from_edges(&[(0, 1), (1, 3), (0, 2), (2, 3)], false);
    let path = g.shortest_path(0, 3).expect("path should exist");

    // Both 0->1->3 and 0->2->3 are shortest paths (length 3)
    assert_eq!(path.len(), 3);
    assert_eq!(path[0], 0);
    assert_eq!(path[2], 3);
    assert!(path[1] == 1 || path[1] == 2); // Either path is valid
}

#[test]
fn test_shortest_path_linear_chain() {
    // Path graph: 0 -- 1 -- 2 -- 3 -- 4
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)], false);

    // Test various source-target pairs
    let path = g.shortest_path(0, 4).expect("path should exist");
    assert_eq!(path, vec![0, 1, 2, 3, 4]);

    let path = g.shortest_path(0, 2).expect("path should exist");
    assert_eq!(path, vec![0, 1, 2]);

    let path = g.shortest_path(1, 3).expect("path should exist");
    assert_eq!(path, vec![1, 2, 3]);
}

#[test]
fn test_shortest_path_triangle() {
    // Triangle graph
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);

    // All pairs should have path length 2
    let path = g.shortest_path(0, 1).expect("path should exist");
    assert_eq!(path.len(), 2);

    let path = g.shortest_path(0, 2).expect("path should exist");
    assert_eq!(path.len(), 2);

    let path = g.shortest_path(1, 2).expect("path should exist");
    assert_eq!(path.len(), 2);
}

#[test]
fn test_shortest_path_directed() {
    // Directed graph: 0 -> 1 -> 2
    let g = Graph::from_edges(&[(0, 1), (1, 2)], true);

    // Forward paths exist
    let path = g.shortest_path(0, 2).expect("forward path should exist");
    assert_eq!(path, vec![0, 1, 2]);

    // Backward paths don't exist
    assert!(g.shortest_path(2, 0).is_none());
}

#[test]
fn test_shortest_path_cycle() {
    // Cycle graph: 0 -> 1 -> 2 -> 3 -> 0
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], true);

    // Test path that uses cycle
    let path = g.shortest_path(0, 3).expect("path should exist");

    // Direct path 0->1->2->3 (length 4) vs backward 0<-3 (not possible in directed)
    assert_eq!(path.len(), 4);
    assert_eq!(path, vec![0, 1, 2, 3]);
}

#[test]
fn test_shortest_path_star_graph() {
    // Star graph: 0 connected to 1, 2, 3, 4
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (0, 4)], false);

    // Center to leaf: length 2
    let path = g.shortest_path(0, 1).expect("path should exist");
    assert_eq!(path.len(), 2);

    // Leaf to leaf through center: length 3
    let path = g.shortest_path(1, 2).expect("path should exist");
    assert_eq!(path.len(), 3);
    assert_eq!(path[0], 1);
    assert_eq!(path[1], 0); // Must go through center
    assert_eq!(path[2], 2);
}

#[test]
fn test_shortest_path_empty_graph() {
    // Empty graph
    let g = Graph::new(false);
    assert!(g.shortest_path(0, 0).is_none());
}

#[test]
fn test_shortest_path_single_node_graph() {
    // Graph with single self-loop
    let g = Graph::from_edges(&[(0, 0)], false);
    let path = g.shortest_path(0, 0).expect("path should exist");
    assert_eq!(path, vec![0]);
}

#[test]
fn test_shortest_path_complete_graph() {
    // Complete graph K4
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);

    // All pairs should have direct edge (length 2)
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                let path = g.shortest_path(i, j).expect("path should exist");
                assert_eq!(path.len(), 2, "Path from {i} to {j} should be direct");
                assert_eq!(path[0], i);
                assert_eq!(path[1], j);
            }
        }
    }
}

#[test]
fn test_shortest_path_bidirectional() {
    // Undirected: path should exist in both directions
    let g = Graph::from_edges(&[(0, 1), (1, 2)], false);

    let path_forward = g.shortest_path(0, 2).expect("forward path should exist");
    let path_backward = g.shortest_path(2, 0).expect("backward path should exist");

    assert_eq!(path_forward.len(), path_backward.len());
    assert_eq!(path_forward.len(), 3);

    // Paths should be reverses of each other
    let reversed: Vec<_> = path_backward.iter().rev().copied().collect();
    assert_eq!(path_forward, reversed);
}

// ========================================================================
// Dijkstra's Algorithm Tests
// ========================================================================

#[test]
fn test_dijkstra_simple_weighted() {
    // Simple weighted graph
    let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)], false);

    let (path, dist) = g.dijkstra(0, 2).expect("path should exist");
    assert_eq!(dist, 3.0); // 0->1->2 is shorter than 0->2
    assert_eq!(path, vec![0, 1, 2]);
}

#[test]
fn test_dijkstra_same_node() {
    let g = Graph::from_weighted_edges(&[(0, 1, 1.0)], false);
    let (path, dist) = g.dijkstra(0, 0).expect("path should exist");
    assert_eq!(path, vec![0]);
    assert_eq!(dist, 0.0);
}

#[test]
fn test_dijkstra_disconnected() {
    let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (2, 3, 1.0)], false);
    assert!(g.dijkstra(0, 3).is_none());
}

#[test]
fn test_dijkstra_unweighted() {
    // Unweighted graph (uses weight 1.0 for all edges)
    let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
    let (path, dist) = g.dijkstra(0, 2).expect("path should exist");
    assert_eq!(dist, 2.0);
    assert_eq!(path, vec![0, 1, 2]);
}

#[test]
fn test_dijkstra_triangle_weighted() {
    // Triangle with different weights
    let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 5.0)], false);

    let (path, dist) = g.dijkstra(0, 2).expect("path should exist");
    assert_eq!(dist, 2.0); // 0->1->2 (cost 2) vs 0->2 (cost 5)
    assert_eq!(path, vec![0, 1, 2]);
}

#[test]
fn test_dijkstra_multiple_paths() {
    // Graph with multiple paths of different costs
    //     1 ----2.0---- 2
    //    /              |
    //   /               |
    //  0                1.0
    //   \               |
    //    \              |
    //     3 ----1.0---- 4
    let g = Graph::from_weighted_edges(
        &[
            (0, 1, 1.0),
            (1, 2, 2.0),
            (0, 3, 1.0),
            (3, 4, 1.0),
            (4, 2, 1.0),
        ],
        false,
    );

    let (_path, dist) = g.dijkstra(0, 2).expect("path should exist");
    assert_eq!(dist, 3.0); // Best path: 0->3->4->2 or 0->1->2
}

#[test]
fn test_dijkstra_linear_chain() {
    // Weighted linear chain
    let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0)], false);

    let (path, dist) = g.dijkstra(0, 3).expect("path should exist");
    assert_eq!(dist, 6.0);
    assert_eq!(path, vec![0, 1, 2, 3]);
}

#[test]
fn test_dijkstra_directed_graph() {
    // Directed weighted graph
    let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 2.0)], true);

    let (path, dist) = g.dijkstra(0, 2).expect("forward path should exist");
    assert_eq!(dist, 3.0);
    assert_eq!(path, vec![0, 1, 2]);

    // Backward path doesn't exist
    assert!(g.dijkstra(2, 0).is_none());
}

#[test]
fn test_dijkstra_invalid_nodes() {
    let g = Graph::from_weighted_edges(&[(0, 1, 1.0)], false);
    assert!(g.dijkstra(0, 10).is_none());
    assert!(g.dijkstra(10, 0).is_none());
}

#[test]
#[should_panic(expected = "negative edge weights")]
fn test_dijkstra_negative_weights() {
    // Dijkstra should panic on negative weights
    let g = Graph::from_weighted_edges(&[(0, 1, -1.0)], false);
    let _ = g.dijkstra(0, 1);
}

#[test]
fn test_dijkstra_zero_weight_edges() {
    // Zero-weight edges should work
    let g = Graph::from_weighted_edges(&[(0, 1, 0.0), (1, 2, 1.0)], false);
    let (path, dist) = g.dijkstra(0, 2).expect("path should exist");
    assert_eq!(dist, 1.0);
    assert_eq!(path, vec![0, 1, 2]);
}

#[test]
fn test_dijkstra_complete_graph_weighted() {
    // Complete graph K3 with different weights
    let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 3.0)], false);

    // Direct edge 0->2 costs 3.0, but 0->1->2 costs 2.0
    let (path, dist) = g.dijkstra(0, 2).expect("path should exist");
    assert_eq!(dist, 2.0);
    assert_eq!(path, vec![0, 1, 2]);
}

#[test]
fn test_dijkstra_star_graph_weighted() {
    // Star graph with center at node 0
    let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (0, 2, 2.0), (0, 3, 3.0)], false);

    // Path from 1 to 3 must go through 0
    let (path, dist) = g.dijkstra(1, 3).expect("path should exist");
    assert_eq!(dist, 4.0); // 1->0 (1.0) + 0->3 (3.0)
    assert_eq!(path, vec![1, 0, 3]);
}
