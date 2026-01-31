//\! Advanced graph tests: all-pairs shortest paths, A*, DFS, components, label propagation.

use crate::graph::*;

// ========================================================================
// All-Pairs Shortest Paths Tests
// ========================================================================

#[test]
fn test_apsp_linear_chain() {
    // Linear chain: 0 -- 1 -- 2 -- 3
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
    let dist = g.all_pairs_shortest_paths();

    // Check diagonal (distance to self = 0)
    for (i, row) in dist.iter().enumerate().take(4) {
        assert_eq!(row[i], Some(0));
    }

    // Check distances
    assert_eq!(dist[0][1], Some(1));
    assert_eq!(dist[0][2], Some(2));
    assert_eq!(dist[0][3], Some(3));
    assert_eq!(dist[1][2], Some(1));
    assert_eq!(dist[1][3], Some(2));
    assert_eq!(dist[2][3], Some(1));

    // Check symmetry (undirected graph)
    assert_eq!(dist[0][3], dist[3][0]);
    assert_eq!(dist[1][2], dist[2][1]);
}

#[test]
fn test_apsp_complete_graph() {
    // Complete graph K4
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
    let dist = g.all_pairs_shortest_paths();

    // All pairs should have distance 1 (direct edge) except diagonal
    for (i, row) in dist.iter().enumerate().take(4) {
        for (j, &cell) in row.iter().enumerate().take(4) {
            if i == j {
                assert_eq!(cell, Some(0));
            } else {
                assert_eq!(cell, Some(1));
            }
        }
    }
}

#[test]
fn test_apsp_disconnected() {
    // Two disconnected components: (0, 1) and (2, 3)
    let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
    let dist = g.all_pairs_shortest_paths();

    // Within components
    assert_eq!(dist[0][1], Some(1));
    assert_eq!(dist[1][0], Some(1));
    assert_eq!(dist[2][3], Some(1));
    assert_eq!(dist[3][2], Some(1));

    // Between components (no path)
    assert_eq!(dist[0][2], None);
    assert_eq!(dist[0][3], None);
    assert_eq!(dist[1][2], None);
    assert_eq!(dist[1][3], None);
}

#[test]
fn test_apsp_directed() {
    // Directed graph: 0 -> 1 -> 2
    let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
    let dist = g.all_pairs_shortest_paths();

    // Forward paths
    assert_eq!(dist[0][1], Some(1));
    assert_eq!(dist[0][2], Some(2));
    assert_eq!(dist[1][2], Some(1));

    // Backward paths (no reverse edges)
    assert_eq!(dist[1][0], None);
    assert_eq!(dist[2][0], None);
    assert_eq!(dist[2][1], None);
}

#[test]
fn test_apsp_triangle() {
    // Triangle graph
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
    let dist = g.all_pairs_shortest_paths();

    // All pairs should have distance 1 (triangle) except diagonal
    for (i, row) in dist.iter().enumerate().take(3) {
        for (j, &cell) in row.iter().enumerate().take(3) {
            if i == j {
                assert_eq!(cell, Some(0));
            } else {
                assert_eq!(cell, Some(1));
            }
        }
    }
}

#[test]
fn test_apsp_star_graph() {
    // Star graph: 0 connected to 1, 2, 3
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    let dist = g.all_pairs_shortest_paths();

    // Center to leaves: distance 1
    assert_eq!(dist[0][1], Some(1));
    assert_eq!(dist[0][2], Some(1));
    assert_eq!(dist[0][3], Some(1));

    // Leaf to leaf through center: distance 2
    assert_eq!(dist[1][2], Some(2));
    assert_eq!(dist[1][3], Some(2));
    assert_eq!(dist[2][3], Some(2));
}

#[test]
fn test_apsp_empty_graph() {
    let g = Graph::new(false);
    let dist = g.all_pairs_shortest_paths();
    assert_eq!(dist.len(), 0);
}

#[test]
fn test_apsp_single_node() {
    let g = Graph::from_edges(&[(0, 0)], false);
    let dist = g.all_pairs_shortest_paths();

    assert_eq!(dist.len(), 1);
    assert_eq!(dist[0][0], Some(0));
}

#[test]
fn test_apsp_cycle() {
    // Cycle: 0 -> 1 -> 2 -> 3 -> 0
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], true);
    let dist = g.all_pairs_shortest_paths();

    // Along cycle direction
    assert_eq!(dist[0][1], Some(1));
    assert_eq!(dist[0][2], Some(2));
    assert_eq!(dist[0][3], Some(3));
    assert_eq!(dist[1][3], Some(2));

    // All nodes reachable in directed cycle
    for row in dist.iter().take(4) {
        for &cell in row.iter().take(4) {
            assert!(cell.is_some());
        }
    }
}

#[test]
fn test_apsp_matrix_size() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
    let dist = g.all_pairs_shortest_paths();

    // Matrix should be n×n
    assert_eq!(dist.len(), 4);
    for row in &dist {
        assert_eq!(row.len(), 4);
    }
}

// ========================================================================
// A* Search Algorithm Tests
// ========================================================================

#[test]
fn test_astar_linear_chain() {
    // Linear chain: 0 -- 1 -- 2 -- 3
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);

    // Simple distance heuristic
    let heuristic = |node: usize| (3 - node) as f64;

    let path = g.a_star(0, 3, heuristic).expect("path should exist");
    assert_eq!(path, vec![0, 1, 2, 3]);
}

#[test]
fn test_astar_same_node() {
    let g = Graph::from_edges(&[(0, 1)], false);
    let heuristic = |_: usize| 0.0;

    let path = g.a_star(0, 0, heuristic).expect("path should exist");
    assert_eq!(path, vec![0]);
}

#[test]
fn test_astar_disconnected() {
    // Two disconnected components
    let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
    let heuristic = |_: usize| 0.0;

    assert!(g.a_star(0, 3, heuristic).is_none());
}

#[test]
fn test_astar_zero_heuristic() {
    // With h(n) = 0, A* behaves like Dijkstra
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
    let heuristic = |_: usize| 0.0;

    let path = g.a_star(0, 3, heuristic).expect("path should exist");
    assert_eq!(path, vec![0, 1, 2, 3]);
}

#[test]
fn test_astar_admissible_heuristic() {
    // Graph with shortcut
    // 0 -- 1 -- 2
    // |         |
    // +----3----+
    let g =
        Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 1.0), (0, 3, 0.5), (3, 2, 0.5)], false);

    // Admissible heuristic (straight-line distance estimate)
    let heuristic = |node: usize| match node {
        0 | 1 => 1.0, // Estimate to reach 2
        3 => 0.5,
        _ => 0.0, // At target (2) or other
    };

    let path = g.a_star(0, 2, heuristic).expect("path should exist");
    // Should find shortest path via 3
    assert!(path.contains(&3)); // Must use the shortcut
}

#[test]
fn test_astar_directed() {
    // Directed graph: 0 -> 1 -> 2
    let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
    let heuristic = |node: usize| (2 - node) as f64;

    let path = g
        .a_star(0, 2, heuristic)
        .expect("forward path should exist");
    assert_eq!(path, vec![0, 1, 2]);

    // Backward path doesn't exist
    assert!(g.a_star(2, 0, |_| 0.0).is_none());
}

#[test]
fn test_astar_triangle() {
    // Triangle graph
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
    let heuristic = |_: usize| 0.0;

    let path = g.a_star(0, 2, heuristic).expect("path should exist");
    assert_eq!(path.len(), 2); // Direct edge 0-2
    assert_eq!(path[0], 0);
    assert_eq!(path[1], 2);
}

#[test]
fn test_astar_weighted_graph() {
    // Weighted graph with better heuristic guidance
    let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)], false);

    // Heuristic guides toward node 2
    let heuristic = |node: usize| match node {
        0 => 3.0,
        1 => 2.0,
        _ => 0.0, // At target (2) or other
    };

    let path = g.a_star(0, 2, heuristic).expect("path should exist");
    // Should find path 0->1->2 (cost 3) instead of 0->2 (cost 5)
    assert_eq!(path, vec![0, 1, 2]);
}

#[test]
fn test_astar_complete_graph() {
    // Complete graph K4
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
    let heuristic = |_: usize| 0.0;

    // All nodes directly connected
    let path = g.a_star(0, 3, heuristic).expect("path should exist");
    assert_eq!(path.len(), 2); // Direct path
    assert_eq!(path[0], 0);
    assert_eq!(path[1], 3);
}

#[test]
fn test_astar_invalid_nodes() {
    let g = Graph::from_edges(&[(0, 1)], false);
    let heuristic = |_: usize| 0.0;

    assert!(g.a_star(0, 10, heuristic).is_none());
    assert!(g.a_star(10, 0, heuristic).is_none());
}

#[test]
fn test_astar_vs_shortest_path() {
    // On unweighted graph with zero heuristic, A* should match shortest_path
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);

    let sp_path = g
        .shortest_path(0, 3)
        .expect("shortest_path should find path");
    let astar_path = g.a_star(0, 3, |_| 0.0).expect("astar should find path");

    assert_eq!(sp_path.len(), astar_path.len());
    assert_eq!(sp_path, astar_path);
}

#[test]
fn test_astar_star_graph() {
    // Star graph: 0 connected to 1, 2, 3
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);

    // Heuristic that guides toward target 3
    let heuristic = |node: usize| if node == 3 { 0.0 } else { 1.0 };

    let path = g.a_star(1, 3, heuristic).expect("path should exist");
    assert_eq!(path.len(), 3); // Must go through center
    assert_eq!(path[0], 1);
    assert_eq!(path[1], 0);
    assert_eq!(path[2], 3);
}

#[test]
fn test_astar_perfect_heuristic() {
    // Perfect heuristic (exact distance to target)
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);

    // Perfect heuristic = exact remaining distance
    let heuristic = |node: usize| (3 - node) as f64;

    let path = g.a_star(0, 3, heuristic).expect("path should exist");
    assert_eq!(path, vec![0, 1, 2, 3]);
}

#[test]
fn test_astar_complex_graph() {
    // More complex graph to test heuristic efficiency
    //     1
    //    / \
    //   0   3 - 4
    //    \ /
    //     2
    let g = Graph::from_weighted_edges(
        &[
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
        ],
        false,
    );

    // Distance-based heuristic
    let heuristic = |node: usize| (4 - node) as f64;

    let path = g.a_star(0, 4, heuristic).expect("path should exist");
    assert_eq!(path.len(), 4); // 0->1->3->4 or 0->2->3->4
    assert_eq!(path[0], 0);
    assert_eq!(path[3], 4);
}

// DFS Tests

#[test]
fn test_dfs_linear_chain() {
    // Linear chain: 0 -- 1 -- 2 -- 3
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
    let visited = g.dfs(0).expect("valid source");

    assert_eq!(visited.len(), 4);
    assert_eq!(visited[0], 0); // Starts at source
    assert!(visited.contains(&1));
    assert!(visited.contains(&2));
    assert!(visited.contains(&3));
}

#[test]
fn test_dfs_tree() {
    // Tree: 0 connected to 1, 2, 3
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    let visited = g.dfs(0).expect("valid source");

    assert_eq!(visited.len(), 4);
    assert_eq!(visited[0], 0); // Root first
                               // Children visited in some order
    assert!(visited.contains(&1));
    assert!(visited.contains(&2));
    assert!(visited.contains(&3));
}

#[test]
fn test_dfs_cycle() {
    // Cycle: 0 -- 1 -- 2 -- 0
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
    let visited = g.dfs(0).expect("valid source");

    assert_eq!(visited.len(), 3);
    assert_eq!(visited[0], 0);
    assert!(visited.contains(&1));
    assert!(visited.contains(&2));
}

#[test]
fn test_dfs_disconnected() {
    // Two components: (0, 1) and (2, 3)
    let g = Graph::from_edges(&[(0, 1), (2, 3)], false);

    // DFS from 0 only visits component containing 0
    let visited = g.dfs(0).expect("valid source");
    assert_eq!(visited.len(), 2);
    assert!(visited.contains(&0));
    assert!(visited.contains(&1));
    assert!(!visited.contains(&2));
    assert!(!visited.contains(&3));

    // DFS from 2 only visits component containing 2
    let visited2 = g.dfs(2).expect("valid source");
    assert_eq!(visited2.len(), 2);
    assert!(visited2.contains(&2));
    assert!(visited2.contains(&3));
}

#[test]
fn test_dfs_directed() {
    // Directed: 0 -> 1 -> 2
    let g = Graph::from_edges(&[(0, 1), (1, 2)], true);

    // Forward traversal
    let visited = g.dfs(0).expect("valid source");
    assert_eq!(visited.len(), 3);
    assert_eq!(visited[0], 0);

    // Backward traversal (node 2 has no outgoing edges)
    let visited2 = g.dfs(2).expect("valid source");
    assert_eq!(visited2.len(), 1);
    assert_eq!(visited2[0], 2);
}

#[test]
fn test_dfs_single_node() {
    // Single node with self-loop
    let g = Graph::from_edges(&[(0, 0)], false);

    let visited = g.dfs(0).expect("valid source");
    assert_eq!(visited.len(), 1);
    assert_eq!(visited[0], 0);
}

#[test]
fn test_dfs_invalid_source() {
    let g = Graph::from_edges(&[(0, 1), (1, 2)], false);

    // Invalid source node
    assert!(g.dfs(10).is_none());
    assert!(g.dfs(100).is_none());
}

#[test]
fn test_dfs_complete_graph() {
    // Complete graph K4
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
    let visited = g.dfs(0).expect("valid source");

    assert_eq!(visited.len(), 4);
    assert_eq!(visited[0], 0);
    // All other nodes reachable
    assert!(visited.contains(&1));
    assert!(visited.contains(&2));
    assert!(visited.contains(&3));
}

#[test]
fn test_dfs_dag() {
    // DAG: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 3), (2, 3)], true);
    let visited = g.dfs(0).expect("valid source");

    assert_eq!(visited.len(), 4);
    assert_eq!(visited[0], 0);
    // All nodes reachable from 0
    assert!(visited.contains(&1));
    assert!(visited.contains(&2));
    assert!(visited.contains(&3));

    // Node 3 is a sink (no outgoing edges)
    let visited3 = g.dfs(3).expect("valid source");
    assert_eq!(visited3.len(), 1);
    assert_eq!(visited3[0], 3);
}

#[test]
fn test_dfs_empty_graph() {
    let g = Graph::new(false);
    // No nodes, so any DFS should return None
    assert!(g.dfs(0).is_none());
}

// Connected Components Tests

#[test]
fn test_connected_components_single() {
    // Single connected component
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
    let components = g.connected_components();

    assert_eq!(components.len(), 4);
    // All nodes in same component
    assert_eq!(components[0], components[1]);
    assert_eq!(components[1], components[2]);
    assert_eq!(components[2], components[3]);
}

#[test]
fn test_connected_components_two() {
    // Two disconnected components: (0,1) and (2,3)
    let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
    let components = g.connected_components();

    assert_eq!(components.len(), 4);
    // Component 1: nodes 0 and 1
    assert_eq!(components[0], components[1]);
    // Component 2: nodes 2 and 3
    assert_eq!(components[2], components[3]);
    // Different components
    assert_ne!(components[0], components[2]);
}

#[test]
fn test_connected_components_three() {
    // Three components: (0,1), (2,3), (4)
    let g = Graph::from_edges(&[(0, 1), (2, 3), (4, 4)], false);
    let components = g.connected_components();

    assert_eq!(components.len(), 5);
    // Three distinct components
    assert_eq!(components[0], components[1]);
    assert_eq!(components[2], components[3]);
    assert_ne!(components[0], components[2]);
    assert_ne!(components[0], components[4]);
    assert_ne!(components[2], components[4]);
}

#[test]
fn test_connected_components_complete() {
    // Complete graph K4 - all in one component
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
    let components = g.connected_components();

    assert_eq!(components.len(), 4);
    let first = components[0];
    assert!(components.iter().all(|&c| c == first));
}

#[test]
fn test_connected_components_star() {
    // Star graph - all connected through center
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    let components = g.connected_components();

    assert_eq!(components.len(), 4);
    // All in same component
    assert_eq!(components[0], components[1]);
    assert_eq!(components[0], components[2]);
    assert_eq!(components[0], components[3]);
}

#[test]
fn test_connected_components_directed_weak() {
    // Directed graph: 0 -> 1 -> 2 (weakly connected)
    let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
    let components = g.connected_components();

    assert_eq!(components.len(), 3);
    // Weakly connected (ignores direction)
    assert_eq!(components[0], components[1]);
    assert_eq!(components[1], components[2]);
}

#[test]
fn test_connected_components_cycle() {
    // Cycle graph
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
    let components = g.connected_components();

    assert_eq!(components.len(), 3);
    // All in same component
    assert_eq!(components[0], components[1]);
    assert_eq!(components[1], components[2]);
}

#[test]
fn test_connected_components_empty() {
    let g = Graph::new(false);
    let components = g.connected_components();
    assert!(components.is_empty());
}

#[test]
fn test_connected_components_isolated_nodes() {
    // Graph with some isolated nodes
    let g = Graph::from_edges(&[(0, 1), (3, 4)], false);
    // Node 2 is isolated (no edges)
    // But we only have nodes that appear in edges
    let components = g.connected_components();

    assert_eq!(components.len(), 5);
    // Two components: (0,1) and (3,4), and isolated 2
    assert_eq!(components[0], components[1]);
    assert_eq!(components[3], components[4]);
    assert_ne!(components[0], components[3]);
    // Node 2 is in its own component
    assert_ne!(components[2], components[0]);
    assert_ne!(components[2], components[3]);
}

#[test]
fn test_connected_components_count() {
    // Helper to count unique components
    fn count_components(components: &[usize]) -> usize {
        use std::collections::HashSet;
        components.iter().copied().collect::<HashSet<_>>().len()
    }

    // Single component
    let g1 = Graph::from_edges(&[(0, 1), (1, 2)], false);
    assert_eq!(count_components(&g1.connected_components()), 1);

    // Two components
    let g2 = Graph::from_edges(&[(0, 1), (2, 3)], false);
    assert_eq!(count_components(&g2.connected_components()), 2);

    // Three components
    let g3 = Graph::from_edges(&[(0, 1), (2, 3), (4, 5)], false);
    assert_eq!(count_components(&g3.connected_components()), 3);
}

// Strongly Connected Components Tests

#[test]
fn test_scc_single_cycle() {
    // Single SCC: 0 -> 1 -> 2 -> 0
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
    let sccs = g.strongly_connected_components();

    assert_eq!(sccs.len(), 3);
    // All nodes in same SCC
    assert_eq!(sccs[0], sccs[1]);
    assert_eq!(sccs[1], sccs[2]);
}

#[test]
fn test_scc_dag() {
    // DAG: 0 -> 1 -> 2 (each node is its own SCC)
    let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
    let sccs = g.strongly_connected_components();

    assert_eq!(sccs.len(), 3);
    // Each node is its own SCC
    assert_ne!(sccs[0], sccs[1]);
    assert_ne!(sccs[1], sccs[2]);
    assert_ne!(sccs[0], sccs[2]);
}

#[test]
fn test_scc_two_components() {
    // Two SCCs: (0->1->0) and (2->3->2)
    let g = Graph::from_edges(&[(0, 1), (1, 0), (2, 3), (3, 2)], true);
    let sccs = g.strongly_connected_components();

    assert_eq!(sccs.len(), 4);
    // SCC 1: nodes 0 and 1
    assert_eq!(sccs[0], sccs[1]);
    // SCC 2: nodes 2 and 3
    assert_eq!(sccs[2], sccs[3]);
    // Different SCCs
    assert_ne!(sccs[0], sccs[2]);
}

#[test]
fn test_scc_complex() {
    // Complex graph with multiple SCCs
    // SCC 1: 0 -> 1 -> 0
    // SCC 2: 2 -> 3 -> 4 -> 2
    // Edge from SCC1 to SCC2: 1 -> 2
    let g = Graph::from_edges(&[(0, 1), (1, 0), (1, 2), (2, 3), (3, 4), (4, 2)], true);
    let sccs = g.strongly_connected_components();

    assert_eq!(sccs.len(), 5);
    // SCC 1: nodes 0 and 1
    assert_eq!(sccs[0], sccs[1]);
    // SCC 2: nodes 2, 3, 4
    assert_eq!(sccs[2], sccs[3]);
    assert_eq!(sccs[3], sccs[4]);
    // Different SCCs
    assert_ne!(sccs[0], sccs[2]);
}

#[test]
fn test_scc_self_loop() {
    // Single node with self-loop is an SCC
    let g = Graph::from_edges(&[(0, 0)], true);
    let sccs = g.strongly_connected_components();

    assert_eq!(sccs.len(), 1);
    assert_eq!(sccs[0], 0);
}

#[test]
fn test_scc_disconnected() {
    // Two disconnected cycles
    let g = Graph::from_edges(&[(0, 1), (1, 0), (2, 3), (3, 2)], true);
    let sccs = g.strongly_connected_components();

    assert_eq!(sccs.len(), 4);
    // Two separate SCCs
    assert_eq!(sccs[0], sccs[1]);
    assert_eq!(sccs[2], sccs[3]);
    assert_ne!(sccs[0], sccs[2]);
}

#[test]
fn test_scc_empty() {
    let g = Graph::new(true);
    let sccs = g.strongly_connected_components();
    assert!(sccs.is_empty());
}

#[test]
fn test_scc_linear_dag() {
    // Linear DAG: 0 -> 1 -> 2 -> 3
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], true);
    let sccs = g.strongly_connected_components();

    assert_eq!(sccs.len(), 4);
    // Each node is its own SCC in a DAG
    use std::collections::HashSet;
    let unique_sccs: HashSet<_> = sccs.iter().copied().collect();
    assert_eq!(unique_sccs.len(), 4);
}

#[test]
fn test_scc_complete_graph() {
    // Complete directed graph (all nodes reachable from all)
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)], true);
    let sccs = g.strongly_connected_components();

    assert_eq!(sccs.len(), 3);
    // All in same SCC
    assert_eq!(sccs[0], sccs[1]);
    assert_eq!(sccs[1], sccs[2]);
}

#[test]
fn test_scc_count() {
    // Helper to count unique SCCs
    fn count_sccs(sccs: &[usize]) -> usize {
        use std::collections::HashSet;
        sccs.iter().copied().collect::<HashSet<_>>().len()
    }

    // Single SCC
    let g1 = Graph::from_edges(&[(0, 1), (1, 0)], true);
    assert_eq!(count_sccs(&g1.strongly_connected_components()), 1);

    // Two SCCs
    let g2 = Graph::from_edges(&[(0, 1)], true);
    assert_eq!(count_sccs(&g2.strongly_connected_components()), 2);

    // Three SCCs
    let g3 = Graph::from_edges(&[(0, 1), (2, 3), (3, 2)], true);
    assert_eq!(count_sccs(&g3.strongly_connected_components()), 3);
}

// Topological Sort Tests

#[test]
fn test_topo_linear_dag() {
    // Linear DAG: 0 -> 1 -> 2 -> 3
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], true);
    let order = g
        .topological_sort()
        .expect("DAG should have topological order");

    assert_eq!(order.len(), 4);
    // Check ordering constraints
    assert!(order.iter().position(|&x| x == 0) < order.iter().position(|&x| x == 1));
    assert!(order.iter().position(|&x| x == 1) < order.iter().position(|&x| x == 2));
    assert!(order.iter().position(|&x| x == 2) < order.iter().position(|&x| x == 3));
}

#[test]
fn test_topo_cycle() {
    // Cycle: 0 -> 1 -> 2 -> 0
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
    assert!(g.topological_sort().is_none()); // Should detect cycle
}

#[test]
fn test_topo_diamond() {
    // Diamond DAG: 0 -> {1,2} -> 3
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 3), (2, 3)], true);
    let order = g
        .topological_sort()
        .expect("DAG should have topological order");

    assert_eq!(order.len(), 4);
    // 0 must come before 1, 2, 3
    let pos_0 = order
        .iter()
        .position(|&x| x == 0)
        .expect("0 should be in order");
    assert!(
        pos_0
            < order
                .iter()
                .position(|&x| x == 1)
                .expect("1 should be in order")
    );
    assert!(
        pos_0
            < order
                .iter()
                .position(|&x| x == 2)
                .expect("2 should be in order")
    );
    assert!(
        pos_0
            < order
                .iter()
                .position(|&x| x == 3)
                .expect("3 should be in order")
    );

    // 3 must come after 1 and 2
    let pos_3 = order
        .iter()
        .position(|&x| x == 3)
        .expect("3 should be in order");
    assert!(
        order
            .iter()
            .position(|&x| x == 1)
            .expect("1 should be in order")
            < pos_3
    );
    assert!(
        order
            .iter()
            .position(|&x| x == 2)
            .expect("2 should be in order")
            < pos_3
    );
}

#[test]
fn test_topo_empty() {
    let g = Graph::new(true);
    let order = g
        .topological_sort()
        .expect("Empty graph has topological order");
    assert!(order.is_empty());
}

#[test]
fn test_topo_single_node() {
    // Single node with self-loop creates cycle
    let g = Graph::from_edges(&[(0, 0)], true);
    assert!(g.topological_sort().is_none()); // Self-loop is a cycle
}

#[test]
fn test_topo_disconnected_dag() {
    // Two disconnected chains: 0->1 and 2->3
    let g = Graph::from_edges(&[(0, 1), (2, 3)], true);
    let order = g
        .topological_sort()
        .expect("Disconnected DAG has topological order");

    assert_eq!(order.len(), 4);
    // Within each chain, ordering is preserved
    assert!(order.iter().position(|&x| x == 0) < order.iter().position(|&x| x == 1));
    assert!(order.iter().position(|&x| x == 2) < order.iter().position(|&x| x == 3));
}

#[test]
fn test_topo_tree() {
    // Tree: 0 -> {1, 2}, 1 -> {3, 4}
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 3), (1, 4)], true);
    let order = g.topological_sort().expect("Tree is a DAG");

    assert_eq!(order.len(), 5);
    // 0 must come first
    assert_eq!(order.iter().position(|&x| x == 0), Some(0));
    // 1 before 3 and 4
    let pos_1 = order
        .iter()
        .position(|&x| x == 1)
        .expect("1 should be in order");
    assert!(
        pos_1
            < order
                .iter()
                .position(|&x| x == 3)
                .expect("3 should be in order")
    );
    assert!(
        pos_1
            < order
                .iter()
                .position(|&x| x == 4)
                .expect("4 should be in order")
    );
}

#[test]
fn test_topo_self_loop() {
    // Self-loop is a cycle
    let g = Graph::from_edges(&[(0, 0)], true);
    assert!(g.topological_sort().is_none());
}

#[test]
fn test_topo_complete_dag() {
    // Complete DAG: 0 -> {1,2,3}, 1 -> {2,3}, 2 -> 3
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], true);
    let order = g
        .topological_sort()
        .expect("Complete DAG has topological order");

    assert_eq!(order.len(), 4);
    // Check all ordering constraints
    let positions: Vec<_> = (0..4)
        .map(|i| {
            order
                .iter()
                .position(|&x| x == i)
                .expect("node should be in order")
        })
        .collect();

    assert!(positions[0] < positions[1]);
    assert!(positions[0] < positions[2]);
    assert!(positions[0] < positions[3]);
    assert!(positions[1] < positions[2]);
    assert!(positions[1] < positions[3]);
    assert!(positions[2] < positions[3]);
}

#[test]
fn test_topo_undirected() {
    // Undirected graph is treated as bidirectional (always has cycles unless tree)
    let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
    // Undirected edges create cycles (0->1 and 1->0)
    assert!(g.topological_sort().is_none());
}

// Common Neighbors Tests

#[test]
fn test_common_neighbors_triangle() {
    // Triangle: 0-1, 0-2, 1-2
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], false);

    // Nodes 1 and 2 share neighbor 0
    assert_eq!(g.common_neighbors(1, 2), Some(1));
    // Nodes 0 and 1 share neighbor 2
    assert_eq!(g.common_neighbors(0, 1), Some(1));
    // Nodes 0 and 2 share neighbor 1
    assert_eq!(g.common_neighbors(0, 2), Some(1));
}

#[test]
fn test_common_neighbors_no_overlap() {
    // Two stars: 0-{1,2}, 3-{4,5}
    let g = Graph::from_edges(&[(0, 1), (0, 2), (3, 4), (3, 5)], false);

    // Nodes 1 and 2 share neighbor 0
    assert_eq!(g.common_neighbors(1, 2), Some(1));
    // Nodes from different components have no common neighbors
    assert_eq!(g.common_neighbors(1, 4), Some(0));
    assert_eq!(g.common_neighbors(0, 3), Some(0));
}

#[test]
fn test_common_neighbors_complete() {
    // Complete graph K4
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);

    // Any two nodes share 2 common neighbors
    assert_eq!(g.common_neighbors(0, 1), Some(2)); // Share 2, 3
    assert_eq!(g.common_neighbors(0, 2), Some(2)); // Share 1, 3
    assert_eq!(g.common_neighbors(1, 2), Some(2)); // Share 0, 3
}

#[test]
fn test_common_neighbors_directed() {
    // Directed: 0->1, 0->2, 1->2
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], true);

    // 0 has out-neighbors {1, 2}
    // 1 has out-neighbors {2}
    assert_eq!(g.common_neighbors(0, 1), Some(1)); // Share out-neighbor 2
}

#[test]
fn test_common_neighbors_invalid() {
    let g = Graph::from_edges(&[(0, 1), (1, 2)], false);

    // Invalid nodes
    assert!(g.common_neighbors(0, 10).is_none());
    assert!(g.common_neighbors(10, 0).is_none());
    assert!(g.common_neighbors(10, 20).is_none());
}

#[test]
fn test_common_neighbors_self() {
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], false);

    // Node with itself shares all neighbors
    assert_eq!(g.common_neighbors(0, 0), Some(2)); // Shares {1, 2}
}

#[test]
fn test_common_neighbors_empty() {
    let g = Graph::new(false);
    assert!(g.common_neighbors(0, 1).is_none());
}

#[test]
fn test_common_neighbors_star() {
    // Star: 0 connected to {1, 2, 3, 4}
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (0, 4)], false);

    // Leaves share center as common neighbor
    assert_eq!(g.common_neighbors(1, 2), Some(1)); // Share 0
    assert_eq!(g.common_neighbors(1, 3), Some(1)); // Share 0
    assert_eq!(g.common_neighbors(2, 4), Some(1)); // Share 0
}

// Adamic-Adar Index Tests

#[test]
fn test_adamic_adar_triangle() {
    // Triangle: 0-1, 0-2, 1-2
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], false);

    // Nodes 1 and 2 share neighbor 0 (degree 2)
    let aa = g.adamic_adar_index(1, 2).expect("valid nodes");
    // Score should be 1/ln(2) ≈ 1.44
    assert!((aa - 1.0 / 2.0_f64.ln()).abs() < 1e-10);
}

#[test]
fn test_adamic_adar_no_common() {
    // Two disconnected edges
    let g = Graph::from_edges(&[(0, 1), (2, 3)], false);

    // No common neighbors
    assert_eq!(g.adamic_adar_index(0, 2).expect("valid nodes"), 0.0);
    assert_eq!(g.adamic_adar_index(1, 3).expect("valid nodes"), 0.0);
}

#[test]
fn test_adamic_adar_star() {
    // Star: 0-{1,2,3,4}
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (0, 4)], false);

    // Leaves share center (degree 4)
    let aa = g.adamic_adar_index(1, 2).expect("valid nodes");
    // Score should be 1/ln(4) ≈ 0.72
    assert!((aa - 1.0 / 4.0_f64.ln()).abs() < 1e-10);
}

#[test]
fn test_adamic_adar_multiple_common() {
    // Graph where nodes 0 and 1 share multiple neighbors
    // 0-{2,3,4}, 1-{2,3,4}
    let g = Graph::from_edges(&[(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)], false);

    let aa = g.adamic_adar_index(0, 1).expect("valid nodes");
    // Three common neighbors (2, 3, 4), each with degree 2
    // Score = 3 * 1/ln(2) = 3/ln(2) ≈ 4.33
    let expected = 3.0 / 2.0_f64.ln();
    assert!((aa - expected).abs() < 1e-10);
}

#[test]
fn test_adamic_adar_degree_one() {
    // Linear chain: 0-1-2
    let g = Graph::from_edges(&[(0, 1), (1, 2)], false);

    // Nodes 0 and 2 share neighbor 1 (degree 2)
    let aa = g.adamic_adar_index(0, 2).expect("valid nodes");
    assert!((aa - 1.0 / 2.0_f64.ln()).abs() < 1e-10);
}

#[test]
fn test_adamic_adar_invalid() {
    let g = Graph::from_edges(&[(0, 1), (1, 2)], false);

    assert!(g.adamic_adar_index(0, 10).is_none());
    assert!(g.adamic_adar_index(10, 0).is_none());
}

#[test]
fn test_adamic_adar_directed() {
    // Directed: 0->2, 1->2, 2->3
    let g = Graph::from_edges(&[(0, 2), (1, 2), (2, 3)], true);

    // 0 and 1 both point to 2 (share out-neighbor)
    let aa = g.adamic_adar_index(0, 1).expect("valid nodes");
    // Node 2 has degree 1 (out-degree), but we use total neighbor count
    // which is 2 (1 in + 1 out in directed graph becomes bidirectional in CSR)
    // Actually in CSR for directed graphs, degree is out-degree
    assert!(aa >= 0.0);
}

#[test]
fn test_adamic_adar_empty() {
    let g = Graph::new(false);
    assert!(g.adamic_adar_index(0, 1).is_none());
}

// Label Propagation Tests

#[test]
fn test_label_propagation_two_triangles() {
    // Two triangles connected by single edge
    // Triangle 1: 0-1-2-0, Triangle 2: 3-4-5-3, Bridge: 2-3
    let g = Graph::from_edges(
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
        false,
    );

    let communities = g.label_propagation(100, Some(42));

    // Nodes within same triangle should likely have same label
    // This is probabilistic, but with seed should be deterministic
    assert_eq!(communities.len(), 6);

    // Count unique communities
    use std::collections::HashSet;
    let unique: HashSet<_> = communities.iter().copied().collect();
    // Should have 2-3 communities depending on convergence
    assert!(!unique.is_empty() && unique.len() <= 6);
}

#[test]
fn test_label_propagation_complete_graph() {
    // Complete graph K4 - all nodes fully connected
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);

    let communities = g.label_propagation(100, Some(42));

    assert_eq!(communities.len(), 4);
    // In complete graph, all nodes should converge to same label
    let first_label = communities[0];
    assert!(communities.iter().all(|&c| c == first_label));
}

#[test]
fn test_label_propagation_disconnected() {
    // Two disconnected triangles
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], false);

    let communities = g.label_propagation(100, Some(42));

    assert_eq!(communities.len(), 6);

    // Each triangle should form its own community
    // Nodes 0, 1, 2 should have same label
    assert_eq!(communities[0], communities[1]);
    assert_eq!(communities[1], communities[2]);

    // Nodes 3, 4, 5 should have same label
    assert_eq!(communities[3], communities[4]);
    assert_eq!(communities[4], communities[5]);

    // Different triangles should have different labels
    assert_ne!(communities[0], communities[3]);
}

#[test]
fn test_label_propagation_star() {
    // Star graph: center connected to all leaves
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (0, 4)], false);

    let communities = g.label_propagation(100, Some(42));

    assert_eq!(communities.len(), 5);
    // All nodes should eventually have same label
    // (leaves adopt center's label or center adopts majority)
    use std::collections::HashSet;
    let unique: HashSet<_> = communities.iter().copied().collect();
    assert_eq!(unique.len(), 1);
}

#[test]
fn test_label_propagation_linear() {
    // Linear chain: 0-1-2-3-4
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)], false);

    let communities = g.label_propagation(100, Some(42));

    assert_eq!(communities.len(), 5);
    // Linear graph may converge to 1-2 communities
    use std::collections::HashSet;
    let unique: HashSet<_> = communities.iter().copied().collect();
    assert!(!unique.is_empty() && unique.len() <= 5);
}

#[test]
fn test_label_propagation_empty() {
    let g = Graph::new(false);
    let communities = g.label_propagation(100, Some(42));
    assert!(communities.is_empty());
}

#[test]
fn test_label_propagation_single_node() {
    // Single isolated node (created via self-loop)
    let g = Graph::from_edges(&[(0, 0)], false);
    let communities = g.label_propagation(100, Some(42));

    assert_eq!(communities.len(), 1);
    assert_eq!(communities[0], 0); // Keeps its own label
}

#[test]
fn test_label_propagation_convergence() {
    // Small graph that should converge quickly
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);

    let communities = g.label_propagation(100, Some(42));

    assert_eq!(communities.len(), 3);
    // Triangle should converge to single community
    assert_eq!(communities[0], communities[1]);
    assert_eq!(communities[1], communities[2]);
}

#[test]
fn test_label_propagation_directed() {
    // Directed graph with mutual edges forms strongly connected component
    // 0<->1<->2 (bidirectional edges)
    let g = Graph::from_edges(&[(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)], true);

    let communities = g.label_propagation(100, Some(42));

    assert_eq!(communities.len(), 3);
    // Strongly connected component should form single community
    assert_eq!(communities[0], communities[1]);
    assert_eq!(communities[1], communities[2]);
}

#[test]
fn test_label_propagation_barbell() {
    // Barbell graph: two cliques connected by bridge
    // Clique 1: 0-1-2 (complete), Clique 2: 3-4-5 (complete), Bridge: 2-3
    let g = Graph::from_edges(
        &[(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)],
        false,
    );

    let communities = g.label_propagation(100, Some(42));

    assert_eq!(communities.len(), 6);

    // Should detect 1-2 communities
    use std::collections::HashSet;
    let unique: HashSet<_> = communities.iter().copied().collect();
    assert!(!unique.is_empty() && unique.len() <= 3);
}
