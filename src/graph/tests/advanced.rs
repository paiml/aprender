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

include!("advanced_part_02.rs");
include!("advanced_part_03.rs");
