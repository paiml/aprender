//\! Tests for graph algorithms.

use crate::graph::*;

#[test]
fn test_empty_graph() {
    let g = Graph::new(false);
    assert_eq!(g.num_nodes(), 0);
    assert_eq!(g.num_edges(), 0);
    assert!(!g.is_directed());
}

#[test]
fn test_directed_graph() {
    let g = Graph::new(true);
    assert!(g.is_directed());
}

#[test]
fn test_from_edges_empty() {
    let g = Graph::from_edges(&[], false);
    assert_eq!(g.num_nodes(), 0);
    assert_eq!(g.num_edges(), 0);
}

#[test]
fn test_from_edges_undirected() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
    assert_eq!(g.num_nodes(), 3);
    assert_eq!(g.num_edges(), 3);

    // Check neighbors (should be sorted)
    assert_eq!(g.neighbors(0), &[1, 2]);
    assert_eq!(g.neighbors(1), &[0, 2]);
    assert_eq!(g.neighbors(2), &[0, 1]);
}

#[test]
fn test_from_edges_directed() {
    let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
    assert_eq!(g.num_nodes(), 3);
    assert_eq!(g.num_edges(), 2);

    // Directed: edges only go one way
    assert_eq!(g.neighbors(0), &[1]);
    assert_eq!(g.neighbors(1), &[2]);
    assert!(g.neighbors(2).is_empty()); // no outgoing edges
}

#[test]
fn test_from_edges_with_gaps() {
    // Node IDs don't have to be contiguous
    let g = Graph::from_edges(&[(0, 5), (5, 10)], false);
    assert_eq!(g.num_nodes(), 11); // max node + 1
    assert_eq!(g.num_edges(), 2);

    assert_eq!(g.neighbors(0), &[5]);
    assert_eq!(g.neighbors(5), &[0, 10]);
    assert!(g.neighbors(1).is_empty()); // isolated node
}

#[test]
fn test_from_edges_duplicate_edges() {
    // Duplicate edges should be deduplicated
    let g = Graph::from_edges(&[(0, 1), (0, 1), (1, 0)], false);
    assert_eq!(g.num_nodes(), 2);

    // Should only have one edge (0,1) in undirected graph
    assert_eq!(g.neighbors(0), &[1]);
    assert_eq!(g.neighbors(1), &[0]);
}

#[test]
fn test_from_edges_self_loop() {
    let g = Graph::from_edges(&[(0, 0), (0, 1)], false);
    assert_eq!(g.num_nodes(), 2);

    // Self-loop should appear once
    assert_eq!(g.neighbors(0), &[0, 1]);
}

#[test]
fn test_neighbors_invalid_node() {
    let g = Graph::from_edges(&[(0, 1)], false);
    assert!(g.neighbors(999).is_empty()); // non-existent node
}

#[test]
fn test_degree_centrality_empty() {
    let g = Graph::new(false);
    let dc = g.degree_centrality();
    assert_eq!(dc.len(), 0);
}

#[test]
fn test_degree_centrality_single_node() {
    let g = Graph::from_edges(&[(0, 0)], false);
    let dc = g.degree_centrality();
    assert_eq!(dc[&0], 0.0); // single node, normalized degree is 0
}

#[test]
fn test_degree_centrality_star_graph() {
    // Star graph: center node connected to 3 leaves
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    let dc = g.degree_centrality();

    assert_eq!(dc[&0], 1.0); // center: degree 3 / (4-1) = 1.0
    assert!((dc[&1] - 1.0 / 3.0).abs() < 1e-6); // leaves: degree 1 / 3
    assert!((dc[&2] - 1.0 / 3.0).abs() < 1e-6);
    assert!((dc[&3] - 1.0 / 3.0).abs() < 1e-6);
}

#[test]
fn test_degree_centrality_complete_graph() {
    // Complete graph K4: every node connected to every other
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
    let dc = g.degree_centrality();

    // All nodes have degree 3 in K4, normalized: 3/3 = 1.0
    for v in 0..4 {
        assert_eq!(dc[&v], 1.0);
    }
}

#[test]
fn test_degree_centrality_path_graph() {
    // Path graph: 0 -- 1 -- 2 -- 3
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
    let dc = g.degree_centrality();

    // Endpoints have degree 1, middle nodes have degree 2
    assert!((dc[&0] - 1.0 / 3.0).abs() < 1e-6);
    assert!((dc[&1] - 2.0 / 3.0).abs() < 1e-6);
    assert!((dc[&2] - 2.0 / 3.0).abs() < 1e-6);
    assert!((dc[&3] - 1.0 / 3.0).abs() < 1e-6);
}

#[test]
fn test_degree_centrality_directed() {
    // Directed: only count outgoing edges
    let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], true);
    let dc = g.degree_centrality();

    assert!((dc[&0] - 2.0 / 2.0).abs() < 1e-6); // 2 outgoing edges
    assert!((dc[&1] - 1.0 / 2.0).abs() < 1e-6); // 1 outgoing edge
    assert_eq!(dc[&2], 0.0); // 0 outgoing edges
}

// PageRank tests

#[test]
fn test_pagerank_empty() {
    let g = Graph::new(true);
    let pr = g
        .pagerank(0.85, 100, 1e-6)
        .expect("pagerank should succeed for empty graph");
    assert!(pr.is_empty());
}

#[test]
fn test_pagerank_single_node() {
    let g = Graph::from_edges(&[(0, 0)], true);
    let pr = g
        .pagerank(0.85, 100, 1e-6)
        .expect("pagerank should succeed for single node graph");
    assert_eq!(pr.len(), 1);
    assert!((pr[0] - 1.0).abs() < 1e-6); // Single node has all rank
}

#[test]
fn test_pagerank_sum_equals_one() {
    // PageRank scores must sum to 1.0 (within numerical precision)
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
    let pr = g
        .pagerank(0.85, 100, 1e-6)
        .expect("pagerank should converge for cycle graph");
    let sum: f64 = pr.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10); // Kahan ensures high precision
}

#[test]
fn test_pagerank_cycle_graph() {
    // Cycle graph: 0 -> 1 -> 2 -> 0
    // All nodes should have equal PageRank (by symmetry)
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
    let pr = g
        .pagerank(0.85, 100, 1e-6)
        .expect("pagerank should converge for symmetric cycle");

    assert_eq!(pr.len(), 3);
    // All nodes have equal rank in symmetric cycle
    assert!((pr[0] - 1.0 / 3.0).abs() < 1e-6);
    assert!((pr[1] - 1.0 / 3.0).abs() < 1e-6);
    assert!((pr[2] - 1.0 / 3.0).abs() < 1e-6);
}

#[test]
fn test_pagerank_star_graph_directed() {
    // Star graph: 0 -> {1, 2, 3}
    // Node 0 distributes rank equally to 1, 2, 3
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], true);
    let pr = g
        .pagerank(0.85, 100, 1e-6)
        .expect("pagerank should converge for directed star graph");

    assert_eq!(pr.len(), 4);
    // Leaves have no incoming edges except from 0
    // Node 0 has no incoming edges (lowest rank)
    assert!(pr[0] < pr[1]); // 0 has lowest rank
    assert!((pr[1] - pr[2]).abs() < 1e-6); // leaves have equal rank
    assert!((pr[2] - pr[3]).abs() < 1e-6);
}

#[test]
fn test_pagerank_convergence() {
    // Test that PageRank converges within max_iter
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (1, 0)], true);
    let pr = g
        .pagerank(0.85, 100, 1e-6)
        .expect("pagerank should converge within max iterations");

    // Should converge (not hit max_iter)
    assert_eq!(pr.len(), 3);
    assert!((pr.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

#[test]
fn test_pagerank_no_outgoing_edges() {
    // Node with no outgoing edges (dangling node)
    let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
    let pr = g
        .pagerank(0.85, 100, 1e-6)
        .expect("pagerank should handle dangling nodes correctly");

    // Node 2 has no outgoing edges, but should still have rank
    assert_eq!(pr.len(), 3);
    assert!(pr[2] > 0.0);
    assert!((pr.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

#[test]
fn test_pagerank_undirected() {
    // Undirected graph: each edge goes both ways
    let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
    let pr = g
        .pagerank(0.85, 100, 1e-6)
        .expect("pagerank should converge for undirected path graph");

    assert_eq!(pr.len(), 3);
    // Middle node should have highest rank
    assert!(pr[1] > pr[0]);
    assert!(pr[1] > pr[2]);
    assert!((pr[0] - pr[2]).abs() < 1e-6); // endpoints equal
}

// Betweenness centrality tests (moved to centrality module, using GraphCentrality trait)

#[test]
fn test_betweenness_centrality_empty() {
    let g = Graph::new(false);
    let bc = g.betweenness_centrality();
    assert!(bc.is_empty());
}

#[test]
fn test_betweenness_centrality_single_node() {
    let g = Graph::from_edges(&[(0, 0)], false);
    let bc = g.betweenness_centrality();
    assert_eq!(bc.len(), 1);
    assert_eq!(bc[0], 0.0); // Single node has no betweenness
}

#[test]
fn test_betweenness_centrality_path_graph() {
    // Path graph: 0 -- 1 -- 2
    // Middle node lies on all paths between endpoints
    let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
    let bc = g.betweenness_centrality();

    assert_eq!(bc.len(), 3);
    // Middle node has highest betweenness (all paths go through it)
    assert!(bc[1] > bc[0]);
    assert!(bc[1] > bc[2]);
    // Endpoints should have equal betweenness (by symmetry)
    assert!((bc[0] - bc[2]).abs() < 1e-6);
}

#[test]
fn test_betweenness_centrality_star_graph() {
    // Star graph: center (0) connected to leaves {1, 2, 3}
    // Center lies on all paths between leaves
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    let bc = g.betweenness_centrality();

    assert_eq!(bc.len(), 4);
    // Center has highest betweenness
    assert!(bc[0] > bc[1]);
    assert!(bc[0] > bc[2]);
    assert!(bc[0] > bc[3]);
    // Leaves should have equal betweenness (by symmetry)
    assert!((bc[1] - bc[2]).abs() < 1e-6);
    assert!((bc[2] - bc[3]).abs() < 1e-6);
}

#[test]
fn test_betweenness_centrality_cycle_graph() {
    // Cycle graph: 0 -- 1 -- 2 -- 3 -- 0
    // All nodes have equal betweenness by symmetry
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], false);
    let bc = g.betweenness_centrality();

    assert_eq!(bc.len(), 4);
    // All nodes should have equal betweenness
    for i in 0..4 {
        for j in i + 1..4 {
            assert!((bc[i] - bc[j]).abs() < 1e-6);
        }
    }
}

#[test]
fn test_betweenness_centrality_complete_graph() {
    // Complete graph K4: every node connected to every other
    // All nodes have equal betweenness by symmetry
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
    let bc = g.betweenness_centrality();

    assert_eq!(bc.len(), 4);
    // All nodes should have equal betweenness (by symmetry)
    for i in 0..4 {
        for j in i + 1..4 {
            assert!((bc[i] - bc[j]).abs() < 1e-6);
        }
    }
}

#[test]
fn test_betweenness_centrality_bridge_graph() {
    // Bridge graph: (0 -- 1) -- 2 -- (3 -- 4)
    // Node 2 is a bridge and should have high betweenness
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)], false);
    let bc = g.betweenness_centrality();

    assert_eq!(bc.len(), 5);
    // Bridge node (2) has highest betweenness
    assert!(bc[2] > bc[0]);
    assert!(bc[2] > bc[1]);
    assert!(bc[2] > bc[3]);
    assert!(bc[2] > bc[4]);
    // Nodes 1 and 3 also have some betweenness (but less than 2)
    assert!(bc[1] > bc[0]);
    assert!(bc[3] > bc[4]);
}

#[test]
fn test_betweenness_centrality_directed() {
    // Directed path: 0 -> 1 -> 2
    let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
    let bc = g.betweenness_centrality();

    assert_eq!(bc.len(), 3);
    // In a directed path, middle node should have positive betweenness
    // (it lies on the path from 0 to 2)
    // All nodes should have some betweenness in directed graphs
    assert!(bc.iter().any(|&x| x > 0.0));
}

#[test]
fn test_betweenness_centrality_disconnected() {
    // Disconnected graph: (0 -- 1) and (2 -- 3)
    let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
    let bc = g.betweenness_centrality();

    assert_eq!(bc.len(), 4);
    // Nodes within same component should have equal betweenness
    assert!((bc[0] - bc[1]).abs() < 1e-6);
    assert!((bc[2] - bc[3]).abs() < 1e-6);
}

// Community Detection Tests

#[test]
fn test_modularity_empty_graph() {
    let g = Graph::new(false);
    let communities = vec![];
    let modularity = g.modularity(&communities);
    assert_eq!(modularity, 0.0);
}

#[test]
fn test_modularity_single_community() {
    // Triangle: all nodes in one community
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
    let communities = vec![vec![0, 1, 2]];
    let modularity = g.modularity(&communities);
    // For single community covering whole graph, Q = 0
    assert!((modularity - 0.0).abs() < 1e-6);
}

#[test]
fn test_modularity_two_communities() {
    // Two triangles connected by single edge: 0-1-2 and 3-4-5, edge 2-3
    let g = Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (2, 0), // Triangle 1
            (3, 4),
            (4, 5),
            (5, 3), // Triangle 2
            (2, 3), // Inter-community edge
        ],
        false,
    );

    let communities = vec![vec![0, 1, 2], vec![3, 4, 5]];
    let modularity = g.modularity(&communities);

    // Should have positive modularity (good community structure)
    assert!(modularity > 0.0);
    assert!(modularity < 1.0); // Not perfect due to inter-community edge
}

#[test]
fn test_modularity_perfect_split() {
    // Two disconnected triangles
    let g = Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (2, 0), // Triangle 1
            (3, 4),
            (4, 5),
            (5, 3), // Triangle 2
        ],
        false,
    );

    let communities = vec![vec![0, 1, 2], vec![3, 4, 5]];
    let modularity = g.modularity(&communities);

    // Perfect split should have high modularity
    assert!(modularity > 0.5);
}

include!("core_part_02.rs");
include!("core_part_03.rs");
include!("core_part_04.rs");
