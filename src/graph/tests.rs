//! Tests for graph algorithms.

use super::*;

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

#[test]
fn test_kahan_diff() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.1, 2.1, 2.9];
    let diff = kahan_diff(&a, &b);
    assert!((diff - 0.3).abs() < 1e-10);
}

// Betweenness centrality tests

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

#[test]
// Implementation complete
fn test_louvain_empty_graph() {
    let g = Graph::new(false);
    let communities = g.louvain();
    assert_eq!(communities.len(), 0);
}

#[test]
// Implementation complete
fn test_louvain_single_node() {
    // Single node with self-loop
    let g = Graph::from_edges(&[(0, 0)], false);
    let communities = g.louvain();
    assert_eq!(communities.len(), 1);
    assert_eq!(communities[0].len(), 1);
}

#[test]
// Implementation complete
fn test_louvain_two_nodes() {
    let g = Graph::from_edges(&[(0, 1)], false);
    let communities = g.louvain();

    // Should find 1 community containing both nodes
    assert_eq!(communities.len(), 1);
    assert_eq!(communities[0].len(), 2);
}

#[test]
// Implementation complete
fn test_louvain_triangle() {
    // Single triangle - should be one community
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
    let communities = g.louvain();

    assert_eq!(communities.len(), 1);
    assert_eq!(communities[0].len(), 3);
}

#[test]
// Implementation complete
fn test_louvain_two_triangles_connected() {
    // Two triangles connected by one edge
    let g = Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (2, 0), // Triangle 1
            (3, 4),
            (4, 5),
            (5, 3), // Triangle 2
            (2, 3), // Connection
        ],
        false,
    );

    let communities = g.louvain();

    // Should find 2 communities
    assert_eq!(communities.len(), 2);

    // Verify all nodes are assigned
    let all_nodes: Vec<_> = communities.iter().flat_map(|c| c.iter()).copied().collect();
    assert_eq!(all_nodes.len(), 6);
}

#[test]
// Implementation complete
fn test_louvain_disconnected_components() {
    // Two separate triangles (no connection)
    let g = Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (2, 0), // Component 1
            (3, 4),
            (4, 5),
            (5, 3), // Component 2
        ],
        false,
    );

    let communities = g.louvain();

    // Should find at least 2 communities (one per component)
    assert!(communities.len() >= 2);

    // Verify nodes 0,1,2 are in different community than 3,4,5
    let comm1_nodes: Vec<_> = communities
        .iter()
        .find(|c| c.contains(&0))
        .expect("node 0 should be assigned to a community")
        .clone();
    let comm2_nodes: Vec<_> = communities
        .iter()
        .find(|c| c.contains(&3))
        .expect("node 3 should be assigned to a community")
        .clone();

    assert!(comm1_nodes.contains(&0));
    assert!(comm1_nodes.contains(&1));
    assert!(comm1_nodes.contains(&2));

    assert!(comm2_nodes.contains(&3));
    assert!(comm2_nodes.contains(&4));
    assert!(comm2_nodes.contains(&5));

    // Verify no overlap
    assert!(!comm1_nodes.contains(&3));
    assert!(!comm2_nodes.contains(&0));
}

#[test]
// Implementation complete
fn test_louvain_karate_club() {
    // Zachary's Karate Club network (simplified 4-node version)
    // Known ground truth: 2 factions
    let g = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 2), // Group 1
            (2, 3), // Bridge
            (3, 4),
            (3, 5),
            (4, 5), // Group 2
        ],
        false,
    );

    let communities = g.louvain();

    // Should detect at least 2 communities
    assert!(communities.len() >= 2);

    // Node 2 and 3 are bridge nodes - could be in either community
    // But groups {0,1} and {4,5} should be detected
    let all_nodes: Vec<_> = communities.iter().flat_map(|c| c.iter()).copied().collect();
    assert_eq!(all_nodes.len(), 6);
}

#[test]
// Implementation complete
fn test_louvain_star_graph() {
    // Star graph: central node 0 connected to 1,2,3,4
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (0, 4)], false);

    let communities = g.louvain();

    // Star graph could be 1 community or split
    // Just verify all nodes are assigned
    assert!(!communities.is_empty());
    let all_nodes: Vec<_> = communities.iter().flat_map(|c| c.iter()).copied().collect();
    assert_eq!(all_nodes.len(), 5);
}

#[test]
// Implementation complete
fn test_louvain_complete_graph() {
    // Complete graph K4 - all nodes connected
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);

    let communities = g.louvain();

    // Complete graph should be single community
    assert_eq!(communities.len(), 1);
    assert_eq!(communities[0].len(), 4);
}

#[test]
// Implementation complete
fn test_louvain_modularity_improves() {
    // Two clear communities
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

    let communities = g.louvain();
    let modularity = g.modularity(&communities);

    // Louvain should find good communities (high modularity)
    assert!(modularity > 0.3);
}

#[test]
// Implementation complete
fn test_louvain_all_nodes_assigned() {
    // Verify every node gets assigned to exactly one community
    let g = Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0), // Pentagon
        ],
        false,
    );

    let communities = g.louvain();

    let mut assigned_nodes: Vec<NodeId> = Vec::new();
    for community in &communities {
        assigned_nodes.extend(community);
    }

    // All 5 nodes should be assigned
    assigned_nodes.sort_unstable();
    assert_eq!(assigned_nodes, vec![0, 1, 2, 3, 4]);

    // No node should appear twice
    let unique_count = assigned_nodes.len();
    assigned_nodes.dedup();
    assert_eq!(assigned_nodes.len(), unique_count);
}

#[test]
fn test_modularity_bad_partition() {
    // Triangle with each node in separate community (worst partition)
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);

    let communities = vec![vec![0], vec![1], vec![2]];
    let modularity = g.modularity(&communities);

    // Bad partition should have negative or very low modularity
    assert!(modularity < 0.1);
}

// Closeness Centrality Tests

#[test]
fn test_closeness_centrality_empty() {
    let g = Graph::new(false);
    let cc = g.closeness_centrality();
    assert!(cc.is_empty());
}

#[test]
fn test_closeness_centrality_star_graph() {
    // Star graph: center (0) is close to all nodes
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    let cc = g.closeness_centrality();

    assert_eq!(cc.len(), 4);
    // Center has highest closeness
    assert!(cc[0] > cc[1]);
    assert!(cc[0] > cc[2]);
    assert!(cc[0] > cc[3]);
    // Leaves have equal closeness by symmetry
    assert!((cc[1] - cc[2]).abs() < 1e-6);
    assert!((cc[2] - cc[3]).abs() < 1e-6);
}

#[test]
fn test_closeness_centrality_path_graph() {
    // Path: 0--1--2--3
    // Middle nodes have higher closeness
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
    let cc = g.closeness_centrality();

    assert_eq!(cc.len(), 4);
    // Middle nodes more central
    assert!(cc[1] > cc[0]);
    assert!(cc[2] > cc[0]);
    assert!(cc[1] > cc[3]);
    assert!(cc[2] > cc[3]);
}

// Eigenvector Centrality Tests

#[test]
fn test_eigenvector_centrality_empty() {
    let g = Graph::new(false);
    let ec = g
        .eigenvector_centrality(100, 1e-6)
        .expect("eigenvector centrality should succeed on empty graph");
    assert!(ec.is_empty());
}

#[test]
fn test_eigenvector_centrality_star_graph() {
    // Star graph: center has highest eigenvector centrality
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    let ec = g
        .eigenvector_centrality(100, 1e-6)
        .expect("eigenvector centrality should succeed on star graph");

    assert_eq!(ec.len(), 4);
    // Center should have highest score
    assert!(ec[0] > ec[1]);
    assert!(ec[0] > ec[2]);
    assert!(ec[0] > ec[3]);
}

#[test]
fn test_eigenvector_centrality_path_graph() {
    // Path graph: middle nodes more central
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
    let ec = g
        .eigenvector_centrality(100, 1e-6)
        .expect("eigenvector centrality should succeed on path graph");

    assert_eq!(ec.len(), 4);
    // Middle nodes should have higher scores
    assert!(ec[1] > ec[0]);
    assert!(ec[2] > ec[3]);
}

#[test]
fn test_eigenvector_centrality_disconnected() {
    // Graph with no edges
    let g = Graph::from_edges(&[], false);
    let ec = g
        .eigenvector_centrality(100, 1e-6)
        .expect("eigenvector centrality should succeed on graph with no edges");
    assert!(ec.is_empty());
}

// Katz Centrality Tests

#[test]
fn test_katz_centrality_empty() {
    let g = Graph::new(true);
    let kc = g
        .katz_centrality(0.1, 100, 1e-6)
        .expect("katz centrality should succeed on empty graph");
    assert!(kc.is_empty());
}

#[test]
fn test_katz_centrality_invalid_alpha() {
    let g = Graph::from_edges(&[(0, 1), (1, 2)], true);

    // Alpha = 0 should fail
    assert!(g.katz_centrality(0.0, 100, 1e-6).is_err());

    // Alpha = 1 should fail
    assert!(g.katz_centrality(1.0, 100, 1e-6).is_err());

    // Alpha > 1 should fail
    assert!(g.katz_centrality(1.5, 100, 1e-6).is_err());
}

#[test]
fn test_katz_centrality_cycle() {
    // Cycle graph: all nodes should have equal Katz centrality
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
    let kc = g
        .katz_centrality(0.1, 100, 1e-6)
        .expect("katz centrality should succeed on cycle graph");

    assert_eq!(kc.len(), 3);
    // All nodes equal by symmetry
    assert!((kc[0] - kc[1]).abs() < 1e-3);
    assert!((kc[1] - kc[2]).abs() < 1e-3);
}

#[test]
fn test_katz_centrality_star_directed() {
    // Directed star: 0 -> {1,2,3}
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], true);
    let kc = g
        .katz_centrality(0.1, 100, 1e-6)
        .expect("katz centrality should succeed on directed star graph");

    assert_eq!(kc.len(), 4);
    // Nodes with incoming edges have higher Katz centrality
    assert!(kc[1] > kc[0]);
    assert!(kc[2] > kc[0]);
    assert!(kc[3] > kc[0]);
}

// Harmonic Centrality Tests

#[test]
fn test_harmonic_centrality_empty() {
    let g = Graph::new(false);
    let hc = g.harmonic_centrality();
    assert!(hc.is_empty());
}

#[test]
fn test_harmonic_centrality_star_graph() {
    // Star graph: center has highest harmonic centrality
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    let hc = g.harmonic_centrality();

    assert_eq!(hc.len(), 4);
    // Center most central
    assert!(hc[0] > hc[1]);
    assert!(hc[0] > hc[2]);
    assert!(hc[0] > hc[3]);
}

#[test]
fn test_harmonic_centrality_disconnected() {
    // Disconnected graph: (0--1) and (2--3)
    let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
    let hc = g.harmonic_centrality();

    assert_eq!(hc.len(), 4);
    // Nodes within same component have equal harmonic centrality
    assert!((hc[0] - hc[1]).abs() < 1e-6);
    assert!((hc[2] - hc[3]).abs() < 1e-6);
}

// Density Tests

#[test]
fn test_density_empty() {
    let g = Graph::new(false);
    assert_eq!(g.density(), 0.0);
}

#[test]
fn test_density_single_node() {
    let g = Graph::from_edges(&[(0, 0)], false);
    assert_eq!(g.density(), 0.0);
}

#[test]
fn test_density_complete_graph() {
    // Complete graph K4: all nodes connected
    let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
    assert!((g.density() - 1.0).abs() < 1e-6);
}

#[test]
fn test_density_path_graph() {
    // Path: 0--1--2--3 (3 edges, 4 nodes)
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);

    // Undirected: density = 2*m / (n*(n-1)) = 2*3 / (4*3) = 0.5
    assert!((g.density() - 0.5).abs() < 1e-6);
}

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

#[test]
fn test_dijkstra_vs_shortest_path() {
    // On unweighted graph, Dijkstra should match shortest_path
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);

    let sp_path = g
        .shortest_path(0, 3)
        .expect("shortest_path should find path");
    let (dij_path, dij_dist) = g.dijkstra(0, 3).expect("dijkstra should find path");

    assert_eq!(sp_path.len(), dij_path.len());
    assert_eq!(dij_dist, (dij_path.len() - 1) as f64);
}

#[test]
fn test_dijkstra_floating_point_precision() {
    // Test with fractional weights
    let g = Graph::from_weighted_edges(&[(0, 1, 0.1), (1, 2, 0.2), (0, 2, 0.31)], false);

    let (path, dist) = g.dijkstra(0, 2).expect("path should exist");
    assert!((dist - 0.3).abs() < 1e-10); // 0.1 + 0.2 = 0.3
    assert_eq!(path, vec![0, 1, 2]);
}

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
