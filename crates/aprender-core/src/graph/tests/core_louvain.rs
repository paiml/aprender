
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
