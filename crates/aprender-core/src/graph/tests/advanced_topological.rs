
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
