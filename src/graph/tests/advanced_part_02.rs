
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
