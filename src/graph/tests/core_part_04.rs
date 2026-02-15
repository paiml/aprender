
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
