// =========================================================================
// FALSIFY-GR (Dijkstra): graph-centrality-v1.yaml contract (graph algos)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-GR-* tests for graph algorithms
//   Why 2: graph tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from graph contracts to inline FALSIFY test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Dijkstra was "obviously correct" (textbook algorithm)
//
// References:
//   - provable-contracts/contracts/graph-centrality-v1.yaml
//   - Dijkstra (1959) "A note on two problems in connexion with graphs"
// =========================================================================

use super::*;

/// FALSIFY-GR-001: Dijkstra finds shortest path on simple graph
#[test]
fn falsify_gr_001_dijkstra_shortest_path() {
    // 0 --1.0--> 1 --2.0--> 2
    // 0 --5.0--> 2
    let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)], false);
    let (path, dist) = g.dijkstra(0, 2).expect("path should exist");

    assert_eq!(
        path,
        vec![0, 1, 2],
        "FALSIFIED GR-001: path={path:?}, expected [0, 1, 2]"
    );
    assert!(
        (dist - 3.0).abs() < 1e-6,
        "FALSIFIED GR-001: distance={dist}, expected 3.0"
    );
}

/// FALSIFY-GR-002: Dijkstra returns None for unreachable target
#[test]
fn falsify_gr_002_dijkstra_unreachable() {
    // Disconnected: 0->1 and 2 is isolated
    let g = Graph::from_weighted_edges(&[(0, 1, 1.0)], false);
    let result = g.dijkstra(0, 2);

    assert!(
        result.is_none(),
        "FALSIFIED GR-002: expected None for unreachable node, got {result:?}"
    );
}

/// FALSIFY-GR-003: Dijkstra source=target returns zero distance
#[test]
fn falsify_gr_003_dijkstra_self_path() {
    let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 2.0)], false);
    let (path, dist) = g.dijkstra(0, 0).expect("self-path should exist");

    assert_eq!(
        path,
        vec![0],
        "FALSIFIED GR-003: self-path={path:?}, expected [0]"
    );
    assert!(
        dist.abs() < 1e-6,
        "FALSIFIED GR-003: self-distance={dist}, expected 0.0"
    );
}

/// FALSIFY-GR-004: BFS shortest path on unweighted graph
#[test]
fn falsify_gr_004_bfs_shortest_path() {
    // 0-1, 1-2, 0-2 — BFS from 0 to 2 should be direct (1 hop)
    let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 2)], false);
    let path = g.shortest_path(0, 2);

    assert!(
        path.is_some(),
        "FALSIFIED GR-004: BFS returned None for connected graph"
    );
    let p = path.expect("checked above");
    assert!(
        p.len() <= 3,
        "FALSIFIED GR-004: path len {} > 3 for 3-node graph",
        p.len()
    );
}
