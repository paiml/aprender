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

mod gr_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-GR-003-prop: Self-path returns distance 0 for random chain graphs
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_gr_003_prop_self_path_zero(
            n in 3..=8usize,
            src in 0..8usize,
        ) {
            let src = src % n;
            let edges: Vec<(usize, usize, f64)> =
                (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
            let g = Graph::from_weighted_edges(&edges, false);

            let (path, dist) = g.dijkstra(src, src).expect("self-path must exist");
            prop_assert!(
                path == vec![src],
                "FALSIFIED GR-003-prop: self-path={:?}, expected [{}]",
                path, src
            );
            prop_assert!(
                dist.abs() < 1e-6,
                "FALSIFIED GR-003-prop: self-distance={}, expected 0.0",
                dist
            );
        }
    }

    /// FALSIFY-GR-004-prop: BFS finds path in connected chain graph
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_gr_004_prop_bfs_connected(
            n in 3..=8usize,
        ) {
            let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
            let g = Graph::from_edges(&edges, false);

            let path = g.shortest_path(0, n - 1);
            prop_assert!(
                path.is_some(),
                "FALSIFIED GR-004-prop: BFS returned None in connected chain (n={})",
                n
            );
            let p = path.expect("checked");
            prop_assert!(
                p.len() <= n,
                "FALSIFIED GR-004-prop: path len {} > n={} for chain graph",
                p.len(), n
            );
        }
    }
}
