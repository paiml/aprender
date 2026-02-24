// =========================================================================
// FALSIFY-CD: Community detection contract (aprender graph)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-CD-* tests for community detection
//   Why 2: community detection tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for community detection yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Label propagation was "obviously correct" (textbook algorithm)
//
// References:
//   - Raghavan, Albert, Kumara (2007) "Near linear time algorithm to detect
//     community structures in large-scale networks"
// =========================================================================

use super::*;

/// FALSIFY-CD-001: Labels length matches node count
#[test]
fn falsify_cd_001_labels_length() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], false);

    let labels = g.label_propagation(100, Some(42));
    assert_eq!(
        labels.len(),
        g.num_nodes(),
        "FALSIFIED CD-001: labels len={}, expected {}",
        labels.len(),
        g.num_nodes()
    );
}

/// FALSIFY-CD-002: Connected clique nodes share a community label
#[test]
fn falsify_cd_002_clique_same_community() {
    // Two fully connected cliques with a single bridge
    let g = Graph::from_edges(
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
        false,
    );

    let labels = g.label_propagation(100, Some(42));

    // Within clique A, nodes should share a label
    assert_eq!(
        labels[0], labels[1],
        "FALSIFIED CD-002: nodes 0 and 1 in different communities"
    );
    assert_eq!(
        labels[1], labels[2],
        "FALSIFIED CD-002: nodes 1 and 2 in different communities"
    );

    // Within clique B
    assert_eq!(
        labels[3], labels[4],
        "FALSIFIED CD-002: nodes 3 and 4 in different communities"
    );
    assert_eq!(
        labels[4], labels[5],
        "FALSIFIED CD-002: nodes 4 and 5 in different communities"
    );
}

/// FALSIFY-CD-003: Deterministic with same seed
#[test]
fn falsify_cd_003_deterministic_with_seed() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 3)], false);

    let labels1 = g.label_propagation(100, Some(123));
    let labels2 = g.label_propagation(100, Some(123));

    assert_eq!(
        labels1, labels2,
        "FALSIFIED CD-003: same seed gave different results"
    );
}

mod cd_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-CD-001-prop: Labels length matches node count for random graphs
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_cd_001_prop_labels_length(
            n in 4..=10usize,
            seed in 0..500u32,
        ) {
            // Build a chain graph: 0-1-2-..-(n-1)
            let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
            let g = Graph::from_edges(&edges, false);

            let labels = g.label_propagation(50, Some(seed as u64));
            prop_assert_eq!(
                labels.len(),
                g.num_nodes(),
                "FALSIFIED CD-001-prop: labels len {} != nodes {}",
                labels.len(), g.num_nodes()
            );
        }
    }

    /// FALSIFY-CD-003-prop: Deterministic with same seed for random graphs
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_cd_003_prop_deterministic(
            n in 4..=8usize,
            seed in 0..500u32,
        ) {
            let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
            let g = Graph::from_edges(&edges, false);

            let l1 = g.label_propagation(50, Some(seed as u64));
            let l2 = g.label_propagation(50, Some(seed as u64));
            prop_assert_eq!(
                l1, l2,
                "FALSIFIED CD-003-prop: same seed gave different results"
            );
        }
    }
}
