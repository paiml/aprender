// =========================================================================
// FALSIFY-PR: pagerank-kernel-v1.yaml contract (aprender Graph::pagerank)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had proptest PageRank tests but zero inline FALSIFY-PR-* tests
//   Why 2: proptests live in tests/contracts/, not near the implementation
//   Why 3: no mapping from pagerank-kernel-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: PageRank was "obviously correct" (standard power iteration)
//
// References:
//   - provable-contracts/contracts/pagerank-kernel-v1.yaml
//   - Brin & Page (1998) "The Anatomy of a Large-Scale Hypertextual Web Search Engine"
// =========================================================================

use super::*;

/// FALSIFY-PR-001: Probability distribution — sum(r) ≈ 1 and r_i >= 0
#[test]
fn falsify_pr_001_probability_distribution() {
    // Simple directed graph: 0→1→2→0 (cycle)
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
    let ranks = g
        .pagerank(0.85, 100, 1e-8)
        .expect("pagerank should converge");

    // Non-negativity
    for (i, &r) in ranks.iter().enumerate() {
        assert!(r >= 0.0, "FALSIFIED PR-001: rank[{i}] = {r} < 0");
    }

    // Sums to 1
    let total: f64 = ranks.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-6,
        "FALSIFIED PR-001: sum(ranks) = {total}, expected 1.0"
    );
}

/// FALSIFY-PR-002: Symmetric cycle — all nodes get equal rank
///
/// In a symmetric cycle (0→1→2→0), all nodes have equal importance.
#[test]
fn falsify_pr_002_symmetric_equal_rank() {
    let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
    let ranks = g
        .pagerank(0.85, 100, 1e-8)
        .expect("pagerank should converge");

    let expected = 1.0 / 3.0;
    for (i, &r) in ranks.iter().enumerate() {
        assert!(
            (r - expected).abs() < 1e-4,
            "FALSIFIED PR-002: rank[{i}] = {r}, expected {expected} (symmetric cycle)"
        );
    }
}

/// FALSIFY-PR-003: Hub structure — node with more incoming links ranks higher
///
/// If node 2 has more incoming edges, it should have a higher rank.
#[test]
fn falsify_pr_003_hub_ranks_higher() {
    // 0→2, 1→2, 2→0 — node 2 has 2 incoming, nodes 0,1 have 1 each
    let g = Graph::from_edges(&[(0, 2), (1, 2), (2, 0)], true);
    let ranks = g
        .pagerank(0.85, 100, 1e-8)
        .expect("pagerank should converge");

    assert!(
        ranks[2] > ranks[0],
        "FALSIFIED PR-003: hub rank[2]={} not > rank[0]={}",
        ranks[2],
        ranks[0]
    );
}

/// FALSIFY-PR-004: Single node — rank is 1.0
#[test]
fn falsify_pr_004_single_node() {
    let g = Graph::from_edges(&[(0, 0)], true);
    let ranks = g
        .pagerank(0.85, 100, 1e-8)
        .expect("pagerank should converge");

    assert_eq!(ranks.len(), 1);
    assert!(
        (ranks[0] - 1.0).abs() < 1e-6,
        "FALSIFIED PR-004: single node rank = {}, expected 1.0",
        ranks[0]
    );
}

/// FALSIFY-PR-005: Empty graph — returns empty ranks
#[test]
fn falsify_pr_005_empty_graph() {
    let g = Graph::new(true);
    let ranks = g.pagerank(0.85, 100, 1e-8).expect("empty graph");
    assert!(
        ranks.is_empty(),
        "FALSIFIED PR-005: empty graph returned {} ranks",
        ranks.len()
    );
}
