// =========================================================================
// FALSIFY-RK: Ranking metrics contract (aprender metrics)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-RK-* tests for ranking metrics
//   Why 2: ranking tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for ranking metrics yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Hit@K/MRR/NDCG were "obviously correct" (standard formulae)
//
// References:
//   - Jarvelin & Kekalainen (2002) "Cumulated gain-based evaluation of IR"
// =========================================================================

use crate::metrics::ranking::*;

/// FALSIFY-RK-001: Hit@K is binary: 0.0 or 1.0
#[test]
fn falsify_rk_001_hit_at_k_binary() {
    let predictions = vec![5, 3, 1, 4, 2];

    for k in 1..=5 {
        let h = hit_at_k(&predictions, &3, k);
        assert!(
            h == 0.0 || h == 1.0,
            "FALSIFIED RK-001: hit_at_k={h} for k={k}, expected 0.0 or 1.0"
        );
    }
}

/// FALSIFY-RK-002: Hit@K is monotone non-decreasing in K
#[test]
fn falsify_rk_002_hit_monotone_in_k() {
    let predictions = vec![5, 3, 1, 4, 2];
    let target = 1;

    let mut prev = 0.0_f32;
    for k in 1..=5 {
        let h = hit_at_k(&predictions, &target, k);
        assert!(
            h >= prev,
            "FALSIFIED RK-002: hit_at_k decreased from {prev} to {h} at k={k}"
        );
        prev = h;
    }
}

/// FALSIFY-RK-003: NDCG is in [0, 1]
#[test]
fn falsify_rk_003_ndcg_bounded() {
    let relevance = vec![3.0, 2.0, 3.0, 0.0, 1.0, 2.0];

    for k in 1..=6 {
        let score = ndcg_at_k(&relevance, k);
        assert!(
            (0.0..=1.0001).contains(&score),
            "FALSIFIED RK-003: NDCG@{k}={score}, expected in [0,1]"
        );
    }
}

/// FALSIFY-RK-004: Reciprocal rank is in [0, 1]
#[test]
fn falsify_rk_004_reciprocal_rank_bounded() {
    let predictions = vec![5, 3, 1, 4, 2];

    let rr = reciprocal_rank(&predictions, &3);
    assert!(
        (0.0..=1.0).contains(&rr),
        "FALSIFIED RK-004: reciprocal_rank={rr}, expected in [0,1]"
    );

    // target=3 is at position 2 → RR = 1/2 = 0.5
    assert!(
        (rr - 0.5).abs() < 1e-6,
        "FALSIFIED RK-004: reciprocal_rank={rr}, expected 0.5"
    );
}

mod rk_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-RK-001-prop: Hit@K is binary for random predictions
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_rk_001_prop_hit_binary(
            n in 3..=10usize,
            seed in 0..500u32,
        ) {
            let predictions: Vec<usize> = (0..n).map(|i| ((i + seed as usize) % n)).collect();
            let target = seed as usize % n;

            for k in 1..=n {
                let h = hit_at_k(&predictions, &target, k);
                prop_assert!(
                    h == 0.0 || h == 1.0,
                    "FALSIFIED RK-001-prop: hit_at_k={} for k={}, expected 0 or 1",
                    h, k
                );
            }
        }
    }

    /// FALSIFY-RK-003-prop: NDCG in [0, 1] for random relevance
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_rk_003_prop_ndcg_bounded(
            n in 3..=8usize,
            seed in 0..500u32,
        ) {
            let relevance: Vec<f32> = (0..n)
                .map(|i| (((i as f32 + seed as f32) * 0.37).sin().abs() * 5.0).floor())
                .collect();

            for k in 1..=n {
                let score = ndcg_at_k(&relevance, k);
                prop_assert!(
                    (-0.001..=1.001).contains(&score),
                    "FALSIFIED RK-003-prop: NDCG@{}={} not in [0,1]",
                    k, score
                );
            }
        }
    }
}
