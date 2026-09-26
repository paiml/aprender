//! Falsifiers for the neighbor indexes (#3149, contracts/neighbor-index-v1.yaml).

use super::*;
use proptest::prelude::*;

/// Deterministic points. `grid` draws small integers so exact distance ties
/// are common; otherwise coordinates are continuous in [-100, 100).
fn points(seed: u64, n: usize, d: usize, grid: bool) -> Vec<f32> {
    let mut s = seed;
    (0..n * d)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            if grid {
                (z % 7) as f32 - 3.0
            } else {
                ((z >> 40) as f32 / (1u64 << 24) as f32) * 200.0 - 100.0
            }
        })
        .collect()
}

const TREE_METRICS: [Metric; 5] = [
    Metric::Euclidean,
    Metric::Manhattan,
    Metric::Chebyshev,
    Metric::Minkowski(1.5),
    Metric::Minkowski(3.0),
];

fn trees(data: &[f32], n: usize, d: usize, m: Metric) -> [(&'static str, SpatialIndex); 2] {
    [
        (
            "kd",
            SpatialIndex::build(data, n, d, m, NeighborAlgorithm::KdTree).expect("kd builds"),
        ),
        (
            "ball",
            SpatialIndex::build(data, n, d, m, NeighborAlgorithm::BallTree).expect("ball builds"),
        ),
    ]
}

/// Neighbors as comparable bits: index plus the exact distance bit pattern.
fn bits(v: &[Neighbor]) -> Vec<(usize, u32)> {
    v.iter()
        .map(|nb| (nb.index, nb.distance.to_bits()))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// FALSIFY-NBR-001: a tree's k-nearest set is EXACTLY brute force's, in the
    /// same (distance, index) order, bit for bit, ties included.
    #[test]
    fn falsify_nbr_001_knn_trees_equal_brute(
        n in 10usize..2000, d in 1usize..30, seed in any::<u64>(),
        k in 1usize..40, grid in any::<bool>(), mi in 0usize..5,
    ) {
        let m = TREE_METRICS[mi];
        let data = points(seed, n, d, grid);
        let brute = BruteForce::new(&data, n, d, m);
        let queries = points(seed ^ 0xA5A5, 6, d, grid);
        for (name, t) in trees(&data, n, d, m) {
            // Queries off the data, and every 97th data point itself (distance 0).
            let on_data = (0..n).step_by(97).map(|i| data[i * d..(i + 1) * d].to_vec());
            for q in queries.chunks(d).map(<[f32]>::to_vec).chain(on_data) {
                prop_assert_eq!(bits(&t.k_nearest(&q, k)), bits(&brute.k_nearest(&q, k)),
                    "{} n={} d={} k={} {:?}", name, n, d, k, m);
            }
        }
    }

    /// FALSIFY-NBR-002: radius queries agree with brute force, including a
    /// radius equal to an existing neighbor distance (the boundary is inclusive).
    #[test]
    fn falsify_nbr_002_radius_trees_equal_brute(
        n in 10usize..2000, d in 1usize..30, seed in any::<u64>(),
        j in 0usize..60, grid in any::<bool>(), mi in 0usize..5,
    ) {
        let m = TREE_METRICS[mi];
        let data = points(seed, n, d, grid);
        let brute = BruteForce::new(&data, n, d, m);
        let queries = points(seed ^ 0x5A5A, 4, d, grid);
        for (name, t) in trees(&data, n, d, m) {
            for q in queries.chunks(d) {
                let all = brute.k_nearest(q, n);
                let r = all[j.min(n - 1)].distance;
                prop_assert_eq!(bits(&t.within_radius(q, r)), bits(&brute.within_radius(q, r)),
                    "{} n={} d={} r={} {:?}", name, n, d, r, m);
            }
        }
    }

    /// FALSIFY-NBR-005: the metric axioms. Symmetry is exact; the triangle
    /// inequality holds within rounding for every true metric.
    #[test]
    fn falsify_nbr_005_metric_axioms(
        d in 1usize..20, seed in any::<u64>(), grid in any::<bool>(),
    ) {
        let p = points(seed, 3, d, grid);
        let (a, b, c) = (&p[..d], &p[d..2 * d], &p[2 * d..]);
        for m in TREE_METRICS.iter().copied().chain([Metric::Cosine]) {
            let ab = m.distance(a, b);
            prop_assert!(ab >= 0.0, "{:?} negative: {}", m, ab);
            prop_assert_eq!(ab.to_bits(), m.distance(b, a).to_bits(), "{:?} asymmetric", m);
            if m == Metric::Cosine {
                // Cosine is not a metric; d(x,x) is 0 only for a nonzero x.
                if a.iter().any(|&x| x != 0.0) {
                    prop_assert!(m.distance(a, a) < 1e-6);
                }
                continue;
            }
            prop_assert!(m.distance(a, a) == 0.0, "{:?} d(x,x) != 0", m);
            let (bc, ac) = (m.distance(b, c), m.distance(a, c));
            prop_assert!(ac <= (ab + bc) * (1.0 + 1e-5) + 1e-5,
                "{:?} triangle: {} > {} + {}", m, ac, ab, bc);
        }
    }
}

/// Mean distance evaluations per k=5 query over 50 uniform queries.
fn mean_evals(algo: NeighborAlgorithm, n: usize) -> f64 {
    let d = 3;
    let data = points(42, n, d, false);
    let idx = SpatialIndex::build(&data, n, d, Metric::Euclidean, algo).expect("builds");
    let queries = points(7, 50, d, false);
    let total: usize = queries
        .chunks(d)
        .map(|q| idx.k_nearest_with_stats(q, 5).1.distance_evaluations)
        .sum();
    total as f64 / 50.0
}

/// FALSIFY-NBR-003: work per query grows sub-linearly. A 100x larger data set
/// may cost at most 10x the distance evaluations (linear would be 100x), and
/// brute force, the control, must show the linear growth the bound rejects.
#[test]
fn falsify_nbr_003_node_visits_sublinear() {
    let brute = [1_000, 100_000].map(|n| mean_evals(NeighborAlgorithm::BruteForce, n));
    assert!(
        brute[1] / brute[0] > 99.0,
        "control: brute force must scale linearly, got {brute:?}"
    );
    for algo in [NeighborAlgorithm::KdTree, NeighborAlgorithm::BallTree] {
        let e = [1_000, 10_000, 100_000].map(|n| mean_evals(algo, n));
        assert!(
            e[2] / e[0] < 10.0,
            "{algo:?}: evaluations {e:?} grow faster than sub-linear"
        );
        assert!(e[1] <= e[2], "{algo:?}: evaluations {e:?} not monotone");
        assert!(
            e[2] < 1_000.0,
            "{algo:?}: {} of 100000 points scanned",
            e[2]
        );
    }
}

/// FALSIFY-NBR-004: degenerate inputs return the right answer and never panic.
#[test]
fn falsify_nbr_004_degenerate_inputs() {
    let algos = [
        NeighborAlgorithm::BruteForce,
        NeighborAlgorithm::KdTree,
        NeighborAlgorithm::BallTree,
    ];
    for algo in algos {
        // n = 0
        let empty = SpatialIndex::build(&[], 0, 3, Metric::Euclidean, algo).expect("n=0");
        assert!(empty.is_empty());
        assert!(empty.k_nearest(&[0.0; 3], 5).is_empty(), "{algo:?}");
        assert!(empty.within_radius(&[0.0; 3], 1e9).is_empty(), "{algo:?}");

        // n = 1, k > n, k = 0
        let one = SpatialIndex::build(&[1.0, 2.0], 1, 2, Metric::Euclidean, algo).expect("n=1");
        assert_eq!(
            bits(&one.k_nearest(&[1.0, 2.0], 10)),
            vec![(0, 0)],
            "{algo:?}"
        );
        assert!(one.k_nearest(&[1.0, 2.0], 0).is_empty(), "{algo:?} k=0");

        // All-identical points: every point ties at distance 0; lowest indices win.
        let same = vec![7.0_f32; 500 * 4];
        let idx = SpatialIndex::build(&same, 500, 4, Metric::Euclidean, algo).expect("same");
        let got: Vec<usize> = idx
            .k_nearest(&[7.0; 4], 3)
            .iter()
            .map(|nb| nb.index)
            .collect();
        assert_eq!(got, vec![0, 1, 2], "{algo:?} identical points");
        assert_eq!(idx.within_radius(&[7.0; 4], 0.0).len(), 500, "{algo:?}");
        assert_eq!(idx.k_nearest(&[7.0; 4], 1000).len(), 500, "{algo:?} k>n");

        // Duplicate coordinates and d = 1.
        let dup: Vec<f32> = (0..300).map(|i| (i % 5) as f32).collect();
        let idx = SpatialIndex::build(&dup, 300, 1, Metric::Manhattan, algo).expect("d=1");
        let brute = BruteForce::new(&dup, 300, 1, Metric::Manhattan);
        for q in [-1.0, 0.0, 2.5, 4.0, 9.0] {
            assert_eq!(
                bits(&idx.k_nearest(&[q], 70)),
                bits(&brute.k_nearest(&[q], 70))
            );
            assert_eq!(
                bits(&idx.within_radius(&[q], 1.0)),
                bits(&brute.within_radius(&[q], 1.0))
            );
        }

        // d = 0: every point is at distance 0.
        let idx = SpatialIndex::build(&[], 5, 0, Metric::Euclidean, algo).expect("d=0");
        assert_eq!(idx.k_nearest(&[], 2).len(), 2, "{algo:?} d=0");
    }
}

/// FALSIFY-NBR-006: a tree is never built for a metric it cannot prune with;
/// `Auto` routes those to brute force instead.
#[test]
fn falsify_nbr_006_non_metric_distances_refuse_trees() {
    let data = points(1, 200, 3, false);
    for m in [Metric::Cosine, Metric::Minkowski(0.5)] {
        assert!(SpatialIndex::build(&data, 200, 3, m, NeighborAlgorithm::KdTree).is_err());
        assert!(SpatialIndex::build(&data, 200, 3, m, NeighborAlgorithm::BallTree).is_err());
        let auto = SpatialIndex::build(&data, 200, 3, m, NeighborAlgorithm::Auto).expect("auto");
        assert_eq!(auto.algorithm(), NeighborAlgorithm::BruteForce, "{m:?}");
    }
}

#[test]
fn auto_resolution_follows_the_documented_rule() {
    use NeighborAlgorithm as A;
    let e = Metric::Euclidean;
    assert_eq!(A::Auto.resolve(64, 3, e), A::BruteForce);
    assert_eq!(A::Auto.resolve(65, 3, e), A::KdTree);
    assert_eq!(A::Auto.resolve(10_000, 16, e), A::KdTree);
    assert_eq!(A::Auto.resolve(10_000, 17, e), A::BallTree);
    assert_eq!(A::Auto.resolve(10_000, 64, e), A::BallTree);
    assert_eq!(A::Auto.resolve(10_000, 65, e), A::BruteForce);
    assert_eq!(A::Auto.resolve(10_000, 3, Metric::Cosine), A::BruteForce);
    assert_eq!(
        A::KdTree.resolve(5, 3, e),
        A::KdTree,
        "explicit choice is kept"
    );
}

#[test]
fn build_rejects_a_wrong_data_length() {
    assert!(
        SpatialIndex::build(&[1.0; 5], 2, 3, Metric::Euclidean, NeighborAlgorithm::Auto).is_err()
    );
}

#[test]
#[should_panic(expected = "query has 2 coordinates")]
fn query_of_wrong_dimension_panics() {
    let idx = SpatialIndex::build(
        &[0.0; 6],
        2,
        3,
        Metric::Euclidean,
        NeighborAlgorithm::KdTree,
    )
    .expect("builds");
    let _ = idx.k_nearest(&[0.0, 0.0], 1);
}
