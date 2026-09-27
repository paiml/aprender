// =========================================================================
// FALSIFY-DT: decision-tree-v1.yaml contract (aprender DecisionTreeClassifier)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had proptest DT tests but zero inline FALSIFY-DT-* tests
//   Why 2: proptests live in tests/contracts/, not near the implementation
//   Why 3: no mapping from decision-tree-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: CART was "obviously correct" (textbook algorithm)
//
// References:
//   - provable-contracts/contracts/decision-tree-v1.yaml
//   - Breiman et al. (1984) "Classification and Regression Trees"
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-DT-001: Predictions in label range — predict(x) ∈ training labels
#[test]
fn falsify_dt_001_predictions_in_label_range() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0],
    )
    .expect("valid matrix");
    let y = vec![0_usize, 0, 1, 1, 2, 2];

    let mut dt = DecisionTreeClassifier::new();
    dt.fit(&x, &y).expect("fit succeeds");

    let preds = dt.predict(&x);
    for (i, &p) in preds.iter().enumerate() {
        assert!(
            p <= 2,
            "FALSIFIED DT-001: prediction[{i}] = {p}, not in [0, 2]"
        );
    }
}

/// FALSIFY-DT-002: Deterministic — same input produces same output
#[test]
fn falsify_dt_002_deterministic() {
    let x =
        Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).expect("valid matrix");
    let y = vec![0_usize, 0, 1, 1];

    let mut dt = DecisionTreeClassifier::new();
    dt.fit(&x, &y).expect("fit");

    let p1 = dt.predict(&x);
    let p2 = dt.predict(&x);
    assert_eq!(p1, p2, "FALSIFIED DT-002: predictions differ on same input");
}

/// FALSIFY-DT-003: Perfect fit on separable data
///
/// A decision tree should perfectly classify linearly separable data.
#[test]
fn falsify_dt_003_perfect_separable() {
    let x = Matrix::from_vec(4, 1, vec![0.0, 1.0, 10.0, 11.0]).expect("valid matrix");
    let y = vec![0_usize, 0, 1, 1];

    let mut dt = DecisionTreeClassifier::new();
    dt.fit(&x, &y).expect("fit");

    let preds = dt.predict(&x);
    assert_eq!(
        preds, y,
        "FALSIFIED DT-003: tree cannot perfectly fit separable data"
    );
}

/// FALSIFY-DT-004: Prediction count matches input count
#[test]
fn falsify_dt_004_prediction_count() {
    let x_train =
        Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).expect("valid");
    let y_train = vec![0_usize, 0, 1, 1];

    let mut dt = DecisionTreeClassifier::new();
    dt.fit(&x_train, &y_train).expect("fit");

    let x_test = Matrix::from_vec(3, 2, vec![0.5, 0.5, 1.5, 1.5, 2.5, 2.5]).expect("valid");
    let preds = dt.predict(&x_test);
    assert_eq!(
        preds.len(),
        3,
        "FALSIFIED DT-004: {} predictions for 3 inputs",
        preds.len()
    );
}

/// FALSIFY-DT-008: the single-pass split search is bit-identical to the rescan
///
/// #3815 replaced an `O(n²)` rescan-per-candidate split search with a
/// sort-once single pass. A split search that picks a *different* threshold is
/// a different classifier, so "equivalent" is not enough — the two must agree
/// on the raw bits of both the threshold and the gain.
///
/// The oracle is the previous implementation itself
/// (`find_best_split_for_feature_rescan`), kept verbatim under `cfg(test)`, so
/// this compares the new algorithm against the code it replaced rather than
/// against a second derivation of the new algorithm's own premise.
#[test]
fn falsify_dt_008_split_search_matches_rescan_bit_for_bit() {
    use crate::tree::helpers::{find_best_split_for_feature, find_best_split_for_feature_rescan};

    fn check(x: &[f32], y: &[usize], case: &str) {
        let fast = find_best_split_for_feature(x, y);
        let oracle = find_best_split_for_feature_rescan(x, y);

        match (fast, oracle) {
            (None, None) => {}
            (Some((tf, gf)), Some((to, go))) => assert_eq!(
                (tf.to_bits(), gf.to_bits()),
                (to.to_bits(), go.to_bits()),
                "FALSIFIED DT-008 [{case}]: single pass chose ({tf}, {gf}), rescan chose ({to}, {go})"
            ),
            (f, o) => panic!("FALSIFIED DT-008 [{case}]: single pass {f:?} vs rescan {o:?}"),
        }
    }

    // Structured edge cases: too few samples, one distinct value, pure labels,
    // an exact two-way split, and values closer together than the 1e-10
    // tolerance the candidate scan dedups on.
    check(&[1.0], &[0], "single sample");
    check(&[1.0, 1.0, 1.0], &[0, 1, 0], "all values identical");
    check(&[0.0, 1.0, 2.0, 3.0], &[1, 1, 1, 1], "pure labels");
    check(
        &[0.0, 1.0, 10.0, 11.0],
        &[0, 0, 1, 1],
        "clean two-way split",
    );
    check(
        &[1.0, 1.0 + 1e-12, 2.0, 2.0],
        &[0, 1, 1, 0],
        "sub-tolerance neighbours",
    );
    check(
        &[-3.5, -3.5, 0.0, 7.25, 7.25],
        &[2, 2, 0, 5, 5],
        "sparse class labels",
    );

    // A sample sitting EXACTLY on a candidate threshold, which is the only way
    // `value <= threshold` and `value < threshold` can disagree. It needs the
    // 1e-10 dedup tolerance to be coarser than the float grid, so it only
    // exists at magnitudes near 1e-10: `1e-10` is dropped as a duplicate of
    // `0.0`, `2e-10` is kept, and `midpoint(0, 2e-10)` lands back on the
    // dropped value. Randomised columns at ordinary magnitudes never reach
    // this — a `<=`→`<` mutant survived them all.
    check(
        &[0.0, 1e-10, 2e-10, 3e-10],
        &[0, 1, 0, 1],
        "sample on the threshold, alternating",
    );
    check(
        &[0.0, 1e-10, 2e-10, 3e-10],
        &[0, 0, 1, 1],
        "sample on the threshold, separable",
    );
    check(
        &[0.0, 1e-10, 1e-10, 2e-10],
        &[1, 0, 0, 1],
        "threshold sample duplicated",
    );
    check(
        &[-2e-10, -1e-10, 0.0, 1e-10],
        &[0, 0, 1, 1],
        "sample on the threshold, negative",
    );

    // Randomised cases. A small quantised value range is deliberate: it forces
    // duplicate feature values, empty-side candidates and gain ties, which is
    // where an index-shifted rewrite would diverge from the rescan.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize
    };

    for case in 0..500 {
        let n = 2 + next() % 64;
        let n_classes = 2 + next() % 4;
        let spread = 1 + next() % 10;
        // Half the cases at ordinary magnitudes, half at the 1e-10 scale where
        // the dedup tolerance is coarser than the float grid and samples can
        // land exactly on a candidate threshold.
        let quantum = if case % 2 == 0 { 0.25 } else { 1e-10 };
        let x: Vec<f32> = (0..n).map(|_| (next() % spread) as f32 * quantum).collect();
        let y: Vec<usize> = (0..n).map(|_| next() % n_classes).collect();
        check(
            &x,
            &y,
            &format!("random case {case} (n={n}, k={n_classes}, q={quantum})"),
        );
    }

    // A continuous column, where every value is its own candidate threshold —
    // the shape that made the rescan quadratic.
    let n = 400;
    let x: Vec<f32> = (0..n).map(|_| next() as f32 / usize::MAX as f32).collect();
    let y: Vec<usize> = (0..n).map(|i| usize::from(x[i] > 0.4)).collect();
    check(&x, &y, "continuous column");
}

/// FALSIFY-DT-009: root split search is sub-quadratic in row count
///
/// #3815. Both sides of the 60s budget are measured, in this same debug build,
/// on this shape (200k rows, one continuous column, `max_depth = 1`, so exactly
/// one split is searched and the timing is the split search and nothing else):
///
/// | split search | 200k rows | vs budget |
/// |---|---|---|
/// | single pass (this code) | 0.25s | 240x under |
/// | rescan (pre-#3815) | ~1713s | 28x over |
///
/// The rescan figure is `find_best_split_for_feature_rescan` timed directly at
/// 6250/12500/25000 rows — 1.44s / 6.22s / 26.76s, a 4.3x cost per doubling —
/// extrapolated over the remaining three doublings at that measured exponent.
/// So the budget cannot flake (240x of headroom) and cannot pass if the rescan
/// comes back (28x over).
#[test]
fn falsify_dt_009_root_split_search_is_subquadratic() {
    use std::time::{Duration, Instant};

    const N: usize = 200_000;
    const BUDGET: Duration = Duration::from_secs(60);

    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32
    };

    let values: Vec<f32> = (0..N).map(|_| next()).collect();
    let y: Vec<usize> = values.iter().map(|&v| usize::from(v > 0.6)).collect();
    let x = Matrix::from_vec(N, 1, values).expect("valid matrix");

    // max_depth = 1 → exactly one split is searched, so the measurement is the
    // split search and nothing else.
    let mut dt = DecisionTreeClassifier::new().with_max_depth(1);
    let started = Instant::now();
    dt.fit(&x, &y).expect("fit");
    let elapsed = started.elapsed();

    assert!(
        elapsed < BUDGET,
        "FALSIFIED DT-009: one root split over {N} rows took {elapsed:?} (budget {BUDGET:?}) \
         — split search is quadratic in row count again"
    );
}
