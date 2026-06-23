// =========================================================================
// FALSIFY-KNN: knn tests (part of linear-models-v1.yaml / classification suite)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-KNN-* tests
//   Why 2: KNN tests only in tests/contracts/, not near implementation
//   Why 3: no mapping to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: KNN was "obviously correct" (nearest neighbor lookup)
//
// References:
//   - Cover & Hart (1967) "Nearest Neighbor Pattern Classification"
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-KNN-001: Predictions in training label set
#[test]
fn falsify_knn_001_predictions_in_label_range() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 5.0, 5.0, 5.5, 5.5, 6.0, 5.0],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut knn = KNearestNeighbors::new(3);
    knn.fit(&x, &y).expect("fit");

    let preds = knn.predict(&x).expect("predict");
    for (i, &p) in preds.iter().enumerate() {
        assert!(
            p <= 1,
            "FALSIFIED KNN-001: prediction[{i}] = {p}, not in {{0, 1}}"
        );
    }
}

/// FALSIFY-KNN-002: Prediction count matches input count
#[test]
fn falsify_knn_002_prediction_count() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 5.0, 5.0, 5.5, 5.5, 6.0, 5.0],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut knn = KNearestNeighbors::new(3);
    knn.fit(&x, &y).expect("fit");

    let x_test = Matrix::from_vec(3, 2, vec![0.2, 0.2, 3.0, 3.0, 5.8, 5.8]).expect("valid");
    let preds = knn.predict(&x_test).expect("predict");
    assert_eq!(
        preds.len(),
        3,
        "FALSIFIED KNN-002: {} predictions for 3 inputs",
        preds.len()
    );
}

/// FALSIFY-KNN-003: Well-separated clusters classified correctly
#[test]
fn falsify_knn_003_separable_data() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 100.0, 100.0, 100.1, 100.1, 100.2, 100.2,
        ],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut knn = KNearestNeighbors::new(3);
    knn.fit(&x, &y).expect("fit");

    let preds = knn.predict(&x).expect("predict");
    assert_eq!(
        preds, y,
        "FALSIFIED KNN-003: KNN cannot classify well-separated clusters"
    );
}

/// FALSIFY-KNN-004: Deterministic predictions
#[test]
fn falsify_knn_004_deterministic() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut knn = KNearestNeighbors::new(1);
    knn.fit(&x, &y).expect("fit");

    let p1 = knn.predict(&x).expect("predict 1");
    let p2 = knn.predict(&x).expect("predict 2");
    assert_eq!(
        p1, p2,
        "FALSIFIED KNN-004: predictions differ on same input"
    );
}

/// FALSIFY-KNN-005: partial-select (`select_nth_unstable_by`) preserves predictions
/// under ties at the k-boundary.
///
/// PMAT-733 replaced the per-query full `sort_by(distance)` with an O(n) partial select.
/// The only case where a partial select could differ from a stable sort is when several
/// training points sit at *exactly* the same distance straddling index `k`. This fixture
/// engineers that: query (0,0) is equidistant (dist 1) from FOUR points, two of class 0
/// and two of class 1; k=3 must pick a deterministic, tie-broken k-set. The prediction
/// must be stable across repeated calls and independent of any sort instability.
#[test]
fn falsify_knn_005_partial_select_ties_stable() {
    // Four points at L2 distance 1 from the origin (two class 0, two class 1),
    // plus one far point. With k=3 the k-set is drawn from the four tied points.
    let x = Matrix::from_vec(
        5,
        2,
        vec![
            1.0, 0.0, // class 0, dist 1
            -1.0, 0.0, // class 0, dist 1
            0.0, 1.0, // class 1, dist 1
            0.0, -1.0, // class 1, dist 1
            9.0, 9.0, // class 1, far
        ],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 1, 1, 1];

    let mut knn = KNearestNeighbors::new(3);
    knn.fit(&x, &y).expect("fit");

    let query = Matrix::from_vec(1, 2, vec![0.0, 0.0]).expect("valid");
    let p1 = knn.predict(&query).expect("predict 1");
    let p2 = knn.predict(&query).expect("predict 2");

    // Determinism is the load-bearing guarantee: the tie-break (ascending training index)
    // must yield the same k-set, hence the same vote, on every call.
    assert_eq!(
        p1, p2,
        "FALSIFIED KNN-005: tie-broken prediction not deterministic"
    );
    // Tie-break picks indices {0,1,2} (smallest indices among the four tied) -> classes
    // {0,0,1} -> majority class 0. Pin the exact value so a regression in the tie-break
    // ordering is caught.
    assert_eq!(
        p1[0], 0,
        "FALSIFIED KNN-005: tie-break k-set changed (expected class 0)"
    );
}

// =========================================================================
// FALSIFY-KNN-TIE (PMAT-865): on a VOTE-COUNT tie among modes, the predicted
// label MUST be the smallest tied label, deterministically, matching
// scikit-learn KNeighborsClassifier.predict (scipy.stats.mode / argmax over
// bincount both return the lowest index on ties).
//
// RED (pre-fix): majority_vote/weighted_vote used a HashMap + max_by_key/max_by,
// which return the LAST maximal element in iteration order. HashMap iteration
// order is randomized per-process (Rust RandomState), so a tie between labels
// 0 and 2 could resolve to EITHER 0 or 2 depending on the process.
// GREEN (post-fix): BTreeMap (ascending labels) + keep-first-on-strict-greater
// always returns the smallest tied label (0).
// =========================================================================

/// FALSIFY-KNN-TIE-001: a balanced 2-neighbor vote tie (label 0 vs label 2,
/// one neighbor each) resolves to the SMALLEST label (0) via the unweighted
/// `majority_vote` helper directly. Repeated across many calls to surface any
/// nondeterminism.
#[test]
fn falsify_knn_tie_001_majority_vote_smallest_label() {
    let knn = KNearestNeighbors::new(2);

    // One neighbor of class 0, one of class 2, identical distance -> exact tie.
    let neighbors: Vec<(f32, usize)> = vec![(1.0, 0), (1.0, 2)];
    // Also test the reversed iteration order to prove order-independence: the
    // result must be the smallest label regardless of neighbor ordering.
    let neighbors_rev: Vec<(f32, usize)> = vec![(1.0, 2), (1.0, 0)];

    for _ in 0..1000 {
        assert_eq!(
            knn.majority_vote(&neighbors),
            0,
            "FALSIFIED KNN-TIE-001: tie {{0,2}} did not resolve to smallest label 0"
        );
        assert_eq!(
            knn.majority_vote(&neighbors_rev),
            0,
            "FALSIFIED KNN-TIE-001: reversed tie {{2,0}} did not resolve to smallest label 0"
        );
    }
}

/// FALSIFY-KNN-TIE-002: a balanced weighted vote tie (equal inverse-distance
/// weights for labels 0 and 2) resolves to the SMALLEST label (0) via the
/// `weighted_vote` helper directly.
#[test]
fn falsify_knn_tie_002_weighted_vote_smallest_label() {
    let knn = KNearestNeighbors::new(2).with_weights(true);

    // Identical distances => identical inverse-distance weights => weight tie.
    let neighbors: Vec<(f32, usize)> = vec![(2.0, 0), (2.0, 2)];
    let neighbors_rev: Vec<(f32, usize)> = vec![(2.0, 2), (2.0, 0)];

    for _ in 0..1000 {
        assert_eq!(
            knn.weighted_vote(&neighbors),
            0,
            "FALSIFIED KNN-TIE-002: weighted tie {{0,2}} did not resolve to smallest label 0"
        );
        assert_eq!(
            knn.weighted_vote(&neighbors_rev),
            0,
            "FALSIFIED KNN-TIE-002: reversed weighted tie {{2,0}} did not resolve to smallest label 0"
        );
    }
}

/// FALSIFY-KNN-TIE-003: end-to-end via the public `predict` API. A k=2 query
/// equidistant from one class-0 point and one class-2 point is a true vote tie;
/// the prediction MUST be the smallest label (0), deterministically across
/// repeated `predict` calls.
#[test]
fn falsify_knn_tie_003_predict_tie_smallest_label() {
    // Two training points equidistant from the query (0.0, 0.0):
    //   (1, 0) class 0, dist 1 ; (-1, 0) class 2, dist 1.
    let x = Matrix::from_vec(2, 2, vec![1.0, 0.0, -1.0, 0.0]).expect("valid");
    let y = vec![0_usize, 2];

    let mut knn = KNearestNeighbors::new(2);
    knn.fit(&x, &y).expect("fit");

    let query = Matrix::from_vec(1, 2, vec![0.0, 0.0]).expect("valid");

    // Determinism + correctness: smallest tied label (0) on every call.
    for _ in 0..200 {
        let pred = knn.predict(&query).expect("predict");
        assert_eq!(
            pred[0], 0,
            "FALSIFIED KNN-TIE-003: 2-way vote tie {{0,2}} did not resolve to smallest label 0"
        );
    }
}

/// FALSIFY-KNN-TIE-004: a three-way mode tie among labels 0, 1, 2 (k=3, one
/// neighbor each) resolves to the smallest label (0).
#[test]
fn falsify_knn_tie_004_three_way_tie_smallest_label() {
    let knn = KNearestNeighbors::new(3);
    let neighbors: Vec<(f32, usize)> = vec![(1.0, 1), (1.0, 2), (1.0, 0)];
    for _ in 0..1000 {
        assert_eq!(
            knn.majority_vote(&neighbors),
            0,
            "FALSIFIED KNN-TIE-004: 3-way tie {{0,1,2}} did not resolve to smallest label 0"
        );
    }
}

/// FALSIFY-KNN-WEIGHTED-ZERO-DISTANCE: with `weights="distance"`, an exact
/// duplicate of a training point (distance == 0) must dominate the vote — sklearn
/// gives a zero-distance neighbor INFINITE weight, so only zero-distance neighbors
/// vote. apr previously capped the zero-distance weight at 1.0, letting a cluster of
/// near (but non-zero) neighbors of a *different* label override the exact match,
/// diverging from sklearn.
///
/// Oracle (sklearn, pinned):
///   X=[[0,0],[1,0],[0,1],[1,1]], y=[1,0,0,0], KNeighborsClassifier(k=3, weights="distance")
///   predict([[0,0]]) -> [1]   predict_proba([[0,0]]) -> [[0, 1]]
/// (label 1 is the exact-match point; the other 2 of the k=3 neighbors are label 0.)
///
/// OBLIG-KNN-WEIGHTED-ZERO-DISTANCE.
#[test]
fn falsify_knn_weighted_zero_distance_dominates() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).expect("valid");
    let y = vec![1_usize, 0, 0, 0];

    let mut knn = KNearestNeighbors::new(3).with_weights(true);
    knn.fit(&x, &y).expect("fit");

    // Query == training point [0,0] (label 1). Its k=3 neighbors are
    // [0,0] (d=0, label 1), [1,0] (d=1, label 0), [0,1] (d=1, label 0).
    let q = Matrix::from_vec(1, 2, vec![0.0, 0.0]).expect("valid");

    // sklearn oracle: exact-match label dominates.
    let pred = knn.predict(&q).expect("predict")[0];
    assert_eq!(
        pred, 1,
        "FALSIFIED KNN-WEIGHTED-ZERO-DISTANCE: predict={pred}, sklearn oracle=1 \
         (zero-distance neighbor must get infinite weight, not capped at 1.0)"
    );

    // predict_proba must be [0, 1] (only the zero-distance neighbor votes).
    let proba = knn.predict_proba(&q).expect("proba");
    assert!(
        (proba[0][0] - 0.0).abs() < 1e-6 && (proba[0][1] - 1.0).abs() < 1e-6,
        "FALSIFIED KNN-WEIGHTED-ZERO-DISTANCE: proba={:?}, sklearn oracle=[0.0, 1.0]",
        proba[0]
    );
}

/// FALSIFY-KNN-WEIGHTED-NONZERO-DISTANCE: a query with NO exact match still uses the
/// ordinary 1/d weighting and must match sklearn (guards against the zero-distance
/// special case regressing the normal weighted path).
///
/// Oracle (sklearn, pinned):
///   X=[[0,0],[1,0],[0,1],[1,1]], y=[1,0,0,0], KNeighborsClassifier(k=3, weights="distance")
///   predict([[0.1,0.1]]) -> [1]  (closest point [0,0] label 1 dominates by 1/d weight)
#[test]
fn falsify_knn_weighted_nonzero_distance_parity() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).expect("valid");
    let y = vec![1_usize, 0, 0, 0];

    let mut knn = KNearestNeighbors::new(3).with_weights(true);
    knn.fit(&x, &y).expect("fit");

    let q = Matrix::from_vec(1, 2, vec![0.1, 0.1]).expect("valid");
    let pred = knn.predict(&q).expect("predict")[0];
    assert_eq!(
        pred, 1,
        "FALSIFIED KNN-WEIGHTED-NONZERO: predict={pred}, sklearn oracle=1"
    );
}

mod knn_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// Reference k-set: the multiset of `(training_index)` chosen by a *full stable sort*
    /// keyed on `(distance, ascending index)` — i.e. the exact selection rule the old
    /// `sort_by(distance)` produced. Returns the chosen indices, sorted, for comparison.
    fn reference_k_set(distances: &[(f32, usize, usize)], k: usize) -> Vec<usize> {
        let mut d = distances.to_vec();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        let mut idx: Vec<usize> = d[..k].iter().map(|&(_, i, _)| i).collect();
        idx.sort_unstable();
        idx
    }

    // FALSIFY-KNN-005-prop: `select_k_nearest` (the O(n) partial select that replaced the
    // full sort) picks exactly the same k-set of training points as a full stable sort,
    // even with heavy distance ties. This is the load-bearing correctness invariant of the
    // PMAT-733 optimization: identical k-set => identical vote => identical prediction.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn falsify_knn_005_prop_select_matches_stable_sort(
            // Small integer distances => many ties straddling the k boundary.
            dists in proptest::collection::vec(0u8..4, 1usize..=16),
            k_raw in 1usize..=16,
        ) {
            let n = dists.len();
            let k = k_raw.min(n);
            // (distance, training_index, label) — label unused by selection.
            let buf: Vec<(f32, usize, usize)> = dists
                .iter()
                .enumerate()
                .map(|(i, &d)| (d as f32, i, i % 3))
                .collect();

            let want = reference_k_set(&buf, k);

            let mut buf_sel = buf.clone();
            let chosen = super::super::select_k_nearest(&mut buf_sel, k);
            prop_assert_eq!(chosen.len(), k, "select_k_nearest returned wrong count");

            // Recover the indices the partial select kept: match each chosen (dist,label)
            // back via the index column of buf_sel[..k] (the index survives in buf_sel).
            let mut got: Vec<usize> = buf_sel[..k].iter().map(|&(_, i, _)| i).collect();
            got.sort_unstable();

            prop_assert_eq!(
                got, want,
                "FALSIFIED KNN-005-prop: partial-select k-set != stable-sort k-set (k={})",
                k
            );
        }
    }

    // FALSIFY-KNN-002-prop: Prediction count matches input count
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_knn_002_prop_prediction_count(
            n_train in 6..=12usize,
            n_test in 3..=8usize,
            seed in 0..200u32,
        ) {
            // Construct two well-separated clusters for training
            let mut x_data = Vec::with_capacity(n_train * 2);
            let mut y_data = Vec::with_capacity(n_train);
            let half = n_train / 2;
            for i in 0..half {
                let offset = (seed as f32 + i as f32) * 0.01;
                x_data.push(0.0 + offset);
                x_data.push(0.0 + offset);
                y_data.push(0_usize);
            }
            for i in 0..(n_train - half) {
                let offset = (seed as f32 + i as f32) * 0.01;
                x_data.push(10.0 + offset);
                x_data.push(10.0 + offset);
                y_data.push(1_usize);
            }
            let x = Matrix::from_vec(n_train, 2, x_data).expect("valid");

            let mut knn = KNearestNeighbors::new(3.min(half));
            knn.fit(&x, &y_data).expect("fit");

            let x_test_data: Vec<f32> = (0..n_test * 2)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 5.0 + 5.0)
                .collect();
            let x_test = Matrix::from_vec(n_test, 2, x_test_data).expect("valid");

            let preds = knn.predict(&x_test).expect("predict");
            prop_assert_eq!(
                preds.len(),
                n_test,
                "FALSIFIED KNN-002-prop: {} predictions for {} inputs",
                preds.len(), n_test
            );
        }
    }

    // FALSIFY-KNN-004-prop: Deterministic predictions for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_knn_004_prop_deterministic(
            seed in 0..200u32,
        ) {
            let x = Matrix::from_vec(
                6, 2,
                vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0,
                     5.0, 5.0, 5.5, 5.5, 6.0, 5.0],
            ).expect("valid");
            let y = vec![0_usize, 0, 0, 1, 1, 1];

            let k = ((seed % 3) + 1) as usize;
            let mut knn = KNearestNeighbors::new(k);
            knn.fit(&x, &y).expect("fit");

            let p1 = knn.predict(&x).expect("predict 1");
            let p2 = knn.predict(&x).expect("predict 2");
            prop_assert_eq!(
                p1, p2,
                "FALSIFIED KNN-004-prop: predictions differ (k={})",
                k
            );
        }
    }
}
