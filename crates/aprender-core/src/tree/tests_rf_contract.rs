// =========================================================================
// FALSIFY-RF: random-forest-v1.yaml contract (aprender RandomForestClassifier)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had proptest RF tests but zero inline FALSIFY-RF-* tests
//   Why 2: proptests live in tests/contracts/, not near the implementation
//   Why 3: no mapping from random-forest-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Random Forest was "obviously correct" (bagged decision trees)
//
// References:
//   - provable-contracts/contracts/random-forest-v1.yaml
//   - Breiman (2001) "Random Forests"
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-RF-001: Predictions in training label set
#[test]
fn falsify_rf_001_predictions_in_label_range() {
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 1.5, 0.5, 5.0, 5.0, 5.5, 5.5, 6.0, 5.0, 6.5, 5.5,
        ],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 0, 1, 1, 1, 1];

    let mut rf = RandomForestClassifier::new(10).with_random_state(42);
    rf.fit(&x, &y).expect("fit");

    let preds = rf.predict(&x);
    for (i, &p) in preds.iter().enumerate() {
        assert!(
            p <= 1,
            "FALSIFIED RF-001: prediction[{i}] = {p}, not in training labels {{0, 1}}"
        );
    }
}

/// FALSIFY-RF-002: Prediction count equals input sample count
#[test]
fn falsify_rf_002_prediction_count() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut rf = RandomForestClassifier::new(5).with_random_state(42);
    rf.fit(&x, &y).expect("fit");

    let preds = rf.predict(&x);
    assert_eq!(
        preds.len(),
        6,
        "FALSIFIED RF-002: {} predictions for 6 inputs",
        preds.len()
    );
}

/// FALSIFY-RF-003: Deterministic with same seed
#[test]
fn falsify_rf_003_deterministic_with_seed() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut rf1 = RandomForestClassifier::new(5).with_random_state(42);
    rf1.fit(&x, &y).expect("fit 1");
    let p1 = rf1.predict(&x);

    let mut rf2 = RandomForestClassifier::new(5).with_random_state(42);
    rf2.fit(&x, &y).expect("fit 2");
    let p2 = rf2.predict(&x);

    assert_eq!(
        p1, p2,
        "FALSIFIED RF-003: same seed produces different predictions"
    );
}

/// FALSIFY-RF-004: More trees generally doesn't degrade accuracy on training data
///
/// With sufficiently separated clusters, random forest should achieve high training accuracy.
#[test]
fn falsify_rf_004_ensemble_not_worse_than_random() {
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 10.0, 10.0, 10.1, 10.1, 10.2, 10.2, 10.3, 10.3,
        ],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 0, 1, 1, 1, 1];

    let mut rf = RandomForestClassifier::new(20).with_random_state(42);
    rf.fit(&x, &y).expect("fit");

    let preds = rf.predict(&x);
    let correct: usize = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
    let accuracy = correct as f32 / y.len() as f32;

    assert!(
        accuracy > 0.5,
        "FALSIFIED RF-004: accuracy={accuracy} <= 0.5 (worse than random)"
    );
}

/// FALSIFY-RF-005: parallel fit preserves the seed→tree binding
///
/// #3816 builds the `n_estimators` trees with rayon instead of a sequential
/// loop. Reordering execution across threads must not change which bootstrap
/// sample tree `i` gets: the ensemble is only reproducible if tree `i` still
/// carries seed `random_state + i` and the OOB complement that goes with it.
///
/// The oracle is `bootstrap_sample` called directly, outside the forest — so a
/// forest that silently paired trees with the wrong seeds (or collected them
/// out of order) fails here even though every tree individually looks fine.
#[test]
fn falsify_rf_005_parallel_fit_preserves_seed_to_tree_binding() {
    let x = Matrix::from_vec(
        10,
        2,
        vec![
            0.0, 0.0, 1.0, 0.5, 2.0, 1.0, 3.0, 1.5, 4.0, 2.0, 5.0, 2.5, 6.0, 3.0, 7.0, 3.5, 8.0,
            4.0, 9.0, 4.5,
        ],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    const SEED: u64 = 7;
    let mut rf = RandomForestClassifier::new(8)
        .with_max_depth(3)
        .with_random_state(SEED);
    rf.fit(&x, &y).expect("fit");

    assert_eq!(rf.trees.len(), 8, "FALSIFIED RF-005: wrong tree count");
    assert_eq!(rf.oob_indices.len(), 8, "FALSIFIED RF-005: wrong OOB count");

    for i in 0..8 {
        let indices = bootstrap_sample(10, Some(SEED + i as u64));

        // OOB set for tree i is the complement of ITS OWN bootstrap sample.
        let expected_oob: Vec<usize> = (0..10).filter(|idx| !indices.contains(idx)).collect();
        assert_eq!(
            rf.oob_indices[i], expected_oob,
            "FALSIFIED RF-005: tree {i} carries the OOB set of a different seed"
        );

        // And the tree itself must be the one that bootstrap sample produces.
        let mut data = Vec::with_capacity(indices.len() * 2);
        let mut labels = Vec::with_capacity(indices.len());
        for &idx in &indices {
            data.push(x.get(idx, 0));
            data.push(x.get(idx, 1));
            labels.push(y[idx]);
        }
        let boot_x = Matrix::from_vec(indices.len(), 2, data).expect("valid");
        let mut expected_tree = DecisionTreeClassifier::new().with_max_depth(3);
        expected_tree.fit(&boot_x, &labels).expect("fit");

        assert_eq!(
            rf.trees[i].predict(&x),
            expected_tree.predict(&x),
            "FALSIFIED RF-005: tree {i} is not the tree seed {} produces",
            SEED + i as u64
        );
    }
}

/// FALSIFY-RF-006: batched voting equals per-sample voting
///
/// #3816 stopped re-running `tree.predict(x)` once per (sample, tree) pair and
/// aggregates votes from one prediction vector per tree instead. The oracle is
/// the per-sample form the batch replaced — each tree asked about one row at a
/// time — so an off-by-one in the transposed indexing cannot hide.
#[test]
fn falsify_rf_006_batched_votes_match_per_sample_votes() {
    let x = Matrix::from_vec(
        12,
        2,
        vec![
            0.0, 0.0, 0.5, 0.2, 1.0, 0.4, 1.5, 0.6, 2.0, 0.8, 2.5, 1.0, 5.0, 3.0, 5.5, 3.2, 6.0,
            3.4, 6.5, 3.6, 7.0, 3.8, 7.5, 4.0,
        ],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 1];

    let mut rf = RandomForestClassifier::new(9)
        .with_max_depth(3)
        .with_random_state(11);
    rf.fit(&x, &y).expect("fit");

    let batched = rf.predict(&x);
    let proba = rf.predict_proba(&x);
    let (n_samples, n_classes) = proba.shape();
    let n_trees = rf.trees.len();

    for sample_idx in 0..n_samples {
        // Oracle: ask every tree about this one row, on its own.
        let row = Matrix::from_vec(1, 2, vec![x.get(sample_idx, 0), x.get(sample_idx, 1)])
            .expect("valid");
        let mut votes = vec![0usize; n_classes];
        for tree in &rf.trees {
            let voted = tree.predict(&row)[0];
            assert!(
                voted < n_classes,
                "FALSIFIED RF-006: class {voted} out of range"
            );
            votes[voted] += 1;
        }

        let expected = votes
            .iter()
            .enumerate()
            .max_by_key(|&(class, &count)| (count, std::cmp::Reverse(class)))
            .map(|(class, _)| class)
            .expect("at least one class");

        assert_eq!(
            batched[sample_idx], expected,
            "FALSIFIED RF-006: batched vote at row {sample_idx} disagrees with per-sample vote"
        );

        for class in 0..n_classes {
            let want = votes[class] as f32 / n_trees as f32;
            assert!(
                (proba.get(sample_idx, class) - want).abs() < 1e-6,
                "FALSIFIED RF-006: predict_proba[{sample_idx}][{class}] = {} != {want}",
                proba.get(sample_idx, class)
            );
        }
    }
}
