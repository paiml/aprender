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
