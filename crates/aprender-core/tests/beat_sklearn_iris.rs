//! FALSIFY-BEAT-SKLEARN-IRIS — the Pillar-1 beat-benchmark.
//!
//! Mission ([[project_mission_four_pillars]]): aprender must BEAT scikit-learn at
//! its canonical task, where "beat" is a *falsifiable* benchmark — apr ≥ sklearn
//! on accuracy on the SAME data/split. This gate fails CI if apr's
//! `RandomForestClassifier` regresses below sklearn's pinned Iris accuracy.
//!
//! ## Pinned scikit-learn baseline
//! `RandomForestClassifier(n_estimators=100)` on the canonical Iris dataset, with
//! a DETERMINISTIC split (sample index `i % 3 == 0` → test; n_train=100,
//! n_test=50). Over `random_state` 0..4: test_acc **mean 0.9560, min 0.9400,
//! max 0.9600**. Pinned 2026-06-11 via `uv run --with scikit-learn`. apr must
//! reach **≥ 0.92** (sklearn's floor minus a 2pp margin for RF-implementation
//! differences) — a fail means apr underperforms sklearn on its own hello-world.
//!
//! The same deterministic split is used on both sides, so the comparison is
//! apples-to-apples (apr's `train_test_split` is RNG-based and would NOT match
//! sklearn's, hence the explicit `i % 3` split here).

use aprender::datasets::load_iris;
use aprender::tree::RandomForestClassifier;
use aprender::Matrix;

/// sklearn's pinned minimum test accuracy on this exact split (see module docs).
const SKLEARN_IRIS_FLOOR: f64 = 0.94;
/// apr must come within 2pp of sklearn's floor to "match/beat" on accuracy.
const BEAT_THRESHOLD: f64 = SKLEARN_IRIS_FLOOR - 0.02;

#[test]
fn beat_sklearn_iris_accuracy() {
    let (x, y) = load_iris();
    let n_features = x.n_cols();

    // Deterministic split: i % 3 == 0 -> test. Iris is stored in class-order
    // blocks of 50, so i%3 lands evenly across all three classes.
    let mut x_train = Vec::new();
    let mut y_train: Vec<usize> = Vec::new();
    let mut x_test = Vec::new();
    let mut y_test: Vec<usize> = Vec::new();
    for i in 0..x.n_rows() {
        let row: Vec<f32> = (0..n_features).map(|j| x.get(i, j)).collect();
        if i % 3 == 0 {
            x_test.extend_from_slice(&row);
            y_test.push(y[i]);
        } else {
            x_train.extend_from_slice(&row);
            y_train.push(y[i]);
        }
    }
    let n_train = y_train.len();
    let n_test = y_test.len();
    assert_eq!((n_train, n_test), (100, 50), "deterministic split shape");

    let x_train = Matrix::from_vec(n_train, n_features, x_train).expect("train dims");
    let x_test = Matrix::from_vec(n_test, n_features, x_test).expect("test dims");

    let mut rf = RandomForestClassifier::new(100)
        .with_max_depth(10)
        .with_random_state(42);
    rf.fit(&x_train, &y_train).expect("fit iris");
    let preds = rf.predict(&x_test);

    let correct = preds.iter().zip(&y_test).filter(|(p, t)| p == t).count();
    let acc = correct as f64 / n_test as f64;

    // Beat-benchmarks report their number, not just pass/fail.
    eprintln!("BEAT-SKLEARN-IRIS: apr RandomForestClassifier test_acc = {acc:.4} (scikit-learn 0.9560 mean / 0.9400 floor on same split)");

    assert!(
        acc >= BEAT_THRESHOLD,
        "FALSIFY-BEAT-SKLEARN-IRIS: apr RandomForestClassifier test_acc {acc:.4} < {BEAT_THRESHOLD:.2} \
         (scikit-learn baseline 0.94-0.96 on the same deterministic i%3 split) — apr regressed below sklearn"
    );
}
