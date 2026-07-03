//! Pillar-1 (scikit-learn) CORRECTNESS beat: apr's `GaussianNB` matches
//! scikit-learn on ACCURACY on the same data/split — a falsifiable, per-PR
//! CI-gated benchmark. This is the accuracy half of GaussianNB's replace+beat
//! story; the speed half (`beat_sklearn_gaussiannb_speed`, ~4.9× faster after
//! the ln-hoist) already runs nightly. Together: provably accuracy-equal AND
//! faster than sklearn on its own hello-world.
//!
//! Canonical task: fit `GaussianNB` on the canonical Iris dataset with a
//! DETERMINISTIC split (sample index `i % 3 == 0` → test; n_train=100,
//! n_test=50 — identical to `beat_sklearn_iris`, so the comparison is
//! apples-to-apples). GaussianNB is closed-form/deterministic (no
//! `random_state`), so there is a single accuracy value. sklearn 1.9.0 scores
//! **1.0000** on this split (pinned 2026-07-03 via `uv run --with
//! scikit-learn`). apr must reach `>= beat_threshold` from the contract.

use aprender::classification::GaussianNB;
use aprender::datasets::load_iris;
use aprender::primitives::Matrix;
use serde::Deserialize;

#[derive(Deserialize)]
struct BeatContract {
    beat: BeatParams,
}

#[derive(Deserialize)]
struct BeatParams {
    /// apr must reach `>= beat_threshold` or CI fails.
    beat_threshold: f64,
    /// sklearn's pinned accuracy floor on this split (report line).
    baseline_floor: f64,
    /// sklearn's pinned accuracy on this split (report line).
    baseline_value: f64,
    /// The CI gate this contract is enforced by — must match this test binary.
    ci_gate_name: String,
}

fn load_beat() -> BeatParams {
    const YAML: &str =
        include_str!("../../../contracts/apr-sklearn-gaussiannb-accuracy-beat-v1.yaml");
    let contract: BeatContract = serde_yaml::from_str(YAML)
        .expect("parse contracts/apr-sklearn-gaussiannb-accuracy-beat-v1.yaml");
    contract.beat
}

#[test]
fn beat_sklearn_gaussiannb_accuracy() {
    let beat = load_beat();
    // Self-consistency: the contract names the gate that enforces it.
    assert_eq!(
        beat.ci_gate_name, "beat_sklearn_gaussiannb_accuracy",
        "contract ci_gate_name must match this test binary"
    );

    let (x, y) = load_iris();
    let n_features = x.n_cols();

    // Deterministic split: i % 3 == 0 -> test (same as beat_sklearn_iris).
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

    let mut gnb = GaussianNB::new();
    gnb.fit(&x_train, &y_train).expect("fit iris GaussianNB");
    let preds = gnb.predict(&x_test).expect("predict iris GaussianNB");

    let correct = preds.iter().zip(&y_test).filter(|(p, t)| p == t).count();
    let acc = correct as f64 / n_test as f64;

    eprintln!(
        "BEAT-SKLEARN-GAUSSIANNB-ACCURACY: apr GaussianNB test_acc = {acc:.4} \
         (scikit-learn {:.4} on same split; contract threshold {:.4})",
        beat.baseline_value, beat.beat_threshold
    );

    assert!(
        acc >= beat.beat_threshold,
        "FALSIFY-BEAT-SKLEARN-GAUSSIANNB-ACCURACY: apr GaussianNB test_acc {acc:.4} < {:.4} \
         (contract apr-sklearn-gaussiannb-accuracy-beat-v1.yaml; scikit-learn {:.4}/{:.4} on the \
         same deterministic i%3 split) — apr regressed below sklearn",
        beat.beat_threshold,
        beat.baseline_value,
        beat.baseline_floor
    );
}
