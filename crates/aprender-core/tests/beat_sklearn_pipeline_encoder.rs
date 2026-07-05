//! Pillar-1 (scikit-learn) CORRECTNESS beat: apr's sklearn-style `Pipeline`
//! composes a preprocessing ENCODER with an estimator and matches scikit-learn's
//! `make_pipeline` on the same categorical data — a falsifiable, per-PR CI-gated
//! benchmark.
//!
//! apr already ships `OneHotEncoder`/`OrdinalEncoder` (both `impl Transformer`)
//! and a `Pipeline`, but nothing GATED that the encoder→estimator composition
//! works end-to-end and agrees with sklearn. Closes PMAT-733 with two checks:
//!   1. apr `OneHotEncoder`'s dense transform is BYTE-IDENTICAL to sklearn
//!      `OneHotEncoder(handle_unknown='ignore')` on a pinned fixture (no margin).
//!   2. apr `Pipeline(OneHotEncoder → LogisticRegression)` reaches `>=
//!      beat_threshold` test accuracy on a deterministic categorical dataset where
//!      sklearn `make_pipeline(OneHotEncoder, LogisticRegression)` scores 1.0000.
//!
//! Oracle pinned 2026-07-04 via `uv run --with scikit-learn` (sklearn 1.9.0).

use aprender::classification::LogisticRegression;
use aprender::pipeline::Pipeline;
use aprender::preprocessing::OneHotEncoder;
use aprender::primitives::{Matrix, Vector};
use aprender::traits::Transformer;
use serde::Deserialize;

#[derive(Deserialize)]
struct BeatContract {
    beat: BeatParams,
}

#[derive(Deserialize)]
struct BeatParams {
    beat_threshold: f64,
    baseline_value: f64,
    ci_gate_name: String,
}

fn load_beat() -> BeatParams {
    const YAML: &str = include_str!("../../../contracts/apr-sklearn-pipeline-encoder-beat-v1.yaml");
    let contract: BeatContract = serde_yaml::from_str(YAML)
        .expect("parse contracts/apr-sklearn-pipeline-encoder-beat-v1.yaml");
    contract.beat
}

/// Deterministic categorical dataset (identical formula on both apr and sklearn).
/// 3 categorical features, target = 1 iff (f0==2 or f1==0). Split i%4==0 → test.
fn categorical_split() -> (Matrix<f32>, Vector<f32>, Matrix<f32>, Vec<usize>) {
    const N: usize = 108;
    let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..N {
        let (f0, f1, f2) = ((i % 3) as f32, ((i / 3) % 3) as f32, ((i / 9) % 3) as f32);
        let target = usize::from(f0 == 2.0 || f1 == 0.0);
        if i % 4 == 0 {
            xte.extend_from_slice(&[f0, f1, f2]);
            yte.push(target);
        } else {
            xtr.extend_from_slice(&[f0, f1, f2]);
            ytr.push(target as f32);
        }
    }
    let (ntr, nte) = (ytr.len(), yte.len());
    let xtr = Matrix::from_vec(ntr, 3, xtr).expect("train dims");
    let xte = Matrix::from_vec(nte, 3, xte).expect("test dims");
    (xtr, Vector::from_vec(ytr), xte, yte)
}

#[test]
fn beat_sklearn_pipeline_encoder() {
    let beat = load_beat();
    assert_eq!(
        beat.ci_gate_name, "beat_sklearn_pipeline_encoder",
        "contract ci_gate_name must match this test binary"
    );

    // --- (1) Exact OneHotEncoder transform parity vs sklearn (NO margin) ---
    // Fit on all 27 category combinations so every column has all 3 levels
    // (same category sets as sklearn fit on the training split → width 9).
    let full = Matrix::from_vec(
        27,
        3,
        (0..27)
            .flat_map(|i| [(i % 3) as f32, (i / 3 % 3) as f32, (i / 9 % 3) as f32])
            .collect(),
    )
    .expect("fit matrix");
    let mut oh = OneHotEncoder::new();
    oh.fit(&full).expect("fit OHE");
    let fixture = Matrix::from_vec(4, 3, vec![0., 0., 0., 1., 2., 1., 2., 1., 2., 0., 1., 0.])
        .expect("fixture");
    let out = oh.transform(&fixture).expect("transform");
    // sklearn OneHotEncoder(handle_unknown='ignore') dense oracle (width 9).
    let oracle = [
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0],
    ];
    assert_eq!(out.shape(), (4, 9), "OHE width must match sklearn (9)");
    for (i, row) in oracle.iter().enumerate() {
        for (j, &e) in row.iter().enumerate() {
            assert!(
                (out.get(i, j) - e as f32).abs() < 1e-6,
                "FALSIFY-BEAT-PIPELINE-OHE-PARITY: apr OneHotEncoder[{i}][{j}] = {} != sklearn {e}",
                out.get(i, j)
            );
        }
    }

    // --- (2) Pipeline(OneHotEncoder → LogisticRegression) composition accuracy ---
    let (xtr, ytr, xte, yte) = categorical_split();
    assert_eq!(
        (ytr.len(), yte.len()),
        (81, 27),
        "deterministic split shape"
    );
    let mut pipe = Pipeline::new(
        vec![Box::new(OneHotEncoder::new())],
        Box::new(LogisticRegression::new().with_max_iter(1000)),
    );
    pipe.fit(&xtr, &ytr).expect("pipeline fit");
    let preds = pipe.predict(&xte).expect("pipeline predict");
    let correct = preds
        .as_slice()
        .iter()
        .zip(&yte)
        .filter(|(p, &t)| p.round() as usize == t)
        .count();
    let acc = correct as f64 / yte.len() as f64;

    eprintln!(
        "BEAT-SKLEARN-PIPELINE-ENCODER: apr Pipeline(OneHotEncoder->LogReg) test_acc = {acc:.4} \
         (scikit-learn make_pipeline {:.4} on same split; threshold {:.4}); OHE transform \
         byte-identical to sklearn",
        beat.baseline_value, beat.beat_threshold
    );

    assert!(
        acc >= beat.beat_threshold,
        "FALSIFY-BEAT-SKLEARN-PIPELINE-ENCODER: apr Pipeline(OneHotEncoder->LogReg) test_acc \
         {acc:.4} < {:.4} (contract apr-sklearn-pipeline-encoder-beat-v1.yaml; sklearn \
         make_pipeline {:.4} on the same categorical split) — the encoder→estimator \
         composition regressed below sklearn",
        beat.beat_threshold,
        beat.baseline_value
    );
}
