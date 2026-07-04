//! Pillar-1 (scikit-learn) CORRECTNESS beat: apr's score-based classification
//! metrics are NUMERICALLY EQUAL to scikit-learn on the same inputs — a
//! falsifiable, per-PR CI-gated benchmark. Covers the full probabilistic-metric
//! surface: `roc_auc_score`, `log_loss`, `average_precision_score`, and (new in
//! PMAT-730) the array-returning `roc_curve` and `precision_recall_curve`.
//!
//! Where the accuracy beats (`beat_sklearn_iris`, `beat_sklearn_gaussiannb_accuracy`)
//! prove apr's CLASSIFIERS match sklearn, this proves the METRIC layer those
//! classifiers are scored with matches sklearn — element-wise, including sklearn's
//! `+inf` leading ROC threshold and the terminal `(precision=1, recall=0)` PR
//! sentinel. Metric parity is exact (no solver/RNG variance), so a single pinned
//! fixture is a complete falsifier. Oracle pinned 2026-07-04 via
//! `uv run --with scikit-learn` (sklearn 1.9.0).

use aprender::metrics::{
    average_precision_score, log_loss, precision_recall_curve, roc_auc_score, roc_curve,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct BeatContract {
    beat: BeatParams,
}

#[derive(Deserialize)]
struct BeatParams {
    /// apr's max element-wise deviation from sklearn must be `<= beat_threshold`.
    beat_threshold: f64,
    /// The CI gate this contract is enforced by — must match this test binary.
    ci_gate_name: String,
}

fn load_beat() -> BeatParams {
    const YAML: &str = include_str!("../../../contracts/apr-sklearn-metrics-parity-beat-v1.yaml");
    let contract: BeatContract = serde_yaml::from_str(YAML)
        .expect("parse contracts/apr-sklearn-metrics-parity-beat-v1.yaml");
    contract.beat
}

// Pinned 8-sample binary fixture (identical to the probabilistic.rs unit tests).
const YT: [usize; 8] = [0, 0, 1, 1, 1, 0, 1, 0];
const YS: [f32; 8] = [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.55];

/// Max |apr - sklearn| over two equal-length vectors, treating matching infinities
/// as zero deviation (sklearn's leading ROC threshold is +inf).
fn max_dev(got: &[f32], oracle: &[f32]) -> f64 {
    assert_eq!(
        got.len(),
        oracle.len(),
        "length mismatch vs sklearn: apr={got:?} sklearn={oracle:?}"
    );
    got.iter()
        .zip(oracle)
        .map(|(&g, &o)| {
            if o.is_infinite() {
                assert!(
                    g.is_infinite() && g.signum() == o.signum(),
                    "expected {o}, got {g}"
                );
                0.0
            } else {
                f64::from((g - o).abs())
            }
        })
        .fold(0.0_f64, f64::max)
}

#[test]
fn beat_sklearn_metrics_parity() {
    let beat = load_beat();
    assert_eq!(
        beat.ci_gate_name, "beat_sklearn_metrics_parity",
        "contract ci_gate_name must match this test binary"
    );

    // --- Scalar metrics (sklearn 1.9.0 oracles) ---
    let mut worst = 0.0_f64;
    worst = worst.max(f64::from((roc_auc_score(&YT, &YS) - 0.875).abs()));
    worst = worst.max(f64::from((log_loss(&YT, &YS) - 0.421_605).abs()));
    worst = worst.max(f64::from(
        (average_precision_score(&YT, &YS) - 0.916_667).abs(),
    ));

    // --- roc_curve (sklearn drop_intermediate=True) ---
    let (fpr, tpr, rthr) = roc_curve(&YT, &YS);
    worst = worst.max(max_dev(&fpr, &[0.0, 0.0, 0.0, 0.5, 0.5, 1.0]));
    worst = worst.max(max_dev(&tpr, &[0.0, 0.25, 0.75, 0.75, 1.0, 1.0]));
    worst = worst.max(max_dev(&rthr, &[f32::INFINITY, 0.9, 0.7, 0.4, 0.35, 0.1]));

    // --- precision_recall_curve (terminal (1,0) sentinel) ---
    let (prec, rec, pthr) = precision_recall_curve(&YT, &YS);
    worst = worst.max(max_dev(
        &prec,
        &[0.5, 0.571_429, 0.666_667, 0.6, 0.75, 1.0, 1.0, 1.0, 1.0],
    ));
    worst = worst.max(max_dev(
        &rec,
        &[1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.5, 0.25, 0.0],
    ));
    worst = worst.max(max_dev(&pthr, &[0.1, 0.2, 0.35, 0.4, 0.55, 0.7, 0.8, 0.9]));

    eprintln!(
        "BEAT-SKLEARN-METRICS-PARITY: apr max metric deviation vs scikit-learn \
         1.9.0 = {worst:.2e} (contract threshold {:.1e})",
        beat.beat_threshold
    );

    assert!(
        worst <= beat.beat_threshold,
        "FALSIFY-BEAT-SKLEARN-METRICS-PARITY: apr metric deviation {worst:.3e} > {:.1e} \
         (contract apr-sklearn-metrics-parity-beat-v1.yaml; sklearn 1.9.0 on the pinned \
         8-sample fixture) — a probabilistic metric regressed away from scikit-learn",
        beat.beat_threshold
    );
}
