// =========================================================================
// FALSIFY-MC: metrics-classification-v1.yaml contract (aprender classification metrics)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-MC-* tests for classification metrics
//   Why 2: metric tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from metrics-classification-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Metrics were "obviously correct" (TP/FP/FN counting)
//
// References:
//   - provable-contracts/contracts/metrics-classification-v1.yaml
//   - Sokolova & Lapalme (2009) "A systematic analysis of performance measures"
// =========================================================================

use super::*;

/// FALSIFY-MC-001: Accuracy ∈ [0, 1]
#[test]
fn falsify_mc_001_accuracy_bounded() {
    let y_true = vec![0, 1, 2, 0, 1, 2];
    let y_pred = vec![0, 2, 1, 0, 0, 1];

    let acc = accuracy(&y_pred, &y_true);
    assert!(
        (0.0..=1.0).contains(&acc),
        "FALSIFIED MC-001: accuracy={acc} not in [0, 1]"
    );
}

/// FALSIFY-MC-002: Perfect predictions → accuracy = 1.0
#[test]
fn falsify_mc_002_perfect_accuracy() {
    let y = vec![0, 1, 2, 0, 1, 2];
    let acc = accuracy(&y, &y);
    assert!(
        (acc - 1.0).abs() < 1e-6,
        "FALSIFIED MC-002: accuracy={acc} for perfect predictions, expected 1.0"
    );
}

/// FALSIFY-MC-003: Precision ∈ [0, 1]
#[test]
fn falsify_mc_003_precision_bounded() {
    let y_true = vec![0, 1, 2, 0, 1, 2];
    let y_pred = vec![0, 2, 1, 0, 0, 1];

    let prec = precision(&y_pred, &y_true, Average::Macro);
    assert!(
        (0.0..=1.0).contains(&prec),
        "FALSIFIED MC-003: precision={prec} not in [0, 1]"
    );
}

/// FALSIFY-MC-004: F1 = 1.0 for perfect predictions
#[test]
fn falsify_mc_004_perfect_f1() {
    let y = vec![0, 1, 2, 0, 1, 2];
    let f1 = f1_score(&y, &y, Average::Macro);
    assert!(
        (f1 - 1.0).abs() < 1e-6,
        "FALSIFIED MC-004: F1={f1} for perfect predictions, expected 1.0"
    );
}

/// FALSIFY-MC-005: Recall ∈ [0, 1]
#[test]
fn falsify_mc_005_recall_bounded() {
    let y_true = vec![0, 1, 2, 0, 1, 2];
    let y_pred = vec![0, 2, 1, 0, 0, 1];

    let rec = recall(&y_pred, &y_true, Average::Macro);
    assert!(
        (0.0..=1.0).contains(&rec),
        "FALSIFIED MC-005: recall={rec} not in [0, 1]"
    );
}

mod mc_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-MC-001-prop: Accuracy in [0, 1] for random labels
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn falsify_mc_001_prop_accuracy_bounded(
            seed in 0..1000u32,
            n in 5..=30usize,
        ) {
            let y_true: Vec<usize> = (0..n)
                .map(|i| ((i as u32 + seed) % 3) as usize)
                .collect();
            let y_pred: Vec<usize> = (0..n)
                .map(|i| ((i as u32 + seed + 1) % 3) as usize)
                .collect();

            let acc = accuracy(&y_pred, &y_true);
            prop_assert!(
                (0.0..=1.0).contains(&acc),
                "FALSIFIED MC-001-prop: accuracy={} not in [0,1]",
                acc
            );
        }
    }

    /// FALSIFY-MC-002-prop: Perfect predictions → accuracy = 1.0
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_mc_002_prop_perfect_accuracy(
            n in 5..=30usize,
            seed in 0..500u32,
        ) {
            let y: Vec<usize> = (0..n)
                .map(|i| ((i as u32 + seed) % 3) as usize)
                .collect();
            let acc = accuracy(&y, &y);
            prop_assert!(
                (acc - 1.0).abs() < 1e-6,
                "FALSIFIED MC-002-prop: accuracy={} for perfect predictions",
                acc
            );
        }
    }
}
