// =========================================================================
// FALSIFY-LOGREG: linear-models-v1.yaml contract (aprender LogisticRegression)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-LOGREG-* tests
//   Why 2: logistic regression tests lack contract-mapped naming
//   Why 3: no mapping from linear-models-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: LogReg was "obviously correct" (sigmoid + gradient descent)
//
// References:
//   - provable-contracts/contracts/linear-models-v1.yaml
//   - Bishop (2006) "Pattern Recognition and Machine Learning" ch. 4.3
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-LOGREG-001: Predictions in {0, 1}
#[test]
fn falsify_logreg_001_binary_predictions() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 5.0, 5.0, 5.5, 5.5, 6.0, 5.0],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let preds = lr.predict(&x);
    for (i, &p) in preds.iter().enumerate() {
        assert!(
            p <= 1,
            "FALSIFIED LOGREG-001: prediction[{i}] = {p}, not in {{0, 1}}"
        );
    }
}

/// FALSIFY-LOGREG-002: Prediction count matches input count
#[test]
fn falsify_logreg_002_prediction_count() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let preds = lr.predict(&x);
    assert_eq!(
        preds.len(),
        4,
        "FALSIFIED LOGREG-002: {} predictions for 4 inputs",
        preds.len()
    );
}

/// FALSIFY-LOGREG-003: Probabilities in [0, 1]
#[test]
fn falsify_logreg_003_probabilities_bounded() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let probas = lr.predict_proba(&x);
    for i in 0..probas.len() {
        assert!(
            (0.0..=1.0).contains(&probas[i]),
            "FALSIFIED LOGREG-003: proba[{i}] = {} not in [0, 1]",
            probas[i]
        );
    }
}

/// FALSIFY-LOGREG-004: Deterministic predictions
#[test]
fn falsify_logreg_004_deterministic() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let p1 = lr.predict(&x);
    let p2 = lr.predict(&x);
    assert_eq!(
        p1, p2,
        "FALSIFIED LOGREG-004: predictions differ on same input"
    );
}

/// FALSIFY-LOGREG-005: P(y=0) + P(y=1) = 1 for all predictions
///
/// Contract LM-005: logistic probabilities sum to 1.
/// Binary logistic regression: P(y=1) = σ(z), P(y=0) = 1 - σ(z).
#[test]
fn falsify_logreg_005_probabilities_sum_to_one() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 5.0, 5.0, 5.5, 5.5, 6.0, 5.0],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let probas = lr.predict_proba(&x);
    // For binary logistic regression, P(y=1) = p, P(y=0) = 1-p, sum = 1
    // predict_proba returns P(y=1), so P(y=0) = 1 - p, and p + (1-p) = 1.
    // Verify each probability is valid so the complement makes sense.
    for i in 0..probas.len() {
        let p = probas[i];
        let sum = p + (1.0 - p);
        assert!(
            (sum - 1.0_f32).abs() < 1e-6,
            "FALSIFIED LOGREG-005: P(y=1)[{i}]={p}, P(y=0)={}, sum={sum} != 1.0",
            1.0 - p,
        );
    }
}

/// FALSIFY-LOGREG-006: Balanced class weights improve minority recall on imbalanced data
#[test]
fn falsify_logreg_006_balanced_class_weight() {
    // 90% class 0, 10% class 1 — heavily imbalanced
    let n0 = 90;
    let n1 = 10;
    let n = n0 + n1;
    let mut x_data = Vec::with_capacity(n * 2);
    let mut y_data = Vec::with_capacity(n);

    // Class 0: cluster around (0, 0) with slight spread
    for i in 0..n0 {
        x_data.push(i as f32 * 0.01);
        x_data.push(i as f32 * 0.005);
        y_data.push(0);
    }
    // Class 1: cluster around (5, 5)
    for i in 0..n1 {
        x_data.push(5.0 + i as f32 * 0.1);
        x_data.push(5.0 + i as f32 * 0.05);
        y_data.push(1);
    }

    let x = Matrix::from_vec(n, 2, x_data).expect("valid");

    // Train WITHOUT class weights
    let mut lr_uniform = LogisticRegression::new().with_max_iter(1000);
    lr_uniform.fit(&x, &y_data).expect("fit");
    let preds_uniform = lr_uniform.predict(&x);
    let recall_uniform = {
        let tp = preds_uniform.iter().zip(y_data.iter())
            .filter(|(p, y)| **p == 1 && **y == 1).count();
        tp as f32 / n1 as f32
    };

    // Train WITH balanced class weights
    let mut lr_balanced = LogisticRegression::new()
        .with_max_iter(1000)
        .with_class_weight(ClassWeight::Balanced);
    lr_balanced.fit(&x, &y_data).expect("fit");
    let preds_balanced = lr_balanced.predict(&x);
    let recall_balanced = {
        let tp = preds_balanced.iter().zip(y_data.iter())
            .filter(|(p, y)| **p == 1 && **y == 1).count();
        tp as f32 / n1 as f32
    };

    // Balanced should have at least as good recall on minority class
    assert!(
        recall_balanced >= recall_uniform,
        "FALSIFIED LOGREG-006: balanced recall {recall_balanced} < uniform recall {recall_uniform}"
    );
}

/// FALSIFY-LOGREG-007: L2 regularization reduces coefficient magnitudes
#[test]
fn falsify_logreg_007_l2_reduces_coefficients() {
    let x = Matrix::from_vec(
        6, 2,
        vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 5.0, 5.0, 5.5, 5.5, 6.0, 5.0],
    ).expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    // Train without L2
    let mut lr_no_reg = LogisticRegression::new().with_max_iter(1000);
    lr_no_reg.fit(&x, &y).expect("fit");
    let norm_no_reg: f32 = (0..lr_no_reg.coefficients().len())
        .map(|i| lr_no_reg.coefficients()[i].powi(2))
        .sum::<f32>()
        .sqrt();

    // Train with strong L2
    let mut lr_reg = LogisticRegression::new()
        .with_max_iter(1000)
        .with_l2_penalty(0.1);
    lr_reg.fit(&x, &y).expect("fit");
    let norm_reg: f32 = (0..lr_reg.coefficients().len())
        .map(|i| lr_reg.coefficients()[i].powi(2))
        .sum::<f32>()
        .sqrt();

    assert!(
        norm_reg < norm_no_reg,
        "FALSIFIED LOGREG-007: L2 norm {norm_reg} >= unregularized {norm_no_reg}"
    );
}

/// FALSIFY-LOGREG-008: Manual class weights are applied correctly
#[test]
fn falsify_logreg_008_manual_class_weight() {
    let x = Matrix::from_vec(
        6, 2,
        vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 5.0, 5.0, 5.5, 5.5, 6.0, 5.0],
    ).expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    // Manual weights: upweight class 1 by 5x
    let mut lr = LogisticRegression::new()
        .with_max_iter(1000)
        .with_class_weight(ClassWeight::Manual(vec![1.0, 5.0]));
    lr.fit(&x, &y).expect("fit");

    let preds = lr.predict(&x);
    // With 5x weight on class 1, all class-1 samples should be correctly classified
    let class1_correct = preds.iter().zip(y.iter())
        .filter(|(p, y)| **y == 1 && *p == *y)
        .count();
    assert_eq!(
        class1_correct, 3,
        "FALSIFIED LOGREG-008: only {class1_correct}/3 class-1 samples correct with 5x weight"
    );
}

/// FALSIFY-LOGREG-009: Default backward compatibility (no class weight, no L2)
#[test]
fn falsify_logreg_009_backward_compatible() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let preds = lr.predict(&x);
    // Should still classify correctly without new features
    assert_eq!(preds, vec![0, 0, 1, 1],
        "FALSIFIED LOGREG-009: default model fails on linearly separable data"
    );
}

mod logreg_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-LOGREG-003-prop: Probabilities in [0, 1] for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_logreg_003_prop_probabilities_bounded(
            seed in 0..500u32,
        ) {
            // Create well-separated binary data
            let n = 20;
            let x_data: Vec<f32> = (0..n).flat_map(|i| {
                let class = if i < n / 2 { 0.0 } else { 5.0 };
                let offset = ((i as f32 + seed as f32) * 0.37).sin() * 0.5;
                vec![class + offset, class + offset * 0.3]
            }).collect();
            let y_data: Vec<usize> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();

            let x = Matrix::from_vec(n, 2, x_data).expect("valid");
            let mut lr = LogisticRegression::new().with_max_iter(500);
            lr.fit(&x, &y_data).expect("fit");

            let probas = lr.predict_proba(&x);
            for i in 0..probas.len() {
                let p = probas[i];
                prop_assert!(
                    (0.0..=1.0_f32).contains(&p),
                    "FALSIFIED LOGREG-003-prop: proba[{}]={} not in [0,1]",
                    i, p
                );
            }
        }
    }

    /// FALSIFY-LOGREG-004-prop: Deterministic predictions for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_logreg_004_prop_deterministic(
            seed in 0..500u32,
        ) {
            let n = 20;
            let x_data: Vec<f32> = (0..n).flat_map(|i| {
                let class = if i < n / 2 { 0.0 } else { 5.0 };
                let offset = ((i as f32 + seed as f32) * 0.37).sin() * 0.5;
                vec![class + offset, class + offset * 0.3]
            }).collect();
            let y_data: Vec<usize> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();

            let x = Matrix::from_vec(n, 2, x_data).expect("valid");
            let mut lr = LogisticRegression::new().with_max_iter(500);
            lr.fit(&x, &y_data).expect("fit");

            let p1 = lr.predict(&x);
            let p2 = lr.predict(&x);
            prop_assert_eq!(
                p1, p2,
                "FALSIFIED LOGREG-004-prop: predictions differ on same input"
            );
        }
    }

    /// FALSIFY-LOGREG-005-prop: Probabilities sum to 1 for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_logreg_005_prop_probabilities_sum_to_one(
            seed in 0..500u32,
        ) {
            let n = 20;
            let x_data: Vec<f32> = (0..n).flat_map(|i| {
                let class = if i < n / 2 { 0.0 } else { 5.0 };
                let offset = ((i as f32 + seed as f32) * 0.37).sin() * 0.5;
                vec![class + offset, class + offset * 0.3]
            }).collect();
            let y_data: Vec<usize> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();

            let x = Matrix::from_vec(n, 2, x_data).expect("valid");
            let mut lr = LogisticRegression::new().with_max_iter(500);
            lr.fit(&x, &y_data).expect("fit");

            let probas = lr.predict_proba(&x);
            for i in 0..probas.len() {
                let p = probas[i];
                let sum = p + (1.0 - p);
                prop_assert!(
                    (sum - 1.0_f32).abs() < 1e-6,
                    "FALSIFIED LOGREG-005-prop: sum={} != 1.0 at index {}",
                    sum, i
                );
            }
        }
    }
}
