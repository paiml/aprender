// =========================================================================
// FALSIFY-GLM: glm-v1.yaml contract (aprender GLM)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-GLM-* tests
//   Why 2: GLM tests only in tests/contracts/, not near implementation
//   Why 3: no mapping from glm-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: GLM was "obviously correct" (IRLS + link functions)
//
// References:
//   - provable-contracts/contracts/glm-v1.yaml
//   - McCullagh & Nelder (1989) "Generalized Linear Models"
// =========================================================================

use super::*;
use crate::primitives::{Matrix, Vector};

/// Helper: create Poisson-appropriate data where log(y) ≈ linear in x
fn poisson_data() -> (Matrix<f32>, Vector<f32>) {
    // y ≈ exp(0.5 + 0.3*x) — clean Poisson data
    let x = Matrix::from_vec(8, 1, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).expect("valid");
    let y = Vector::from_slice(&[1.6, 2.1, 2.8, 3.7, 5.0, 6.6, 8.8, 11.6]);
    (x, y)
}

/// FALSIFY-GLM-001: Predictions are finite
#[test]
fn falsify_glm_001_finite_predictions() {
    let (x, y) = poisson_data();

    let mut glm = GLM::new(Family::Poisson).with_max_iter(2000);
    glm.fit(&x, &y).expect("fit");

    let preds = glm.predict(&x).expect("predict");
    for i in 0..preds.len() {
        assert!(
            preds[i].is_finite(),
            "FALSIFIED GLM-001: prediction[{i}] = {} is not finite",
            preds[i]
        );
    }
}

/// FALSIFY-GLM-002: Prediction count matches input count
#[test]
fn falsify_glm_002_prediction_count() {
    let (x, y) = poisson_data();

    let mut glm = GLM::new(Family::Poisson).with_max_iter(2000);
    glm.fit(&x, &y).expect("fit");

    let preds = glm.predict(&x).expect("predict");
    assert_eq!(
        preds.len(),
        8,
        "FALSIFIED GLM-002: {} predictions for 8 inputs",
        preds.len()
    );
}

/// FALSIFY-GLM-003: Poisson predictions are non-negative (link = log → μ = exp(η) > 0)
#[test]
fn falsify_glm_003_poisson_non_negative() {
    let (x, y) = poisson_data();

    let mut glm = GLM::new(Family::Poisson).with_max_iter(2000);
    glm.fit(&x, &y).expect("fit");

    let preds = glm.predict(&x).expect("predict");
    for i in 0..preds.len() {
        assert!(
            preds[i] >= 0.0,
            "FALSIFIED GLM-003: Poisson prediction[{i}] = {} < 0",
            preds[i]
        );
    }
}

/// FALSIFY-GLM-004: Deterministic predictions
#[test]
fn falsify_glm_004_deterministic() {
    let (x, y) = poisson_data();

    let mut glm = GLM::new(Family::Poisson).with_max_iter(2000);
    glm.fit(&x, &y).expect("fit");

    let p1 = glm.predict(&x).expect("predict 1");
    let p2 = glm.predict(&x).expect("predict 2");
    for i in 0..p1.len() {
        assert_eq!(
            p1[i], p2[i],
            "FALSIFIED GLM-004: prediction differs at index {i}"
        );
    }
}

mod glm_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-GLM-002-prop: Prediction count matches for different data sizes
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn falsify_glm_002_prop_prediction_count(
            extra in 0..=5usize,
        ) {
            let n = 8 + extra;
            let x_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let x = Matrix::from_vec(n, 1, x_data).expect("valid");
            let y_data: Vec<f32> = (0..n).map(|i| (0.5 + 0.3 * i as f32).exp()).collect();
            let y = Vector::from_vec(y_data);

            let mut glm = GLM::new(Family::Poisson).with_max_iter(5000);
            prop_assume!(glm.fit(&x, &y).is_ok());

            let preds = glm.predict(&x).expect("predict");
            prop_assert_eq!(
                preds.len(),
                n,
                "FALSIFIED GLM-002-prop: {} predictions for {} inputs",
                preds.len(), n
            );
        }
    }

    /// FALSIFY-GLM-003-prop: Poisson predictions non-negative for different sizes
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn falsify_glm_003_prop_poisson_nonneg(
            extra in 0..=5usize,
        ) {
            let n = 8 + extra;
            let x_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let x = Matrix::from_vec(n, 1, x_data).expect("valid");
            let y_data: Vec<f32> = (0..n).map(|i| (0.5 + 0.3 * i as f32).exp()).collect();
            let y = Vector::from_vec(y_data);

            let mut glm = GLM::new(Family::Poisson).with_max_iter(5000);
            prop_assume!(glm.fit(&x, &y).is_ok());

            let preds = glm.predict(&x).expect("predict");
            for i in 0..preds.len() {
                prop_assert!(
                    preds[i] >= 0.0,
                    "FALSIFIED GLM-003-prop: Poisson prediction[{}]={} < 0",
                    i, preds[i]
                );
            }
        }
    }
}
