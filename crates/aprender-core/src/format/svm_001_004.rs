// SHIP-TWO-001 — `svm-v1` algorithm-level PARTIAL discharge
// for FALSIFY-SVM-001..004.
//
// Contract: `contracts/svm-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Four Linear-SVM gates from Cortes & Vapnik (1995) / ESL §12:
//
// - SVM-001 (binary prediction): every predict(x) ∈ {0, 1}.
// - SVM-002 (deterministic prediction): predict(X) ≡ predict(X).
// - SVM-003 (separable accuracy): well-separated 2D binary clusters
//   give accuracy > 0.9.
// - SVM-004 (fit-predict consistency): predictions are subset of
//   training labels.
//
// In-module reference: `decision_function`, `svm_predict`,
// `hinge_loss`, `linear_svm_subgradient_train`.

/// Minimum accuracy floor on separable data (per FALSIFY-SVM-003).
pub const AC_SVM_003_MIN_SEPARABLE_ACCURACY: f32 = 0.9;

/// Allowed binary labels.
pub const AC_SVM_001_BINARY_LABELS: [usize; 2] = [0, 1];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvmVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference SVM.
// -----------------------------------------------------------------------------

/// Hinge loss for one sample: max(0, 1 - y(w·x + b)) where y ∈ {-1, +1}.
#[must_use]
pub fn hinge_loss(w: &[f32], b: f32, x: &[f32], y_signed: f32) -> f32 {
    let dot: f32 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
    let margin = y_signed * (dot + b);
    (1.0 - margin).max(0.0)
}

/// Decision function f(x) = w·x + b.
#[must_use]
pub fn decision_function(w: &[f32], b: f32, x: &[f32]) -> f32 {
    let dot: f32 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
    dot + b
}

/// Predict {0, 1} from sign(w·x + b). Tie at 0 ⇒ class 1 (sklearn convention).
#[must_use]
pub fn svm_predict(w: &[f32], b: f32, x: &[f32]) -> usize {
    if decision_function(w, b, x) >= 0.0 {
        1
    } else {
        0
    }
}

/// Reference linear-SVM training via primal subgradient descent.
/// Trains on (x, y_binary) where y_binary ∈ {0, 1}; internally maps to
/// y_signed ∈ {-1, +1}.
///
/// Returns `(w, b)`.
#[must_use]
pub fn linear_svm_subgradient_train(
    x: &[f32],
    y_binary: &[usize],
    n_samples: usize,
    n_features: usize,
    n_iters: usize,
    learning_rate: f32,
    c: f32,
) -> Option<(Vec<f32>, f32)> {
    if n_samples == 0 || n_features == 0 {
        return None;
    }
    if x.len() != n_samples * n_features || y_binary.len() != n_samples {
        return None;
    }
    if y_binary.iter().any(|&y| y > 1) {
        return None;
    }

    let mut w = vec![0.0_f32; n_features];
    let mut b = 0.0_f32;
    for _ in 0..n_iters {
        for i in 0..n_samples {
            let xi = &x[i * n_features..(i + 1) * n_features];
            let y_signed = if y_binary[i] == 1 { 1.0_f32 } else { -1.0 };
            let dot: f32 = w.iter().zip(xi.iter()).map(|(wj, xj)| wj * xj).sum();
            let margin = y_signed * (dot + b);
            // Subgradient with regularization 0.5||w||^2 (no reg here for simplicity)
            if margin < 1.0 {
                for j in 0..n_features {
                    w[j] += learning_rate * c * y_signed * xi[j];
                }
                b += learning_rate * c * y_signed;
            }
        }
    }
    Some((w, b))
}

// -----------------------------------------------------------------------------
// Verdict 1: SVM-001 — binary prediction.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_binary_prediction(predictions: &[usize]) -> SvmVerdict {
    if predictions.is_empty() {
        return SvmVerdict::Fail;
    }
    for &p in predictions {
        if !AC_SVM_001_BINARY_LABELS.contains(&p) {
            return SvmVerdict::Fail;
        }
    }
    SvmVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 2: SVM-002 — deterministic prediction.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_deterministic_prediction(
    run1: &[usize],
    run2: &[usize],
) -> SvmVerdict {
    if run1.len() != run2.len() {
        return SvmVerdict::Fail;
    }
    if run1.is_empty() {
        return SvmVerdict::Fail;
    }
    if run1 == run2 {
        SvmVerdict::Pass
    } else {
        SvmVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: SVM-003 — separable accuracy ≥ 0.9.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_separable_accuracy(accuracy: f32) -> SvmVerdict {
    if !accuracy.is_finite() {
        return SvmVerdict::Fail;
    }
    if accuracy >= AC_SVM_003_MIN_SEPARABLE_ACCURACY {
        SvmVerdict::Pass
    } else {
        SvmVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: SVM-004 — fit-predict consistency.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_fit_predict_consistency(
    predictions: &[usize],
    training_labels: &[usize],
) -> SvmVerdict {
    if training_labels.is_empty() {
        return if predictions.is_empty() {
            SvmVerdict::Pass
        } else {
            SvmVerdict::Fail
        };
    }
    let label_set: std::collections::HashSet<usize> =
        training_labels.iter().copied().collect();
    for &p in predictions {
        if !label_set.contains(&p) {
            return SvmVerdict::Fail;
        }
    }
    SvmVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_min_separable_accuracy_09() {
        assert_eq!(AC_SVM_003_MIN_SEPARABLE_ACCURACY, 0.9);
    }

    #[test]
    fn provenance_binary_labels_0_1() {
        assert_eq!(AC_SVM_001_BINARY_LABELS, [0, 1]);
    }

    // -------------------------------------------------------------------------
    // Section 2: Domain — reference functions.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_hinge_loss_zero_at_correct_with_margin() {
        // y=+1, margin = 2 ≥ 1 ⇒ L = 0.
        let w = vec![1.0_f32, 0.0];
        let x = vec![2.0_f32, 0.0];
        let l = hinge_loss(&w, 0.0, &x, 1.0);
        assert_eq!(l, 0.0);
    }

    #[test]
    fn domain_hinge_loss_positive_at_violation() {
        // y=+1, margin = 0.5 < 1 ⇒ L = 0.5.
        let w = vec![0.5_f32, 0.0];
        let x = vec![1.0_f32, 0.0];
        let l = hinge_loss(&w, 0.0, &x, 1.0);
        assert!((l - 0.5).abs() < 1e-6);
    }

    #[test]
    fn domain_hinge_loss_nonneg() {
        // Sweep many (w, b, x, y) combos.
        for &w0 in &[-1.0_f32, 0.0, 1.0] {
            for &b in &[-2.0_f32, 0.0, 2.0] {
                for &x0 in &[-3.0_f32, 0.0, 3.0] {
                    for &y in &[-1.0_f32, 1.0] {
                        let w = vec![w0];
                        let x = vec![x0];
                        let l = hinge_loss(&w, b, &x, y);
                        assert!(l >= 0.0, "L < 0 at w={w0} b={b} x={x0} y={y}");
                    }
                }
            }
        }
    }

    #[test]
    fn domain_decision_function_basic() {
        let w = vec![1.0_f32, -1.0];
        let b = 2.0;
        let x = vec![3.0_f32, 1.0];
        // f = 1*3 + (-1)*1 + 2 = 4.
        assert!((decision_function(&w, b, &x) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn domain_predict_threshold_at_zero() {
        let w = vec![1.0_f32];
        // f(x=2)=2 ⇒ 1; f(x=-2)=-2 ⇒ 0; f(x=0)=0 ⇒ 1 (sklearn ≥ tie).
        assert_eq!(svm_predict(&w, 0.0, &[2.0]), 1);
        assert_eq!(svm_predict(&w, 0.0, &[-2.0]), 0);
        assert_eq!(svm_predict(&w, 0.0, &[0.0]), 1);
    }

    // -------------------------------------------------------------------------
    // Section 3: SVM-001 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn svm001_pass_all_zeros() {
        let preds = vec![0_usize; 10];
        assert_eq!(
            verdict_from_binary_prediction(&preds),
            SvmVerdict::Pass
        );
    }

    #[test]
    fn svm001_pass_all_ones() {
        let preds = vec![1_usize; 10];
        assert_eq!(
            verdict_from_binary_prediction(&preds),
            SvmVerdict::Pass
        );
    }

    #[test]
    fn svm001_pass_mixed_binary() {
        let preds = vec![0_usize, 1, 0, 1, 1, 0];
        assert_eq!(
            verdict_from_binary_prediction(&preds),
            SvmVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: SVM-001 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn svm001_fail_label_2() {
        // Bug: forgot to map sign() to {0, 1}, returned class 2.
        let preds = vec![0_usize, 1, 2];
        assert_eq!(
            verdict_from_binary_prediction(&preds),
            SvmVerdict::Fail
        );
    }

    #[test]
    fn svm001_fail_label_minus_one() {
        // Returned signed labels ∈ {-1, +1} (cast to usize gives huge value).
        let preds = vec![0_usize, 1, usize::MAX];
        assert_eq!(
            verdict_from_binary_prediction(&preds),
            SvmVerdict::Fail
        );
    }

    #[test]
    fn svm001_fail_empty() {
        let preds: Vec<usize> = vec![];
        assert_eq!(
            verdict_from_binary_prediction(&preds),
            SvmVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: SVM-002 — determinism.
    // -------------------------------------------------------------------------
    #[test]
    fn svm002_pass_identical_runs() {
        let r1 = vec![0_usize, 1, 1, 0, 1];
        let r2 = vec![0_usize, 1, 1, 0, 1];
        assert_eq!(
            verdict_from_deterministic_prediction(&r1, &r2),
            SvmVerdict::Pass
        );
    }

    #[test]
    fn svm002_fail_one_off() {
        let r1 = vec![0_usize, 1, 1];
        let r2 = vec![0_usize, 1, 0]; // last differs
        assert_eq!(
            verdict_from_deterministic_prediction(&r1, &r2),
            SvmVerdict::Fail
        );
    }

    #[test]
    fn svm002_fail_length_mismatch() {
        let r1 = vec![0_usize, 1];
        let r2 = vec![0_usize];
        assert_eq!(
            verdict_from_deterministic_prediction(&r1, &r2),
            SvmVerdict::Fail
        );
    }

    #[test]
    fn svm002_fail_empty() {
        let v: Vec<usize> = vec![];
        assert_eq!(
            verdict_from_deterministic_prediction(&v, &v),
            SvmVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: SVM-003 — separable accuracy.
    // -------------------------------------------------------------------------
    #[test]
    fn svm003_pass_perfect() {
        assert_eq!(
            verdict_from_separable_accuracy(1.0),
            SvmVerdict::Pass
        );
    }

    #[test]
    fn svm003_pass_at_threshold() {
        assert_eq!(
            verdict_from_separable_accuracy(0.9),
            SvmVerdict::Pass
        );
    }

    #[test]
    fn svm003_fail_below_threshold() {
        assert_eq!(
            verdict_from_separable_accuracy(0.89),
            SvmVerdict::Fail
        );
    }

    #[test]
    fn svm003_fail_random_chance() {
        assert_eq!(
            verdict_from_separable_accuracy(0.5),
            SvmVerdict::Fail
        );
    }

    #[test]
    fn svm003_fail_nan() {
        assert_eq!(
            verdict_from_separable_accuracy(f32::NAN),
            SvmVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: SVM-004 — fit-predict consistency.
    // -------------------------------------------------------------------------
    #[test]
    fn svm004_pass_predictions_subset() {
        let preds = vec![0_usize, 1, 0, 1];
        let labels = vec![0_usize, 1];
        assert_eq!(
            verdict_from_fit_predict_consistency(&preds, &labels),
            SvmVerdict::Pass
        );
    }

    #[test]
    fn svm004_pass_empty_predictions() {
        let preds: Vec<usize> = vec![];
        let labels = vec![0_usize, 1];
        assert_eq!(
            verdict_from_fit_predict_consistency(&preds, &labels),
            SvmVerdict::Pass
        );
    }

    #[test]
    fn svm004_fail_unseen_label_in_predictions() {
        let preds = vec![0_usize, 1, 5];
        let labels = vec![0_usize, 1];
        assert_eq!(
            verdict_from_fit_predict_consistency(&preds, &labels),
            SvmVerdict::Fail
        );
    }

    #[test]
    fn svm004_fail_no_training_labels() {
        let preds = vec![0_usize];
        let labels: Vec<usize> = vec![];
        assert_eq!(
            verdict_from_fit_predict_consistency(&preds, &labels),
            SvmVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 8: Domain — fit/predict end-to-end on toy data.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_fit_predict_separable_2d() {
        // 4 points: (-2, 0), (-1, 0) → class 0; (1, 0), (2, 0) → class 1.
        let x = vec![-2.0_f32, 0.0, -1.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let y = vec![0_usize, 0, 1, 1];
        let (w, b) =
            linear_svm_subgradient_train(&x, &y, 4, 2, 100, 0.01, 1.0).unwrap();

        // Predict on training points; should classify all correctly.
        let mut preds = Vec::new();
        for i in 0..4 {
            preds.push(svm_predict(&w, b, &x[i * 2..i * 2 + 2]));
        }
        let correct = preds.iter().zip(y.iter()).filter(|(p, l)| p == l).count();
        let acc = correct as f32 / y.len() as f32;
        assert_eq!(
            verdict_from_separable_accuracy(acc),
            SvmVerdict::Pass,
            "preds={preds:?} expected={y:?} acc={acc}"
        );
    }

    #[test]
    fn domain_fit_rejects_invalid_input() {
        // Wrong x size.
        let x = vec![1.0_f32];
        let y = vec![0_usize, 1];
        assert!(
            linear_svm_subgradient_train(&x, &y, 2, 1, 10, 0.01, 1.0).is_none()
        );
    }

    #[test]
    fn domain_fit_rejects_label_above_one() {
        let x = vec![1.0_f32, 2.0];
        let y = vec![0_usize, 5]; // 5 > 1
        assert!(
            linear_svm_subgradient_train(&x, &y, 2, 1, 10, 0.01, 1.0).is_none()
        );
    }

    // -------------------------------------------------------------------------
    // Section 9: Sweep — accuracy threshold band.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_separable_accuracy_around_threshold() {
        let test_cases = [
            (0.0_f32, SvmVerdict::Fail),
            (0.5, SvmVerdict::Fail),
            (0.89, SvmVerdict::Fail),
            (0.9, SvmVerdict::Pass),
            (0.95, SvmVerdict::Pass),
            (1.0, SvmVerdict::Pass),
        ];
        for (acc, expected) in test_cases {
            let v = verdict_from_separable_accuracy(acc);
            assert_eq!(v, expected, "acc={acc}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 10: Realistic — contract regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_forgot_to_map_decision_to_binary() {
        // SVM-001 if_fails: "Prediction not mapped to {0, 1}".
        // sign() returned {-1, +1} cast to usize.
        let preds = vec![0_usize, 1, usize::MAX, 1];
        assert_eq!(
            verdict_from_binary_prediction(&preds),
            SvmVerdict::Fail
        );
    }

    #[test]
    fn realistic_non_deterministic_lr_caught() {
        // SVM-002 if_fails: "Non-deterministic learning rate or state".
        let r1 = vec![0_usize, 1, 0, 1];
        let r2 = vec![1_usize, 1, 0, 1]; // first flipped
        assert_eq!(
            verdict_from_deterministic_prediction(&r1, &r2),
            SvmVerdict::Fail
        );
    }

    #[test]
    fn realistic_convergence_failure_caught() {
        // SVM-003 if_fails: "Convergence failure or wrong sign convention"
        // ⇒ accuracy near random.
        assert_eq!(
            verdict_from_separable_accuracy(0.55),
            SvmVerdict::Fail
        );
    }

    #[test]
    fn realistic_label_mapping_error_caught() {
        // SVM-004 if_fails: "Label mapping error".
        let preds = vec![3_usize, 7, 9]; // none in {0, 1}
        let labels = vec![0_usize, 1];
        assert_eq!(
            verdict_from_fit_predict_consistency(&preds, &labels),
            SvmVerdict::Fail
        );
    }

    #[test]
    fn realistic_full_pipeline_passes_all_4_gates() {
        // Train on separable 1D data.
        let x = vec![-3.0_f32, -2.0, -1.0, 1.0, 2.0, 3.0];
        let y = vec![0_usize, 0, 0, 1, 1, 1];
        let (w, b) =
            linear_svm_subgradient_train(&x, &y, 6, 1, 100, 0.01, 1.0).unwrap();

        let mut preds = Vec::new();
        for i in 0..6 {
            preds.push(svm_predict(&w, b, &x[i..i + 1]));
        }

        // Gate 1: binary.
        assert_eq!(verdict_from_binary_prediction(&preds), SvmVerdict::Pass);

        // Gate 2: deterministic.
        let preds_again: Vec<usize> = (0..6).map(|i| svm_predict(&w, b, &x[i..i + 1])).collect();
        assert_eq!(
            verdict_from_deterministic_prediction(&preds, &preds_again),
            SvmVerdict::Pass
        );

        // Gate 3: separable accuracy ≥ 0.9.
        let correct = preds.iter().zip(y.iter()).filter(|(p, l)| p == l).count();
        let acc = correct as f32 / y.len() as f32;
        assert_eq!(verdict_from_separable_accuracy(acc), SvmVerdict::Pass);

        // Gate 4: fit-predict consistency.
        assert_eq!(
            verdict_from_fit_predict_consistency(&preds, &y),
            SvmVerdict::Pass
        );
    }
}
