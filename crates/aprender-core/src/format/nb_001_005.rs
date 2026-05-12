// SHIP-TWO-001 — `naive-bayes-v1` algorithm-level PARTIAL discharge
// for FALSIFY-NB-001..005.
//
// Contract: `contracts/naive-bayes-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Gaussian Naive Bayes — five gates:
//
// - NB-001 (prior sums to 1): Σ_k P(C_k) = 1 after fit.
// - NB-002 (prior bounded): each P(C_k) ∈ (0, 1) for observed classes.
// - NB-003 (prediction in classes): predict(x) ∈ training_classes.
// - NB-004 (prediction deterministic): predict(x) = predict(x).
// - NB-005 (separable accuracy): accuracy > 0.9 on well-separated
//   Gaussian clusters.
//
// All five are PURE algorithm-level — no kernel selection, no SIMD path
// — provable via in-module reference Gaussian-NB classifier.

/// Floor of valid probability bound — strict.
pub const AC_NB_002_PRIOR_LOWER_EXCLUSIVE: f32 = 0.0;

/// Ceiling of valid probability bound — strict.
pub const AC_NB_002_PRIOR_UPPER_EXCLUSIVE: f32 = 1.0;

/// Tolerance on Σ priors == 1 (allow f32 round-off).
pub const AC_NB_001_PRIOR_SUM_EPS: f32 = 1e-5;

/// Minimum accuracy for NB-005 separable-cluster gate.
pub const AC_NB_005_MIN_ACCURACY: f32 = 0.9;

/// Lower bound for variance — Gaussian PDF is undefined at σ²=0.
pub const AC_NB_VARIANCE_FLOOR: f32 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NbVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference Gaussian-NB.
// -----------------------------------------------------------------------------

/// Fit a Gaussian Naive Bayes classifier from `(x, y)`.
///
/// `x` is row-major `[n_samples, n_features]`; `y` is `[n_samples]` of
/// class indices in `0..k`. Returns (priors, means, variances) where:
///
/// - `priors[k]`        — P(C_k) = count_k / n
/// - `means[k * d + j]` — μ_jk
/// - `vars[k * d + j]`  — σ²_jk (variance-floored at AC_NB_VARIANCE_FLOOR)
#[must_use]
pub fn fit_gnb(
    x: &[f32],
    y: &[usize],
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if n_samples == 0 || n_features == 0 || n_classes == 0 {
        return None;
    }
    if x.len() != n_samples * n_features || y.len() != n_samples {
        return None;
    }
    if y.iter().any(|&c| c >= n_classes) {
        return None;
    }

    let mut counts = vec![0_usize; n_classes];
    for &c in y {
        counts[c] += 1;
    }

    let mut means = vec![0.0_f32; n_classes * n_features];
    for i in 0..n_samples {
        let c = y[i];
        for j in 0..n_features {
            means[c * n_features + j] += x[i * n_features + j];
        }
    }
    for k in 0..n_classes {
        let n_k = counts[k] as f32;
        if n_k > 0.0 {
            for j in 0..n_features {
                means[k * n_features + j] /= n_k;
            }
        }
    }

    let mut vars = vec![0.0_f32; n_classes * n_features];
    for i in 0..n_samples {
        let c = y[i];
        for j in 0..n_features {
            let d = x[i * n_features + j] - means[c * n_features + j];
            vars[c * n_features + j] += d * d;
        }
    }
    for k in 0..n_classes {
        let n_k = counts[k] as f32;
        if n_k > 0.0 {
            for j in 0..n_features {
                let v = vars[k * n_features + j] / n_k;
                vars[k * n_features + j] = v.max(AC_NB_VARIANCE_FLOOR);
            }
        } else {
            for j in 0..n_features {
                vars[k * n_features + j] = 1.0;
            }
        }
    }

    let n = n_samples as f32;
    let priors: Vec<f32> = counts.iter().map(|&c| c as f32 / n).collect();

    Some((priors, means, vars))
}

/// Predict the class index by argmax of log P(C_k) + Σ_j log P(x_j | C_k).
#[must_use]
pub fn predict_gnb(
    x: &[f32],
    priors: &[f32],
    means: &[f32],
    vars: &[f32],
    n_features: usize,
) -> usize {
    let n_classes = priors.len();
    let mut best_k: usize = 0;
    let mut best_lp: f32 = f32::NEG_INFINITY;

    for k in 0..n_classes {
        if priors[k] <= 0.0 {
            // Class unobserved in fit — log(0) is -inf, never argmax.
            continue;
        }
        let mut lp = priors[k].ln();
        for j in 0..n_features {
            let mu = means[k * n_features + j];
            let var = vars[k * n_features + j];
            let dx = x[j] - mu;
            // log Gaussian PDF: -0.5 * ln(2π·σ²) - dx²/(2σ²)
            let two_pi = std::f32::consts::TAU;
            lp += -0.5 * (two_pi * var).ln() - 0.5 * (dx * dx) / var;
        }
        if lp > best_lp {
            best_lp = lp;
            best_k = k;
        }
    }
    best_k
}

// -----------------------------------------------------------------------------
// Verdict 1: NB-001 — Σ_k P(C_k) = 1.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_prior_sum_to_one(priors: &[f32]) -> NbVerdict {
    if priors.is_empty() {
        return NbVerdict::Fail;
    }
    if priors.iter().any(|p| !p.is_finite()) {
        return NbVerdict::Fail;
    }
    let sum: f32 = priors.iter().sum();
    if (sum - 1.0).abs() < AC_NB_001_PRIOR_SUM_EPS {
        NbVerdict::Pass
    } else {
        NbVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: NB-002 — each P(C_k) ∈ (0, 1) for observed classes.
// -----------------------------------------------------------------------------

/// `priors` is P(C_k); `observed_mask[k] = true` iff class k appears in
/// training labels. For unobserved classes the strict lower bound is
/// relaxed (P(C_k) = 0 is permissible — a class never seen).
#[must_use]
pub fn verdict_from_prior_bounded(priors: &[f32], observed_mask: &[bool]) -> NbVerdict {
    if priors.is_empty() || priors.len() != observed_mask.len() {
        return NbVerdict::Fail;
    }
    for (i, &p) in priors.iter().enumerate() {
        if !p.is_finite() {
            return NbVerdict::Fail;
        }
        if observed_mask[i] {
            if p <= AC_NB_002_PRIOR_LOWER_EXCLUSIVE
                || p >= AC_NB_002_PRIOR_UPPER_EXCLUSIVE
            {
                return NbVerdict::Fail;
            }
        } else if !(0.0..=1.0).contains(&p) {
            return NbVerdict::Fail;
        }
    }
    NbVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 3: NB-003 — predict(x) ∈ training_classes.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_prediction_in_classes(
    predictions: &[usize],
    n_training_classes: usize,
) -> NbVerdict {
    if n_training_classes == 0 {
        return NbVerdict::Fail;
    }
    for &p in predictions {
        if p >= n_training_classes {
            return NbVerdict::Fail;
        }
    }
    NbVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 4: NB-004 — prediction deterministic.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_prediction_deterministic(
    predictions_run1: &[usize],
    predictions_run2: &[usize],
) -> NbVerdict {
    if predictions_run1.len() != predictions_run2.len() {
        return NbVerdict::Fail;
    }
    if predictions_run1 == predictions_run2 {
        NbVerdict::Pass
    } else {
        NbVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 5: NB-005 — separable-cluster accuracy ≥ 0.9.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_separable_accuracy(accuracy: f32) -> NbVerdict {
    if !accuracy.is_finite() {
        return NbVerdict::Fail;
    }
    if accuracy >= AC_NB_005_MIN_ACCURACY {
        NbVerdict::Pass
    } else {
        NbVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_prior_lower_zero_exclusive() {
        assert_eq!(AC_NB_002_PRIOR_LOWER_EXCLUSIVE, 0.0);
    }

    #[test]
    fn provenance_prior_upper_one_exclusive() {
        assert_eq!(AC_NB_002_PRIOR_UPPER_EXCLUSIVE, 1.0);
    }

    #[test]
    fn provenance_separable_accuracy_floor_is_0_9() {
        assert_eq!(AC_NB_005_MIN_ACCURACY, 0.9);
    }

    #[test]
    fn provenance_prior_sum_eps_is_1e_5() {
        assert_eq!(AC_NB_001_PRIOR_SUM_EPS, 1e-5);
    }

    // -------------------------------------------------------------------------
    // Section 2: NB-001 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn nb001_pass_two_class_balanced() {
        let priors = vec![0.5_f32, 0.5];
        assert_eq!(verdict_from_prior_sum_to_one(&priors), NbVerdict::Pass);
    }

    #[test]
    fn nb001_pass_three_class_unbalanced() {
        let priors = vec![0.7_f32, 0.2, 0.1];
        assert_eq!(verdict_from_prior_sum_to_one(&priors), NbVerdict::Pass);
    }

    #[test]
    fn nb001_pass_after_fit_balanced() {
        // 6 samples, 3 classes, 2 each — priors should be [1/3, 1/3, 1/3].
        let x = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let y = vec![0_usize, 0, 1, 1, 2, 2];
        let (priors, _, _) = fit_gnb(&x, &y, 6, 2, 3).unwrap();
        assert_eq!(verdict_from_prior_sum_to_one(&priors), NbVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: NB-001 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn nb001_fail_priors_sum_to_half() {
        let priors = vec![0.25_f32, 0.25];
        assert_eq!(verdict_from_prior_sum_to_one(&priors), NbVerdict::Fail);
    }

    #[test]
    fn nb001_fail_priors_sum_to_two() {
        let priors = vec![1.0_f32, 1.0];
        assert_eq!(verdict_from_prior_sum_to_one(&priors), NbVerdict::Fail);
    }

    #[test]
    fn nb001_fail_priors_empty() {
        let priors: Vec<f32> = vec![];
        assert_eq!(verdict_from_prior_sum_to_one(&priors), NbVerdict::Fail);
    }

    #[test]
    fn nb001_fail_priors_nan() {
        let priors = vec![0.5_f32, f32::NAN];
        assert_eq!(verdict_from_prior_sum_to_one(&priors), NbVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: NB-002 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn nb002_pass_two_class_observed() {
        let priors = vec![0.6_f32, 0.4];
        let observed = vec![true, true];
        assert_eq!(
            verdict_from_prior_bounded(&priors, &observed),
            NbVerdict::Pass
        );
    }

    #[test]
    fn nb002_pass_unobserved_class_zero_allowed() {
        // class 2 wasn't seen — its prior is 0, mask[2]=false: allowed.
        let priors = vec![0.5_f32, 0.5, 0.0];
        let observed = vec![true, true, false];
        assert_eq!(
            verdict_from_prior_bounded(&priors, &observed),
            NbVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: NB-002 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn nb002_fail_observed_class_with_prior_zero() {
        // observed[k]=true ⇒ p must be > 0.
        let priors = vec![0.0_f32, 1.0];
        let observed = vec![true, true];
        assert_eq!(
            verdict_from_prior_bounded(&priors, &observed),
            NbVerdict::Fail
        );
    }

    #[test]
    fn nb002_fail_observed_class_with_prior_one() {
        // observed[k]=true ⇒ p must be < 1 (open interval).
        let priors = vec![1.0_f32, 0.0];
        let observed = vec![true, true];
        assert_eq!(
            verdict_from_prior_bounded(&priors, &observed),
            NbVerdict::Fail
        );
    }

    #[test]
    fn nb002_fail_negative_prior() {
        let priors = vec![-0.1_f32, 1.1];
        let observed = vec![true, true];
        assert_eq!(
            verdict_from_prior_bounded(&priors, &observed),
            NbVerdict::Fail
        );
    }

    #[test]
    fn nb002_fail_nan_prior() {
        let priors = vec![f32::NAN, 0.5];
        let observed = vec![true, true];
        assert_eq!(
            verdict_from_prior_bounded(&priors, &observed),
            NbVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: NB-003 — predictions ∈ training_classes.
    // -------------------------------------------------------------------------
    #[test]
    fn nb003_pass_all_in_range() {
        let preds = vec![0_usize, 1, 2, 0, 1];
        assert_eq!(
            verdict_from_prediction_in_classes(&preds, 3),
            NbVerdict::Pass
        );
    }

    #[test]
    fn nb003_pass_empty_predictions() {
        let preds: Vec<usize> = vec![];
        assert_eq!(
            verdict_from_prediction_in_classes(&preds, 5),
            NbVerdict::Pass
        );
    }

    #[test]
    fn nb003_fail_one_out_of_range() {
        let preds = vec![0_usize, 1, 2, 5];
        assert_eq!(
            verdict_from_prediction_in_classes(&preds, 3),
            NbVerdict::Fail
        );
    }

    #[test]
    fn nb003_fail_no_training_classes() {
        let preds = vec![0_usize];
        assert_eq!(
            verdict_from_prediction_in_classes(&preds, 0),
            NbVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: NB-004 — determinism.
    // -------------------------------------------------------------------------
    #[test]
    fn nb004_pass_identical_runs() {
        let r1 = vec![0_usize, 1, 0, 2];
        let r2 = vec![0_usize, 1, 0, 2];
        assert_eq!(
            verdict_from_prediction_deterministic(&r1, &r2),
            NbVerdict::Pass
        );
    }

    #[test]
    fn nb004_pass_empty_both() {
        let r: Vec<usize> = vec![];
        assert_eq!(
            verdict_from_prediction_deterministic(&r, &r),
            NbVerdict::Pass
        );
    }

    #[test]
    fn nb004_fail_one_off() {
        let r1 = vec![0_usize, 1, 0, 2];
        let r2 = vec![0_usize, 1, 0, 1]; // last differs
        assert_eq!(
            verdict_from_prediction_deterministic(&r1, &r2),
            NbVerdict::Fail
        );
    }

    #[test]
    fn nb004_fail_length_mismatch() {
        let r1 = vec![0_usize, 1, 2];
        let r2 = vec![0_usize, 1];
        assert_eq!(
            verdict_from_prediction_deterministic(&r1, &r2),
            NbVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 8: NB-005 — separable accuracy.
    // -------------------------------------------------------------------------
    #[test]
    fn nb005_pass_perfect_accuracy() {
        assert_eq!(
            verdict_from_separable_accuracy(1.0),
            NbVerdict::Pass
        );
    }

    #[test]
    fn nb005_pass_at_threshold() {
        assert_eq!(
            verdict_from_separable_accuracy(0.9),
            NbVerdict::Pass
        );
    }

    #[test]
    fn nb005_fail_below_threshold() {
        assert_eq!(
            verdict_from_separable_accuracy(0.89),
            NbVerdict::Fail
        );
    }

    #[test]
    fn nb005_fail_random_chance() {
        assert_eq!(
            verdict_from_separable_accuracy(0.5),
            NbVerdict::Fail
        );
    }

    #[test]
    fn nb005_fail_nan() {
        assert_eq!(
            verdict_from_separable_accuracy(f32::NAN),
            NbVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 9: Domain — fit/predict end-to-end on toy data.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_fit_predict_two_well_separated_clusters() {
        // Cluster 0 around (0,0); cluster 1 around (10,10). 2 features.
        let x = vec![
            0.0, 0.0, 0.1, -0.1, -0.2, 0.2, // class 0
            10.0, 10.0, 10.1, 9.9, 9.8, 10.2, // class 1
        ];
        let y = vec![0_usize, 0, 0, 1, 1, 1];
        let (priors, means, vars) = fit_gnb(&x, &y, 6, 2, 2).unwrap();

        // Priors balanced.
        assert_eq!(verdict_from_prior_sum_to_one(&priors), NbVerdict::Pass);
        let observed = vec![true, true];
        assert_eq!(
            verdict_from_prior_bounded(&priors, &observed),
            NbVerdict::Pass
        );

        // Predict at the cluster centers.
        let p0 = predict_gnb(&[0.0, 0.0], &priors, &means, &vars, 2);
        let p1 = predict_gnb(&[10.0, 10.0], &priors, &means, &vars, 2);
        assert_eq!(p0, 0);
        assert_eq!(p1, 1);

        // Predict twice — deterministic.
        let p0_again = predict_gnb(&[0.0, 0.0], &priors, &means, &vars, 2);
        assert_eq!(
            verdict_from_prediction_deterministic(&[p0], &[p0_again]),
            NbVerdict::Pass
        );

        // All predictions in [0, 2).
        let preds = vec![p0, p1];
        assert_eq!(
            verdict_from_prediction_in_classes(&preds, 2),
            NbVerdict::Pass
        );
    }

    #[test]
    fn domain_separable_accuracy_high() {
        // Generate 4 well-separated 2D clusters at (±5, ±5); fit + predict
        // on training data; check accuracy.
        let centers = [(-5.0, -5.0), (5.0, -5.0), (-5.0, 5.0), (5.0, 5.0)];
        let mut x = Vec::new();
        let mut y = Vec::new();
        // 5 perturbations per center.
        let pert = [
            (0.0_f32, 0.0),
            (0.1, 0.0),
            (-0.1, 0.0),
            (0.0, 0.1),
            (0.0, -0.1),
        ];
        for (k, (cx, cy)) in centers.iter().enumerate() {
            for (dx, dy) in pert.iter() {
                x.push(cx + dx);
                x.push(cy + dy);
                y.push(k);
            }
        }
        let n = y.len();
        let (priors, means, vars) = fit_gnb(&x, &y, n, 2, 4).unwrap();
        let mut correct = 0;
        for i in 0..n {
            let pred = predict_gnb(&x[i * 2..i * 2 + 2], &priors, &means, &vars, 2);
            if pred == y[i] {
                correct += 1;
            }
        }
        let acc = correct as f32 / n as f32;
        assert_eq!(
            verdict_from_separable_accuracy(acc),
            NbVerdict::Pass,
            "accuracy={acc}"
        );
    }

    // -------------------------------------------------------------------------
    // Section 10: Sweep — boundary fail modes.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_fail_priors_off_by_more_than_eps() {
        for sum in [0.5_f32, 0.99, 1.01, 1.5, 2.0] {
            let priors = vec![sum / 2.0, sum / 2.0];
            assert_eq!(
                verdict_from_prior_sum_to_one(&priors),
                NbVerdict::Fail,
                "sum={sum}"
            );
        }
    }

    #[test]
    fn sweep_pass_priors_within_eps() {
        for offset in [-1e-6_f32, 0.0, 1e-6] {
            let priors = vec![0.5 + offset, 0.5];
            assert_eq!(
                verdict_from_prior_sum_to_one(&priors),
                NbVerdict::Pass,
                "offset={offset}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 11: Realistic — contract regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_integer_truncation_caught() {
        // The contract failure for NB-001: "Normalization not applied or
        // integer truncation". 3/10 = 0.3 in f32 — but if someone wrote
        // `count / n_samples` as integer truncation, all priors would be
        // 0 ⇒ sum != 1.
        let priors_truncated = vec![0.0_f32, 0.0, 0.0];
        assert_eq!(
            verdict_from_prior_sum_to_one(&priors_truncated),
            NbVerdict::Fail
        );
    }

    #[test]
    fn realistic_argmax_returns_invalid_index() {
        // The contract failure for NB-003: "Argmax over log-posteriors
        // returns invalid index" — e.g., off-by-one or buffer overrun.
        let preds_invalid = vec![0_usize, 1, 99]; // 99 ∉ training classes
        assert_eq!(
            verdict_from_prediction_in_classes(&preds_invalid, 5),
            NbVerdict::Fail
        );
    }

    #[test]
    fn realistic_non_deterministic_floating_point() {
        // The contract failure for NB-004: "Non-deterministic floating
        // point or random state leakage" — same input, different output
        // means a hidden state was consulted.
        let r1 = vec![0_usize, 1, 2, 0];
        let r2 = vec![0_usize, 1, 2, 1]; // last flipped
        assert_eq!(
            verdict_from_prediction_deterministic(&r1, &r2),
            NbVerdict::Fail
        );
    }

    #[test]
    fn realistic_fit_rejects_invalid_input_shapes() {
        // n_samples=2, n_features=3 ⇒ x must have 6 entries.
        let x = vec![1.0_f32, 2.0]; // wrong size
        let y = vec![0_usize, 1];
        assert!(fit_gnb(&x, &y, 2, 3, 2).is_none());
    }

    #[test]
    fn realistic_fit_rejects_class_index_out_of_range() {
        let x = vec![1.0_f32, 2.0];
        let y = vec![0_usize, 5]; // 5 >= n_classes=2
        assert!(fit_gnb(&x, &y, 2, 1, 2).is_none());
    }
}
