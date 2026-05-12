// SHIP-TWO-001 — `decision-tree-v1` algorithm-level PARTIAL discharge
// for FALSIFY-DT-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/decision-tree-v1.yaml`.

// ===========================================================================
// Reference impurity / regression metrics
// ===========================================================================

use std::collections::HashMap;

#[must_use]
pub fn gini_impurity(labels: &[u32]) -> Option<f64> {
    if labels.is_empty() { return None; }
    let n = labels.len() as f64;
    let mut counts: HashMap<u32, u64> = HashMap::new();
    for &l in labels { *counts.entry(l).or_insert(0) += 1; }
    let mut sum_p_sq = 0.0_f64;
    for &c in counts.values() {
        let p = c as f64 / n;
        sum_p_sq += p * p;
    }
    Some(1.0 - sum_p_sq)
}

#[must_use]
pub fn weighted_child_gini(left: &[u32], right: &[u32]) -> Option<f64> {
    if left.is_empty() && right.is_empty() { return None; }
    let n_total = (left.len() + right.len()) as f64;
    let g_l = if left.is_empty() { 0.0 } else { gini_impurity(left)? };
    let g_r = if right.is_empty() { 0.0 } else { gini_impurity(right)? };
    let w_l = left.len() as f64 / n_total;
    let w_r = right.len() as f64 / n_total;
    Some(w_l * g_l + w_r * g_r)
}

#[must_use]
pub fn mse_targets(targets: &[f32]) -> Option<f64> {
    if targets.is_empty() { return None; }
    if targets.iter().any(|t| !t.is_finite()) { return None; }
    let n = targets.len() as f64;
    let mean = targets.iter().map(|t| *t as f64).sum::<f64>() / n;
    Some(targets.iter().map(|t| ((*t as f64) - mean).powi(2)).sum::<f64>() / n)
}

// ===========================================================================
// DT-001 — Gini bounded in [0, 1)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dt001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_gini_bounded(labels: &[u32]) -> Dt001Verdict {
    match gini_impurity(labels) {
        Some(g) if g.is_finite() && (0.0..1.0).contains(&g) => Dt001Verdict::Pass,
        // Pure single-class: g=0 is bounded (lower edge); allow.
        Some(g) if g == 0.0 => Dt001Verdict::Pass,
        _ => Dt001Verdict::Fail,
    }
}

// ===========================================================================
// DT-002 — Gini == 0 for pure node (all labels identical)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dt002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_gini_pure_zero(labels: &[u32]) -> Dt002Verdict {
    if labels.is_empty() { return Dt002Verdict::Fail; }
    let first = labels[0];
    if !labels.iter().all(|l| *l == first) { return Dt002Verdict::Fail; }
    match gini_impurity(labels) {
        Some(g) if g.abs() < 1e-12 => Dt002Verdict::Pass,
        _ => Dt002Verdict::Fail,
    }
}

// ===========================================================================
// DT-003 — Weighted child Gini <= parent Gini
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dt003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_gini_split_reduction(parent: &[u32], left: &[u32], right: &[u32]) -> Dt003Verdict {
    if left.len() + right.len() != parent.len() { return Dt003Verdict::Fail; }
    let g_p = match gini_impurity(parent) { Some(v) => v, None => return Dt003Verdict::Fail };
    let g_w = match weighted_child_gini(left, right) { Some(v) => v, None => return Dt003Verdict::Fail };
    if g_w <= g_p + 1e-9 { Dt003Verdict::Pass } else { Dt003Verdict::Fail }
}

// ===========================================================================
// DT-004 — MSE non-negative
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dt004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mse_nonneg(targets: &[f32]) -> Dt004Verdict {
    match mse_targets(targets) {
        Some(m) if m.is_finite() && m >= -1e-12 => Dt004Verdict::Pass,
        _ => Dt004Verdict::Fail,
    }
}

// ===========================================================================
// DT-005 — MSE == 0 for constant target
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dt005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mse_zero_constant(c: f32, n: usize) -> Dt005Verdict {
    if n == 0 || !c.is_finite() { return Dt005Verdict::Fail; }
    let targets = vec![c; n];
    match mse_targets(&targets) {
        Some(m) if m.abs() < 1e-9 => Dt005Verdict::Pass,
        _ => Dt005Verdict::Fail,
    }
}

// ===========================================================================
// DT-006 — Prediction deterministic
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dt006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_prediction_determinism(repeats: &[u32]) -> Dt006Verdict {
    if repeats.len() < 2 { return Dt006Verdict::Fail; }
    let first = repeats[0];
    if repeats.iter().all(|p| *p == first) { Dt006Verdict::Pass } else { Dt006Verdict::Fail }
}

// ===========================================================================
// DT-007 — Predictions ⊆ training classes
// ===========================================================================

use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dt007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_predictions_in_class_range(predictions: &[u32], training_classes: &[u32]) -> Dt007Verdict {
    if predictions.is_empty() || training_classes.is_empty() { return Dt007Verdict::Fail; }
    let train_set: HashSet<u32> = training_classes.iter().copied().collect();
    for p in predictions {
        if !train_set.contains(p) { return Dt007Verdict::Fail; }
    }
    Dt007Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference impl spot checks
    #[test] fn ref_gini_balanced_binary() {
        let g = gini_impurity(&[0_u32, 0, 1, 1]).unwrap();
        assert!((g - 0.5).abs() < 1e-12);
    }
    #[test] fn ref_gini_pure() {
        let g = gini_impurity(&[3_u32; 5]).unwrap();
        assert!(g.abs() < 1e-12);
    }
    #[test] fn ref_mse_constant() {
        let m = mse_targets(&[5.0_f32; 8]).unwrap();
        assert!(m.abs() < 1e-9);
    }

    // DT-001
    #[test] fn dt001_pass_pure() { assert_eq!(verdict_from_gini_bounded(&[1_u32; 4]), Dt001Verdict::Pass); }
    #[test] fn dt001_pass_balanced() { assert_eq!(verdict_from_gini_bounded(&[0_u32, 1, 0, 1]), Dt001Verdict::Pass); }
    #[test] fn dt001_pass_three_class() {
        assert_eq!(verdict_from_gini_bounded(&[0_u32, 1, 2, 0, 1, 2]), Dt001Verdict::Pass);
    }
    #[test] fn dt001_fail_empty() { assert_eq!(verdict_from_gini_bounded(&[]), Dt001Verdict::Fail); }

    // DT-002
    #[test] fn dt002_pass_constant_zero() { assert_eq!(verdict_from_gini_pure_zero(&[0_u32; 5]), Dt002Verdict::Pass); }
    #[test] fn dt002_pass_constant_seven() { assert_eq!(verdict_from_gini_pure_zero(&[7_u32; 3]), Dt002Verdict::Pass); }
    #[test] fn dt002_fail_mixed() { assert_eq!(verdict_from_gini_pure_zero(&[0_u32, 1, 0]), Dt002Verdict::Fail); }
    #[test] fn dt002_fail_empty() { assert_eq!(verdict_from_gini_pure_zero(&[]), Dt002Verdict::Fail); }

    // DT-003
    #[test] fn dt003_pass_perfect_split() {
        let parent = vec![0_u32, 0, 1, 1];
        let left = vec![0_u32, 0];
        let right = vec![1_u32, 1];
        assert_eq!(verdict_from_gini_split_reduction(&parent, &left, &right), Dt003Verdict::Pass);
    }
    #[test] fn dt003_pass_no_change() {
        // Split that doesn't reduce impurity (still <= parent vacuously).
        let parent = vec![0_u32, 1, 0, 1];
        let left = vec![0_u32, 1];
        let right = vec![0_u32, 1];
        assert_eq!(verdict_from_gini_split_reduction(&parent, &left, &right), Dt003Verdict::Pass);
    }
    #[test] fn dt003_fail_size_mismatch() {
        let parent = vec![0_u32, 1];
        let left = vec![0_u32];
        let right = vec![0_u32, 1];
        assert_eq!(verdict_from_gini_split_reduction(&parent, &left, &right), Dt003Verdict::Fail);
    }

    // DT-004
    #[test] fn dt004_pass_random() {
        let t = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(verdict_from_mse_nonneg(&t), Dt004Verdict::Pass);
    }
    #[test] fn dt004_pass_constant() {
        let t = vec![5.0_f32; 8];
        assert_eq!(verdict_from_mse_nonneg(&t), Dt004Verdict::Pass);
    }
    #[test] fn dt004_fail_empty() { assert_eq!(verdict_from_mse_nonneg(&[]), Dt004Verdict::Fail); }
    #[test] fn dt004_fail_nan() {
        let t = vec![1.0_f32, f32::NAN];
        assert_eq!(verdict_from_mse_nonneg(&t), Dt004Verdict::Fail);
    }

    // DT-005
    #[test] fn dt005_pass_n8() { assert_eq!(verdict_from_mse_zero_constant(3.0, 8), Dt005Verdict::Pass); }
    #[test] fn dt005_pass_n128() { assert_eq!(verdict_from_mse_zero_constant(0.5, 128), Dt005Verdict::Pass); }
    #[test] fn dt005_fail_n_zero() { assert_eq!(verdict_from_mse_zero_constant(1.0, 0), Dt005Verdict::Fail); }
    #[test] fn dt005_fail_nan_c() { assert_eq!(verdict_from_mse_zero_constant(f32::NAN, 8), Dt005Verdict::Fail); }

    // DT-006
    #[test] fn dt006_pass_consistent() {
        assert_eq!(verdict_from_prediction_determinism(&[5_u32, 5, 5, 5]), Dt006Verdict::Pass);
    }
    #[test] fn dt006_fail_drift() {
        assert_eq!(verdict_from_prediction_determinism(&[5_u32, 5, 6]), Dt006Verdict::Fail);
    }
    #[test] fn dt006_fail_too_few() {
        assert_eq!(verdict_from_prediction_determinism(&[5_u32]), Dt006Verdict::Fail);
    }

    // DT-007
    #[test] fn dt007_pass_subset() {
        let preds = vec![0_u32, 1, 2, 0];
        let train = vec![0_u32, 1, 2];
        assert_eq!(verdict_from_predictions_in_class_range(&preds, &train), Dt007Verdict::Pass);
    }
    #[test] fn dt007_fail_unseen_class() {
        let preds = vec![0_u32, 1, 5];
        let train = vec![0_u32, 1, 2];
        assert_eq!(verdict_from_predictions_in_class_range(&preds, &train), Dt007Verdict::Fail);
    }
    #[test] fn dt007_fail_empty_train() {
        assert_eq!(verdict_from_predictions_in_class_range(&[0_u32], &[]), Dt007Verdict::Fail);
    }
}
