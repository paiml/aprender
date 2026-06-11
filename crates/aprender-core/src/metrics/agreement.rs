//! Agreement / imbalance-robust classification metrics.
//!
//! `balanced_accuracy_score` and `matthews_corrcoef` (MCC) — label-based metrics
//! (operate on predicted *labels*, not scores) matching `sklearn.metrics`.
//! Pillar 1 (beat scikit-learn): both are standard for imbalanced classification
//! and were absent from apr's metric surface.

/// Number of classes implied by the maximum label seen in either vector.
fn n_classes(y_pred: &[usize], y_true: &[usize]) -> usize {
    y_true
        .iter()
        .chain(y_pred.iter())
        .max()
        .map_or(0, |&m| m + 1)
}

/// Balanced accuracy — the mean of per-class recall over the classes present in
/// `y_true`. Robust to class imbalance. Matches
/// `sklearn.metrics.balanced_accuracy_score`.
///
/// # Panics
/// Panics if `y_pred` and `y_true` differ in length.
#[must_use]
pub fn balanced_accuracy_score(y_pred: &[usize], y_true: &[usize]) -> f32 {
    assert_eq!(
        y_pred.len(),
        y_true.len(),
        "balanced_accuracy_score: length mismatch"
    );
    let k = n_classes(y_pred, y_true);
    if k == 0 {
        return 0.0;
    }
    let mut tp = vec![0usize; k];
    let mut total = vec![0usize; k];
    for (&p, &t) in y_pred.iter().zip(y_true) {
        total[t] += 1;
        if p == t {
            tp[t] += 1;
        }
    }
    let mut sum = 0.0f64;
    let mut present = 0usize;
    for c in 0..k {
        if total[c] > 0 {
            sum += tp[c] as f64 / total[c] as f64;
            present += 1;
        }
    }
    if present == 0 {
        return 0.0;
    }
    (sum / present as f64) as f32
}

/// Matthews correlation coefficient (multiclass generalization), matching
/// `sklearn.metrics.matthews_corrcoef`. Returns a value in [−1, 1] (1 = perfect,
/// 0 = no better than random); 0 when a variance term is degenerate.
///
/// # Panics
/// Panics if `y_pred` and `y_true` differ in length.
#[must_use]
pub fn matthews_corrcoef(y_pred: &[usize], y_true: &[usize]) -> f32 {
    assert_eq!(
        y_pred.len(),
        y_true.len(),
        "matthews_corrcoef: length mismatch"
    );
    let k = n_classes(y_pred, y_true);
    if k == 0 {
        return 0.0;
    }
    // Confusion matrix c_mat[true][pred] (i64 to avoid overflow in the products).
    let mut c_mat = vec![vec![0i64; k]; k];
    for (&p, &t) in y_pred.iter().zip(y_true) {
        c_mat[t][p] += 1;
    }
    let s = y_true.len() as i64;
    let correct: i64 = (0..k).map(|c| c_mat[c][c]).sum();
    let t_k: Vec<i64> = (0..k).map(|c| (0..k).map(|j| c_mat[c][j]).sum()).collect(); // true totals (rows)
    let p_k: Vec<i64> = (0..k).map(|c| (0..k).map(|i| c_mat[i][c]).sum()).collect(); // pred totals (cols)
    let sum_pt: i64 = (0..k).map(|c| p_k[c] * t_k[c]).sum();
    let sum_p2: i64 = p_k.iter().map(|&x| x * x).sum();
    let sum_t2: i64 = t_k.iter().map(|&x| x * x).sum();
    let num = (correct * s - sum_pt) as f64;
    let den = (((s * s - sum_p2) as f64) * ((s * s - sum_t2) as f64)).sqrt();
    if den == 0.0 {
        0.0
    } else {
        (num / den) as f32
    }
}

/// Cohen's kappa — inter-rater agreement corrected for chance, matching
/// `sklearn.metrics.cohen_kappa_score`. `κ = (p_o − p_e) / (1 − p_e)`, where
/// `p_o` is observed agreement (accuracy) and `p_e` is chance agreement. Returns
/// 0.0 when chance agreement is total (κ undefined).
///
/// # Panics
/// Panics if `y_pred` and `y_true` differ in length.
#[must_use]
pub fn cohen_kappa_score(y_pred: &[usize], y_true: &[usize]) -> f32 {
    assert_eq!(
        y_pred.len(),
        y_true.len(),
        "cohen_kappa_score: length mismatch"
    );
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let k = n_classes(y_pred, y_true);
    let mut pred_count = vec![0.0f64; k];
    let mut true_count = vec![0.0f64; k];
    let mut agree = 0usize;
    for (&p, &t) in y_pred.iter().zip(y_true) {
        pred_count[p] += 1.0;
        true_count[t] += 1.0;
        if p == t {
            agree += 1;
        }
    }
    let nf = n as f64;
    let p_o = agree as f64 / nf;
    let p_e: f64 = (0..k)
        .map(|c| (pred_count[c] / nf) * (true_count[c] / nf))
        .sum();
    if (1.0 - p_e).abs() < 1e-12 {
        return 0.0;
    }
    ((p_o - p_e) / (1.0 - p_e)) as f32
}

/// Hamming loss — the fraction of labels predicted incorrectly, matching
/// `sklearn.metrics.hamming_loss` for the multiclass single-label case.
///
/// # Panics
/// Panics if `y_pred` and `y_true` differ in length.
#[must_use]
pub fn hamming_loss(y_pred: &[usize], y_true: &[usize]) -> f32 {
    assert_eq!(y_pred.len(), y_true.len(), "hamming_loss: length mismatch");
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let mismatches = y_pred.iter().zip(y_true).filter(|(&p, &t)| p != t).count();
    mismatches as f32 / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // Oracle values pinned from scikit-learn 2026-06-11.
    const YT: [usize; 8] = [0, 0, 1, 1, 2, 2, 0, 1];
    const YP: [usize; 8] = [0, 1, 1, 1, 2, 0, 0, 2];

    /// FT-METRIC-BALACC: matches `sklearn.metrics.balanced_accuracy_score` within 1e-4.
    #[test]
    fn balanced_accuracy_matches_sklearn() {
        assert!((balanced_accuracy_score(&YP, &YT) - 0.611_111).abs() < 1e-4);
        assert!(
            (balanced_accuracy_score(&[0, 1, 1, 1, 0], &[0, 0, 1, 1, 1]) - 0.583_333).abs() < 1e-4
        );
    }

    /// FT-METRIC-MCC: matches `sklearn.metrics.matthews_corrcoef` within 1e-4.
    #[test]
    fn matthews_corrcoef_matches_sklearn() {
        assert!((matthews_corrcoef(&YP, &YT) - 0.428_571).abs() < 1e-4);
        assert!((matthews_corrcoef(&[0, 1, 1, 1, 0], &[0, 0, 1, 1, 1]) - 0.166_667).abs() < 1e-4);
        assert!((matthews_corrcoef(&[0, 1, 0, 1], &[0, 1, 0, 1]) - 1.0).abs() < 1e-4);
    }

    /// FT-METRIC-KAPPA: matches `sklearn.metrics.cohen_kappa_score` within 1e-4.
    #[test]
    fn cohen_kappa_matches_sklearn() {
        assert!((cohen_kappa_score(&YP, &YT) - 0.428_571).abs() < 1e-4);
        assert!((cohen_kappa_score(&[0, 1, 0, 1], &[0, 1, 0, 1]) - 1.0).abs() < 1e-4);
    }

    /// FT-METRIC-HAMMING: matches `sklearn.metrics.hamming_loss` within 1e-6.
    #[test]
    fn hamming_loss_matches_sklearn() {
        assert!((hamming_loss(&YP, &YT) - 0.375).abs() < 1e-6);
        assert!((hamming_loss(&[0, 1, 2], &[0, 1, 2])).abs() < 1e-6);
    }
}
