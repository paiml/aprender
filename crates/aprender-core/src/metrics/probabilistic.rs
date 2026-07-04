//! Probabilistic classification metrics — ROC AUC and log loss.
//!
//! These operate on predicted scores/probabilities (not hard labels), matching
//! `sklearn.metrics.roc_auc_score` and `sklearn.metrics.log_loss`. Pillar 1
//! (beat scikit-learn): closes a verified-absent gap in the classification
//! metric surface (apr had accuracy/precision/recall/f1 but no score-based
//! metrics, so generic sklearn-style classifier evaluation couldn't run).

use core::cmp::Ordering;

/// Machine epsilon for IEEE-754 binary64, matching `np.finfo(float64).eps`.
/// sklearn clips/floors with this exact value (e.g. `log_loss` clamps
/// predictions to `[EPS, 1 − EPS]`); the legacy `1e-15` floor diverged by >4×
/// on boundary inputs. PMAT-929.
const FINFO_F64_EPS: f64 = f64::EPSILON; // 2.220446049250313e-16

/// Binary ROC AUC via the Mann–Whitney U statistic (rank-based, tie-averaged).
///
/// `y_true`: labels in {0, 1}. `y_score`: predicted scores (higher ⇒ more
/// likely positive). Returns the area under the ROC curve, matching
/// `sklearn.metrics.roc_auc_score`. Returns `NaN` if only one class is present
/// (AUC is undefined), mirroring sklearn raising in that case.
///
/// # Panics
/// Panics if `y_true` and `y_score` differ in length.
#[must_use]
pub fn roc_auc_score(y_true: &[usize], y_score: &[f32]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_score.len(),
        "roc_auc_score: y_true/y_score length mismatch"
    );
    let n = y_true.len();
    if n == 0 {
        return f32::NAN;
    }
    // Rank scores ascending, averaging ranks within tie blocks.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        y_score[a]
            .partial_cmp(&y_score[b])
            .unwrap_or(Ordering::Equal)
    });
    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && y_score[idx[j + 1]] == y_score[idx[i]] {
            j += 1;
        }
        let avg_rank = (i + j) as f32 / 2.0 + 1.0; // 1-based average rank over the tie block
        for &orig in &idx[i..=j] {
            ranks[orig] = avg_rank;
        }
        i = j + 1;
    }
    let n_pos = y_true.iter().filter(|&&y| y == 1).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f32::NAN;
    }
    let sum_ranks_pos: f32 = (0..n).filter(|&k| y_true[k] == 1).map(|k| ranks[k]).sum();
    (sum_ranks_pos - (n_pos * (n_pos + 1)) as f32 / 2.0) / (n_pos as f32 * n_neg as f32)
}

/// Binary log loss (cross-entropy), matching `sklearn.metrics.log_loss`.
///
/// `y_true`: labels in {0, 1}. `y_prob`: predicted P(y = 1), clamped to
/// `[eps, 1 − eps]` with `eps = finfo(float64).eps` (≈2.22e-16) to keep the
/// log finite — matching sklearn's clip exactly (PMAT-929). Accumulates in f64
/// to match sklearn's precision.
///
/// # Panics
/// Panics if `y_true` and `y_prob` differ in length.
#[must_use]
pub fn log_loss(y_true: &[usize], y_prob: &[f32]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_prob.len(),
        "log_loss: y_true/y_prob length mismatch"
    );
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for k in 0..n {
        let p = f64::from(y_prob[k]).clamp(FINFO_F64_EPS, 1.0 - FINFO_F64_EPS);
        let y = y_true[k] as f64;
        sum += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
    }
    (sum / n as f64) as f32
}

/// Binary average precision (area under the precision–recall curve), matching
/// `sklearn.metrics.average_precision_score`.
///
/// Computed as the step-function PR area `Σ (Rₙ − Rₙ₋₁)·Pₙ` over score
/// thresholds (descending), with tied scores grouped into one threshold (so the
/// result is rank-only, not threshold-position dependent). `y_true`: labels in
/// {0, 1}. `y_score`: predicted scores (higher ⇒ more likely positive).
/// Returns `0.0` when there are no positive samples, matching sklearn's
/// convention ("No positive class found"; recall→1, AP→0.0) — PMAT-929. Also
/// returns `0.0` for empty input.
///
/// # Panics
/// Panics if `y_true` and `y_score` differ in length.
#[must_use]
pub fn average_precision_score(y_true: &[usize], y_score: &[f32]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_score.len(),
        "average_precision_score: y_true/y_score length mismatch"
    );
    let n = y_true.len();
    let n_pos = y_true.iter().filter(|&&y| y == 1).count();
    if n == 0 || n_pos == 0 {
        // sklearn convention: AP is 0.0 with no positives (not undefined).
        return 0.0;
    }
    // Sort by score descending.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(Ordering::Equal)
    });
    let (mut tp, mut fp) = (0usize, 0usize);
    let mut ap = 0.0f64;
    let mut prev_recall = 0.0f64;
    let mut i = 0;
    while i < n {
        // Group tied scores into a single threshold block.
        let mut j = i;
        while j < n && y_score[idx[j]] == y_score[idx[i]] {
            if y_true[idx[j]] == 1 {
                tp += 1;
            } else {
                fp += 1;
            }
            j += 1;
        }
        let recall = tp as f64 / n_pos as f64;
        let precision = tp as f64 / (tp + fp) as f64;
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
        i = j;
    }
    ap as f32
}

/// Shared `sklearn.metrics._ranking._binary_clf_curve` for `pos_label = 1`.
///
/// Sorts samples by descending score, then at each DISTINCT score value records
/// the cumulative true-positive (`tps`) and false-positive (`fps`) counts and the
/// threshold. Returns `(fps, tps, thresholds)` aligned, in descending-threshold
/// order (tied scores collapse into a single threshold). `fps.last()` is the
/// total negative count, `tps.last()` the total positive count.
fn binary_clf_curve(y_true: &[usize], y_score: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = y_true.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(Ordering::Equal)
    });
    let (mut fps, mut tps, mut thr) = (Vec::new(), Vec::new(), Vec::new());
    let (mut tp, mut fp) = (0.0f32, 0.0f32);
    let mut i = 0;
    while i < n {
        let s = y_score[idx[i]];
        while i < n && y_score[idx[i]] == s {
            if y_true[idx[i]] == 1 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            i += 1;
        }
        fps.push(fp);
        tps.push(tp);
        thr.push(s);
    }
    (fps, tps, thr)
}

/// Receiver-operating-characteristic curve, matching `sklearn.metrics.roc_curve`
/// (default `drop_intermediate=True`).
///
/// Returns `(fpr, tpr, thresholds)`. As in sklearn ≥1.3, an initial point
/// `(fpr=0, tpr=0, threshold=+∞)` is prepended so the curve starts at the origin,
/// and collinear intermediate points are dropped (kept iff an endpoint or the
/// second difference of `fps`/`tps` is non-zero) — this changes the point set but
/// not the curve the points trace. `y_true`: labels in {0, 1}. `y_score`:
/// predicted scores (higher ⇒ more likely positive). With only one class present
/// the corresponding rate is undefined and its axis is filled with `NaN`
/// (mirroring sklearn's warn-and-nan behaviour).
///
/// # Panics
/// Panics if `y_true` and `y_score` differ in length.
#[must_use]
pub fn roc_curve(y_true: &[usize], y_score: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(
        y_true.len(),
        y_score.len(),
        "roc_curve: y_true/y_score length mismatch"
    );
    let (fps, tps, thr) = binary_clf_curve(y_true, y_score);
    let m = fps.len();
    // drop_intermediate: keep endpoints + points whose 2nd diff of fps OR tps is
    // non-zero (i.e. a corner of the ROC step function). Only applied when m > 2,
    // matching sklearn's guard.
    let apply_drop = m > 2;
    let (mut kfps, mut ktps, mut kthr) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..m {
        let keep = !apply_drop
            || i == 0
            || i == m - 1
            || (fps[i + 1] - fps[i]) - (fps[i] - fps[i - 1]) != 0.0
            || (tps[i + 1] - tps[i]) - (tps[i] - tps[i - 1]) != 0.0;
        if keep {
            kfps.push(fps[i]);
            ktps.push(tps[i]);
            kthr.push(thr[i]);
        }
    }
    let n_neg = fps.last().copied().unwrap_or(0.0);
    let n_pos = tps.last().copied().unwrap_or(0.0);
    let mut fpr = vec![0.0f32];
    let mut tpr = vec![0.0f32];
    let mut thresholds = vec![f32::INFINITY];
    for i in 0..kfps.len() {
        fpr.push(if n_neg > 0.0 {
            kfps[i] / n_neg
        } else {
            f32::NAN
        });
        tpr.push(if n_pos > 0.0 {
            ktps[i] / n_pos
        } else {
            f32::NAN
        });
        thresholds.push(kthr[i]);
    }
    (fpr, tpr, thresholds)
}

/// Precision–recall curve, matching `sklearn.metrics.precision_recall_curve`.
///
/// Returns `(precision, recall, thresholds)` with `precision.len() ==
/// recall.len() == thresholds.len() + 1`: thresholds are in ascending order and a
/// terminal `(precision=1, recall=0)` sentinel point (with no threshold) is
/// appended, exactly as sklearn does. `y_true`: labels in {0, 1}. `y_score`:
/// predicted scores (higher ⇒ more likely positive).
///
/// # Panics
/// Panics if `y_true` and `y_score` differ in length.
#[must_use]
pub fn precision_recall_curve(y_true: &[usize], y_score: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(
        y_true.len(),
        y_score.len(),
        "precision_recall_curve: y_true/y_score length mismatch"
    );
    let (fps, tps, thr) = binary_clf_curve(y_true, y_score);
    let n_pos = tps.last().copied().unwrap_or(0.0);
    let m = fps.len();
    let mut precision = Vec::with_capacity(m + 1);
    let mut recall = Vec::with_capacity(m + 1);
    for i in (0..m).rev() {
        let denom = tps[i] + fps[i];
        precision.push(if denom > 0.0 { tps[i] / denom } else { 1.0 });
        recall.push(if n_pos > 0.0 { tps[i] / n_pos } else { 0.0 });
    }
    // Terminal sentinel: precision → 1, recall → 0 (no associated threshold).
    precision.push(1.0);
    recall.push(0.0);
    let thresholds: Vec<f32> = thr.iter().rev().copied().collect();
    (precision, recall, thresholds)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Oracle values pinned from scikit-learn 2026-06-11 (`uv run --with scikit-learn`).
    const YT: [usize; 8] = [0, 0, 1, 1, 1, 0, 1, 0];
    const YS: [f32; 8] = [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.55];

    /// FT-METRIC-ROCAUC: matches `sklearn.metrics.roc_auc_score` within 1e-4.
    #[test]
    fn roc_auc_matches_sklearn() {
        assert!((roc_auc_score(&YT, &YS) - 0.875).abs() < 1e-4);
        assert!((roc_auc_score(&[0, 0, 1, 1], &[0.1, 0.2, 0.8, 0.9]) - 1.0).abs() < 1e-4);
        // tie-averaging (sklearn = 0.75 here)
        assert!((roc_auc_score(&[0, 1, 0, 1], &[0.5, 0.5, 0.5, 0.9]) - 0.75).abs() < 1e-4);
        // one class present -> undefined
        assert!(roc_auc_score(&[1, 1], &[0.5, 0.6]).is_nan());
    }

    /// FT-METRIC-LOGLOSS: matches `sklearn.metrics.log_loss` within 1e-4.
    #[test]
    fn log_loss_matches_sklearn() {
        assert!((log_loss(&YT, &YS) - 0.421_605).abs() < 1e-4);
        // near-perfect predictions -> ~0
        assert!(log_loss(&[0, 1], &[1e-9, 1.0 - 1e-9]) < 1e-3);
    }

    /// FT-METRIC-LOGLOSS-EPS (PMAT-929): probabilities at the exact bounds
    /// {0.0, 1.0} must be clamped by `finfo(float64).eps` (≈2.220446e-16), not
    /// the legacy 1e-15 floor. sklearn 1.9.0 oracle:
    /// `log_loss([0,1],[0.0,1.0]) = 2.220446049250313e-16` (= −log(1−eps)).
    /// With a 1e-15 floor apr returned ≈9.99e-16 — a >4× divergence. log(0)
    /// must never produce inf/nan.
    #[test]
    fn log_loss_clamps_finfo_eps_at_bounds() {
        // Both predictions perfect at the exact boundary ⇒ loss = −log(1−eps).
        let got = log_loss(&[0, 1], &[0.0, 1.0]);
        assert!(got.is_finite(), "log(0) leaked inf/nan: {got}");
        // sklearn oracle: 2.220446049250313e-16 (f32 round ≈ 2.220446e-16).
        let sklearn = 2.220_446e-16_f32;
        assert!(
            (got - sklearn).abs() < 1e-18,
            "log_loss bound-clamp diverges from sklearn: apr={got}, sklearn={sklearn}"
        );
    }

    /// FT-METRIC-AVGPREC: matches `sklearn.metrics.average_precision_score` within 1e-4.
    #[test]
    fn average_precision_matches_sklearn() {
        assert!((average_precision_score(&YT, &YS) - 0.916_667).abs() < 1e-4);
        assert!((average_precision_score(&[0, 0, 1, 1], &[0.1, 0.2, 0.8, 0.9]) - 1.0).abs() < 1e-4);
        assert!((average_precision_score(&[1, 1, 0, 0], &[0.9, 0.8, 0.2, 0.1]) - 1.0).abs() < 1e-4);
    }

    /// FT-METRIC-AVGPREC-NOPOS (PMAT-929): with NO positive samples sklearn
    /// returns 0.0 (warns "No positive class found", recall→1, AP→0.0), NOT
    /// nan. sklearn 1.9.0 oracle: `average_precision_score([0,0,0],…) = 0.0`.
    /// apr previously returned `f32::NAN`, which poisons downstream means.
    #[test]
    fn average_precision_no_positives_is_zero() {
        let got = average_precision_score(&[0, 0, 0], &[0.1, 0.2, 0.3]);
        assert!(!got.is_nan(), "AP with no positives returned nan, not 0.0");
        assert!(
            (got - 0.0).abs() < 1e-7,
            "AP no-positive diverges from sklearn (expected 0.0): apr={got}"
        );
        // Two-element no-positive case (legacy test used .is_nan() here).
        assert_eq!(average_precision_score(&[0, 0], &[0.1, 0.2]), 0.0);
    }

    fn approx_vec(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length {got:?} != {want:?}");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            if w.is_infinite() {
                assert!(
                    g.is_infinite() && g.signum() == w.signum(),
                    "{what}[{i}]: {g} != {w}"
                );
            } else {
                assert!((g - w).abs() < tol, "{what}[{i}]: {g} != {w}");
            }
        }
    }

    /// FT-METRIC-ROCCURVE: matches `sklearn.metrics.roc_curve` (default
    /// drop_intermediate=True) on the pinned fixture. sklearn 1.9.0 oracle
    /// (2026-07-04): fpr=[0,0,0,.5,.5,1], tpr=[0,.25,.75,.75,1,1],
    /// thr=[inf,.9,.7,.4,.35,.1].
    #[test]
    fn roc_curve_matches_sklearn() {
        let (fpr, tpr, thr) = roc_curve(&YT, &YS);
        approx_vec(&fpr, &[0.0, 0.0, 0.0, 0.5, 0.5, 1.0], 1e-4, "fpr");
        approx_vec(&tpr, &[0.0, 0.25, 0.75, 0.75, 1.0, 1.0], 1e-4, "tpr");
        approx_vec(
            &thr,
            &[f32::INFINITY, 0.9, 0.7, 0.4, 0.35, 0.1],
            1e-4,
            "thresholds",
        );
        // A perfectly separable case (sklearn 1.9.0 oracle).
        let (fpr, tpr, _) = roc_curve(&[0, 0, 1, 1], &[0.1, 0.2, 0.8, 0.9]);
        approx_vec(&fpr, &[0.0, 0.0, 0.0, 1.0], 1e-4, "sep fpr");
        approx_vec(&tpr, &[0.0, 0.5, 1.0, 1.0], 1e-4, "sep tpr");
    }

    /// FT-METRIC-PRCURVE: matches `sklearn.metrics.precision_recall_curve` on the
    /// pinned fixture. sklearn 1.9.0 oracle (2026-07-04): precision len 9, recall
    /// len 9, thresholds len 8; ends at the (precision=1, recall=0) sentinel.
    #[test]
    fn precision_recall_curve_matches_sklearn() {
        let (prec, rec, thr) = precision_recall_curve(&YT, &YS);
        approx_vec(
            &prec,
            &[0.5, 0.571_429, 0.666_667, 0.6, 0.75, 1.0, 1.0, 1.0, 1.0],
            1e-4,
            "precision",
        );
        approx_vec(
            &rec,
            &[1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.5, 0.25, 0.0],
            1e-4,
            "recall",
        );
        approx_vec(
            &thr,
            &[0.1, 0.2, 0.35, 0.4, 0.55, 0.7, 0.8, 0.9],
            1e-4,
            "thresholds",
        );
        assert_eq!(prec.len(), rec.len(), "precision/recall length parity");
        assert_eq!(prec.len(), thr.len() + 1, "sentinel: len(P) == len(thr)+1");
    }
}
