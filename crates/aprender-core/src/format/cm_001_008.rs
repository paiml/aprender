// SHIP-TWO-001 — `metrics-classification-v1` algorithm-level PARTIAL
// discharge for FALSIFY-CM-001..008 (closes 8/8 sweep).
//
// Contract: `contracts/metrics-classification-v1.yaml`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvgMode { Macro, Micro, Weighted }

// ===========================================================================
// Reference scalar metrics (multi-class classification)
// ===========================================================================

/// `n_classes x n_classes` confusion matrix; `cm[t][p]` is the count
/// of samples whose true label is `t` and predicted label is `p`.
#[must_use]
pub fn confusion_matrix(y_true: &[u32], y_pred: &[u32], n_classes: usize) -> Option<Vec<Vec<u64>>> {
    if y_true.len() != y_pred.len() || y_true.is_empty() || n_classes == 0 { return None; }
    let mut cm = vec![vec![0_u64; n_classes]; n_classes];
    for (t, p) in y_true.iter().zip(y_pred) {
        if (*t as usize) >= n_classes || (*p as usize) >= n_classes { return None; }
        cm[*t as usize][*p as usize] += 1;
    }
    Some(cm)
}

#[must_use]
pub fn accuracy(y_true: &[u32], y_pred: &[u32]) -> Option<f64> {
    if y_true.len() != y_pred.len() || y_true.is_empty() { return None; }
    let correct = y_true.iter().zip(y_pred).filter(|(a, b)| a == b).count() as f64;
    Some(correct / y_true.len() as f64)
}

fn pr_per_class(cm: &[Vec<u64>]) -> Vec<(f64, f64, u64)> {
    let n = cm.len();
    let mut out = Vec::with_capacity(n);
    for c in 0..n {
        let tp = cm[c][c] as f64;
        let mut col_sum = 0_u64;
        let mut row_sum = 0_u64;
        for i in 0..n { col_sum += cm[i][c]; row_sum += cm[c][i]; }
        let p = if col_sum == 0 { 0.0 } else { tp / col_sum as f64 };
        let r = if row_sum == 0 { 0.0 } else { tp / row_sum as f64 };
        out.push((p, r, row_sum));
    }
    out
}

#[must_use]
pub fn precision(y_true: &[u32], y_pred: &[u32], n_classes: usize, mode: AvgMode) -> Option<f64> {
    let cm = confusion_matrix(y_true, y_pred, n_classes)?;
    let pr = pr_per_class(&cm);
    Some(match mode {
        AvgMode::Macro => pr.iter().map(|(p, _, _)| *p).sum::<f64>() / n_classes as f64,
        AvgMode::Micro => {
            let total: u64 = cm.iter().flat_map(|r| r.iter()).sum();
            let tp: u64 = (0..n_classes).map(|c| cm[c][c]).sum();
            if total == 0 { 0.0 } else { tp as f64 / total as f64 }
        }
        AvgMode::Weighted => {
            let total: u64 = pr.iter().map(|(_, _, s)| *s).sum();
            if total == 0 { 0.0 } else {
                pr.iter().map(|(p, _, s)| p * (*s as f64)).sum::<f64>() / total as f64
            }
        }
    })
}

#[must_use]
pub fn recall(y_true: &[u32], y_pred: &[u32], n_classes: usize, mode: AvgMode) -> Option<f64> {
    let cm = confusion_matrix(y_true, y_pred, n_classes)?;
    let pr = pr_per_class(&cm);
    Some(match mode {
        AvgMode::Macro => pr.iter().map(|(_, r, _)| *r).sum::<f64>() / n_classes as f64,
        AvgMode::Micro => {
            let total: u64 = cm.iter().flat_map(|r| r.iter()).sum();
            let tp: u64 = (0..n_classes).map(|c| cm[c][c]).sum();
            if total == 0 { 0.0 } else { tp as f64 / total as f64 }
        }
        AvgMode::Weighted => {
            let total: u64 = pr.iter().map(|(_, _, s)| *s).sum();
            if total == 0 { 0.0 } else {
                pr.iter().map(|(_, r, s)| r * (*s as f64)).sum::<f64>() / total as f64
            }
        }
    })
}

#[must_use]
pub fn f1_score(y_true: &[u32], y_pred: &[u32], n_classes: usize, mode: AvgMode) -> Option<f64> {
    let p = precision(y_true, y_pred, n_classes, mode)?;
    let r = recall(y_true, y_pred, n_classes, mode)?;
    if p + r == 0.0 { return Some(0.0); }
    Some(2.0 * p * r / (p + r))
}

// ===========================================================================
// CM-001 — Accuracy bounded in [0, 1]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cm001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_accuracy_bounded(y_true: &[u32], y_pred: &[u32]) -> Cm001Verdict {
    match accuracy(y_true, y_pred) {
        Some(a) if (0.0..=1.0).contains(&a) => Cm001Verdict::Pass,
        _ => Cm001Verdict::Fail,
    }
}

// ===========================================================================
// CM-002 — Precision bounded for all averaging modes
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cm002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_precision_bounded(y_true: &[u32], y_pred: &[u32], n_classes: usize) -> Cm002Verdict {
    for mode in [AvgMode::Macro, AvgMode::Micro, AvgMode::Weighted] {
        match precision(y_true, y_pred, n_classes, mode) {
            Some(p) if (0.0..=1.0).contains(&p) => {}
            _ => return Cm002Verdict::Fail,
        }
    }
    Cm002Verdict::Pass
}

// ===========================================================================
// CM-003 — F1 ≤ max(precision, recall)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cm003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_f1_harmonic_mean(
    y_true: &[u32], y_pred: &[u32], n_classes: usize,
) -> Cm003Verdict {
    for mode in [AvgMode::Macro, AvgMode::Micro, AvgMode::Weighted] {
        let p = match precision(y_true, y_pred, n_classes, mode) { Some(v) => v, None => return Cm003Verdict::Fail };
        let r = match recall(y_true, y_pred, n_classes, mode) { Some(v) => v, None => return Cm003Verdict::Fail };
        let f1 = match f1_score(y_true, y_pred, n_classes, mode) { Some(v) => v, None => return Cm003Verdict::Fail };
        if f1 > p.max(r) + 1e-9 { return Cm003Verdict::Fail; }
    }
    Cm003Verdict::Pass
}

// ===========================================================================
// CM-004 — Confusion matrix conservation: sum(CM) == n
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cm004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_cm_conservation(y_true: &[u32], y_pred: &[u32], n_classes: usize) -> Cm004Verdict {
    let cm = match confusion_matrix(y_true, y_pred, n_classes) { Some(v) => v, None => return Cm004Verdict::Fail };
    let total: u64 = cm.iter().flat_map(|r| r.iter()).sum();
    if total as usize == y_true.len() { Cm004Verdict::Pass } else { Cm004Verdict::Fail }
}

// ===========================================================================
// CM-005 — Perfect classification: ŷ = y ⇒ all metrics = 1
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cm005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_perfect_classification(y_true: &[u32], n_classes: usize) -> Cm005Verdict {
    if y_true.is_empty() { return Cm005Verdict::Fail; }
    // Every class must appear at least once for "perfect" to be 1.0
    // across all averaging modes (otherwise unobserved classes give 0/0
    // → 0 precision/recall under our convention).
    let mut seen = vec![false; n_classes];
    for &y in y_true { if (y as usize) < n_classes { seen[y as usize] = true; } }
    if !seen.iter().all(|s| *s) { return Cm005Verdict::Fail; }
    let y_pred = y_true.to_vec();
    let acc = match accuracy(y_true, &y_pred) { Some(v) => v, None => return Cm005Verdict::Fail };
    if (acc - 1.0).abs() > 1e-12 { return Cm005Verdict::Fail; }
    for mode in [AvgMode::Macro, AvgMode::Micro, AvgMode::Weighted] {
        let p = precision(y_true, &y_pred, n_classes, mode).expect("metric defined for valid inputs");
        let r = recall(y_true, &y_pred, n_classes, mode).expect("metric defined for valid inputs");
        let f = f1_score(y_true, &y_pred, n_classes, mode).expect("metric defined for valid inputs");
        if (p - 1.0).abs() > 1e-12 || (r - 1.0).abs() > 1e-12 || (f - 1.0).abs() > 1e-12 {
            return Cm005Verdict::Fail;
        }
    }
    Cm005Verdict::Pass
}

// ===========================================================================
// CM-006 — Micro-average identity: micro_p == micro_r == accuracy
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cm006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_micro_identity(y_true: &[u32], y_pred: &[u32], n_classes: usize) -> Cm006Verdict {
    let acc = match accuracy(y_true, y_pred) { Some(v) => v, None => return Cm006Verdict::Fail };
    let mp = match precision(y_true, y_pred, n_classes, AvgMode::Micro) { Some(v) => v, None => return Cm006Verdict::Fail };
    let mr = match recall(y_true, y_pred, n_classes, AvgMode::Micro) { Some(v) => v, None => return Cm006Verdict::Fail };
    if (mp - acc).abs() > 1e-12 { return Cm006Verdict::Fail; }
    if (mr - acc).abs() > 1e-12 { return Cm006Verdict::Fail; }
    Cm006Verdict::Pass
}

// ===========================================================================
// CM-007 — Recall bounded for all averaging modes
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cm007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_recall_bounded(y_true: &[u32], y_pred: &[u32], n_classes: usize) -> Cm007Verdict {
    for mode in [AvgMode::Macro, AvgMode::Micro, AvgMode::Weighted] {
        match recall(y_true, y_pred, n_classes, mode) {
            Some(r) if (0.0..=1.0).contains(&r) => {}
            _ => return Cm007Verdict::Fail,
        }
    }
    Cm007Verdict::Pass
}

// ===========================================================================
// CM-008 — F1 bounded
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cm008Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_f1_bounded(y_true: &[u32], y_pred: &[u32], n_classes: usize) -> Cm008Verdict {
    for mode in [AvgMode::Macro, AvgMode::Micro, AvgMode::Weighted] {
        match f1_score(y_true, y_pred, n_classes, mode) {
            Some(f) if (0.0..=1.0).contains(&f) => {}
            _ => return Cm008Verdict::Fail,
        }
    }
    Cm008Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (Vec<u32>, Vec<u32>) {
        // 6 samples, 3 classes; 4 correct.
        let y = vec![0, 1, 2, 0, 1, 2];
        let p = vec![0, 1, 0, 0, 2, 2];
        (y, p)
    }

    // CM-001
    #[test] fn cm001_pass_normal() {
        let (y, p) = sample();
        assert_eq!(verdict_from_accuracy_bounded(&y, &p), Cm001Verdict::Pass);
    }
    #[test] fn cm001_pass_perfect() {
        let y = vec![0, 1, 2];
        assert_eq!(verdict_from_accuracy_bounded(&y, &y), Cm001Verdict::Pass);
    }
    #[test] fn cm001_fail_empty() {
        assert_eq!(verdict_from_accuracy_bounded(&[], &[]), Cm001Verdict::Fail);
    }
    #[test] fn cm001_fail_length_mismatch() {
        assert_eq!(verdict_from_accuracy_bounded(&[0, 1], &[0]), Cm001Verdict::Fail);
    }

    // CM-002
    #[test] fn cm002_pass_normal() {
        let (y, p) = sample();
        assert_eq!(verdict_from_precision_bounded(&y, &p, 3), Cm002Verdict::Pass);
    }
    #[test] fn cm002_fail_label_oob() {
        // pred 5 with n_classes=3 → confusion_matrix returns None.
        assert_eq!(verdict_from_precision_bounded(&[0, 1], &[0, 5], 3), Cm002Verdict::Fail);
    }

    // CM-003
    #[test] fn cm003_pass_normal() {
        let (y, p) = sample();
        assert_eq!(verdict_from_f1_harmonic_mean(&y, &p, 3), Cm003Verdict::Pass);
    }

    // CM-004
    #[test] fn cm004_pass_conservation() {
        let (y, p) = sample();
        assert_eq!(verdict_from_cm_conservation(&y, &p, 3), Cm004Verdict::Pass);
    }
    #[test] fn cm004_fail_zero_classes() {
        let (y, p) = sample();
        assert_eq!(verdict_from_cm_conservation(&y, &p, 0), Cm004Verdict::Fail);
    }

    // CM-005
    #[test] fn cm005_pass_perfect() {
        let y = vec![0, 1, 2, 0, 1, 2];
        assert_eq!(verdict_from_perfect_classification(&y, 3), Cm005Verdict::Pass);
    }
    #[test] fn cm005_fail_unseen_class() {
        // Classes {0, 1} represented but n_classes = 3 — class 2 unseen.
        let y = vec![0, 1, 0, 1];
        assert_eq!(verdict_from_perfect_classification(&y, 3), Cm005Verdict::Fail);
    }
    #[test] fn cm005_fail_empty() {
        assert_eq!(verdict_from_perfect_classification(&[], 3), Cm005Verdict::Fail);
    }

    // CM-006
    #[test] fn cm006_pass_normal() {
        let (y, p) = sample();
        assert_eq!(verdict_from_micro_identity(&y, &p, 3), Cm006Verdict::Pass);
    }
    #[test] fn cm006_pass_perfect() {
        let y = vec![0, 1, 2, 0, 1, 2];
        assert_eq!(verdict_from_micro_identity(&y, &y, 3), Cm006Verdict::Pass);
    }

    // CM-007
    #[test] fn cm007_pass_normal() {
        let (y, p) = sample();
        assert_eq!(verdict_from_recall_bounded(&y, &p, 3), Cm007Verdict::Pass);
    }

    // CM-008
    #[test] fn cm008_pass_normal() {
        let (y, p) = sample();
        assert_eq!(verdict_from_f1_bounded(&y, &p, 3), Cm008Verdict::Pass);
    }
    #[test] fn cm008_fail_oob_label() {
        assert_eq!(verdict_from_f1_bounded(&[0, 1], &[0, 5], 3), Cm008Verdict::Fail);
    }

    // Reference impl spot checks
    #[test] fn ref_perfect_acc() {
        let y = vec![0, 1, 2, 0];
        let acc = accuracy(&y, &y).expect("metric defined for valid inputs");
        assert!((acc - 1.0).abs() < 1e-12);
    }

    #[test] fn ref_three_quarter_accuracy() {
        let y = vec![0, 1, 2, 0];
        let p = vec![0, 1, 2, 1];
        let acc = accuracy(&y, &p).expect("metric defined for valid inputs");
        assert!((acc - 0.75).abs() < 1e-12);
    }

    #[test] fn ref_micro_equals_accuracy() {
        let (y, p) = sample();
        let acc = accuracy(&y, &p).expect("metric defined for valid inputs");
        let mp = precision(&y, &p, 3, AvgMode::Micro).expect("metric defined for valid inputs");
        let mr = recall(&y, &p, 3, AvgMode::Micro).expect("metric defined for valid inputs");
        assert!((acc - mp).abs() < 1e-12);
        assert!((acc - mr).abs() < 1e-12);
    }
}
