//! Classification metrics for evaluating classifier performance.
//!
//! Provides accuracy, precision, recall, F1-score, and confusion matrix
//! computation for multi-class classification tasks.

use crate::primitives::Matrix;

/// Averaging strategy for multi-class metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Average {
    /// Calculate metrics for each label, return unweighted mean.
    Macro,
    /// Calculate metrics globally by counting total TP, FP, FN.
    Micro,
    /// Weighted mean by support (number of true instances per label).
    Weighted,
}

/// Compute classification accuracy.
///
/// accuracy = `correct_predictions` / `total_predictions`
///
/// # Arguments
///
/// * `y_pred` - Predicted class labels
/// * `y_true` - True class labels
///
/// # Returns
///
/// Accuracy score between 0.0 and 1.0
///
/// # Panics
///
/// Panics if vectors have different lengths or are empty.
///
/// # Examples
///
/// ```
/// use aprender::metrics::classification::accuracy;
///
/// let y_true = vec![0, 1, 2, 0, 1, 2];
/// let y_pred = vec![0, 2, 1, 0, 0, 1];
/// let acc = accuracy(&y_pred, &y_true);
/// assert!((acc - 0.333333).abs() < 0.001);
/// ```
#[must_use]
#[provable_contracts_macros::contract("metrics-classification-v1", equation = "accuracy")]
pub fn accuracy(y_pred: &[usize], y_true: &[usize]) -> f32 {
    contract_pre_accuracy!();
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");

    let correct = y_pred
        .iter()
        .zip(y_true.iter())
        .filter(|(p, t)| p == t)
        .count();

    correct as f32 / y_true.len() as f32
}

/// Compute precision score.
///
/// precision = TP / (TP + FP)
///
/// # Arguments
///
/// * `y_pred` - Predicted class labels
/// * `y_true` - True class labels
/// * `average` - Averaging strategy for multi-class
///
/// # Returns
///
/// Precision score between 0.0 and 1.0
///
/// # Panics
///
/// Panics if vectors have different lengths or are empty.
///
/// # Examples
///
/// ```
/// use aprender::metrics::classification::{precision, Average};
///
/// let y_true = vec![0, 1, 2, 0, 1, 2];
/// let y_pred = vec![0, 2, 1, 0, 0, 1];
/// let prec = precision(&y_pred, &y_true, Average::Macro);
/// assert!(prec >= 0.0 && prec <= 1.0);
/// ```
#[must_use]
#[provable_contracts_macros::contract("metrics-classification-v1", equation = "precision")]
pub fn precision(y_pred: &[usize], y_true: &[usize], average: Average) -> f32 {
    contract_pre_precision!();
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");

    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .max()
        .map_or(0, |&m| m + 1);

    if n_classes == 0 {
        return 0.0;
    }

    let (tp, fp, _, support) = compute_tp_fp_fn(y_pred, y_true, n_classes);

    match average {
        Average::Micro => {
            let total_tp: usize = tp.iter().sum();
            let total_fp: usize = fp.iter().sum();
            if total_tp + total_fp == 0 {
                0.0
            } else {
                total_tp as f32 / (total_tp + total_fp) as f32
            }
        }
        Average::Macro => {
            // Average only over labels present in y_true ∪ y_pred (PMAT-844),
            // matching sklearn's default labels = unique_labels(y_true, y_pred).
            let present = present_classes(&fp, &support);
            if present.is_empty() {
                return 0.0;
            }
            let sum: f32 = present.iter().map(|&i| class_precision(tp[i], fp[i])).sum();
            sum / present.len() as f32
        }
        Average::Weighted => {
            let total_support: usize = support.iter().sum();
            if total_support == 0 {
                return 0.0;
            }
            (0..n_classes)
                .map(|i| {
                    let prec = if tp[i] + fp[i] == 0 {
                        0.0
                    } else {
                        tp[i] as f32 / (tp[i] + fp[i]) as f32
                    };
                    prec * support[i] as f32 / total_support as f32
                })
                .sum()
        }
    }
}

/// Compute recall score.
///
/// recall = TP / (TP + FN)
///
/// # Arguments
///
/// * `y_pred` - Predicted class labels
/// * `y_true` - True class labels
/// * `average` - Averaging strategy for multi-class
///
/// # Returns
///
/// Recall score between 0.0 and 1.0
///
/// # Panics
///
/// Panics if vectors have different lengths or are empty.
///
/// # Examples
///
/// ```
/// use aprender::metrics::classification::{recall, Average};
///
/// let y_true = vec![0, 1, 2, 0, 1, 2];
/// let y_pred = vec![0, 2, 1, 0, 0, 1];
/// let rec = recall(&y_pred, &y_true, Average::Macro);
/// assert!(rec >= 0.0 && rec <= 1.0);
/// ```
#[must_use]
#[provable_contracts_macros::contract("metrics-classification-v1", equation = "recall")]
pub fn recall(y_pred: &[usize], y_true: &[usize], average: Average) -> f32 {
    contract_pre_recall!();
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");

    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .max()
        .map_or(0, |&m| m + 1);

    if n_classes == 0 {
        return 0.0;
    }

    let (tp, fp, fn_counts, support) = compute_tp_fp_fn(y_pred, y_true, n_classes);

    match average {
        Average::Micro => {
            let total_tp: usize = tp.iter().sum();
            let total_fn: usize = fn_counts.iter().sum();
            if total_tp + total_fn == 0 {
                0.0
            } else {
                total_tp as f32 / (total_tp + total_fn) as f32
            }
        }
        Average::Macro => {
            // Average only over labels present in y_true ∪ y_pred (PMAT-844).
            let present = present_classes(&fp, &support);
            if present.is_empty() {
                return 0.0;
            }
            let sum: f32 = present
                .iter()
                .map(|&i| class_recall(tp[i], fn_counts[i]))
                .sum();
            sum / present.len() as f32
        }
        Average::Weighted => {
            let total_support: usize = support.iter().sum();
            if total_support == 0 {
                return 0.0;
            }
            (0..n_classes)
                .map(|i| {
                    let rec = if tp[i] + fn_counts[i] == 0 {
                        0.0
                    } else {
                        tp[i] as f32 / (tp[i] + fn_counts[i]) as f32
                    };
                    rec * support[i] as f32 / total_support as f32
                })
                .sum()
        }
    }
}

/// Compute precision for a class given true positives and false positives.
fn class_precision(tp: usize, fp: usize) -> f32 {
    if tp + fp == 0 {
        0.0
    } else {
        tp as f32 / (tp + fp) as f32
    }
}

/// Compute recall for a class given true positives and false negatives.
fn class_recall(tp: usize, fn_count: usize) -> f32 {
    if tp + fn_count == 0 {
        0.0
    } else {
        tp as f32 / (tp + fn_count) as f32
    }
}

/// Compute F1 score from precision and recall.
fn f1_from_prec_rec(precision: f32, recall: f32) -> f32 {
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// Compute F1 score for a single class.
fn class_f1(tp: usize, fp: usize, fn_count: usize) -> f32 {
    let prec = class_precision(tp, fp);
    let rec = class_recall(tp, fn_count);
    f1_from_prec_rec(prec, rec)
}

/// Compute F1 score (harmonic mean of precision and recall).
///
/// F1 = 2 * (precision * recall) / (precision + recall)
///
/// # Arguments
///
/// * `y_pred` - Predicted class labels
/// * `y_true` - True class labels
/// * `average` - Averaging strategy for multi-class
///
/// # Returns
///
/// F1 score between 0.0 and 1.0
///
/// # Panics
///
/// Panics if vectors have different lengths or are empty.
///
/// # Examples
///
/// ```
/// use aprender::metrics::classification::{f1_score, Average};
///
/// let y_true = vec![0, 1, 2, 0, 1, 2];
/// let y_pred = vec![0, 2, 1, 0, 0, 1];
/// let f1 = f1_score(&y_pred, &y_true, Average::Macro);
/// assert!(f1 >= 0.0 && f1 <= 1.0);
/// ```
#[must_use]
#[provable_contracts_macros::contract("metrics-classification-v1", equation = "f1_score")]
pub fn f1_score(y_pred: &[usize], y_true: &[usize], average: Average) -> f32 {
    contract_pre_f1_score!();
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");

    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .max()
        .map_or(0, |&m| m + 1);

    if n_classes == 0 {
        return 0.0;
    }

    let (tp, fp, fn_counts, support) = compute_tp_fp_fn(y_pred, y_true, n_classes);

    match average {
        Average::Micro => {
            let total_tp: usize = tp.iter().sum();
            let total_fp: usize = fp.iter().sum();
            let total_fn: usize = fn_counts.iter().sum();
            class_f1(total_tp, total_fp, total_fn)
        }
        Average::Macro => {
            // Average only over labels present in y_true ∪ y_pred (PMAT-844).
            let present = present_classes(&fp, &support);
            if present.is_empty() {
                return 0.0;
            }
            let f1_sum: f32 = present
                .iter()
                .map(|&i| class_f1(tp[i], fp[i], fn_counts[i]))
                .sum();
            f1_sum / present.len() as f32
        }
        Average::Weighted => {
            let total_support: usize = support.iter().sum();
            if total_support == 0 {
                return 0.0;
            }
            (0..n_classes)
                .map(|i| {
                    let f1 = class_f1(tp[i], fp[i], fn_counts[i]);
                    f1 * support[i] as f32 / total_support as f32
                })
                .sum()
        }
    }
}

/// Per-class F1 in f64: `2 tp / (2 tp + fp + fn)`, `0` when the class has no true, predicted
/// or missed row (`zero_division = 0`).
///
/// This is the form `scripts/laya_train/metrics.py` `_f1` computes (`2.0 * tp / den`), and it
/// is exact up to one rounding: the integers are exact in f64 and the single division rounds
/// once. It is deliberately NOT `2PR / (P + R)`, the f32 form [`f1_score`] uses, which rounds
/// three times.
fn class_f1_f64(tp: usize, fp: usize, fn_count: usize) -> f64 {
    let den = 2 * tp + fp + fn_count;
    if den == 0 {
        0.0
    } else {
        (2 * tp) as f64 / den as f64
    }
}

/// Macro-F1 in f64 with an exactly-rounded mean — the Laya gate's `macro_f1`
/// (contracts/laya-finetune-gate-v1.yaml `numeric_agreement.macro_f1`).
///
/// ```text
/// L        = labels present in y_true ∪ y_pred           (PMAT-844, sklearn's default)
/// F1_c     = 2 tp_c / (2 tp_c + fp_c + fn_c)              in f64, 0 when the denominator is 0
/// macro_f1 = fsum(F1_c for c in L) / |L|                  (metrics::fsum, exactly rounded)
/// ```
///
/// Bit-identical to `scripts/laya_train/metrics.py` `macro_f1` (`math.fsum(f1s) / len(f1s)`),
/// so the Rust verifier and the Python trainer reach the same verdict at an exact threshold.
/// It shares the confusion counting and the present-label rule with [`f1_score`], which it does
/// NOT replace: `f1_score`'s f32 results are frozen for its other callers.
///
/// # Panics
///
/// Panics if the inputs differ in length or are empty.
#[must_use]
pub fn macro_f1_f64(y_pred: &[usize], y_true: &[usize]) -> f64 {
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");
    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .max()
        .map_or(0, |&m| m + 1);
    let (tp, fp, fn_counts, support) = compute_tp_fp_fn(y_pred, y_true, n_classes);
    let present = present_classes(&fp, &support);
    let f1s = present
        .iter()
        .map(|&i| class_f1_f64(tp[i], fp[i], fn_counts[i]));
    super::fsum(f1s) / present.len() as f64
}

/// The mean f64 F1 over the GIVEN labels with an exactly-rounded mean — the Laya gate's `f_avg`
/// (TweetEval stance: `against`, `favor`; laya-finetune-gate-v1 `numeric_agreement.f_avg`).
///
/// ```text
/// f_avg = fsum(F1_c for c in labels) / |labels|        F1_c as in macro_f1_f64
/// ```
///
/// A label absent from both `y_true` and `y_pred` scores 0 (`zero_division = 0`), exactly as
/// `scripts/laya_train/metrics.py` `f_avg`, which this equals bit for bit.
///
/// # Panics
///
/// Panics if the inputs differ in length, are empty, or `labels` is empty.
#[must_use]
pub fn mean_f1_over_labels_f64(y_pred: &[usize], y_true: &[usize], labels: &[usize]) -> f64 {
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");
    assert!(!labels.is_empty(), "f_avg needs at least one label");
    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .chain(labels.iter())
        .max()
        .map_or(0, |&m| m + 1);
    let (tp, fp, fn_counts, _) = compute_tp_fp_fn(y_pred, y_true, n_classes);
    let f1s = labels
        .iter()
        .map(|&c| class_f1_f64(tp[c], fp[c], fn_counts[c]));
    super::fsum(f1s) / labels.len() as f64
}

/// Compute per-class precision scores.
///
/// Returns a vector of precision values, one per class (ordered by class index).
/// For binary classification, index 1 is the positive-class precision.
///
/// precision_i = TP_i / (TP_i + FP_i)
///
/// # Arguments
///
/// * `y_pred` - Predicted class labels
/// * `y_true` - True class labels
///
/// # Returns
///
/// Vector of per-class precision scores (each in 0.0..=1.0)
///
/// # Panics
///
/// Panics if vectors have different lengths or are empty.
///
/// # Examples
///
/// ```
/// use aprender::metrics::classification::precision_per_class;
///
/// let y_true = vec![1, 0, 1, 0];
/// let y_pred = vec![1, 1, 0, 0];
/// let per_class = precision_per_class(&y_pred, &y_true);
/// assert_eq!(per_class.len(), 2);
/// // class 0: TP=1, FP=1 → 0.5
/// // class 1: TP=1, FP=1 → 0.5
/// assert!((per_class[0] - 0.5).abs() < 1e-5);
/// assert!((per_class[1] - 0.5).abs() < 1e-5);
/// ```
#[must_use]
pub fn precision_per_class(y_pred: &[usize], y_true: &[usize]) -> Vec<f32> {
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");

    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .max()
        .map_or(0, |&m| m + 1);

    let (tp, fp, _, _) = compute_tp_fp_fn(y_pred, y_true, n_classes);
    (0..n_classes)
        .map(|i| class_precision(tp[i], fp[i]))
        .collect()
}

/// Compute per-class recall scores.
///
/// Returns a vector of recall values, one per class (ordered by class index).
/// For binary classification, index 1 is the positive-class recall.
///
/// recall_i = TP_i / (TP_i + FN_i)
///
/// # Arguments
///
/// * `y_pred` - Predicted class labels
/// * `y_true` - True class labels
///
/// # Returns
///
/// Vector of per-class recall scores (each in 0.0..=1.0)
///
/// # Panics
///
/// Panics if vectors have different lengths or are empty.
///
/// # Examples
///
/// ```
/// use aprender::metrics::classification::recall_per_class;
///
/// let y_true = vec![1, 0, 1, 0];
/// let y_pred = vec![1, 1, 0, 0];
/// let per_class = recall_per_class(&y_pred, &y_true);
/// assert_eq!(per_class.len(), 2);
/// // class 0: TP=1, FN=1 → 0.5
/// // class 1: TP=1, FN=1 → 0.5
/// assert!((per_class[0] - 0.5).abs() < 1e-5);
/// assert!((per_class[1] - 0.5).abs() < 1e-5);
/// ```
#[must_use]
pub fn recall_per_class(y_pred: &[usize], y_true: &[usize]) -> Vec<f32> {
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");

    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .max()
        .map_or(0, |&m| m + 1);

    let (tp, _, fn_counts, _) = compute_tp_fp_fn(y_pred, y_true, n_classes);
    (0..n_classes)
        .map(|i| class_recall(tp[i], fn_counts[i]))
        .collect()
}

include!("classification_include_01.rs");

#[cfg(test)]
#[path = "tests_classification_contract.rs"]
mod tests_classification_contract;

/// Jaccard similarity score (intersection-over-union per class), matching
/// `sklearn.metrics.jaccard_score`. Per class `J = TP / (TP + FP + FN)`,
/// combined per the `average` strategy.
///
/// # Panics
/// Panics if the inputs differ in length or are empty.
#[must_use]
pub fn jaccard_score(y_pred: &[usize], y_true: &[usize], average: Average) -> f32 {
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");
    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .max()
        .map_or(0, |&m| m + 1);
    if n_classes == 0 {
        return 0.0;
    }
    let (tp, fp, fn_counts, support) = compute_tp_fp_fn(y_pred, y_true, n_classes);
    let class_jaccard = |tp: usize, fp: usize, fnc: usize| -> f32 {
        let denom = tp + fp + fnc;
        if denom == 0 {
            0.0
        } else {
            tp as f32 / denom as f32
        }
    };
    match average {
        Average::Micro => class_jaccard(tp.iter().sum(), fp.iter().sum(), fn_counts.iter().sum()),
        Average::Macro => {
            // Average only over labels present in y_true ∪ y_pred (PMAT-844).
            let present = present_classes(&fp, &support);
            if present.is_empty() {
                return 0.0;
            }
            present
                .iter()
                .map(|&i| class_jaccard(tp[i], fp[i], fn_counts[i]))
                .sum::<f32>()
                / present.len() as f32
        }
        Average::Weighted => {
            let total: usize = support.iter().sum();
            if total == 0 {
                return 0.0;
            }
            (0..n_classes)
                .map(|i| {
                    class_jaccard(tp[i], fp[i], fn_counts[i]) * support[i] as f32 / total as f32
                })
                .sum()
        }
    }
}

/// F-beta score, matching `sklearn.metrics.fbeta_score`. Generalizes F1
/// (`beta = 1`); `beta < 1` weights precision, `beta > 1` weights recall.
/// Per class `Fβ = (1+β²)·TP / ((1+β²)·TP + β²·FN + FP)`.
///
/// # Panics
/// Panics if the inputs differ in length or are empty.
#[must_use]
pub fn fbeta_score(y_pred: &[usize], y_true: &[usize], beta: f32, average: Average) -> f32 {
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");
    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .max()
        .map_or(0, |&m| m + 1);
    if n_classes == 0 {
        return 0.0;
    }
    let (tp, fp, fn_counts, support) = compute_tp_fp_fn(y_pred, y_true, n_classes);
    let b2 = beta * beta;
    let class_fbeta = |tp: usize, fp: usize, fnc: usize| -> f32 {
        let num = (1.0 + b2) * tp as f32;
        let denom = (1.0 + b2) * tp as f32 + b2 * fnc as f32 + fp as f32;
        if denom == 0.0 {
            0.0
        } else {
            num / denom
        }
    };
    match average {
        Average::Micro => class_fbeta(tp.iter().sum(), fp.iter().sum(), fn_counts.iter().sum()),
        Average::Macro => {
            // Average only over labels present in y_true ∪ y_pred (PMAT-844).
            let present = present_classes(&fp, &support);
            if present.is_empty() {
                return 0.0;
            }
            present
                .iter()
                .map(|&i| class_fbeta(tp[i], fp[i], fn_counts[i]))
                .sum::<f32>()
                / present.len() as f32
        }
        Average::Weighted => {
            let total: usize = support.iter().sum();
            if total == 0 {
                return 0.0;
            }
            (0..n_classes)
                .map(|i| class_fbeta(tp[i], fp[i], fn_counts[i]) * support[i] as f32 / total as f32)
                .sum()
        }
    }
}

#[cfg(test)]
mod tests_jaccard_fbeta {
    use super::*;

    /// FT-METRIC-JACCARD / FBETA: match sklearn within 1e-4.
    #[test]
    fn jaccard_and_fbeta_match_sklearn() {
        let yt = [0usize, 0, 1, 1, 1, 0, 1, 0];
        let yp = [0usize, 1, 1, 1, 0, 0, 1, 1];
        assert!((jaccard_score(&yp, &yt, Average::Macro) - 0.45).abs() < 1e-4);
        assert!((jaccard_score(&yp, &yt, Average::Micro) - 0.454_545).abs() < 1e-4);
        assert!((fbeta_score(&yp, &yt, 0.5, Average::Macro) - 0.625).abs() < 1e-4);
        assert!((fbeta_score(&yp, &yt, 2.0, Average::Macro) - 0.620_301).abs() < 1e-4);
    }
}

#[cfg(test)]
mod tests_macro_present_labels {
    use super::*;

    /// PMAT-844 / PO-MACRO-001: `Average::Macro` must average per-class metrics
    /// only over the labels ACTUALLY PRESENT (sorted union of y_true ∪ y_pred),
    /// matching `sklearn.metrics.precision_recall_fscore_support(average='macro')`
    /// whose default `labels = unique_labels(y_true, y_pred)`.
    ///
    /// Before the fix, the divisor was `max(labels)+1`, so absent intermediate
    /// class indices (e.g. class 1 when labels are {0, 2}) contributed spurious
    /// 0.0 terms. A PERFECT classifier on {0, 2} scored 2/3 instead of 1.0.
    ///
    /// RED (buggy): precision/recall/f1 = 0.6667 for the perfect {0,2} case,
    ///              precision = 0.3333 for the swapped {0,2} case.
    /// GREEN (fixed): 1.0 and 0.5 respectively (verified against scikit-learn).
    #[test]
    fn macro_average_noncontiguous_labels() {
        // Perfect classifier on non-contiguous labels {0, 2} → all metrics 1.0.
        let yt = vec![0usize, 2, 0, 2];
        let yp = vec![0usize, 2, 0, 2];
        let p = precision(&yp, &yt, Average::Macro);
        let r = recall(&yp, &yt, Average::Macro);
        let f = f1_score(&yp, &yt, Average::Macro);
        assert!(
            (p - 1.0).abs() < 1e-6,
            "FALSIFIED PMAT-844: macro precision={p} for perfect {{0,2}}, expected 1.0"
        );
        assert!(
            (r - 1.0).abs() < 1e-6,
            "FALSIFIED PMAT-844: macro recall={r} for perfect {{0,2}}, expected 1.0"
        );
        assert!(
            (f - 1.0).abs() < 1e-6,
            "FALSIFIED PMAT-844: macro f1={f} for perfect {{0,2}}, expected 1.0"
        );

        // Swapped predictions on {0, 2}: each class has precision 0.5 → macro 0.5.
        let yt2 = vec![0usize, 0, 2, 2];
        let yp2 = vec![0usize, 2, 2, 0];
        let p2 = precision(&yp2, &yt2, Average::Macro);
        assert!(
            (p2 - 0.5).abs() < 1e-6,
            "FALSIFIED PMAT-844: macro precision={p2} for swapped {{0,2}}, expected 0.5"
        );
    }

    /// PMAT-844 sibling metrics: `jaccard_score` and `fbeta_score` share the
    /// same Macro divisor and must also restrict to present labels.
    /// Perfect classifier on {0, 2} → both 1.0 (sklearn parity).
    #[test]
    fn macro_average_noncontiguous_jaccard_fbeta() {
        let yt = vec![0usize, 2, 0, 2];
        let yp = vec![0usize, 2, 0, 2];
        assert!((jaccard_score(&yp, &yt, Average::Macro) - 1.0).abs() < 1e-6);
        assert!((fbeta_score(&yp, &yt, 1.0, Average::Macro) - 1.0).abs() < 1e-6);
        assert!((fbeta_score(&yp, &yt, 0.5, Average::Macro) - 1.0).abs() < 1e-6);
    }

    /// Regression guard: contiguous labels {0, 1, 2} must be unchanged by the
    /// present-label fix (every class is present, so divisor == n_classes).
    #[test]
    fn macro_average_contiguous_labels_unchanged() {
        // Perfect classifier on dense {0, 1, 2} → all metrics 1.0.
        let y = vec![0usize, 1, 2, 0, 1, 2];
        assert!((precision(&y, &y, Average::Macro) - 1.0).abs() < 1e-6);
        assert!((recall(&y, &y, Average::Macro) - 1.0).abs() < 1e-6);
        assert!((f1_score(&y, &y, Average::Macro) - 1.0).abs() < 1e-6);

        // Known imperfect dense case keeps its established value (all 3 classes
        // present, divisor unchanged at 3).
        let yt = vec![0usize, 1, 2, 0, 1, 2];
        let yp = vec![0usize, 2, 1, 0, 0, 1];
        // class0: tp=2,fp=1 → 2/3; class1: tp=0,fp=2 → 0; class2: tp=0,fp=1 → 0.
        let expected = (2.0_f32 / 3.0) / 3.0;
        assert!((precision(&yp, &yt, Average::Macro) - expected).abs() < 1e-6);
    }
}

#[cfg(test)]
mod tests_macro_f1_f64 {
    use super::*;

    /// The Laya gate's f64 macro-F1 (plan 08-27): `2tp / (2tp + fp + fn)` per present label,
    /// exactly-rounded mean. Frozen values are `scripts/laya_train/metrics.py` `macro_f1` bits.
    #[test]
    fn macro_f1_f64_matches_metrics_py_bits() {
        // Always predicts 0 on y = [0, 1, 2, 0]: F1 = (2/3, 0, 0) over {0, 1, 2} -> 2/9.
        let got = macro_f1_f64(&[0, 0, 0, 0], &[0, 1, 2, 0]);
        assert_eq!(got.to_bits(), (2.0_f64 / 3.0 / 3.0).to_bits());
        // 13/60 on the 9-row exact-margin case's fine-tuned predictions: 0x3fcbbbbbbbbbbbbc.
        let y = [0, 0, 0, 1, 1, 1, 2, 2, 2];
        let ft = macro_f1_f64(&[2, 2, 2, 0, 1, 2, 0, 1, 2], &y);
        let zs = macro_f1_f64(&[2, 2, 2, 2, 2, 2, 2, 2, 2], &y);
        assert_eq!(ft.to_bits(), 0x3fcb_bbbb_bbbb_bbbc, "fine-tuned 13/60");
        assert_eq!(zs.to_bits(), 0x3fc5_5555_5555_5555, "zero-shot 1/6");
        // The margin is 1/20 exactly in rationals and 0.05000000000000002 in f64: PASS.
        assert!(ft - zs >= 0.05);
        // Present-label rule is the f32 function's (PMAT-844): perfect {0, 2} -> 1.
        assert_eq!(macro_f1_f64(&[0, 2, 0, 2], &[0, 2, 0, 2]), 1.0);
    }

    /// f_avg: the mean over the GIVEN labels; a label absent everywhere scores 0.
    #[test]
    fn mean_f1_over_labels_f64_follows_metrics_py() {
        // Always predicts 0 on y = [0, 1, 2, 0]: F1(1) = F1(2) = 0.
        assert_eq!(
            mean_f1_over_labels_f64(&[0, 0, 0, 0], &[0, 1, 2, 0], &[1, 2]),
            0.0
        );
        // pred = [0, 0, 0, 1] on y = [0, 1, 0, 1]: F1(0) = 4/5, F1(1) = 2/3.
        let got = mean_f1_over_labels_f64(&[0, 0, 0, 1], &[0, 1, 0, 1], &[0, 1]);
        assert_eq!(got.to_bits(), ((0.8_f64 + 2.0 / 3.0) / 2.0).to_bits());
        // A label index beyond every observed class is absent: 0, not a panic.
        assert_eq!(mean_f1_over_labels_f64(&[0, 1], &[0, 1], &[5]), 0.0);
    }
}
