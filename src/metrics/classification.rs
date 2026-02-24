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
            let precisions: Vec<f32> = (0..n_classes)
                .map(|i| {
                    if tp[i] + fp[i] == 0 {
                        0.0
                    } else {
                        tp[i] as f32 / (tp[i] + fp[i]) as f32
                    }
                })
                .collect();
            precisions.iter().sum::<f32>() / n_classes as f32
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

    let (tp, _, fn_counts, support) = compute_tp_fp_fn(y_pred, y_true, n_classes);

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
            let recalls: Vec<f32> = (0..n_classes)
                .map(|i| {
                    if tp[i] + fn_counts[i] == 0 {
                        0.0
                    } else {
                        tp[i] as f32 / (tp[i] + fn_counts[i]) as f32
                    }
                })
                .collect();
            recalls.iter().sum::<f32>() / n_classes as f32
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
            let f1_sum: f32 = (0..n_classes)
                .map(|i| class_f1(tp[i], fp[i], fn_counts[i]))
                .sum();
            f1_sum / n_classes as f32
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
