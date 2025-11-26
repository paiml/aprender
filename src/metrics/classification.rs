//! Classification metrics for evaluating classifier performance.
//!
//! Provides accuracy, precision, recall, F1-score, and confusion matrix
//! computation for multi-class classification tasks.

use crate::primitives::Matrix;
use std::fmt::Write;

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
/// accuracy = correct_predictions / total_predictions
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

            let prec = if total_tp + total_fp == 0 {
                0.0
            } else {
                total_tp as f32 / (total_tp + total_fp) as f32
            };
            let rec = if total_tp + total_fn == 0 {
                0.0
            } else {
                total_tp as f32 / (total_tp + total_fn) as f32
            };

            if prec + rec == 0.0 {
                0.0
            } else {
                2.0 * prec * rec / (prec + rec)
            }
        }
        Average::Macro => {
            let f1s: Vec<f32> = (0..n_classes)
                .map(|i| {
                    let prec = if tp[i] + fp[i] == 0 {
                        0.0
                    } else {
                        tp[i] as f32 / (tp[i] + fp[i]) as f32
                    };
                    let rec = if tp[i] + fn_counts[i] == 0 {
                        0.0
                    } else {
                        tp[i] as f32 / (tp[i] + fn_counts[i]) as f32
                    };
                    if prec + rec == 0.0 {
                        0.0
                    } else {
                        2.0 * prec * rec / (prec + rec)
                    }
                })
                .collect();
            f1s.iter().sum::<f32>() / n_classes as f32
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
                    let rec = if tp[i] + fn_counts[i] == 0 {
                        0.0
                    } else {
                        tp[i] as f32 / (tp[i] + fn_counts[i]) as f32
                    };
                    let f1 = if prec + rec == 0.0 {
                        0.0
                    } else {
                        2.0 * prec * rec / (prec + rec)
                    };
                    f1 * support[i] as f32 / total_support as f32
                })
                .sum()
        }
    }
}

/// Compute confusion matrix.
///
/// Returns a matrix where element [i,j] is the count of samples
/// with true label i and predicted label j.
///
/// # Arguments
///
/// * `y_pred` - Predicted class labels
/// * `y_true` - True class labels
///
/// # Returns
///
/// Confusion matrix as Matrix<usize>
///
/// # Panics
///
/// Panics if vectors have different lengths or are empty.
///
/// # Examples
///
/// ```
/// use aprender::metrics::classification::confusion_matrix;
///
/// let y_true = vec![0, 0, 1, 1, 2, 2];
/// let y_pred = vec![0, 1, 1, 1, 2, 0];
/// let cm = confusion_matrix(&y_pred, &y_true);
/// assert_eq!(cm.n_rows(), 3);
/// assert_eq!(cm.n_cols(), 3);
/// ```
#[must_use]
pub fn confusion_matrix(y_pred: &[usize], y_true: &[usize]) -> Matrix<usize> {
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");

    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .max()
        .map_or(0, |&m| m + 1);

    let mut data = vec![0usize; n_classes * n_classes];

    for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
        data[true_label * n_classes + pred_label] += 1;
    }

    Matrix::from_vec(n_classes, n_classes, data)
        .expect("Confusion matrix dimensions match data length")
}

/// Generate a text classification report (sklearn-style).
///
/// # Arguments
///
/// * `y_pred` - Predicted class labels
/// * `y_true` - True class labels
///
/// # Returns
///
/// Formatted string with per-class and aggregate metrics.
///
/// # Panics
///
/// Panics if vectors have different lengths or are empty.
///
/// # Examples
///
/// ```
/// use aprender::metrics::classification::classification_report;
///
/// let y_true = vec![0, 0, 1, 1, 2, 2];
/// let y_pred = vec![0, 1, 1, 1, 2, 0];
/// let report = classification_report(&y_pred, &y_true);
/// assert!(report.contains("precision"));
/// assert!(report.contains("recall"));
/// ```
#[must_use]
pub fn classification_report(y_pred: &[usize], y_true: &[usize]) -> String {
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");

    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .max()
        .map_or(0, |&m| m + 1);
    let (tp, fp, fn_counts, support) = compute_tp_fp_fn(y_pred, y_true, n_classes);

    let mut report = String::new();
    report.push_str("              precision    recall  f1-score   support\n\n");

    for i in 0..n_classes {
        let prec = if tp[i] + fp[i] == 0 {
            0.0
        } else {
            tp[i] as f32 / (tp[i] + fp[i]) as f32
        };
        let rec = if tp[i] + fn_counts[i] == 0 {
            0.0
        } else {
            tp[i] as f32 / (tp[i] + fn_counts[i]) as f32
        };
        let f1 = if prec + rec == 0.0 {
            0.0
        } else {
            2.0 * prec * rec / (prec + rec)
        };

        let _ = writeln!(
            report,
            "    {i:>8}      {prec:>5.2}     {rec:>5.2}     {f1:>5.2}      {:>4}",
            support[i]
        );
    }

    report.push('\n');

    let total_support: usize = support.iter().sum();
    let macro_prec = precision(y_pred, y_true, Average::Macro);
    let macro_rec = recall(y_pred, y_true, Average::Macro);
    let macro_f1 = f1_score(y_pred, y_true, Average::Macro);

    let _ = writeln!(
        report,
        "   macro avg      {macro_prec:>5.2}     {macro_rec:>5.2}     {macro_f1:>5.2}      {total_support:>4}"
    );

    let weighted_prec = precision(y_pred, y_true, Average::Weighted);
    let weighted_rec = recall(y_pred, y_true, Average::Weighted);
    let weighted_f1 = f1_score(y_pred, y_true, Average::Weighted);

    let _ = writeln!(
        report,
        "weighted avg      {weighted_prec:>5.2}     {weighted_rec:>5.2}     {weighted_f1:>5.2}      {total_support:>4}"
    );

    report
}

/// Helper function to compute TP, FP, FN for each class.
fn compute_tp_fp_fn(
    y_pred: &[usize],
    y_true: &[usize],
    n_classes: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut tp = vec![0usize; n_classes];
    let mut fp = vec![0usize; n_classes];
    let mut fn_counts = vec![0usize; n_classes];
    let mut support = vec![0usize; n_classes];

    for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
        support[true_label] += 1;

        if true_label == pred_label {
            tp[true_label] += 1;
        } else {
            fp[pred_label] += 1;
            fn_counts[true_label] += 1;
        }
    }

    (tp, fp, fn_counts, support)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== ACCURACY TESTS ====================

    #[test]
    fn test_accuracy_perfect() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 2, 0, 1, 2];
        let acc = accuracy(&y_pred, &y_true);
        assert!((acc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_accuracy_worst() {
        let y_true = vec![0, 1, 2];
        let y_pred = vec![1, 2, 0];
        let acc = accuracy(&y_pred, &y_true);
        assert!((acc - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_accuracy_partial() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 2, 1, 0, 0, 1];
        let acc = accuracy(&y_pred, &y_true);
        // 2 correct out of 6
        assert!((acc - 2.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_accuracy_binary() {
        let y_true = vec![0, 0, 1, 1];
        let y_pred = vec![0, 1, 1, 0];
        let acc = accuracy(&y_pred, &y_true);
        assert!((acc - 0.5).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_accuracy_length_mismatch() {
        let y_true = vec![0, 1, 2];
        let y_pred = vec![0, 1];
        let _ = accuracy(&y_pred, &y_true);
    }

    #[test]
    #[should_panic(expected = "empty")]
    fn test_accuracy_empty() {
        let y_true: Vec<usize> = vec![];
        let y_pred: Vec<usize> = vec![];
        let _ = accuracy(&y_pred, &y_true);
    }

    // ==================== PRECISION TESTS ====================

    #[test]
    fn test_precision_perfect() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 2, 0, 1, 2];
        let prec = precision(&y_pred, &y_true, Average::Macro);
        assert!((prec - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_precision_macro_vs_micro() {
        let y_true = vec![0, 0, 0, 1, 1, 2];
        let y_pred = vec![0, 0, 1, 1, 0, 2];
        // Macro: average per-class precision
        // Micro: global TP / (TP + FP)
        let macro_prec = precision(&y_pred, &y_true, Average::Macro);
        let micro_prec = precision(&y_pred, &y_true, Average::Micro);

        assert!((0.0..=1.0).contains(&macro_prec));
        assert!((0.0..=1.0).contains(&micro_prec));
    }

    #[test]
    fn test_precision_weighted() {
        let y_true = vec![0, 0, 0, 0, 1, 2];
        let y_pred = vec![0, 0, 0, 0, 1, 2];
        let weighted = precision(&y_pred, &y_true, Average::Weighted);
        assert!((weighted - 1.0).abs() < 1e-6);
    }

    // ==================== RECALL TESTS ====================

    #[test]
    fn test_recall_perfect() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 2, 0, 1, 2];
        let rec = recall(&y_pred, &y_true, Average::Macro);
        assert!((rec - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_recall_zero_for_class() {
        // Class 2 never predicted
        let y_true = vec![0, 1, 2];
        let y_pred = vec![0, 1, 0];
        let rec = recall(&y_pred, &y_true, Average::Macro);
        // Class 0: 1/1, Class 1: 1/1, Class 2: 0/1
        // Macro avg = (1 + 1 + 0) / 3 = 2/3
        assert!((rec - 2.0 / 3.0).abs() < 1e-6);
    }

    // ==================== F1 SCORE TESTS ====================

    #[test]
    fn test_f1_perfect() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 2, 0, 1, 2];
        let f1 = f1_score(&y_pred, &y_true, Average::Macro);
        assert!((f1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f1_harmonic_mean_property() {
        let y_true = vec![0, 0, 1, 1, 2, 2];
        let y_pred = vec![0, 1, 1, 0, 2, 2];

        let prec = precision(&y_pred, &y_true, Average::Micro);
        let rec = recall(&y_pred, &y_true, Average::Micro);
        let f1 = f1_score(&y_pred, &y_true, Average::Micro);

        // F1 = 2 * P * R / (P + R)
        let expected = 2.0 * prec * rec / (prec + rec);
        assert!((f1 - expected).abs() < 1e-6);
    }

    #[test]
    fn test_f1_between_precision_recall() {
        let y_true = vec![0, 0, 1, 1, 2, 2];
        let y_pred = vec![0, 1, 1, 0, 2, 1];

        let prec = precision(&y_pred, &y_true, Average::Macro);
        let rec = recall(&y_pred, &y_true, Average::Macro);
        let f1 = f1_score(&y_pred, &y_true, Average::Macro);

        // F1 should be <= min(P, R) when they differ (harmonic mean property)
        // Actually F1 <= max(P, R) always
        assert!(f1 <= prec.max(rec) + 1e-6);
    }

    // ==================== CONFUSION MATRIX TESTS ====================

    #[test]
    fn test_confusion_matrix_perfect() {
        let y_true = vec![0, 1, 2];
        let y_pred = vec![0, 1, 2];
        let cm = confusion_matrix(&y_pred, &y_true);

        assert_eq!(cm.n_rows(), 3);
        assert_eq!(cm.n_cols(), 3);

        // Diagonal should be 1s, off-diagonal 0s
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(cm.get(i, j), 1);
                } else {
                    assert_eq!(cm.get(i, j), 0);
                }
            }
        }
    }

    #[test]
    fn test_confusion_matrix_off_diagonal() {
        let y_true = vec![0, 0, 1, 1];
        let y_pred = vec![0, 1, 1, 0];
        let cm = confusion_matrix(&y_pred, &y_true);

        // True\Pred |  0  |  1  |
        // ----------+-----+-----+
        //     0     |  1  |  1  |
        //     1     |  1  |  1  |
        assert_eq!(cm.get(0, 0), 1); // TP for class 0
        assert_eq!(cm.get(0, 1), 1); // FN for class 0 (predicted 1)
        assert_eq!(cm.get(1, 0), 1); // FN for class 1 (predicted 0)
        assert_eq!(cm.get(1, 1), 1); // TP for class 1
    }

    #[test]
    fn test_confusion_matrix_sum_equals_total() {
        let y_true = vec![0, 0, 1, 1, 2, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0, 0, 1, 1];
        let cm = confusion_matrix(&y_pred, &y_true);

        let mut total: usize = 0;
        for i in 0..cm.n_rows() {
            for j in 0..cm.n_cols() {
                total += cm.get(i, j);
            }
        }

        assert_eq!(total, y_true.len());
    }

    // ==================== CLASSIFICATION REPORT TESTS ====================

    #[test]
    fn test_classification_report_format() {
        let y_true = vec![0, 0, 1, 1, 2, 2];
        let y_pred = vec![0, 1, 1, 1, 2, 0];
        let report = classification_report(&y_pred, &y_true);

        assert!(report.contains("precision"));
        assert!(report.contains("recall"));
        assert!(report.contains("f1-score"));
        assert!(report.contains("support"));
        assert!(report.contains("macro avg"));
        assert!(report.contains("weighted avg"));
    }

    #[test]
    fn test_classification_report_perfect() {
        let y_true = vec![0, 1, 2];
        let y_pred = vec![0, 1, 2];
        let report = classification_report(&y_pred, &y_true);

        // Should contain 1.00 for perfect scores
        assert!(report.contains("1.00"));
    }

    // ==================== EDGE CASES ====================

    #[test]
    fn test_single_class() {
        let y_true = vec![0, 0, 0];
        let y_pred = vec![0, 0, 0];

        let acc = accuracy(&y_pred, &y_true);
        let prec = precision(&y_pred, &y_true, Average::Macro);
        let rec = recall(&y_pred, &y_true, Average::Macro);
        let f1 = f1_score(&y_pred, &y_true, Average::Macro);

        assert!((acc - 1.0).abs() < 1e-6);
        assert!((prec - 1.0).abs() < 1e-6);
        assert!((rec - 1.0).abs() < 1e-6);
        assert!((f1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_binary_classification() {
        let y_true = vec![0, 0, 1, 1, 1, 0];
        let y_pred = vec![0, 1, 1, 1, 0, 0];

        let acc = accuracy(&y_pred, &y_true);
        assert!((acc - 4.0 / 6.0).abs() < 1e-6);

        let cm = confusion_matrix(&y_pred, &y_true);
        // TP0=2, FP0=1, FN0=1, TP1=2, FP1=1, FN1=1
        assert_eq!(cm.get(0, 0), 2); // TN for class 1 / TP for class 0
        assert_eq!(cm.get(1, 1), 2); // TP for class 1
    }

    // ==================== PROPERTY-BASED TESTS (proptest) ====================

    #[test]
    fn test_accuracy_bounds() {
        // Accuracy is always in [0, 1]
        for _ in 0..100 {
            let y_true: Vec<usize> = (0..50).map(|i| i % 5).collect();
            let y_pred: Vec<usize> = (0..50).map(|i| (i + 1) % 5).collect();
            let acc = accuracy(&y_pred, &y_true);
            assert!((0.0..=1.0).contains(&acc));
        }
    }

    #[test]
    fn test_metrics_consistency() {
        // For perfect predictions, all metrics should be 1.0
        let y_true: Vec<usize> = (0..100).map(|i| i % 10).collect();
        let y_pred = y_true.clone();

        assert!((accuracy(&y_pred, &y_true) - 1.0).abs() < 1e-6);
        assert!((precision(&y_pred, &y_true, Average::Macro) - 1.0).abs() < 1e-6);
        assert!((recall(&y_pred, &y_true, Average::Macro) - 1.0).abs() < 1e-6);
        assert!((f1_score(&y_pred, &y_true, Average::Macro) - 1.0).abs() < 1e-6);
    }
}
