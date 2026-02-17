use super::{compute_tp_fp_fn, f1_score, precision, recall, Average};
use std::fmt::Write;

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

#[cfg(test)]
mod tests {
    use super::super::{accuracy, confusion_matrix};
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

    // ==================== PER-CLASS METRIC TESTS ====================

    #[test]
    fn test_precision_per_class_perfect() {
        use super::super::precision_per_class;
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 2, 0, 1, 2];
        let per_class = precision_per_class(&y_pred, &y_true);
        assert_eq!(per_class.len(), 3);
        for &p in &per_class {
            assert!((p - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_precision_per_class_partial() {
        use super::super::precision_per_class;
        // Class 0: pred=[0,0], true=[0,1] → TP=1, FP=1 → 0.5
        // Class 1: pred=[1], true=[1] → TP=1, FP=0 → 1.0
        let y_true = vec![0, 1, 1];
        let y_pred = vec![0, 0, 1];
        let per_class = precision_per_class(&y_pred, &y_true);
        assert!((per_class[0] - 0.5).abs() < 1e-6);
        assert!((per_class[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_recall_per_class_perfect() {
        use super::super::recall_per_class;
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 2, 0, 1, 2];
        let per_class = recall_per_class(&y_pred, &y_true);
        assert_eq!(per_class.len(), 3);
        for &r in &per_class {
            assert!((r - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_recall_per_class_partial() {
        use super::super::recall_per_class;
        // Class 0: true=[0], pred=[1] → TP=0, FN=1 → 0.0
        // Class 1: true=[1,1], pred=[1,0] → TP=1, FN=1 → 0.5
        let y_true = vec![0, 1, 1];
        let y_pred = vec![1, 1, 0];
        let per_class = recall_per_class(&y_pred, &y_true);
        assert!((per_class[0] - 0.0).abs() < 1e-6);
        assert!((per_class[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_f1_per_class_perfect() {
        use super::super::f1_per_class;
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 2, 0, 1, 2];
        let per_class = f1_per_class(&y_pred, &y_true);
        assert_eq!(per_class.len(), 3);
        for &f in &per_class {
            assert!((f - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_f1_per_class_harmonic_mean() {
        use super::super::{f1_per_class, precision_per_class, recall_per_class};
        let y_true = vec![0, 0, 1, 1, 1];
        let y_pred = vec![0, 1, 1, 1, 0];
        let p = precision_per_class(&y_pred, &y_true);
        let r = recall_per_class(&y_pred, &y_true);
        let f = f1_per_class(&y_pred, &y_true);
        for i in 0..p.len() {
            if p[i] + r[i] > 0.0 {
                let expected = 2.0 * p[i] * r[i] / (p[i] + r[i]);
                assert!((f[i] - expected).abs() < 1e-6);
            } else {
                assert!((f[i] - 0.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_per_class_consistency_with_macro() {
        use super::super::{f1_per_class, precision_per_class, recall_per_class};
        let y_true = vec![0, 0, 1, 1, 2, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0];
        // Macro average should equal mean of per-class
        let p_per = precision_per_class(&y_pred, &y_true);
        let r_per = recall_per_class(&y_pred, &y_true);
        let f_per = f1_per_class(&y_pred, &y_true);
        let p_macro = precision(&y_pred, &y_true, Average::Macro);
        let r_macro = recall(&y_pred, &y_true, Average::Macro);
        let f_macro = f1_score(&y_pred, &y_true, Average::Macro);
        let p_mean: f32 = p_per.iter().sum::<f32>() / p_per.len() as f32;
        let r_mean: f32 = r_per.iter().sum::<f32>() / r_per.len() as f32;
        let f_mean: f32 = f_per.iter().sum::<f32>() / f_per.len() as f32;
        assert!((p_mean - p_macro).abs() < 1e-6);
        assert!((r_mean - r_macro).abs() < 1e-6);
        assert!((f_mean - f_macro).abs() < 1e-6);
    }
}
