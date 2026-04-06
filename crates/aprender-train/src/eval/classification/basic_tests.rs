//! Basic tests for classification metrics
//!
//! Tests for confusion matrix, multiclass metrics, and core functionality.

#[cfg(test)]
mod tests {
    use crate::eval::classification::{
        classification_report, confusion_matrix, Average, ConfusionMatrix, MultiClassMetrics,
    };

    #[test]
    fn test_confusion_matrix_basic() {
        let y_pred = vec![0, 1, 1, 2, 0, 1];
        let y_true = vec![0, 1, 0, 2, 0, 2];
        let cm = confusion_matrix(&y_pred, &y_true);

        assert_eq!(cm.n_classes(), 3);
        assert_eq!(cm.get(0, 0), 2); // True 0, predicted 0
        assert_eq!(cm.get(0, 1), 1); // True 0, predicted 1
        assert_eq!(cm.get(1, 1), 1); // True 1, predicted 1
        assert_eq!(cm.get(2, 1), 1); // True 2, predicted 1
        assert_eq!(cm.get(2, 2), 1); // True 2, predicted 2
    }

    #[test]
    fn test_confusion_matrix_perfect() {
        let y_pred = vec![0, 1, 2, 0, 1, 2];
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let cm = confusion_matrix(&y_pred, &y_true);

        assert_eq!(cm.accuracy(), 1.0);
        assert_eq!(cm.get(0, 0), 2);
        assert_eq!(cm.get(1, 1), 2);
        assert_eq!(cm.get(2, 2), 2);
    }

    #[test]
    fn test_confusion_matrix_tp_fp_fn() {
        let y_pred = vec![1, 1, 0, 1];
        let y_true = vec![1, 0, 0, 1];
        let cm = confusion_matrix(&y_pred, &y_true);

        // For class 1:
        // TP = 2 (predicted 1, was 1)
        // FP = 1 (predicted 1, was 0)
        // FN = 0 (was 1, predicted 0)
        assert_eq!(cm.true_positives(1), 2);
        assert_eq!(cm.false_positives(1), 1);
        assert_eq!(cm.false_negatives(1), 0);

        // For class 0:
        // TP = 1 (predicted 0, was 0)
        // FP = 0 (predicted 0, was 1)
        // FN = 1 (was 0, predicted 1)
        assert_eq!(cm.true_positives(0), 1);
        assert_eq!(cm.false_positives(0), 0);
        assert_eq!(cm.false_negatives(0), 1);
    }

    #[test]
    fn test_multiclass_metrics() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        // Class 0: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
        assert!((metrics.precision[0] - 0.5).abs() < 1e-6);
        assert!((metrics.recall[0] - 0.5).abs() < 1e-6);

        // Class 1: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
        assert!((metrics.precision[1] - 0.5).abs() < 1e-6);
        assert!((metrics.recall[1] - 0.5).abs() < 1e-6);

        // Class 2: TP=1, FP=0, FN=0 -> P=1, R=1, F1=1
        assert!((metrics.precision[2] - 1.0).abs() < 1e-6);
        assert!((metrics.recall[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_macro_average() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        // Macro average: (0.5 + 0.5 + 1.0) / 3 = 0.667
        let macro_f1 = metrics.f1_avg(Average::Macro);
        assert!((macro_f1 - 0.6666666).abs() < 0.01);
    }

    #[test]
    fn test_weighted_average() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        // Support: class 0 = 2, class 1 = 2, class 2 = 1
        // Weighted F1: (0.5*2 + 0.5*2 + 1.0*1) / 5 = 3/5 = 0.6
        let weighted_f1 = metrics.f1_avg(Average::Weighted);
        assert!((weighted_f1 - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_classification_report() {
        let y_pred = vec![0, 1, 1, 2, 0, 1];
        let y_true = vec![0, 1, 0, 2, 0, 2];
        let report = classification_report(&y_pred, &y_true);

        assert!(report.contains("precision"));
        assert!(report.contains("recall"));
        assert!(report.contains("f1-score"));
        assert!(report.contains("support"));
        assert!(report.contains("macro avg"));
        assert!(report.contains("weighted avg"));
        assert!(report.contains("Accuracy"));
    }

    #[test]
    fn test_empty_input() {
        let y_pred: Vec<usize> = vec![];
        let y_true: Vec<usize> = vec![];
        let cm = confusion_matrix(&y_pred, &y_true);

        assert_eq!(cm.n_classes(), 0);
        assert_eq!(cm.accuracy(), 0.0);
    }

    #[test]
    fn test_binary_classification() {
        let y_pred = vec![0, 1, 1, 0, 1, 0];
        let y_true = vec![0, 1, 0, 0, 1, 1];
        let cm = confusion_matrix(&y_pred, &y_true);

        assert_eq!(cm.n_classes(), 2);

        // Class 0: TP=2, FP=1, FN=1
        // Class 1: TP=2, FP=1, FN=1
        assert_eq!(cm.true_positives(0), 2);
        assert_eq!(cm.true_positives(1), 2);
    }

    #[test]
    fn test_support() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let cm = confusion_matrix(&y_pred, &y_true);

        // Support is count of true instances per class
        assert_eq!(cm.support(0), 2); // Two samples with true label 0
        assert_eq!(cm.support(1), 2); // Two samples with true label 1
        assert_eq!(cm.support(2), 1); // One sample with true label 2
    }

    #[test]
    fn test_display() {
        let y_pred = vec![0, 1, 0];
        let y_true = vec![0, 1, 1];
        let cm = confusion_matrix(&y_pred, &y_true);

        let display = format!("{cm}");
        assert!(display.contains("Confusion Matrix"));
        assert!(display.contains("Pred"));
        assert!(display.contains("True"));
    }

    #[test]
    fn test_average_enum_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Average::Macro);
        set.insert(Average::Micro);
        set.insert(Average::Weighted);
        set.insert(Average::None);
        set.insert(Average::Macro); // Duplicate
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_average_enum_clone_copy() {
        let avg = Average::Weighted;
        let copied = avg;
        let cloned = avg;
        assert_eq!(avg, copied);
        assert_eq!(avg, cloned);
    }

    #[test]
    fn test_confusion_matrix_labels() {
        let cm = ConfusionMatrix::new(3);
        let labels = cm.labels();
        assert_eq!(labels, &[0, 1, 2]);
    }

    #[test]
    fn test_confusion_matrix_matrix_accessor() {
        let y_pred = vec![0, 1, 0];
        let y_true = vec![0, 1, 1];
        let cm = confusion_matrix(&y_pred, &y_true);
        let matrix = cm.matrix();
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0][0], 1); // True 0, pred 0
        assert_eq!(matrix[1][0], 1); // True 1, pred 0
    }

    #[test]
    fn test_confusion_matrix_total() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let cm = confusion_matrix(&y_pred, &y_true);
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_confusion_matrix_true_negatives() {
        // Simple binary case for clarity
        let y_pred = vec![1, 1, 0, 0];
        let y_true = vec![1, 0, 0, 1];
        let cm = confusion_matrix(&y_pred, &y_true);

        // For class 0: TN = samples that are not 0 and not predicted as 0
        // True 1, pred 1: 1 case = TN for class 0
        let tn_0 = cm.true_negatives(0);
        assert_eq!(tn_0, 1);

        // For class 1: TN = samples that are not 1 and not predicted as 1
        // True 0, pred 0: 1 case = TN for class 1
        let tn_1 = cm.true_negatives(1);
        assert_eq!(tn_1, 1);
    }

    #[test]
    fn test_confusion_matrix_clone() {
        let y_pred = vec![0, 1, 0];
        let y_true = vec![0, 1, 1];
        let cm = confusion_matrix(&y_pred, &y_true);
        let cloned = cm.clone();

        assert_eq!(cm.n_classes(), cloned.n_classes());
        assert_eq!(cm.get(0, 0), cloned.get(0, 0));
    }

    #[test]
    fn test_multiclass_metrics_clone() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        let cloned = metrics.clone();

        assert_eq!(metrics.n_classes, cloned.n_classes);
        assert_eq!(metrics.precision.len(), cloned.precision.len());
    }

    // ── Confusion matrix coverage improvement tests ──────────────────

    #[test]
    fn test_confusion_matrix_from_predictions_with_min_classes() {
        // Observed classes are 0,1 but min_classes forces 4
        let y_pred = vec![0, 1, 0, 1];
        let y_true = vec![0, 1, 1, 0];
        let cm = ConfusionMatrix::from_predictions_with_min_classes(&y_pred, &y_true, 4);
        assert_eq!(cm.n_classes(), 4);
        // Classes 2,3 should have zero counts everywhere
        assert_eq!(cm.support(2), 0);
        assert_eq!(cm.support(3), 0);
        assert_eq!(cm.true_positives(2), 0);
        assert_eq!(cm.false_positives(2), 0);
        assert_eq!(cm.false_negatives(2), 0);
    }

    #[test]
    fn test_confusion_matrix_from_predictions_with_min_classes_lower_than_observed() {
        // min_classes < observed → observed wins
        let y_pred = vec![0, 1, 2];
        let y_true = vec![0, 1, 2];
        let cm = ConfusionMatrix::from_predictions_with_min_classes(&y_pred, &y_true, 1);
        assert_eq!(cm.n_classes(), 3); // observed = 3, min_classes = 1 → 3
    }

    #[test]
    fn test_confusion_matrix_from_predictions_with_min_classes_zero() {
        let y_pred = vec![0, 1];
        let y_true = vec![0, 1];
        let cm = ConfusionMatrix::from_predictions_with_min_classes(&y_pred, &y_true, 0);
        assert_eq!(cm.n_classes(), 2);
    }

    #[test]
    fn test_confusion_matrix_single_class() {
        let y_pred = vec![0, 0, 0];
        let y_true = vec![0, 0, 0];
        let cm = confusion_matrix(&y_pred, &y_true);
        assert_eq!(cm.n_classes(), 1);
        assert_eq!(cm.true_positives(0), 3);
        assert_eq!(cm.false_positives(0), 0);
        assert_eq!(cm.false_negatives(0), 0);
        assert_eq!(cm.true_negatives(0), 0); // total - tp - fp - fn = 3 - 3 - 0 - 0 = 0
        assert_eq!(cm.accuracy(), 1.0);
        assert_eq!(cm.total(), 3);
    }

    #[test]
    fn test_confusion_matrix_new_constructor() {
        let cm = ConfusionMatrix::new(5);
        assert_eq!(cm.n_classes(), 5);
        assert_eq!(cm.labels(), &[0, 1, 2, 3, 4]);
        assert_eq!(cm.total(), 0);
        assert_eq!(cm.accuracy(), 0.0);
        // All cells should be zero
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(cm.get(i, j), 0);
            }
        }
    }

    #[test]
    fn test_confusion_matrix_new_zero_classes() {
        let cm = ConfusionMatrix::new(0);
        assert_eq!(cm.n_classes(), 0);
        assert!(cm.labels().is_empty());
        assert_eq!(cm.total(), 0);
        assert_eq!(cm.accuracy(), 0.0);
    }

    #[test]
    fn test_confusion_matrix_display_formatting() {
        let cm = ConfusionMatrix::new(2);
        let display = format!("{cm}");
        assert!(display.contains("Confusion Matrix:"));
        assert!(display.contains("Pred 0"));
        assert!(display.contains("Pred 1"));
        assert!(display.contains("True 0"));
        assert!(display.contains("True 1"));
    }

    #[test]
    fn test_confusion_matrix_display_with_data() {
        let y_pred = vec![0, 1, 0, 1];
        let y_true = vec![0, 0, 1, 1];
        let cm = confusion_matrix(&y_pred, &y_true);
        let display = format!("{cm}");
        // Should contain the counts in the display
        assert!(display.contains('1'), "Display should contain count values");
    }

    #[test]
    fn test_confusion_matrix_true_negatives_multiclass() {
        // 3-class: perfect predictions
        let y_pred = vec![0, 1, 2, 0, 1, 2];
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let cm = confusion_matrix(&y_pred, &y_true);

        // For class 0: TN = total - TP(0) - FP(0) - FN(0) = 6 - 2 - 0 - 0 = 4
        assert_eq!(cm.true_negatives(0), 4);
        assert_eq!(cm.true_negatives(1), 4);
        assert_eq!(cm.true_negatives(2), 4);
    }

    #[test]
    fn test_confusion_matrix_support_sums_to_total() {
        let y_pred = vec![0, 1, 2, 0, 1];
        let y_true = vec![0, 1, 0, 2, 1];
        let cm = confusion_matrix(&y_pred, &y_true);
        let total_support: usize = (0..cm.n_classes()).map(|c| cm.support(c)).sum();
        assert_eq!(total_support, cm.total());
    }

    #[test]
    fn test_confusion_matrix_accuracy_all_wrong() {
        // Every prediction is wrong
        let y_pred = vec![1, 0, 1, 0];
        let y_true = vec![0, 1, 0, 1];
        let cm = confusion_matrix(&y_pred, &y_true);
        assert!((cm.accuracy() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_confusion_matrix_debug() {
        let cm = ConfusionMatrix::new(2);
        let debug = format!("{cm:?}");
        assert!(debug.contains("ConfusionMatrix"));
    }

    #[test]
    fn test_confusion_matrix_from_predictions_empty() {
        let y_pred: Vec<usize> = vec![];
        let y_true: Vec<usize> = vec![];
        let cm = ConfusionMatrix::from_predictions(&y_pred, &y_true);
        assert_eq!(cm.n_classes(), 0);
        assert_eq!(cm.total(), 0);
    }

    #[test]
    fn test_confusion_matrix_from_predictions_with_min_classes_empty_data() {
        let cm = ConfusionMatrix::from_predictions_with_min_classes(&[], &[], 3);
        assert_eq!(cm.n_classes(), 3);
        assert_eq!(cm.total(), 0);
        assert_eq!(cm.accuracy(), 0.0);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_confusion_matrix_mismatched_lengths() {
        let y_pred = vec![0, 1];
        let y_true = vec![0];
        let _ = ConfusionMatrix::from_predictions(&y_pred, &y_true);
    }

    // ── MultiClassMetrics coverage improvement ──────────────────────

    #[test]
    fn test_multiclass_metrics_empty_predictions() {
        let metrics = MultiClassMetrics::from_predictions(&[], &[]);
        assert_eq!(metrics.n_classes, 0);
        assert!(metrics.precision.is_empty());
        assert!(metrics.recall.is_empty());
        assert!(metrics.f1.is_empty());
        assert!(metrics.support.is_empty());
    }

    #[test]
    fn test_multiclass_metrics_single_class() {
        let y_pred = vec![0, 0, 0];
        let y_true = vec![0, 0, 0];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        assert_eq!(metrics.n_classes, 1);
        assert!((metrics.precision[0] - 1.0).abs() < 1e-6);
        assert!((metrics.recall[0] - 1.0).abs() < 1e-6);
        assert!((metrics.f1[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiclass_metrics_zero_tp_class() {
        // Class 2 has zero TP, zero FP → precision = 0.0
        let y_pred = vec![0, 1, 0, 1];
        let y_true = vec![0, 1, 2, 2];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        assert!(
            (metrics.precision[2] - 0.0).abs() < 1e-6,
            "Class with no predictions should have precision 0"
        );
        assert!(
            (metrics.recall[2] - 0.0).abs() < 1e-6,
            "Class never correctly predicted should have recall 0"
        );
        assert!(
            (metrics.f1[2] - 0.0).abs() < 1e-6,
            "F1 should be 0 when precision and recall are 0"
        );
    }

    #[test]
    fn test_multiclass_metrics_micro_average() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        // Micro currently falls back to macro
        let micro_f1 = metrics.f1_avg(Average::Micro);
        let macro_f1 = metrics.f1_avg(Average::Macro);
        assert!((micro_f1 - macro_f1).abs() < 1e-6, "Micro should fall back to macro");
    }

    #[test]
    fn test_multiclass_metrics_none_average() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        // None currently falls back to macro
        let none_f1 = metrics.f1_avg(Average::None);
        let macro_f1 = metrics.f1_avg(Average::Macro);
        assert!((none_f1 - macro_f1).abs() < 1e-6, "None should fall back to macro");
    }

    #[test]
    fn test_multiclass_metrics_weighted_all_same_support() {
        // When all classes have equal support, weighted = macro
        let y_pred = vec![0, 1, 2, 1, 0, 2];
        let y_true = vec![0, 1, 2, 0, 1, 2]; // 2 per class
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        let weighted_f1 = metrics.f1_avg(Average::Weighted);
        let macro_f1 = metrics.f1_avg(Average::Macro);
        assert!(
            (weighted_f1 - macro_f1).abs() < 1e-6,
            "Weighted and macro should match when all classes have equal support"
        );
    }

    #[test]
    fn test_multiclass_metrics_weighted_zero_support() {
        // All zero → weighted returns 0.0
        let metrics = MultiClassMetrics::from_predictions(&[], &[]);
        let weighted = metrics.f1_avg(Average::Weighted);
        assert!((weighted - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiclass_metrics_precision_avg_macro_empty() {
        let metrics = MultiClassMetrics::from_predictions(&[], &[]);
        assert!((metrics.precision_avg(Average::Macro) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiclass_metrics_recall_avg() {
        let y_pred = vec![0, 1, 0];
        let y_true = vec![0, 0, 1];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        let recall_macro = metrics.recall_avg(Average::Macro);
        let recall_weighted = metrics.recall_avg(Average::Weighted);
        // Both should be valid f64
        assert!(recall_macro.is_finite());
        assert!(recall_weighted.is_finite());
    }

    #[test]
    fn test_multiclass_metrics_from_confusion_matrix() {
        let cm = ConfusionMatrix::new(3);
        let metrics = MultiClassMetrics::from_confusion_matrix(&cm);
        assert_eq!(metrics.n_classes, 3);
        // All zeros → all metrics 0.0
        for i in 0..3 {
            assert!((metrics.precision[i] - 0.0).abs() < 1e-6);
            assert!((metrics.recall[i] - 0.0).abs() < 1e-6);
            assert!((metrics.f1[i] - 0.0).abs() < 1e-6);
            assert_eq!(metrics.support[i], 0);
        }
    }

    // ── Average enum coverage ──────────────────────────────────────

    #[test]
    fn test_average_enum_debug() {
        assert_eq!(format!("{:?}", Average::Macro), "Macro");
        assert_eq!(format!("{:?}", Average::Micro), "Micro");
        assert_eq!(format!("{:?}", Average::Weighted), "Weighted");
        assert_eq!(format!("{:?}", Average::None), "None");
    }

    // ── Classification report coverage ──────────────────────────────

    #[test]
    fn test_classification_report_single_class() {
        let y_pred = vec![0, 0, 0];
        let y_true = vec![0, 0, 0];
        let report = classification_report(&y_pred, &y_true);
        assert!(report.contains("precision"));
        assert!(report.contains("Accuracy: 1.0"));
    }

    #[test]
    fn test_classification_report_binary() {
        let y_pred = vec![0, 1, 0, 1];
        let y_true = vec![0, 1, 1, 0];
        let report = classification_report(&y_pred, &y_true);
        assert!(report.contains("Class 0"));
        assert!(report.contains("Class 1"));
        assert!(report.contains("macro avg"));
        assert!(report.contains("weighted avg"));
        assert!(report.contains("Accuracy: 0.5"));
    }

    // =========================================================================
    // Additional coverage tests for ConfusionMatrix
    // =========================================================================

    #[test]
    fn test_confusion_matrix_display_format() {
        let y_pred = vec![0, 1, 1];
        let y_true = vec![0, 1, 0];
        let cm = confusion_matrix(&y_pred, &y_true);
        let display = format!("{cm}");
        assert!(display.contains("Confusion Matrix:"));
        assert!(display.contains("Pred 0"));
        assert!(display.contains("True 0"));
    }

    #[test]
    fn test_confusion_matrix_support_per_class() {
        let y_pred = vec![0, 0, 1, 1, 1];
        let y_true = vec![0, 0, 0, 1, 1];
        let cm = confusion_matrix(&y_pred, &y_true);

        // Support is total true instances per class
        assert_eq!(cm.support(0), 3); // 3 true class-0 samples
        assert_eq!(cm.support(1), 2); // 2 true class-1 samples
    }

    #[test]
    fn test_confusion_matrix_accuracy_empty_matrix() {
        let cm = ConfusionMatrix::new(2);
        assert_eq!(cm.accuracy(), 0.0);
    }

    #[test]
    fn test_confusion_matrix_raw_matrix_access() {
        let cm = ConfusionMatrix::new(2);
        let mat = cm.matrix();
        assert_eq!(mat.len(), 2);
        assert_eq!(mat[0], vec![0, 0]);
        assert_eq!(mat[1], vec![0, 0]);
    }
}
