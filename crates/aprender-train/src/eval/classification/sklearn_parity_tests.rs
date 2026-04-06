//! sklearn parity tests for classification metrics
//!
//! These tests verify that our metrics match sklearn reference values
//! to within 1e-6 precision.
//!
//! Reference values computed with sklearn 1.4.0:
//! ```python
//! from sklearn.metrics import (accuracy_score, precision_score,
//!                              recall_score, f1_score, confusion_matrix)
//! ```

#[cfg(test)]
mod tests {
    use crate::eval::classification::{
        confusion_matrix, Average, ConfusionMatrix, MultiClassMetrics,
    };

    #[test]
    fn test_sklearn_parity_accuracy() {
        // sklearn: accuracy_score([0, 0, 1, 1, 2, 2, 0, 1, 2],
        //                         [0, 1, 1, 2, 2, 0, 0, 1, 2]) = 0.6666666666666666
        let y_true = vec![0, 0, 1, 1, 2, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0, 0, 1, 2];

        let cm = confusion_matrix(&y_pred, &y_true);
        let acc = cm.accuracy();

        assert!(
            (acc - 0.6666666666666666).abs() < 1e-6,
            "Accuracy {acc} does not match sklearn reference 0.6666666666666666"
        );
    }

    #[test]
    fn test_sklearn_parity_precision_macro() {
        // sklearn: precision_score(..., average='macro') = 0.6666666666666666
        let y_true = vec![0, 0, 1, 1, 2, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0, 0, 1, 2];

        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        let p_macro = metrics.precision_avg(Average::Macro);

        assert!(
            (p_macro - 0.6666666666666666).abs() < 1e-6,
            "Macro precision {p_macro} does not match sklearn reference"
        );
    }

    #[test]
    fn test_sklearn_parity_recall_macro() {
        // sklearn: recall_score(..., average='macro') = 0.6666666666666666
        let y_true = vec![0, 0, 1, 1, 2, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0, 0, 1, 2];

        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        let r_macro = metrics.recall_avg(Average::Macro);

        assert!(
            (r_macro - 0.6666666666666666).abs() < 1e-6,
            "Macro recall {r_macro} does not match sklearn reference"
        );
    }

    #[test]
    fn test_sklearn_parity_f1_macro() {
        // sklearn: f1_score(..., average='macro') = 0.6666666666666666
        let y_true = vec![0, 0, 1, 1, 2, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0, 0, 1, 2];

        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        let f1_macro = metrics.f1_avg(Average::Macro);

        assert!(
            (f1_macro - 0.6666666666666666).abs() < 1e-6,
            "Macro F1 {f1_macro} does not match sklearn reference"
        );
    }

    #[test]
    fn test_sklearn_parity_micro_averages() {
        // sklearn: For this dataset, micro = macro = 0.6666666666666666
        let y_true = vec![0, 0, 1, 1, 2, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0, 0, 1, 2];

        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        let p_micro = metrics.precision_avg(Average::Micro);
        let r_micro = metrics.recall_avg(Average::Micro);
        let f1_micro = metrics.f1_avg(Average::Micro);

        assert!(
            (p_micro - 0.6666666666666666).abs() < 1e-6,
            "Micro precision {p_micro} does not match sklearn reference"
        );
        assert!(
            (r_micro - 0.6666666666666666).abs() < 1e-6,
            "Micro recall {r_micro} does not match sklearn reference"
        );
        assert!(
            (f1_micro - 0.6666666666666666).abs() < 1e-6,
            "Micro F1 {f1_micro} does not match sklearn reference"
        );
    }

    #[test]
    fn test_sklearn_parity_weighted_averages() {
        // sklearn: weighted averages = 0.6666666666666666 for balanced classes
        let y_true = vec![0, 0, 1, 1, 2, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 1, 2, 2, 0, 0, 1, 2];

        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        let p_weighted = metrics.precision_avg(Average::Weighted);
        let r_weighted = metrics.recall_avg(Average::Weighted);
        let f1_weighted = metrics.f1_avg(Average::Weighted);

        assert!(
            (p_weighted - 0.6666666666666666).abs() < 1e-6,
            "Weighted precision {p_weighted} does not match sklearn reference"
        );
        assert!(
            (r_weighted - 0.6666666666666666).abs() < 1e-6,
            "Weighted recall {r_weighted} does not match sklearn reference"
        );
        assert!(
            (f1_weighted - 0.6666666666666666).abs() < 1e-6,
            "Weighted F1 {f1_weighted} does not match sklearn reference"
        );
    }

    #[test]
    fn test_sklearn_parity_imbalanced() {
        // Test with imbalanced classes
        // y_true = [0, 0, 0, 0, 0, 1, 1, 2]
        // y_pred = [0, 0, 0, 1, 1, 1, 0, 2]
        //
        // Manual calculation:
        // Class 0: TP=3, FP=1, FN=2 -> P=0.75, R=0.6, F1=0.6667
        // Class 1: TP=1, FP=2, FN=1 -> P=0.333, R=0.5, F1=0.4
        // Class 2: TP=1, FP=0, FN=0 -> P=1, R=1, F1=1
        // Macro F1 = (0.6667 + 0.4 + 1.0) / 3 = 0.6889
        let y_true = vec![0, 0, 0, 0, 0, 1, 1, 2];
        let y_pred = vec![0, 0, 0, 1, 1, 1, 0, 2];

        let cm = confusion_matrix(&y_pred, &y_true);
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        // Accuracy: 5 correct out of 8
        assert!(
            (cm.accuracy() - 0.625).abs() < 1e-6,
            "Accuracy {} does not match reference 0.625",
            cm.accuracy()
        );

        // Macro F1: (0.6667 + 0.4 + 1.0) / 3 = 0.6889
        let f1_macro = metrics.f1_avg(Average::Macro);
        assert!(
            (f1_macro - 0.6888888888888888).abs() < 1e-6,
            "Macro F1 {f1_macro} does not match reference 0.6889"
        );
    }

    #[test]
    fn test_sklearn_parity_binary() {
        // Binary classification parity
        // y_true = [0, 0, 1, 1, 0, 1, 0, 1]
        // y_pred = [0, 1, 1, 0, 0, 1, 1, 1]
        //
        // Manual calculation:
        // Class 0: TP=2, FP=1, FN=2 -> P=2/3=0.6667, R=2/4=0.5, F1=0.5714
        // Class 1: TP=3, FP=2, FN=1 -> P=3/5=0.6, R=3/4=0.75, F1=0.6667
        // Macro F1 = (0.5714 + 0.6667) / 2 = 0.6190
        let y_true = vec![0, 0, 1, 1, 0, 1, 0, 1];
        let y_pred = vec![0, 1, 1, 0, 0, 1, 1, 1];

        let cm = confusion_matrix(&y_pred, &y_true);
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        // Accuracy: 5 correct out of 8
        assert!(
            (cm.accuracy() - 0.625).abs() < 1e-6,
            "Accuracy {} does not match reference 0.625",
            cm.accuracy()
        );

        // Macro F1: (0.5714 + 0.6667) / 2 = 0.6190
        let f1_macro = metrics.f1_avg(Average::Macro);
        assert!(
            (f1_macro - 0.6190476190476191).abs() < 1e-6,
            "Macro F1 {f1_macro} does not match reference 0.6190"
        );
    }

    #[test]
    fn test_multiclass_metrics_average_none() {
        let y_pred = vec![0, 1, 1, 2, 0];
        let y_true = vec![0, 1, 0, 2, 1];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        // Average::None should behave like Macro (returns single value)
        let f1_none = metrics.f1_avg(Average::None);
        let f1_macro = metrics.f1_avg(Average::Macro);
        assert!((f1_none - f1_macro).abs() < 1e-6);
    }

    #[test]
    fn test_multiclass_metrics_empty_support() {
        // Test with all predictions correct - verify weighted average doesn't panic with 0 support
        let y_pred = vec![0, 0, 0];
        let y_true = vec![0, 0, 0];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);

        let weighted = metrics.f1_avg(Average::Weighted);
        assert!((weighted - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_confusion_matrix_from_confusion_matrix() {
        let cm = ConfusionMatrix::new(3);
        let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

        // Empty matrix - all metrics should be 0
        assert_eq!(metrics.n_classes, 3);
        for i in 0..3 {
            assert!((metrics.precision[i] - 0.0).abs() < 1e-6);
            assert!((metrics.recall[i] - 0.0).abs() < 1e-6);
            assert!((metrics.f1[i] - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_macro_average_empty_values() {
        let metrics = MultiClassMetrics {
            precision: vec![],
            recall: vec![],
            f1: vec![],
            support: vec![],
            n_classes: 0,
        };

        assert!((metrics.precision_avg(Average::Macro) - 0.0).abs() < 1e-6);
        assert!((metrics.recall_avg(Average::Macro) - 0.0).abs() < 1e-6);
        assert!((metrics.f1_avg(Average::Macro) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_average_zero_support() {
        let metrics = MultiClassMetrics {
            precision: vec![0.5, 0.5],
            recall: vec![0.5, 0.5],
            f1: vec![0.5, 0.5],
            support: vec![0, 0], // Zero support
            n_classes: 2,
        };

        // Zero total support should return 0
        assert!((metrics.precision_avg(Average::Weighted) - 0.0).abs() < 1e-6);
    }
}
