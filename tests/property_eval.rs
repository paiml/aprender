//! Property tests for Model Evaluation Framework (APR-073)
//!
//! Ensures evaluation metrics satisfy mathematical invariants:
//! - Metrics bounded to [0, 1]
//! - No NaN or Infinity values
//! - Consistent results with empty/edge cases
//! - Confusion matrix invariants

use entrenar::eval::{
    classification_report, confusion_matrix, Average, DriftDetector, DriftTest, EvalConfig, Metric,
    ModelEvaluator, MultiClassMetrics,
};
use proptest::collection::vec;
use proptest::prelude::*;

// =============================================================================
// Strategy Helpers
// =============================================================================

/// Generate a vector of class labels in range [0, n_classes)
fn class_labels(
    n_classes: usize,
    len: impl Into<proptest::collection::SizeRange>,
) -> impl Strategy<Value = Vec<usize>> {
    vec(0..n_classes, len)
}

/// Generate pair of prediction/true labels with same length
fn label_pair(
    n_classes: usize,
    len: std::ops::Range<usize>,
) -> impl Strategy<Value = (Vec<usize>, Vec<usize>)> {
    len.prop_flat_map(move |l| (vec(0..n_classes, l), vec(0..n_classes, l)))
}

// =============================================================================
// Classification Metric Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    // -------------------------------------------------------------------------
    // Accuracy Properties
    // -------------------------------------------------------------------------

    #[test]
    fn prop_accuracy_bounded(
        (y_pred, y_true) in label_pair(5, 10..100)
    ) {
        let cm = confusion_matrix(&y_pred, &y_true);
        let acc = cm.accuracy();

        prop_assert!(
            (0.0..=1.0).contains(&acc),
            "Accuracy {} not in [0, 1]",
            acc
        );
        prop_assert!(
            !acc.is_nan() && !acc.is_infinite(),
            "Accuracy {} is NaN or Inf",
            acc
        );
    }

    #[test]
    fn prop_accuracy_perfect_predictions(
        y in class_labels(5, 10..100)
    ) {
        let cm = confusion_matrix(&y, &y);
        let acc = cm.accuracy();

        prop_assert!(
            (acc - 1.0).abs() < 1e-6,
            "Perfect predictions should have accuracy 1.0, got {}",
            acc
        );
    }

    // -------------------------------------------------------------------------
    // Precision/Recall/F1 Properties
    // -------------------------------------------------------------------------

    #[test]
    fn prop_precision_bounded(
        (y_pred, y_true) in label_pair(5, 10..100)
    ) {
        let cm = confusion_matrix(&y_pred, &y_true);
        let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

        for avg in [Average::Macro, Average::Micro, Average::Weighted] {
            let p = metrics.precision_avg(avg);
            prop_assert!(
                (0.0..=1.0).contains(&p),
                "Precision({:?}) {} not in [0, 1]",
                avg, p
            );
            prop_assert!(
                !p.is_nan() && !p.is_infinite(),
                "Precision({:?}) {} is NaN or Inf",
                avg, p
            );
        }
    }

    #[test]
    fn prop_recall_bounded(
        (y_pred, y_true) in label_pair(5, 10..100)
    ) {
        let cm = confusion_matrix(&y_pred, &y_true);
        let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

        for avg in [Average::Macro, Average::Micro, Average::Weighted] {
            let r = metrics.recall_avg(avg);
            prop_assert!(
                (0.0..=1.0).contains(&r),
                "Recall({:?}) {} not in [0, 1]",
                avg, r
            );
            prop_assert!(
                !r.is_nan() && !r.is_infinite(),
                "Recall({:?}) {} is NaN or Inf",
                avg, r
            );
        }
    }

    #[test]
    fn prop_f1_bounded(
        (y_pred, y_true) in label_pair(5, 10..100)
    ) {
        let cm = confusion_matrix(&y_pred, &y_true);
        let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

        for avg in [Average::Macro, Average::Micro, Average::Weighted] {
            let f1 = metrics.f1_avg(avg);
            prop_assert!(
                (0.0..=1.0).contains(&f1),
                "F1({:?}) {} not in [0, 1]",
                avg, f1
            );
            prop_assert!(
                !f1.is_nan() && !f1.is_infinite(),
                "F1({:?}) {} is NaN or Inf",
                avg, f1
            );
        }
    }

    #[test]
    fn prop_f1_consistent_across_averages(
        (y_pred, y_true) in label_pair(5, 10..100)
    ) {
        // F1 should be consistent: all averaging methods produce valid results
        let cm = confusion_matrix(&y_pred, &y_true);
        let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

        let f1_macro = metrics.f1_avg(Average::Macro);
        let f1_micro = metrics.f1_avg(Average::Micro);
        let f1_weighted = metrics.f1_avg(Average::Weighted);

        // All should be in [0, 1] and not NaN
        for (name, f1) in [("Macro", f1_macro), ("Micro", f1_micro), ("Weighted", f1_weighted)] {
            prop_assert!(
                (0.0..=1.0).contains(&f1),
                "F1({}) {} not in [0, 1]",
                name, f1
            );
            prop_assert!(
                !f1.is_nan() && !f1.is_infinite(),
                "F1({}) {} is NaN or Inf",
                name, f1
            );
        }
    }

    // -------------------------------------------------------------------------
    // Confusion Matrix Properties
    // -------------------------------------------------------------------------

    #[test]
    fn prop_confusion_matrix_sum_equals_samples(
        (y_pred, y_true) in label_pair(5, 10..100)
    ) {
        let cm = confusion_matrix(&y_pred, &y_true);
        let total: usize = cm.matrix().iter().flat_map(|row| row.iter()).sum();

        prop_assert_eq!(
            total,
            y_true.len(),
            "Confusion matrix total {} != sample count {}",
            total,
            y_true.len()
        );
    }

    #[test]
    fn prop_confusion_matrix_diagonal_counts_correct(
        y in class_labels(5, 10..100)
    ) {
        // For perfect predictions, diagonal should sum to len
        let cm = confusion_matrix(&y, &y);
        let diagonal_sum: usize = (0..cm.n_classes())
            .map(|i| cm.matrix()[i][i])
            .sum();

        prop_assert_eq!(
            diagonal_sum,
            y.len(),
            "Diagonal sum {} != sample count {} for perfect predictions",
            diagonal_sum,
            y.len()
        );
    }

    // -------------------------------------------------------------------------
    // Classification Report Properties
    // -------------------------------------------------------------------------

    #[test]
    fn prop_classification_report_no_panic(
        (y_pred, y_true) in label_pair(5, 1..50)
    ) {
        // Just verify no panics
        let report = classification_report(&y_pred, &y_true);
        prop_assert!(!report.is_empty());
    }

    // -------------------------------------------------------------------------
    // ModelEvaluator Properties
    // -------------------------------------------------------------------------

    #[test]
    fn prop_evaluator_all_metrics_valid(
        (y_pred, y_true) in label_pair(5, 10..100)
    ) {
        let config = EvalConfig {
            metrics: vec![
                Metric::Accuracy,
                Metric::Precision(Average::Macro),
                Metric::Precision(Average::Micro),
                Metric::Precision(Average::Weighted),
                Metric::Recall(Average::Macro),
                Metric::Recall(Average::Micro),
                Metric::Recall(Average::Weighted),
                Metric::F1(Average::Macro),
                Metric::F1(Average::Micro),
                Metric::F1(Average::Weighted),
            ],
            ..Default::default()
        };

        let evaluator = ModelEvaluator::new(config);
        let result = evaluator.evaluate_classification("test", &y_pred, &y_true).expect("operation should succeed");

        for (metric, &score) in &result.scores {
            prop_assert!(
                (0.0..=1.0).contains(&score),
                "Metric {:?} = {} not in [0, 1]",
                metric, score
            );
            prop_assert!(
                !score.is_nan() && !score.is_infinite(),
                "Metric {:?} = {} is NaN or Inf",
                metric, score
            );
        }
    }
}

// =============================================================================
// Drift Detection Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    #[test]
    fn prop_ks_statistic_bounded(
        baseline in vec(0.0f64..100.0, 50..100),
        current in vec(0.0f64..100.0, 50..100)
    ) {
        let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);

        let baseline_data: Vec<Vec<f64>> = baseline.iter().map(|&v| vec![v]).collect();
        let current_data: Vec<Vec<f64>> = current.iter().map(|&v| vec![v]).collect();

        detector.set_baseline(&baseline_data);
        let results = detector.check(&current_data);

        for result in results {
            prop_assert!(
                result.statistic >= 0.0 && result.statistic <= 1.0,
                "KS statistic {} not in [0, 1]",
                result.statistic
            );
        }
    }

    #[test]
    fn prop_psi_non_negative(
        baseline in vec(0.0f64..100.0, 50..100),
        current in vec(0.0f64..100.0, 50..100)
    ) {
        let mut detector = DriftDetector::new(vec![DriftTest::PSI { threshold: 0.2 }]);

        let baseline_data: Vec<Vec<f64>> = baseline.iter().map(|&v| vec![v]).collect();
        let current_data: Vec<Vec<f64>> = current.iter().map(|&v| vec![v]).collect();

        detector.set_baseline(&baseline_data);
        let results = detector.check(&current_data);

        for result in results {
            prop_assert!(
                result.statistic >= 0.0,
                "PSI {} should be non-negative",
                result.statistic
            );
            prop_assert!(
                !result.statistic.is_nan() && !result.statistic.is_infinite(),
                "PSI {} is NaN or Inf",
                result.statistic
            );
        }
    }

    #[test]
    fn prop_identical_distributions_low_ks(
        data in vec(0.0f64..100.0, 50..100)
    ) {
        let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);

        let data_vecs: Vec<Vec<f64>> = data.iter().map(|&v| vec![v]).collect();

        detector.set_baseline(&data_vecs);
        let results = detector.check(&data_vecs);

        for result in results {
            // Same distribution should have KS statistic near 0
            prop_assert!(
                result.statistic < 0.5,
                "KS statistic {} too high for identical distributions",
                result.statistic
            );
        }
    }

    #[test]
    fn prop_identical_distributions_low_psi(
        data in vec(0.0f64..100.0, 50..100)
    ) {
        let mut detector = DriftDetector::new(vec![DriftTest::PSI { threshold: 0.2 }]);

        let data_vecs: Vec<Vec<f64>> = data.iter().map(|&v| vec![v]).collect();

        detector.set_baseline(&data_vecs);
        let results = detector.check(&data_vecs);

        for result in results {
            // Same distribution should have PSI near 0
            prop_assert!(
                result.statistic < 0.1,
                "PSI {} too high for identical distributions",
                result.statistic
            );
        }
    }
}

// =============================================================================
// Edge Case Tests (Not proptest but important coverage)
// =============================================================================

#[test]
fn test_single_class_metrics() {
    // All same class
    let y = vec![0, 0, 0, 0, 0];
    let cm = confusion_matrix(&y, &y);
    let acc = cm.accuracy();
    assert!((acc - 1.0).abs() < 1e-6);
}

#[test]
fn test_binary_classification_metrics() {
    let y_pred = vec![0, 1, 1, 0, 1, 0, 1, 1];
    let y_true = vec![0, 1, 0, 0, 1, 1, 1, 0];

    let cm = confusion_matrix(&y_pred, &y_true);
    let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

    // All metrics should be valid
    assert!(cm.accuracy() >= 0.0 && cm.accuracy() <= 1.0);
    assert!(metrics.precision_avg(Average::Macro) >= 0.0);
    assert!(metrics.recall_avg(Average::Macro) >= 0.0);
    assert!(metrics.f1_avg(Average::Macro) >= 0.0);
}

#[test]
fn test_highly_imbalanced_classes() {
    // 99 samples of class 0, 1 sample of class 1
    let mut y_pred = vec![0; 99];
    y_pred.push(0); // Predict 0 for the class 1 sample
    let mut y_true = vec![0; 99];
    y_true.push(1);

    let cm = confusion_matrix(&y_pred, &y_true);
    let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

    // Should not panic or produce NaN
    let acc = cm.accuracy();
    assert!((acc - 0.99).abs() < 0.01);

    let p = metrics.precision_avg(Average::Weighted);
    let r = metrics.recall_avg(Average::Weighted);
    let f1 = metrics.f1_avg(Average::Weighted);

    assert!(!p.is_nan());
    assert!(!r.is_nan());
    assert!(!f1.is_nan());
}
