//! Tests for evaluation metrics

use crate::Tensor;

use super::{Accuracy, F1Score, Metric, Precision, R2Score, Recall, MAE, RMSE};

#[test]
fn test_accuracy_perfect() {
    let metric = Accuracy::default();
    let pred = Tensor::from_vec(vec![0.9, 0.1, 0.8, 0.2], false);
    let target = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], false);

    let acc = metric.compute(&pred, &target);
    assert!((acc - 1.0).abs() < 1e-5);
}

#[test]
fn test_accuracy_half() {
    let metric = Accuracy::default();
    let pred = Tensor::from_vec(vec![0.9, 0.9, 0.1, 0.1], false);
    let target = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], false);

    let acc = metric.compute(&pred, &target);
    assert!((acc - 0.5).abs() < 1e-5);
}

#[test]
fn test_precision() {
    let metric = Precision::default();
    // 2 predicted positives (0.9, 0.8), 1 TP (0.9 -> 1.0)
    let pred = Tensor::from_vec(vec![0.9, 0.8, 0.2], false);
    let target = Tensor::from_vec(vec![1.0, 0.0, 0.0], false);

    let prec = metric.compute(&pred, &target);
    assert!((prec - 0.5).abs() < 1e-5);
}

#[test]
fn test_recall() {
    let metric = Recall::default();
    // 2 actual positives, 1 correctly predicted
    let pred = Tensor::from_vec(vec![0.9, 0.2, 0.8], false);
    let target = Tensor::from_vec(vec![1.0, 1.0, 0.0], false);

    let rec = metric.compute(&pred, &target);
    assert!((rec - 0.5).abs() < 1e-5);
}

#[test]
fn test_f1_score() {
    let metric = F1Score::default();
    let pred = Tensor::from_vec(vec![0.9, 0.8, 0.2, 0.1], false);
    let target = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], false);

    let f1 = metric.compute(&pred, &target);
    // Precision = 0.5 (1/2), Recall = 0.5 (1/2)
    // F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
    assert!((f1 - 0.5).abs() < 1e-5);
}

#[test]
fn test_f1_perfect() {
    let metric = F1Score::default();
    let pred = Tensor::from_vec(vec![0.9, 0.1], false);
    let target = Tensor::from_vec(vec![1.0, 0.0], false);

    let f1 = metric.compute(&pred, &target);
    assert!((f1 - 1.0).abs() < 1e-5);
}

#[test]
fn test_r2_perfect() {
    let metric = R2Score;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

    let r2 = metric.compute(&pred, &target);
    assert!((r2 - 1.0).abs() < 1e-5);
}

#[test]
fn test_r2_mean_prediction() {
    let metric = R2Score;
    // Predicting mean of targets
    let pred = Tensor::from_vec(vec![2.0, 2.0, 2.0], false);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

    let r2 = metric.compute(&pred, &target);
    assert!(r2.abs() < 1e-5); // R² ≈ 0
}

#[test]
fn test_mae() {
    let metric = MAE;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);

    let mae = metric.compute(&pred, &target);
    assert!((mae - 0.5).abs() < 1e-5);
}

#[test]
fn test_mae_perfect() {
    let metric = MAE;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

    let mae = metric.compute(&pred, &target);
    assert!(mae < 1e-5);
}

#[test]
fn test_rmse() {
    let metric = RMSE;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    let target = Tensor::from_vec(vec![2.0, 3.0, 4.0], false);

    let rmse = metric.compute(&pred, &target);
    // MSE = mean([1, 1, 1]) = 1, RMSE = 1
    assert!((rmse - 1.0).abs() < 1e-5);
}

#[test]
fn test_rmse_perfect() {
    let metric = RMSE;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);

    let rmse = metric.compute(&pred, &target);
    assert!(rmse < 1e-5);
}

#[test]
fn test_higher_is_better() {
    assert!(Accuracy::default().higher_is_better());
    assert!(Precision::default().higher_is_better());
    assert!(Recall::default().higher_is_better());
    assert!(F1Score::default().higher_is_better());
    assert!(R2Score.higher_is_better());
    assert!(!MAE.higher_is_better());
    assert!(!RMSE.higher_is_better());
}

#[test]
fn test_metric_names() {
    assert_eq!(Accuracy::default().name(), "Accuracy");
    assert_eq!(Precision::default().name(), "Precision");
    assert_eq!(Recall::default().name(), "Recall");
    assert_eq!(F1Score::default().name(), "F1");
    assert_eq!(R2Score.name(), "R²");
    assert_eq!(MAE.name(), "MAE");
    assert_eq!(RMSE.name(), "RMSE");
}

#[test]
fn test_empty_input() {
    let metric = Accuracy::default();
    let pred = Tensor::from_vec(vec![], false);
    let target = Tensor::from_vec(vec![], false);

    let acc = metric.compute(&pred, &target);
    assert_eq!(acc, 0.0);
}

#[test]
fn test_precision_no_predictions() {
    let metric = Precision::default();
    let pred = Tensor::from_vec(vec![0.1, 0.2, 0.3], false);
    let target = Tensor::from_vec(vec![1.0, 1.0, 1.0], false);

    let prec = metric.compute(&pred, &target);
    assert_eq!(prec, 0.0); // No positive predictions
}

#[test]
fn test_recall_no_positives() {
    let metric = Recall::default();
    let pred = Tensor::from_vec(vec![0.9, 0.8, 0.7], false);
    let target = Tensor::from_vec(vec![0.0, 0.0, 0.0], false);

    let rec = metric.compute(&pred, &target);
    assert_eq!(rec, 0.0); // No actual positives
}

#[test]
fn test_f1_zero_precision_and_recall() {
    let metric = F1Score::default();
    // Neither precision nor recall will be > 0
    let pred = Tensor::from_vec(vec![0.1, 0.2], false);
    let target = Tensor::from_vec(vec![0.0, 0.0], false);

    let f1 = metric.compute(&pred, &target);
    assert_eq!(f1, 0.0);
}

#[test]
fn test_metric_with_custom_threshold() {
    let metric = Accuracy::new(0.3);
    // Predictions > 0.3 are positive
    let pred = Tensor::from_vec(vec![0.4, 0.2, 0.35], false);
    let target = Tensor::from_vec(vec![1.0, 0.0, 1.0], false);

    let acc = metric.compute(&pred, &target);
    assert!((acc - 1.0).abs() < 1e-5); // 3/3 correct
}

#[test]
fn test_precision_with_custom_threshold() {
    let metric = Precision::new(0.3);
    let pred = Tensor::from_vec(vec![0.4, 0.2, 0.35], false);
    let target = Tensor::from_vec(vec![1.0, 0.0, 1.0], false);

    let prec = metric.compute(&pred, &target);
    assert!((prec - 1.0).abs() < 1e-5); // 2 true positives, 0 false positives
}

#[test]
fn test_recall_with_custom_threshold() {
    let metric = Recall::new(0.3);
    let pred = Tensor::from_vec(vec![0.4, 0.2, 0.35], false);
    let target = Tensor::from_vec(vec![1.0, 0.0, 1.0], false);

    let rec = metric.compute(&pred, &target);
    assert!((rec - 1.0).abs() < 1e-5); // 2 true positives out of 2 actual positives
}

#[test]
fn test_f1_with_custom_threshold() {
    let metric = F1Score::new(0.3);
    let pred = Tensor::from_vec(vec![0.4, 0.2, 0.35], false);
    let target = Tensor::from_vec(vec![1.0, 0.0, 1.0], false);

    let f1 = metric.compute(&pred, &target);
    assert!((f1 - 1.0).abs() < 1e-5); // Perfect precision and recall
}

#[test]
fn test_accuracy_default_threshold() {
    let metric = Accuracy::default_threshold();
    assert!((metric.threshold - 0.5).abs() < 1e-5);
}

#[test]
fn test_accuracy_clone() {
    let metric = Accuracy::new(0.5);
    let cloned = metric.clone();
    assert!((metric.threshold - cloned.threshold).abs() < 1e-5);
}

#[test]
fn test_precision_clone() {
    let metric = Precision::new(0.5);
    let cloned = metric.clone();
    assert!((metric.threshold - cloned.threshold).abs() < 1e-5);
}

#[test]
fn test_recall_clone() {
    let metric = Recall::new(0.5);
    let cloned = metric.clone();
    assert!((metric.threshold - cloned.threshold).abs() < 1e-5);
}

#[test]
fn test_f1_clone() {
    let metric = F1Score::new(0.5);
    let cloned = metric.clone();
    // F1 stores precision and recall internally
    assert_eq!(metric.name(), cloned.name());
}
