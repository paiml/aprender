use super::*;
use crate::linear_model::LinearRegression;

#[test]
fn test_model_result_stats() {
    let mut result = ModelResult::new("test");
    result.cv_scores = vec![0.9, 0.85, 0.88, 0.92, 0.87];
    result.compute_stats();

    assert!((result.mean_score - 0.884).abs() < 0.001);
    assert!(result.std_score > 0.0);
}

#[test]
fn test_model_result_empty_scores() {
    let mut result = ModelResult::new("test");
    result.compute_stats();
    assert!((result.mean_score - 0.0).abs() < 1e-6);
}

#[test]
fn test_model_result_single_score() {
    let mut result = ModelResult::new("test");
    result.cv_scores = vec![0.9];
    result.compute_stats();
    assert!((result.mean_score - 0.9).abs() < 1e-6);
    assert!((result.std_score - 0.0).abs() < 1e-6);
}

#[test]
fn test_evaluator_regression() {
    let x = Matrix::from_vec(12, 1, (0..12).map(|i| i as f32).collect()).expect("valid dimensions");
    let y = Vector::from_slice(&[
        1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0,
    ]);

    let evaluator = ModelEvaluator::new(TaskType::Regression).with_cv_folds(3);

    let mut model = LinearRegression::new();
    let result = evaluator
        .evaluate(&mut model, "LinReg", &x, &y)
        .expect("should succeed");

    assert_eq!(result.name, "LinReg");
    assert_eq!(result.cv_scores.len(), 3);
    assert!(result.mean_score > 0.9);
}

#[test]
fn test_evaluator_too_few_samples() {
    let x = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("valid dimensions");
    let y = Vector::from_slice(&[1.0, 2.0]);

    let evaluator = ModelEvaluator::new(TaskType::Regression).with_cv_folds(5);

    let mut model = LinearRegression::new();
    let result = evaluator.evaluate(&mut model, "LinReg", &x, &y);

    assert!(result.is_err());
}

#[test]
fn test_comparison_result_best() {
    let mut result1 = ModelResult::new("Model1");
    result1.mean_score = 0.85;

    let mut result2 = ModelResult::new("Model2");
    result2.mean_score = 0.92;

    let mut result3 = ModelResult::new("Model3");
    result3.mean_score = 0.88;

    let comparison = ComparisonResult {
        models: vec![result1, result2, result3],
        task_type: TaskType::Regression,
        primary_metric: "R²".to_string(),
    };

    let best = comparison.best_model().expect("should have best");
    assert_eq!(best.name, "Model2");
}

#[test]
fn test_comparison_result_ranked() {
    let mut result1 = ModelResult::new("Model1");
    result1.mean_score = 0.85;

    let mut result2 = ModelResult::new("Model2");
    result2.mean_score = 0.92;

    let comparison = ComparisonResult {
        models: vec![result1, result2],
        task_type: TaskType::Classification,
        primary_metric: "accuracy".to_string(),
    };

    let ranked = comparison.ranked();
    assert_eq!(ranked[0].name, "Model2");
    assert_eq!(ranked[1].name, "Model1");
}

#[test]
fn test_comparison_report() {
    let mut result = ModelResult::new("TestModel");
    result.mean_score = 0.9;
    result.std_score = 0.05;

    let comparison = ComparisonResult {
        models: vec![result],
        task_type: TaskType::Regression,
        primary_metric: "R²".to_string(),
    };

    let report = comparison.report();
    assert!(report.contains("TestModel"));
    assert!(report.contains("0.9"));
    assert!(report.contains("R²"));
}

#[test]
fn test_evaluate_regression_metrics() {
    let y_true = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_pred = Vector::from_slice(&[1.1, 2.0, 2.9, 4.1, 5.0]);

    let metrics = evaluate_regression(&y_pred, &y_true);

    assert!(metrics.contains_key("r2"));
    assert!(metrics.contains_key("mse"));
    assert!(metrics.contains_key("rmse"));
    assert!(*metrics.get("r2").expect("has r2") > 0.9);
}

#[test]
fn test_evaluate_classification_metrics() {
    let y_true = vec![0, 0, 1, 1, 2, 2];
    let y_pred = vec![0, 0, 1, 1, 2, 2];

    let metrics = evaluate_classification(&y_pred, &y_true);

    assert!(metrics.contains_key("accuracy"));
    assert!(metrics.contains_key("f1_macro"));
    assert!((*metrics.get("accuracy").expect("has accuracy") - 1.0).abs() < 1e-6);
}

#[test]
fn test_evaluator_with_options() {
    let evaluator = ModelEvaluator::new(TaskType::Classification)
        .with_cv_folds(10)
        .with_random_state(42);

    assert_eq!(evaluator.cv_folds, 10);
    assert_eq!(evaluator.random_state, Some(42));
}

#[test]
fn test_model_result_add_metric() {
    let mut result = ModelResult::new("test");
    result.add_metric("custom_metric", 0.95);

    assert_eq!(
        *result.metrics.get("custom_metric").expect("has metric"),
        0.95
    );
}

// ================================================================
// Additional coverage tests for missed branches
// ================================================================

#[test]
fn test_comparison_result_best_model_empty() {
    let comparison = ComparisonResult {
        models: vec![],
        task_type: TaskType::Regression,
        primary_metric: "R\u{00b2}".to_string(),
    };
    assert!(comparison.best_model().is_none());
}

#[test]
fn test_comparison_result_ranked_empty() {
    let comparison = ComparisonResult {
        models: vec![],
        task_type: TaskType::Regression,
        primary_metric: "R\u{00b2}".to_string(),
    };
    let ranked = comparison.ranked();
    assert!(ranked.is_empty());
}

#[test]
fn test_comparison_report_empty_models() {
    let comparison = ComparisonResult {
        models: vec![],
        task_type: TaskType::Classification,
        primary_metric: "accuracy".to_string(),
    };
    let report = comparison.report();
    assert!(report.contains("Classification"));
    assert!(report.contains("accuracy"));
    // No best model line since models is empty
    assert!(!report.contains("Best model:"));
}

#[test]
fn test_comparison_report_classification_task() {
    let mut result = ModelResult::new("LR");
    result.mean_score = 0.85;
    result.std_score = 0.02;

    let comparison = ComparisonResult {
        models: vec![result],
        task_type: TaskType::Classification,
        primary_metric: "accuracy".to_string(),
    };

    let report = comparison.report();
    assert!(report.contains("Classification"));
    assert!(report.contains("accuracy"));
    assert!(report.contains("LR"));
    assert!(report.contains("Best model:"));
}

#[test]
fn test_model_result_train_time_ms() {
    let mut result = ModelResult::new("timed");
    result.train_time_ms = Some(1234);
    assert_eq!(result.train_time_ms, Some(1234));
}

#[test]
fn test_model_result_compute_stats_two_scores() {
    let mut result = ModelResult::new("test");
    result.cv_scores = vec![0.8, 0.9];
    result.compute_stats();

    assert!((result.mean_score - 0.85).abs() < 1e-6);
    // std of [0.8, 0.9] with sample std = sqrt(0.005) ~ 0.0707
    assert!(result.std_score > 0.0);
}

#[test]
fn test_evaluator_with_cv_folds_minimum_clamp() {
    // cv_folds.max(2) means requesting 1 or 0 should yield 2
    let evaluator = ModelEvaluator::new(TaskType::Regression).with_cv_folds(1);
    assert_eq!(evaluator.cv_folds, 2);

    let evaluator0 = ModelEvaluator::new(TaskType::Regression).with_cv_folds(0);
    assert_eq!(evaluator0.cv_folds, 2);
}

#[test]
fn test_evaluate_classification_imperfect() {
    let y_true = vec![0, 0, 1, 1, 2, 2];
    let y_pred = vec![0, 1, 1, 0, 2, 1]; // Some errors

    let metrics = evaluate_classification(&y_pred, &y_true);

    let acc = *metrics.get("accuracy").expect("has accuracy");
    assert!(acc < 1.0);
    assert!(acc > 0.0);

    let f1 = *metrics.get("f1_macro").expect("has f1_macro");
    assert!(f1 < 1.0);
    assert!(f1 > 0.0);

    // Weighted metrics should also be present
    assert!(metrics.contains_key("precision_weighted"));
    assert!(metrics.contains_key("recall_weighted"));
    assert!(metrics.contains_key("f1_weighted"));
}

#[test]
fn test_comparison_best_model_with_nan_score() {
    let mut r1 = ModelResult::new("Normal");
    r1.mean_score = 0.85;

    let mut r2 = ModelResult::new("NaN");
    r2.mean_score = f32::NAN;

    let comparison = ComparisonResult {
        models: vec![r1, r2],
        task_type: TaskType::Regression,
        primary_metric: "R\u{00b2}".to_string(),
    };

    // Should return Some even with NaN (uses Ordering::Equal fallback)
    let best = comparison.best_model();
    assert!(best.is_some());
}

#[test]
fn test_comparison_ranked_with_equal_scores() {
    let mut r1 = ModelResult::new("A");
    r1.mean_score = 0.9;

    let mut r2 = ModelResult::new("B");
    r2.mean_score = 0.9;

    let comparison = ComparisonResult {
        models: vec![r1, r2],
        task_type: TaskType::Regression,
        primary_metric: "R\u{00b2}".to_string(),
    };

    let ranked = comparison.ranked();
    assert_eq!(ranked.len(), 2);
}

#[test]
fn test_model_result_multiple_metrics() {
    let mut result = ModelResult::new("multi");
    result.add_metric("f1", 0.88);
    result.add_metric("auc", 0.92);
    result.add_metric("precision", 0.85);

    assert_eq!(result.metrics.len(), 3);
    assert_eq!(*result.metrics.get("f1").expect("has f1"), 0.88);
    assert_eq!(*result.metrics.get("auc").expect("has auc"), 0.92);
}

#[test]
fn test_model_result_clone_and_debug() {
    let mut result = ModelResult::new("test");
    result.cv_scores = vec![0.9, 0.85];
    result.mean_score = 0.875;
    result.std_score = 0.025;

    let cloned = result.clone();
    assert_eq!(cloned.name, "test");
    assert_eq!(cloned.cv_scores.len(), 2);

    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("ModelResult"));
}

#[test]
fn test_comparison_result_clone_and_debug() {
    let comparison = ComparisonResult {
        models: vec![],
        task_type: TaskType::Regression,
        primary_metric: "R\u{00b2}".to_string(),
    };

    let cloned = comparison.clone();
    assert_eq!(cloned.task_type, TaskType::Regression);

    let debug_str = format!("{:?}", comparison);
    assert!(debug_str.contains("ComparisonResult"));
}

#[test]
fn test_task_type_eq() {
    assert_eq!(TaskType::Regression, TaskType::Regression);
    assert_eq!(TaskType::Classification, TaskType::Classification);
    assert_ne!(TaskType::Regression, TaskType::Classification);
}

#[test]
fn test_evaluator_debug() {
    let evaluator = ModelEvaluator::new(TaskType::Regression);
    let debug_str = format!("{:?}", evaluator);
    assert!(debug_str.contains("ModelEvaluator"));
}
