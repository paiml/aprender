//! Model evaluation framework for comparing multiple models.
//!
//! Provides `ModelEvaluator` for systematic model comparison with
//! cross-validation, metric collection, and ranking.

use std::fmt::Write;

use crate::error::{AprenderError, Result};
use crate::metrics::classification::{accuracy, f1_score, precision, recall, Average};
use crate::metrics::{mse, r_squared, rmse};
use crate::model_selection::{cross_validate, KFold};
use crate::primitives::{Matrix, Vector};
use crate::traits::Estimator;

/// Results from evaluating a single model.
#[derive(Clone, Debug)]
pub struct ModelResult {
    /// Model name/identifier
    pub name: String,
    /// Cross-validation scores (one per fold)
    pub cv_scores: Vec<f32>,
    /// Mean CV score
    pub mean_score: f32,
    /// Standard deviation of CV scores
    pub std_score: f32,
    /// Training time in milliseconds (if measured)
    pub train_time_ms: Option<u64>,
    /// Additional metrics (name -> value)
    pub metrics: std::collections::HashMap<String, f32>,
}

impl ModelResult {
    /// Create a new model result.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            cv_scores: Vec::new(),
            mean_score: 0.0,
            std_score: 0.0,
            train_time_ms: None,
            metrics: std::collections::HashMap::new(),
        }
    }

    /// Compute mean and std from CV scores.
    pub fn compute_stats(&mut self) {
        if self.cv_scores.is_empty() {
            return;
        }

        let n = self.cv_scores.len() as f32;
        self.mean_score = self.cv_scores.iter().sum::<f32>() / n;

        if self.cv_scores.len() > 1 {
            let variance: f32 = self
                .cv_scores
                .iter()
                .map(|s| (s - self.mean_score).powi(2))
                .sum::<f32>()
                / (n - 1.0);
            self.std_score = variance.sqrt();
        }
    }

    /// Add a custom metric.
    pub fn add_metric(&mut self, name: &str, value: f32) {
        self.metrics.insert(name.to_string(), value);
    }
}

/// Type of machine learning task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskType {
    /// Regression task (continuous target)
    Regression,
    /// Classification task (discrete labels)
    Classification,
}

/// Comparison results from evaluating multiple models.
#[derive(Clone, Debug)]
pub struct ComparisonResult {
    /// Results for each model
    pub models: Vec<ModelResult>,
    /// Task type (regression or classification)
    pub task_type: TaskType,
    /// Primary metric used for ranking
    pub primary_metric: String,
}

impl ComparisonResult {
    /// Get the best model by mean score (higher is better).
    #[must_use]
    pub fn best_model(&self) -> Option<&ModelResult> {
        self.models.iter().max_by(|a, b| {
            a.mean_score
                .partial_cmp(&b.mean_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Rank models by mean score (descending).
    #[must_use]
    pub fn ranked(&self) -> Vec<&ModelResult> {
        let mut ranked: Vec<_> = self.models.iter().collect();
        ranked.sort_by(|a, b| {
            b.mean_score
                .partial_cmp(&a.mean_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// Generate a comparison report.
    #[must_use]
    pub fn report(&self) -> String {
        let mut report = String::new();
        let _ = writeln!(report, "Model Comparison Report ({:?})", self.task_type);
        let _ = writeln!(report, "Primary metric: {}", self.primary_metric);
        let _ = writeln!(report, "{}", "=".repeat(60));
        let _ = writeln!(report, "{:<20} {:>10} {:>10}", "Model", "Mean", "Std");
        let _ = writeln!(report, "{}", "-".repeat(60));

        for model in self.ranked() {
            let _ = writeln!(
                report,
                "{:<20} {:>10.4} {:>10.4}",
                model.name, model.mean_score, model.std_score
            );
        }

        if let Some(best) = self.best_model() {
            let _ = writeln!(report, "{}", "-".repeat(60));
            let _ = writeln!(
                report,
                "Best model: {} (score: {:.4})",
                best.name, best.mean_score
            );
        }

        report
    }
}

/// Model evaluator for systematic comparison.
///
/// # Examples
///
/// ```
/// use aprender::metrics::evaluator::{ModelEvaluator, TaskType};
/// use aprender::linear_model::LinearRegression;
/// use aprender::primitives::{Matrix, Vector};
///
/// let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect()).unwrap();
/// let y = Vector::from_slice(&[1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]);
///
/// let evaluator = ModelEvaluator::new(TaskType::Regression)
///     .with_cv_folds(3);
///
/// let mut model = LinearRegression::new();
/// let result = evaluator.evaluate(&mut model, "LinReg", &x, &y).unwrap();
/// assert!(result.mean_score > 0.9);
/// ```
#[derive(Debug)]
pub struct ModelEvaluator {
    /// Task type
    task_type: TaskType,
    /// Number of CV folds
    cv_folds: usize,
    /// Random seed for reproducibility
    random_state: Option<u64>,
}

impl ModelEvaluator {
    /// Create a new evaluator.
    #[must_use]
    pub fn new(task_type: TaskType) -> Self {
        Self {
            task_type,
            cv_folds: 5,
            random_state: None,
        }
    }

    /// Set number of cross-validation folds.
    #[must_use]
    pub fn with_cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = folds.max(2);
        self
    }

    /// Set random state for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Evaluate a single model using cross-validation.
    ///
    /// # Errors
    ///
    /// Returns error if cross-validation fails.
    pub fn evaluate<E: Estimator + Clone>(
        &self,
        model: &mut E,
        name: &str,
        x: &Matrix<f32>,
        y: &Vector<f32>,
    ) -> Result<ModelResult> {
        if x.n_rows() < self.cv_folds {
            return Err(AprenderError::DimensionMismatch {
                expected: format!(
                    "at least {} samples for {} folds",
                    self.cv_folds, self.cv_folds
                ),
                actual: format!("{} samples", x.n_rows()),
            });
        }

        let mut result = ModelResult::new(name);

        // Cross-validation
        let kfold = KFold::new(self.cv_folds);
        let cv_result = cross_validate(model, x, y, &kfold)?;
        result.cv_scores = cv_result.scores;
        result.compute_stats();

        Ok(result)
    }

    /// Compare multiple models.
    ///
    /// # Errors
    ///
    /// Returns error if any model evaluation fails.
    pub fn compare<E: Estimator + Clone>(
        &self,
        models: Vec<(&mut E, &str)>,
        x: &Matrix<f32>,
        y: &Vector<f32>,
    ) -> Result<ComparisonResult> {
        let mut results = Vec::new();

        for (model, name) in models {
            let result = self.evaluate(model, name, x, y)?;
            results.push(result);
        }

        let primary_metric = match self.task_type {
            TaskType::Regression => "R²".to_string(),
            TaskType::Classification => "accuracy".to_string(),
        };

        Ok(ComparisonResult {
            models: results,
            task_type: self.task_type,
            primary_metric,
        })
    }
}

/// Evaluate classification metrics on predictions.
#[must_use]
pub fn evaluate_classification(
    y_pred: &[usize],
    y_true: &[usize],
) -> std::collections::HashMap<String, f32> {
    let mut metrics = std::collections::HashMap::new();

    metrics.insert("accuracy".to_string(), accuracy(y_pred, y_true));
    metrics.insert(
        "precision_macro".to_string(),
        precision(y_pred, y_true, Average::Macro),
    );
    metrics.insert(
        "recall_macro".to_string(),
        recall(y_pred, y_true, Average::Macro),
    );
    metrics.insert(
        "f1_macro".to_string(),
        f1_score(y_pred, y_true, Average::Macro),
    );
    metrics.insert(
        "precision_weighted".to_string(),
        precision(y_pred, y_true, Average::Weighted),
    );
    metrics.insert(
        "recall_weighted".to_string(),
        recall(y_pred, y_true, Average::Weighted),
    );
    metrics.insert(
        "f1_weighted".to_string(),
        f1_score(y_pred, y_true, Average::Weighted),
    );

    metrics
}

/// Evaluate regression metrics on predictions.
#[must_use]
pub fn evaluate_regression(
    y_pred: &Vector<f32>,
    y_true: &Vector<f32>,
) -> std::collections::HashMap<String, f32> {
    let mut metrics = std::collections::HashMap::new();

    metrics.insert("r2".to_string(), r_squared(y_pred, y_true));
    metrics.insert("mse".to_string(), mse(y_pred, y_true));
    metrics.insert("rmse".to_string(), rmse(y_pred, y_true));

    metrics
}

#[cfg(test)]
mod tests {
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
        let x =
            Matrix::from_vec(12, 1, (0..12).map(|i| i as f32).collect()).expect("valid dimensions");
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
}
