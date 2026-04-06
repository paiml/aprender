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
/// let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect()).expect("valid matrix dims");
/// let y = Vector::from_slice(&[1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]);
///
/// let evaluator = ModelEvaluator::new(TaskType::Regression)
///     .with_cv_folds(3);
///
/// let mut model = LinearRegression::new();
/// let result = evaluator.evaluate(&mut model, "LinReg", &x, &y).expect("evaluation should succeed");
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
#[path = "evaluator_tests.rs"]
mod tests;
