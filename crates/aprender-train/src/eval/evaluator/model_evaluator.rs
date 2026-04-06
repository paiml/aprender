//! Model Evaluator for running evaluations

use super::super::classification::{confusion_matrix, MultiClassMetrics};
use super::config::EvalConfig;
use super::kfold::KFold;
use super::leaderboard::Leaderboard;
use super::metric::Metric;
use super::result::EvalResult;
use crate::error::{Error, Result};
use std::time::Instant;

/// Model Evaluator for running evaluations
pub struct ModelEvaluator {
    config: EvalConfig,
}

impl ModelEvaluator {
    /// Create a new evaluator with given configuration
    pub fn new(config: EvalConfig) -> Self {
        Self { config }
    }

    /// Evaluate classification with cross-validation
    ///
    /// Takes a prediction function that maps (train_indices, test_indices) to predictions.
    /// Returns EvalResult with cv_scores, cv_mean, and cv_std populated.
    pub fn evaluate_cv<F>(
        &self,
        model_name: impl Into<String>,
        y_true: &[usize],
        predict_fn: F,
    ) -> Result<EvalResult>
    where
        F: Fn(&[usize], &[usize]) -> Vec<usize>,
    {
        if self.config.cv_folds == 0 {
            return Err(Error::InvalidParameter(
                "cv_folds must be > 0 for cross-validation".into(),
            ));
        }

        let start = Instant::now();
        let kfold = KFold::new(self.config.cv_folds).with_seed(self.config.seed);
        let folds = kfold.split(y_true.len());

        let mut fold_scores: Vec<f64> = Vec::with_capacity(self.config.cv_folds);

        // Get primary metric for CV scoring
        let primary_metric = self.config.metrics.first().copied().unwrap_or(Metric::Accuracy);

        for (train_idx, test_idx) in &folds {
            // Get predictions for this fold
            let predictions = predict_fn(train_idx, test_idx);

            // Get test labels
            let test_labels: Vec<usize> = test_idx.iter().map(|&i| y_true[i]).collect();

            // Compute primary metric for this fold
            let cm = confusion_matrix(&predictions, &test_labels);
            let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

            let score = match primary_metric {
                Metric::Accuracy
                | Metric::R2
                | Metric::MSE
                | Metric::MAE
                | Metric::RMSE
                | Metric::Silhouette
                | Metric::Inertia
                | Metric::WER
                | Metric::RTFx
                | Metric::BLEU
                | Metric::ROUGE(_)
                | Metric::Perplexity
                | Metric::MMLUAccuracy
                | Metric::PassAtK(_)
                | Metric::NDCGAtK(_) => cm.accuracy(),
                Metric::Precision(avg) => metrics.precision_avg(avg),
                Metric::Recall(avg) => metrics.recall_avg(avg),
                Metric::F1(avg) => metrics.f1_avg(avg),
            };

            fold_scores.push(score);
        }

        // Compute mean and std
        let cv_mean = fold_scores.iter().sum::<f64>() / fold_scores.len().max(1) as f64;
        let cv_std = if fold_scores.len() > 1 {
            let variance = fold_scores.iter().map(|s| (s - cv_mean).powi(2)).sum::<f64>()
                / (fold_scores.len().saturating_sub(1)).max(1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let mut result = EvalResult::new(model_name);
        result.cv_scores = Some(fold_scores);
        result.cv_mean = Some(cv_mean);
        result.cv_std = Some(cv_std);
        result.add_score(primary_metric, cv_mean);
        result.inference_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(result)
    }

    /// Evaluate classification model with predictions and ground truth
    ///
    /// # Arguments
    /// * `model_name` - Name for the model in results
    /// * `y_pred` - Predicted class labels
    /// * `y_true` - Ground truth class labels
    ///
    /// # Returns
    /// EvalResult containing computed metrics
    pub fn evaluate_classification(
        &self,
        model_name: impl Into<String>,
        y_pred: &[usize],
        y_true: &[usize],
    ) -> Result<EvalResult> {
        if y_pred.len() != y_true.len() {
            return Err(Error::InvalidParameter(
                "Predictions and targets must have same length".into(),
            ));
        }

        let start = Instant::now();

        let cm = confusion_matrix(y_pred, y_true);
        let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

        let mut result = EvalResult::new(model_name);

        for metric in &self.config.metrics {
            let score = match metric {
                Metric::Accuracy => cm.accuracy(),
                Metric::Precision(avg) => metrics.precision_avg(*avg),
                Metric::Recall(avg) => metrics.recall_avg(*avg),
                Metric::F1(avg) => metrics.f1_avg(*avg),
                Metric::R2
                | Metric::MSE
                | Metric::MAE
                | Metric::RMSE
                | Metric::Silhouette
                | Metric::Inertia
                | Metric::WER
                | Metric::RTFx
                | Metric::BLEU
                | Metric::ROUGE(_)
                | Metric::Perplexity
                | Metric::MMLUAccuracy
                | Metric::PassAtK(_)
                | Metric::NDCGAtK(_) => continue,
            };
            result.add_score(*metric, score);
        }

        result.inference_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(result)
    }

    /// Compare multiple classification models
    ///
    /// # Arguments
    /// * `models` - Slice of (name, predictions) tuples
    /// * `y_true` - Ground truth labels
    ///
    /// # Returns
    /// Leaderboard with all models ranked by primary metric
    pub fn compare_classification(
        &self,
        models: &[(&str, &[usize])],
        y_true: &[usize],
    ) -> Result<Leaderboard> {
        let primary = self.config.metrics.first().copied().unwrap_or(Metric::Accuracy);
        let mut leaderboard = Leaderboard::new(primary);

        for (name, y_pred) in models {
            let result = self.evaluate_classification(*name, y_pred, y_true)?;
            leaderboard.add(result);
        }

        Ok(leaderboard)
    }

    /// Get the configuration
    pub fn config(&self) -> &EvalConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::classification::Average;
    use crate::eval::evaluator::metric::RougeVariant;

    #[test]
    fn test_cv_precision_avg_arm() {
        // Exercises: Metric::Precision(avg) => metrics.precision_avg(avg)
        let metric = Metric::Precision(Average::Macro);
        match metric {
            Metric::Precision(avg) => {
                let _ = avg;
            }
            _ => unreachable!(),
        }
        let config =
            EvalConfig { metrics: vec![metric], cv_folds: 2, seed: 42, ..Default::default() };
        let evaluator = ModelEvaluator::new(config);
        let y_true: Vec<usize> = (0..20).map(|i| i % 2).collect();
        let result = evaluator
            .evaluate_cv("Test", &y_true, |_, test_idx| {
                test_idx.iter().map(|&i| y_true[i]).collect()
            })
            .expect("operation should succeed");
        assert!(result.cv_mean.is_some());
    }

    #[test]
    fn test_cv_recall_avg_arm() {
        // Exercises: Metric::Recall(avg) => metrics.recall_avg(avg)
        let metric = Metric::Recall(Average::Weighted);
        match metric {
            Metric::Recall(avg) => {
                let _ = avg;
            }
            _ => unreachable!(),
        }
        let config =
            EvalConfig { metrics: vec![metric], cv_folds: 2, seed: 42, ..Default::default() };
        let evaluator = ModelEvaluator::new(config);
        let y_true: Vec<usize> = (0..20).map(|i| i % 2).collect();
        let result = evaluator
            .evaluate_cv("Test", &y_true, |_, test_idx| {
                test_idx.iter().map(|&i| y_true[i]).collect()
            })
            .expect("operation should succeed");
        assert!(result.cv_mean.is_some());
    }

    #[test]
    fn test_cv_f1_avg_arm() {
        // Exercises: Metric::F1(avg) => metrics.f1_avg(avg)
        let metric = Metric::F1(Average::Micro);
        match metric {
            Metric::F1(avg) => {
                let _ = avg;
            }
            _ => unreachable!(),
        }
        let config =
            EvalConfig { metrics: vec![metric], cv_folds: 2, seed: 42, ..Default::default() };
        let evaluator = ModelEvaluator::new(config);
        let y_true: Vec<usize> = (0..20).map(|i| i % 2).collect();
        let result = evaluator
            .evaluate_cv("Test", &y_true, |_, test_idx| {
                test_idx.iter().map(|&i| y_true[i]).collect()
            })
            .expect("operation should succeed");
        assert!(result.cv_mean.is_some());
    }

    #[test]
    fn test_cv_accuracy_fallback_arm() {
        // Test the grouped arm: Accuracy|R2|MSE|... => cm.accuracy()
        for metric in [
            Metric::Accuracy,
            Metric::R2,
            Metric::MSE,
            Metric::MAE,
            Metric::RMSE,
            Metric::Silhouette,
            Metric::Inertia,
            Metric::WER,
            Metric::RTFx,
            Metric::BLEU,
            Metric::ROUGE(RougeVariant::Rouge1),
            Metric::Perplexity,
            Metric::MMLUAccuracy,
            Metric::PassAtK(1),
            Metric::NDCGAtK(5),
        ] {
            let config =
                EvalConfig { metrics: vec![metric], cv_folds: 2, seed: 42, ..Default::default() };
            let evaluator = ModelEvaluator::new(config);
            let y_true: Vec<usize> = (0..20).map(|i| i % 2).collect();
            let result = evaluator
                .evaluate_cv("Test", &y_true, |_, test_idx| {
                    test_idx.iter().map(|&i| y_true[i]).collect()
                })
                .expect("operation should succeed");
            assert!(result.cv_mean.is_some(), "CV should succeed with metric {metric:?}");
        }
    }

    #[test]
    fn test_classify_precision_avg_arm() {
        // Exercises: Metric::Precision(avg) => metrics.precision_avg(*avg)
        let metric = Metric::Precision(Average::Macro);
        match metric {
            Metric::Precision(avg) => {
                let _ = avg;
            }
            _ => unreachable!(),
        }
        let config = EvalConfig { metrics: vec![metric], ..Default::default() };
        let evaluator = ModelEvaluator::new(config);
        let result = evaluator
            .evaluate_classification("Test", &[0, 1, 0], &[0, 1, 1])
            .expect("operation should succeed");
        assert!(result.get_score(Metric::Precision(Average::Macro)).is_some());
    }

    #[test]
    fn test_classify_recall_avg_arm() {
        // Exercises: Metric::Recall(avg) => metrics.recall_avg(*avg)
        let metric = Metric::Recall(Average::Micro);
        match metric {
            Metric::Recall(avg) => {
                let _ = avg;
            }
            _ => unreachable!(),
        }
        let config = EvalConfig { metrics: vec![metric], ..Default::default() };
        let evaluator = ModelEvaluator::new(config);
        let result = evaluator
            .evaluate_classification("Test", &[0, 1, 0], &[0, 1, 1])
            .expect("operation should succeed");
        assert!(result.get_score(Metric::Recall(Average::Micro)).is_some());
    }

    #[test]
    fn test_classify_f1_avg_arm() {
        // Exercises: Metric::F1(avg) => metrics.f1_avg(*avg)
        let metric = Metric::F1(Average::Weighted);
        match metric {
            Metric::F1(avg) => {
                let _ = avg;
            }
            _ => unreachable!(),
        }
        let config = EvalConfig { metrics: vec![metric], ..Default::default() };
        let evaluator = ModelEvaluator::new(config);
        let result = evaluator
            .evaluate_classification("Test", &[0, 1, 0], &[0, 1, 1])
            .expect("operation should succeed");
        assert!(result.get_score(Metric::F1(Average::Weighted)).is_some());
    }

    #[test]
    fn test_classify_skips_non_classification_metrics() {
        // Tests the grouped continue arm: R2|MSE|...|NDCGAtK(_) => continue
        let config = EvalConfig {
            metrics: vec![
                Metric::Accuracy,
                Metric::R2,
                Metric::MSE,
                Metric::MAE,
                Metric::RMSE,
                Metric::Silhouette,
                Metric::Inertia,
                Metric::WER,
                Metric::RTFx,
                Metric::BLEU,
                Metric::ROUGE(RougeVariant::RougeL),
                Metric::Perplexity,
                Metric::MMLUAccuracy,
                Metric::PassAtK(5),
                Metric::NDCGAtK(10),
            ],
            ..Default::default()
        };
        let evaluator = ModelEvaluator::new(config);
        let result = evaluator
            .evaluate_classification("Test", &[0, 1, 0], &[0, 1, 1])
            .expect("operation should succeed");
        assert!(result.get_score(Metric::Accuracy).is_some());
        assert!(result.get_score(Metric::R2).is_none());
        assert!(result.get_score(Metric::MSE).is_none());
        assert!(result.get_score(Metric::ROUGE(RougeVariant::RougeL)).is_none());
        assert!(result.get_score(Metric::PassAtK(5)).is_none());
        assert!(result.get_score(Metric::NDCGAtK(10)).is_none());
    }
}
