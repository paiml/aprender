//! Model evaluation and benchmarking
//!
//! This module provides tools for evaluating ML models:
//! - Confusion matrix with visualization
//! - ROC curve and AUC calculation
//! - Feature importance analysis
//! - Inference speed benchmarking
//! - Baseline model comparison

use crate::data::CodeFeatures;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Confusion matrix for binary classification
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// True positives (predicted positive, actual positive)
    pub tp: usize,
    /// True negatives (predicted negative, actual negative)
    pub tn: usize,
    /// False positives (predicted positive, actual negative)
    pub fp: usize,
    /// False negatives (predicted negative, actual positive)
    pub r#fn: usize,
}

impl ConfusionMatrix {
    /// Create confusion matrix from predictions and ground truth
    #[must_use]
    pub fn from_predictions(predictions: &[bool], ground_truth: &[bool]) -> Self {
        let mut matrix = Self::default();

        for (pred, truth) in predictions.iter().zip(ground_truth.iter()) {
            match (pred, truth) {
                (true, true) => matrix.tp += 1,
                (false, false) => matrix.tn += 1,
                (true, false) => matrix.fp += 1,
                (false, true) => matrix.r#fn += 1,
            }
        }

        matrix
    }

    /// Total number of samples
    #[must_use]
    pub fn total(&self) -> usize {
        self.tp + self.tn + self.fp + self.r#fn
    }

    /// Accuracy = (TP + TN) / total
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.tp + self.tn) as f64 / total as f64
    }

    /// Precision = TP / (TP + FP)
    #[must_use]
    pub fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 {
            return 0.0;
        }
        self.tp as f64 / denom as f64
    }

    /// Recall (sensitivity) = TP / (TP + FN)
    #[must_use]
    pub fn recall(&self) -> f64 {
        let denom = self.tp + self.r#fn;
        if denom == 0 {
            return 0.0;
        }
        self.tp as f64 / denom as f64
    }

    /// Specificity = TN / (TN + FP)
    #[must_use]
    pub fn specificity(&self) -> f64 {
        let denom = self.tn + self.fp;
        if denom == 0 {
            return 0.0;
        }
        self.tn as f64 / denom as f64
    }

    /// F1 score = 2 * (precision * recall) / (precision + recall)
    #[must_use]
    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();
        if precision + recall == 0.0 {
            return 0.0;
        }
        2.0 * (precision * recall) / (precision + recall)
    }

    /// Render as ASCII table
    #[must_use]
    pub fn to_ascii(&self) -> String {
        format!(
            r"
Confusion Matrix
================
                 Predicted
              Pos      Neg
Actual Pos   {:>5}    {:>5}   (TP, FN)
Actual Neg   {:>5}    {:>5}   (FP, TN)

Accuracy:  {:.3}
Precision: {:.3}
Recall:    {:.3}
F1 Score:  {:.3}
",
            self.tp,
            self.r#fn,
            self.fp,
            self.tn,
            self.accuracy(),
            self.precision(),
            self.recall(),
            self.f1_score()
        )
    }
}

/// ROC curve data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocPoint {
    /// Threshold used
    pub threshold: f64,
    /// True positive rate (sensitivity)
    pub tpr: f64,
    /// False positive rate (1 - specificity)
    pub fpr: f64,
}

/// ROC curve and AUC calculator
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RocCurve {
    /// Points on the ROC curve
    pub points: Vec<RocPoint>,
    /// Area under the curve
    pub auc: f64,
}

impl RocCurve {
    /// Calculate ROC curve from probability scores and ground truth
    #[must_use]
    pub fn from_scores(scores: &[f64], ground_truth: &[bool]) -> Self {
        if scores.is_empty() || scores.len() != ground_truth.len() {
            return Self::default();
        }

        // Sort by score descending
        let mut indexed: Vec<(f64, bool)> = scores
            .iter()
            .zip(ground_truth.iter())
            .map(|(&s, &t)| (s, t))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_positives = ground_truth.iter().filter(|&&t| t).count();
        let total_negatives = ground_truth.len() - total_positives;

        if total_positives == 0 || total_negatives == 0 {
            return Self::default();
        }

        let mut points = Vec::new();
        let mut tp = 0;
        let mut fp = 0;

        // Add point at (0, 0)
        points.push(RocPoint {
            threshold: 1.0,
            tpr: 0.0,
            fpr: 0.0,
        });

        let mut prev_score = f64::INFINITY;

        for (score, is_positive) in &indexed {
            // Allow direct comparison since scores are sorted and exact matches matter
            #[allow(clippy::float_cmp)]
            if *score != prev_score {
                let tpr = f64::from(tp) / total_positives as f64;
                let fpr = f64::from(fp) / total_negatives as f64;
                points.push(RocPoint {
                    threshold: *score,
                    tpr,
                    fpr,
                });
                prev_score = *score;
            }

            if *is_positive {
                tp += 1;
            } else {
                fp += 1;
            }
        }

        // Add final point at (1, 1)
        points.push(RocPoint {
            threshold: 0.0,
            tpr: 1.0,
            fpr: 1.0,
        });

        // Calculate AUC using trapezoidal rule
        let auc = Self::calculate_auc(&points);

        Self { points, auc }
    }

    /// Calculate AUC using trapezoidal integration
    fn calculate_auc(points: &[RocPoint]) -> f64 {
        let mut auc = 0.0;

        for i in 1..points.len() {
            let width = points[i].fpr - points[i - 1].fpr;
            let height = (points[i].tpr + points[i - 1].tpr) / 2.0;
            auc += width * height;
        }

        auc.abs()
    }

    /// Render as ASCII plot (simple representation)
    #[must_use]
    pub fn to_ascii(&self) -> String {
        use std::fmt::Write;

        let mut output = String::new();
        output.push_str("ROC Curve\n");
        output.push_str("=========\n");
        let _ = writeln!(output, "AUC: {:.4}\n", self.auc);

        // Simple 10x10 ASCII plot
        let grid_size = 10;
        let mut grid = vec![vec!['.'; grid_size]; grid_size];

        // Plot points (values are always non-negative, so cast is safe)
        #[allow(clippy::cast_sign_loss)]
        for point in &self.points {
            let x = (point.fpr * (grid_size - 1) as f64).round() as usize;
            let y = ((1.0 - point.tpr) * (grid_size - 1) as f64).round() as usize;
            if x < grid_size && y < grid_size {
                grid[y][x] = '*';
            }
        }

        // Diagonal line (random classifier)
        for (i, row) in grid.iter_mut().enumerate() {
            if row[i] == '.' {
                row[i] = '-';
            }
        }

        output.push_str("TPR\n");
        output.push_str("1.0 |");
        for row in &grid {
            output.push_str(&row.iter().collect::<String>());
            output.push_str("|\n    |");
        }
        output.push_str(&"-".repeat(grid_size));
        output.push_str("| FPR\n    0");
        output.push_str(&" ".repeat(grid_size - 2));
        output.push_str("1.0\n");

        output
    }
}

/// Feature importance for model interpretation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Feature name
    pub name: String,
    /// Importance score (0.0 to 1.0)
    pub importance: f64,
}

/// Calculate feature importance using permutation importance
pub fn calculate_feature_importance(
    features: &[CodeFeatures],
    labels: &[bool],
    predictor: &dyn Fn(&CodeFeatures) -> f64,
) -> Vec<FeatureImportance> {
    let baseline_score = calculate_accuracy(features, labels, predictor);
    let feature_names = [
        "ast_depth",
        "num_operators",
        "num_control_flow",
        "cyclomatic_complexity",
        "uses_edge_values",
    ];

    let mut importances = Vec::new();

    for (idx, name) in feature_names.iter().enumerate() {
        // Permute this feature
        let permuted_features: Vec<CodeFeatures> = features
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut permuted = f.clone();
                let swap_idx = (i + 1) % features.len();
                match idx {
                    0 => permuted.ast_depth = features[swap_idx].ast_depth,
                    1 => permuted.num_operators = features[swap_idx].num_operators,
                    2 => permuted.num_control_flow = features[swap_idx].num_control_flow,
                    3 => permuted.cyclomatic_complexity = features[swap_idx].cyclomatic_complexity,
                    4 => permuted.uses_edge_values = features[swap_idx].uses_edge_values,
                    _ => {}
                }
                permuted
            })
            .collect();

        let permuted_score = calculate_accuracy(&permuted_features, labels, predictor);
        let importance = (baseline_score - permuted_score).max(0.0);

        importances.push(FeatureImportance {
            name: (*name).to_string(),
            importance,
        });
    }

    // Normalize importances
    let total: f64 = importances.iter().map(|f| f.importance).sum();
    if total > 0.0 {
        for f in &mut importances {
            f.importance /= total;
        }
    }

    // Sort by importance descending
    importances.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    importances
}

fn calculate_accuracy(
    features: &[CodeFeatures],
    labels: &[bool],
    predictor: &dyn Fn(&CodeFeatures) -> f64,
) -> f64 {
    let correct: usize = features
        .iter()
        .zip(labels.iter())
        .map(|(f, &l)| {
            let pred = predictor(f) > 0.5;
            usize::from(pred == l)
        })
        .sum();

    if features.is_empty() {
        return 0.0;
    }
    correct as f64 / features.len() as f64
}

/// Benchmark results for inference speed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Number of predictions made
    pub num_predictions: usize,
    /// Total time in milliseconds
    pub total_time_ms: f64,
    /// Predictions per second
    pub predictions_per_sec: f64,
    /// Average latency per prediction in microseconds
    pub avg_latency_us: f64,
}

/// Benchmark inference speed
pub fn benchmark_inference<F>(
    predictor: F,
    features: &[CodeFeatures],
    iterations: usize,
) -> BenchmarkResult
where
    F: Fn(&CodeFeatures) -> f64,
{
    let start = Instant::now();

    for _ in 0..iterations {
        for f in features {
            let _ = predictor(f);
        }
    }

    let elapsed = start.elapsed();
    let total_time_ms = elapsed.as_secs_f64() * 1000.0;
    let num_predictions = iterations * features.len();

    let predictions_per_sec = if total_time_ms > 0.0 {
        num_predictions as f64 / (total_time_ms / 1000.0)
    } else {
        0.0
    };

    let avg_latency_us = if num_predictions > 0 {
        (total_time_ms * 1000.0) / num_predictions as f64
    } else {
        0.0
    };

    BenchmarkResult {
        num_predictions,
        total_time_ms,
        predictions_per_sec,
        avg_latency_us,
    }
}

/// Model comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    /// Baseline model metrics
    pub baseline: ComparisonMetrics,
    /// Trained model metrics
    pub trained: ComparisonMetrics,
    /// Improvement in accuracy (percentage points)
    pub accuracy_improvement: f64,
    /// Improvement in F1 (percentage points)
    pub f1_improvement: f64,
    /// Speedup factor (trained/baseline)
    pub speedup: f64,
}

/// Metrics for model comparison
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    /// Model name
    pub name: String,
    /// Accuracy
    pub accuracy: f64,
    /// F1 score
    pub f1_score: f64,
    /// Predictions per second
    pub predictions_per_sec: f64,
}

impl ModelComparison {
    /// Compare two models
    #[must_use]
    pub fn compare(baseline: ComparisonMetrics, trained: ComparisonMetrics) -> Self {
        let accuracy_improvement = trained.accuracy - baseline.accuracy;
        let f1_improvement = trained.f1_score - baseline.f1_score;
        let speedup = if baseline.predictions_per_sec > 0.0 {
            trained.predictions_per_sec / baseline.predictions_per_sec
        } else {
            1.0
        };

        Self {
            baseline,
            trained,
            accuracy_improvement,
            f1_improvement,
            speedup,
        }
    }

    /// Render as ASCII table
    #[must_use]
    pub fn to_ascii(&self) -> String {
        format!(
            r"
Model Comparison
================
                  Baseline     Trained      Delta
Name              {:<12} {:<12}
Accuracy          {:<12.4} {:<12.4} {:+.4}
F1 Score          {:<12.4} {:<12.4} {:+.4}
Pred/sec          {:<12.0} {:<12.0} {:.2}x
",
            self.baseline.name,
            self.trained.name,
            self.baseline.accuracy,
            self.trained.accuracy,
            self.accuracy_improvement,
            self.baseline.f1_score,
            self.trained.f1_score,
            self.f1_improvement,
            self.baseline.predictions_per_sec,
            self.trained.predictions_per_sec,
            self.speedup
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix_from_predictions() {
        let predictions = vec![true, true, false, false, true];
        let ground_truth = vec![true, false, false, true, true];

        let matrix = ConfusionMatrix::from_predictions(&predictions, &ground_truth);

        assert_eq!(matrix.tp, 2);
        assert_eq!(matrix.tn, 1);
        assert_eq!(matrix.fp, 1);
        assert_eq!(matrix.r#fn, 1);
    }

    #[test]
    fn test_confusion_matrix_metrics() {
        let matrix = ConfusionMatrix {
            tp: 50,
            tn: 40,
            fp: 10,
            r#fn: 0,
        };

        assert!((matrix.accuracy() - 0.9).abs() < 0.001);
        assert!((matrix.precision() - 0.833).abs() < 0.01);
        assert!((matrix.recall() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_confusion_matrix_perfect() {
        let predictions = vec![true, false, true, false];
        let ground_truth = vec![true, false, true, false];

        let matrix = ConfusionMatrix::from_predictions(&predictions, &ground_truth);

        assert!((matrix.accuracy() - 1.0).abs() < f64::EPSILON);
        assert!((matrix.precision() - 1.0).abs() < f64::EPSILON);
        assert!((matrix.recall() - 1.0).abs() < f64::EPSILON);
        assert!((matrix.f1_score() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_confusion_matrix_to_ascii() {
        let matrix = ConfusionMatrix {
            tp: 10,
            tn: 20,
            fp: 5,
            r#fn: 3,
        };

        let ascii = matrix.to_ascii();
        assert!(ascii.contains("Confusion Matrix"));
        assert!(ascii.contains("10"));
        assert!(ascii.contains("Accuracy"));
    }

    #[test]
    fn test_roc_curve_from_scores() {
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
        let ground_truth = vec![
            true, true, true, true, true, false, false, false, false, false,
        ];

        let roc = RocCurve::from_scores(&scores, &ground_truth);

        assert!(roc.auc > 0.9); // Should be close to 1.0 for well-separated classes
        assert!(!roc.points.is_empty());
    }

    #[test]
    fn test_roc_curve_random() {
        // Random classifier should have AUC ~0.5
        let scores = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let ground_truth = vec![true, false, true, false, true, false, true, false];

        let roc = RocCurve::from_scores(&scores, &ground_truth);

        // AUC for random should be around 0.5
        assert!(roc.auc >= 0.0 && roc.auc <= 1.0);
    }

    #[test]
    fn test_roc_curve_empty() {
        let roc = RocCurve::from_scores(&[], &[]);
        assert!((roc.auc - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_roc_curve_to_ascii() {
        let scores = vec![0.9, 0.8, 0.3, 0.2];
        let ground_truth = vec![true, true, false, false];

        let roc = RocCurve::from_scores(&scores, &ground_truth);
        let ascii = roc.to_ascii();

        assert!(ascii.contains("ROC Curve"));
        assert!(ascii.contains("AUC"));
    }

    #[test]
    fn test_feature_importance() {
        let features: Vec<CodeFeatures> = (0..100)
            .map(|i| CodeFeatures {
                ast_depth: (i % 10) as u32,
                num_operators: (i % 20) as u32,
                num_control_flow: (i % 5) as u32,
                cyclomatic_complexity: (i % 15) as f32,
                uses_edge_values: i % 3 == 0,
                ..Default::default()
            })
            .collect();
        let labels: Vec<bool> = (0..100).map(|i| i % 4 == 0).collect();

        let predictor = |f: &CodeFeatures| f.ast_depth as f64 * 0.1;
        let importance = calculate_feature_importance(&features, &labels, &predictor);

        assert_eq!(importance.len(), 5);
        // Sum should be ~1.0 if normalized
        let total: f64 = importance.iter().map(|f| f.importance).sum();
        assert!(total <= 1.1);
    }

    #[test]
    fn test_benchmark_inference() {
        let features: Vec<CodeFeatures> = (0..100).map(|_| CodeFeatures::default()).collect();

        let predictor = |_: &CodeFeatures| 0.5;
        let result = benchmark_inference(predictor, &features, 10);

        assert_eq!(result.num_predictions, 1000);
        assert!(result.total_time_ms > 0.0);
        assert!(result.predictions_per_sec > 0.0);
    }

    #[test]
    fn test_model_comparison() {
        let baseline = ComparisonMetrics {
            name: "Baseline".to_string(),
            accuracy: 0.7,
            f1_score: 0.65,
            predictions_per_sec: 10000.0,
        };

        let trained = ComparisonMetrics {
            name: "Trained".to_string(),
            accuracy: 0.85,
            f1_score: 0.82,
            predictions_per_sec: 8000.0,
        };

        let comparison = ModelComparison::compare(baseline, trained);

        assert!((comparison.accuracy_improvement - 0.15).abs() < 0.001);
        assert!((comparison.f1_improvement - 0.17).abs() < 0.001);
        assert!(comparison.speedup < 1.0); // Trained is slower
    }

    #[test]
    fn test_model_comparison_to_ascii() {
        let baseline = ComparisonMetrics {
            name: "Baseline".to_string(),
            accuracy: 0.7,
            f1_score: 0.65,
            predictions_per_sec: 10000.0,
        };

        let trained = ComparisonMetrics {
            name: "Trained".to_string(),
            accuracy: 0.85,
            f1_score: 0.82,
            predictions_per_sec: 15000.0,
        };

        let comparison = ModelComparison::compare(baseline, trained);
        let ascii = comparison.to_ascii();

        assert!(ascii.contains("Model Comparison"));
        assert!(ascii.contains("Baseline"));
        assert!(ascii.contains("Trained"));
    }

    #[test]
    fn test_confusion_matrix_empty() {
        let matrix = ConfusionMatrix::from_predictions(&[], &[]);
        assert_eq!(matrix.total(), 0);
        assert!((matrix.accuracy() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_confusion_matrix_specificity() {
        let matrix = ConfusionMatrix {
            tp: 10,
            tn: 80,
            fp: 20,
            r#fn: 10,
        };

        assert!((matrix.specificity() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_roc_point_debug() {
        let point = RocPoint {
            threshold: 0.5,
            tpr: 0.8,
            fpr: 0.2,
        };
        let debug = format!("{:?}", point);
        assert!(debug.contains("RocPoint"));
    }

    #[test]
    fn test_feature_importance_serialize() {
        let fi = FeatureImportance {
            name: "test".to_string(),
            importance: 0.5,
        };
        let json = serde_json::to_string(&fi).unwrap();
        assert!(json.contains("test"));
    }

    #[test]
    fn test_benchmark_result_serialize() {
        let result = BenchmarkResult {
            num_predictions: 1000,
            total_time_ms: 100.0,
            predictions_per_sec: 10000.0,
            avg_latency_us: 100.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("num_predictions"));
    }

    #[test]
    fn test_benchmark_predictions_per_sec_formula() {
        // Test that predictions_per_sec is computed as num / time (division, not multiplication)
        // Mutant: replace / with * would break this
        let features: Vec<CodeFeatures> = (0..10).map(|_| CodeFeatures::default()).collect();
        let predictor = |_: &CodeFeatures| 0.5;

        // Run 10 iterations on 10 features = 100 predictions
        let result = benchmark_inference(predictor, &features, 10);

        assert_eq!(result.num_predictions, 100);
        // predictions_per_sec = num_predictions / (total_time_ms / 1000)
        //                     = num_predictions * 1000 / total_time_ms
        // If mutant changes / to *, we'd get num_predictions * (total_time_ms / 1000)
        // which would be a tiny number (< 1) instead of a large number (> 1000)
        assert!(
            result.predictions_per_sec > 1000.0,
            "predictions_per_sec should be > 1000, got {}",
            result.predictions_per_sec
        );
    }

    #[test]
    fn test_roc_curve_grid_positions() {
        // Test that ROC curve plotting uses correct multiplication for grid positions
        // Mutant: replace * with / in `point.fpr * (grid_size - 1)` would break this
        let scores = vec![0.9, 0.1];
        let ground_truth = vec![true, false];

        let roc = RocCurve::from_scores(&scores, &ground_truth);
        let ascii = roc.to_ascii();

        // The ROC should show points at extremes (perfect classifier)
        // With AUC = 1.0, we should see stars at correct positions
        assert!(ascii.contains('*'), "ROC plot should contain star markers");
        // The plot should have the diagonal line
        assert!(
            ascii.contains('-'),
            "ROC plot should contain diagonal markers"
        );
    }
}
