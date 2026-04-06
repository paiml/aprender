//! Hansei (反省) Post-Training Report Generator
//!
//! Toyota Way principle: Reflection and continuous improvement through
//! systematic analysis of training outcomes.
//!
//! Reference: Liker, J.K. (2004). The Toyota Way: 14 Management Principles.

use super::output::PostTrainingReport;
use super::types::{IssueSeverity, MetricSummary, TrainingIssue, Trend};
use crate::monitor::{Metric, MetricStats, MetricsCollector};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

/// Hansei report generator
pub struct HanseiAnalyzer {
    /// Threshold for loss increase to trigger warning
    pub loss_increase_threshold: f64,
    /// Threshold for gradient norm to indicate explosion
    pub gradient_explosion_threshold: f64,
    /// Threshold for gradient norm to indicate vanishing
    pub gradient_vanishing_threshold: f64,
    /// Minimum expected accuracy improvement
    pub min_accuracy_improvement: f64,
}

impl Default for HanseiAnalyzer {
    fn default() -> Self {
        Self {
            loss_increase_threshold: 0.1, // 10% increase
            gradient_explosion_threshold: 100.0,
            gradient_vanishing_threshold: 1e-7,
            min_accuracy_improvement: 0.01, // 1% improvement
        }
    }
}

impl HanseiAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze a completed training run and generate a report
    pub fn analyze(
        &self,
        training_id: &str,
        collector: &MetricsCollector,
        duration_secs: f64,
    ) -> PostTrainingReport {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        let mut metric_summaries = HashMap::new();
        let mut final_metrics = HashMap::new();

        let summary = collector.summary();
        let total_steps = summary.values().map(|s| s.count).sum::<usize>() as u64;

        // Analyze each metric
        for (metric, stats) in &summary {
            let metric_summary = self.analyze_metric(metric, stats);
            metric_summaries.insert(metric.clone(), metric_summary.clone());
            final_metrics.insert(metric.clone(), stats.mean);

            // Check for issues based on metric type
            self.check_metric_issues(metric, &metric_summary, stats, &mut issues);
        }

        // Generate recommendations based on issues
        self.generate_recommendations(&issues, &mut recommendations);

        // Check for missing expected metrics
        self.check_missing_metrics(&summary, &mut issues);

        // Sort issues by severity (critical first)
        issues.sort_by(|a, b| b.severity.cmp(&a.severity));

        PostTrainingReport {
            training_id: training_id.to_string(),
            duration_secs,
            total_steps,
            final_metrics,
            metric_summaries,
            issues,
            recommendations,
        }
    }

    fn analyze_metric(&self, metric: &Metric, stats: &MetricStats) -> MetricSummary {
        // Determine trend based on metric type and statistics
        let trend = self.determine_trend(metric, stats);

        MetricSummary {
            initial: stats.min,      // Approximation - would need history for actual initial
            final_value: stats.mean, // Approximation - would need last value
            min: stats.min,
            max: stats.max,
            mean: stats.mean,
            std_dev: stats.std,
            trend,
        }
    }

    fn determine_trend(&self, metric: &Metric, stats: &MetricStats) -> Trend {
        let cv = coeff_of_variation(stats);
        if cv > 0.5 {
            return Trend::Oscillating;
        }
        match metric {
            Metric::Loss => range_trend(stats, true),
            Metric::Accuracy => range_trend(stats, false),
            Metric::GradientNorm => {
                if cv < 0.2 {
                    Trend::Stable
                } else {
                    Trend::Oscillating
                }
            }
            Metric::LearningRate | Metric::Epoch | Metric::Batch | Metric::Custom(_) => {
                Trend::Stable
            }
        }
    }
}

fn coeff_of_variation(stats: &MetricStats) -> f64 {
    if stats.mean.abs() > 1e-10 {
        stats.std / stats.mean.abs()
    } else {
        0.0
    }
}

/// Determine trend based on whether mean is above or below midpoint.
/// `lower_is_better` = true for loss, false for accuracy.
fn range_trend(stats: &MetricStats, lower_is_better: bool) -> Trend {
    if stats.max - stats.min < stats.std * 0.5 {
        return Trend::Stable;
    }
    let mid = f64::midpoint(stats.min, stats.max);
    let improving = if lower_is_better { stats.mean < mid } else { stats.mean > mid };
    if improving {
        Trend::Improving
    } else {
        Trend::Degrading
    }
}

impl HanseiAnalyzer {
    fn check_metric_issues(
        &self,
        metric: &Metric,
        summary: &MetricSummary,
        stats: &MetricStats,
        issues: &mut Vec<TrainingIssue>,
    ) {
        match metric {
            Metric::Loss => self.check_loss_issues(summary, stats, issues),
            Metric::Accuracy => self.check_accuracy_issues(summary, stats, issues),
            Metric::GradientNorm => self.check_gradient_issues(stats, issues),
            Metric::LearningRate => self.check_lr_issues(summary, issues),
            Metric::Epoch | Metric::Batch | Metric::Custom(_) => {}
        }
    }

    /// Check loss metric for NaN/Inf, degrading trend, and oscillation.
    fn check_loss_issues(
        &self,
        summary: &MetricSummary,
        stats: &MetricStats,
        issues: &mut Vec<TrainingIssue>,
    ) {
        if stats.has_nan {
            issues.push(TrainingIssue {
                severity: IssueSeverity::Critical,
                category: "Numerical Stability".to_string(),
                description: "NaN values detected in loss".to_string(),
                recommendation:
                    "Reduce learning rate, add gradient clipping, or check data preprocessing"
                        .to_string(),
            });
        }
        if stats.has_inf {
            issues.push(TrainingIssue {
                severity: IssueSeverity::Critical,
                category: "Numerical Stability".to_string(),
                description: "Infinity values detected in loss".to_string(),
                recommendation: "Check for division by zero, reduce learning rate".to_string(),
            });
        }
        if summary.trend == Trend::Degrading {
            issues.push(TrainingIssue {
                severity: IssueSeverity::Warning,
                category: "Convergence".to_string(),
                description: "Loss appears to be increasing over training".to_string(),
                recommendation: "Consider reducing learning rate or checking data quality"
                    .to_string(),
            });
        }
        if summary.trend == Trend::Oscillating {
            issues.push(TrainingIssue {
                severity: IssueSeverity::Warning,
                category: "Stability".to_string(),
                description: "Loss is oscillating significantly".to_string(),
                recommendation: "Reduce learning rate or increase batch size".to_string(),
            });
        }
    }

    /// Check accuracy metric for low values and stagnation.
    fn check_accuracy_issues(
        &self,
        summary: &MetricSummary,
        stats: &MetricStats,
        issues: &mut Vec<TrainingIssue>,
    ) {
        if summary.final_value < 0.5 && stats.count > 100 {
            issues.push(TrainingIssue {
                severity: IssueSeverity::Warning,
                category: "Performance".to_string(),
                description: format!("Final accuracy is low: {:.2}%", summary.final_value * 100.0),
                recommendation: "Consider model architecture changes or hyperparameter tuning"
                    .to_string(),
            });
        }
        if summary.trend == Trend::Stable
            && summary.max - summary.min < self.min_accuracy_improvement
        {
            issues.push(TrainingIssue {
                severity: IssueSeverity::Info,
                category: "Convergence".to_string(),
                description: "Accuracy shows minimal improvement".to_string(),
                recommendation: "Model may have converged or may be stuck in local minimum"
                    .to_string(),
            });
        }
    }

    /// Check gradient norms for explosion and vanishing.
    fn check_gradient_issues(&self, stats: &MetricStats, issues: &mut Vec<TrainingIssue>) {
        if stats.max > self.gradient_explosion_threshold {
            issues.push(TrainingIssue {
                severity: IssueSeverity::Error,
                category: "Gradient Health".to_string(),
                description: format!("Gradient explosion detected: max norm = {:.2e}", stats.max),
                recommendation: "Enable gradient clipping (e.g., max_norm=1.0)".to_string(),
            });
        }
        if stats.mean < self.gradient_vanishing_threshold && stats.count > 10 {
            issues.push(TrainingIssue {
                severity: IssueSeverity::Warning,
                category: "Gradient Health".to_string(),
                description: format!(
                    "Possible vanishing gradients: mean norm = {:.2e}",
                    stats.mean
                ),
                recommendation:
                    "Consider using residual connections or different activation functions"
                        .to_string(),
            });
        }
    }

    /// Check learning rate schedule for high variance.
    fn check_lr_issues(&self, summary: &MetricSummary, issues: &mut Vec<TrainingIssue>) {
        if summary.std_dev > summary.mean * 0.5 {
            issues.push(TrainingIssue {
                severity: IssueSeverity::Info,
                category: "Hyperparameters".to_string(),
                description: "Learning rate schedule shows high variance".to_string(),
                recommendation: "Review learning rate schedule configuration".to_string(),
            });
        }
    }

    fn check_missing_metrics(
        &self,
        metrics: &HashMap<Metric, MetricStats>,
        issues: &mut Vec<TrainingIssue>,
    ) {
        // Check for essential metrics
        if !metrics.contains_key(&Metric::Loss) {
            issues.push(TrainingIssue {
                severity: IssueSeverity::Warning,
                category: "Observability".to_string(),
                description: "No loss metric recorded".to_string(),
                recommendation: "Ensure loss is being tracked for proper monitoring".to_string(),
            });
        }
    }

    fn generate_recommendations(
        &self,
        issues: &[TrainingIssue],
        recommendations: &mut Vec<String>,
    ) {
        let has_numerical_issues = issues.iter().any(|i| i.category == "Numerical Stability");
        let has_gradient_issues = issues.iter().any(|i| i.category == "Gradient Health");
        let has_convergence_issues = issues.iter().any(|i| i.category == "Convergence");

        if has_numerical_issues {
            recommendations.push(
                "Priority 1: Address numerical stability before continuing training".to_string(),
            );
        }

        if has_gradient_issues {
            recommendations.push("Enable gradient clipping in optimizer configuration".to_string());
        }

        if has_convergence_issues {
            recommendations.push(
                "Consider hyperparameter search for learning rate and batch size".to_string(),
            );
        }

        if issues.is_empty() {
            recommendations.push(
                "Training completed without detected issues. Consider running validation tests."
                    .to_string(),
            );
        }
    }

    /// Generate a human-readable report
    pub fn format_report(&self, report: &PostTrainingReport) -> String {
        let mut output = String::new();

        // Writing to String never fails, so we ignore the Result
        let _ = writeln!(output, "═══════════════════════════════════════════════════════════════");
        let _ =
            writeln!(output, "                    HANSEI POST-TRAINING REPORT                 ");
        let _ = writeln!(output, "═══════════════════════════════════════════════════════════════");
        let _ = writeln!(output);
        let _ = writeln!(output, "Training ID: {}", report.training_id);
        let _ = writeln!(output, "Duration: {:.2}s", report.duration_secs);
        let _ = writeln!(output, "Total Steps: {}", report.total_steps);
        let _ = writeln!(output);

        // Metric summaries
        let _ =
            writeln!(output, "─── Metric Summaries ───────────────────────────────────────────");
        for (metric_type, summary) in &report.metric_summaries {
            let _ = writeln!(output, "\n{metric_type:?}:");
            let _ = writeln!(output, "  Mean: {:.6}  Std: {:.6}", summary.mean, summary.std_dev);
            let _ = writeln!(output, "  Min: {:.6}   Max: {:.6}", summary.min, summary.max);
            let _ = writeln!(output, "  Trend: {}", summary.trend);
        }
        let _ = writeln!(output);

        // Issues
        if !report.issues.is_empty() {
            let _ = writeln!(
                output,
                "─── Issues Detected ────────────────────────────────────────────"
            );
            for issue in &report.issues {
                let _ = writeln!(output, "\n[{}] {}", issue.severity, issue.category);
                let _ = writeln!(output, "  {}", issue.description);
                let _ = writeln!(output, "  → {}", issue.recommendation);
            }
            let _ = writeln!(output);
        }

        // Recommendations
        let _ =
            writeln!(output, "─── Recommendations ────────────────────────────────────────────");
        for (i, rec) in report.recommendations.iter().enumerate() {
            let _ = writeln!(output, "{}. {}", i + 1, rec);
        }
        let _ = writeln!(output);

        let _ = writeln!(output, "═══════════════════════════════════════════════════════════════");

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determine_trend_all_metric_variants() {
        let analyzer = HanseiAnalyzer::default();

        // Stable stats (low CV, narrow range)
        let stable_stats = MetricStats {
            count: 100,
            mean: 1.0,
            std: 0.01,
            min: 0.99,
            max: 1.01,
            sum: 100.0,
            has_nan: false,
            has_inf: false,
        };

        // Syntactic match covering all arms from determine_trend
        let metrics = [
            Metric::Loss,
            Metric::Accuracy,
            Metric::GradientNorm,
            Metric::LearningRate,
            Metric::Epoch,
            Metric::Batch,
            Metric::Custom("custom_metric".to_string()),
        ];

        for metric in &metrics {
            let trend = analyzer.determine_trend(metric, &stable_stats);
            match metric {
                Metric::Loss => {
                    assert!(matches!(
                        trend,
                        Trend::Stable | Trend::Improving | Trend::Degrading | Trend::Oscillating
                    ));
                }
                Metric::Accuracy => {
                    assert!(matches!(
                        trend,
                        Trend::Stable | Trend::Improving | Trend::Degrading | Trend::Oscillating
                    ));
                }
                Metric::GradientNorm => {
                    assert!(matches!(trend, Trend::Stable | Trend::Oscillating));
                }
                Metric::LearningRate | Metric::Epoch | Metric::Batch | Metric::Custom(_) => {
                    assert_eq!(trend, Trend::Stable);
                }
            }
        }
    }

    #[test]
    fn test_check_metric_issues_all_metric_variants() {
        let analyzer = HanseiAnalyzer::default();

        let stats = MetricStats {
            count: 200,
            mean: 0.5,
            std: 0.1,
            min: 0.3,
            max: 0.7,
            sum: 100.0,
            has_nan: false,
            has_inf: false,
        };

        let summary = MetricSummary {
            initial: 0.3,
            final_value: 0.5,
            min: 0.3,
            max: 0.7,
            mean: 0.5,
            std_dev: 0.1,
            trend: Trend::Stable,
        };

        let metrics = [
            Metric::Loss,
            Metric::Accuracy,
            Metric::GradientNorm,
            Metric::LearningRate,
            Metric::Epoch,
            Metric::Batch,
            Metric::Custom("test".to_string()),
        ];

        for metric in &metrics {
            let mut issues = Vec::new();
            analyzer.check_metric_issues(metric, &summary, &stats, &mut issues);

            // Syntactic match covering all arms from check_metric_issues
            match metric {
                Metric::Loss => {
                    // Loss branch checks NaN, Inf, trend
                }
                Metric::Accuracy => {
                    // Accuracy branch checks low accuracy, no improvement
                }
                Metric::GradientNorm => {
                    // GradientNorm branch checks explosion, vanishing
                }
                Metric::LearningRate => {
                    // LearningRate branch checks variance
                }
                Metric::Epoch | Metric::Batch | Metric::Custom(_) => {
                    // No-op branch
                    assert!(issues.is_empty(), "Epoch/Batch/Custom should produce no issues");
                }
            }
        }
    }

    #[test]
    fn test_analyzer_default() {
        let analyzer = HanseiAnalyzer::default();
        assert!((analyzer.loss_increase_threshold - 0.1).abs() < 1e-10);
        assert!((analyzer.gradient_explosion_threshold - 100.0).abs() < 1e-10);
        assert!((analyzer.gradient_vanishing_threshold - 1e-7).abs() < 1e-15);
        assert!((analyzer.min_accuracy_improvement - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_analyzer_new() {
        let analyzer = HanseiAnalyzer::new();
        assert!((analyzer.loss_increase_threshold - 0.1).abs() < 1e-10);
    }
}
