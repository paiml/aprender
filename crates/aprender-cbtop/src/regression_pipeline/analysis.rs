//! Regression analysis types for comparing benchmark results.

use super::metrics::BenchmarkMetric;
use super::types::PipelineStatus;

use super::metrics::BenchmarkResults;

/// Comparison result for a single metric
#[derive(Debug, Clone)]
pub struct MetricRegression {
    /// Metric name
    pub name: String,
    /// Baseline mean
    pub baseline_mean: f64,
    /// Current mean
    pub current_mean: f64,
    /// Percent change
    pub percent_change: f64,
    /// Whether this is a regression
    pub is_regression: bool,
    /// Whether this is a warning
    pub is_warning: bool,
    /// Unit
    pub unit: String,
}

impl MetricRegression {
    /// Create from two metrics
    pub fn from_metrics(
        baseline: &BenchmarkMetric,
        current: &BenchmarkMetric,
        regression_threshold: f64,
        warning_threshold: f64,
    ) -> Self {
        let baseline_mean = baseline.mean();
        let current_mean = current.mean();

        let percent_change = if baseline_mean.abs() > 1e-10 {
            ((current_mean - baseline_mean) / baseline_mean) * 100.0
        } else {
            0.0
        };

        // For latency-like metrics (lower is better), positive change is regression
        // For throughput-like metrics (higher is better), negative change is regression
        let is_latency_metric = baseline.name.contains("latency")
            || baseline.name.contains("time")
            || baseline.name.contains("duration");

        let regression_change = if is_latency_metric {
            percent_change // Increase is bad
        } else {
            -percent_change // Decrease is bad
        };

        Self {
            name: baseline.name.clone(),
            baseline_mean,
            current_mean,
            percent_change,
            is_regression: regression_change >= regression_threshold,
            is_warning: regression_change >= warning_threshold
                && regression_change < regression_threshold,
            unit: baseline.unit.clone(),
        }
    }
}

/// Complete regression analysis result
#[derive(Debug, Clone)]
pub struct RegressionAnalysis {
    /// Baseline results
    pub baseline: BenchmarkResults,
    /// Current results
    pub current: BenchmarkResults,
    /// Per-metric regressions
    pub regressions: Vec<MetricRegression>,
    /// Overall status
    pub status: PipelineStatus,
    /// Analysis duration in milliseconds
    pub analysis_duration_ms: u64,
    /// Summary message
    pub summary: String,
}

impl RegressionAnalysis {
    /// Count significant regressions
    pub fn regression_count(&self) -> usize {
        self.regressions.iter().filter(|r| r.is_regression).count()
    }

    /// Count warnings
    pub fn warning_count(&self) -> usize {
        self.regressions.iter().filter(|r| r.is_warning).count()
    }

    /// Count improvements
    pub fn improvement_count(&self) -> usize {
        self.regressions
            .iter()
            .filter(|r| !r.is_regression && !r.is_warning && r.percent_change.abs() > 1.0)
            .filter(|r| {
                // Check if change is improvement
                let is_latency = r.name.contains("latency") || r.name.contains("time");
                if is_latency {
                    r.percent_change < 0.0 // Decrease is good
                } else {
                    r.percent_change > 0.0 // Increase is good
                }
            })
            .count()
    }

    /// Get worst regression
    pub fn worst_regression(&self) -> Option<&MetricRegression> {
        self.regressions.iter().filter(|r| r.is_regression).max_by(|a, b| {
            a.percent_change
                .abs()
                .partial_cmp(&b.percent_change.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}
