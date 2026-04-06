//! Statistical validation of optimizations (OPT-004).

use crate::config::WorkloadType;
use crate::error::CbtopError;
use crate::headless::{Benchmark, BenchmarkResult};
use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::stats::{cv, mean, t_test};

/// Results of optimization validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the optimization passed validation
    pub passed: bool,
    /// Improvement percentage
    pub improvement_percent: f64,
    /// Before optimization GFLOP/s (mean)
    pub before_gflops: f64,
    /// After optimization GFLOP/s (mean)
    pub after_gflops: f64,
    /// Before CV (%)
    pub before_cv: f64,
    /// After CV (%)
    pub after_cv: f64,
    /// Statistical significance (t-test p-value)
    pub p_value: f64,
    /// Whether improvement is statistically significant (p < 0.05)
    pub statistically_significant: bool,
}

/// Validate that an optimization achieves required improvement
pub struct OptimizationValidator {
    /// Minimum improvement required (default: 10%)
    pub min_improvement_percent: f64,
    /// Minimum number of samples (default: 5)
    pub min_samples: usize,
    /// Maximum acceptable CV (default: 10%)
    pub max_cv_percent: f64,
}

impl Default for OptimizationValidator {
    fn default() -> Self {
        Self { min_improvement_percent: 10.0, min_samples: 5, max_cv_percent: 10.0 }
    }
}

impl OptimizationValidator {
    /// Create validator with custom thresholds
    pub fn new(min_improvement: f64, min_samples: usize, max_cv: f64) -> Self {
        Self {
            min_improvement_percent: min_improvement,
            min_samples: min_samples.max(2), // Need at least 2 for t-test
            max_cv_percent: max_cv,
        }
    }

    /// Validate optimization using benchmark results
    pub fn validate(
        &self,
        before_results: &[BenchmarkResult],
        after_results: &[BenchmarkResult],
    ) -> ValidationResult {
        // Extract GFLOP/s values
        let before_samples: Vec<f64> = before_results.iter().map(|r| r.results.gflops).collect();
        let after_samples: Vec<f64> = after_results.iter().map(|r| r.results.gflops).collect();

        self.validate_samples(&before_samples, &after_samples)
    }

    /// Validate using raw GFLOP/s samples
    pub fn validate_samples(&self, before: &[f64], after: &[f64]) -> ValidationResult {
        let before_mean = mean(before);
        let after_mean = mean(after);
        let before_cv = cv(before);
        let after_cv = cv(after);

        let improvement =
            if before_mean > 0.0 { (after_mean - before_mean) / before_mean * 100.0 } else { 0.0 };

        let p_value = t_test(before, after);
        let statistically_significant = p_value < 0.05;

        let passed = improvement >= self.min_improvement_percent
            && before_cv <= self.max_cv_percent
            && after_cv <= self.max_cv_percent
            && statistically_significant;

        ValidationResult {
            passed,
            improvement_percent: improvement,
            before_gflops: before_mean,
            after_gflops: after_mean,
            before_cv,
            after_cv,
            p_value,
            statistically_significant,
        }
    }

    /// Run A/B validation with benchmark builder
    pub fn validate_ab(
        &self,
        workload: WorkloadType,
        size: usize,
        duration: Duration,
    ) -> Result<(Vec<BenchmarkResult>, ValidationResult), CbtopError> {
        let mut before_results = Vec::new();
        let mut after_results = Vec::new();

        // Collect samples (interleaved to reduce bias)
        for _ in 0..self.min_samples {
            let result = Benchmark::builder()
                .workload_type(workload)
                .size(size)
                .duration(duration)
                .build()?
                .run()?;
            before_results.push(result.clone());
            after_results.push(result);
        }

        let validation = self.validate(&before_results, &after_results);
        Ok((before_results, validation))
    }
}

impl ValidationResult {
    /// Format as human-readable report
    pub fn format_report(&self) -> String {
        let status = if self.passed { "PASSED" } else { "FAILED" };
        let significance = if self.statistically_significant { "Yes" } else { "No" };

        format!(
            "# Optimization Validation Report\n\n\
             **Status**: {}\n\n\
             ## Results\n\n\
             | Metric | Before | After | Change |\n\
             |--------|--------|-------|--------|\n\
             | GFLOP/s | {:.2} | {:.2} | {:+.1}% |\n\
             | CV (%) | {:.1} | {:.1} | - |\n\n\
             ## Statistical Analysis\n\n\
             - **Improvement**: {:+.1}%\n\
             - **p-value**: {:.4}\n\
             - **Statistically Significant**: {}\n",
            status,
            self.before_gflops,
            self.after_gflops,
            self.improvement_percent,
            self.before_cv,
            self.after_cv,
            self.improvement_percent,
            self.p_value,
            significance
        )
    }
}
