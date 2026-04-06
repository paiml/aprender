//! CI/CD integration recipe for cost-performance benchmarking.

use crate::experiment::CostPerformancePoint;
use crate::recipes::RecipeResult;
use std::collections::HashMap;

/// CI/CD integration recipe for cost-performance benchmarking
#[derive(Debug)]
pub struct CiCdBenchmarkRecipe {
    /// Benchmark name
    pub(crate) name: String,
    /// Threshold checks
    pub(crate) thresholds: HashMap<String, f64>,
    /// Results from runs
    results: Vec<CostPerformancePoint>,
}

impl CiCdBenchmarkRecipe {
    /// Create a new CI/CD benchmark recipe
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), thresholds: HashMap::new(), results: Vec::new() }
    }

    /// Add a performance threshold (fails if below)
    pub fn add_min_performance_threshold(&mut self, metric: impl Into<String>, min_value: f64) {
        self.thresholds.insert(format!("min_{}", metric.into()), min_value);
    }

    /// Add a cost threshold (fails if above)
    pub fn add_max_cost_threshold(&mut self, max_cost: f64) {
        self.thresholds.insert("max_cost".to_string(), max_cost);
    }

    /// Add a result
    pub fn add_result(&mut self, point: CostPerformancePoint) {
        self.results.push(point);
    }

    /// Check thresholds and return CI/CD result
    pub fn check(&self) -> RecipeResult {
        let mut result = RecipeResult::success(&self.name);
        let mut all_passed = true;

        for point in &self.results {
            // Check cost threshold
            if let Some(&max_cost) = self.thresholds.get("max_cost") {
                if point.cost > max_cost {
                    all_passed = false;
                    result = result
                        .with_metric(format!("{}_cost_exceeded", point.id), point.cost - max_cost);
                }
            }

            // Check performance thresholds
            for (key, &threshold) in &self.thresholds {
                if let Some(metric_name) = key.strip_prefix("min_") {
                    if metric_name == "performance" && point.performance < threshold {
                        all_passed = false;
                        result = result.with_metric(
                            format!("{}_performance_below_threshold", point.id),
                            threshold - point.performance,
                        );
                    }
                }
            }
        }

        result = result.with_metric("all_checks_passed", if all_passed { 1.0 } else { 0.0 });

        if !all_passed {
            result.success = false;
            result.error = Some("One or more threshold checks failed".to_string());
        }

        result
    }
}
