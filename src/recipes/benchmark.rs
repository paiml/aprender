//! Cost-performance benchmarking recipe implementation.

use crate::experiment::{CostPerformanceBenchmark, CostPerformancePoint, ExperimentRun};
use crate::recipes::RecipeResult;
use std::collections::HashMap;

/// Cost-performance benchmarking recipe
#[derive(Debug)]
pub struct CostPerformanceBenchmarkRecipe {
    benchmark: CostPerformanceBenchmark,
    pub(crate) budget_constraint: Option<f64>,
    performance_target: Option<f64>,
}

impl CostPerformanceBenchmarkRecipe {
    /// Create a new benchmarking recipe
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            benchmark: CostPerformanceBenchmark::new(name),
            budget_constraint: None,
            performance_target: None,
        }
    }

    /// Set a budget constraint
    pub fn with_budget(mut self, max_cost: f64) -> Self {
        self.budget_constraint = Some(max_cost);
        self
    }

    /// Set a performance target
    pub fn with_performance_target(mut self, target: f64) -> Self {
        self.performance_target = Some(target);
        self
    }

    /// Add an experiment run as a data point
    pub fn add_run(&mut self, run: &ExperimentRun, performance_metric: &str) {
        let performance = run.metrics.get(performance_metric).copied().unwrap_or(0.0);
        let cost = run.cost.as_ref().map(|c| c.total_cost_usd).unwrap_or(0.0);
        let energy = run.energy.as_ref().map(|e| e.total_joules).unwrap_or(0.0);

        let mut metadata = HashMap::new();
        metadata.insert("paradigm".to_string(), format!("{:?}", run.paradigm));
        metadata.insert("device".to_string(), format!("{:?}", run.device));
        metadata.insert("platform".to_string(), format!("{:?}", run.platform));

        self.benchmark.add_point(CostPerformancePoint {
            id: run.run_id.clone(),
            performance,
            cost,
            energy_joules: energy,
            latency_ms: None,
            metadata,
        });
    }

    /// Run the benchmark analysis
    pub fn analyze(&mut self) -> RecipeResult {
        contract_pre_analyze!(self);
        let mut result = RecipeResult::success("cost-performance-benchmark");

        // Compute Pareto frontier
        let frontier = self.benchmark.compute_pareto_frontier().to_vec();
        result = result.with_metric("pareto_optimal_count", frontier.len() as f64);
        result = result.with_metric("total_configurations", self.benchmark.points.len() as f64);

        // Find best within budget if constraint set
        if let Some(budget) = self.budget_constraint {
            if let Some(best) = self.benchmark.best_within_budget(budget) {
                result = result.with_metric("best_in_budget_performance", best.performance);
                result = result.with_metric("best_in_budget_cost", best.cost);
            }
        }

        // Check if any point meets performance target
        if let Some(target) = self.performance_target {
            let meets_target = self.benchmark.points.iter().any(|p| p.performance >= target);
            result = result.with_metric("meets_target", if meets_target { 1.0 } else { 0.0 });

            // Find cheapest that meets target
            let cheapest =
                self.benchmark.points.iter().filter(|p| p.performance >= target).min_by(|a, b| {
                    a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal)
                });

            if let Some(cheapest) = cheapest {
                result = result.with_metric("cheapest_meeting_target_cost", cheapest.cost);
            }
        }

        // Add efficiency scores
        let efficiency = self.benchmark.efficiency_scores();
        if !efficiency.is_empty() {
            let max_efficiency = efficiency.iter().map(|e| e.1).fold(f64::NEG_INFINITY, f64::max);
            result = result.with_metric("max_efficiency", max_efficiency);
        }

        result
    }

    /// Get the benchmark
    pub fn benchmark(&self) -> &CostPerformanceBenchmark {
        &self.benchmark
    }

    /// Get mutable benchmark
    pub fn benchmark_mut(&mut self) -> &mut CostPerformanceBenchmark {
        &mut self.benchmark
    }
}
