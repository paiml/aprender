//! Cost-performance benchmarking with Pareto frontier analysis.
//!
//! This module provides tools for analyzing cost vs performance tradeoffs
//! and identifying Pareto-optimal configurations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A point in the cost-performance space
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostPerformancePoint {
    /// Unique identifier for this configuration
    pub id: String,
    /// Performance metric (e.g., accuracy, F1, throughput)
    pub performance: f64,
    /// Cost in USD
    pub cost: f64,
    /// Energy consumption in joules
    pub energy_joules: f64,
    /// Latency in milliseconds (for inference)
    pub latency_ms: Option<f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Cost-performance benchmark with Pareto frontier analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostPerformanceBenchmark {
    /// Name of the benchmark
    pub name: String,
    /// All data points
    pub points: Vec<CostPerformancePoint>,
    /// Pareto-optimal points (computed lazily)
    pareto_frontier: Option<Vec<usize>>,
}

impl CostPerformanceBenchmark {
    /// Create a new benchmark
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), points: Vec::new(), pareto_frontier: None }
    }

    /// Add a data point
    pub fn add_point(&mut self, point: CostPerformancePoint) {
        self.points.push(point);
        self.pareto_frontier = None; // Invalidate cache
    }

    /// Compute the Pareto frontier (maximize performance, minimize cost)
    pub fn compute_pareto_frontier(&mut self) -> &[usize] {
        if let Some(ref frontier) = self.pareto_frontier {
            return frontier;
        }

        let mut frontier = Vec::new();

        for (i, point) in self.points.iter().enumerate() {
            let mut is_dominated = false;

            for (j, other) in self.points.iter().enumerate() {
                if i == j {
                    continue;
                }

                // Other dominates point if: better or equal on all, strictly better on at least one
                let other_better_perf = other.performance >= point.performance;
                let other_better_cost = other.cost <= point.cost;
                let other_strictly_better =
                    other.performance > point.performance || other.cost < point.cost;

                if other_better_perf && other_better_cost && other_strictly_better {
                    is_dominated = true;
                    break;
                }
            }

            if !is_dominated {
                frontier.push(i);
            }
        }

        // Sort by performance descending
        frontier.sort_by(|&a, &b| {
            self.points[b]
                .performance
                .partial_cmp(&self.points[a].performance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.pareto_frontier = Some(frontier);
        self.pareto_frontier.as_ref().expect("just assigned Some above")
    }

    /// Get Pareto-optimal points
    pub fn pareto_optimal_points(&mut self) -> Vec<&CostPerformancePoint> {
        let frontier = self.compute_pareto_frontier().to_vec();
        frontier.iter().map(|&i| &self.points[i]).collect()
    }

    /// Find the best point within a cost budget
    pub fn best_within_budget(&mut self, max_cost: f64) -> Option<&CostPerformancePoint> {
        self.compute_pareto_frontier();

        self.pareto_frontier
            .as_ref()
            .expect("compute_pareto_frontier ensures Some")
            .iter()
            .map(|&i| &self.points[i])
            .filter(|p| p.cost <= max_cost)
            .max_by(|a, b| {
                a.performance.partial_cmp(&b.performance).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Calculate cost-performance efficiency (performance per dollar)
    pub fn efficiency_scores(&self) -> Vec<(usize, f64)> {
        self.points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let efficiency = if p.cost > 0.0 { p.performance / p.cost } else { f64::INFINITY };
                (i, efficiency)
            })
            .collect()
    }
}
