//! Distillation benchmarking and hyperparameter sweep tool.
//!
//! This crate provides tools for:
//! - Systematic hyperparameter sweeps
//! - Statistical analysis of results
//! - Comparison of distillation strategies
//! - Cost-performance analysis and recommendations
//!
//! # Toyota Way Principles
//!
//! - **Kaizen**: Data-driven optimization through systematic experimentation
//! - **Muda Elimination**: Avoid wasted training runs through early stopping
//! - **Visual Control**: Clear visualization of benchmark results

pub mod cost;
pub mod stats;
pub mod strategies;
pub mod sweep;

pub use cost::{
    ConfigParams, Constraints, CostModel, CostPerformanceAnalysis, CostPerformancePoint,
    Recommendation,
};
pub use stats::{StatisticalAnalyzer, TestResult};
pub use strategies::{DistillStrategy, StrategyComparison};
pub use sweep::{SweepConfig, SweepResult, Sweeper};

use entrenar_common::Result;

/// Run a temperature sweep.
pub fn temperature_sweep(
    range: std::ops::Range<f32>,
    step: f32,
    runs_per_point: usize,
) -> Result<SweepResult> {
    let config = SweepConfig::temperature(range, step).with_runs(runs_per_point);
    Sweeper::new(config).run()
}

/// Compare multiple distillation strategies.
pub fn compare_strategies(strategies: &[DistillStrategy]) -> Result<StrategyComparison> {
    strategies::compare(strategies)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_sweep_returns_results() {
        let result = temperature_sweep(1.0..4.0, 1.0, 1);
        assert!(result.is_ok());
    }
}
