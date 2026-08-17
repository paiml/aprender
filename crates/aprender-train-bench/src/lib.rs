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
///
/// # Errors
///
/// Always -- see [`Sweeper::run`] and #2519. Nothing in this crate trains.
pub fn temperature_sweep(
    range: std::ops::Range<f32>,
    step: f32,
    runs_per_point: usize,
) -> Result<SweepResult> {
    let config = SweepConfig::temperature(range, step).with_runs(runs_per_point);
    Sweeper::new(config).run()
}

/// Compare multiple distillation strategies.
///
/// # Errors
///
/// Always -- see [`strategies::compare`] and #2519.
pub fn compare_strategies(strategies: &[DistillStrategy]) -> Result<StrategyComparison> {
    strategies::compare(strategies)
}

#[cfg(test)]
mod tests {
    use super::*;

    // #2519: this asserted `is_ok()` on a sweep that never trained, which is
    // exactly the shape of test that locks a fabrication in -- it passes
    // precisely because the output is invented.
    #[test]
    fn test_temperature_sweep_refuses_without_training() {
        let result = temperature_sweep(1.0..4.0, 1.0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_compare_strategies_refuses_without_training() {
        let result = compare_strategies(&[DistillStrategy::kd_only()]);
        assert!(result.is_err());
    }
}
