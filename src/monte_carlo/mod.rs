//! Monte Carlo Simulation Framework
//!
//! Comprehensive tools for financial modeling and risk analysis using
//! Monte Carlo methods.
//!
//! # Overview
//!
//! This module provides:
//! - **Simulation Engine**: Reproducible RNG, variance reduction, convergence diagnostics
//! - **Risk Metrics**: `VaR`, `CVaR`, Maximum Drawdown, risk-adjusted ratios
//! - **Financial Models**: GBM, GARCH, jump-diffusion for stock prices
//! - **Business Models**: Revenue projection with Bayesian priors
//!
//! # Design Philosophy
//!
//! Built following Toyota Way principles:
//! - **Jidoka**: Fail-fast error handling with actionable messages
//! - **Kaizen**: Continuous improvement through comprehensive testing
//! - **Heijunka**: Leveled workload via variance reduction techniques
//!
//! # Example
//!
//! ```
//! use aprender::monte_carlo::prelude::*;
//!
//! // Create reproducible RNG
//! let mut rng = MonteCarloRng::new(42);
//!
//! // Generate random samples
//! let sample = rng.normal(0.0, 1.0);
//! assert!(sample.is_finite());
//!
//! // Calculate VaR from returns
//! let returns = vec![-0.05, -0.02, 0.01, 0.03, 0.05, 0.02, -0.01, 0.04, -0.03, 0.00];
//! let var_95 = VaR::historical(&returns, 0.95);
//! assert!(var_95 >= 0.0);
//! ```
//!
//! # References
//!
//! - Glasserman (2003), "Monte Carlo Methods in Financial Engineering"
//! - Jorion (2006), "Value at Risk"
//! - Hull (2018), "Options, Futures, and Other Derivatives"

pub mod engine;
pub mod models;
pub mod risk;

/// Error types for Monte Carlo operations
pub mod error;

pub use error::{MonteCarloError, Result};

/// Prelude for convenient imports
pub mod prelude {
    pub use super::engine::{
        percentile, Budget, ConvergenceDiagnostics, MonteCarloEngine, MonteCarloRng, PathMetadata,
        Percentiles, SimulationModel, SimulationPath, SimulationResult, Statistics, TimeHorizon,
        TimeStep, VarianceReduction,
    };
    pub use super::error::{MonteCarloError, Result};
    pub use super::models::{EmpiricalBootstrap, GeometricBrownianMotion, MertonJumpDiffusion};
    pub use super::risk::{
        calmar_ratio, gain_to_pain_ratio, information_ratio, jensens_alpha, omega_ratio,
        sharpe_ratio, sharpe_ratio_annualized, sortino_ratio, treynor_ratio, CVaR,
        DrawdownAnalysis, DrawdownStatistics, RiskReport, VaR,
    };
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_prelude_imports() {
        // Verify all prelude items are accessible
        let _rng = MonteCarloRng::new(42);
        let _engine = MonteCarloEngine::reproducible(42);
        let _horizon = TimeHorizon::years(1);
    }

    #[test]
    fn test_end_to_end_simulation() {
        let engine = MonteCarloEngine::reproducible(42)
            .with_n_simulations(100)
            .with_variance_reduction(VarianceReduction::Antithetic);

        let model = GeometricBrownianMotion::new(100.0, 0.08, 0.20);
        let horizon = TimeHorizon::years(1);
        let result = engine.simulate(&model, &horizon);

        assert_eq!(result.n_paths(), 100);

        let stats = result.final_value_statistics();
        assert!(stats.mean > 0.0);
    }

    #[test]
    fn test_risk_report_integration() {
        let mut rng = MonteCarloRng::new(42);
        let paths: Vec<SimulationPath> = (0..100)
            .map(|i| {
                let values: Vec<f64> = (0..12)
                    .map(|j| {
                        if j == 0 {
                            100.0
                        } else {
                            100.0 * (1.0 + rng.normal(0.01, 0.05))
                        }
                    })
                    .collect();

                SimulationPath::new(
                    (0..12).map(|j| j as f64 / 12.0).collect(),
                    values,
                    PathMetadata {
                        path_id: i,
                        seed: 42,
                        is_antithetic: false,
                    },
                )
            })
            .collect();

        let report = RiskReport::from_paths(&paths, 0.02).expect("Should generate report");

        assert!(report.var_90 >= 0.0);
        assert!(report.cvar_90 >= report.var_90 - 0.01);
        assert!(report.sharpe_ratio.is_finite());
    }
}
