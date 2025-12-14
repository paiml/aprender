//! Risk metrics for Monte Carlo simulations
//!
//! Implements Value-at-Risk (VaR), Conditional VaR (CVaR/Expected Shortfall),
//! maximum drawdown, and risk-adjusted return metrics.
//!
//! References:
//! - Jorion (2006), "Value at Risk"
//! - Artzner et al. (1999), "Coherent Measures of Risk"

mod cvar;
mod drawdown;
mod ratios;
mod var;

pub use cvar::CVaR;
pub use drawdown::{DrawdownAnalysis, DrawdownStatistics};
pub use ratios::{
    calmar_ratio, gain_to_pain_ratio, information_ratio, jensens_alpha, omega_ratio, sharpe_ratio,
    sharpe_ratio_annualized, sortino_ratio, treynor_ratio,
};
pub use var::VaR;

use crate::monte_carlo::engine::{SimulationPath, Statistics};
use crate::monte_carlo::error::{MonteCarloError, Result};

/// Comprehensive risk report from simulation
#[derive(Debug, Clone)]
pub struct RiskReport {
    /// Value at Risk at various confidence levels
    pub var_90: f64,
    /// VaR at 95% confidence
    pub var_95: f64,
    /// VaR at 99% confidence
    pub var_99: f64,

    /// Conditional VaR (Expected Shortfall) at 90%
    pub cvar_90: f64,
    /// CVaR at 95% confidence
    pub cvar_95: f64,
    /// CVaR at 99% confidence
    pub cvar_99: f64,

    /// Drawdown statistics
    pub drawdown: DrawdownStatistics,

    /// Risk-adjusted return ratios
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,

    /// Return statistics
    pub return_statistics: Statistics,
}

impl RiskReport {
    /// Generate comprehensive risk report from simulation paths
    pub fn from_paths(paths: &[SimulationPath], risk_free_rate: f64) -> Result<Self> {
        if paths.is_empty() {
            return Err(MonteCarloError::EmptyData {
                context: "Cannot generate risk report from empty paths".to_string(),
            });
        }

        let returns: Vec<f64> = paths
            .iter()
            .filter_map(SimulationPath::total_return)
            .collect();

        if returns.is_empty() {
            return Err(MonteCarloError::EmptyData {
                context: "No valid returns in paths".to_string(),
            });
        }

        let return_statistics = Statistics::from_values(&returns);

        // VaR calculations
        let var_90 = VaR::historical(&returns, 0.90);
        let var_95 = VaR::historical(&returns, 0.95);
        let var_99 = VaR::historical(&returns, 0.99);

        // CVaR calculations
        let cvar_90 = CVaR::from_returns(&returns, 0.90);
        let cvar_95 = CVaR::from_returns(&returns, 0.95);
        let cvar_99 = CVaR::from_returns(&returns, 0.99);

        // Drawdown analysis
        let drawdown = DrawdownAnalysis::from_paths(paths);

        // Risk-adjusted ratios
        let sharpe = sharpe_ratio(&returns, risk_free_rate);
        let sortino = sortino_ratio(&returns, risk_free_rate, 0.0);
        let calmar = calmar_ratio(return_statistics.mean, drawdown.median);

        Ok(Self {
            var_90,
            var_95,
            var_99,
            cvar_90,
            cvar_95,
            cvar_99,
            drawdown,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            calmar_ratio: calmar,
            return_statistics,
        })
    }

    /// Generate a text summary of the risk report
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Risk Report\n\
             ===========\n\
             \n\
             Returns:\n\
             - Mean: {:.2}%\n\
             - Std Dev: {:.2}%\n\
             - Min: {:.2}%\n\
             - Max: {:.2}%\n\
             \n\
             Value at Risk:\n\
             - VaR(90%): {:.2}%\n\
             - VaR(95%): {:.2}%\n\
             - VaR(99%): {:.2}%\n\
             \n\
             Expected Shortfall (CVaR):\n\
             - CVaR(90%): {:.2}%\n\
             - CVaR(95%): {:.2}%\n\
             - CVaR(99%): {:.2}%\n\
             \n\
             Drawdown:\n\
             - Mean Max Drawdown: {:.2}%\n\
             - Median Max Drawdown: {:.2}%\n\
             - 95th Percentile: {:.2}%\n\
             - Worst Case: {:.2}%\n\
             \n\
             Risk-Adjusted Returns:\n\
             - Sharpe Ratio: {:.2}\n\
             - Sortino Ratio: {:.2}\n\
             - Calmar Ratio: {:.2}",
            self.return_statistics.mean * 100.0,
            self.return_statistics.std * 100.0,
            self.return_statistics.min * 100.0,
            self.return_statistics.max * 100.0,
            self.var_90 * 100.0,
            self.var_95 * 100.0,
            self.var_99 * 100.0,
            self.cvar_90 * 100.0,
            self.cvar_95 * 100.0,
            self.cvar_99 * 100.0,
            self.drawdown.mean * 100.0,
            self.drawdown.median * 100.0,
            self.drawdown.p95 * 100.0,
            self.drawdown.worst * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.calmar_ratio,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::engine::{MonteCarloRng, PathMetadata};

    fn create_test_paths(n: usize, seed: u64) -> Vec<SimulationPath> {
        let mut rng = MonteCarloRng::new(seed);

        (0..n)
            .map(|i| {
                let returns: Vec<f64> = (0..12)
                    .map(|j| {
                        if j == 0 {
                            100.0
                        } else {
                            // Simulate monthly returns
                            let prev = if j > 0 {
                                100.0 + (j as f64 - 1.0) * 2.0
                            } else {
                                100.0
                            };
                            prev * (1.0 + rng.normal(0.01, 0.05))
                        }
                    })
                    .collect();

                let mut values = vec![100.0];
                for r in returns.iter().skip(1) {
                    values.push(*r);
                }

                SimulationPath::new(
                    (0..12).map(|j| j as f64 / 12.0).collect(),
                    values,
                    PathMetadata {
                        path_id: i,
                        seed,
                        is_antithetic: false,
                    },
                )
            })
            .collect()
    }

    #[test]
    fn test_risk_report_from_paths() {
        let paths = create_test_paths(1000, 42);
        let report = RiskReport::from_paths(&paths, 0.02).unwrap();

        // VaR should be increasing with confidence
        assert!(report.var_90 <= report.var_95);
        assert!(report.var_95 <= report.var_99);

        // CVaR should be >= VaR (it's expected shortfall beyond VaR)
        assert!(report.cvar_90 >= report.var_90 - 0.001);
        assert!(report.cvar_95 >= report.var_95 - 0.001);
        assert!(report.cvar_99 >= report.var_99 - 0.001);
    }

    #[test]
    fn test_risk_report_empty_paths() {
        let result = RiskReport::from_paths(&[], 0.02);
        assert!(result.is_err());
    }

    #[test]
    fn test_risk_report_summary() {
        let paths = create_test_paths(100, 42);
        let report = RiskReport::from_paths(&paths, 0.02).unwrap();
        let summary = report.summary();

        assert!(summary.contains("Risk Report"));
        assert!(summary.contains("Value at Risk"));
        assert!(summary.contains("Expected Shortfall"));
        assert!(summary.contains("Sharpe Ratio"));
    }
}
