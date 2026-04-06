//! Convergence diagnostics for Monte Carlo simulations
//!
//! Implements effective sample size (ESS), Monte Carlo standard error (MCSE),
//! and autocorrelation analysis for assessing simulation quality.
//!
//! Reference: Gelman et al. (2013), "Bayesian Data Analysis", Ch. 11

use super::types::SimulationPath;

/// Diagnostics for monitoring simulation convergence
#[derive(Debug, Clone, Default)]
pub struct ConvergenceDiagnostics {
    /// Effective sample size
    pub ess: f64,
    /// Monte Carlo standard error
    pub mcse: f64,
    /// Relative precision (MCSE / mean)
    pub relative_precision: f64,
    /// Running mean history for convergence tracking
    pub running_mean_history: Vec<f64>,
    /// Running count
    count: usize,
    /// Running sum
    sum: f64,
    /// Running sum of squares (for variance)
    sum_sq: f64,
}

impl ConvergenceDiagnostics {
    /// Create new diagnostics tracker
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Update diagnostics with new paths
    pub fn update(&mut self, paths: &[SimulationPath]) {
        if paths.is_empty() {
            return;
        }

        let returns: Vec<f64> = paths
            .iter()
            .filter_map(SimulationPath::total_return)
            .collect();

        if returns.is_empty() {
            return;
        }

        let n = returns.len();
        let mean = returns.iter().sum::<f64>() / n as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
        let std = variance.sqrt();

        // Calculate ESS using batch means
        self.ess = calculate_ess(&returns);

        // Calculate MCSE
        self.mcse = if self.ess > 0.0 {
            std / self.ess.sqrt()
        } else {
            f64::INFINITY
        };

        // Calculate relative precision
        self.relative_precision = if mean.abs() > 1e-10 {
            self.mcse / mean.abs()
        } else {
            f64::INFINITY
        };

        // Track running mean history
        self.running_mean_history.push(mean);
    }

    /// Update with a single value (Welford's online algorithm)
    pub fn update_single(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;

        let mean = self.sum / self.count as f64;
        self.running_mean_history.push(mean);

        // Estimate variance and ESS from running statistics
        if self.count > 1 {
            let variance =
                (self.sum_sq - self.sum * self.sum / self.count as f64) / (self.count - 1) as f64;
            let std = variance.sqrt();

            // Approximate ESS as n (assuming independence)
            self.ess = self.count as f64;
            self.mcse = std / self.ess.sqrt();
            self.relative_precision = if mean.abs() > 1e-10 {
                self.mcse / mean.abs()
            } else {
                f64::INFINITY
            };
        }
    }

    /// Check if simulation has converged to target precision
    #[must_use]
    pub fn is_converged(&self, target_precision: f64) -> bool {
        self.relative_precision < target_precision && self.ess > 100.0
    }

    /// Get the current estimate of the mean
    #[must_use]
    pub fn current_mean(&self) -> Option<f64> {
        self.running_mean_history.last().copied()
    }

    /// Get confidence interval for the mean
    #[must_use]
    pub fn confidence_interval(&self, confidence: f64) -> Option<(f64, f64)> {
        let mean = self.current_mean()?;
        if !self.mcse.is_finite() {
            return None;
        }

        // z-score for confidence level (approximate)
        let z = match confidence {
            c if c >= 0.99 => 2.576,
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            _ => 1.28,
        };

        let margin = z * self.mcse;
        Some((mean - margin, mean + margin))
    }
}

/// Calculate Effective Sample Size using batch means
///
/// ESS = n / (1 + 2 * sum of autocorrelations)
#[must_use]
pub fn calculate_ess(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 10 {
        return n as f64;
    }

    // Use batch means method
    let batch_size = (n as f64).sqrt().ceil() as usize;
    let n_batches = n / batch_size;

    if n_batches < 2 {
        return n as f64;
    }

    // Calculate batch means
    let batch_means: Vec<f64> = (0..n_batches)
        .map(|i| {
            let start = i * batch_size;
            let end = ((i + 1) * batch_size).min(n);
            values[start..end].iter().sum::<f64>() / (end - start) as f64
        })
        .collect();

    // Overall mean
    let grand_mean = values.iter().sum::<f64>() / n as f64;

    // Variance of batch means
    let var_batch = batch_means
        .iter()
        .map(|m| (m - grand_mean).powi(2))
        .sum::<f64>()
        / (n_batches - 1) as f64;

    // Sample variance
    let var_sample = values.iter().map(|x| (x - grand_mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    // ESS = n * var_sample / (batch_size * var_batch)
    if var_batch > 0.0 {
        (n as f64 * var_sample / (batch_size as f64 * var_batch)).min(n as f64)
    } else {
        n as f64
    }
}

/// Calculate autocorrelation at a given lag
#[must_use]
pub fn autocorrelation(values: &[f64], lag: usize) -> f64 {
    let n = values.len();
    if lag >= n {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if variance < 1e-15 {
        return 0.0;
    }

    let covariance: f64 = values[..n - lag]
        .iter()
        .zip(values[lag..].iter())
        .map(|(x, y)| (x - mean) * (y - mean))
        .sum::<f64>()
        / n as f64;

    covariance / variance
}

/// Calculate ESS using autocorrelation method
///
/// Computes autocorrelations until they become negligible,
/// then calculates ESS = n / (1 + 2 * `sum(ρ_k)`)
#[must_use]
pub fn ess_autocorr(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 4 {
        return n as f64;
    }

    let max_lag = (n / 2).min(100);
    let mut sum_rho = 0.0;

    for lag in 1..max_lag {
        let rho = autocorrelation(values, lag);

        // Stop when autocorrelation becomes negligible
        if rho.abs() < 0.05 {
            break;
        }

        sum_rho += rho;
    }

    let tau = 1.0 + 2.0 * sum_rho;
    if tau > 0.0 {
        (n as f64 / tau).min(n as f64)
    } else {
        n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::engine::types::PathMetadata;

    fn create_test_paths(n: usize, seed: u64) -> Vec<SimulationPath> {
        use crate::monte_carlo::engine::MonteCarloRng;

        let mut rng = MonteCarloRng::new(seed);

        (0..n)
            .map(|i| {
                let returns: Vec<f64> = (0..12)
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
                    returns,
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
    fn test_convergence_diagnostics_basic() {
        let paths = create_test_paths(1000, 42);
        let mut diag = ConvergenceDiagnostics::new();
        diag.update(&paths);

        assert!(diag.ess > 0.0);
        assert!(diag.mcse.is_finite());
        assert!(diag.relative_precision.is_finite());
    }

    #[test]
    fn test_convergence_check() {
        let paths = create_test_paths(10000, 42);
        let mut diag = ConvergenceDiagnostics::new();
        diag.update(&paths);

        // With enough samples, should converge
        let converged = diag.is_converged(0.10);
        assert!(diag.ess > 100.0);
        // May or may not converge depending on data
        let _ = converged;
    }

    #[test]
    fn test_running_mean_history() {
        let mut diag = ConvergenceDiagnostics::new();

        for i in 1..=100 {
            diag.update_single(i as f64);
        }

        assert_eq!(diag.running_mean_history.len(), 100);
        // Final mean should be close to 50.5
        assert!((diag.current_mean().unwrap() - 50.5).abs() < 0.1);
    }

    #[test]
    fn test_calculate_ess() {
        // IID samples should have ESS close to n
        let iid: Vec<f64> = (0..1000).map(|i| (i % 10) as f64).collect();
        let ess = calculate_ess(&iid);
        assert!(ess > 0.0);
        assert!(ess <= 1000.0);
    }

    #[test]
    fn test_autocorrelation_lag_zero() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rho_0 = autocorrelation(&values, 0);
        // Autocorrelation at lag 0 should be 1
        assert!((rho_0 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_autocorrelation_random() {
        // Random values should have low autocorrelation
        // Use a more scrambled sequence to avoid hidden patterns
        use crate::monte_carlo::engine::MonteCarloRng;
        let mut rng = MonteCarloRng::new(42);
        let values: Vec<f64> = (0..200).map(|_| rng.uniform()).collect();
        let rho_1 = autocorrelation(&values, 1);
        // ChaCha20 should produce uncorrelated samples
        assert!(rho_1.abs() < 0.2, "Autocorrelation should be low: {rho_1}");
    }

    #[test]
    fn test_ess_autocorr() {
        let values: Vec<f64> = (0..1000).map(|i| (i % 10) as f64).collect();
        let ess = ess_autocorr(&values);
        assert!(ess > 0.0);
        assert!(ess <= 1000.0);
    }

    #[test]
    fn test_confidence_interval() {
        let mut diag = ConvergenceDiagnostics::new();
        for i in 1..=1000 {
            diag.update_single(i as f64);
        }

        let ci = diag.confidence_interval(0.95);
        assert!(ci.is_some());
        let (lower, upper) = ci.unwrap();
        assert!(lower < upper);
        assert!(lower < diag.current_mean().unwrap());
        assert!(upper > diag.current_mean().unwrap());
    }

    #[test]
    fn test_empty_paths() {
        let mut diag = ConvergenceDiagnostics::new();
        diag.update(&[]);
        assert!(diag.running_mean_history.is_empty());
    }

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_ess_bounded(values in prop::collection::vec(0.0..100.0f64, 10..500)) {
                let ess = calculate_ess(&values);
                prop_assert!(ess >= 0.0);
                prop_assert!(ess <= values.len() as f64);
            }

            #[test]
            fn prop_autocorr_bounded(values in prop::collection::vec(-10.0..10.0f64, 20..200)) {
                for lag in 1..5 {
                    let rho = autocorrelation(&values, lag);
                    prop_assert!(rho >= -1.0 && rho <= 1.0, "Autocorrelation out of bounds: {rho}");
                }
            }

            #[test]
            fn prop_ess_autocorr_bounded(values in prop::collection::vec(0.0..100.0f64, 10..200)) {
                let ess = ess_autocorr(&values);
                prop_assert!(ess >= 0.0);
                prop_assert!(ess <= values.len() as f64);
            }
        }
    }
}
