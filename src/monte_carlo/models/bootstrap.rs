//! Empirical Bootstrap Model
//!
//! Simulates future prices by resampling from historical returns.
//! Non-parametric approach that preserves empirical distribution properties.
//!
//! Reference: Efron (1979), "Bootstrap Methods: Another Look at the Jackknife"

use crate::monte_carlo::engine::{
    MonteCarloRng, PathMetadata, SimulationModel, SimulationPath, TimeHorizon,
};

/// Empirical Bootstrap model for stock prices
///
/// Generates future price paths by randomly sampling from
/// historical returns with replacement.
///
/// # Advantages
/// - Preserves empirical distribution (fat tails, skewness)
/// - No parametric assumptions
/// - Captures actual market behavior
///
/// # Example
/// ```
/// use aprender::monte_carlo::prelude::*;
///
/// // Historical monthly returns
/// let returns = vec![0.02, -0.01, 0.03, 0.01, -0.02, 0.04, -0.03, 0.02];
///
/// let model = EmpiricalBootstrap::new(100.0, returns);
/// let engine = MonteCarloEngine::reproducible(42).with_n_simulations(1000);
/// let result = engine.simulate(&model, &TimeHorizon::months(12));
/// ```
#[derive(Debug, Clone)]
pub struct EmpiricalBootstrap {
    /// Initial price
    pub initial_price: f64,
    /// Historical returns to sample from
    pub historical_returns: Vec<f64>,
    /// Use block bootstrap for autocorrelation preservation
    pub block_size: Option<usize>,
}

impl EmpiricalBootstrap {
    /// Create a new empirical bootstrap model
    ///
    /// # Arguments
    /// * `initial_price` - Starting price
    /// * `historical_returns` - Vector of historical period returns
    #[must_use]
    pub fn new(initial_price: f64, historical_returns: Vec<f64>) -> Self {
        Self {
            initial_price,
            historical_returns,
            block_size: None,
        }
    }

    /// Enable block bootstrap for preserving autocorrelation
    ///
    /// Block size determines the length of consecutive returns
    /// sampled together, preserving short-term momentum effects.
    #[must_use]
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = Some(block_size);
        self
    }

    /// Get summary statistics of historical returns
    #[must_use]
    pub fn historical_stats(&self) -> HistoricalStats {
        if self.historical_returns.is_empty() {
            return HistoricalStats::default();
        }

        let n = self.historical_returns.len() as f64;
        let mean = self.historical_returns.iter().sum::<f64>() / n;
        let variance = self
            .historical_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0).max(1.0);
        let std = variance.sqrt();

        let min = self
            .historical_returns
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .historical_returns
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        HistoricalStats {
            mean,
            std,
            min,
            max,
            n: self.historical_returns.len(),
        }
    }

    /// Sample a single return
    fn sample_return(&self, rng: &mut MonteCarloRng) -> f64 {
        if self.historical_returns.is_empty() {
            return 0.0;
        }

        let idx = (rng.uniform() * self.historical_returns.len() as f64) as usize;
        self.historical_returns[idx.min(self.historical_returns.len() - 1)]
    }

    /// Sample a block of consecutive returns
    fn sample_block(&self, rng: &mut MonteCarloRng, block_size: usize) -> Vec<f64> {
        if self.historical_returns.is_empty() {
            return vec![0.0; block_size];
        }

        let max_start = self.historical_returns.len().saturating_sub(block_size);
        let start = if max_start > 0 {
            (rng.uniform() * max_start as f64) as usize
        } else {
            0
        };

        let end = (start + block_size).min(self.historical_returns.len());
        self.historical_returns[start..end].to_vec()
    }
}

/// Summary statistics for historical returns
#[derive(Debug, Clone, Default)]
pub struct HistoricalStats {
    /// Mean return
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Minimum return
    pub min: f64,
    /// Maximum return
    pub max: f64,
    /// Number of observations
    pub n: usize,
}

impl SimulationModel for EmpiricalBootstrap {
    fn name(&self) -> &'static str {
        "EmpiricalBootstrap"
    }

    fn generate_path(
        &self,
        rng: &mut MonteCarloRng,
        time_horizon: &TimeHorizon,
        path_id: usize,
    ) -> SimulationPath {
        let n = time_horizon.n_steps();

        let mut values = Vec::with_capacity(n + 1);
        let mut price = self.initial_price;
        values.push(price);

        if let Some(block_size) = self.block_size {
            // Block bootstrap
            let mut remaining = n;
            while remaining > 0 {
                let block = self.sample_block(rng, block_size.min(remaining));
                for &ret in &block {
                    price *= 1.0 + ret;
                    values.push(price);
                }
                remaining = remaining.saturating_sub(block.len());
            }
        } else {
            // Simple bootstrap
            for _ in 0..n {
                let ret = self.sample_return(rng);
                price *= 1.0 + ret;
                values.push(price);
            }
        }

        SimulationPath::new(
            time_horizon.time_points(),
            values,
            PathMetadata {
                path_id,
                seed: rng.seed(),
                is_antithetic: false,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::engine::MonteCarloEngine;

    fn sample_returns() -> Vec<f64> {
        vec![
            0.02, -0.01, 0.03, 0.01, -0.02, 0.04, -0.03, 0.02, 0.01, -0.01, 0.02, 0.03,
        ]
    }

    #[test]
    fn test_bootstrap_basic() {
        let model = EmpiricalBootstrap::new(100.0, sample_returns());
        let engine = MonteCarloEngine::reproducible(42).with_n_simulations(1000);
        let horizon = TimeHorizon::months(12);

        let result = engine.simulate(&model, &horizon);

        assert_eq!(result.n_paths(), 1000);
        assert!(result.final_value_statistics().mean > 0.0);
    }

    #[test]
    fn test_bootstrap_positive_prices() {
        let model = EmpiricalBootstrap::new(100.0, sample_returns());
        let engine = MonteCarloEngine::reproducible(42).with_n_simulations(100);
        let horizon = TimeHorizon::years(1);

        let result = engine.simulate(&model, &horizon);

        for path in &result.paths {
            for &value in &path.values {
                assert!(value > 0.0, "Prices must stay positive");
            }
        }
    }

    #[test]
    fn test_bootstrap_reproducibility() {
        let model = EmpiricalBootstrap::new(100.0, sample_returns());
        let horizon = TimeHorizon::months(12);

        let result1 = MonteCarloEngine::reproducible(42)
            .with_n_simulations(100)
            .simulate(&model, &horizon);

        let result2 = MonteCarloEngine::reproducible(42)
            .with_n_simulations(100)
            .simulate(&model, &horizon);

        for (p1, p2) in result1.paths.iter().zip(result2.paths.iter()) {
            for (v1, v2) in p1.values.iter().zip(p2.values.iter()) {
                assert!((v1 - v2).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_bootstrap_block() {
        let model = EmpiricalBootstrap::new(100.0, sample_returns()).with_block_size(3);
        let engine = MonteCarloEngine::reproducible(42).with_n_simulations(100);
        let horizon = TimeHorizon::months(12);

        let result = engine.simulate(&model, &horizon);

        assert!(result.n_paths() > 0);
    }

    #[test]
    fn test_historical_stats() {
        let model = EmpiricalBootstrap::new(100.0, sample_returns());
        let stats = model.historical_stats();

        assert_eq!(stats.n, 12);
        assert!(stats.std > 0.0);
        assert!(stats.min < stats.max);
    }

    #[test]
    fn test_empty_returns() {
        let model = EmpiricalBootstrap::new(100.0, vec![]);
        let engine = MonteCarloEngine::reproducible(42).with_n_simulations(10);
        let horizon = TimeHorizon::months(6);

        let result = engine.simulate(&model, &horizon);

        // Should still work, just with zero returns
        for path in &result.paths {
            assert!((path.values[0] - 100.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_preserves_return_distribution() {
        use crate::monte_carlo::engine::TimeStep;

        // Bootstrap should approximately preserve the mean and std of returns
        let returns = sample_returns();
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;

        let model = EmpiricalBootstrap::new(100.0, returns);
        let engine = MonteCarloEngine::reproducible(42).with_n_simulations(10000);
        // Use monthly step to match the assumed frequency of historical returns
        let horizon = TimeHorizon::months(1).with_step(TimeStep::Monthly);

        let result = engine.simulate(&model, &horizon);

        // Calculate realized returns
        let realized_returns: Vec<f64> = result
            .paths
            .iter()
            .filter_map(|p| p.total_return())
            .collect();

        let realized_mean = realized_returns.iter().sum::<f64>() / realized_returns.len() as f64;

        // Should be close to historical mean
        assert!(
            (realized_mean - mean).abs() < 0.01,
            "Realized mean {} should be close to historical mean {}",
            realized_mean,
            mean
        );
    }

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_bootstrap_positive_prices(
                initial in 10.0..1000.0f64,
                returns in prop::collection::vec(-0.2..0.2f64, 10..50),
                seed: u64,
            ) {
                let model = EmpiricalBootstrap::new(initial, returns);
                let engine = MonteCarloEngine::reproducible(seed).with_n_simulations(50);
                let horizon = TimeHorizon::months(12);

                let result = engine.simulate(&model, &horizon);

                for path in &result.paths {
                    for &value in &path.values {
                        prop_assert!(value > 0.0, "Prices must be positive: {value}");
                    }
                }
            }
        }
    }
}
