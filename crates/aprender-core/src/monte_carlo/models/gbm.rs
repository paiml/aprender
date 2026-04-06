//! Geometric Brownian Motion (GBM) model
//!
//! The standard model for stock price dynamics:
//! dS = μS dt + σS dW
//!
//! Reference: Black & Scholes (1973)

use crate::monte_carlo::engine::{
    MonteCarloRng, PathMetadata, SimulationModel, SimulationPath, TimeHorizon,
};

/// Geometric Brownian Motion model for stock prices
///
/// Models stock prices as:
/// S(t) = S(0) × exp((μ - σ²/2)t + σW(t))
///
/// # Example
/// ```
/// use aprender::monte_carlo::prelude::*;
///
/// let model = GeometricBrownianMotion::new(100.0, 0.08, 0.20);
/// let engine = MonteCarloEngine::reproducible(42).with_n_simulations(1000);
/// let result = engine.simulate(&model, &TimeHorizon::years(1));
///
/// assert!(result.final_value_statistics().mean > 100.0);
/// ```
#[derive(Debug, Clone)]
pub struct GeometricBrownianMotion {
    /// Initial stock price
    pub initial_price: f64,
    /// Drift rate (expected return, annualized)
    pub mu: f64,
    /// Volatility (standard deviation, annualized)
    pub sigma: f64,
    /// Dividend yield (continuous, annualized)
    pub dividend_yield: f64,
}

impl GeometricBrownianMotion {
    /// Create a new GBM model
    ///
    /// # Arguments
    /// * `initial_price` - Starting price (S₀)
    /// * `mu` - Drift rate (expected return, e.g., 0.08 for 8%)
    /// * `sigma` - Volatility (e.g., 0.20 for 20%)
    #[must_use]
    pub fn new(initial_price: f64, mu: f64, sigma: f64) -> Self {
        Self {
            initial_price,
            mu,
            sigma,
            dividend_yield: 0.0,
        }
    }

    /// Add dividend yield to the model
    #[must_use]
    pub fn with_dividend_yield(mut self, yield_rate: f64) -> Self {
        self.dividend_yield = yield_rate;
        self
    }

    /// Calculate expected price at time t
    #[must_use]
    pub fn expected_price(&self, t: f64) -> f64 {
        self.initial_price * ((self.mu - self.dividend_yield) * t).exp()
    }

    /// Calculate price variance at time t
    #[must_use]
    pub fn price_variance(&self, t: f64) -> f64 {
        let s0 = self.initial_price;
        let mu_adj = self.mu - self.dividend_yield;

        s0 * s0 * (2.0 * mu_adj * t).exp() * ((self.sigma * self.sigma * t).exp() - 1.0)
    }
}

impl SimulationModel for GeometricBrownianMotion {
    fn name(&self) -> &'static str {
        "GeometricBrownianMotion"
    }

    fn generate_path(
        &self,
        rng: &mut MonteCarloRng,
        time_horizon: &TimeHorizon,
        path_id: usize,
    ) -> SimulationPath {
        let n = time_horizon.n_steps();
        let dt = time_horizon.dt();
        let sqrt_dt = dt.sqrt();

        // Adjusted drift for dividends
        let drift = (self.mu - self.dividend_yield - 0.5 * self.sigma * self.sigma) * dt;

        let mut values = Vec::with_capacity(n + 1);
        let mut price = self.initial_price;
        values.push(price);

        for _ in 0..n {
            let z = rng.standard_normal();
            let log_return = drift + self.sigma * sqrt_dt * z;
            price *= log_return.exp();
            values.push(price);
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

    fn generate_antithetic_path(
        &self,
        rng: &mut MonteCarloRng,
        time_horizon: &TimeHorizon,
        path_id: usize,
    ) -> SimulationPath {
        let n = time_horizon.n_steps();
        let dt = time_horizon.dt();
        let sqrt_dt = dt.sqrt();

        let drift = (self.mu - self.dividend_yield - 0.5 * self.sigma * self.sigma) * dt;

        let mut values = Vec::with_capacity(n + 1);
        let mut price = self.initial_price;
        values.push(price);

        for _ in 0..n {
            // Use negated random numbers for antithetic path
            let z = -rng.standard_normal();
            let log_return = drift + self.sigma * sqrt_dt * z;
            price *= log_return.exp();
            values.push(price);
        }

        SimulationPath::new(
            time_horizon.time_points(),
            values,
            PathMetadata {
                path_id,
                seed: rng.seed(),
                is_antithetic: true,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::engine::{MonteCarloEngine, VarianceReduction};

    #[test]
    fn test_gbm_basic() {
        let model = GeometricBrownianMotion::new(100.0, 0.08, 0.20);
        let engine = MonteCarloEngine::reproducible(42).with_n_simulations(1000);
        let horizon = TimeHorizon::years(1);

        let result = engine.simulate(&model, &horizon);

        assert_eq!(result.n_paths(), 1000);

        let stats = result.final_value_statistics();
        // With 8% drift, expected value should be around 108
        assert!(stats.mean > 100.0, "Mean = {}", stats.mean);
        assert!(stats.std > 0.0);
    }

    #[test]
    fn test_gbm_reproducibility() {
        let model = GeometricBrownianMotion::new(100.0, 0.08, 0.20);
        let horizon = TimeHorizon::years(1);

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
    fn test_gbm_with_dividend() {
        let model = GeometricBrownianMotion::new(100.0, 0.08, 0.20).with_dividend_yield(0.02);
        let engine = MonteCarloEngine::reproducible(42).with_n_simulations(1000);
        let horizon = TimeHorizon::years(1);

        let result = engine.simulate(&model, &horizon);
        let stats = result.final_value_statistics();

        // With 2% dividend, effective return should be around 6%
        assert!(stats.mean > 100.0);
    }

    #[test]
    fn test_gbm_expected_price() {
        let model = GeometricBrownianMotion::new(100.0, 0.10, 0.20);

        let expected_1y = model.expected_price(1.0);
        // E[S(1)] = 100 * exp(0.10) ≈ 110.52
        assert!((expected_1y - 110.52).abs() < 0.1);
    }

    #[test]
    fn test_gbm_variance_reduction() {
        let model = GeometricBrownianMotion::new(100.0, 0.08, 0.20);
        let horizon = TimeHorizon::years(1);

        // Without antithetic
        let result_std = MonteCarloEngine::reproducible(42)
            .with_n_simulations(1000)
            .with_variance_reduction(VarianceReduction::None)
            .simulate(&model, &horizon);

        // With antithetic
        let result_anti = MonteCarloEngine::reproducible(42)
            .with_n_simulations(1000)
            .with_variance_reduction(VarianceReduction::Antithetic)
            .simulate(&model, &horizon);

        // Means should be similar
        let mean_std = result_std.final_value_statistics().mean;
        let mean_anti = result_anti.final_value_statistics().mean;

        assert!(
            (mean_std - mean_anti).abs() / mean_std < 0.10,
            "Means should be similar: {} vs {}",
            mean_std,
            mean_anti
        );
    }

    #[test]
    fn test_gbm_time_evolution() {
        let model = GeometricBrownianMotion::new(100.0, 0.08, 0.20);
        let engine = MonteCarloEngine::reproducible(42).with_n_simulations(100);
        let horizon =
            TimeHorizon::years(1).with_step(crate::monte_carlo::engine::TimeStep::Monthly);

        let result = engine.simulate(&model, &horizon);
        let stats_over_time = result.statistics_over_time();

        assert_eq!(stats_over_time.len(), 13); // 12 months + initial

        // Mean should generally increase over time
        assert!(stats_over_time.last().unwrap().mean > stats_over_time.first().unwrap().mean);
    }

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_gbm_positive_prices(
                initial in 10.0..1000.0f64,
                mu in -0.2..0.5f64,
                sigma in 0.05..0.8f64,
                seed: u64,
            ) {
                let model = GeometricBrownianMotion::new(initial, mu, sigma);
                let engine = MonteCarloEngine::reproducible(seed).with_n_simulations(100);
                let horizon = TimeHorizon::years(1);

                let result = engine.simulate(&model, &horizon);

                for path in &result.paths {
                    for &value in &path.values {
                        prop_assert!(value > 0.0, "GBM prices must be positive: {value}");
                    }
                }
            }

            #[test]
            fn prop_gbm_expected_value_convergence(
                initial in 50.0..200.0f64,
                mu in 0.0..0.2f64,
                sigma in 0.1..0.4f64,
            ) {
                let model = GeometricBrownianMotion::new(initial, mu, sigma);
                let engine = MonteCarloEngine::reproducible(42).with_n_simulations(10000);
                let horizon = TimeHorizon::years(1);

                let result = engine.simulate(&model, &horizon);
                let expected = model.expected_price(1.0);
                let actual = result.final_value_statistics().mean;

                // Should be within 10% of expected
                let relative_error = (actual - expected).abs() / expected;
                prop_assert!(
                    relative_error < 0.10,
                    "Expected {expected}, got {actual}, error {relative_error}"
                );
            }
        }
    }
}
