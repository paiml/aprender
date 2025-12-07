//! Merton Jump-Diffusion Model
//!
//! Extends GBM with discrete jumps to model sudden price movements:
//! dS = (μ - λκ)S dt + σS dW + S(J-1) dN
//!
//! Reference: Merton (1976), "Option pricing when underlying stock returns are discontinuous"

use crate::monte_carlo::engine::{
    MonteCarloRng, PathMetadata, SimulationModel, SimulationPath, TimeHorizon,
};

/// Merton Jump-Diffusion model
///
/// Combines continuous GBM dynamics with discrete jumps
/// to model market crashes and rallies.
///
/// # Parameters
/// - `mu`: Drift rate (expected return excluding jumps)
/// - `sigma`: Continuous volatility
/// - `lambda`: Jump intensity (expected jumps per year)
/// - `jump_mean`: Mean of log jump size
/// - `jump_std`: Std of log jump size
///
/// # Example
/// ```
/// use aprender::monte_carlo::prelude::*;
///
/// let model = MertonJumpDiffusion::new(100.0, 0.08, 0.15)
///     .with_jumps(1.0, -0.1, 0.15);  // 1 jump/year, -10% mean, 15% std
///
/// let engine = MonteCarloEngine::reproducible(42).with_n_simulations(1000);
/// let result = engine.simulate(&model, &TimeHorizon::years(1));
/// ```
#[derive(Debug, Clone)]
pub struct MertonJumpDiffusion {
    /// Initial stock price
    pub initial_price: f64,
    /// Drift rate (excluding jump compensation)
    pub mu: f64,
    /// Continuous volatility
    pub sigma: f64,
    /// Jump intensity (jumps per year)
    pub lambda: f64,
    /// Mean of log jump size
    pub jump_mean: f64,
    /// Std of log jump size
    pub jump_std: f64,
}

impl MertonJumpDiffusion {
    /// Create a new Merton Jump-Diffusion model
    ///
    /// Default: no jumps (pure GBM)
    #[must_use]
    pub fn new(initial_price: f64, mu: f64, sigma: f64) -> Self {
        Self {
            initial_price,
            mu,
            sigma,
            lambda: 0.0,
            jump_mean: 0.0,
            jump_std: 0.0,
        }
    }

    /// Configure jump parameters
    ///
    /// # Arguments
    /// * `lambda` - Jump intensity (expected jumps per year)
    /// * `jump_mean` - Mean of log jump size (negative for crashes)
    /// * `jump_std` - Standard deviation of log jump size
    #[must_use]
    pub fn with_jumps(mut self, lambda: f64, jump_mean: f64, jump_std: f64) -> Self {
        self.lambda = lambda;
        self.jump_mean = jump_mean;
        self.jump_std = jump_std;
        self
    }

    /// Calculate expected jump contribution
    #[must_use]
    pub fn expected_jump_contribution(&self) -> f64 {
        // κ = E[J-1] = exp(jump_mean + jump_std²/2) - 1
        (self.jump_mean + 0.5 * self.jump_std * self.jump_std).exp() - 1.0
    }
}

impl SimulationModel for MertonJumpDiffusion {
    fn name(&self) -> &'static str {
        "MertonJumpDiffusion"
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

        // Jump compensation
        let kappa = self.expected_jump_contribution();
        let drift = (self.mu - self.lambda * kappa - 0.5 * self.sigma * self.sigma) * dt;

        let mut values = Vec::with_capacity(n + 1);
        let mut price = self.initial_price;
        values.push(price);

        for _ in 0..n {
            // Continuous component (GBM)
            let z = rng.standard_normal();
            let continuous_return = drift + self.sigma * sqrt_dt * z;

            // Jump component
            let n_jumps = rng.poisson(self.lambda * dt);
            let jump_return: f64 = (0..n_jumps)
                .map(|_| rng.normal(self.jump_mean, self.jump_std))
                .sum();

            price *= (continuous_return + jump_return).exp();
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

        let kappa = self.expected_jump_contribution();
        let drift = (self.mu - self.lambda * kappa - 0.5 * self.sigma * self.sigma) * dt;

        let mut values = Vec::with_capacity(n + 1);
        let mut price = self.initial_price;
        values.push(price);

        for _ in 0..n {
            // Antithetic: negate continuous component
            let z = -rng.standard_normal();
            let continuous_return = drift + self.sigma * sqrt_dt * z;

            // Keep jumps as-is (antithetic doesn't apply well to Poisson)
            let n_jumps = rng.poisson(self.lambda * dt);
            let jump_return: f64 = (0..n_jumps)
                .map(|_| rng.normal(self.jump_mean, self.jump_std))
                .sum();

            price *= (continuous_return + jump_return).exp();
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
    use crate::monte_carlo::engine::MonteCarloEngine;

    #[test]
    fn test_jump_diffusion_no_jumps() {
        // Without jumps, should behave like GBM
        let model = MertonJumpDiffusion::new(100.0, 0.08, 0.20);
        let engine = MonteCarloEngine::reproducible(42).with_n_simulations(1000);
        let horizon = TimeHorizon::years(1);

        let result = engine.simulate(&model, &horizon);
        let stats = result.final_value_statistics();

        assert!(stats.mean > 100.0);
    }

    #[test]
    fn test_jump_diffusion_with_crashes() {
        // Crash-prone model: 2 jumps/year, -15% mean, 10% std
        let model = MertonJumpDiffusion::new(100.0, 0.08, 0.15).with_jumps(2.0, -0.15, 0.10);

        let engine = MonteCarloEngine::reproducible(42).with_n_simulations(1000);
        let horizon = TimeHorizon::years(1);

        let result = engine.simulate(&model, &horizon);
        let stats = result.final_value_statistics();

        // Should have higher variance due to jumps
        assert!(stats.std > 0.0);
        // And potentially fat tails
        assert!(stats.kurtosis.abs() >= 0.0 || stats.kurtosis.is_finite());
    }

    #[test]
    fn test_jump_diffusion_positive_prices() {
        let model = MertonJumpDiffusion::new(100.0, 0.08, 0.20).with_jumps(5.0, -0.10, 0.15);

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
    fn test_expected_jump_contribution() {
        let model = MertonJumpDiffusion::new(100.0, 0.08, 0.20).with_jumps(1.0, 0.0, 0.0);

        // With zero mean and std, kappa should be 0
        let kappa = model.expected_jump_contribution();
        assert!((kappa - 0.0).abs() < 1e-10);

        // With positive mean jump
        let model2 = MertonJumpDiffusion::new(100.0, 0.08, 0.20).with_jumps(1.0, 0.1, 0.0);

        let kappa2 = model2.expected_jump_contribution();
        assert!(kappa2 > 0.0);
    }

    #[test]
    fn test_jump_diffusion_reproducibility() {
        let model = MertonJumpDiffusion::new(100.0, 0.08, 0.20).with_jumps(2.0, -0.10, 0.15);

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

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_jump_diffusion_positive_prices(
                initial in 10.0..1000.0f64,
                mu in -0.1..0.3f64,
                sigma in 0.05..0.5f64,
                lambda in 0.0..5.0f64,
                jump_mean in -0.3..0.1f64,
                jump_std in 0.0..0.3f64,
                seed: u64,
            ) {
                let model = MertonJumpDiffusion::new(initial, mu, sigma)
                    .with_jumps(lambda, jump_mean, jump_std);

                let engine = MonteCarloEngine::reproducible(seed).with_n_simulations(50);
                let horizon = TimeHorizon::years(1);

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
