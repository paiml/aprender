//! Business and financial models for Monte Carlo simulation
//!
//! This module provides domain-specific models that wrap the core
//! Monte Carlo functionality from `aprender::monte_carlo`.

use aprender::monte_carlo::prelude::*;

/// Bayesian Revenue Model for business forecasting
///
/// Models product revenue with uncertain growth using Bayesian priors.
/// Combines historical data with domain expertise for robust forecasting.
///
/// # Example
///
/// ```rust,ignore
/// use aprender_monte_carlo::models::BayesianRevenueModel;
///
/// let products = vec![
///     ProductData { name: "Widget", base_revenue: 100_000.0, growth_rate: 0.10, volatility: 0.15 },
///     ProductData { name: "Gadget", base_revenue: 50_000.0, growth_rate: 0.20, volatility: 0.25 },
/// ];
///
/// let model = BayesianRevenueModel::new(products);
/// let engine = MonteCarloEngine::reproducible(42).with_n_simulations(10_000);
/// let result = engine.simulate(&model, &TimeHorizon::quarters(8));
/// ```
#[derive(Debug, Clone)]
pub struct BayesianRevenueModel {
    /// Product data
    products: Vec<ProductData>,
    /// Use correlated simulation
    correlated: bool,
}

/// Individual product data
#[derive(Debug, Clone)]
pub struct ProductData {
    /// Product name
    pub name: String,
    /// Base revenue (current period)
    pub base_revenue: f64,
    /// Expected growth rate per period
    pub growth_rate: f64,
    /// Volatility (standard deviation of growth)
    pub volatility: f64,
}

impl BayesianRevenueModel {
    /// Create a new Bayesian revenue model
    #[must_use]
    pub fn new(products: Vec<ProductData>) -> Self {
        Self {
            products,
            correlated: false,
        }
    }

    /// Enable correlated simulation (products move together)
    #[must_use]
    pub fn with_correlation(mut self) -> Self {
        self.correlated = true;
        self
    }

    /// Get total base revenue
    #[must_use]
    pub fn total_base_revenue(&self) -> f64 {
        self.products.iter().map(|p| p.base_revenue).sum()
    }

    /// Get weighted average growth rate
    #[must_use]
    pub fn weighted_growth_rate(&self) -> f64 {
        let total = self.total_base_revenue();
        if total <= 0.0 {
            return 0.0;
        }
        self.products
            .iter()
            .map(|p| p.growth_rate * p.base_revenue)
            .sum::<f64>()
            / total
    }

    /// Get weighted average volatility
    #[must_use]
    pub fn weighted_volatility(&self) -> f64 {
        let total = self.total_base_revenue();
        if total <= 0.0 {
            return 0.0;
        }
        // Use sqrt of weighted variance for portfolio effect
        let weighted_var: f64 = self
            .products
            .iter()
            .map(|p| (p.volatility * p.base_revenue / total).powi(2))
            .sum();
        weighted_var.sqrt()
    }
}

impl SimulationModel for BayesianRevenueModel {
    fn name(&self) -> &'static str {
        "BayesianRevenue"
    }

    fn generate_path(
        &self,
        rng: &mut MonteCarloRng,
        time_horizon: &TimeHorizon,
        path_id: usize,
    ) -> SimulationPath {
        let n = time_horizon.n_steps();
        let dt = time_horizon.dt();

        // Start with total base revenue
        let initial = self.total_base_revenue();
        let mu = self.weighted_growth_rate();
        let sigma = self.weighted_volatility();

        let mut values = Vec::with_capacity(n + 1);
        let mut current = initial;
        values.push(current);

        // Common shock for correlation
        let common_shock = if self.correlated {
            rng.standard_normal()
        } else {
            0.0
        };

        for _ in 0..n {
            // GBM-like growth with random shocks
            let z = if self.correlated {
                0.7 * common_shock + 0.3 * rng.standard_normal() // 70% correlated
            } else {
                rng.standard_normal()
            };

            // Log-normal growth
            let growth = ((mu - 0.5 * sigma.powi(2)) * dt + sigma * dt.sqrt() * z).exp();
            current *= growth;
            values.push(current);
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

        let initial = self.total_base_revenue();
        let mu = self.weighted_growth_rate();
        let sigma = self.weighted_volatility();

        let mut values = Vec::with_capacity(n + 1);
        let mut current = initial;
        values.push(current);

        for _ in 0..n {
            // Antithetic: negate the random shock
            let z = -rng.standard_normal();
            let growth = ((mu - 0.5 * sigma.powi(2)) * dt + sigma * dt.sqrt() * z).exp();
            current *= growth;
            values.push(current);
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

/// Create a revenue model from CSV data
impl ProductData {
    /// Create a new product
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        base_revenue: f64,
        growth_rate: f64,
        volatility: f64,
    ) -> Self {
        Self {
            name: name.into(),
            base_revenue,
            growth_rate,
            volatility,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_model_creation() {
        let products = vec![
            ProductData::new("Widget", 100_000.0, 0.10, 0.15),
            ProductData::new("Gadget", 50_000.0, 0.20, 0.25),
        ];

        let model = BayesianRevenueModel::new(products);

        assert!((model.total_base_revenue() - 150_000.0).abs() < 0.01);
    }

    #[test]
    fn test_weighted_metrics() {
        let products = vec![
            ProductData::new("A", 100.0, 0.10, 0.20),
            ProductData::new("B", 100.0, 0.20, 0.30),
        ];

        let model = BayesianRevenueModel::new(products);

        // Equal weights: average should be midpoint
        assert!((model.weighted_growth_rate() - 0.15).abs() < 0.01);
    }

    #[test]
    fn test_simulation() {
        let products = vec![ProductData::new("Test", 100_000.0, 0.10, 0.15)];

        let model = BayesianRevenueModel::new(products);
        let engine = MonteCarloEngine::reproducible(42).with_n_simulations(100);
        let horizon = TimeHorizon::quarters(4);

        let result = engine.simulate(&model, &horizon);

        assert_eq!(result.n_paths(), 100);
        assert!(result.final_value_statistics().mean > 0.0);
    }

    #[test]
    fn test_correlated_simulation() {
        let products = vec![
            ProductData::new("A", 100_000.0, 0.10, 0.15),
            ProductData::new("B", 50_000.0, 0.15, 0.20),
        ];

        let model = BayesianRevenueModel::new(products).with_correlation();
        let engine = MonteCarloEngine::reproducible(42).with_n_simulations(100);
        let horizon = TimeHorizon::quarters(4);

        let result = engine.simulate(&model, &horizon);

        assert_eq!(result.n_paths(), 100);
    }
}
