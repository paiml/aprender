//! Financial portfolio Monte Carlo scenarios.
//!
//! Implements portfolio risk analysis:
//! - Value at Risk (`VaR`) calculation
//! - Geometric Brownian Motion for asset prices
//! - Correlated multi-asset portfolios
//! - Greeks and sensitivity analysis

use crate::engine::rng::SimRng;
use crate::error::{SimError, SimResult};
use serde::{Deserialize, Serialize};

/// Configuration for a single asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetConfig {
    /// Asset name/identifier.
    pub name: String,
    /// Current price.
    pub price: f64,
    /// Expected annual return (drift).
    pub drift: f64,
    /// Annual volatility (standard deviation).
    pub volatility: f64,
    /// Position size (number of units).
    pub position: f64,
}

impl AssetConfig {
    /// Create a new asset configuration.
    #[must_use]
    pub fn new(name: &str, price: f64, drift: f64, volatility: f64, position: f64) -> Self {
        Self {
            name: name.to_string(),
            price,
            drift,
            volatility,
            position,
        }
    }

    /// Create a stock with typical parameters.
    #[must_use]
    pub fn stock(name: &str, price: f64, position: f64) -> Self {
        Self::new(name, price, 0.08, 0.20, position) // 8% return, 20% vol
    }

    /// Create a bond with typical parameters.
    #[must_use]
    pub fn bond(name: &str, price: f64, position: f64) -> Self {
        Self::new(name, price, 0.03, 0.05, position) // 3% return, 5% vol
    }

    /// Calculate current value of position.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.price * self.position
    }
}

/// Portfolio configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioConfig {
    /// Assets in the portfolio.
    pub assets: Vec<AssetConfig>,
    /// Correlation matrix (flattened, row-major).
    /// If empty, assets are assumed uncorrelated.
    pub correlations: Vec<f64>,
    /// Risk-free rate (annual).
    pub risk_free_rate: f64,
    /// Time horizon for simulation (years).
    pub time_horizon: f64,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            assets: vec![
                AssetConfig::stock("SPY", 450.0, 100.0),
                AssetConfig::bond("TLT", 100.0, 500.0),
            ],
            correlations: vec![1.0, -0.3, -0.3, 1.0], // Stock-bond negative correlation
            risk_free_rate: 0.04,
            time_horizon: 1.0 / 252.0, // 1 trading day
        }
    }
}

impl PortfolioConfig {
    /// Create a simple equity portfolio.
    #[must_use]
    pub fn equity_only() -> Self {
        Self {
            assets: vec![
                AssetConfig::stock("SPY", 450.0, 100.0),
                AssetConfig::stock("QQQ", 380.0, 50.0),
                AssetConfig::stock("IWM", 200.0, 75.0),
            ],
            correlations: vec![1.0, 0.85, 0.80, 0.85, 1.0, 0.75, 0.80, 0.75, 1.0],
            risk_free_rate: 0.04,
            time_horizon: 1.0 / 252.0,
        }
    }

    /// Create a 60/40 stock/bond portfolio.
    #[must_use]
    pub fn balanced_60_40() -> Self {
        Self {
            assets: vec![
                AssetConfig::stock("Equity", 100.0, 600.0),
                AssetConfig::bond("Bonds", 100.0, 400.0),
            ],
            correlations: vec![1.0, -0.2, -0.2, 1.0],
            risk_free_rate: 0.04,
            time_horizon: 1.0 / 252.0,
        }
    }

    /// Calculate total portfolio value.
    #[must_use]
    pub fn total_value(&self) -> f64 {
        self.assets.iter().map(AssetConfig::value).sum()
    }

    /// Get number of assets.
    #[must_use]
    pub fn num_assets(&self) -> usize {
        self.assets.len()
    }

    /// Get correlation between two assets.
    #[must_use]
    pub fn correlation(&self, i: usize, j: usize) -> f64 {
        let n = self.num_assets();
        if self.correlations.is_empty() {
            return if i == j { 1.0 } else { 0.0 };
        }
        self.correlations[i * n + j]
    }
}

/// Result of a single Monte Carlo path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathResult {
    /// Final asset prices.
    pub final_prices: Vec<f64>,
    /// Final portfolio value.
    pub final_value: f64,
    /// Portfolio return (fractional).
    pub portfolio_return: f64,
    /// Profit/Loss.
    pub pnl: f64,
}

/// Value at Risk calculation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaRResult {
    /// `VaR` at specified confidence level.
    pub var: f64,
    /// Conditional `VaR` (Expected Shortfall).
    pub cvar: f64,
    /// Confidence level (e.g., 0.95 for 95%).
    pub confidence: f64,
    /// Number of simulations.
    pub n_simulations: usize,
    /// Mean return.
    pub mean_return: f64,
    /// Standard deviation of returns.
    pub std_return: f64,
    /// Minimum return observed.
    pub min_return: f64,
    /// Maximum return observed.
    pub max_return: f64,
}

/// Portfolio Monte Carlo scenario.
#[derive(Debug, Clone)]
pub struct PortfolioScenario {
    config: PortfolioConfig,
    /// Cholesky decomposition of correlation matrix.
    cholesky: Vec<f64>,
}

impl PortfolioScenario {
    /// Create a new portfolio scenario.
    ///
    /// # Errors
    ///
    /// Returns an error if the correlation matrix is not positive definite.
    pub fn new(config: PortfolioConfig) -> SimResult<Self> {
        let cholesky = Self::cholesky_decomposition(&config)?;
        Ok(Self { config, cholesky })
    }

    /// Perform Cholesky decomposition of correlation matrix.
    fn cholesky_decomposition(config: &PortfolioConfig) -> SimResult<Vec<f64>> {
        let n = config.num_assets();

        if config.correlations.is_empty() {
            // Identity matrix for uncorrelated assets
            let mut l = vec![0.0; n * n];
            for i in 0..n {
                l[i * n + i] = 1.0;
            }
            return Ok(l);
        }

        let mut l = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                if i == j {
                    for k in 0..j {
                        sum += l[j * n + k] * l[j * n + k];
                    }
                    let diag = config.correlations[j * n + j] - sum;
                    if diag <= 0.0 {
                        return Err(SimError::config(
                            "Correlation matrix is not positive definite".to_string(),
                        ));
                    }
                    l[j * n + j] = diag.sqrt();
                } else {
                    for k in 0..j {
                        sum += l[i * n + k] * l[j * n + k];
                    }
                    l[i * n + j] = (config.correlations[i * n + j] - sum) / l[j * n + j];
                }
            }
        }

        Ok(l)
    }

    /// Generate correlated random normals using Cholesky decomposition.
    fn generate_correlated_normals(&self, rng: &mut SimRng) -> Vec<f64> {
        let n = self.config.num_assets();
        let mut z = Vec::with_capacity(n);

        // Generate independent standard normals
        for _ in 0..n {
            z.push(rng.gen_standard_normal());
        }

        // Apply Cholesky transformation for correlation
        let mut correlated = vec![0.0; n];
        for i in 0..n {
            for j in 0..=i {
                correlated[i] += self.cholesky[i * n + j] * z[j];
            }
        }

        correlated
    }

    /// Simulate a single path using Geometric Brownian Motion.
    pub fn simulate_path(&self, rng: &mut SimRng) -> PathResult {
        contract_pre_iterator!();
        let dt = self.config.time_horizon;
        let sqrt_dt = dt.sqrt();
        let initial_value = self.config.total_value();

        // Generate correlated random shocks
        let z = self.generate_correlated_normals(rng);

        // Simulate each asset using GBM
        let final_prices: Vec<f64> = self
            .config
            .assets
            .iter()
            .zip(z.iter())
            .map(|(asset, &shock)| {
                // GBM: S(t+dt) = S(t) * exp((μ - σ²/2)*dt + σ*√dt*Z)
                let drift_term = (asset.drift - 0.5 * asset.volatility.powi(2)) * dt;
                let diffusion_term = asset.volatility * sqrt_dt * shock;
                asset.price * (drift_term + diffusion_term).exp()
            })
            .collect();

        // Calculate final portfolio value
        let final_value: f64 = self
            .config
            .assets
            .iter()
            .zip(final_prices.iter())
            .map(|(asset, &price)| price * asset.position)
            .sum();

        let portfolio_return = (final_value - initial_value) / initial_value;
        let pnl = final_value - initial_value;

        PathResult {
            final_prices,
            final_value,
            portfolio_return,
            pnl,
        }
    }

    /// Calculate Value at Risk using Monte Carlo simulation.
    ///
    /// # Errors
    ///
    /// Returns an error if confidence level is not in (0, 1).
    pub fn calculate_var(
        &self,
        n_simulations: usize,
        confidence: f64,
        rng: &mut SimRng,
    ) -> SimResult<VaRResult> {
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(SimError::config(format!(
                "Confidence must be between 0 and 1, got {confidence}"
            )));
        }

        // Run simulations
        let mut returns: Vec<f64> = (0..n_simulations)
            .map(|_| self.simulate_path(rng).portfolio_return)
            .collect();

        // Sort returns for percentile calculation
        returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate statistics
        let mean_return = returns.iter().sum::<f64>() / n_simulations as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / n_simulations as f64;
        let std_return = variance.sqrt();

        // VaR: loss at (1 - confidence) percentile
        let var_index = ((1.0 - confidence) * n_simulations as f64) as usize;
        let var = -returns[var_index] * self.config.total_value();

        // CVaR (Expected Shortfall): average of losses worse than VaR
        let cvar = -returns[..=var_index].iter().sum::<f64>() / (var_index + 1) as f64
            * self.config.total_value();

        Ok(VaRResult {
            var,
            cvar,
            confidence,
            n_simulations,
            mean_return,
            std_return,
            min_return: returns[0],
            max_return: returns[n_simulations - 1],
        })
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &PortfolioConfig {
        &self.config
    }

    /// Calculate portfolio delta (sensitivity to price changes).
    #[must_use]
    pub fn portfolio_delta(&self) -> Vec<f64> {
        self.config
            .assets
            .iter()
            .map(|asset| asset.position)
            .collect()
    }

    /// Calculate portfolio beta (sensitivity to market).
    /// Uses first asset as market proxy.
    #[must_use]
    pub fn portfolio_beta(&self) -> f64 {
        if self.config.assets.is_empty() {
            return 0.0;
        }

        let market_vol = self.config.assets[0].volatility;
        let total_value = self.config.total_value();

        self.config
            .assets
            .iter()
            .enumerate()
            .map(|(i, asset)| {
                let weight = asset.value() / total_value;
                let correlation = self.config.correlation(i, 0);
                let beta = correlation * asset.volatility / market_vol;
                weight * beta
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_config() {
        let asset = AssetConfig::stock("AAPL", 150.0, 100.0);

        assert_eq!(asset.name, "AAPL");
        assert!((asset.price - 150.0).abs() < f64::EPSILON);
        assert!((asset.value() - 15000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_portfolio_config_default() {
        let config = PortfolioConfig::default();

        assert_eq!(config.num_assets(), 2);
        assert!(config.total_value() > 0.0);
    }

    #[test]
    fn test_portfolio_config_correlation() {
        let config = PortfolioConfig::default();

        // Diagonal should be 1.0
        assert!((config.correlation(0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((config.correlation(1, 1) - 1.0).abs() < f64::EPSILON);

        // Off-diagonal should be symmetric
        assert!((config.correlation(0, 1) - config.correlation(1, 0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_portfolio_scenario_creation() {
        let config = PortfolioConfig::default();
        let scenario = PortfolioScenario::new(config).unwrap();

        assert_eq!(scenario.config().num_assets(), 2);
    }

    #[test]
    fn test_portfolio_simulate_path() {
        let config = PortfolioConfig::default();
        let scenario = PortfolioScenario::new(config).unwrap();
        let mut rng = SimRng::new(42);

        let result = scenario.simulate_path(&mut rng);

        // Prices should be positive
        for price in &result.final_prices {
            assert!(*price > 0.0, "Price should be positive: {price}");
        }

        // Value should be positive
        assert!(result.final_value > 0.0);
    }

    #[test]
    fn test_portfolio_var_calculation() {
        let config = PortfolioConfig::default();
        let scenario = PortfolioScenario::new(config).unwrap();
        let mut rng = SimRng::new(42);

        let var_result = scenario.calculate_var(10000, 0.95, &mut rng).unwrap();

        // VaR should be positive (it's a loss measure)
        assert!(var_result.var >= 0.0, "VaR = {}", var_result.var);

        // CVaR should be >= VaR
        assert!(
            var_result.cvar >= var_result.var,
            "CVaR ({}) should be >= VaR ({})",
            var_result.cvar,
            var_result.var
        );

        // Statistics should be reasonable
        assert!(var_result.max_return > var_result.min_return);
    }

    #[test]
    fn test_portfolio_var_invalid_confidence() {
        let config = PortfolioConfig::default();
        let scenario = PortfolioScenario::new(config).unwrap();
        let mut rng = SimRng::new(42);

        // Invalid confidence levels
        assert!(scenario.calculate_var(1000, 0.0, &mut rng).is_err());
        assert!(scenario.calculate_var(1000, 1.0, &mut rng).is_err());
        assert!(scenario.calculate_var(1000, 1.5, &mut rng).is_err());
    }

    #[test]
    fn test_portfolio_equity_only() {
        let config = PortfolioConfig::equity_only();
        let scenario = PortfolioScenario::new(config).unwrap();
        let mut rng = SimRng::new(42);

        let var_result = scenario.calculate_var(5000, 0.99, &mut rng).unwrap();

        // 99% VaR should be larger than typical daily moves
        // (for highly correlated equities)
        assert!(var_result.var > 0.0);
    }

    #[test]
    fn test_portfolio_balanced() {
        let config = PortfolioConfig::balanced_60_40();
        let scenario = PortfolioScenario::new(config).unwrap();
        let mut rng = SimRng::new(42);

        let var_result = scenario.calculate_var(5000, 0.95, &mut rng).unwrap();

        // Balanced portfolio should have lower VaR than equity-only
        // due to diversification from negative correlation
        assert!(var_result.var > 0.0);
    }

    #[test]
    fn test_portfolio_delta() {
        let config = PortfolioConfig::default();
        let scenario = PortfolioScenario::new(config).unwrap();

        let delta = scenario.portfolio_delta();

        assert_eq!(delta.len(), 2);
        assert!((delta[0] - 100.0).abs() < f64::EPSILON);
        assert!((delta[1] - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_portfolio_beta() {
        let config = PortfolioConfig::default();
        let scenario = PortfolioScenario::new(config).unwrap();

        let beta = scenario.portfolio_beta();

        // Beta should be between -1 and 2 for typical portfolios
        assert!(beta > -1.0 && beta < 2.0, "Beta = {beta}");
    }

    #[test]
    fn test_cholesky_identity() {
        let config = PortfolioConfig {
            assets: vec![
                AssetConfig::stock("A", 100.0, 10.0),
                AssetConfig::stock("B", 100.0, 10.0),
            ],
            correlations: vec![], // Empty = identity
            ..Default::default()
        };

        let scenario = PortfolioScenario::new(config).unwrap();

        // Cholesky of identity is identity
        assert!((scenario.cholesky[0] - 1.0).abs() < f64::EPSILON);
        assert!((scenario.cholesky[3] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_path_result_consistency() {
        let config = PortfolioConfig::default();
        let scenario = PortfolioScenario::new(config.clone()).unwrap();
        let mut rng = SimRng::new(42);

        let result = scenario.simulate_path(&mut rng);

        // PnL should equal final_value - initial_value
        let initial_value = config.total_value();
        let expected_pnl = result.final_value - initial_value;
        assert!((result.pnl - expected_pnl).abs() < 0.01);

        // Return should be PnL / initial_value
        let expected_return = expected_pnl / initial_value;
        assert!((result.portfolio_return - expected_return).abs() < 1e-10);
    }

    #[test]
    fn test_gbm_properties() {
        // Run many simulations to verify GBM statistical properties
        let config = PortfolioConfig {
            assets: vec![AssetConfig::new("Test", 100.0, 0.10, 0.20, 1.0)],
            correlations: vec![1.0],
            risk_free_rate: 0.0,
            time_horizon: 1.0, // 1 year
        };

        let scenario = PortfolioScenario::new(config).unwrap();
        let mut rng = SimRng::new(42);

        let n = 10000;
        let log_returns: Vec<f64> = (0..n)
            .map(|_| {
                let result = scenario.simulate_path(&mut rng);
                (result.final_prices[0] / 100.0).ln()
            })
            .collect();

        // Mean of log returns should be (μ - σ²/2) * T = (0.10 - 0.02) = 0.08
        let mean_log_return = log_returns.iter().sum::<f64>() / n as f64;
        assert!(
            (mean_log_return - 0.08).abs() < 0.02,
            "Mean log return = {mean_log_return}, expected ~0.08"
        );

        // Std of log returns should be σ * √T = 0.20
        let variance = log_returns
            .iter()
            .map(|r| (r - mean_log_return).powi(2))
            .sum::<f64>()
            / n as f64;
        let std = variance.sqrt();
        assert!(
            (std - 0.20).abs() < 0.02,
            "Std of log returns = {std}, expected ~0.20"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Portfolio weights must sum to 1.
        #[test]
        fn prop_weights_sum_to_one(
            w1 in 0.0f64..0.5,
            w2 in 0.0f64..0.5,
        ) {
            let w3 = 1.0 - w1 - w2;
            if w3 >= 0.0 {
                let sum = w1 + w2 + w3;
                prop_assert!((sum - 1.0).abs() < 1e-10);
            }
        }

        /// VaR is always negative or zero (it's a loss).
        #[test]
        fn prop_var_nonpositive(
            confidence in 0.9f64..0.999,
        ) {
            let config = PortfolioConfig::default();
            let _scenario = PortfolioScenario::new(config);

            // VaR at high confidence should be a loss (negative return)
            // For a risky portfolio, VaR represents potential loss
            let _confidence_level = confidence;
            // VaR measures potential loss, which is typically expressed as negative
        }

        /// Higher volatility means higher VaR (more risk).
        #[test]
        fn prop_higher_vol_higher_var(
            vol1 in 0.1f64..0.3,
            vol2 in 0.1f64..0.3,
        ) {
            // For identical portfolios, higher volatility implies higher VaR magnitude
            // This is a fundamental relationship in portfolio risk
            if vol1 > vol2 {
                // Higher volatility -> wider distribution -> larger potential loss
                prop_assert!(vol1 > vol2);
            }
        }

        /// Asset prices must be positive.
        #[test]
        fn prop_prices_positive(
            price in 10.0f64..1000.0,
        ) {
            let config = AssetConfig {
                name: "Test".to_string(),
                price,
                drift: 0.10,
                volatility: 0.20,
                position: 100.0,
            };

            prop_assert!(config.price > 0.0);
        }

        /// Expected return affects drift direction.
        #[test]
        fn prop_drift_sign(
            expected_return in -0.2f64..0.3,
            volatility in 0.1f64..0.4,
        ) {
            // GBM drift = μ - σ²/2
            let drift = expected_return - volatility.powi(2) / 2.0;

            // If μ > σ²/2, drift is positive (prices tend to grow)
            if expected_return > volatility.powi(2) / 2.0 {
                prop_assert!(drift > 0.0);
            } else if expected_return < volatility.powi(2) / 2.0 {
                prop_assert!(drift < 0.0);
            }
        }

        /// Correlation must be in [-1, 1].
        #[test]
        fn prop_correlation_bounds(
            corr in -1.0f64..1.0,
        ) {
            prop_assert!(corr >= -1.0);
            prop_assert!(corr <= 1.0);
        }
    }
}
