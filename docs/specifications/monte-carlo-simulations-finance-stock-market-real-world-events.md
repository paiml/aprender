# Monte Carlo Simulations for Finance, Stock Market, and Real-World Events

**Version:** 1.0
**Date:** 2025-12-07
**Status:** Planning (Awaiting Team Review)
**Target Release:** v0.15.0+ (aprender-monte-carlo sub-crate)
**Methodology:** EXTREME TDD + Toyota Way + Six Sigma + Google AI Engineering Principles

---

## Executive Summary

This specification defines a production-grade Monte Carlo simulation framework for financial modeling, stock market analysis, and real-world business event forecasting. Building upon the Bayesian probability foundations established in `comprehensive-bayesian-probability-spec.md`, this module provides practitioners with scientifically rigorous tools for:

1. **Revenue Forecasting**: Product portfolio projections with uncertainty quantification
2. **Stock Market Analysis**: Historical-data-driven portfolio simulations
3. **Risk Assessment**: Value-at-Risk (VaR), Conditional VaR (CVaR), drawdown analysis
4. **Scenario Planning**: Stress testing and what-if analysis with embedded S&P 500 historical data

**Design Philosophy**: "Genchi Genbutsu" (Go and see) - Users provide their own CSV data; we provide mathematically sound simulation engines with transparent assumptions.

**Sub-Crate**: `aprender-monte-carlo` with embedded inflation-adjusted S&P 500 historical returns (1928-present)

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Core Simulation Engine](#3-core-simulation-engine)
4. [Business Revenue Simulation](#4-business-revenue-simulation)
5. [Stock Market Simulation](#5-stock-market-simulation)
6. [CSV Data Interface](#6-csv-data-interface)
7. [Risk Metrics and Analysis](#7-risk-metrics-and-analysis)
8. [Embedded S&P 500 Historical Data](#8-embedded-sp-500-historical-data)
9. [Implementation Architecture](#9-implementation-architecture)
10. [Quality Standards](#10-quality-standards)
11. [Academic References](#11-academic-references)
12. [Toyota Way / Six Sigma Integration](#12-toyota-way--six-sigma-integration)
13. [Google AI Engineering Principles](#13-google-ai-engineering-principles)
14. [Implementation Roadmap](#14-implementation-roadmap)

---

## 1. Design Philosophy

### 1.1 Bayesian-Monte Carlo Integration

**Extension of Bayesian Foundations**: Monte Carlo methods are computational implementations of Bayesian inference when analytical solutions are intractable [1]. This specification bridges the gap between the theoretical Bayesian framework and practical financial simulation.

```text
Posterior Expectation (Bayesian):
  E[g(θ)|D] = ∫ g(θ) P(θ|D) dθ

Monte Carlo Approximation:
  E[g(θ)|D] ≈ (1/N) Σᵢ g(θᵢ),  θᵢ ~ P(θ|D)

Convergence: O(1/√N) regardless of dimensionality [2]
```

### 1.2 Toyota Way Principles Applied

| Principle | Application |
|-----------|-------------|
| **Genchi Genbutsu** | Users provide real CSV data; we don't assume distributions |
| **Jidoka** | Automatic validation of input data; halt on anomalies |
| **Kaizen** | Incremental simulation refinement based on new data |
| **Heijunka** | Balanced computational load across simulation paths |
| **Standardized Work** | Reproducible simulations via explicit seed management |
| **Muda Elimination** | Variance reduction techniques (antithetic, stratified) |

### 1.3 Six Sigma Quality Gates

```rust
/// Six Sigma simulation quality metrics
pub struct SixSigmaMetrics {
    /// Process capability index (Cpk)
    /// Target: Cpk ≥ 1.33 for simulation accuracy vs analytical solutions
    pub cpk: f64,

    /// Defects per million opportunities (DPMO)
    /// Target: < 3.4 DPMO for critical financial calculations
    pub dpmo: f64,

    /// Measurement System Analysis (MSA)
    /// Gage R&R < 10% for reproducibility
    pub gage_rr: f64,
}
```

### 1.4 Google AI Engineering Principles [3]

1. **ML Test Score**: Comprehensive testing of statistical properties
2. **Data Validation**: Schema enforcement on CSV inputs
3. **Model Monitoring**: Convergence diagnostics during simulation
4. **Feature Engineering**: Automatic detection of data characteristics
5. **Reproducibility**: Deterministic results with explicit seeds

---

## 2. Mathematical Foundations

### 2.1 Law of Large Numbers and Central Limit Theorem

**Monte Carlo Estimator**:
```text
θ̂ₙ = (1/N) Σᵢ₌₁ᴺ f(Xᵢ),  Xᵢ ~ P(X)

Properties:
  - Unbiased: E[θ̂ₙ] = E[f(X)]
  - Consistent: θ̂ₙ →ᵖ E[f(X)] as N → ∞ (SLLN)
  - Asymptotically Normal: √N(θ̂ₙ - θ) →ᵈ N(0, σ²) (CLT)

Standard Error: SE = σ / √N
```

**Reference**: Glasserman (2003), *Monte Carlo Methods in Financial Engineering* [4]

### 2.2 Variance Reduction Techniques

**Antithetic Variates** [5]:
```text
Given U ~ Uniform(0,1), generate paths using both U and (1-U)
Variance reduction: Var(θ̂_AV) = (1 + ρ)/2 × Var(θ̂)
where ρ = Corr(f(U), f(1-U)) < 0 for monotonic f
```

**Stratified Sampling**:
```text
Partition sample space into K strata
Sample nₖ points from stratum k
θ̂_SS = Σₖ (nₖ/N) × θ̂ₖ
Variance reduction: Always ≤ simple random sampling
```

**Control Variates** [6]:
```text
θ̂_CV = θ̂ - c(Ŷ - E[Y])
where Y is a correlated variable with known expectation
Optimal c* = Cov(θ̂, Ŷ) / Var(Ŷ)
Variance reduction factor: 1 - ρ²(θ̂, Ŷ)
```

```rust
/// Variance reduction configuration
pub enum VarianceReduction {
    /// No variance reduction (baseline)
    None,

    /// Antithetic variates for symmetric distributions
    Antithetic,

    /// Stratified sampling with K strata
    Stratified { strata: usize },

    /// Control variates with known expectation
    ControlVariate {
        control_fn: Box<dyn Fn(&SimulationPath) -> f64 + Send + Sync>,
        known_expectation: f64,
    },

    /// Combined techniques
    Combined(Vec<VarianceReduction>),
}
```

### 2.3 Convergence Diagnostics

**Effective Sample Size** [7]:
```text
ESS = N / (1 + 2Σₖ ρₖ)
where ρₖ is autocorrelation at lag k
```

**Monte Carlo Standard Error**:
```text
MCSE = σ̂ / √ESS
Convergence criterion: MCSE < ε × |θ̂|
```

```rust
/// Convergence diagnostics for Monte Carlo simulations
pub struct ConvergenceDiagnostics {
    /// Effective sample size
    pub ess: f64,

    /// Monte Carlo standard error
    pub mcse: f64,

    /// Relative precision (MCSE / |estimate|)
    pub relative_precision: f64,

    /// Running mean convergence (should stabilize)
    pub running_mean_history: Vec<f64>,

    /// Batch means for variance estimation
    pub batch_means: Vec<f64>,
}

impl ConvergenceDiagnostics {
    /// Check if simulation has converged to desired precision
    pub fn is_converged(&self, target_precision: f64) -> bool {
        self.relative_precision < target_precision
    }
}
```

---

## 3. Core Simulation Engine

### 3.1 Random Number Generation

**Requirement**: Cryptographically-secure PRNG with explicit seeding for reproducibility [8].

```rust
use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha20Rng;

/// Reproducible random number generator
pub struct MonteCarloRng {
    rng: ChaCha20Rng,
    seed: u64,
    draws: usize,
}

impl MonteCarloRng {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
            seed,
            draws: 0,
        }
    }

    /// Standard normal via Box-Muller transform
    pub fn standard_normal(&mut self) -> f64 {
        let u1: f64 = self.rng.gen();
        let u2: f64 = self.rng.gen();
        self.draws += 2;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Correlated multivariate normal via Cholesky decomposition
    pub fn multivariate_normal(&mut self, mean: &[f64], cholesky_l: &Matrix) -> Vec<f64> {
        let n = mean.len();
        let z: Vec<f64> = (0..n).map(|_| self.standard_normal()).collect();
        // x = μ + L × z
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[i] = mean[i];
            for j in 0..=i {
                x[i] += cholesky_l[(i, j)] * z[j];
            }
        }
        x
    }
}
```

### 3.2 Simulation Path Generation

```rust
/// A single simulation path through time
#[derive(Debug, Clone)]
pub struct SimulationPath {
    /// Time points (e.g., months, quarters, years)
    pub time: Vec<f64>,

    /// Values at each time point
    pub values: Vec<f64>,

    /// Metadata for this path
    pub metadata: PathMetadata,
}

#[derive(Debug, Clone)]
pub struct PathMetadata {
    pub path_id: usize,
    pub seed: u64,
    pub is_antithetic: bool,
}

/// Core simulation engine
pub struct MonteCarloEngine {
    /// Number of simulation paths
    pub n_simulations: usize,

    /// Random seed for reproducibility
    pub seed: u64,

    /// Variance reduction technique
    pub variance_reduction: VarianceReduction,

    /// Convergence target (relative precision)
    pub target_precision: f64,

    /// Maximum simulations before stopping
    pub max_simulations: usize,
}

impl MonteCarloEngine {
    /// Run simulation with given model
    pub fn simulate<M: SimulationModel>(
        &self,
        model: &M,
        time_horizon: &TimeHorizon,
    ) -> SimulationResult {
        let mut rng = MonteCarloRng::new(self.seed);
        let mut paths = Vec::with_capacity(self.n_simulations);
        let mut diagnostics = ConvergenceDiagnostics::default();

        for i in 0..self.n_simulations {
            // Generate primary path
            let path = model.generate_path(&mut rng, time_horizon, i);
            paths.push(path);

            // Antithetic path if enabled
            if matches!(self.variance_reduction, VarianceReduction::Antithetic) {
                let antithetic = model.generate_antithetic_path(&mut rng, time_horizon, i);
                paths.push(antithetic);
            }

            // Update convergence diagnostics
            diagnostics.update(&paths);

            // Early stopping if converged
            if diagnostics.is_converged(self.target_precision) {
                break;
            }
        }

        SimulationResult {
            paths,
            diagnostics,
            model_name: model.name().to_string(),
        }
    }
}
```

### 3.3 Simulation Models

```rust
/// Trait for simulation models
pub trait SimulationModel: Send + Sync {
    fn name(&self) -> &str;

    /// Generate a single simulation path
    fn generate_path(
        &self,
        rng: &mut MonteCarloRng,
        time_horizon: &TimeHorizon,
        path_id: usize,
    ) -> SimulationPath;

    /// Generate antithetic path (for variance reduction)
    fn generate_antithetic_path(
        &self,
        rng: &mut MonteCarloRng,
        time_horizon: &TimeHorizon,
        path_id: usize,
    ) -> SimulationPath;
}

/// Time horizon specification
pub struct TimeHorizon {
    /// Start date
    pub start: Date,

    /// End date
    pub end: Date,

    /// Time step granularity
    pub step: TimeStep,
}

pub enum TimeStep {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annually,
    Custom { days: usize },
}
```

---

## 4. Business Revenue Simulation

### 4.1 Product Revenue Model

**Use Case**: A company sells 10 products with quarterly revenue ranging from $2K-$20K per quarter. They plan to add 10 more products. What can they expect?

```rust
/// Product revenue distribution (learned from historical data)
#[derive(Debug, Clone)]
pub struct ProductRevenueDistribution {
    /// Distribution type fitted to historical data
    pub distribution: RevenueDistribution,

    /// Seasonality factors by quarter [Q1, Q2, Q3, Q4]
    pub seasonality: [f64; 4],

    /// Year-over-year growth rate
    pub growth_rate: f64,

    /// Correlation with market conditions
    pub market_beta: f64,
}

pub enum RevenueDistribution {
    /// Normal distribution (symmetric around mean)
    Normal { mean: f64, std: f64 },

    /// Log-normal (common for revenue, always positive)
    LogNormal { mu: f64, sigma: f64 },

    /// Truncated normal (bounded range)
    TruncatedNormal { mean: f64, std: f64, lower: f64, upper: f64 },

    /// Empirical (non-parametric, from historical data)
    Empirical { samples: Vec<f64> },

    /// Mixture of distributions (multiple product types)
    Mixture { components: Vec<(f64, Box<RevenueDistribution>)> },
}

/// Business revenue simulation model
pub struct BusinessRevenueModel {
    /// Existing products with learned distributions
    pub existing_products: Vec<ProductRevenueDistribution>,

    /// New products (use prior from existing or specified)
    pub new_products: Vec<ProductRevenueDistribution>,

    /// Correlation matrix between products
    pub correlation_matrix: Matrix,

    /// Market condition model
    pub market_model: Option<MarketConditionModel>,
}

impl BusinessRevenueModel {
    /// Create from CSV of historical product revenues
    ///
    /// CSV Format:
    /// ```csv
    /// date,product_id,revenue
    /// 2023-Q1,product_1,15000
    /// 2023-Q1,product_2,8500
    /// ...
    /// ```
    pub fn from_csv(path: &Path, config: RevenueModelConfig) -> Result<Self, MonteCarloError> {
        let data = CsvLoader::load_revenue_data(path)?;

        // Fit distribution to each product
        let existing_products: Vec<ProductRevenueDistribution> = data
            .group_by_product()
            .map(|(product_id, revenues)| {
                let dist = fit_revenue_distribution(&revenues, config.distribution_type);
                let seasonality = estimate_seasonality(&revenues, &data.dates);
                let growth = estimate_growth_rate(&revenues, &data.dates);

                ProductRevenueDistribution {
                    distribution: dist,
                    seasonality,
                    growth_rate: growth,
                    market_beta: config.default_beta,
                }
            })
            .collect();

        // Estimate correlation matrix
        let correlation_matrix = estimate_correlation(&data)?;

        Ok(Self {
            existing_products,
            new_products: Vec::new(),
            correlation_matrix,
            market_model: None,
        })
    }

    /// Add new products using prior from existing products (Bayesian)
    pub fn add_new_products(&mut self, n_products: usize, prior: NewProductPrior) {
        match prior {
            NewProductPrior::SimilarToExisting => {
                // Use hierarchical Bayesian model
                // New product ~ mixture of existing product distributions
                let mixture = create_mixture_distribution(&self.existing_products);
                for _ in 0..n_products {
                    self.new_products.push(ProductRevenueDistribution {
                        distribution: mixture.clone(),
                        seasonality: average_seasonality(&self.existing_products),
                        growth_rate: average_growth(&self.existing_products),
                        market_beta: 1.0,
                    });
                }
            }
            NewProductPrior::Specified(dist) => {
                for _ in 0..n_products {
                    self.new_products.push(dist.clone());
                }
            }
            NewProductPrior::Pessimistic => {
                // Use lower quartile of existing products
                let lower_quartile = percentile_distribution(&self.existing_products, 0.25);
                for _ in 0..n_products {
                    self.new_products.push(lower_quartile.clone());
                }
            }
        }

        // Expand correlation matrix for new products
        self.expand_correlation_matrix(n_products);
    }
}
```

### 4.2 Revenue Simulation Output

```rust
/// Revenue simulation results
pub struct RevenueSimulationResult {
    /// Total revenue by period (e.g., quarterly)
    pub total_revenue_by_period: Vec<PeriodStatistics>,

    /// Total annual revenue
    pub annual_revenue: AnnualStatistics,

    /// Product-level breakdowns
    pub product_breakdowns: Vec<ProductStatistics>,

    /// Probability of hitting targets
    pub target_probabilities: Vec<TargetProbability>,

    /// Raw simulation paths (for further analysis)
    pub paths: Vec<SimulationPath>,
}

#[derive(Debug, Clone)]
pub struct PeriodStatistics {
    pub period: String,  // e.g., "2024-Q1"
    pub mean: f64,
    pub median: f64,
    pub std: f64,
    pub percentiles: Percentiles,
    pub confidence_interval_95: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct Percentiles {
    pub p5: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
}

#[derive(Debug, Clone)]
pub struct TargetProbability {
    pub target_name: String,
    pub target_value: f64,
    pub probability_of_achieving: f64,
    pub expected_shortfall_if_miss: f64,
}
```

### 4.3 CLI Interface for Revenue Simulation

```bash
# Simulate existing 10-product portfolio
aprender-monte-carlo revenue simulate \
    --data products_historical.csv \
    --periods 4 \
    --horizon 2025 \
    --n-simulations 10000 \
    --output forecast_existing.json

# Add 10 new products and simulate total
aprender-monte-carlo revenue simulate \
    --data products_historical.csv \
    --new-products 10 \
    --new-product-prior similar \
    --periods 4 \
    --horizon 2025 \
    --n-simulations 10000 \
    --output forecast_expanded.json

# Check probability of hitting revenue targets
aprender-monte-carlo revenue targets \
    --data products_historical.csv \
    --targets "Q1:100000,Q2:120000,Annual:500000" \
    --n-simulations 10000
```

---

## 5. Stock Market Simulation

### 5.1 Stock Price Models

**Geometric Brownian Motion (GBM)** [9]:
```text
dS = μSdt + σSdW

Discretization (Euler-Maruyama):
S(t+Δt) = S(t) × exp((μ - σ²/2)Δt + σ√Δt × Z)
where Z ~ N(0,1)
```

**Jump-Diffusion Model (Merton)** [10]:
```text
dS = μSdt + σSdW + S(J-1)dN

where:
  J = jump multiplier (log-normal)
  N = Poisson process with intensity λ
```

**GARCH(1,1) for Volatility Clustering** [11]:
```text
σ²(t) = ω + α × r²(t-1) + β × σ²(t-1)

where:
  r(t) = log return
  ω, α, β = GARCH parameters
  Stationarity: α + β < 1
```

```rust
/// Stock price simulation models
pub enum StockModel {
    /// Geometric Brownian Motion (Black-Scholes)
    GBM {
        mu: f64,    // Drift (expected return)
        sigma: f64, // Volatility
    },

    /// GBM with mean-reverting volatility
    Heston {
        mu: f64,           // Drift
        kappa: f64,        // Mean reversion speed
        theta: f64,        // Long-run variance
        xi: f64,           // Vol of vol
        rho: f64,          // Correlation with price
        v0: f64,           // Initial variance
    },

    /// Jump-diffusion (Merton model)
    MertonJumpDiffusion {
        mu: f64,
        sigma: f64,
        lambda: f64,      // Jump intensity
        jump_mean: f64,   // Mean jump size
        jump_std: f64,    // Jump size volatility
    },

    /// Empirical bootstrap from historical returns
    EmpiricalBootstrap {
        historical_returns: Vec<f64>,
        block_size: usize,  // For block bootstrap
    },

    /// GARCH(1,1) with time-varying volatility
    Garch {
        omega: f64,
        alpha: f64,
        beta: f64,
        initial_variance: f64,
    },
}

impl StockModel {
    /// Calibrate from historical data
    pub fn calibrate_from_csv(
        path: &Path,
        model_type: StockModelType,
    ) -> Result<Self, MonteCarloError> {
        let prices = CsvLoader::load_price_data(path)?;
        let returns = compute_log_returns(&prices);

        match model_type {
            StockModelType::GBM => {
                let mu = returns.mean() * 252.0;  // Annualize
                let sigma = returns.std() * 252.0_f64.sqrt();
                Ok(StockModel::GBM { mu, sigma })
            }
            StockModelType::Garch => {
                let (omega, alpha, beta) = fit_garch(&returns)?;
                let initial_variance = returns.variance();
                Ok(StockModel::Garch { omega, alpha, beta, initial_variance })
            }
            StockModelType::Empirical => {
                Ok(StockModel::EmpiricalBootstrap {
                    historical_returns: returns,
                    block_size: 20,  // ~1 month of trading days
                })
            }
            // ... other model types
        }
    }
}
```

### 5.2 Portfolio Simulation

```rust
/// Portfolio of stocks/assets
pub struct Portfolio {
    /// Asset weights (sum to 1.0)
    pub weights: Vec<f64>,

    /// Asset models
    pub assets: Vec<AssetModel>,

    /// Correlation matrix
    pub correlation: Matrix,

    /// Rebalancing strategy
    pub rebalancing: RebalancingStrategy,
}

pub struct AssetModel {
    pub ticker: String,
    pub initial_price: f64,
    pub model: StockModel,
    pub dividend_yield: f64,
}

pub enum RebalancingStrategy {
    /// No rebalancing (buy and hold)
    None,

    /// Rebalance at fixed intervals
    Periodic { frequency: TimeStep },

    /// Rebalance when weights drift beyond threshold
    Threshold { tolerance: f64 },
}

/// Portfolio simulation engine
pub struct PortfolioSimulator {
    pub portfolio: Portfolio,
    pub engine: MonteCarloEngine,
}

impl PortfolioSimulator {
    /// Simulate portfolio value over time
    pub fn simulate(&self, time_horizon: &TimeHorizon) -> PortfolioSimulationResult {
        let mut rng = MonteCarloRng::new(self.engine.seed);
        let n_assets = self.portfolio.assets.len();
        let n_steps = time_horizon.n_steps();
        let mut paths = Vec::new();

        // Generate correlated random numbers
        let cholesky = self.portfolio.correlation.cholesky()
            .expect("Correlation matrix must be positive definite");

        for sim_id in 0..self.engine.n_simulations {
            let mut portfolio_values = vec![self.initial_value(); n_steps + 1];
            let mut asset_prices = self.initial_prices();
            let mut weights = self.portfolio.weights.clone();

            for t in 0..n_steps {
                // Generate correlated innovations
                let z = rng.multivariate_normal(&vec![0.0; n_assets], &cholesky);

                // Update each asset price
                for (i, asset) in self.portfolio.assets.iter().enumerate() {
                    asset_prices[i] = self.step_asset_price(
                        asset_prices[i],
                        &asset.model,
                        z[i],
                        time_horizon.dt(),
                    );
                }

                // Calculate portfolio value
                portfolio_values[t + 1] = self.calculate_portfolio_value(
                    &asset_prices,
                    &weights,
                    time_horizon.dt(),
                );

                // Rebalance if needed
                if self.should_rebalance(t, &weights, &asset_prices) {
                    weights = self.rebalance(&asset_prices);
                }
            }

            paths.push(SimulationPath {
                time: time_horizon.time_points(),
                values: portfolio_values,
                metadata: PathMetadata { path_id: sim_id, seed: self.engine.seed, is_antithetic: false },
            });
        }

        PortfolioSimulationResult::from_paths(paths, &self.portfolio)
    }
}
```

### 5.3 Stock Screening and Filtering

```rust
/// Stock selection criteria for simulation
pub struct StockScreeningCriteria {
    /// Minimum annual return (historical)
    pub min_annual_return: Option<f64>,

    /// Maximum volatility
    pub max_volatility: Option<f64>,

    /// Minimum dividend yield
    pub min_dividend_yield: Option<f64>,

    /// Maximum dividend yield (avoid yield traps)
    pub max_dividend_yield: Option<f64>,

    /// Market cap range
    pub market_cap_range: Option<(f64, f64)>,

    /// Sector filter
    pub sectors: Option<Vec<String>>,

    /// Custom filters
    pub custom_filters: Vec<Box<dyn Fn(&StockData) -> bool + Send + Sync>>,
}

impl StockScreeningCriteria {
    /// Filter stocks from universe
    pub fn apply(&self, universe: &[StockData]) -> Vec<&StockData> {
        universe.iter().filter(|stock| {
            self.min_annual_return.map_or(true, |min| stock.annual_return >= min) &&
            self.max_volatility.map_or(true, |max| stock.volatility <= max) &&
            self.min_dividend_yield.map_or(true, |min| stock.dividend_yield >= min) &&
            self.max_dividend_yield.map_or(true, |max| stock.dividend_yield <= max) &&
            self.custom_filters.iter().all(|f| f(stock))
        }).collect()
    }
}
```

### 5.4 CLI Interface for Stock Simulation

```bash
# Simulate single stock
aprender-monte-carlo stock simulate \
    --ticker AAPL \
    --data aapl_historical.csv \
    --model gbm \
    --horizon 5y \
    --n-simulations 10000 \
    --output aapl_forecast.json

# Simulate portfolio
aprender-monte-carlo portfolio simulate \
    --portfolio portfolio.csv \
    --data market_data/ \
    --horizon 10y \
    --rebalance quarterly \
    --n-simulations 10000 \
    --output portfolio_forecast.json

# Screen and simulate filtered stocks
aprender-monte-carlo stock screen \
    --data sp500_stocks.csv \
    --min-dividend 0.02 \
    --max-volatility 0.25 \
    --min-return 0.08 \
    --simulate 10y \
    --n-simulations 5000 \
    --output screened_results.json
```

---

## 6. CSV Data Interface

### 6.1 Supported CSV Formats

**Revenue Data Format**:
```csv
# Revenue data with products and time periods
date,product_id,revenue,units_sold
2023-Q1,product_1,15000,120
2023-Q1,product_2,8500,85
2023-Q2,product_1,17500,140
2023-Q2,product_2,9200,92
```

**Stock Price Data Format**:
```csv
# Daily stock prices (OHLCV)
date,open,high,low,close,volume,adj_close,dividend
2024-01-02,185.23,186.15,183.79,185.64,45000000,185.64,0.00
2024-01-03,184.50,185.89,183.12,184.25,52000000,184.25,0.00
```

**Portfolio Weights Format**:
```csv
# Portfolio composition
ticker,weight,initial_investment
AAPL,0.25,25000
GOOGL,0.20,20000
MSFT,0.20,20000
VTI,0.35,35000
```

### 6.2 CSV Loader with Validation

```rust
/// CSV data loader with schema validation (Google AI Engineering: Data Validation)
pub struct CsvLoader;

impl CsvLoader {
    /// Load revenue data with validation
    pub fn load_revenue_data(path: &Path) -> Result<RevenueData, MonteCarloError> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut records = Vec::new();

        for (line_num, result) in reader.records().enumerate() {
            let record = result.map_err(|e| MonteCarloError::CsvParse {
                line: line_num + 2,  // +1 for header, +1 for 1-indexing
                cause: e.to_string(),
            })?;

            // Validate required fields
            let date = Self::parse_date(&record[0], line_num)?;
            let product_id = &record[1];
            let revenue: f64 = record[2].parse().map_err(|_| {
                MonteCarloError::InvalidValue {
                    field: "revenue",
                    value: record[2].to_string(),
                    line: line_num + 2,
                }
            })?;

            // Jidoka: Halt on invalid data
            if revenue < 0.0 {
                return Err(MonteCarloError::InvalidValue {
                    field: "revenue",
                    value: format!("{} (negative revenue)", revenue),
                    line: line_num + 2,
                });
            }

            records.push(RevenueRecord { date, product_id: product_id.to_string(), revenue });
        }

        // Validate data completeness
        Self::validate_revenue_data(&records)?;

        Ok(RevenueData { records })
    }

    /// Load stock price data with validation
    pub fn load_price_data(path: &Path) -> Result<PriceData, MonteCarloError> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut records = Vec::new();

        for (line_num, result) in reader.records().enumerate() {
            let record = result?;

            let date = Self::parse_date(&record[0], line_num)?;
            let close: f64 = record.get(4)
                .or_else(|| record.get(1))  // Fallback to first price column
                .ok_or(MonteCarloError::MissingField { field: "close", line: line_num + 2 })?
                .parse()?;

            let adj_close: f64 = record.get(6)
                .and_then(|s| s.parse().ok())
                .unwrap_or(close);

            let dividend: f64 = record.get(7)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);

            records.push(PriceRecord { date, close, adj_close, dividend });
        }

        // Sort by date
        records.sort_by(|a, b| a.date.cmp(&b.date));

        // Validate no missing dates (warn only)
        Self::check_missing_dates(&records);

        Ok(PriceData { records })
    }
}
```

### 6.3 Automatic Distribution Fitting

```rust
/// Fit probability distribution to data (Bayesian model selection)
pub fn fit_revenue_distribution(
    data: &[f64],
    preference: DistributionPreference,
) -> RevenueDistribution {
    // Candidate distributions
    let candidates = vec![
        fit_normal(data),
        fit_lognormal(data),
        fit_truncated_normal(data),
    ];

    // Select best using AIC/BIC (Bayesian Information Criterion)
    let best = candidates.into_iter()
        .min_by(|a, b| {
            a.bic(data).partial_cmp(&b.bic(data)).unwrap()
        })
        .expect("At least one candidate distribution");

    match preference {
        DistributionPreference::BestFit => best,
        DistributionPreference::Conservative => {
            // Use heavier tails for risk management
            fit_student_t(data).unwrap_or(best)
        }
        DistributionPreference::Empirical => {
            RevenueDistribution::Empirical { samples: data.to_vec() }
        }
    }
}
```

---

## 7. Risk Metrics and Analysis

### 7.1 Value at Risk (VaR)

**Definition** [12]: VaR at confidence level α is the threshold value such that the probability of loss exceeding VaR is (1-α).

```text
P(Loss > VaR_α) = 1 - α

For α = 0.95:
  VaR₉₅ = -Percentile(Returns, 5%)
```

```rust
/// Value at Risk calculation
pub struct VaR;

impl VaR {
    /// Historical VaR from simulation paths
    pub fn historical(paths: &[SimulationPath], confidence: f64) -> f64 {
        let returns: Vec<f64> = paths.iter()
            .map(|p| (p.values.last().unwrap() - p.values.first().unwrap()) / p.values.first().unwrap())
            .collect();

        // Sort returns and find percentile
        let mut sorted = returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
        -sorted[idx]  // VaR is positive (loss)
    }

    /// Parametric VaR assuming normal distribution
    pub fn parametric(mean: f64, std: f64, confidence: f64) -> f64 {
        // VaR = -μ + σ × Φ⁻¹(1-α)
        let z = quantile_normal(1.0 - confidence);
        -mean + std * z
    }

    /// Cornish-Fisher VaR (accounts for skewness and kurtosis)
    pub fn cornish_fisher(
        mean: f64,
        std: f64,
        skewness: f64,
        kurtosis: f64,
        confidence: f64,
    ) -> f64 {
        let z = quantile_normal(1.0 - confidence);
        let z_cf = z
            + (z.powi(2) - 1.0) * skewness / 6.0
            + (z.powi(3) - 3.0 * z) * (kurtosis - 3.0) / 24.0
            - (2.0 * z.powi(3) - 5.0 * z) * skewness.powi(2) / 36.0;
        -mean + std * z_cf
    }
}
```

### 7.2 Conditional Value at Risk (CVaR / Expected Shortfall)

**Definition** [13]: CVaR is the expected loss given that loss exceeds VaR.

```text
CVaR_α = E[Loss | Loss > VaR_α]

CVaR is coherent (VaR is not):
  - Subadditive: CVaR(A+B) ≤ CVaR(A) + CVaR(B)
  - Convex: Portfolio optimization is tractable
```

```rust
/// Conditional Value at Risk (Expected Shortfall)
pub struct CVaR;

impl CVaR {
    /// Calculate CVaR from simulation paths
    pub fn from_paths(paths: &[SimulationPath], confidence: f64) -> f64 {
        let returns: Vec<f64> = paths.iter()
            .map(|p| {
                let start = p.values.first().unwrap();
                let end = p.values.last().unwrap();
                (end - start) / start
            })
            .collect();

        let var = VaR::historical(paths, confidence);

        // Average of returns worse than VaR
        let tail_losses: Vec<f64> = returns.iter()
            .filter(|&&r| -r > var)
            .map(|&r| -r)
            .collect();

        if tail_losses.is_empty() {
            var
        } else {
            tail_losses.iter().sum::<f64>() / tail_losses.len() as f64
        }
    }
}
```

### 7.3 Maximum Drawdown

```rust
/// Maximum drawdown analysis
pub struct DrawdownAnalysis;

impl DrawdownAnalysis {
    /// Calculate maximum drawdown from a path
    pub fn max_drawdown(values: &[f64]) -> f64 {
        let mut max_so_far = values[0];
        let mut max_dd = 0.0;

        for &value in values.iter().skip(1) {
            max_so_far = max_so_far.max(value);
            let dd = (max_so_far - value) / max_so_far;
            max_dd = max_dd.max(dd);
        }

        max_dd
    }

    /// Drawdown statistics from simulation
    pub fn from_paths(paths: &[SimulationPath]) -> DrawdownStatistics {
        let drawdowns: Vec<f64> = paths.iter()
            .map(|p| Self::max_drawdown(&p.values))
            .collect();

        DrawdownStatistics {
            mean: drawdowns.iter().sum::<f64>() / drawdowns.len() as f64,
            median: percentile(&drawdowns, 0.5),
            p95: percentile(&drawdowns, 0.95),
            p99: percentile(&drawdowns, 0.99),
            worst: drawdowns.iter().cloned().fold(0.0, f64::max),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DrawdownStatistics {
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub worst: f64,
}
```

### 7.4 Risk-Adjusted Return Metrics

```rust
/// Sharpe Ratio [14]
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
    let excess_returns: Vec<f64> = returns.iter()
        .map(|r| r - risk_free_rate)
        .collect();

    let mean_excess = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
    let std = standard_deviation(&excess_returns);

    mean_excess / std
}

/// Sortino Ratio (penalizes downside volatility only) [15]
pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64, target: f64) -> f64 {
    let excess_returns: Vec<f64> = returns.iter()
        .map(|r| r - risk_free_rate)
        .collect();

    let mean_excess = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;

    // Downside deviation
    let downside_returns: Vec<f64> = returns.iter()
        .filter(|&&r| r < target)
        .map(|&r| (r - target).powi(2))
        .collect();

    let downside_std = if downside_returns.is_empty() {
        0.0
    } else {
        (downside_returns.iter().sum::<f64>() / downside_returns.len() as f64).sqrt()
    };

    if downside_std == 0.0 {
        f64::INFINITY
    } else {
        mean_excess / downside_std
    }
}

/// Calmar Ratio (return / max drawdown) [16]
pub fn calmar_ratio(total_return: f64, max_drawdown: f64) -> f64 {
    if max_drawdown == 0.0 {
        f64::INFINITY
    } else {
        total_return / max_drawdown
    }
}
```

---

## 8. Embedded S&P 500 Historical Data

### 8.1 Data Source and Coverage

**Data Set**: S&P 500 Total Return Index (with dividends reinvested)
**Coverage**: 1928-present (97 years)
**Frequency**: Monthly (annual summaries)
**Adjustment**: Inflation-adjusted using CPI-U [17]

```rust
/// Embedded S&P 500 historical data
pub mod sp500_data {
    /// Monthly returns (inflation-adjusted)
    pub static MONTHLY_RETURNS: &[(u16, u8, f64)] = &[
        // (year, month, real_return)
        (1928, 1, 0.0043),
        (1928, 2, -0.0215),
        // ... comprehensive data through present
    ];

    /// Annual returns summary
    pub static ANNUAL_RETURNS: &[(u16, f64)] = &[
        (1928, 0.4381),
        (1929, -0.0839),
        (1930, -0.2512),
        (1931, -0.4384),
        (1932, -0.0864),
        (1933, 0.5399),
        // ... through present
        (2024, 0.2365),  // Updated annually
    ];

    /// Key historical statistics (pre-computed)
    pub const HISTORICAL_STATS: HistoricalStats = HistoricalStats {
        nominal_cagr: 0.0997,           // ~10% nominal CAGR
        real_cagr: 0.0694,              // ~7% real CAGR
        nominal_std: 0.1574,            // ~16% annual volatility
        real_std: 0.1712,               // ~17% real volatility
        worst_year: -0.4384,            // 1931
        best_year: 0.5399,              // 1933
        worst_drawdown: 0.8674,         // Great Depression peak-to-trough
        positive_years_pct: 0.73,       // 73% of years positive
    };
}

#[derive(Debug, Clone, Copy)]
pub struct HistoricalStats {
    pub nominal_cagr: f64,
    pub real_cagr: f64,
    pub nominal_std: f64,
    pub real_std: f64,
    pub worst_year: f64,
    pub best_year: f64,
    pub worst_drawdown: f64,
    pub positive_years_pct: f64,
}
```

### 8.2 Historical Context Simulations

```rust
/// Simulate using historical scenarios
pub struct HistoricalScenarioSimulator {
    /// Use actual historical sequences (not fitted model)
    pub use_block_bootstrap: bool,

    /// Block size for bootstrap (years)
    pub block_size: usize,

    /// Include specific scenarios
    pub include_scenarios: Vec<HistoricalScenario>,
}

pub enum HistoricalScenario {
    /// Great Depression (1929-1932)
    GreatDepression,

    /// Stagflation (1973-1974)
    Stagflation1970s,

    /// Black Monday (1987)
    BlackMonday,

    /// Dot-com Crash (2000-2002)
    DotComCrash,

    /// Great Financial Crisis (2007-2009)
    GFC2008,

    /// COVID Crash (2020)
    CovidCrash,

    /// Custom date range
    Custom { start: Date, end: Date },
}

impl HistoricalScenarioSimulator {
    /// Run simulation with historical scenarios included
    pub fn simulate_with_scenarios(
        &self,
        portfolio: &Portfolio,
        time_horizon: &TimeHorizon,
        n_simulations: usize,
    ) -> ScenarioSimulationResult {
        let mut results = Vec::new();

        // Regular Monte Carlo simulations
        let engine = MonteCarloEngine::default();
        let regular = engine.simulate(portfolio, time_horizon);
        results.extend(regular.paths);

        // Historical scenario paths
        for scenario in &self.include_scenarios {
            let scenario_returns = self.get_scenario_returns(scenario);
            let scenario_path = self.apply_scenario_returns(
                portfolio,
                &scenario_returns,
                time_horizon,
            );
            results.push(scenario_path);
        }

        ScenarioSimulationResult {
            all_paths: results,
            scenario_outcomes: self.extract_scenario_outcomes(&results),
        }
    }

    fn get_scenario_returns(&self, scenario: &HistoricalScenario) -> Vec<f64> {
        match scenario {
            HistoricalScenario::GreatDepression => {
                // Extract 1929-1932 returns from embedded data
                sp500_data::MONTHLY_RETURNS.iter()
                    .filter(|(y, _, _)| *y >= 1929 && *y <= 1932)
                    .map(|(_, _, r)| *r)
                    .collect()
            }
            HistoricalScenario::GFC2008 => {
                sp500_data::MONTHLY_RETURNS.iter()
                    .filter(|(y, _, _)| *y >= 2007 && *y <= 2009)
                    .map(|(_, _, r)| *r)
                    .collect()
            }
            // ... other scenarios
            HistoricalScenario::Custom { start, end } => {
                sp500_data::MONTHLY_RETURNS.iter()
                    .filter(|(y, m, _)| {
                        let date = Date::from_ymd(*y as i32, *m, 1);
                        date >= *start && date <= *end
                    })
                    .map(|(_, _, r)| *r)
                    .collect()
            }
        }
    }
}
```

### 8.3 Rolling Period Analysis

```rust
/// Rolling period analysis using historical data
pub struct RollingPeriodAnalysis;

impl RollingPeriodAnalysis {
    /// Analyze all rolling N-year periods in S&P 500 history
    pub fn rolling_returns(years: usize) -> RollingPeriodStats {
        let annual_returns = sp500_data::ANNUAL_RETURNS;
        let n_periods = annual_returns.len() - years + 1;

        let mut period_returns = Vec::with_capacity(n_periods);

        for start in 0..n_periods {
            let end = start + years;
            let total_return: f64 = annual_returns[start..end]
                .iter()
                .map(|(_, r)| 1.0 + r)
                .product();
            let annualized = total_return.powf(1.0 / years as f64) - 1.0;
            period_returns.push(annualized);
        }

        RollingPeriodStats {
            years,
            n_periods,
            mean: mean(&period_returns),
            median: percentile(&period_returns, 0.5),
            min: period_returns.iter().cloned().fold(f64::INFINITY, f64::min),
            max: period_returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            positive_pct: period_returns.iter().filter(|&&r| r > 0.0).count() as f64 / n_periods as f64,
            worst_period_start: find_worst_period_start(&annual_returns, &period_returns),
        }
    }

    /// Pre-computed rolling statistics for common periods
    pub fn summary() -> Vec<RollingPeriodStats> {
        vec![
            Self::rolling_returns(1),
            Self::rolling_returns(5),
            Self::rolling_returns(10),
            Self::rolling_returns(15),
            Self::rolling_returns(20),
            Self::rolling_returns(30),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct RollingPeriodStats {
    pub years: usize,
    pub n_periods: usize,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub positive_pct: f64,
    pub worst_period_start: u16,
}
```

---

## 9. Implementation Architecture

### 9.1 Crate Structure

```
crates/aprender-monte-carlo/
├── Cargo.toml
├── src/
│   ├── lib.rs                   # Library entry point
│   ├── main.rs                  # CLI entry point
│   ├── engine/
│   │   ├── mod.rs               # MonteCarloEngine
│   │   ├── rng.rs               # Random number generation
│   │   ├── variance.rs          # Variance reduction techniques
│   │   └── convergence.rs       # Convergence diagnostics
│   ├── models/
│   │   ├── mod.rs               # SimulationModel trait
│   │   ├── revenue.rs           # Business revenue models
│   │   ├── stock.rs             # Stock price models
│   │   └── portfolio.rs         # Portfolio models
│   ├── risk/
│   │   ├── mod.rs               # Risk metrics
│   │   ├── var.rs               # Value at Risk
│   │   ├── cvar.rs              # Conditional VaR
│   │   └── drawdown.rs          # Drawdown analysis
│   ├── data/
│   │   ├── mod.rs               # Data loading
│   │   ├── csv.rs               # CSV loader with validation
│   │   ├── sp500.rs             # Embedded S&P 500 data
│   │   └── validation.rs        # Data validation (Jidoka)
│   ├── cli/
│   │   ├── mod.rs               # CLI commands
│   │   ├── revenue.rs           # Revenue simulation CLI
│   │   ├── stock.rs             # Stock simulation CLI
│   │   └── portfolio.rs         # Portfolio simulation CLI
│   └── error.rs                 # MonteCarloError enum
├── data/
│   └── sp500_monthly.csv        # Embedded historical data
└── tests/
    ├── integration/
    │   ├── revenue_tests.rs
    │   ├── stock_tests.rs
    │   └── cli_tests.rs
    ├── property_tests.rs        # Proptest convergence properties
    └── fixtures/
        ├── sample_revenue.csv
        └── sample_prices.csv
```

### 9.2 Core Dependencies

```toml
[package]
name = "aprender-monte-carlo"
version = "0.1.0"
edition = "2021"
description = "Monte Carlo simulations for finance and business forecasting"

[dependencies]
aprender = { version = "0.14", path = "../aprender" }  # Core ML primitives
rand = "0.8"
rand_chacha = "0.3"                                     # Cryptographic PRNG
csv = "1.3"                                             # CSV parsing
chrono = { version = "0.4", features = ["serde"] }      # Date handling
clap = { version = "4.4", features = ["derive"] }       # CLI

[dev-dependencies]
proptest = "1.4"
criterion = "0.5"
```

### 9.3 Error Handling (Jidoka)

```rust
/// Monte Carlo simulation errors with actionable hints
#[derive(Debug)]
pub enum MonteCarloError {
    /// CSV parsing error
    CsvParse {
        line: usize,
        cause: String,
    },

    /// Missing required field
    MissingField {
        field: &'static str,
        line: usize,
    },

    /// Invalid value in data
    InvalidValue {
        field: &'static str,
        value: String,
        line: usize,
    },

    /// Insufficient data for reliable simulation
    InsufficientData {
        required: usize,
        provided: usize,
        hint: String,
    },

    /// Simulation did not converge
    ConvergenceFailure {
        simulations_run: usize,
        current_precision: f64,
        target_precision: f64,
        hint: String,
    },

    /// Invalid model parameters
    InvalidModelParams {
        param: &'static str,
        value: f64,
        constraint: String,
    },

    /// Correlation matrix not positive definite
    InvalidCorrelationMatrix {
        cause: String,
    },
}

impl std::fmt::Display for MonteCarloError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientData { required, provided, hint } => {
                write!(
                    f,
                    "Insufficient data: {} records provided, {} required\nHint: {}",
                    provided, required, hint
                )
            }
            Self::ConvergenceFailure { simulations_run, current_precision, target_precision, hint } => {
                write!(
                    f,
                    "Simulation did not converge after {} runs\n\
                     Current precision: {:.4}%, Target: {:.4}%\n\
                     Hint: {}",
                    simulations_run, current_precision * 100.0, target_precision * 100.0, hint
                )
            }
            // ... other variants
            _ => write!(f, "{:?}", self),
        }
    }
}
```

---

## 10. Quality Standards

### 10.1 Testing Requirements

| Category | Target | Methodology |
|----------|--------|-------------|
| Unit Test Coverage | ≥95% | `cargo llvm-cov` |
| Property Tests | 100+ | `proptest` |
| Integration Tests | 30+ | CLI e2e tests |
| Convergence Tests | 50+ | Statistical properties |
| Mutation Score | ≥80% | `cargo mutants` |

### 10.2 Statistical Property Tests

```rust
use proptest::prelude::*;

proptest! {
    /// Monte Carlo mean converges to true mean (Law of Large Numbers)
    #[test]
    fn mc_mean_converges(
        true_mean in -100.0..100.0f64,
        true_std in 0.1..10.0f64,
        seed in any::<u64>(),
    ) {
        let mut rng = MonteCarloRng::new(seed);
        let n = 10_000;

        let samples: Vec<f64> = (0..n)
            .map(|_| true_mean + true_std * rng.standard_normal())
            .collect();

        let sample_mean = samples.iter().sum::<f64>() / n as f64;
        let se = true_std / (n as f64).sqrt();

        // 3-sigma bound (99.7% confidence)
        prop_assert!((sample_mean - true_mean).abs() < 3.0 * se);
    }

    /// VaR is monotonic in confidence level
    #[test]
    fn var_monotonic_in_confidence(
        returns in prop::collection::vec(-0.5..0.5f64, 100..1000),
    ) {
        let var_90 = VaR::from_returns(&returns, 0.90);
        let var_95 = VaR::from_returns(&returns, 0.95);
        let var_99 = VaR::from_returns(&returns, 0.99);

        prop_assert!(var_90 <= var_95);
        prop_assert!(var_95 <= var_99);
    }

    /// CVaR ≥ VaR (CVaR is expected shortfall beyond VaR)
    #[test]
    fn cvar_geq_var(
        returns in prop::collection::vec(-0.5..0.5f64, 100..1000),
        confidence in 0.9..0.99f64,
    ) {
        let var = VaR::from_returns(&returns, confidence);
        let cvar = CVaR::from_returns(&returns, confidence);

        prop_assert!(cvar >= var - 1e-10);  // Numerical tolerance
    }

    /// Antithetic variates reduce variance for monotonic functions
    #[test]
    fn antithetic_reduces_variance(seed in any::<u64>()) {
        let n = 10_000;
        let mut rng = MonteCarloRng::new(seed);

        // Simple Monte Carlo
        let simple_estimates: Vec<f64> = (0..100).map(|_| {
            let samples: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
            samples.iter().map(|&u| u.powi(2)).sum::<f64>() / n as f64
        }).collect();

        // Antithetic Monte Carlo
        rng = MonteCarloRng::new(seed);
        let antithetic_estimates: Vec<f64> = (0..100).map(|_| {
            let mut sum = 0.0;
            for _ in 0..n/2 {
                let u = rng.uniform();
                sum += u.powi(2) + (1.0 - u).powi(2);
            }
            sum / n as f64
        }).collect();

        let var_simple = variance(&simple_estimates);
        let var_antithetic = variance(&antithetic_estimates);

        prop_assert!(var_antithetic < var_simple);
    }
}
```

### 10.3 Numerical Stability Tests

```rust
#[test]
fn test_var_extreme_returns() {
    // Test VaR with extreme values
    let returns = vec![-0.99, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 10.0];

    let var_95 = VaR::from_returns(&returns, 0.95);
    assert!(var_95.is_finite());
    assert!(var_95 > 0.0);  // VaR should be positive (loss)
}

#[test]
fn test_cvar_all_positive_returns() {
    // CVaR when all returns are positive
    let returns = vec![0.01, 0.02, 0.03, 0.05, 0.10];

    let cvar_95 = CVaR::from_returns(&returns, 0.95);
    assert!(cvar_95.is_finite());
    // CVaR should be negative or zero (no loss in tail)
}

#[test]
fn test_convergence_diagnostics_short_series() {
    // Edge case: very short simulation
    let paths: Vec<SimulationPath> = (0..10).map(|i| {
        SimulationPath {
            time: vec![0.0, 1.0],
            values: vec![100.0, 100.0 + (i as f64)],
            metadata: PathMetadata::default(),
        }
    }).collect();

    let diagnostics = ConvergenceDiagnostics::from_paths(&paths);

    // Should not panic, should indicate low confidence
    assert!(!diagnostics.is_converged(0.01));
}
```

---

## 11. Academic References

### 11.1 Monte Carlo Foundations

**[1] Robert, C. P., & Casella, G. (2004).** *Monte Carlo Statistical Methods* (2nd ed.). Springer.
- **Foundation**: Comprehensive treatment of Monte Carlo methods and Bayesian computation
- **Key insight**: MCMC as computational implementation of Bayesian inference
- **Application**: Variance reduction techniques, convergence diagnostics

**[2] Kroese, D. P., Taimre, T., & Botev, Z. I. (2011).** *Handbook of Monte Carlo Methods*. Wiley.
- **Foundation**: Modern Monte Carlo techniques
- **Key insight**: O(1/√N) convergence rate is dimension-independent
- **Application**: Stratified sampling, importance sampling

**[3] Sculley, D., et al. (2015).** "Hidden Technical Debt in Machine Learning Systems." *NeurIPS 2015*.
- **Foundation**: Google AI engineering principles
- **Key insight**: ML systems require monitoring, validation, reproducibility
- **Application**: Data validation schemas, convergence monitoring

### 11.2 Financial Monte Carlo

**[4] Glasserman, P. (2003).** *Monte Carlo Methods in Financial Engineering*. Springer.
- **Foundation**: Authoritative reference for financial MC
- **Key insight**: Path-dependent options, Greeks estimation, variance reduction
- **Application**: GBM simulation, VaR calculation

**[5] Hammersley, J. M., & Handscomb, D. C. (1964).** *Monte Carlo Methods*. Chapman & Hall.
- **Foundation**: Original treatment of antithetic variates
- **Key insight**: Negative correlation reduces variance
- **Application**: Antithetic path generation

**[6] Lavenberg, S. S., & Welch, P. D. (1981).** "A Perspective on the Use of Control Variables to Increase the Efficiency of Monte Carlo Simulations." *Management Science*, 27(3), 322-335.
- **Foundation**: Control variates methodology
- **Key insight**: Leverage known expectations to reduce variance
- **Application**: Control variate implementation

**[7] Gelman, A., et al. (2013).** *Bayesian Data Analysis* (3rd ed.). CRC Press.
- **Foundation**: Effective sample size, convergence diagnostics
- **Key insight**: ESS accounts for autocorrelation
- **Application**: Convergence monitoring

### 11.3 Stochastic Processes

**[8] L'Ecuyer, P. (2012).** "Random Number Generation." *Handbook of Computational Statistics*.
- **Foundation**: PRNG theory and practice
- **Key insight**: Reproducibility requires explicit seeding
- **Application**: ChaCha20 PRNG selection

**[9] Black, F., & Scholes, M. (1973).** "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637-654.
- **Foundation**: Geometric Brownian Motion for stock prices
- **Key insight**: Log-normal distribution of prices
- **Application**: GBM stock model

**[10] Merton, R. C. (1976).** "Option Pricing When Underlying Stock Returns Are Discontinuous." *Journal of Financial Economics*, 3(1-2), 125-144.
- **Foundation**: Jump-diffusion models
- **Key insight**: Fat tails from jumps
- **Application**: Merton jump-diffusion implementation

**[11] Bollerslev, T. (1986).** "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.
- **Foundation**: GARCH models
- **Key insight**: Volatility clustering in financial returns
- **Application**: GARCH(1,1) model implementation

### 11.4 Risk Metrics

**[12] Jorion, P. (2006).** *Value at Risk: The New Benchmark for Managing Financial Risk* (3rd ed.). McGraw-Hill.
- **Foundation**: VaR methodology
- **Key insight**: VaR as industry standard risk measure
- **Application**: VaR calculation methods

**[13] Artzner, P., et al. (1999).** "Coherent Measures of Risk." *Mathematical Finance*, 9(3), 203-228.
- **Foundation**: Coherent risk measures, CVaR
- **Key insight**: VaR is not subadditive; CVaR is coherent
- **Application**: CVaR implementation

**[14] Sharpe, W. F. (1966).** "Mutual Fund Performance." *Journal of Business*, 39(1), 119-138.
- **Foundation**: Sharpe ratio
- **Key insight**: Risk-adjusted return measurement
- **Application**: Sharpe ratio calculation

**[15] Sortino, F. A., & van der Meer, R. (1991).** "Downside Risk." *Journal of Portfolio Management*, 17(4), 27-31.
- **Foundation**: Sortino ratio
- **Key insight**: Focus on downside risk only
- **Application**: Sortino ratio implementation

**[16] Young, T. W. (1991).** "Calmar Ratio: A Smoother Tool." *Futures*, 20(1), 40.
- **Foundation**: Calmar ratio
- **Key insight**: Return relative to maximum drawdown
- **Application**: Calmar ratio calculation

### 11.5 Historical Data and Market Analysis

**[17] Shiller, R. J. (2015).** *Irrational Exuberance* (3rd ed.). Princeton University Press.
- **Foundation**: Long-term stock market data
- **Key insight**: Inflation-adjusted returns, mean reversion
- **Application**: S&P 500 historical data source

**[18] Dimson, E., Marsh, P., & Staunton, M. (2002).** *Triumph of the Optimists: 101 Years of Global Investment Returns*. Princeton University Press.
- **Foundation**: Global equity returns analysis
- **Key insight**: Equity risk premium, survivorship bias
- **Application**: Historical return context

**[19] Fama, E. F., & French, K. R. (1993).** "Common Risk Factors in the Returns on Stocks and Bonds." *Journal of Financial Economics*, 33(1), 3-56.
- **Foundation**: Factor models
- **Key insight**: Size and value factors
- **Application**: Multi-factor return modeling

### 11.6 Software Engineering and Quality

**[20] Liker, J. K. (2004).** *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill.
- **Foundation**: Toyota Production System
- **Key insight**: Jidoka, Genchi Genbutsu, Kaizen
- **Application**: Quality principles throughout

**[21] George, M. L. (2002).** *Lean Six Sigma: Combining Six Sigma Quality with Lean Speed*. McGraw-Hill.
- **Foundation**: Six Sigma methodology
- **Key insight**: DMAIC, process capability indices
- **Application**: Cpk metrics, quality gates

**[22] Matsumoto, M., & Nishimura, T. (1998).** "Mersenne Twister: A 623-Dimensionally Equidistributed Uniform Pseudo-Random Number Generator." *ACM Transactions on Modeling and Computer Simulation*, 8(1), 3-30.
- **Foundation**: Modern PRNG theory
- **Key insight**: High-dimensional equidistribution
- **Application**: PRNG validation

**[23] Bernstein, D. J. (2008).** "ChaCha, a Variant of Salsa20." *SASC 2008*.
- **Foundation**: ChaCha cryptographic PRNG
- **Key insight**: Security and speed properties
- **Application**: ChaCha20Rng selection

**[24] Owen, A. B. (2013).** *Monte Carlo Theory, Methods and Examples*. (Online book).
- **Foundation**: Modern MC theory
- **Key insight**: Quasi-Monte Carlo, convergence rates
- **Application**: Theoretical underpinning

**[25] Rubinstein, R. Y., & Kroese, D. P. (2016).** *Simulation and the Monte Carlo Method* (3rd ed.). Wiley.
- **Foundation**: Simulation methodology
- **Key insight**: Practical implementation guidance
- **Application**: Algorithm implementations

### 11.7 Advanced Methods and Implementation Guidance

**[26] Niederreiter, H. (1992).** *Random Number Generation and Quasi-Monte Carlo Methods*. SIAM.
- **Guidance**: Use Sobol sequences (Quasi-Monte Carlo) for high-dimensional portfolio simulations to achieve O(1/N) convergence, significantly faster than standard Monte Carlo's O(1/√N).

**[27] Cont, R., & Tankov, P. (2004).** *Financial Modelling with Jump Processes*. Chapman & Hall/CRC.
- **Guidance**: For stress-testing (e.g., 2008/2020 scenarios), pure GBM is insufficient. Implement Merton's model with calibrated jump parameters to capture "fat tail" events accurately.

**[28] McNeil, A. J., Frey, R., & Embrechts, P. (2015).** *Quantitative Risk Management* (2nd ed.). Princeton.
- **Guidance**: For extreme confidence levels (e.g., 99.9% VaR), use Extreme Value Theory (EVT) and Generalized Pareto Distributions (GPD) rather than simple historical bootstrapping which underestimates tail risk.

**[29] Giles, M. B. (2008).** "Multilevel Monte Carlo Path Simulation." *Operations Research*, 56(3).
- **Guidance**: Implement Multilevel Monte Carlo (MLMC) to optimize computational cost by combining simulations with different time step granularities.

**[30] Rockafellar, R. T., & Uryasev, S. (2000).** "Optimization of Conditional Value-at-Risk." *Journal of Risk*, 2(3).
- **Guidance**: When implementing portfolio optimization, minimize CVaR (Expected Shortfall) rather than Variance. CVaR is a coherent risk measure and convex, allowing for efficient linear programming solutions.

**[31] Talay, D., & Tubaro, L. (1990).** "Expansion of the Global Error..." *Stochastic Analysis and Applications*, 8(4).
- **Guidance**: Be aware of discretization error (bias) in Euler-Maruyama. For higher precision without smaller steps, consider the Milstein scheme which includes second-order terms.

**[32] Heckman, J. J. (1979).** "Sample Selection Bias as a Specification Error." *Econometrica*, 47(1).
- **Guidance**: "Genchi Genbutsu" requires validating that user CSV data is representative. Implement checks for survivorship bias (e.g., ensuring delisted stocks are included in historical backtests).

**[33] Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1).
- **Guidance**: Use Random Forests to estimate conditional probabilities for "real-world events" (e.g., product success rates) based on non-linear feature interactions in the input data.

**[34] Pearl, J. (2009).** *Causality*. Cambridge University Press.
- **Guidance**: When simulating business logic, distinguish between correlation and causation. Allow users to define causal graphs for "what-if" interventions (e.g., "if we increase ad spend, revenue increases") distinct from mere correlation.

**[35] Hansen, N. (2016).** "The CMA Evolution Strategy: A Tutorial." *arXiv:1604.00772*.
- **Guidance**: For calibrating complex models (like Heston or Jump-Diffusion) to market data, use Covariance Matrix Adaptation Evolution Strategy (CMA-ES) as a robust global optimizer.

---

## 12. Toyota Way / Six Sigma Integration

### 12.1 DMAIC Cycle for Simulation Quality

**Define**: Clear specification of simulation objectives and acceptance criteria.

**Measure**: Track convergence metrics, ESS, MCSE during simulation.

**Analyze**: Identify sources of variance, detect data quality issues.

**Improve**: Apply variance reduction techniques, refine models.

**Control**: Continuous monitoring of simulation quality.

```rust
/// DMAIC quality tracking
pub struct DMAICTracker {
    /// Define phase: objectives
    pub objectives: SimulationObjectives,

    /// Measure phase: metrics
    pub metrics: ConvergenceDiagnostics,

    /// Analyze phase: variance sources
    pub variance_analysis: VarianceAnalysis,

    /// Improve phase: applied techniques
    pub improvements: Vec<VarianceReduction>,

    /// Control phase: final validation
    pub control_chart: ControlChart,
}

/// Control chart for monitoring simulation quality
pub struct ControlChart {
    /// Upper control limit (UCL = μ + 3σ)
    pub ucl: f64,

    /// Lower control limit (LCL = μ - 3σ)
    pub lcl: f64,

    /// Center line (mean)
    pub cl: f64,

    /// Observations
    pub observations: Vec<f64>,

    /// Out-of-control signals
    pub signals: Vec<ControlSignal>,
}

pub enum ControlSignal {
    /// Single point outside control limits
    BeyondLimits { index: usize, value: f64 },

    /// 7+ points in a row on same side of center
    Run { start: usize, end: usize, side: Side },

    /// 7+ points trending up or down
    Trend { start: usize, end: usize, direction: Direction },
}
```

### 12.2 Poka-Yoke (Error-Proofing)

```rust
/// Input validation (error-proofing)
pub mod validation {
    /// Validate portfolio weights sum to 1
    pub fn validate_weights(weights: &[f64]) -> Result<(), MonteCarloError> {
        let sum: f64 = weights.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(MonteCarloError::InvalidValue {
                field: "weights",
                value: format!("sum = {:.6}", sum),
                line: 0,
            });
        }

        for (i, &w) in weights.iter().enumerate() {
            if w < 0.0 || w > 1.0 {
                return Err(MonteCarloError::InvalidValue {
                    field: "weights",
                    value: format!("weight[{}] = {} (must be in [0, 1])", i, w),
                    line: 0,
                });
            }
        }

        Ok(())
    }

    /// Validate correlation matrix
    pub fn validate_correlation_matrix(corr: &Matrix) -> Result<(), MonteCarloError> {
        let n = corr.nrows();

        // Must be square
        if corr.ncols() != n {
            return Err(MonteCarloError::InvalidCorrelationMatrix {
                cause: format!("Matrix must be square, got {}x{}", n, corr.ncols()),
            });
        }

        // Diagonal must be 1
        for i in 0..n {
            if (corr[(i, i)] - 1.0).abs() > 1e-6 {
                return Err(MonteCarloError::InvalidCorrelationMatrix {
                    cause: format!("Diagonal element [{},{}] = {} (must be 1)", i, i, corr[(i, i)]),
                });
            }
        }

        // Must be symmetric
        for i in 0..n {
            for j in i+1..n {
                if (corr[(i, j)] - corr[(j, i)]).abs() > 1e-6 {
                    return Err(MonteCarloError::InvalidCorrelationMatrix {
                        cause: format!("Not symmetric: [{},{}]={}, [{},{}]={}",
                            i, j, corr[(i, j)], j, i, corr[(j, i)]),
                    });
                }
            }
        }

        // Must be positive semi-definite (check via Cholesky)
        corr.cholesky().map_err(|_| MonteCarloError::InvalidCorrelationMatrix {
            cause: "Matrix is not positive semi-definite".to_string(),
        })?;

        Ok(())
    }
}
```

---

## 13. Google AI Engineering Principles

### 13.1 ML Test Score Components [3]

| Component | Implementation | Score Target |
|-----------|---------------|--------------|
| Data Tests | CSV validation, schema checks | ✅ |
| Model Tests | Convergence tests, property tests | ✅ |
| ML Infrastructure Tests | CLI integration tests | ✅ |
| Monitoring Tests | Convergence diagnostics | ✅ |

### 13.2 Data Validation Schema

```rust
/// Data schema for validation (Google AI: Feature Engineering)
pub struct DataSchema {
    /// Required columns
    pub required_columns: Vec<ColumnSpec>,

    /// Optional columns
    pub optional_columns: Vec<ColumnSpec>,

    /// Row-level constraints
    pub row_constraints: Vec<Box<dyn Fn(&Record) -> bool + Send + Sync>>,
}

pub struct ColumnSpec {
    pub name: String,
    pub dtype: DataType,
    pub nullable: bool,
    pub range: Option<(f64, f64)>,
}

impl DataSchema {
    pub fn revenue_schema() -> Self {
        Self {
            required_columns: vec![
                ColumnSpec { name: "date".into(), dtype: DataType::Date, nullable: false, range: None },
                ColumnSpec { name: "product_id".into(), dtype: DataType::String, nullable: false, range: None },
                ColumnSpec { name: "revenue".into(), dtype: DataType::Float, nullable: false, range: Some((0.0, f64::MAX)) },
            ],
            optional_columns: vec![
                ColumnSpec { name: "units_sold".into(), dtype: DataType::Integer, nullable: true, range: Some((0.0, f64::MAX)) },
            ],
            row_constraints: vec![],
        }
    }

    pub fn price_schema() -> Self {
        Self {
            required_columns: vec![
                ColumnSpec { name: "date".into(), dtype: DataType::Date, nullable: false, range: None },
                ColumnSpec { name: "close".into(), dtype: DataType::Float, nullable: false, range: Some((0.0, f64::MAX)) },
            ],
            optional_columns: vec![
                ColumnSpec { name: "open".into(), dtype: DataType::Float, nullable: true, range: Some((0.0, f64::MAX)) },
                ColumnSpec { name: "high".into(), dtype: DataType::Float, nullable: true, range: Some((0.0, f64::MAX)) },
                ColumnSpec { name: "low".into(), dtype: DataType::Float, nullable: true, range: Some((0.0, f64::MAX)) },
                ColumnSpec { name: "volume".into(), dtype: DataType::Integer, nullable: true, range: Some((0.0, f64::MAX)) },
                ColumnSpec { name: "adj_close".into(), dtype: DataType::Float, nullable: true, range: Some((0.0, f64::MAX)) },
                ColumnSpec { name: "dividend".into(), dtype: DataType::Float, nullable: true, range: Some((0.0, f64::MAX)) },
            ],
            row_constraints: vec![
                Box::new(|r: &Record| {
                    // High >= Low
                    r.get("high").and_then(|h| r.get("low").map(|l| h >= l)).unwrap_or(true)
                }),
            ],
        }
    }
}
```

### 13.3 Reproducibility Guarantees

```rust
/// Reproducibility configuration
pub struct ReproducibilityConfig {
    /// Master seed for all random operations
    pub seed: u64,

    /// Lock random number generator state
    pub deterministic: bool,

    /// Log all random draws for debugging
    pub trace_random: bool,
}

impl MonteCarloEngine {
    /// Create reproducible engine
    pub fn reproducible(seed: u64) -> Self {
        Self {
            seed,
            variance_reduction: VarianceReduction::None,
            target_precision: 0.01,
            max_simulations: 100_000,
            n_simulations: 10_000,
        }
    }

    /// Verify reproducibility
    #[cfg(test)]
    pub fn verify_reproducible(&self) {
        let result1 = self.simulate(&TestModel::default(), &TimeHorizon::one_year());
        let result2 = self.simulate(&TestModel::default(), &TimeHorizon::one_year());

        // Results must be identical
        assert_eq!(result1.paths.len(), result2.paths.len());
        for (p1, p2) in result1.paths.iter().zip(result2.paths.iter()) {
            assert_eq!(p1.values, p2.values);
        }
    }
}
```

---

## 14. Implementation Roadmap

### Phase 1: Core Engine (v0.15.0, 4-5 weeks)

**Focus**: Foundation and reproducibility

- [ ] `MonteCarloRng` with explicit seeding
- [ ] `SimulationPath` and `SimulationModel` trait
- [ ] `MonteCarloEngine` with convergence diagnostics
- [ ] Variance reduction (Antithetic, Stratified)
- [ ] CSV loader with schema validation
- [ ] 80+ unit tests, 30+ property tests

**Deliverable**: Reproducible simulation engine

---

### Phase 2: Business Revenue Models (v0.16.0, 3-4 weeks)

**Focus**: Revenue forecasting

- [ ] `ProductRevenueDistribution` with multiple distributions
- [ ] `BusinessRevenueModel` from CSV
- [ ] Seasonality detection and modeling
- [ ] Correlation matrix estimation
- [ ] New product prior (Bayesian hierarchical)
- [ ] Revenue simulation CLI
- [ ] 50+ tests, integration tests

**Deliverable**: Product portfolio forecasting

---

### Phase 3: Stock Market Models (v0.17.0, 4-5 weeks)

**Focus**: Stock and portfolio simulation

- [ ] `StockModel` (GBM, GARCH, Merton, Empirical)
- [ ] Model calibration from CSV
- [ ] `Portfolio` with rebalancing
- [ ] Correlated asset simulation
- [ ] Stock screening and filtering
- [ ] Stock/portfolio simulation CLI
- [ ] 60+ tests, benchmark tests

**Deliverable**: Stock market simulation toolkit

---

### Phase 4: Risk Metrics (v0.18.0, 3-4 weeks)

**Focus**: Risk analysis

- [ ] `VaR` (Historical, Parametric, Cornish-Fisher)
- [ ] `CVaR` (Expected Shortfall)
- [ ] `DrawdownAnalysis`
- [ ] Risk-adjusted ratios (Sharpe, Sortino, Calmar)
- [ ] Risk dashboard CLI
- [ ] 40+ tests, numerical stability tests

**Deliverable**: Comprehensive risk analytics

---

### Phase 5: S&P 500 Historical Data (v0.19.0, 2-3 weeks)

**Focus**: Embedded historical context

- [ ] Embedded S&P 500 monthly returns (1928-present)
- [ ] `HistoricalScenarioSimulator`
- [ ] `RollingPeriodAnalysis`
- [ ] Historical scenario CLI
- [ ] Data source documentation
- [ ] 30+ tests, data validation tests

**Deliverable**: Historical context for simulations

---

### Phase 6: Documentation & Polish (v0.20.0, 2 weeks)

**Focus**: Production readiness

- [ ] Comprehensive rustdoc
- [ ] Book chapter for aprender-book
- [ ] Example notebooks
- [ ] Performance benchmarks
- [ ] CLI help improvements
- [ ] Final quality validation

**Deliverable**: Production-ready sub-crate

---

## Appendix A: CLI Reference

```bash
# Revenue Simulation
aprender-monte-carlo revenue simulate --data <CSV> --periods <N> --horizon <YEAR> [OPTIONS]
aprender-monte-carlo revenue targets --data <CSV> --targets <LIST> [OPTIONS]

# Stock Simulation
aprender-monte-carlo stock simulate --ticker <SYM> --data <CSV> --model <TYPE> --horizon <PERIOD> [OPTIONS]
aprender-monte-carlo stock calibrate --data <CSV> --model <TYPE> [OPTIONS]
aprender-monte-carlo stock screen --data <CSV> --min-dividend <PCT> --max-volatility <PCT> [OPTIONS]

# Portfolio Simulation
aprender-monte-carlo portfolio simulate --portfolio <CSV> --data <DIR> --horizon <PERIOD> [OPTIONS]
aprender-monte-carlo portfolio optimize --data <DIR> --objective <sharpe|sortino|min-var> [OPTIONS]

# Risk Analysis
aprender-monte-carlo risk var --paths <JSON> --confidence <PCT> [OPTIONS]
aprender-monte-carlo risk report --paths <JSON> [OPTIONS]

# Historical Analysis
aprender-monte-carlo history rolling --years <N> [OPTIONS]
aprender-monte-carlo history scenarios --scenario <NAME> [OPTIONS]
aprender-monte-carlo history stress-test --portfolio <CSV> [OPTIONS]

# Common Options
--n-simulations <N>    Number of simulations (default: 10000)
--seed <SEED>          Random seed for reproducibility
--variance-reduction <TYPE>   none|antithetic|stratified (default: antithetic)
--output <FILE>        Output file (JSON or CSV)
--quiet                Suppress progress output
```

---

## Appendix B: Example Usage

```rust
use aprender_monte_carlo::{
    engine::MonteCarloEngine,
    models::revenue::{BusinessRevenueModel, NewProductPrior},
    models::stock::{StockModel, Portfolio},
    risk::{VaR, CVaR, DrawdownAnalysis},
};

// Example 1: Revenue forecasting with new products
fn forecast_revenue() -> Result<(), MonteCarloError> {
    // Load historical revenue data
    let mut model = BusinessRevenueModel::from_csv(
        Path::new("products_historical.csv"),
        RevenueModelConfig::default(),
    )?;

    // Add 10 new products using learned prior
    model.add_new_products(10, NewProductPrior::SimilarToExisting);

    // Run simulation
    let engine = MonteCarloEngine::reproducible(42);
    let result = engine.simulate(&model, &TimeHorizon::quarters(4))?;

    // Analyze results
    println!("Annual Revenue Forecast:");
    println!("  Mean: ${:.0}", result.annual_revenue.mean);
    println!("  95% CI: ${:.0} - ${:.0}",
        result.annual_revenue.confidence_interval_95.0,
        result.annual_revenue.confidence_interval_95.1);

    Ok(())
}

// Example 2: Portfolio simulation with risk metrics
fn analyze_portfolio() -> Result<(), MonteCarloError> {
    // Load portfolio
    let portfolio = Portfolio::from_csv(Path::new("portfolio.csv"))?;

    // Calibrate models from historical data
    portfolio.calibrate_from_directory(Path::new("market_data/"))?;

    // Run 10-year simulation
    let engine = MonteCarloEngine::default()
        .with_n_simulations(50_000)
        .with_variance_reduction(VarianceReduction::Antithetic);

    let result = engine.simulate(&portfolio, &TimeHorizon::years(10))?;

    // Risk metrics
    let var_95 = VaR::historical(&result.paths, 0.95);
    let cvar_95 = CVaR::from_paths(&result.paths, 0.95);
    let drawdown = DrawdownAnalysis::from_paths(&result.paths);

    println!("10-Year Portfolio Analysis:");
    println!("  Expected Return: {:.1}%", result.annualized_return * 100.0);
    println!("  95% VaR: {:.1}%", var_95 * 100.0);
    println!("  95% CVaR: {:.1}%", cvar_95 * 100.0);
    println!("  Median Max Drawdown: {:.1}%", drawdown.median * 100.0);

    Ok(())
}
```

---

## Status

**Current Status**: Planning (Awaiting Team Review)

**Review Checklist**:
- [ ] API design review
- [ ] Risk metrics validation
- [ ] Historical data source verification
- [ ] Six Sigma quality gates approval
- [ ] Toyota Way compliance check
- [ ] Google AI engineering principles review

**Next Steps**:
1. Team review of this specification
2. Approval of scope and timeline
3. Begin Phase 1 implementation

---

**SPECIFICATION COMPLETE**

**Version:** 1.0
**Date:** 2025-12-07
**Author:** Claude (Anthropic)
**Review Required:** Yes - Awaiting team approval before implementation
**References:** 25 peer-reviewed publications
