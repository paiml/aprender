# Case Study: Monte Carlo Financial Simulation

This case study demonstrates Aprender's Monte Carlo framework for financial modeling and risk analysis.

## Overview

The `monte_carlo` module provides:
- **Simulation Engine**: Reproducible RNG, variance reduction, convergence diagnostics
- **Financial Models**: GBM, Merton jump-diffusion, empirical bootstrap
- **Risk Metrics**: VaR, CVaR, drawdown analysis
- **Risk Ratios**: Sharpe, Sortino, Calmar, Treynor, Information, Omega

## Basic Simulation

```rust
use aprender::monte_carlo::prelude::*;

fn main() {
    // Create reproducible simulation engine
    let engine = MonteCarloEngine::reproducible(42)
        .with_n_simulations(10_000)
        .with_variance_reduction(VarianceReduction::Antithetic);

    // Define stock model: S₀=$100, μ=8%, σ=20%
    let model = GeometricBrownianMotion::new(100.0, 0.08, 0.20);

    // Simulate 1 year with monthly steps
    let horizon = TimeHorizon::years(1);
    let result = engine.simulate(&model, &horizon);

    // Analyze results
    println!("Simulated {} paths", result.n_paths());

    let stats = result.final_value_statistics();
    println!("Final Value Statistics:");
    println!("  Mean: ${:.2}", stats.mean);
    println!("  Std Dev: ${:.2}", stats.std);
    println!("  Min: ${:.2}", stats.min);
    println!("  Max: ${:.2}", stats.max);
}
```

## Financial Models

### Geometric Brownian Motion

```rust
use aprender::monte_carlo::prelude::*;

// Standard GBM model
let gbm = GeometricBrownianMotion::new(
    100.0,  // Initial price S₀
    0.08,   // Drift μ (8% annual return)
    0.20,   // Volatility σ (20% annual)
);

// Simulate
let engine = MonteCarloEngine::reproducible(42);
let result = engine.simulate(&gbm, &TimeHorizon::years(1));
```

### Merton Jump-Diffusion

For modeling crash risk:

```rust
use aprender::monte_carlo::prelude::*;

// Jump-diffusion with crash risk
let jump_model = MertonJumpDiffusion::new(
    100.0,   // Initial price
    0.08,    // Drift
    0.15,    // Diffusion volatility (lower due to jumps)
    1.0,     // Jump intensity λ (1 jump/year on average)
    -0.05,   // Mean jump size (5% drop)
    0.10,    // Jump size volatility
);

let engine = MonteCarloEngine::reproducible(42)
    .with_n_simulations(50_000);  // More sims for jump processes

let result = engine.simulate(&jump_model, &TimeHorizon::years(1));

// Jump models show fatter tails
let stats = result.final_value_statistics();
println!("With jumps - Min: ${:.2}, Max: ${:.2}", stats.min, stats.max);
```

### Empirical Bootstrap

Non-parametric simulation from historical data:

```rust
use aprender::monte_carlo::prelude::*;

// Historical daily returns
let historical_returns = vec![
    0.01, -0.02, 0.005, 0.015, -0.01, 0.02, -0.005,
    0.008, -0.015, 0.012, 0.003, -0.008, 0.018, -0.003,
    // ... more historical data
];

// Bootstrap model preserves empirical distribution
let bootstrap = EmpiricalBootstrap::new(100.0, &historical_returns);

let engine = MonteCarloEngine::reproducible(42);
let result = engine.simulate(&bootstrap, &TimeHorizon::days(252));
```

## Risk Metrics

### Value at Risk (VaR)

```rust
use aprender::monte_carlo::prelude::*;

// Historical VaR from return series
let returns = vec![-0.05, -0.02, 0.01, 0.03, 0.05, 0.02, -0.01, 0.04, -0.03, 0.00];

// 95% VaR: maximum loss at 95% confidence
let var_95 = VaR::historical(&returns, 0.95);
println!("95% VaR: {:.2}%", var_95 * 100.0);

// Multiple confidence levels
let var_90 = VaR::historical(&returns, 0.90);
let var_99 = VaR::historical(&returns, 0.99);

println!("VaR Ladder:");
println!("  90%: {:.2}%", var_90 * 100.0);
println!("  95%: {:.2}%", var_95 * 100.0);
println!("  99%: {:.2}%", var_99 * 100.0);
```

### Conditional VaR (Expected Shortfall)

```rust
use aprender::monte_carlo::prelude::*;

let returns = vec![-0.05, -0.02, 0.01, 0.03, 0.05, 0.02, -0.01, 0.04, -0.03, 0.00];

// CVaR: expected loss given we exceed VaR
let cvar_95 = CVaR::from_returns(&returns, 0.95);
let var_95 = VaR::historical(&returns, 0.95);

println!("95% VaR: {:.2}%", var_95 * 100.0);
println!("95% CVaR: {:.2}%", cvar_95 * 100.0);
println!("CVaR captures tail risk beyond VaR");

// CVaR is always >= VaR (more conservative)
assert!(cvar_95 >= var_95 - 0.001);
```

### Drawdown Analysis

```rust
use aprender::monte_carlo::prelude::*;

// Analyze drawdowns from simulation paths
let engine = MonteCarloEngine::reproducible(42)
    .with_n_simulations(1000);
let model = GeometricBrownianMotion::new(100.0, 0.08, 0.20);
let result = engine.simulate(&model, &TimeHorizon::years(5));

// Get drawdown statistics across all paths
let drawdown_stats = DrawdownAnalysis::from_paths(result.paths());

println!("Drawdown Statistics (5-year horizon):");
println!("  Mean Max Drawdown: {:.1}%", drawdown_stats.mean * 100.0);
println!("  Median Max Drawdown: {:.1}%", drawdown_stats.median * 100.0);
println!("  95th Percentile: {:.1}%", drawdown_stats.p95 * 100.0);
println!("  Worst Case: {:.1}%", drawdown_stats.max * 100.0);
```

## Risk-Adjusted Ratios

```rust
use aprender::monte_carlo::prelude::*;

let returns = vec![0.02, 0.01, -0.01, 0.03, 0.02, -0.02, 0.01, 0.04, -0.01, 0.02];
let risk_free_rate = 0.02;  // 2% annual
let benchmark_returns = vec![0.01, 0.005, -0.005, 0.02, 0.01, -0.01, 0.005, 0.02, 0.0, 0.01];

// Sharpe Ratio: return per unit of total risk
let sharpe = sharpe_ratio(&returns, risk_free_rate);
println!("Sharpe Ratio: {:.2}", sharpe);

// Sortino Ratio: return per unit of downside risk
let sortino = sortino_ratio(&returns, risk_free_rate, 0.0);
println!("Sortino Ratio: {:.2}", sortino);

// Information Ratio: excess return vs benchmark per tracking error
let info_ratio = information_ratio(&returns, &benchmark_returns);
println!("Information Ratio: {:.2}", info_ratio);

// Treynor Ratio: return per unit of systematic risk
let beta = 1.2;
let treynor = treynor_ratio(&returns, risk_free_rate, beta);
println!("Treynor Ratio: {:.2}", treynor);

// Omega Ratio: probability-weighted gains/losses
let threshold = 0.0;
let omega = omega_ratio(&returns, threshold);
println!("Omega Ratio: {:.2}", omega);

// Jensen's Alpha: excess return over CAPM prediction
let market_return = 0.10;
let alpha = jensens_alpha(&returns, risk_free_rate, beta, market_return);
println!("Jensen's Alpha: {:.2}%", alpha * 100.0);
```

## Comprehensive Risk Report

```rust
use aprender::monte_carlo::prelude::*;

fn generate_risk_report() {
    // Run simulation
    let engine = MonteCarloEngine::reproducible(42)
        .with_n_simulations(10_000)
        .with_variance_reduction(VarianceReduction::Antithetic);

    let model = GeometricBrownianMotion::new(100.0, 0.08, 0.20);
    let result = engine.simulate(&model, &TimeHorizon::years(1));

    // Generate comprehensive report
    let risk_free_rate = 0.02;
    let report = RiskReport::from_paths(result.paths(), risk_free_rate)
        .expect("Should generate report");

    // Print summary
    println!("{}", report.summary());

    // Or access individual metrics
    println!("\nKey Metrics:");
    println!("  95% VaR: {:.2}%", report.var_95 * 100.0);
    println!("  95% CVaR: {:.2}%", report.cvar_95 * 100.0);
    println!("  Sharpe Ratio: {:.2}", report.sharpe_ratio);
    println!("  Max Drawdown (median): {:.2}%", report.drawdown.median * 100.0);
}
```

## Variance Reduction

### Antithetic Variates

```rust
use aprender::monte_carlo::prelude::*;

// Without variance reduction
let engine_basic = MonteCarloEngine::reproducible(42)
    .with_n_simulations(10_000)
    .with_variance_reduction(VarianceReduction::None);

// With antithetic variates
let engine_antithetic = MonteCarloEngine::reproducible(42)
    .with_n_simulations(10_000)
    .with_variance_reduction(VarianceReduction::Antithetic);

let model = GeometricBrownianMotion::new(100.0, 0.08, 0.20);
let horizon = TimeHorizon::years(1);

let result_basic = engine_basic.simulate(&model, &horizon);
let result_antithetic = engine_antithetic.simulate(&model, &horizon);

let stats_basic = result_basic.final_value_statistics();
let stats_antithetic = result_antithetic.final_value_statistics();

println!("Basic - Mean: ${:.2}, Std: ${:.2}", stats_basic.mean, stats_basic.std);
println!("Antithetic - Mean: ${:.2}, Std: ${:.2}", stats_antithetic.mean, stats_antithetic.std);
// Antithetic should have lower standard error
```

## Convergence Monitoring

```rust
use aprender::monte_carlo::prelude::*;

// Engine with convergence target
let engine = MonteCarloEngine::reproducible(42)
    .with_n_simulations(100_000)
    .with_target_precision(0.01)  // 1% relative precision
    .with_max_simulations(100_000);

let model = GeometricBrownianMotion::new(100.0, 0.08, 0.20);
let result = engine.simulate(&model, &TimeHorizon::years(1));

// Check convergence diagnostics
let diagnostics = result.diagnostics();
println!("Convergence Diagnostics:");
println!("  Paths used: {}", result.n_paths());
println!("  Converged: {}", diagnostics.is_converged(0.01));
println!("  Relative std error: {:.4}", diagnostics.relative_std_error());
println!("  Effective sample size: {:.0}", diagnostics.effective_sample_size());
```

## Random Number Generation

```rust
use aprender::monte_carlo::prelude::*;

// Reproducible RNG
let mut rng = MonteCarloRng::new(42);

// Standard normal samples
let z1 = rng.normal(0.0, 1.0);
let z2 = rng.normal(0.0, 1.0);

// Uniform samples
let u = rng.uniform(0.0, 1.0);

// Exponential (for Poisson process)
let exp = rng.exponential(1.0);

// Same seed = same sequence
let mut rng2 = MonteCarloRng::new(42);
assert_eq!(rng2.normal(0.0, 1.0), z1);
```

## Time Horizon Configuration

```rust
use aprender::monte_carlo::prelude::*;

// Various time horizons
let daily = TimeHorizon::days(252);      // 1 trading year
let weekly = TimeHorizon::weeks(52);     // 1 year
let monthly = TimeHorizon::months(12);   // 1 year
let yearly = TimeHorizon::years(5);      // 5 years

// Custom horizon
let custom = TimeHorizon::custom(
    0.5,    // Total time (0.5 years = 6 months)
    126,    // Number of steps
);

println!("Daily horizon: {} steps over {} years", daily.n_steps(), daily.total_time());
```

## Portfolio Simulation

```rust
use aprender::monte_carlo::prelude::*;

fn simulate_portfolio() {
    let mut rng = MonteCarloRng::new(42);

    // Define assets
    let assets = vec![
        ("Stock A", 0.10, 0.25),  // (name, return, vol)
        ("Stock B", 0.08, 0.20),
        ("Bonds", 0.04, 0.05),
    ];

    let weights = vec![0.5, 0.3, 0.2];  // Portfolio weights
    let initial_value = 100_000.0;

    // Correlation matrix (simplified)
    let correlations = vec![
        vec![1.0, 0.6, 0.2],
        vec![0.6, 1.0, 0.3],
        vec![0.2, 0.3, 1.0],
    ];

    // Simulate 1000 portfolio paths
    let n_sims = 1000;
    let n_steps = 252;  // Daily for 1 year

    let mut portfolio_values: Vec<f64> = Vec::with_capacity(n_sims);

    for _ in 0..n_sims {
        let mut value = initial_value;

        for _ in 0..n_steps {
            // Simplified: uncorrelated returns for demo
            let mut portfolio_return = 0.0;
            for (i, &(_, mu, sigma)) in assets.iter().enumerate() {
                let daily_return = (mu / 252.0) + (sigma / 252.0_f64.sqrt()) * rng.normal(0.0, 1.0);
                portfolio_return += weights[i] * daily_return;
            }
            value *= 1.0 + portfolio_return;
        }

        portfolio_values.push(value);
    }

    // Calculate portfolio VaR
    let returns: Vec<f64> = portfolio_values.iter()
        .map(|&v| (v - initial_value) / initial_value)
        .collect();

    let var_95 = VaR::historical(&returns, 0.95);
    println!("Portfolio 95% VaR: ${:.0}", var_95 * initial_value);
}
```

## Running Examples

```bash
# Run Monte Carlo examples
cargo run --example monte_carlo_basic
cargo run --example monte_carlo_risk
cargo run --example monte_carlo_portfolio
```

## Feature Flags

The monte_carlo module is included by default. For the separate crate:

```toml
[dependencies]
aprender-monte-carlo = "0.1"
```

## References

- Glasserman (2003), "Monte Carlo Methods in Financial Engineering"
- Jorion (2006), "Value at Risk"
- Hull (2018), "Options, Futures, and Other Derivatives"
