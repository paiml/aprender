# Portfolio Risk (VaR) Scenario

The portfolio scenario provides Value at Risk calculations using Monte Carlo simulation.

## Basic Usage

```rust,ignore
use simular::scenarios::{PortfolioScenario, PortfolioConfig, VaRResult};

let config = PortfolioConfig {
    assets: vec![
        Asset { name: "SPY", weight: 0.6, annual_return: 0.10, volatility: 0.15 },
        Asset { name: "TLT", weight: 0.3, annual_return: 0.04, volatility: 0.12 },
        Asset { name: "GLD", weight: 0.1, annual_return: 0.05, volatility: 0.18 },
    ],
    correlation_matrix: Some(vec![
        vec![1.0, -0.2, 0.0],
        vec![-0.2, 1.0, 0.1],
        vec![0.0, 0.1, 1.0],
    ]),
    initial_value: 1_000_000.0,
    horizon_days: 10,
    seed: 42,
};

let scenario = PortfolioScenario::new(config);
```

## Value at Risk

```rust,ignore
// Calculate VaR at different confidence levels
let var_95 = scenario.var(0.95)?;
let var_99 = scenario.var(0.99)?;

println!("10-day 95% VaR: ${:.2}", var_95);
println!("10-day 99% VaR: ${:.2}", var_99);
```

## Conditional VaR (CVaR/ES)

Expected Shortfall - average loss beyond VaR:

```rust,ignore
let cvar_95 = scenario.cvar(0.95)?;
println!("10-day 95% CVaR: ${:.2}", cvar_95);

// CVaR is always >= VaR
assert!(cvar_95 >= var_95);
```

## Monte Carlo Simulation

```rust,ignore
let config = PortfolioConfig {
    n_simulations: 100_000,
    ..Default::default()
};

let scenario = PortfolioScenario::new(config);

// Run simulation
let results = scenario.simulate()?;

println!("Mean return: {:.2}%", results.mean_return * 100.0);
println!("Std dev: {:.2}%", results.std_dev * 100.0);
println!("Min: {:.2}%", results.min_return * 100.0);
println!("Max: {:.2}%", results.max_return * 100.0);
```

## VaRResult

```rust,ignore
pub struct VaRResult {
    pub confidence: f64,
    pub horizon_days: u32,
    pub var: f64,
    pub cvar: f64,
    pub portfolio_value: f64,
    pub n_simulations: usize,
}
```

## Historical Simulation

```rust,ignore
let config = PortfolioConfig {
    method: VaRMethod::Historical {
        returns: historical_returns,  // Vec<Vec<f64>>
        window_days: 252,             // 1 year
    },
    ..Default::default()
};

let scenario = PortfolioScenario::new(config);
let var = scenario.var(0.95)?;
```

## Parametric VaR

Assumes normal distribution:

```rust,ignore
let config = PortfolioConfig {
    method: VaRMethod::Parametric,
    ..Default::default()
};

let scenario = PortfolioScenario::new(config);

// Analytical calculation
let var = scenario.parametric_var(0.95)?;
```

## Stress Testing

```rust,ignore
let scenario = PortfolioScenario::new(config);

// Define stress scenarios
let stress_tests = vec![
    StressTest { name: "2008 Crisis", equity: -0.40, bonds: 0.10, gold: 0.25 },
    StressTest { name: "2020 Crash", equity: -0.35, bonds: 0.05, gold: 0.10 },
    StressTest { name: "Rate Shock", equity: -0.10, bonds: -0.15, gold: 0.05 },
];

for test in stress_tests {
    let loss = scenario.stress_test(&test)?;
    println!("{}: ${:.2} loss", test.name, loss);
}
```

## Correlation Analysis

```rust,ignore
let scenario = PortfolioScenario::new(config);

// Analyze correlation impact
let uncorrelated_var = scenario.var_uncorrelated(0.95)?;
let correlated_var = scenario.var(0.95)?;

println!("VaR without correlation: ${:.2}", uncorrelated_var);
println!("VaR with correlation: ${:.2}", correlated_var);
println!("Diversification benefit: ${:.2}",
    uncorrelated_var - correlated_var);
```

## Time Scaling

```rust,ignore
// Square root of time rule
let var_1day = scenario.var_1day(0.95)?;
let var_10day = var_1day * (10.0_f64).sqrt();

println!("1-day VaR: ${:.2}", var_1day);
println!("10-day VaR (scaled): ${:.2}", var_10day);
```

## Example: Portfolio Optimization

```rust,ignore
use simular::scenarios::{PortfolioScenario, PortfolioConfig};

fn main() {
    // Test different allocations
    let allocations = vec![
        ("Conservative", vec![0.2, 0.7, 0.1]),
        ("Moderate", vec![0.6, 0.3, 0.1]),
        ("Aggressive", vec![0.9, 0.05, 0.05]),
    ];

    println!("Portfolio       | Return | Vol   | VaR 95%  | Sharpe");
    println!("----------------|--------|-------|----------|-------");

    for (name, weights) in allocations {
        let config = PortfolioConfig {
            assets: vec![
                Asset { name: "Equity", weight: weights[0], annual_return: 0.10, volatility: 0.18 },
                Asset { name: "Bonds", weight: weights[1], annual_return: 0.04, volatility: 0.06 },
                Asset { name: "Gold", weight: weights[2], annual_return: 0.05, volatility: 0.15 },
            ],
            initial_value: 1_000_000.0,
            horizon_days: 252,
            n_simulations: 10_000,
            seed: 42,
            ..Default::default()
        };

        let scenario = PortfolioScenario::new(config);
        let results = scenario.simulate()?;

        let sharpe = (results.mean_return - 0.02) / results.std_dev;

        println!("{:15} | {:>5.1}% | {:>4.1}% | ${:>7.0} | {:>5.2}",
            name,
            results.mean_return * 100.0,
            results.std_dev * 100.0,
            scenario.var(0.95)?,
            sharpe);
    }
}
```

## Risk Metrics

```rust,ignore
let scenario = PortfolioScenario::new(config);
let metrics = scenario.risk_metrics()?;

println!("Volatility: {:.2}%", metrics.volatility * 100.0);
println!("Sharpe Ratio: {:.2}", metrics.sharpe_ratio);
println!("Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
println!("Sortino Ratio: {:.2}", metrics.sortino_ratio);
println!("Calmar Ratio: {:.2}", metrics.calmar_ratio);
```

## Next Steps

- [Epidemic Models](./scenario_epidemic.md) - SIR/SEIR models
- [Monte Carlo Methods](./domain_monte_carlo.md) - Underlying techniques
