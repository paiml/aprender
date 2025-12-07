# aprender-monte-carlo

Monte Carlo simulations for finance, stock market, and business forecasting.

Part of the [aprender](https://crates.io/crates/aprender) machine learning ecosystem.

## Features

- Stock price simulation using Geometric Brownian Motion (GBM)
- Historical S&P 500 data for backtesting
- Risk metrics (VaR, CVaR, Sharpe ratio)
- Variance reduction techniques

## Installation

```bash
cargo install aprender-monte-carlo
```

## Usage

```bash
# Run Monte Carlo simulation
aprender-monte-carlo simulate --ticker AAPL --days 252 --simulations 10000

# Calculate VaR
aprender-monte-carlo risk --ticker AAPL --confidence 0.95
```

## License

MIT
