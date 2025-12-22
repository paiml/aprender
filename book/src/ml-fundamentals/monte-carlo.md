# Monte Carlo Simulation Theory

Monte Carlo methods use random sampling to solve computational problems that are difficult to solve deterministically. Named after the famous casino, these methods are essential for financial modeling, risk analysis, and uncertainty quantification.

## Core Concept

The fundamental idea: **approximate expected values through random sampling**.

For a random variable X with unknown distribution:

```
E[f(X)] ≈ (1/N) Σᵢ f(Xᵢ)
```

As N → ∞, this approximation converges to the true expected value (Law of Large Numbers).

## Standard Error and Convergence

The Monte Carlo estimator's standard error decreases as:

```
SE = σ / √N
```

Where σ is the standard deviation of f(X). Key implications:
- To halve the error, quadruple the samples
- 10,000 simulations → ~1% relative error
- 1,000,000 simulations → ~0.1% relative error

## Financial Models

### Geometric Brownian Motion (GBM)

The standard model for stock prices:

```
dS = μS dt + σS dW
```

Where:
- S = stock price
- μ = drift (expected return)
- σ = volatility
- dW = Wiener process (random walk)

**Discrete simulation:**

```
S(t+Δt) = S(t) × exp((μ - σ²/2)Δt + σ√Δt × Z)
```

Where Z ~ N(0,1).

### Merton Jump-Diffusion

Extends GBM with discontinuous jumps for crash risk:

```
dS = μS dt + σS dW + S dJ
```

Where J is a Poisson jump process:
- λ = jump intensity (jumps per year)
- μⱼ = mean jump size
- σⱼ = jump size volatility

### Empirical Bootstrap

Non-parametric simulation using historical data:

1. Collect historical returns
2. Sample with replacement
3. Compound to form price paths

Advantages:
- No distributional assumptions
- Captures fat tails automatically
- Preserves autocorrelation structure

## Risk Metrics

### Value at Risk (VaR)

VaR answers: "What is the maximum loss at confidence level α?"

```
P(Loss ≤ VaR_α) = α
```

**Example:** 95% daily VaR of $1M means there's a 5% chance of losing more than $1M in one day.

**Calculation methods:**
1. **Historical**: Sort losses, take α percentile
2. **Parametric**: Assume normal distribution, VaR = μ + σ × z_α
3. **Monte Carlo**: Simulate scenarios, compute percentile

### Conditional VaR (CVaR / Expected Shortfall)

CVaR answers: "If we exceed VaR, what's the expected loss?"

```
CVaR_α = E[Loss | Loss > VaR_α]
```

CVaR is a **coherent risk measure** (satisfies subadditivity), unlike VaR.

| Property | VaR | CVaR |
|----------|-----|------|
| Subadditive | No | Yes |
| Tail sensitivity | Low | High |
| Regulatory use | Basel II | Basel III |

### Maximum Drawdown

The largest peak-to-trough decline:

```
MDD = max{(Peak(t) - Trough(t')) / Peak(t)}
```

Where t' > t. Measures worst-case historical loss from any peak.

## Risk-Adjusted Return Ratios

### Sharpe Ratio

Return per unit of total risk:

```
Sharpe = (Rₚ - Rᶠ) / σₚ
```

Where:
- Rₚ = portfolio return
- Rᶠ = risk-free rate
- σₚ = portfolio volatility

| Sharpe | Interpretation |
|--------|----------------|
| < 1.0 | Below average |
| 1.0-2.0 | Good |
| 2.0-3.0 | Very good |
| > 3.0 | Excellent |

### Sortino Ratio

Like Sharpe, but only penalizes downside volatility:

```
Sortino = (Rₚ - Rᶠ) / σ_downside
```

Where σ_downside only considers negative returns.

### Calmar Ratio

Return per unit of drawdown risk:

```
Calmar = Annual Return / Maximum Drawdown
```

### Other Ratios

| Ratio | Formula | Use Case |
|-------|---------|----------|
| Treynor | (Rₚ - Rᶠ) / β | Systematic risk |
| Information | (Rₚ - Rᵦ) / σₑ | Active management |
| Omega | E[gains] / E[losses] | Non-normal returns |
| Jensen's α | Rₚ - [Rᶠ + β(Rₘ - Rᶠ)] | Excess return |

## Variance Reduction Techniques

### Antithetic Variates

For each random sample Z, also use -Z:

```
Estimator = (f(Z) + f(-Z)) / 2
```

This creates negatively correlated samples, reducing variance when f is monotonic.

**Variance reduction factor:** Up to 50% for linear functions.

### Control Variates

Use a correlated variable with known expectation:

```
Adjusted = f(X) - c × (g(X) - E[g(X)])
```

Where c is chosen to minimize variance.

### Importance Sampling

Sample from a different distribution q(x) that emphasizes important regions:

```
E_p[f(X)] = E_q[f(X) × p(X)/q(X)]
```

Critical for rare event simulation (e.g., extreme losses).

### Stratified Sampling

Divide the sample space into strata and sample proportionally:

```
Space = Stratum₁ ∪ Stratum₂ ∪ ... ∪ Stratumₖ
```

Ensures coverage of the entire distribution.

## Convergence Diagnostics

### Effective Sample Size (ESS)

Accounts for correlation between samples:

```
ESS = N / (1 + 2 Σₖ ρₖ)
```

Where ρₖ is the autocorrelation at lag k.

If ESS << N, samples are highly correlated and provide less information.

### R-hat (Gelman-Rubin)

For multiple chains, compare within-chain and between-chain variance:

```
R̂ = √(((n-1)/n × W + (1/n) × B) / W)
```

- R̂ < 1.1: Chains have converged
- R̂ > 1.1: Need more samples

## Reproducibility

Monte Carlo simulations should be **reproducible**:

1. **Seed the RNG**: Use explicit seeds for reproducibility
2. **Document parameters**: Record all simulation settings
3. **Version control**: Track code changes
4. **Validate**: Compare against analytical solutions when possible

## Applications

1. **Option Pricing**: Price path-dependent options (Asian, barrier, lookback)
2. **Portfolio VaR**: Aggregate risk across correlated assets
3. **Credit Risk**: Default correlation and loss distributions
4. **Insurance**: Aggregate claims modeling
5. **Project Finance**: Revenue uncertainty quantification

## References

- Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering"
- Jorion, P. (2006). "Value at Risk"
- Hull, J. (2018). "Options, Futures, and Other Derivatives"
- Artzner, P. et al. (1999). "Coherent Measures of Risk"
