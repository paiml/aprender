# Monte Carlo Methods

The Monte Carlo domain provides stochastic sampling with variance reduction techniques.

## Basic Usage

```rust,ignore
use simular::domains::monte_carlo::MonteCarloEngine;
use simular::engine::rng::SimRng;

let engine = MonteCarloEngine::with_samples(100_000);
let mut rng = SimRng::new(42);

// Estimate E[f(X)] where X ~ Uniform(0,1)
let result = engine.run(|x| x * x, &mut rng);

println!("E[X²] ≈ {:.6} ± {:.6}", result.estimate, result.std_error);
// True value: 1/3 ≈ 0.333333
```

## MonteCarloResult

Every Monte Carlo run returns statistics:

```rust,ignore
pub struct MonteCarloResult {
    pub estimate: f64,           // Point estimate
    pub std_error: f64,          // Standard error
    pub samples: usize,          // Number of samples
    pub confidence_interval: (f64, f64),  // 95% CI
    pub variance_reduction_factor: Option<f64>,
}
```

Usage:
```rust,ignore
let result = engine.run(|x| x, &mut rng);

println!("Estimate: {}", result.estimate);
println!("Std Error: {}", result.std_error);
println!("95% CI: ({}, {})",
    result.confidence_interval.0,
    result.confidence_interval.1);
println!("Contains 0.5? {}", result.contains(0.5));
println!("Relative error: {:.2}%", result.relative_error() * 100.0);
```

## Variance Reduction Techniques

### None (Standard Monte Carlo)

```rust,ignore
use simular::domains::monte_carlo::{MonteCarloEngine, VarianceReduction};

let engine = MonteCarloEngine::new(100_000, VarianceReduction::None);
```

### Antithetic Variates

Uses correlated pairs (U, 1-U) to reduce variance for monotonic functions:

```rust,ignore
let engine = MonteCarloEngine::new(100_000, VarianceReduction::Antithetic);
let result = engine.run(|x| x, &mut rng);

// For f(x) = x, antithetic gives perfect variance reduction
// because E[(f(U) + f(1-U))/2] = E[U + (1-U)]/2 = 0.5 exactly
```

### Control Variates

Use a correlated variable with known expectation:

```rust,ignore
// Estimate E[X²] using X as control variate
// We know E[X] = 0.5
let engine = MonteCarloEngine::new(
    100_000,
    VarianceReduction::ControlVariate {
        control_fn: |x| x,        // Control variable
        expectation: 0.5,         // Known E[X]
    },
);

let result = engine.run(|x| x * x, &mut rng);
```

### Importance Sampling

Sample from a proposal distribution that emphasizes important regions:

```rust,ignore
// Estimate E[X⁴] under Uniform(0,1) using Beta(2,1) proposal
fn sample_beta21(rng: &mut SimRng) -> f64 {
    rng.gen_f64().sqrt()  // Inverse CDF of Beta(2,1)
}

fn likelihood_ratio(x: f64) -> f64 {
    if x < f64::EPSILON { 1.0 } else { 1.0 / (2.0 * x) }
}

let engine = MonteCarloEngine::new(
    100_000,
    VarianceReduction::ImportanceSampling {
        sample_fn: sample_beta21,
        likelihood_ratio,
    },
);

let result = engine.run(|x| x.powi(4), &mut rng);
// True value: E[X⁴] = 1/5 = 0.2
```

### Self-Normalizing Importance Sampling

More robust when normalizing constant is unknown:

```rust,ignore
let engine = MonteCarloEngine::new(
    100_000,
    VarianceReduction::SelfNormalizingIS {
        sample_fn: |rng| rng.gen_f64(),
        weight_fn: |x| x.max(0.001),  // Emphasize larger x
    },
);
```

### Stratified Sampling

Divide domain into strata and sample each:

```rust,ignore
let engine = MonteCarloEngine::new(
    100_000,
    VarianceReduction::Stratified { num_strata: 10 },
);
```

## Multi-Dimensional Monte Carlo

```rust,ignore
// Estimate volume of unit sphere in 3D
let result = engine.run_nd(3, |x| {
    let r2: f64 = x.iter().map(|&xi| xi * xi).sum();
    if r2 <= 1.0 { 8.0 } else { 0.0 }  // 8 = 2³ for [-1,1]³ domain
}, &mut rng);

// True value: V = (4/3)πr³ ≈ 4.189
```

## Work-Stealing Monte Carlo

For variable-duration simulations:

```rust,ignore
use simular::domains::monte_carlo::{WorkStealingMonteCarlo, SimulationTask};

let ws = WorkStealingMonteCarlo::with_workers(4);

// Execute parallel simulations
let (results, stats) = ws.execute_with_stats(100_000, |task| {
    let mut rng = SimRng::new(task.seed);
    let x = rng.gen_f64();
    let y = rng.gen_f64();
    if x * x + y * y <= 1.0 { 4.0 } else { 0.0 }
});

println!("π ≈ {:.6}", stats.estimate);
```

The work-stealing scheduler handles load imbalance when some simulations take longer.

## Example: Pi Estimation

```rust,ignore
use simular::domains::monte_carlo::MonteCarloEngine;
use simular::engine::rng::SimRng;

fn main() {
    let mut rng = SimRng::new(42);

    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let engine = MonteCarloEngine::with_samples(n);

        let result = engine.run_nd(2, |x| {
            if x[0] * x[0] + x[1] * x[1] <= 1.0 { 4.0 } else { 0.0 }
        }, &mut rng);

        let error = (result.estimate - std::f64::consts::PI).abs();
        println!("n={:>7}: π ≈ {:.6}, error = {:.6}, std_err = {:.6}",
            n, result.estimate, error, result.std_error);
    }
}
```

Output:
```text
n=   1000: π ≈ 3.120000, error = 0.021593, std_err = 0.052017
n=  10000: π ≈ 3.143600, error = 0.002007, std_err = 0.016436
n= 100000: π ≈ 3.141120, error = 0.000473, std_err = 0.005197
n=1000000: π ≈ 3.141484, error = 0.000109, std_err = 0.001644
```

## Convergence Rate

Monte Carlo converges at O(n^{-1/2}) regardless of dimension:

```rust,ignore
// Standard error ∝ 1/√n
let engine_small = MonteCarloEngine::with_samples(1_000);
let engine_large = MonteCarloEngine::with_samples(100_000);

let result_small = engine_small.run(|x| x * x, &mut rng);
let result_large = engine_large.run(|x| x * x, &mut rng);

// Error ratio should be ~√100 = 10
let ratio = result_small.std_error / result_large.std_error;
println!("Error ratio: {:.1}", ratio);  // ~10
```

## Comparison of Techniques

| Technique | Variance Reduction | Best For |
|-----------|-------------------|----------|
| Standard | 1x (baseline) | General purpose |
| Antithetic | 2-10x | Monotonic functions |
| Control Variate | 2-100x | Correlated auxiliary |
| Importance | 10-1000x | Rare events, peaked functions |
| Stratified | 2-10x | Smooth functions |

## Next Steps

- [Bayesian Optimization](./domain_optimization.md) - GP-based search
- [Portfolio Risk (VaR)](./scenario_portfolio.md) - Financial Monte Carlo
- [Epidemic Models](./scenario_epidemic.md) - Stochastic SIR/SEIR
