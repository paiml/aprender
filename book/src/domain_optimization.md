# Bayesian Optimization

The optimization domain provides Gaussian Process-based Bayesian optimization for sample-efficient black-box optimization.

## Basic Usage

```rust,ignore
use simular::domains::optimization::{
    BayesianOptimizer, OptimizerConfig, AcquisitionFunction,
};

// Define objective function
let objective = |x: &[f64]| -> f64 {
    x[0].powi(2) + x[1].powi(2)  // Sphere function, min at (0, 0)
};

// Configure optimizer
let config = OptimizerConfig {
    bounds: vec![(-5.0, 5.0), (-5.0, 5.0)],  // 2D search space
    acquisition: AcquisitionFunction::ExpectedImprovement,
    seed: 42,
    ..Default::default()
};

let mut optimizer = BayesianOptimizer::new(config);

// Optimization loop
for _ in 0..20 {
    let x = optimizer.suggest();  // Get next point to evaluate
    let y = objective(&x);        // Evaluate objective
    optimizer.observe(x, y).unwrap();  // Add observation
}

// Get best found solution
let (best_x, best_y) = optimizer.best().unwrap();
println!("Best: x = {:?}, f(x) = {:.6}", best_x, best_y);
```

## OptimizerConfig

```rust,ignore
pub struct OptimizerConfig {
    /// Parameter bounds: (min, max) for each dimension
    pub bounds: Vec<(f64, f64)>,

    /// Acquisition function to use
    pub acquisition: AcquisitionFunction,

    /// Length scale for RBF kernel (default: 1.0)
    pub length_scale: f64,

    /// Signal variance for GP (default: 1.0)
    pub signal_variance: f64,

    /// Noise variance (default: 1e-6)
    pub noise_variance: f64,

    /// Random candidates for acquisition optimization (default: 1000)
    pub n_candidates: usize,

    /// RNG seed for reproducibility
    pub seed: u64,
}
```

## Acquisition Functions

### Expected Improvement (Default)

Balances exploration and exploitation:

```rust,ignore
let config = OptimizerConfig {
    acquisition: AcquisitionFunction::ExpectedImprovement,
    ..Default::default()
};
```

EI = E[max(f_best - f(x), 0)]

### Upper Confidence Bound (UCB)

Tunable exploration via kappa parameter:

```rust,ignore
let config = OptimizerConfig {
    acquisition: AcquisitionFunction::UCB { kappa: 2.0 },
    ..Default::default()
};
```

UCB = μ(x) - κ * σ(x) (for minimization)

Higher kappa = more exploration.

### Probability of Improvement (PoI)

Conservative strategy:

```rust,ignore
let config = OptimizerConfig {
    acquisition: AcquisitionFunction::ProbabilityOfImprovement,
    ..Default::default()
};
```

PoI = P(f(x) < f_best)

## Gaussian Process

The GP surrogate model uses RBF (squared exponential) kernel:

```rust,ignore
use simular::domains::optimization::GaussianProcess;

// Create GP
let mut gp = GaussianProcess::new(
    1.0,    // length_scale
    1.0,    // signal_variance
    1e-6,   // noise_variance
);

// Add observations
gp.add_observation(vec![0.0, 0.0], 1.5);
gp.add_observation(vec![1.0, 1.0], 0.8);

// Fit GP
gp.fit().unwrap();

// Predict at new point
let (mean, variance) = gp.predict(&[0.5, 0.5]);
println!("μ = {:.4}, σ² = {:.4}", mean, variance);
```

### RBF Kernel

k(x, x') = σ² * exp(-||x - x'||² / (2 * l²))

Where:
- σ² = signal variance (amplitude)
- l = length scale (smoothness)

## Example: Rosenbrock Function

```rust,ignore
use simular::domains::optimization::{
    BayesianOptimizer, OptimizerConfig, AcquisitionFunction,
};

fn main() {
    // Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    // Global minimum at (1, 1) with f(1,1) = 0
    let rosenbrock = |x: &[f64]| -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    };

    let config = OptimizerConfig {
        bounds: vec![(-2.0, 2.0), (-2.0, 2.0)],
        acquisition: AcquisitionFunction::ExpectedImprovement,
        length_scale: 0.5,
        seed: 42,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(config);

    println!("Iter |  x[0]  |  x[1]  |  f(x)");
    println!("-----|--------|--------|--------");

    for i in 0..25 {
        let x = optimizer.suggest();
        let y = rosenbrock(&x);
        optimizer.observe(x.clone(), y).unwrap();

        if i < 5 || i >= 22 {
            println!("{:>4} | {:>6.3} | {:>6.3} | {:>6.3}", i+1, x[0], x[1], y);
        } else if i == 5 {
            println!(" ... |  ...   |  ...   |  ...");
        }
    }

    let (best_x, best_y) = optimizer.best().unwrap();
    println!("\nBest: ({:.4}, {:.4}) with f(x) = {:.6}",
        best_x[0], best_x[1], best_y);
    println!("True optimum: (1.0, 1.0) with f(x) = 0");
}
```

## Example: 1D Optimization

```rust,ignore
fn main() {
    // f(x) = sin(x) + 0.1*x², minimum around x ≈ -0.88
    let f = |x: &[f64]| -> f64 {
        x[0].sin() + 0.1 * x[0] * x[0]
    };

    let config = OptimizerConfig {
        bounds: vec![(-3.0, 3.0)],
        acquisition: AcquisitionFunction::ExpectedImprovement,
        seed: 42,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(config);

    for _ in 0..15 {
        let x = optimizer.suggest();
        let y = f(&x);
        optimizer.observe(x, y).unwrap();
    }

    let (best_x, best_y) = optimizer.best().unwrap();
    println!("Best: x = {:.4}, f(x) = {:.6}", best_x[0], best_y);
}
```

## Comparing Acquisition Functions

```rust,ignore
let acquisition_fns = [
    ("EI", AcquisitionFunction::ExpectedImprovement),
    ("UCB(2)", AcquisitionFunction::UCB { kappa: 2.0 }),
    ("UCB(4)", AcquisitionFunction::UCB { kappa: 4.0 }),
    ("PoI", AcquisitionFunction::ProbabilityOfImprovement),
];

for (name, acq) in &acquisition_fns {
    let config = OptimizerConfig {
        bounds: vec![(-2.0, 2.0), (-2.0, 2.0)],
        acquisition: acq.clone(),
        seed: 42,  // Same seed for fair comparison
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(config);

    for _ in 0..20 {
        let x = optimizer.suggest();
        let y = rosenbrock(&x);
        optimizer.observe(x, y).unwrap();
    }

    let (_, best_y) = optimizer.best().unwrap();
    println!("{}: best f(x) = {:.6}", name, best_y);
}
```

## Reproducibility

Same seed = identical optimization trajectory:

```rust,ignore
fn optimize_with_seed(seed: u64) -> f64 {
    let config = OptimizerConfig {
        bounds: vec![(-2.0, 2.0), (-2.0, 2.0)],
        seed,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(config);

    for _ in 0..10 {
        let x = optimizer.suggest();
        let y = rosenbrock(&x);
        optimizer.observe(x, y).unwrap();
    }

    optimizer.best().unwrap().1
}

let result1 = optimize_with_seed(42);
let result2 = optimize_with_seed(42);
assert_eq!(result1, result2);  // Bit-identical!
```

## Hyperparameter Guidelines

| Parameter | Low Value | High Value |
|-----------|-----------|------------|
| `length_scale` | Rugged landscape | Smooth landscape |
| `signal_variance` | Low function variance | High function variance |
| `noise_variance` | Deterministic | Noisy observations |
| `kappa` (UCB) | Exploitation | Exploration |
| `n_candidates` | Fast, less accurate | Slower, more accurate |

## Next Steps

- [ML Training Simulation](./domain_ml.md) - Training optimization
- [Portfolio Risk](./scenario_portfolio.md) - Financial optimization
