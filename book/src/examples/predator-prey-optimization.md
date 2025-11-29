# Predator-Prey Ecosystem Optimization

This example demonstrates using Differential Evolution to optimize parameters of a Lotka-Volterra predator-prey model to match observed population data.

## The Lotka-Volterra Model

The classic predator-prey equations describe population dynamics:

```
dx/dt = αx - βxy    (prey: growth minus predation)
dy/dt = δxy - γy    (predator: growth from prey minus death)
```

Where:
- **x**: Prey population (e.g., rabbits)
- **y**: Predator population (e.g., foxes)
- **α**: Prey birth rate
- **β**: Predation rate
- **δ**: Predator reproduction efficiency
- **γ**: Predator death rate

### Population Dynamics

```text
┌────────────────────────────────────────────────────────┐
│  Population                                             │
│  ▲                                                      │
│  │     ╭──╮        ╭──╮        ╭──╮                    │
│  │    ╱    ╲      ╱    ╲      ╱    ╲     Prey         │
│  │   ╱      ╲    ╱      ╲    ╱      ╲                  │
│  │  ╱        ╲  ╱        ╲  ╱        ╲                 │
│  │ ╱    ╭─╮   ╲╱    ╭─╮   ╲╱    ╭─╮                   │
│  │╱    ╱   ╲       ╱   ╲       ╱   ╲  Predator        │
│  └─────────────────────────────────────────────▶ Time  │
│                                                         │
│  Predators lag behind prey in classic boom-bust cycles  │
└────────────────────────────────────────────────────────┘
```

## Running the Example

```bash
cargo run --example predator_prey_optimization
```

## The Optimization Problem

**Given**: Observed population time series data
**Find**: Parameters (α, β, δ, γ) that minimize error between model and observations

### Why Metaheuristics?

1. **Non-convex objective**: Multiple parameter combinations can produce similar dynamics
2. **Coupled parameters**: Changes in one affect optimal values of others
3. **Numerical simulation**: No analytical gradients available

## Code Walkthrough

### Model Simulation

```rust,ignore
fn simulate_lotka_volterra(
    params: &LotkaVolterraParams,
    x0: f64,      // Initial prey
    y0: f64,      // Initial predator
    dt: f64,      // Time step
    steps: usize, // Simulation length
) -> Vec<(f64, f64)> {
    let mut trajectory = Vec::with_capacity(steps);
    let mut x = x0;
    let mut y = y0;

    for _ in 0..steps {
        trajectory.push((x, y));

        // Lotka-Volterra equations (Euler method)
        let dx = params.alpha * x - params.beta * x * y;
        let dy = params.delta * x * y - params.gamma * y;

        x += dx * dt;
        y += dy * dt;
        x = x.max(0.0);  // Prevent negative populations
        y = y.max(0.0);
    }

    trajectory
}
```

### Optimization Setup

```rust,ignore
use aprender::metaheuristics::{
    Budget, DifferentialEvolution, PerturbativeMetaheuristic, SearchSpace,
};

// Search space: [alpha, beta, delta, gamma]
let space = SearchSpace::Continuous {
    dim: 4,
    lower: vec![0.1, 0.01, 0.01, 0.1],
    upper: vec![2.0, 1.0, 0.5, 1.0],
};

// Objective: Mean Squared Error
let objective = |params_vec: &[f64]| -> f64 {
    let params = LotkaVolterraParams {
        alpha: params_vec[0],
        beta: params_vec[1],
        delta: params_vec[2],
        gamma: params_vec[3],
    };

    let simulated = simulate_lotka_volterra(&params, 10.0, 5.0, 0.1, 100);

    // MSE between observed and simulated
    observed.iter().zip(simulated.iter())
        .map(|((ox, oy), (sx, sy))| (ox - sx).powi(2) + (oy - sy).powi(2))
        .sum::<f64>() / observed.len() as f64
};
```

### Running DE

```rust,ignore
let mut de = DifferentialEvolution::default().with_seed(42);
let result = de.optimize(&objective, &space, Budget::Evaluations(5000));

println!("Recovered parameters:");
println!("  α = {:.4} (true: {:.4})", result.solution[0], true_params.alpha);
println!("  β = {:.4} (true: {:.4})", result.solution[1], true_params.beta);
println!("  δ = {:.4} (true: {:.4})", result.solution[2], true_params.delta);
println!("  γ = {:.4} (true: {:.4})", result.solution[3], true_params.gamma);
```

## Sample Output

```text
=== Predator-Prey Ecosystem Parameter Optimization ===

True parameters (to be recovered):
  α (prey birth rate):     1.100
  β (predation rate):      0.400
  δ (predator growth):     0.100
  γ (predator death rate): 0.400

=== Method 1: Differential Evolution ===
DE Result:
  α = 1.1041 (true: 1.1000)
  β = 0.4013 (true: 0.4000)
  δ = 0.0997 (true: 0.1000)
  γ = 0.3986 (true: 0.4000)
  MSE: 0.000043

Parameter Recovery Error: 0.0046 (excellent!)

=== Population Dynamics with Recovered Parameters ===

Time  Prey(Obs) Prey(Sim)  Pred(Obs) Pred(Sim)
----  --------- ---------  --------- ---------
   0     10.00     10.00       5.00      5.00
  10      2.61      2.61       6.20      6.19
  20      0.76      0.76       4.82      4.82
  30      0.43      0.43       3.40      3.40
```

## Applications

This parameter estimation technique applies to many real-world systems:

| Domain | System | Parameters |
|--------|--------|------------|
| **Ecology** | Predator-prey, competition | Birth/death rates |
| **Epidemiology** | SIR/SEIR models | Transmission, recovery rates |
| **Economics** | Market dynamics | Supply/demand elasticities |
| **Chemistry** | Reaction kinetics | Rate constants |
| **Physics** | Oscillators | Damping, frequency |

## Comparison with Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **DE** | Global search, no gradients | Slower than gradient methods |
| **Grid Search** | Simple, deterministic | Exponential scaling |
| **Bayesian** | Uncertainty quantification | Complex implementation |
| **Gradient Descent** | Fast convergence | Needs differentiable simulator |

## Tips for Parameter Estimation

1. **Normalize data**: Scale populations to similar ranges
2. **Multiple runs**: Use different seeds to assess robustness
3. **Bounds**: Set reasonable parameter ranges from domain knowledge
4. **Regularization**: Add penalty for extreme parameter values

## References

1. Lotka, A.J. (1925). *Elements of Physical Biology*. Williams & Wilkins.
2. Volterra, V. (1926). "Variations and fluctuations in the number of individuals in cohabiting animal species." *Mem. Acad. Lincei*, 2, 31-113.
3. Storn, R. & Price, K. (1997). "Differential Evolution." *Journal of Global Optimization*, 11(4), 341-359.
