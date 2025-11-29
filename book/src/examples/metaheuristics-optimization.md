# Case Study: Metaheuristics Optimization

This example demonstrates derivative-free global optimization using Aprender's metaheuristics module. We compare multiple algorithms on standard benchmark functions.

## Running the Example

```bash
cargo run --example metaheuristics_optimization
```

## Available Algorithms

| Algorithm | Type | Best For |
|-----------|------|----------|
| Differential Evolution | Population | Continuous HPO |
| Particle Swarm | Population | Smooth landscapes |
| Simulated Annealing | Single-point | Discrete/combinatorial |
| Genetic Algorithm | Population | Mixed spaces |
| Harmony Search | Population | Constraint handling |
| CMA-ES | Population | Low-dimension continuous |
| Binary GA | Population | Feature selection |

## Code Walkthrough

### Setting Up

```rust
use aprender::metaheuristics::{
    DifferentialEvolution, ParticleSwarm, SimulatedAnnealing,
    GeneticAlgorithm, HarmonySearch, CmaEs, BinaryGA,
    Budget, SearchSpace, PerturbativeMetaheuristic,
};
```

### Defining Objectives

```rust
// Sphere function: f(x) = Σxᵢ²
let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();

// Rosenbrock: f(x) = Σ[100(xᵢ₊₁-xᵢ²)² + (1-xᵢ)²]
let rosenbrock = |x: &[f64]| -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
};
```

### Running Optimizers

```rust
let dim = 5;
let space = SearchSpace::continuous(dim, -5.0, 5.0);
let budget = Budget::Evaluations(5000);

// Differential Evolution
let mut de = DifferentialEvolution::default().with_seed(42);
let result = de.optimize(&sphere, &space, budget.clone());
println!("DE: f(x*) = {:.6}", result.objective_value);

// CMA-ES
let mut cma = CmaEs::new(dim).with_seed(42);
let result = cma.optimize(&sphere, &space, budget.clone());
println!("CMA-ES: f(x*) = {:.6}", result.objective_value);
```

### Feature Selection with Binary GA

```rust
let feature_objective = |bits: &[f64]| {
    let selected: usize = bits.iter().filter(|&&b| b > 0.5).count();
    if selected == 0 { 100.0 } else { selected as f64 }
};

let space = SearchSpace::binary(10);
let mut ga = BinaryGA::default().with_seed(42);
let result = ga.optimize(&feature_objective, &space, Budget::Evaluations(2000));

let selected = BinaryGA::selected_features(&result.solution);
println!("Selected features: {:?}", selected);
```

## Expected Output

```
=== Metaheuristics Optimization Demo ===

1. Differential Evolution (DE/rand/1/bin)
   Sphere f(x*) = 0.000114
   Solution: [0.0006, -0.0080, ...]
   Evaluations: 5000

2. Particle Swarm Optimization (PSO)
   Sphere f(x*) = 0.000000
   Evaluations: 5000

3. Simulated Annealing (SA)
   Sphere f(x*) = 0.186239
   Evaluations: 450

4. Genetic Algorithm (SBX + Polynomial Mutation)
   Sphere f(x*) = 0.018537
   Evaluations: 5000

5. Harmony Search (HS)
   Sphere f(x*) = 0.000004
   Evaluations: 5000

6. CMA-ES (Covariance Matrix Adaptation)
   Sphere f(x*) = 0.000000
   Evaluations: 5000
```

## Algorithm Selection Guide

### Choose DE when:
- Continuous search space
- Hyperparameter optimization
- Moderate dimensionality (5-50)

### Choose CMA-ES when:
- Low dimensionality (<20)
- Smooth, continuous objectives
- Need automatic step-size adaptation

### Choose PSO when:
- Real-valued optimization
- Want fast convergence on unimodal functions
- Parallel evaluation is possible

### Choose Binary GA when:
- Feature selection problems
- Subset selection
- Binary decision variables

## CEC 2013 Benchmarks

The module includes standard benchmark functions:

```rust
use aprender::metaheuristics::benchmarks;

for info in benchmarks::all_benchmarks() {
    println!("{}: {} ({}, {})",
        info.name,
        if info.multimodal { "multimodal" } else { "unimodal" },
        if info.separable { "separable" } else { "non-separable" },
        format!("[{:.0}, {:.0}]", info.bounds.0, info.bounds.1)
    );
}
```

## See Also

- [Metaheuristics Theory](../ml-fundamentals/metaheuristics.md)
- [Differential Evolution](./differential-evolution.md)
