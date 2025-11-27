# Metaheuristics Theory

Metaheuristics are high-level problem-solving strategies for optimization problems where exact algorithms are impractical. Unlike gradient-based methods, they don't require derivatives and can escape local optima.

## Why Metaheuristics?

Traditional optimization has limitations:

| Method | Limitation |
|--------|------------|
| Gradient Descent | Requires differentiable objectives |
| Newton's Method | Requires Hessian computation |
| Convex Optimization | Assumes convex landscape |
| Grid Search | Exponential scaling with dimensions |

Metaheuristics address these by:
- **Derivative-free**: Work with black-box objectives
- **Global search**: Escape local optima
- **Versatile**: Handle mixed continuous/discrete spaces

## Algorithm Categories

### Perturbative Metaheuristics

Modify complete solutions through perturbation operators:

```text
┌─────────────────────────────────────────────────┐
│  Population-Based                               │
│  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Differential    │  │ Particle Swarm      │  │
│  │ Evolution (DE)  │  │ Optimization (PSO)  │  │
│  │                 │  │                     │  │
│  │ v = a + F(b-c)  │  │ v = wv + c₁r₁(p-x) │  │
│  │                 │  │     + c₂r₂(g-x)    │  │
│  └─────────────────┘  └─────────────────────┘  │
│                                                 │
│  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Genetic         │  │ CMA-ES              │  │
│  │ Algorithm (GA)  │  │                     │  │
│  │                 │  │ Covariance Matrix   │  │
│  │ Selection →     │  │ Adaptation          │  │
│  │ Crossover →     │  │                     │  │
│  │ Mutation        │  │ N(m, σ²C)           │  │
│  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  Single-Solution                                │
│  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Simulated       │  │ Hill Climbing       │  │
│  │ Annealing (SA)  │  │                     │  │
│  │                 │  │ Always accept       │  │
│  │ Accept worse    │  │ improvements        │  │
│  │ with P=e^(-Δ/T) │  │                     │  │
│  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Constructive Metaheuristics

Build solutions incrementally:

```text
┌─────────────────────────────────────────────────┐
│  Ant Colony Optimization (ACO)                  │
│                                                 │
│  τᵢⱼ(t+1) = (1-ρ)τᵢⱼ(t) + Δτᵢⱼ                │
│                                                 │
│  Pheromone guides probabilistic construction    │
│  Best for: TSP, routing, scheduling             │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  Tabu Search                                    │
│                                                 │
│  Memory-based local search                      │
│  Tabu list prevents cycling                     │
│  Aspiration criteria allow exceptions           │
└─────────────────────────────────────────────────┘
```

## Differential Evolution (DE)

DE is the primary algorithm in Aprender's metaheuristics module. It's particularly effective for continuous hyperparameter optimization.

### Algorithm

```text
For each target vector xᵢ in population:
  1. Mutation:    v = xₐ + F·(xᵦ - xᵧ)     # difference vector
  2. Crossover:   u = binomial(xᵢ, v, CR)  # trial vector
  3. Selection:   xᵢ' = u if f(u) ≤ f(xᵢ)  # greedy selection
```

### Mutation Strategies

| Strategy | Formula | Characteristics |
|----------|---------|-----------------|
| DE/rand/1/bin | v = xₐ + F(xᵦ - xᵧ) | Good exploration |
| DE/best/1/bin | v = x_best + F(xₐ - xᵦ) | Fast convergence |
| DE/current-to-best/1/bin | v = xᵢ + F(x_best - xᵢ) + F(xₐ - xᵦ) | Balanced |
| DE/rand/2/bin | v = xₐ + F(xᵦ - xᵧ) + F(xδ - xε) | More exploration |

### Adaptive Variants

**JADE** (Zhang & Sanderson, 2009):
- Adapts F and CR based on successful mutations
- External archive of inferior solutions
- μ_F updated via Lehmer mean
- μ_CR updated via weighted arithmetic mean

**SHADE** (Tanabe & Fukunaga, 2013):
- Success-history based parameter adaptation
- Circular memory buffer for F and CR
- More robust than JADE on multimodal functions

## Search Space Abstraction

Aprender uses a unified `SearchSpace` enum:

```rust
pub enum SearchSpace {
    // Continuous optimization (HPO, function optimization)
    Continuous { dim: usize, lower: Vec<f64>, upper: Vec<f64> },

    // Mixed continuous/discrete (neural architecture search)
    Mixed { dim: usize, lower: Vec<f64>, upper: Vec<f64>, discrete_dims: Vec<usize> },

    // Binary optimization (feature selection)
    Binary { dim: usize },

    // Permutation problems (TSP, scheduling)
    Permutation { size: usize },

    // Graph problems (routing, network design)
    Graph { num_nodes: usize, adjacency: Vec<Vec<(usize, f64)>>, heuristic: Option<Vec<Vec<f64>>> },
}
```

## Budget Control

Three termination strategies:

```rust
pub enum Budget {
    // Precise evaluation counting (recommended for benchmarks)
    Evaluations(usize),

    // Generation/iteration based
    Iterations(usize),

    // Early stopping with convergence detection
    Convergence {
        patience: usize,      // iterations without improvement
        min_delta: f64,       // minimum improvement threshold
        max_evaluations: usize, // safety bound
    },
}
```

## Active Learning (Muda Elimination)

Traditional batch generation ("Push System") produces many redundant samples.
Active Learning implements a "Pull System" - only generating samples while
uncertainty is high (Settles, 2009).

```text
┌─────────────────────────────────────────────────────────────┐
│  Push System (Wasteful)          Pull System (Lean)         │
│  ┌─────────────────────┐         ┌─────────────────────┐   │
│  │ Generate 100K       │         │ Generate batch      │   │
│  │ samples blindly     │         │ while uncertain     │   │
│  │         ↓           │         │         ↓           │   │
│  │ 90% redundant       │         │ Evaluate & update   │   │
│  │ (low info gain)     │         │         ↓           │   │
│  │         ↓           │         │ Check uncertainty   │   │
│  │ Wasted compute      │         │         ↓           │   │
│  └─────────────────────┘         │ Stop when confident │   │
│                                  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Uncertainty Estimation

Uses coefficient of variation (CV = σ/μ):
- **Low CV**: Consistent scores → high confidence → stop
- **High CV**: Variable scores → low confidence → continue

### Usage

```rust
use aprender::automl::{ActiveLearningSearch, DESearch, SearchStrategy};

let base = DESearch::new(10_000).with_jade();
let mut search = ActiveLearningSearch::new(base)
    .with_uncertainty_threshold(0.1)  // Stop when CV < 0.1
    .with_min_samples(20);            // Need at least 20 samples

// Pull system loop
while !search.should_stop() {
    let trials = search.suggest(&space, 10);
    if trials.is_empty() { break; }

    let results = evaluate(&trials);
    search.update(&results);  // Updates uncertainty estimate
}
// Stops early when confidence saturates
```

## When to Use Metaheuristics

### Good Use Cases

1. **Hyperparameter Optimization**: Learning rate, regularization, architecture choices
2. **Black-box Functions**: Simulations, expensive experiments
3. **Multimodal Landscapes**: Many local optima
4. **Mixed Search Spaces**: Continuous + categorical variables

### When to Prefer Other Methods

1. **Convex Problems**: Use convex optimizers (faster convergence)
2. **Differentiable Objectives**: Gradient methods are more efficient
3. **Very Low Budget**: Random search may be comparable
4. **High Dimensions (>100)**: Consider Bayesian optimization

## Benchmark Functions

Standard test functions for algorithm comparison:

| Function | Formula | Characteristics |
|----------|---------|-----------------|
| Sphere | f(x) = Σxᵢ² | Unimodal, separable |
| Rosenbrock | f(x) = Σ[100(xᵢ₊₁-xᵢ²)² + (1-xᵢ)²] | Unimodal, narrow valley |
| Rastrigin | f(x) = 10n + Σ[xᵢ²-10cos(2πxᵢ)] | Highly multimodal |
| Ackley | f(x) = -20exp(-0.2√(Σxᵢ²/n)) - exp(Σcos(2πxᵢ)/n) + 20 + e | Multimodal, nearly flat |

## References

1. Storn, R. & Price, K. (1997). "Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces." *Journal of Global Optimization*, 11(4), 341-359.

2. Zhang, J. & Sanderson, A.C. (2009). "JADE: Adaptive Differential Evolution with Optional External Archive." *IEEE Transactions on Evolutionary Computation*, 13(5), 945-958.

3. Tanabe, R. & Fukunaga, A. (2013). "Success-History Based Parameter Adaptation for Differential Evolution." *IEEE Congress on Evolutionary Computation*, 71-78.

4. Kennedy, J. & Eberhart, R. (1995). "Particle Swarm Optimization." *IEEE International Conference on Neural Networks*, 1942-1948.

5. Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial." *arXiv:1604.00772*.

6. Settles, B. (2009). "Active Learning Literature Survey." *University of Wisconsin-Madison Computer Sciences Technical Report 1648*.
