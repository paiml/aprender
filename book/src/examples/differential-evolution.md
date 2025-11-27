# Case Study: Differential Evolution for Hyperparameter Optimization

This example demonstrates using Differential Evolution (DE) to optimize hyperparameters without requiring gradient information.

## The Problem

Traditional hyperparameter optimization faces challenges:
- Grid search scales exponentially with dimensions
- Random search may miss optimal regions
- Bayesian optimization requires probabilistic modeling

DE provides a simple, effective alternative for continuous hyperparameter spaces.

## Basic Usage

```rust
use aprender::metaheuristics::{
    DifferentialEvolution, SearchSpace, Budget, PerturbativeMetaheuristic
};

// Define a 5D sphere function (minimum at origin)
let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>();

// Create search space: 5 dimensions, bounds [-5, 5]
let space = SearchSpace::continuous(5, -5.0, 5.0);

// Run DE with 10,000 function evaluations
let mut de = DifferentialEvolution::default();
let result = de.optimize(&sphere, &space, Budget::Evaluations(10_000));

println!("Best solution: {:?}", result.solution);
println!("Objective value: {}", result.objective_value);
println!("Evaluations used: {}", result.evaluations);
```

## Hyperparameter Optimization Example

```rust
use aprender::metaheuristics::{
    DifferentialEvolution, SearchSpace, Budget, PerturbativeMetaheuristic
};

// Simulate ML model validation loss as function of hyperparameters
// params[0] = learning_rate (1e-5 to 1e-1)
// params[1] = regularization (1e-6 to 1e-2)
let validation_loss = |params: &[f64]| {
    let lr = params[0];
    let reg = params[1];

    // Simulated loss landscape with optimal around lr=0.01, reg=0.001
    let lr_term = (lr - 0.01).powi(2) / 0.0001;
    let reg_term = (reg - 0.001).powi(2) / 0.000001;
    let noise = 0.1 * (lr * 100.0).sin();  // Local optima

    lr_term + reg_term + noise
};

// Define heterogeneous bounds
let space = SearchSpace::Continuous {
    dim: 2,
    lower: vec![1e-5, 1e-6],
    upper: vec![1e-1, 1e-2],
};

// Configure DE
let mut de = DifferentialEvolution::new()
    .with_seed(42);  // Reproducibility

let result = de.optimize(&validation_loss, &space, Budget::Evaluations(5000));

println!("Optimal learning rate: {:.6}", result.solution[0]);
println!("Optimal regularization: {:.6}", result.solution[1]);
println!("Validation loss: {:.6}", result.objective_value);
```

## Mutation Strategies

Different strategies offer trade-offs:

```rust
use aprender::metaheuristics::{
    DifferentialEvolution, DEStrategy, SearchSpace, Budget, PerturbativeMetaheuristic
};

let objective = |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>();
let space = SearchSpace::continuous(10, -5.0, 5.0);
let budget = Budget::Evaluations(20_000);

// DE/rand/1/bin - Good exploration (default)
let mut de_rand = DifferentialEvolution::new()
    .with_strategy(DEStrategy::Rand1Bin)
    .with_seed(42);
let result_rand = de_rand.optimize(&objective, &space, budget.clone());

// DE/best/1/bin - Fast convergence, risk of premature convergence
let mut de_best = DifferentialEvolution::new()
    .with_strategy(DEStrategy::Best1Bin)
    .with_seed(42);
let result_best = de_best.optimize(&objective, &space, budget.clone());

// DE/current-to-best/1/bin - Balanced approach
let mut de_ctb = DifferentialEvolution::new()
    .with_strategy(DEStrategy::CurrentToBest1Bin)
    .with_seed(42);
let result_ctb = de_ctb.optimize(&objective, &space, budget);

println!("Rand1Bin: {:.6}", result_rand.objective_value);
println!("Best1Bin: {:.6}", result_best.objective_value);
println!("CurrentToBest1Bin: {:.6}", result_ctb.objective_value);
```

## Adaptive DE (JADE)

JADE adapts mutation factor F and crossover rate CR during optimization:

```rust
use aprender::metaheuristics::{
    DifferentialEvolution, SearchSpace, Budget, PerturbativeMetaheuristic
};

// Rastrigin function - highly multimodal
let rastrigin = |x: &[f64]| {
    let n = x.len() as f64;
    10.0 * n + x.iter()
        .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
        .sum::<f64>()
};

let space = SearchSpace::continuous(10, -5.12, 5.12);
let budget = Budget::Evaluations(50_000);

// Standard DE
let mut de_std = DifferentialEvolution::new().with_seed(42);
let result_std = de_std.optimize(&rastrigin, &space, budget.clone());

// JADE adaptive
let mut de_jade = DifferentialEvolution::new()
    .with_jade()
    .with_seed(42);
let result_jade = de_jade.optimize(&rastrigin, &space, budget);

println!("Standard DE: {:.4}", result_std.objective_value);
println!("JADE: {:.4}", result_jade.objective_value);
```

## Early Stopping with Convergence Detection

```rust
use aprender::metaheuristics::{
    DifferentialEvolution, SearchSpace, Budget, PerturbativeMetaheuristic
};

let objective = |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>();
let space = SearchSpace::continuous(5, -5.0, 5.0);

// Stop when no improvement > 1e-8 for 50 iterations
let budget = Budget::Convergence {
    patience: 50,
    min_delta: 1e-8,
    max_evaluations: 100_000,
};

let mut de = DifferentialEvolution::new().with_seed(42);
let result = de.optimize(&objective, &space, budget);

println!("Converged after {} evaluations", result.evaluations);
println!("Final value: {:.10}", result.objective_value);
println!("Termination: {:?}", result.termination);
```

## Convergence History

Track optimization progress for visualization:

```rust
use aprender::metaheuristics::{
    DifferentialEvolution, SearchSpace, Budget, PerturbativeMetaheuristic
};

let objective = |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>();
let space = SearchSpace::continuous(10, -5.0, 5.0);

let mut de = DifferentialEvolution::new().with_seed(42);
let result = de.optimize(&objective, &space, Budget::Iterations(100));

// Print convergence curve
println!("Generation | Best Value");
println!("-----------|-----------");
for (i, &val) in result.history.iter().enumerate().step_by(10) {
    println!("{:10} | {:.6}", i, val);
}
```

## Custom Parameters

Fine-tune DE behavior:

```rust
use aprender::metaheuristics::{
    DifferentialEvolution, DEStrategy, SearchSpace, Budget, PerturbativeMetaheuristic
};

let objective = |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>();
let space = SearchSpace::continuous(20, -10.0, 10.0);

// Custom configuration
let mut de = DifferentialEvolution::with_params(
    100,    // population_size: 100 individuals
    0.7,    // mutation_factor F: slightly lower for stability
    0.85,   // crossover_rate CR: high for good mixing
)
.with_strategy(DEStrategy::CurrentToBest1Bin)
.with_seed(42);

let result = de.optimize(&objective, &space, Budget::Evaluations(50_000));
println!("Result: {:.6}", result.objective_value);
```

## Serialization

Save and restore optimizer state:

```rust
use aprender::metaheuristics::DifferentialEvolution;

let de = DifferentialEvolution::new()
    .with_jade()
    .with_seed(42);

// Serialize to JSON
let json = serde_json::to_string_pretty(&de).unwrap();
println!("{}", json);

// Deserialize
let de_restored: DifferentialEvolution = serde_json::from_str(&json).unwrap();
```

## Active Learning Integration

Wrap DE with `ActiveLearningSearch` for uncertainty-based stopping:

```rust
use aprender::automl::{
    ActiveLearningSearch, DESearch, SearchSpace, SearchStrategy, TrialResult
};
use aprender::automl::params::RandomForestParam as RF;

let space = SearchSpace::new()
    .add_continuous(RF::NEstimators, 10.0, 500.0)
    .add_continuous(RF::MaxDepth, 2.0, 20.0);

// Wrap DE with active learning
let base = DESearch::new(10_000).with_jade().with_seed(42);
let mut search = ActiveLearningSearch::new(base)
    .with_uncertainty_threshold(0.1)  // Stop when CV < 0.1
    .with_min_samples(20);

// Pull system: only generate what's needed
let mut all_results = Vec::new();
while !search.should_stop() {
    let trials = search.suggest(&space, 10);
    if trials.is_empty() { break; }

    // Evaluate trials (your objective function)
    let results: Vec<TrialResult<RF>> = trials.iter().map(|t| {
        let score = evaluate_model(t);  // Your evaluation
        TrialResult { trial: t.clone(), score, metrics: Default::default() }
    }).collect();

    search.update(&results);
    all_results.extend(results);
}

println!("Stopped after {} evaluations (uncertainty: {:.4})",
    all_results.len(), search.uncertainty());
```

This eliminates **Muda** (waste) by stopping when confidence saturates.

## Best Practices

1. **Budget Selection**: Start with `10,000 × dim` evaluations
2. **Population Size**: Default auto-selection usually works well
3. **Strategy Choice**:
   - `Rand1Bin` for unknown landscapes (default)
   - `Best1Bin` for unimodal functions
   - `CurrentToBest1Bin` for balanced exploration/exploitation
4. **Adaptivity**: Use JADE for multimodal problems
5. **Reproducibility**: Always set seed for deterministic results
6. **Convergence**: Use `Budget::Convergence` for expensive objectives
7. **Active Learning**: Wrap with `ActiveLearningSearch` for expensive black-box functions

## Toyota Way Alignment

This implementation follows Toyota Way principles:

- **Jidoka**: Budget system prevents infinite loops
- **Kaizen**: JADE/SHADE continuously improve parameters
- **Muda Elimination**: Early stopping avoids wasted evaluations
- **Standard Work**: Deterministic seeds enable reproducible optimization
