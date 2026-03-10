# Case Study: Evolutionary Merge Optimization

**Ticket**: GH-444 | **Contract**: `merge.md §evolutionary`

## Overview

Evolutionary merge uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
to automatically find optimal merge weights when combining multiple model
checkpoints. Instead of manually tuning weights, the optimizer explores the
weight space and finds the combination that minimizes a user-defined objective
(e.g., perplexity on calibration data).

## API

```rust
use aprender::format::converter::evolutionary_merge::{
    evolutionary_merge, EvolutionaryMergeConfig,
};
use aprender::format::MergeStrategy;

let config = EvolutionaryMergeConfig {
    num_models: 3,
    max_evaluations: 100,
    strategy: MergeStrategy::Weighted,
    sigma: 0.3,    // CMA-ES step size
    seed: 42,      // Reproducibility
    ..Default::default()
};

let result = evolutionary_merge(&models, &config, |merged| {
    // Return score to MINIMIZE (e.g., perplexity)
    evaluate_perplexity(&merged)
});

println!("Optimal weights: {:?}", result.weights);
println!("Best score: {:.4}", result.best_score);

// Use result.merge_options directly with apr_merge
```

## Key Design Decisions

1. **Softmax parameterization**: Raw CMA-ES parameters are mapped through
   softmax to ensure weights are always positive and sum to 1.0.
2. **In-memory merging**: Tensor merging happens in-memory without file I/O,
   enabling fast objective function evaluation during optimization.
3. **Strategy-agnostic**: Works with Average, Weighted, and SLERP strategies.
   TIES/DARE density and drop_rate can also be optimized.

## Falsification Tests

| Test | Property |
|------|----------|
| FALSIFY-EVOL-MERGE-001 | Weights always sum to 1.0 |
| FALSIFY-EVOL-MERGE-002 | CMA-ES improves over equal-weight baseline |
| FALSIFY-EVOL-MERGE-003 | Result produces valid MergeOptions |

## Run the Example

```bash
cargo run --example evolutionary_merge
```
