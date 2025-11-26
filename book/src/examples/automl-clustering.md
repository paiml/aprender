# Case Study: AutoML Clustering (TPE)

This example demonstrates using TPE (Tree-structured Parzen Estimator) to automatically find the optimal number of clusters for K-Means.

## Running the Example

```bash
cargo run --example automl_clustering
```

## Overview

Finding the optimal number of clusters (K) is a fundamental challenge in unsupervised learning. This example shows how to automate this process using aprender's AutoML module with TPE optimization.

**Key Concepts:**
- Type-safe parameter enums (Poka-Yoke design)
- TPE-based Bayesian optimization
- Silhouette score as objective function
- AutoTuner with early stopping

## The Problem

Given unlabeled data, we want to find the best value of K for K-Means clustering. Traditional approaches include:
- Elbow method (manual inspection)
- Silhouette analysis (manual comparison)
- Gap statistic (computationally expensive)

AutoML automates this by treating K as a hyperparameter to optimize.

## Code Walkthrough

### 1. Define Custom Parameter Enum

```rust
use aprender::automl::params::ParamKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum KMeansParam {
    NClusters,
}

impl ParamKey for KMeansParam {
    fn name(&self) -> &'static str {
        match self {
            KMeansParam::NClusters => "n_clusters",
        }
    }
}
```

This provides compile-time safety—typos are caught during compilation, not at runtime.

### 2. Define Search Space

```rust
use aprender::automl::SearchSpace;

let space: SearchSpace<KMeansParam> = SearchSpace::new()
    .add(KMeansParam::NClusters, 2..11); // K ∈ [2, 10]
```

### 3. Configure TPE Optimizer

```rust
use aprender::automl::TPE;

let tpe = TPE::new(15)
    .with_seed(42)
    .with_startup_trials(3)  // Random exploration first
    .with_gamma(0.25);       // Top 25% as "good"
```

TPE configuration:
- **15 trials**: Maximum optimization budget
- **3 startup trials**: Random sampling before model kicks in
- **gamma=0.25**: Top 25% of observations are "good"

### 4. Define Objective Function

```rust
let objective = |trial| {
    let k = trial.get_usize(&KMeansParam::NClusters).unwrap_or(3);

    // Run K-Means multiple times to reduce variance
    let mut scores = Vec::new();
    for seed in [42, 123, 456] {
        let mut kmeans = KMeans::new(k)
            .with_max_iter(100)
            .with_random_state(seed);

        if kmeans.fit(&data).is_ok() {
            let labels = kmeans.predict(&data);
            let score = silhouette_score(&data, &labels);
            scores.push(score);
        }
    }

    // Average silhouette score
    scores.iter().sum::<f32>() / scores.len() as f32
};
```

**Why average multiple runs?** K-Means initialization is stochastic. Averaging reduces variance in the objective.

### 5. Run Optimization

```rust
use aprender::automl::AutoTuner;

let result = AutoTuner::new(tpe)
    .early_stopping(5)  // Stop if stuck for 5 trials
    .maximize(&space, objective);

println!("Best K: {}", result.best_trial.get_usize(&KMeansParam::NClusters));
println!("Best silhouette: {:.4}", result.best_score);
```

## Sample Output

```
AutoML Clustering - TPE Optimization
=====================================

Generated 100 samples with 4 true clusters

Search Space: K ∈ [2, 10]
Objective: Maximize silhouette score

═══════════════════════════════════════════
 Trial │   K   │ Silhouette │   Status
═══════╪═══════╪════════════╪════════════
    1  │    9  │    0.460   │ moderate
    2  │    6  │    0.599   │ good
    3  │    5  │    0.707   │ good
    4  │   10  │    0.498   │ moderate
    5  │   10  │    0.498   │ moderate
    ...
═══════════════════════════════════════════

📊 Summary by K:
   K= 5: silhouette=0.707 (1 trials) ★ BEST
   K= 6: silhouette=0.599 (1 trials)
   K= 9: silhouette=0.460 (1 trials)
   K=10: silhouette=0.498 (5 trials)

🏆 TPE Optimization Results:
   Best K:          5
   Best silhouette: 0.7072
   True K:          4
   Trials run:      8
   Time elapsed:    0.10s

🔍 Final Model Verification:
   Silhouette score: 0.6910
   Inertia:          59.52
   Iterations:       2

📈 Interpretation:
   ✓ TPE found a close approximation (within ±1)
   ✅ Excellent cluster separation (silhouette > 0.5)
```

## Key Observations

1. **TPE found K=5** while true K=4. This is a close approximation—the silhouette metric sometimes favors slightly higher K values when clusters have some overlap.

2. **Early stopping triggered** at 8 trials (instead of 15). TPE identified that K=10 wasn't improving and stopped exploring.

3. **Excellent silhouette score** (0.707 > 0.5) indicates well-separated clusters regardless of the exact K.

4. **Fast optimization** (0.10s) compared to exhaustive search.

## Why TPE Over Grid Search?

| Aspect | Grid Search | TPE |
|--------|-------------|-----|
| Sample efficiency | Evaluates all combinations | Focuses on promising regions |
| Scaling | O(n^d) for d parameters | ~O(n) regardless of d |
| Informed decisions | None | Uses past results to guide search |
| Early stopping | Not built-in | Natural with callbacks |

For this 1D problem, grid search would work fine. TPE shines when:
- You have multiple hyperparameters
- Each evaluation is expensive
- You want to stop early if optimal is found

## Silhouette Score Interpretation

| Score | Interpretation |
|-------|----------------|
| > 0.5 | Strong cluster structure |
| 0.25 - 0.5 | Reasonable structure |
| < 0.25 | Weak or overlapping clusters |
| < 0 | Samples may be in wrong clusters |

## Best Practices

1. **Multiple seeds**: Average multiple K-Means runs to reduce variance
2. **Reasonable search range**: Don't search K > sqrt(n) typically
3. **Early stopping**: Use callbacks to avoid wasted computation
4. **Verify results**: Always examine final clusters qualitatively

## Related Topics

- [AutoML Theory](../ml-fundamentals/automl.md) - Full AutoML documentation
- [K-Means Clustering](./kmeans-clustering.md) - K-Means fundamentals
- [Iris Clustering](./iris-clustering.md) - Basic clustering example
