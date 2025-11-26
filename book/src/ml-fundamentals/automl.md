# AutoML: Automated Machine Learning

Aprender's AutoML module provides type-safe hyperparameter optimization with multiple search strategies, including the state-of-the-art Tree-structured Parzen Estimator (TPE).

## Overview

AutoML automates the tedious process of hyperparameter tuning:

1. **Define search space** with type-safe parameter enums
2. **Choose strategy** (Random, Grid, or TPE)
3. **Run optimization** with callbacks for early stopping and time limits
4. **Get best configuration** automatically

## Key Features

- **Type Safety (Poka-Yoke)**: Parameter keys are enums, not strings—typos caught at compile time
- **Multiple Strategies**: RandomSearch, GridSearch, TPE
- **Callbacks**: TimeBudget, EarlyStopping, ProgressCallback
- **Extensible**: Custom parameter enums for any model family

## Quick Start

```rust
use aprender::automl::{AutoTuner, TPE, SearchSpace};
use aprender::automl::params::RandomForestParam as RF;

// Define type-safe search space
let space = SearchSpace::new()
    .add(RF::NEstimators, 10..500)
    .add(RF::MaxDepth, 2..20);

// Use TPE optimizer with early stopping
let result = AutoTuner::new(TPE::new(100))
    .time_limit_secs(60)
    .early_stopping(20)
    .maximize(&space, |trial| {
        let n = trial.get_usize(&RF::NEstimators).unwrap_or(100);
        let d = trial.get_usize(&RF::MaxDepth).unwrap_or(5);
        evaluate_model(n, d)  // Your objective function
    });

println!("Best: {:?}", result.best_trial);
```

## Type-Safe Parameter Enums

### The Problem with String Keys

Traditional AutoML libraries use string keys for parameters:

```python
# Optuna/scikit-optimize style (error-prone)
space = {
    "n_estimators": (10, 500),
    "max_detph": (2, 20),  # TYPO! Silent bug
}
```

### Aprender's Solution: Poka-Yoke

Aprender uses typed enums that catch typos at compile time:

```rust
use aprender::automl::params::RandomForestParam as RF;

let space = SearchSpace::new()
    .add(RF::NEstimators, 10..500)
    .add(RF::MaxDetph, 2..20);  // Compile error! Typo caught
//       ^^^^^^^^^^^^ Unknown variant
```

### Built-in Parameter Enums

```rust
// Random Forest
use aprender::automl::params::RandomForestParam;
// NEstimators, MaxDepth, MinSamplesLeaf, MaxFeatures, Bootstrap

// Gradient Boosting
use aprender::automl::params::GradientBoostingParam;
// NEstimators, LearningRate, MaxDepth, Subsample

// K-Nearest Neighbors
use aprender::automl::params::KNNParam;
// NNeighbors, Weights, P

// Linear Models
use aprender::automl::params::LinearParam;
// Alpha, L1Ratio, MaxIter, Tol

// Decision Trees
use aprender::automl::params::DecisionTreeParam;
// MaxDepth, MinSamplesLeaf, MinSamplesSplit

// K-Means
use aprender::automl::params::KMeansParam;
// NClusters, MaxIter, NInit
```

### Custom Parameter Enums

```rust
use aprender::automl::params::ParamKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MyModelParam {
    LearningRate,
    HiddenLayers,
    Dropout,
}

impl ParamKey for MyModelParam {
    fn name(&self) -> &'static str {
        match self {
            Self::LearningRate => "learning_rate",
            Self::HiddenLayers => "hidden_layers",
            Self::Dropout => "dropout",
        }
    }
}

impl std::fmt::Display for MyModelParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
```

## Search Space Definition

### Integer Parameters

```rust
let space = SearchSpace::new()
    .add(RF::NEstimators, 10..500)   // [10, 499]
    .add(RF::MaxDepth, 2..20);       // [2, 19]
```

### Continuous Parameters

```rust
let space = SearchSpace::new()
    .add_continuous(Param::LearningRate, 0.001, 0.1)
    .add_log_scale(Param::Alpha, LogScale { low: 1e-4, high: 1.0 });
```

### Categorical Parameters

```rust
let space = SearchSpace::new()
    .add_categorical(RF::MaxFeatures, ["sqrt", "log2", "0.5"])
    .add_bool(RF::Bootstrap, [true, false]);
```

## Search Strategies

### RandomSearch

Best for: Initial exploration, large search spaces

```rust
use aprender::automl::{RandomSearch, SearchStrategy};

let mut search = RandomSearch::new(100)  // 100 trials
    .with_seed(42);                       // Reproducible

let trials = search.suggest(&space, 10);  // Get 10 suggestions
```

**Why Random Search?**

Bergstra & Bengio (2012) showed random search achieves equivalent results to grid search with 60x fewer trials for many problems.

### GridSearch

Best for: Small, discrete search spaces

```rust
use aprender::automl::GridSearch;

let mut search = GridSearch::new(5);  // 5 points per continuous param
let trials = search.suggest(&space, 100);
```

### TPE (Tree-structured Parzen Estimator)

Best for: >10 trials, expensive objective functions

```rust
use aprender::automl::TPE;

let mut tpe = TPE::new(100)
    .with_seed(42)
    .with_startup_trials(10)  // Random before model
    .with_gamma(0.25);        // Top 25% as "good"
```

**How TPE Works:**

1. **Split observations**: Separate into "good" (top γ) and "bad" based on objective values
2. **Fit KDEs**: Build Kernel Density Estimators for good (l) and bad (g) distributions
3. **Sample candidates**: Generate multiple candidates
4. **Select by EI**: Choose candidate maximizing l(x)/g(x) (Expected Improvement)

**TPE Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.25 | Quantile for good/bad split |
| `n_candidates` | 24 | Candidates per iteration |
| `n_startup_trials` | 10 | Random trials before model |

## AutoTuner with Callbacks

### Basic Usage

```rust
use aprender::automl::{AutoTuner, TPE, SearchSpace};

let result = AutoTuner::new(TPE::new(100))
    .maximize(&space, |trial| {
        // Your objective function
        evaluate(trial)
    });

println!("Best score: {}", result.best_score);
println!("Best params: {:?}", result.best_trial);
```

### Time Budget

```rust
let result = AutoTuner::new(TPE::new(1000))
    .time_limit_secs(60)   // Stop after 60 seconds
    .maximize(&space, objective);
```

### Early Stopping

```rust
let result = AutoTuner::new(TPE::new(1000))
    .early_stopping(20)    // Stop if no improvement for 20 trials
    .maximize(&space, objective);
```

### Verbose Progress

```rust
let result = AutoTuner::new(TPE::new(100))
    .verbose()             // Print trial results
    .maximize(&space, objective);

// Output:
// Trial   1: score=0.8234 params={n_estimators=142, max_depth=7}
// Trial   2: score=0.8456 params={n_estimators=287, max_depth=12}
// ...
```

### Combined Callbacks

```rust
let result = AutoTuner::new(TPE::new(500))
    .time_limit_secs(300)    // 5 minute budget
    .early_stopping(30)      // Stop if stuck
    .verbose()               // Show progress
    .maximize(&space, objective);
```

### Custom Callbacks

```rust
use aprender::automl::{Callback, TrialResult};

struct MyCallback {
    best_so_far: f64,
}

impl<P: ParamKey> Callback<P> for MyCallback {
    fn on_trial_end(&mut self, trial_num: usize, result: &TrialResult<P>) {
        if result.score > self.best_so_far {
            self.best_so_far = result.score;
            println!("New best at trial {}: {}", trial_num, result.score);
        }
    }

    fn should_stop(&self) -> bool {
        self.best_so_far > 0.99  // Stop if reached target
    }
}

let result = AutoTuner::new(TPE::new(100))
    .callback(MyCallback { best_so_far: 0.0 })
    .maximize(&space, objective);
```

## TuneResult Structure

```rust
pub struct TuneResult<P: ParamKey> {
    pub best_trial: Trial<P>,       // Best configuration
    pub best_score: f64,            // Best objective value
    pub history: Vec<TrialResult<P>>, // All trial results
    pub elapsed: Duration,          // Total time
    pub n_trials: usize,            // Trials completed
}
```

## Trial Accessors

```rust
let trial: Trial<RF> = /* ... */;

// Type-safe accessors
let n: Option<usize> = trial.get_usize(&RF::NEstimators);
let d: Option<i64> = trial.get_i64(&RF::MaxDepth);
let lr: Option<f64> = trial.get_f64(&Param::LearningRate);
let bootstrap: Option<bool> = trial.get_bool(&RF::Bootstrap);
```

## Real-World Example: aprender-shell

The `aprender-shell tune` command uses TPE to optimize n-gram size:

```rust
fn cmd_tune(history_path: Option<PathBuf>, trials: usize, ratio: f32) {
    use aprender::automl::{AutoTuner, SearchSpace, TPE};
    use aprender::automl::params::ParamKey;

    // Define custom parameter
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum ShellParam { NGram }

    impl ParamKey for ShellParam {
        fn name(&self) -> &'static str { "ngram" }
    }

    let space: SearchSpace<ShellParam> = SearchSpace::new()
        .add(ShellParam::NGram, 2..6);  // n-gram sizes 2-5

    let tpe = TPE::new(trials)
        .with_seed(42)
        .with_startup_trials(2)
        .with_gamma(0.25);

    let result = AutoTuner::new(tpe)
        .early_stopping(4)
        .maximize(&space, |trial| {
            let ngram = trial.get_usize(&ShellParam::NGram).unwrap_or(3);

            // 3-fold cross-validation
            let mut scores = Vec::new();
            for fold in 0..3 {
                let score = validate_model(&commands, ngram, ratio, fold);
                scores.push(score);
            }
            scores.iter().sum::<f64>() / 3.0
        });

    println!("Best n-gram: {}", result.best_trial.get_usize(&ShellParam::NGram).unwrap());
    println!("Best score: {:.3}", result.best_score);
}
```

**Output:**

```
🎯 aprender-shell: AutoML Hyperparameter Tuning (TPE)

📂 History file: /home/user/.zsh_history
📊 Total commands: 21780
🔬 TPE trials: 8

══════════════════════════════════════════════════
 Trial │ N-gram │   Hit@5   │    MRR    │  Score
═══════╪════════╪═══════════╪═══════════╪═════════
    1  │    4   │   26.2%   │  0.182   │  0.282
    2  │    5   │   26.8%   │  0.186   │  0.257
    3  │    2   │   26.2%   │  0.181   │  0.280
══════════════════════════════════════════════════

🏆 Best Configuration (TPE):
   N-gram size: 4
   Score:       0.282
   Trials run:  5
   Time:        51.3s
```

## Best Practices

### 1. Start with Random Search

```rust
// Quick exploration
let result = AutoTuner::new(RandomSearch::new(20))
    .maximize(&space, objective);

// Then refine with TPE
let result = AutoTuner::new(TPE::new(100))
    .maximize(&refined_space, objective);
```

### 2. Use Log Scale for Learning Rates

```rust
let space = SearchSpace::new()
    .add_log_scale(Param::LearningRate, LogScale { low: 1e-5, high: 1e-1 });
```

### 3. Set Reasonable Time Budgets

```rust
// For expensive evaluations
let result = AutoTuner::new(TPE::new(1000))
    .time_limit_mins(30)
    .maximize(&space, expensive_objective);
```

### 4. Combine Early Stopping with Time Budget

```rust
let result = AutoTuner::new(TPE::new(500))
    .time_limit_secs(600)   // Max 10 minutes
    .early_stopping(50)     // Stop if stuck for 50 trials
    .maximize(&space, objective);
```

## Algorithm Comparison

| Strategy | Best For | Sample Efficiency | Overhead |
|----------|----------|-------------------|----------|
| RandomSearch | Large spaces, quick exploration | Low | Minimal |
| GridSearch | Small, discrete spaces | Medium | Minimal |
| TPE | Expensive objectives, >10 trials | High | Low |

## References

1. Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). **Algorithms for Hyper-Parameter Optimization.** NeurIPS.

2. Bergstra, J., & Bengio, Y. (2012). **Random Search for Hyper-Parameter Optimization.** JMLR, 13, 281-305.

## Related Topics

- [Grid Search Hyperparameter Tuning](../examples/grid-search-tuning.md) - Manual grid search
- [Cross-Validation](./cross-validation.md) - CV fundamentals
- [Random Forest](../examples/random-forest.md) - Model to tune
