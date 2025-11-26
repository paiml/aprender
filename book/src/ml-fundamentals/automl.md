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

## Synthetic Data Augmentation

Aprender's `synthetic` module enables automatic data augmentation with quality control and diversity monitoring—particularly powerful for low-resource domains like shell autocomplete.

### The Problem: Limited Training Data

Many ML tasks suffer from insufficient training data:
- Shell autocomplete: Limited user history
- Code translation: Sparse parallel corpora
- Domain-specific NLP: Rare terminology

### The Solution: Quality-Controlled Synthetic Data

```rust
use aprender::synthetic::{SyntheticConfig, DiversityMonitor, DiversityScore};

// Configure augmentation with quality controls
let config = SyntheticConfig::default()
    .with_augmentation_ratio(1.0)    // 100% more data
    .with_quality_threshold(0.7)     // 70% minimum quality
    .with_diversity_weight(0.3);     // Balance quality vs diversity

// Monitor for mode collapse
let mut monitor = DiversityMonitor::new(10)
    .with_collapse_threshold(0.1);
```

### SyntheticConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `augmentation_ratio` | 0.5 | Synthetic/original ratio (1.0 = double data) |
| `quality_threshold` | 0.7 | Minimum score for acceptance [0.0, 1.0] |
| `diversity_weight` | 0.3 | Balance: 0=quality only, 1=diversity only |
| `max_attempts` | 10 | Retries per sample before giving up |

### Generation Strategies

```rust
use aprender::synthetic::GenerationStrategy;

// Available strategies
GenerationStrategy::Template       // Slot-filling templates
GenerationStrategy::EDA            // Easy Data Augmentation
GenerationStrategy::BackTranslation // Via intermediate representation
GenerationStrategy::MixUp          // Embedding interpolation
GenerationStrategy::GrammarBased   // Formal grammar rules
GenerationStrategy::SelfTraining   // Pseudo-labels
GenerationStrategy::WeakSupervision // Labeling functions (Snorkel)
```

### Real-World Example: aprender-shell augment

The `aprender-shell augment` command demonstrates synthetic data power:

```bash
aprender-shell augment -a 1.0 -q 0.6 --monitor-diversity
```

**Output:**

```
🧬 aprender-shell: Data Augmentation (with aprender synthetic)

📂 History file: /home/user/.zsh_history
📊 Real commands: 21789
⚙️  Augmentation ratio: 1.0x
⚙️  Quality threshold:  60.0%
🎯 Target synthetic:   21789 commands
🔢 Known n-grams: 39180

🧪 Generating synthetic commands... done!

📈 Coverage Report:
   Generated:          21789
   Quality filtered:   21430 (rejected 359)
   Known n-grams:      39180
   Total n-grams:      26616
   New n-grams added:  23329
   Coverage gain:      87.7%

📊 Diversity Metrics:
   Mean diversity:     1.000
   ✓  Diversity is healthy

📊 Model Statistics:
   Original commands:   21789
   Synthetic commands:  21430
   Total training:      43219
   Unique n-grams:      65764
   Vocabulary size:     37531
```

### Before vs After Comparison

```
═══════════════════════════════════════════════════════════════
                    📈 IMPROVEMENT SUMMARY
═══════════════════════════════════════════════════════════════

                      BASELINE    AUGMENTED    GAIN
───────────────────────────────────────────────────────────────
  Commands:           21,789      43,219       +98%
  Unique n-grams:     40,852      65,764       +61%
  Vocabulary size:    16,102      37,531       +133%
  Model size:         2,016 KB    3,017 KB     +50%
  Coverage gain:        --        87.7%         ✓
  Diversity:            --        1.000        Healthy
═══════════════════════════════════════════════════════════════
```

### New Capabilities from Synthetic Data

Commands the model never saw in history but now suggests:

```
kubectl suggestions (DevOps):
kubectl exec        0.050
kubectl config      0.050
kubectl delete      0.050

aws suggestions (Cloud):
aws ec2             0.096
aws lambda          0.076
aws iam             0.065

rustup suggestions (Rust):
rustup toolchain    0.107
rustup override     0.107
rustup doc          0.107
```

### DiversityMonitor: Detecting Mode Collapse

```rust
use aprender::synthetic::{DiversityMonitor, DiversityScore};

let mut monitor = DiversityMonitor::new(10)
    .with_collapse_threshold(0.1);

// Record diversity scores during generation
for sample in generated_samples {
    let score = DiversityScore::new(
        mean_distance,   // Pairwise distance
        min_distance,    // Closest pair
        coverage,        // Space coverage
    );
    monitor.record(score);
}

// Check for problems
if monitor.is_collapsing() {
    println!("⚠️  Mode collapse detected!");
}
if monitor.is_trending_down() {
    println!("⚠️  Diversity trending downward");
}

println!("Mean diversity: {:.3}", monitor.mean_diversity());
```

### QualityDegradationDetector

Monitors whether synthetic data is helping or hurting:

```rust
use aprender::synthetic::QualityDegradationDetector;

// Baseline: score without synthetic data
let mut detector = QualityDegradationDetector::new(0.85, 10)
    .with_min_improvement(0.02);

// Record scores from training with synthetic data
detector.record(0.87);  // Better!
detector.record(0.86);
detector.record(0.82);  // Getting worse...

if detector.should_disable_synthetic() {
    println!("Synthetic data is hurting performance");
}

let summary = detector.summary();
println!("Improvement: {:.1}%", summary.improvement * 100.0);
```

### Type-Safe Synthetic Parameters

```rust
use aprender::synthetic::SyntheticParam;
use aprender::automl::SearchSpace;

// Add synthetic params to AutoML search space
let space = SearchSpace::new()
    // Model hyperparameters
    .add(ModelParam::HiddenSize, 64..512)
    // Synthetic data hyperparameters (jointly optimized!)
    .add(SyntheticParam::AugmentationRatio, 0.0..2.0)
    .add(SyntheticParam::QualityThreshold, 0.5..0.95);
```

### Key Benefits

1. **Quality Filtering**: Rejected 359 low-quality commands (1.6%)
2. **Diversity Monitoring**: Confirmed no mode collapse
3. **Coverage Gain**: 87.7% of synthetic data introduced new n-grams
4. **Vocabulary Expansion**: +133% vocabulary size
5. **Joint Optimization**: Augmentation params tuned alongside model

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

## Running the Example

```bash
cargo run --example automl_clustering
```

**Sample Output:**

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
    ...
═══════════════════════════════════════════

🏆 TPE Optimization Results:
   Best K:          5
   Best silhouette: 0.7072
   True K:          4
   Trials run:      8

📈 Interpretation:
   ✓ TPE found a close approximation (within ±1)
   ✅ Excellent cluster separation (silhouette > 0.5)
```

## Related Topics

- [Case Study: AutoML Clustering](../examples/automl-clustering.md) - Full example
- [Grid Search Hyperparameter Tuning](../examples/grid-search-tuning.md) - Manual grid search
- [Cross-Validation](./cross-validation.md) - CV fundamentals
- [Random Forest](../examples/random-forest.md) - Model to tune
