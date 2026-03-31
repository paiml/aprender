# AutoML Specification v1.0

## aprender Automated Machine Learning Module

**Status**: Draft
**Version**: 1.0.0
**Target Release**: v0.12.0

---

## 1. Executive Summary

This specification defines an AutoML module for aprender leveraging trueno's SIMD-accelerated tensor operations. The system automates hyperparameter optimization (HPO), model selection, and ensemble construction while maintaining aprender's zero-unsafe, pure-Rust philosophy.

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     AutoML Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Search  │→ │  Trial   │→ │  Eval    │→ │ Ensemble │    │
│  │  Space   │  │  Runner  │  │  Engine  │  │ Builder  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│       ↑              ↓             ↓             ↓          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Surrogate Model (GP/TPE)                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           trueno SIMD Backend                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 3. Core Components

### 3.1 Search Space Definition

```rust
/// Hyperparameter search space specification
pub enum HyperParam {
    /// Continuous parameter with bounds [low, high]
    Continuous { low: f32, high: f32, log_scale: bool },
    /// Integer parameter with bounds [low, high]
    Integer { low: i32, high: i32 },
    /// Categorical parameter with discrete choices
    Categorical { choices: Vec<String> },
    /// Conditional parameter (depends on parent)
    Conditional { parent: String, condition: Condition, param: Box<HyperParam> },
}

/// Search space for a model family
pub struct SearchSpace {
    pub params: HashMap<String, HyperParam>,
    pub constraints: Vec<Constraint>,
}
```

**Rationale**: Conditional hyperparameters are essential for hierarchical search spaces [1]. Log-scale sampling improves optimization for parameters spanning orders of magnitude [2].

### 3.2 Search Algorithms

#### 3.2.1 Random Search

```rust
pub struct RandomSearch {
    pub n_iter: usize,
    pub seed: u64,
}
```

Random search is surprisingly competitive with grid search and Bayesian methods for low-dimensional spaces, achieving equivalent results with 60x fewer trials in many cases [1].

#### 3.2.2 Bayesian Optimization with Gaussian Processes

```rust
pub struct BayesianOptimizer {
    /// Surrogate model (GP or TPE)
    pub surrogate: SurrogateModel,
    /// Acquisition function
    pub acquisition: AcquisitionFn,
    /// Number of initial random samples
    pub n_initial: usize,
    /// Total budget
    pub n_iter: usize,
}

pub enum AcquisitionFn {
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound { kappa: f32 },
    ThompsonSampling,
}
```

GP-based Bayesian optimization achieves state-of-the-art results on hyperparameter tuning, outperforming manual tuning and grid search [2]. Expected Improvement (EI) provides a principled exploration-exploitation tradeoff.

#### 3.2.3 Tree-structured Parzen Estimator (TPE)

```rust
pub struct TPEOptimizer {
    /// Quantile for splitting good/bad
    pub gamma: f32,  // default: 0.25
    /// Number of candidates to evaluate
    pub n_candidates: usize,
    /// Bandwidth selection
    pub bandwidth: BandwidthMethod,
}
```

TPE handles conditional and categorical hyperparameters more naturally than GP-based methods and scales better to high dimensions [3].

#### 3.2.4 Hyperband (Multi-Fidelity)

```rust
pub struct Hyperband {
    /// Maximum resources per configuration
    pub max_iter: usize,
    /// Downsampling rate
    pub eta: usize,  // default: 3
}
```

Hyperband provides theoretically-grounded early stopping, achieving up to 50x speedup over random search through adaptive resource allocation [4][5].

#### 3.2.5 BOHB (Bayesian Optimization + Hyperband)

```rust
pub struct BOHB {
    /// Hyperband parameters
    pub hyperband: Hyperband,
    /// TPE for configuration sampling
    pub tpe: TPEOptimizer,
    /// Minimum observations before using model
    pub min_points_in_model: usize,
}
```

BOHB combines the sample efficiency of Bayesian optimization with Hyperband's early stopping, achieving strong anytime performance [6].

### 3.3 Model Selection

```rust
pub struct ModelSelector {
    /// Candidate model families
    pub models: Vec<ModelFamily>,
    /// Meta-learning warm-start
    pub warm_start: Option<MetaFeatures>,
    /// Time budget per model
    pub time_budget: Duration,
}

pub enum ModelFamily {
    LinearRegression(SearchSpace),
    Ridge(SearchSpace),
    Lasso(SearchSpace),
    ElasticNet(SearchSpace),
    DecisionTree(SearchSpace),
    RandomForest(SearchSpace),
    GradientBoosting(SearchSpace),
    KNN(SearchSpace),
    SVM(SearchSpace),
    NaiveBayes(SearchSpace),
}
```

Combined Algorithm Selection and Hyperparameter optimization (CASH) treats model selection as a hyperparameter, enabling joint optimization [7][8].

### 3.4 Ensemble Construction

```rust
pub struct EnsembleBuilder {
    /// Selection method
    pub method: EnsembleMethod,
    /// Maximum ensemble size
    pub max_models: usize,
    /// Ensemble weighting
    pub weighting: WeightingStrategy,
}

pub enum EnsembleMethod {
    /// Greedy forward selection [8]
    GreedyEnsembleSelection,
    /// Stacking with meta-learner
    Stacking { meta_learner: Box<dyn Estimator> },
    /// Simple averaging
    Averaging,
}
```

Ensemble selection from libraries of models achieves better generalization than single-model selection, as demonstrated by Auto-sklearn [8].

## 4. API Design

### 4.1 High-Level API

```rust
use aprender::automl::{AutoML, TimeLimit};

let automl = AutoML::default()
    .time_limit(TimeLimit::Minutes(30))
    .metric(Metric::Accuracy)
    .cv_folds(5);

let result = automl.fit(&X_train, &y_train)?;
let predictions = result.predict(&X_test)?;

// Access best model and search history
println!("Best model: {:?}", result.best_model());
println!("Best score: {:.4}", result.best_score());
println!("Trials: {}", result.n_trials());
```

### 4.2 Fine-Grained Control

```rust
use aprender::automl::{SearchSpace, BayesianOptimizer};
use aprender::automl::params::GradientBoostingParam as P;

// Type-safe parameter keys (Poka-Yoke: prevents typos at compile time)
let space = SearchSpace::new()
    .add(P::NEstimators, 10..500)
    .add(P::MaxDepth, 2..20)
    .add(P::LearningRate, (1e-4..1.0).log_scale());

let optimizer = BayesianOptimizer::new()
    .acquisition(AcquisitionFn::ExpectedImprovement)
    .n_initial(10)
    .n_iter(100);

let result = optimizer.minimize(
    |params| cross_val_score(&model, params, &X, &y, 5),
    &space,
)?;
```

**Design Note (Poka-Yoke)**: Parameter keys use enums rather than strings to catch typos at compile time. This eliminates an entire class of runtime errors.

### 4.3 Callback System

```rust
pub trait Callback {
    fn on_trial_start(&mut self, trial: &Trial);
    fn on_trial_complete(&mut self, trial: &Trial, score: f32);
    fn should_stop(&self) -> bool;
}

// Built-in callbacks
automl.add_callback(EarlyStopping::new(patience: 20));
automl.add_callback(ProgressBar::new());
automl.add_callback(TensorBoard::new("./logs"));
```

## 5. trueno Integration

### 5.1 SIMD-Accelerated Surrogate Models

```rust
impl GaussianProcess {
    /// Fit GP using trueno's Cholesky decomposition
    pub fn fit(&mut self, X: &Matrix, y: &Vector) -> Result<()> {
        // K = kernel(X, X) + noise * I
        let K = self.kernel.compute(X, X);
        let K_noisy = K.add_diagonal(self.noise);

        // Cholesky factorization via trueno SIMD
        self.L = trueno::linalg::cholesky(&K_noisy)?;
        self.alpha = trueno::linalg::cho_solve(&self.L, y)?;
        Ok(())
    }

    /// Predict mean and variance
    pub fn predict(&self, X_new: &Matrix) -> (Vector, Vector) {
        let K_s = self.kernel.compute(&self.X_train, X_new);
        let mu = K_s.t().dot(&self.alpha);

        // Variance computation via trueno
        let v = trueno::linalg::triangular_solve(&self.L, &K_s);
        let var = self.kernel.diag(X_new) - v.col_norms_squared();

        (mu, var)
    }
}
```

### 5.2 Parallel Trial Execution

```rust
impl TrialRunner {
    /// Execute trials with SIMD-parallel cross-validation
    pub fn run_parallel(&self, configs: &[Config]) -> Vec<TrialResult> {
        configs.iter().map(|config| {
            // Each fold uses trueno's vectorized operations
            let scores = self.cv_folds.iter().map(|(train, val)| {
                let model = self.build_model(config);
                model.fit(&X[train], &y[train])?;
                model.score(&X[val], &y[val])
            }).collect();

            TrialResult {
                config: config.clone(),
                mean_score: trueno::stats::mean(&scores),
                std_score: trueno::stats::std(&scores),
            }
        }).collect()
    }
}
```

## 6. Default Search Spaces

### 6.1 Type-Safe Parameter Enums

```rust
/// Random Forest hyperparameters (compile-time typo prevention)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RandomForestParam {
    NEstimators,
    MaxDepth,
    MinSamplesSplit,
    MinSamplesLeaf,
    MaxFeatures,
    Bootstrap,
}

/// Gradient Boosting hyperparameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GradientBoostingParam {
    NEstimators,
    LearningRate,
    MaxDepth,
    Subsample,
    MinSamplesLeaf,
}
```

### 6.2 Random Forest

```rust
use RandomForestParam as RF;

pub fn random_forest_space() -> SearchSpace<RandomForestParam> {
    SearchSpace::new()
        .add(RF::NEstimators, 10..500)
        .add(RF::MaxDepth, 2..32)
        .add(RF::MinSamplesSplit, 2..20)
        .add(RF::MinSamplesLeaf, 1..20)
        .add(RF::MaxFeatures, ["sqrt", "log2", "0.5", "0.8", "1.0"])
        .add(RF::Bootstrap, [true, false])
}
```

### 6.3 Gradient Boosting

```rust
use GradientBoostingParam as GB;

pub fn gradient_boosting_space() -> SearchSpace<GradientBoostingParam> {
    SearchSpace::new()
        .add(GB::NEstimators, 50..1000)
        .add(GB::LearningRate, (1e-3..1.0).log_scale())
        .add(GB::MaxDepth, 1..15)
        .add(GB::Subsample, 0.5..1.0)
        .add(GB::MinSamplesLeaf, 1..100)
}
```

## 7. Performance Targets

| Operation | Target | trueno Acceleration |
|-----------|--------|---------------------|
| GP inference (1000 points) | <10ms | 8x via SIMD Cholesky |
| TPE sampling (100 candidates) | <1ms | 4x via vectorized KDE |
| CV fold evaluation | <100ms | 6x via batched predict |
| Ensemble prediction | <5ms | 10x via fused operations |

## 8. References (Peer-Reviewed Publications)

[1] **Bergstra, J., & Bengio, Y. (2012).** Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research*, 13(Feb), 281-305.
> Establishes that random search is more efficient than grid search, requiring 60x fewer trials to find comparable hyperparameter configurations. Foundational for our RandomSearch implementation.

[2] **Snoek, J., Larochelle, H., & Adams, R. P. (2012).** Practical Bayesian Optimization of Machine Learning Algorithms. *Advances in Neural Information Processing Systems*, 25, 2951-2959.
> Demonstrates GP-based Bayesian optimization outperforms manual tuning and grid search. Introduces the SPEARMINT system. Basis for our BayesianOptimizer with Expected Improvement acquisition.

[3] **Bergstra, J., Bardenet, R., Bengio, Y., & Kegl, B. (2011).** Algorithms for Hyper-Parameter Optimization. *Advances in Neural Information Processing Systems*, 24, 2546-2554.
> Introduces Tree-structured Parzen Estimator (TPE) for hyperparameter optimization. Shows TPE handles conditional/categorical parameters better than GP. Core reference for TPEOptimizer.

[4] **Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017).** Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. *Journal of Machine Learning Research*, 18(185), 1-52.
> Presents Hyperband algorithm combining random search with early stopping via successive halving. Achieves 50x speedup. Foundation for our Hyperband implementation.

[5] **Jamieson, K., & Talwalkar, A. (2016).** Non-stochastic Best Arm Identification and Hyperparameter Optimization. *Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 240-248.
> Theoretical foundations for adaptive resource allocation in hyperparameter optimization. Proves optimality guarantees for successive halving. Supports our multi-fidelity approach.

[6] **Falkner, S., Klein, A., & Hutter, F. (2018).** BOHB: Robust and Efficient Hyperparameter Optimization at Scale. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, 1437-1446.
> Combines Bayesian optimization with Hyperband for strong anytime performance. State-of-the-art on HPO benchmarks. Direct basis for our BOHB implementation.

[7] **Thornton, C., Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2013).** Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms. *Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 847-855.
> First to formalize Combined Algorithm Selection and Hyperparameter optimization (CASH). Demonstrates joint model+hyperparameter search outperforms separate optimization. Informs our ModelSelector design.

[8] **Feurer, M., Klein, A., Eggensperger, K., Springenberg, J., Blum, M., & Hutter, F. (2015).** Efficient and Robust Automated Machine Learning. *Advances in Neural Information Processing Systems*, 28, 2962-2970.
> Introduces Auto-sklearn with meta-learning warm-starting and ensemble selection. Demonstrates ensembles of models found during search outperform best single model. Basis for our EnsembleBuilder.

[9] **Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2011).** Sequential Model-based Algorithm Configuration. *Proceedings of the 5th International Conference on Learning and Intelligent Optimization (LION)*, 507-523.
> Introduces SMAC using random forests as surrogate models. Handles categorical/conditional parameters well. Alternative surrogate model for our BayesianOptimizer.

[10] **Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016).** Taking the Human Out of the Loop: A Review of Bayesian Optimization. *Proceedings of the IEEE*, 104(1), 148-175.
> Comprehensive survey of Bayesian optimization methods, acquisition functions, and applications. Provides theoretical grounding for acquisition function selection. Reference for our AcquisitionFn implementations.

## 9. Implementation Roadmap

1. **Phase 1**: RandomSearch, GridSearch, basic SearchSpace (v0.11.0)
2. **Phase 2**: TPE optimizer, Hyperband early stopping (v0.11.1)
3. **Phase 3**: GP-based Bayesian optimization with trueno (v0.12.0)
4. **Phase 4**: BOHB, ensemble selection, meta-learning (v0.13.0)

## 10. Success Criteria

- Match Auto-sklearn performance on OpenML-CC18 benchmark suite
- <5% overhead vs manual HPO for simple search spaces
- Support all aprender estimators via Estimator trait
- Full documentation with examples for each optimizer
