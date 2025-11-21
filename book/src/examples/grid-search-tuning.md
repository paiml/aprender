# Grid Search Hyperparameter Tuning

This example demonstrates grid search for finding optimal regularization hyperparameters using cross-validation with Ridge, Lasso, and ElasticNet regression.

## Overview

Grid search is a systematic way to find the best hyperparameters by:
1. Defining a grid of candidate values
2. Evaluating each combination using cross-validation
3. Selecting parameters that maximize CV score
4. Retraining the final model with optimal parameters

## Running the Example

```bash
cargo run --example grid_search_tuning
```

## Key Concepts

### Why Grid Search?

**Problem**: Default hyperparameters rarely optimal for your specific dataset

**Solution**: Systematically search parameter space to find best values

**Benefits**:
- Automated hyperparameter optimization
- Cross-validation prevents overfitting
- Reproducible model selection
- Better generalization performance

### Grid Search Process

1. **Define parameter grid**: Range of values to try
2. **K-Fold CV**: Split training data into K folds
3. **Evaluate**: Train model on K-1 folds, validate on remaining fold
4. **Average scores**: Mean performance across all K folds
5. **Select best**: Parameters with highest CV score
6. **Final model**: Retrain on all training data with best parameters
7. **Test**: Evaluate on held-out test set

## Examples Demonstrated

### Example 1: Ridge Regression Alpha Tuning

Shows grid search for Ridge regression regularization strength (alpha):

```
Alpha Grid: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

Cross-Validation Scores:
  α=0.001  → R²=0.9510
  α=0.010  → R²=0.9510
  α=0.100  → R²=0.9510  ← Best
  α=1.000  → R²=0.9508
  α=10.000 → R²=0.9428
  α=100.000→ R²=0.8920

Best Parameters: α=0.100, CV Score=0.9510
Test Performance: R²=0.9626
```

**Observation**: Performance degrades with very large alpha (underf itting).

### Example 2: Lasso Regression Alpha Tuning

Demonstrates grid search for Lasso with feature selection:

```
Alpha Grid: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

Best Parameters: α=1.0000
Test Performance: R²=0.9628
Non-zero coefficients: 5/5 (sparse!)
```

**Key Feature**: Lasso performs automatic feature selection by driving some coefficients to exactly zero.

**Alpha guidelines**:
- Too small: Overfitting (no regularization)
- Optimal: Balance between fit and complexity
- Too large: Underfitting (excessive regularization)

### Example 3: ElasticNet with L1 Ratio Tuning

Shows 2D grid search over both `alpha` and `l1_ratio`:

```
Searching over:
  α: [0.001, 0.01, 0.1, 1.0, 10.0]
  l1_ratio: [0.25, 0.5, 0.75]

Best Parameters:
  α=1.000, l1_ratio=0.75
  CV Score: 0.9511
```

**l1_ratio Parameter**:
- `0.0`: Pure Ridge (L2 only)
- `0.5`: Equal mix of Lasso and Ridge
- `1.0`: Pure Lasso (L1 only)

**When to use ElasticNet**:
- Many correlated features (Ridge component)
- Want feature selection (Lasso component)
- Best of both regularization types

### Example 4: Visualizing Alpha vs Score

Compares Ridge and Lasso performance curves:

```
     Alpha      Ridge R²      Lasso R²
----------------------------------------
    0.0001        0.9510        0.9510
    0.0010        0.9510        0.9510
    0.0100        0.9510        0.9510
    0.1000        0.9510        0.9510
    1.0000        0.9508        0.9511
   10.0000        0.9428        0.9480
  100.0000        0.8920        0.8998
```

**Observations**:
- **Plateau region**: Performance stable across small alphas
- **Ridge**: Gradual degradation with large alpha
- **Lasso**: Sharper drop after optimal point
- **Both**: Performance collapses with excessive regularization

### Example 5: Default vs Optimized Comparison

Demonstrates value of hyperparameter tuning:

```
Ridge Regression Comparison:

Default (α=1.0):
  Test R²: 0.9628

Grid Search Optimized (α=0.100):
  CV R²:   0.9510
  Test R²: 0.9626

→ Improvement or similar performance
```

**Interpretation**:
- When default is good: Data well-suited to default parameters
- When improvement significant: Dataset-specific tuning helps
- Always worth checking: Small cost, potential large benefit

## Implementation Details

### Using grid_search_alpha()

```rust
use aprender::model_selection::{grid_search_alpha, KFold};

// Define parameter grid
let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0];

// Setup cross-validation
let kfold = KFold::new(5).with_random_state(42);

// Run grid search
let result = grid_search_alpha(
    "ridge",        // Model type
    &alphas,        // Parameter grid
    &x_train,       // Training features
    &y_train,       // Training targets
    &kfold,         // CV strategy
    None,           // l1_ratio (ElasticNet only)
).unwrap();

// Get best parameters
println!("Best alpha: {}", result.best_alpha);
println!("Best CV score: {}", result.best_score);

// Train final model
let mut model = Ridge::new(result.best_alpha);
model.fit(&x_train, &y_train).unwrap();
```

### GridSearchResult Structure

```rust
pub struct GridSearchResult {
    pub best_alpha: f32,       // Optimal alpha value
    pub best_score: f32,       // Best CV score
    pub alphas: Vec<f32>,      // All alphas tried
    pub scores: Vec<f32>,      // Corresponding scores
}
```

**Methods**:
- `best_index()`: Index of best alpha in grid

## Best Practices

### 1. Define Appropriate Grid

```rust
// ✅ Good: Log-scale grid
let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0];

// ❌ Bad: Linear grid missing optimal region
let alphas = vec![1.0, 2.0, 3.0, 4.0, 5.0];
```

**Guideline**: Use log-scale for regularization parameters.

### 2. Sufficient K-Folds

```rust
// ✅ Good: 5-10 folds typical
let kfold = KFold::new(5).with_random_state(42);

// ❌ Bad: Too few folds (unreliable estimates)
let kfold = KFold::new(2);
```

### 3. Evaluate on Test Set

```rust
// ✅ Correct workflow
let (x_train, x_test, y_train, y_test) = train_test_split(...);
let result = grid_search_alpha(..., &x_train, &y_train, ...);
let mut model = Ridge::new(result.best_alpha);
model.fit(&x_train, &y_train).unwrap();
let test_score = model.score(&x_test, &y_test); // Final evaluation

// ❌ Incorrect: Using CV score as final metric
println!("Final performance: {}", result.best_score); // Wrong!
```

### 4. Use Random State for Reproducibility

```rust
let kfold = KFold::new(5).with_random_state(42);
// Same results every run
```

## Choosing Alpha Ranges

### Ridge Regression
- **Start**: `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`
- **Refine**: Zoom in on best region
- **Typical optimal**: 0.1 - 10.0

### Lasso Regression
- **Start**: `[0.0001, 0.001, 0.01, 0.1, 1.0]`
- **Note**: Usually needs smaller alphas than Ridge
- **Typical optimal**: 0.001 - 1.0

### ElasticNet
- **Alpha**: Same as Ridge/Lasso
- **L1 ratio**: `[0.1, 0.3, 0.5, 0.7, 0.9]` or `[0.25, 0.5, 0.75]`
- **Tip**: Start with 3-5 l1_ratio values

## Common Pitfalls

1. **Fitting grid search on all data**: Always split train/test first
2. **Too fine grid**: Computationally expensive, minimal benefit
3. **Ignoring CV variance**: High variance suggests unstable model
4. **Overfitting to CV**: Test set still needed for final validation
5. **Wrong scale**: Linear grid misses optimal regions

## Computational Cost

**Formula**: `cost = n_alphas × n_folds × cost_per_fit`

**Example**:
- 6 alphas
- 5 folds
- Total fits: 6 × 5 = 30

**Optimization**:
- Start with coarse grid
- Refine around best region
- Use fewer folds for very large datasets

## Related Examples

- [Cross-Validation](./cross-validation.md) - K-Fold CV fundamentals
- [Regularized Regression](./regularized-regression.md) - Ridge, Lasso, ElasticNet
- [Linear Regression](./linear-regression.md) - Baseline model

## Key Takeaways

1. **Grid search automates** hyperparameter optimization
2. **Cross-validation** provides unbiased performance estimates
3. **Log-scale grids** work best for regularization parameters
4. **Ridge degrades gradually**, Lasso more sensitive to alpha
5. **ElasticNet** offers 2D tuning flexibility
6. **Always validate** final model on held-out test set
7. **Reproducibility**: Use random_state for consistent results
8. **Computational cost** scales with grid size and K-folds
