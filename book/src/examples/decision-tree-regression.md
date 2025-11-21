# Decision Tree Regression - Housing Price Prediction

**Status**: ✅ Complete (Verified with 16+ tests)

This case study demonstrates decision tree regression for predicting continuous values (housing prices) using the CART algorithm with Mean Squared Error criterion.

**What You'll Learn**:
- When to use decision trees for regression vs linear models
- How MSE splitting criterion works
- Effect of max_depth on overfitting
- Hyperparameter tuning (min_samples_split, min_samples_leaf)
- Handling non-linear relationships

**Prerequisites**: Basic understanding of regression metrics (R², MSE)

---

## Problem Statement

**Task**: Predict house prices (continuous values) from features like square footage, bedrooms, and age.

**Why Decision Tree Regression?**
- **Non-linear relationships**: Price doesn't scale linearly with size
- **Feature interactions**: Large house + old → different than small house + old
- **Interpretability**: Real estate agents can explain "rules"
- **No feature scaling**: Use raw sqft, years, etc.

**When NOT to use**:
- Linear relationships → Use LinearRegression (simpler, better generalization)
- Need smooth predictions → Trees predict step functions
- Extrapolation beyond training range → Trees can't extrapolate

---

## Dataset

### Simulated Housing Data

```rust
// Features: [sqft, bedrooms, bathrooms, age]
// Target: price (in thousands)
let x_train = Matrix::from_vec(20, 4, vec![
    // Small houses
    1000.0, 2.0, 1.0, 50.0,  // $140k
    1100.0, 2.0, 1.0, 45.0,  // $145k
    1200.0, 2.0, 1.0, 40.0,  // $150k
    1300.0, 2.0, 1.5, 35.0,  // $160k
    // Medium houses
    1500.0, 3.0, 2.0, 25.0,  // $250k
    1600.0, 3.0, 2.0, 20.0,  // $265k
    // ... (more samples)
    // Luxury houses (exponential price increase)
    4000.0, 7.0, 5.0, 0.5,   // $1100k
    4500.0, 8.0, 6.0, 0.5,   // $1350k
]).unwrap();

let y_train = Vector::from_slice(&[
    140.0, 145.0, 150.0, 160.0,  // Small
    250.0, 265.0, 280.0, 295.0,  // Medium
    360.0, 410.0, 480.0, 550.0,  // Large
    650.0, 720.0, 800.0, 920.0,  // Very large
    1100.0, 1350.0, 1600.0, 1950.0,  // Luxury
]);
```

**Data Characteristics**:
- 20 training samples, 4 features
- Price increases non-linearly with size
- Age discount effect
- Multiple price tiers

---

## Implementation

### Step 1: Train Basic Regression Tree

```rust
use aprender::prelude::*;

// Create and configure tree
let mut tree = DecisionTreeRegressor::new()
    .with_max_depth(5);

// Fit to training data
tree.fit(&x_train, &y_train).unwrap();

// Predict on test data
let x_test = Matrix::from_vec(1, 4, vec![
    1900.0, 4.0, 2.0, 12.0  // Medium-large house
]).unwrap();

let predicted_price = tree.predict(&x_test);
println!("Predicted: ${:.0}k", predicted_price.as_slice()[0]);
// Output: Predicted: $295k

// Evaluate with R² score
let r2 = tree.score(&x_train, &y_train);
println!("R² Score: {:.4}", r2);
// Output: R² Score: 1.0000 (perfect on training data)
```

**Key API Methods**:
- `new()`: Create tree with default parameters
- `with_max_depth(depth)`: Limit tree depth (prevent overfitting)
- `fit(&x, &y)`: Train tree on data (MSE criterion)
- `predict(&x)`: Predict continuous values
- `score(&x, &y)`: Compute R² score

**Test Reference**: `src/tree/mod.rs::test_regression_tree_fit_simple_linear`

---

### Step 2: Compare with Linear Regression

Decision trees excel at non-linear patterns. Let's compare:

```rust
// Train both models
let mut tree = DecisionTreeRegressor::new().with_max_depth(5);
let mut linear = LinearRegression::new();

tree.fit(&x_train, &y_train).unwrap();
linear.fit(&x_train, &y_train).unwrap();

// Compare R² scores
let tree_r2 = tree.score(&x_train, &y_train);
let linear_r2 = linear.score(&x_train, &y_train);

println!("Decision Tree R²: {:.4}", tree_r2);   // 1.0000
println!("Linear Regression R²: {:.4}", linear_r2); // 0.9844
println!("Tree advantage: {:.4}", tree_r2 - linear_r2); // 0.0156
```

**Why Tree Performs Better**:
- Captures non-linear price tiers (small/medium/large/luxury)
- Learns feature interactions (size × age)
- No assumption of linear relationship

**When Linear Wins**:
- Truly linear relationships
- Small datasets (better generalization)
- Need smooth predictions

**Test Reference**: `src/tree/mod.rs::test_regression_tree_vs_linear`

---

### Step 3: Understanding MSE Splitting

**How it works**:
1. For each feature and threshold, compute MSE for left and right children
2. Choose split that maximizes variance reduction
3. Leaf nodes predict mean of training samples

**Example Split Decision**:
```text
Parent node: [140, 145, 250, 265, 1100, 1350]
Mean = 541.67, MSE = 184,444

Candidate split: sqft ≤ 1500
  Left:  [140, 145]  → Mean = 142.5, MSE = 6.25
  Right: [250, 265, 1100, 1350] → Mean = 741.25, MSE = 234,756

Weighted MSE = (2/6)*6.25 + (4/6)*234,756 = 156,506
Variance Reduction = 184,444 - 156,506 = 27,938 ✅ Good split!
```

**Pure Node Example**:
```text
Node: [250, 250, 250]
Mean = 250, MSE = 0 → Stop splitting (pure)
```

**Test Reference**: `src/tree/mod.rs::test_regression_tree_constant_target`

---

## Hyperparameter Tuning

### max_depth: Controlling Complexity

```rust
let depths = [2, 3, 5, 10];

for &depth in &depths {
    let mut tree = DecisionTreeRegressor::new().with_max_depth(depth);
    tree.fit(&x_train, &y_train).unwrap();

    let r2 = tree.score(&x_train, &y_train);
    println!("max_depth={}: R² = {:.4}", depth, r2);
}

// Output:
// max_depth=2: R² = 0.9374  (underfitting)
// max_depth=3: R² = 0.9903  (good balance)
// max_depth=5: R² = 1.0000  (perfect fit)
// max_depth=10: R² = 1.0000 (potential overfitting)
```

**Interpretation**:
- **depth=2**: Too shallow, can't capture complexity → underfitting
- **depth=3**: Good balance, likely generalizes well
- **depth=5+**: Perfect training fit, risk of overfitting on test data

**Rule of Thumb**:
- Start with max_depth = 3-5
- Increase if underfitting (low train R²)
- Decrease if overfitting (high train R², low test R²)
- Use cross-validation to find optimal depth

**Test Reference**: `src/tree/mod.rs::test_regression_tree_max_depth`

---

### min_samples_split: Pruning Parameter

```rust
// Default tree (no pruning)
let mut tree_default = DecisionTreeRegressor::new()
    .with_max_depth(10);

// Pruned tree (requires 4 samples to split)
let mut tree_pruned = DecisionTreeRegressor::new()
    .with_max_depth(10)
    .with_min_samples_split(4)
    .with_min_samples_leaf(2);

tree_default.fit(&x_train, &y_train).unwrap();
tree_pruned.fit(&x_train, &y_train).unwrap();

let r2_default = tree_default.score(&x_train, &y_train);
let r2_pruned = tree_pruned.score(&x_train, &y_train);

println!("Default tree R²: {:.4}", r2_default); // 1.0000
println!("Pruned tree R²: {:.4}", r2_pruned);   // 0.9658
```

**Effect of Pruning**:
- **min_samples_split=4**: Don't split nodes with < 4 samples
- **min_samples_leaf=2**: Ensure each leaf has ≥ 2 samples
- **Result**: Simpler tree, prevents overfitting on small groups

**When to Use**:
- Noisy data (prevents fitting to outliers)
- Small datasets (improves generalization)
- Prefer simpler models (Occam's razor)

**Test Reference**: `src/tree/mod.rs::test_regression_tree_min_samples_*`

---

## Non-Linear Patterns

Decision trees naturally handle non-linear relationships. Example with quadratic data:

```rust
// Pure quadratic relationship: y = x²
let x_quad = Matrix::from_vec(10, 1, vec![
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
]).unwrap();

let y_quad = Vector::from_slice(&[
    1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0
]);

// Train both models
let mut tree = DecisionTreeRegressor::new().with_max_depth(4);
let mut linear = LinearRegression::new();

tree.fit(&x_quad, &y_quad).unwrap();
linear.fit(&x_quad, &y_quad).unwrap();

let tree_r2 = tree.score(&x_quad, &y_quad);
let linear_r2 = linear.score(&x_quad, &y_quad);

println!("Decision Tree R²: {:.4}", tree_r2);   // 1.0000
println!("Linear Regression R²: {:.4}", linear_r2); // 0.9498
```

**Why Tree Wins**:
- Learns step function approximation of parabola
- No need for manual feature engineering (x²)
- Captures local patterns

**Linear Model Struggles**:
- Tries to fit straight line to curve
- Needs polynomial features: `[x, x²]`
- Can't learn without feature engineering

**Visualization**:
```text
x    True y   Tree Pred   Linear Pred
1    1        1.0         -11.0
2    4        4.0         0.0
3    9        9.0         11.0
5    25       25.0        33.0
10   100      100.0       88.0
```

Decision tree predictions match exactly (or very close), while linear model has systematic error (underpredicts low, overpredicts high).

**Test Reference**: `src/tree/mod.rs::test_regression_tree_predict_nonlinear`

---

## Edge Cases and Validation

### Constant Target

```rust
// All houses same price (constant target)
let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
let y = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0, 5.0]);

let mut tree = DecisionTreeRegressor::new().with_max_depth(3);
tree.fit(&x, &y).unwrap();

// Should predict constant value
let predictions = tree.predict(&x);
for &pred in predictions.as_slice() {
    assert!((pred - 5.0).abs() < 1e-5); // All ≈ 5.0
}
```

**Behavior**: Tree creates single leaf node (MSE = 0, pure node).

**Test Reference**: `src/tree/mod.rs::test_regression_tree_constant_target`

---

### Single Sample

```rust
// Edge case: only 1 training sample
let x = Matrix::from_vec(1, 2, vec![1.0, 2.0]).unwrap();
let y = Vector::from_slice(&[10.0]);

let mut tree = DecisionTreeRegressor::new().with_max_depth(3);
tree.fit(&x, &y).unwrap();

// Predict on same sample
let pred = tree.predict(&x);
assert!((pred.as_slice()[0] - 10.0).abs() < 1e-5);
```

**Behavior**: Creates single leaf with mean = 10.0.

**Test Reference**: `src/tree/mod.rs::test_regression_tree_single_sample`

---

### Validation Errors

```rust
// Error: Mismatched dimensions
let x = Matrix::from_vec(5, 2, vec![...]).unwrap();
let y = Vector::from_slice(&[1.0, 2.0, 3.0]); // Only 3 labels!

let mut tree = DecisionTreeRegressor::new();
assert!(tree.fit(&x, &y).is_err()); // Returns error

// Error: Predict before fit
let tree = DecisionTreeRegressor::new();
// tree.predict(&x); // Would panic!
```

**Validation Checks**:
- `x.rows() == y.len()` (sample count match)
- Tree must be fitted before predict
- Features count must match between train and test

**Test Reference**: `src/tree/mod.rs::test_regression_tree_validation_*`

---

## Practical Recommendations

### When to Use Decision Tree Regression

✅ **Use when**:
- Non-linear relationships in data
- Feature interactions are important
- Interpretability is needed (can visualize tree)
- No feature scaling available (mixed units)
- Building block for ensembles (Random Forest)

❌ **Don't use when**:
- Linear relationships (use LinearRegression)
- Small datasets (< 50 samples, risk overfitting)
- Need smooth predictions (trees predict step functions)
- Extrapolation required (beyond training range)

### Hyperparameter Selection Guide

| Parameter | Typical Range | Effect | When to Increase | When to Decrease |
|-----------|---------------|--------|------------------|------------------|
| **max_depth** | 3-10 | Tree complexity | Underfitting (low train R²) | Overfitting (train R² >> test R²) |
| **min_samples_split** | 2-10 | Minimum samples to split | Overfitting | Underfitting |
| **min_samples_leaf** | 1-5 | Minimum leaf size | Overfitting | Underfitting |

**Tuning Process**:
1. Start with defaults: `max_depth=5`, `min_samples_split=2`, `min_samples_leaf=1`
2. Check train/test R² (use cross-validation)
3. If overfitting: Decrease max_depth or increase min_samples_*
4. If underfitting: Increase max_depth or decrease min_samples_*
5. Use grid search for optimal combination

### Debugging Checklist

**Low R² on training data**:
- [ ] Tree too shallow (increase max_depth)
- [ ] Too much pruning (decrease min_samples_split/leaf)
- [ ] Data has no predictive signal

**Perfect train R², poor test R²**:
- [ ] Overfitting! (decrease max_depth)
- [ ] Add pruning (increase min_samples_split/leaf)
- [ ] Need more training data

**Unexpected predictions**:
- [ ] Check feature scaling (not needed, but verify units)
- [ ] Inspect tree structure (if implemented)
- [ ] Verify training data quality

---

## Full Example Code

```rust
use aprender::prelude::*;

fn main() {
    // Housing data
    let x_train = Matrix::from_vec(8, 3, vec![
        1500.0, 3.0, 10.0,  // $280k
        2000.0, 4.0, 5.0,   // $350k
        1200.0, 2.0, 30.0,  // $180k
        1800.0, 3.0, 15.0,  // $300k
        2500.0, 5.0, 2.0,   // $450k
        1000.0, 2.0, 50.0,  // $150k
        2200.0, 4.0, 8.0,   // $380k
        1600.0, 3.0, 20.0,  // $260k
    ]).unwrap();

    let y_train = Vector::from_slice(&[
        280.0, 350.0, 180.0, 300.0, 450.0, 150.0, 380.0, 260.0
    ]);

    // Train regression tree
    let mut tree = DecisionTreeRegressor::new()
        .with_max_depth(4)
        .with_min_samples_split(2);

    tree.fit(&x_train, &y_train).unwrap();

    // Evaluate
    let r2 = tree.score(&x_train, &y_train);
    println!("R² Score: {:.3}", r2);

    // Predict on new house
    let x_new = Matrix::from_vec(1, 3, vec![1900.0, 4.0, 12.0]).unwrap();
    let price = tree.predict(&x_new);
    println!("Predicted price: ${:.0}k", price.as_slice()[0]);
}
```

**Run the example**:
```bash
cargo run --example decision_tree_regression
```

---

## Related Reading

**Theory**:
- [Decision Trees Theory](../ml-fundamentals/decision-trees.md) - MSE criterion, CART algorithm
- [Regression Metrics](../ml-fundamentals/regression-metrics.md) - R², MSE, MAE

**Other Algorithms**:
- [Linear Regression](./linear-regression.md) - Baseline comparison
- [Random Forest (Future)](./random-forest-regression.md) - Ensemble of trees

**Code Reference**:
- Implementation: `src/tree/mod.rs` (DecisionTreeRegressor)
- Tests: `src/tree/mod.rs::tests::test_regression_tree_*` (16 tests)
- Example: `examples/decision_tree_regression.rs`

---

## Summary

**Key Takeaways**:
- ✅ Decision tree regression uses MSE criterion (variance reduction)
- ✅ Leaf nodes predict mean of training samples
- ✅ max_depth prevents overfitting (typical: 3-7)
- ✅ Pruning parameters (min_samples_*) add regularization
- ✅ Excels at non-linear relationships without feature engineering
- ✅ Interpretable but can overfit (use ensembles in production)

**Best Practices**:
1. Start with max_depth=5, tune with cross-validation
2. Compare with LinearRegression baseline
3. Use R² for evaluation, check train/test gap
4. Prune with min_samples_split/leaf if overfitting
5. Consider Random Forest for better accuracy

**Verification**: Implementation tested with 16 comprehensive tests in `src/tree/mod.rs`, including edge cases, parameter validation, and comparison with linear regression.

---

**Next**: [Random Forest Regression (Future)](./random-forest-regression.md)

**Previous**: [Decision Tree - Iris Classification](./decision-tree-iris.md)
