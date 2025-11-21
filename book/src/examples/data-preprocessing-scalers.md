# Data Preprocessing with Scalers

This example demonstrates feature scaling with `StandardScaler` and `MinMaxScaler`, two fundamental data preprocessing techniques used before training machine learning models.

## Overview

Feature scaling ensures that all features are on comparable scales, which is crucial for many ML algorithms (especially distance-based methods like K-NN, SVM, and neural networks).

## Running the Example

```bash
cargo run --example data_preprocessing_scalers
```

## Key Concepts

### StandardScaler (Z-score Normalization)

StandardScaler transforms features to have:
- **Mean = 0** (centers data)
- **Standard Deviation = 1** (scales data)

**Formula**: `z = (x - μ) / σ`

**When to use**:
- Data is approximately normally distributed
- Presence of outliers (more robust than MinMax)
- Algorithms sensitive to feature scale (SVM, neural networks)
- Want to preserve relative distances

### MinMaxScaler (Range Normalization)

MinMaxScaler transforms features to a specific range (default `[0, 1]`):

**Formula**: `x' = (x - min) / (max - min)`

**When to use**:
- Need specific output range (e.g., `[0, 1]` for probabilities)
- Data not normally distributed
- No outliers present
- Want to preserve zero values
- Image processing (pixel normalization)

## Examples Demonstrated

### Example 1: StandardScaler Basics

Shows how StandardScaler transforms data with different scales:

```
Original Data:
  Feature 0: [100, 200, 300, 400, 500]
  Feature 1: [1, 2, 3, 4, 5]

Computed Statistics:
  Mean: [300.0, 3.0]
  Std:  [141.42, 1.41]

After StandardScaler:
  Sample 0: [-1.41, -1.41]
  Sample 1: [-0.71, -0.71]
  Sample 2: [ 0.00,  0.00]
  Sample 3: [ 0.71,  0.71]
  Sample 4: [ 1.41,  1.41]
```

Both features now have mean=0 and std=1, despite very different original scales.

### Example 2: MinMaxScaler Basics

Shows how MinMaxScaler transforms to `[0, 1]` range:

```
Original Data:
  Feature 0: [10, 20, 30, 40, 50]
  Feature 1: [100, 200, 300, 400, 500]

After MinMaxScaler [0, 1]:
  Sample 0: [0.00, 0.00]
  Sample 1: [0.25, 0.25]
  Sample 2: [0.50, 0.50]
  Sample 3: [0.75, 0.75]
  Sample 4: [1.00, 1.00]
```

Both features now in `[0, 1]` range with identical relative positions.

### Example 3: Handling Outliers

Demonstrates how each scaler responds to outliers:

```
Data with Outlier: [1, 2, 3, 4, 5, 100]

  Original  StandardScaler  MinMaxScaler
  ----------------------------------------
       1.0           -0.50          0.00
       2.0           -0.47          0.01
       3.0           -0.45          0.02
       4.0           -0.42          0.03
       5.0           -0.39          0.04
     100.0            2.23          1.00
```

**Observations**:
- **StandardScaler**: Outlier is ~2.3 standard deviations from mean (less compression)
- **MinMaxScaler**: Outlier compresses all other values near 0 (heavily affected)

**Recommendation**: Use StandardScaler when outliers are present.

### Example 4: Impact on K-NN Classification

Shows why scaling is critical for distance-based algorithms:

```
Dataset: Employee classification
  Feature 0: Salary (50-95k, range=45)
  Feature 1: Age (25-42 years, range=17)

Test: Salary=70k, Age=33

Without scaling: Distance dominated by salary
With scaling:    Both features contribute equally
```

**Why it matters**:
- K-NN uses Euclidean distance
- Large-scale features (salary) dominate the calculation
- Small differences in age (2-3 years) become negligible
- Scaling equalizes feature importance

### Example 5: Custom Range Scaling

Demonstrates `MinMaxScaler` with custom ranges:

```rust
let scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
```

**Common use cases**:
- `[-1, 1]`: Neural networks with tanh activation
- `[0, 1]`: Probabilities, image pixels (standard)
- `[0, 255]`: 8-bit image processing

### Example 6: Inverse Transformation

Shows how to recover original scale after scaling:

```rust
let scaled = scaler.fit_transform(&original).unwrap();
let recovered = scaler.inverse_transform(&scaled).unwrap();
// recovered == original (within floating point precision)
```

**When to use**:
- Interpreting model coefficients in original units
- Presenting predictions to end users
- Visualizing scaled data
- Debugging transformations

## Best Practices

### 1. Fit Only on Training Data

```rust
// ✅ Correct
let mut scaler = StandardScaler::new();
scaler.fit(&x_train).unwrap();              // Fit on training data
let x_train_scaled = scaler.transform(&x_train).unwrap();
let x_test_scaled = scaler.transform(&x_test).unwrap();  // Same scaler on test

// ❌ Incorrect (data leakage!)
scaler.fit(&x_test).unwrap();  // Never fit on test data
```

### 2. Use fit_transform() for Convenience

```rust
// Shortcut for training data
let x_train_scaled = scaler.fit_transform(&x_train).unwrap();

// Equivalent to:
scaler.fit(&x_train).unwrap();
let x_train_scaled = scaler.transform(&x_train).unwrap();
```

### 3. Save Scaler with Model

The scaler is part of your model pipeline and must be saved/loaded with the model to ensure consistent preprocessing at prediction time.

### 4. Check if Scaler is Fitted

```rust
if scaler.is_fitted() {
    // Safe to transform
}
```

## Decision Guide

### Choose StandardScaler when:
- ✅ Data is approximately normally distributed
- ✅ Outliers are present
- ✅ Using linear models, SVM, neural networks
- ✅ Want interpretable z-scores

### Choose MinMaxScaler when:
- ✅ Need specific output range
- ✅ No outliers present
- ✅ Data not normally distributed
- ✅ Using image data
- ✅ Want to preserve zero values
- ✅ Using algorithms that require specific range (e.g., sigmoid activation)

### Don't Scale when:
- ❌ Using tree-based methods (Decision Trees, Random Forests, GBM)
- ❌ Features already on same scale
- ❌ Scale carries semantic meaning (e.g., age, count data)

## Implementation Details

Both scalers implement the `Transformer` trait with methods:
- `fit(x)` - Compute statistics from data
- `transform(x)` - Apply transformation
- `fit_transform(x)` - Fit then transform
- `inverse_transform(x)` - Reverse transformation

Both scalers:
- Work with `Matrix<f32>` from aprender primitives
- Store statistics (mean/std or min/max) per feature
- Support builder pattern for configuration
- Return `Result` for error handling

## Common Pitfalls

1. **Fitting on test data**: Always fit scaler on training data only
2. **Forgetting to scale test data**: Must apply same transformation to test set
3. **Using wrong scaler**: MinMaxScaler sensitive to outliers
4. **Over-scaling**: Don't scale tree-based models
5. **Losing the scaler**: Save scaler with model for production use

## Related Examples

- [K-Nearest Neighbors](./knn-classification.md) - Distance-based classification
- [Descriptive Statistics](./descriptive-statistics.md) - Computing mean and std
- [Linear Regression](./linear-regression.md) - Model that benefits from scaling

## Key Takeaways

1. **Feature scaling is essential** for distance-based and gradient-based algorithms
2. **StandardScaler** is robust to outliers and preserves relative distances
3. **MinMaxScaler** gives exact range control but is outlier-sensitive
4. **Always fit on training data** and transform both train and test sets
5. **Save scalers with models** for consistent production predictions
6. **Tree-based models don't need scaling** - they're scale-invariant
7. **Use inverse_transform()** to interpret results in original units
