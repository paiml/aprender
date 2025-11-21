# Feature Scaling Theory

Feature scaling is a critical preprocessing step that transforms features to similar scales. Proper scaling dramatically improves convergence speed and model performance, especially for distance-based algorithms and gradient descent optimization.

## Why Feature Scaling Matters

### Problem: Features on Different Scales

Consider a dataset with two features:

```
Feature 1 (salary):    [30,000, 50,000, 80,000, 120,000]  Range: 90,000
Feature 2 (age):       [25, 30, 35, 40]                    Range: 15
```

**Issue**: Salary values are ~6000x larger than age values!

### Impact on Machine Learning Algorithms

#### 1. Gradient Descent

Without scaling, loss surface becomes elongated:

```
Unscaled Loss Surface:
           θ₁ (salary coefficient)
           ↑
      1000 ┤●
       800 ┤ ●
       600 ┤  ●
       400 ┤   ●  ← Very elongated
       200 ┤    ●●●●●●●●●●●●●●●●●
         0 └────────────────────────→
                 θ₂ (age coefficient)

Problem: Gradient descent takes tiny steps in θ₁ direction,
         large steps in θ₂ direction → zig-zagging, slow convergence
```

With scaling, loss surface becomes circular:

```
Scaled Loss Surface:
           θ₁
           ↑
      1.0 ┤
      0.8 ┤    ●●●
      0.6 ┤  ●     ●  ← Circular contours
      0.4 ┤ ●   ✖   ●  (✖ = optimal)
      0.2 ┤  ●     ●
      0.0 └───●●●─────→
                θ₂

Result: Gradient descent takes efficient path to minimum
```

**Convergence speed**: Scaling can improve training time by **10-100x**!

#### 2. Distance-Based Algorithms (K-NN, K-Means, SVM)

Euclidean distance formula:

```
d = √((x₁-y₁)² + (x₂-y₂)²)
```

With unscaled features:

```
Sample A: (salary=50000, age=30)
Sample B: (salary=51000, age=35)

Distance = √((51000-50000)² + (35-30)²)
         = √(1000² + 5²)
         = √(1,000,000 + 25)
         = √1,000,025
         ≈ 1000.01

Contribution to distance:
  Salary: 1,000,000 / 1,000,025 ≈ 99.997%
  Age:           25 / 1,000,025 ≈  0.003%
```

**Problem**: Age is completely ignored! K-NN makes decisions based solely on salary.

With scaled features (both in range [0, 1]):

```
Scaled A: (0.2, 0.33)
Scaled B: (0.3, 0.67)

Distance = √((0.3-0.2)² + (0.67-0.33)²)
         = √(0.01 + 0.1156)
         = √0.1256
         ≈ 0.354

Contribution to distance:
  Salary: 0.01 / 0.1256 ≈ 8%
  Age:   0.1156 / 0.1256 ≈ 92%
```

**Result**: Both features contribute meaningfully to distance calculation.

## Scaling Methods

### Comparison Table

| Method | Formula | Range | Best For | Outlier Sensitive |
|--------|---------|-------|----------|-------------------|
| **StandardScaler** | (x - μ) / σ | Unbounded, ~[-3, 3] | Normal distributions | Low |
| **MinMaxScaler** | (x - min) / (max - min) | [0, 1] or custom | Known bounds needed | High |
| **RobustScaler** | (x - median) / IQR | Unbounded | Data with outliers | Low |
| **MaxAbsScaler** | x / \|max\| | [-1, 1] | Sparse data, preserves zeros | High |
| **Normalization (L2)** | x / ‖x‖₂ | Unit sphere | Text, TF-IDF vectors | N/A |

## StandardScaler: Z-Score Normalization

**Key idea**: Center data at zero, scale by standard deviation.

### Formula

```
x' = (x - μ) / σ

Where:
  μ = mean of feature
  σ = standard deviation of feature
```

### Properties

After standardization:
- **Mean = 0**
- **Standard deviation = 1**
- Distribution shape preserved

### Algorithm

```
1. Fit phase (training data):
   μ = (1/N) Σ xᵢ                    // Compute mean
   σ = √[(1/N) Σ (xᵢ - μ)²]          // Compute std

2. Transform phase:
   x'ᵢ = (xᵢ - μ) / σ                // Scale each sample

3. Inverse transform (optional):
   xᵢ = x'ᵢ × σ + μ                  // Recover original scale
```

### Example

```
Original data: [1, 2, 3, 4, 5]

Step 1: Compute statistics
  μ = (1+2+3+4+5) / 5 = 3
  σ = √[(1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²] / 5
    = √[4 + 1 + 0 + 1 + 4] / 5
    = √2 ≈ 1.414

Step 2: Transform
  x'₁ = (1 - 3) / 1.414 = -1.414
  x'₂ = (2 - 3) / 1.414 = -0.707
  x'₃ = (3 - 3) / 1.414 =  0.000
  x'₄ = (4 - 3) / 1.414 =  0.707
  x'₅ = (5 - 3) / 1.414 =  1.414

Result: [-1.414, -0.707, 0.000, 0.707, 1.414]
  Mean = 0, Std = 1 ✓
```

### aprender Implementation

```rust
use aprender::preprocessing::StandardScaler;
use aprender::primitives::Matrix;

// Create scaler
let mut scaler = StandardScaler::new();

// Fit on training data
scaler.fit(&x_train)?;

// Transform training and test data
let x_train_scaled = scaler.transform(&x_train)?;
let x_test_scaled = scaler.transform(&x_test)?;

// Access learned statistics
println!("Mean: {:?}", scaler.mean());
println!("Std:  {:?}", scaler.std());

// Inverse transform (recover original scale)
let x_recovered = scaler.inverse_transform(&x_train_scaled)?;
```

### Advantages

1. **Robust to outliers**: Outliers affect mean/std less than min/max
2. **Maintains distribution shape**: Useful for normally distributed data
3. **Unbounded output**: Can handle values outside training range
4. **Interpretable**: "How many standard deviations from the mean?"

### Disadvantages

1. **Assumes normality**: Less effective for heavily skewed distributions
2. **Unbounded range**: Output not in [0, 1] if that's required
3. **Outliers still affect**: Mean and std sensitive to extreme values

### When to Use

✅ **Use StandardScaler for**:
- Features with approximately normal distribution
- Gradient-based optimization (neural networks, logistic regression)
- SVM with RBF kernel
- PCA (Principal Component Analysis)
- Data with moderate outliers

❌ **Avoid StandardScaler for**:
- Need strict [0, 1] bounds (use MinMaxScaler)
- Heavy outliers (use RobustScaler)
- Sparse data with many zeros (use MaxAbsScaler)

## MinMaxScaler: Range Normalization

**Key idea**: Scale features to a fixed range, typically [0, 1].

### Formula

```
x' = (x - min) / (max - min)           // Scale to [0, 1]

x' = a + (x - min) × (b - a) / (max - min)  // Scale to [a, b]
```

### Properties

After min-max scaling to [0, 1]:
- **Minimum value → 0**
- **Maximum value → 1**
- Linear transformation (preserves relationships)

### Algorithm

```
1. Fit phase (training data):
   min = minimum value in feature
   max = maximum value in feature
   range = max - min

2. Transform phase:
   x'ᵢ = (xᵢ - min) / range

3. Inverse transform:
   xᵢ = x'ᵢ × range + min
```

### Example

```
Original data: [10, 20, 30, 40, 50]

Step 1: Compute range
  min = 10
  max = 50
  range = 50 - 10 = 40

Step 2: Transform to [0, 1]
  x'₁ = (10 - 10) / 40 = 0.00
  x'₂ = (20 - 10) / 40 = 0.25
  x'₃ = (30 - 10) / 40 = 0.50
  x'₄ = (40 - 10) / 40 = 0.75
  x'₅ = (50 - 10) / 40 = 1.00

Result: [0.00, 0.25, 0.50, 0.75, 1.00]
  Min = 0, Max = 1 ✓
```

### Custom Range Example

Scale to [-1, 1] for neural networks with tanh activation:

```
Original: [10, 20, 30, 40, 50]
Range: [min=10, max=50]

Formula: x' = -1 + (x - 10) × 2 / 40

Result:
  10 → -1.0
  20 → -0.5
  30 →  0.0
  40 →  0.5
  50 →  1.0
```

### aprender Implementation

```rust
use aprender::preprocessing::MinMaxScaler;

// Scale to [0, 1] (default)
let mut scaler = MinMaxScaler::new();

// Scale to custom range [-1, 1]
let mut scaler = MinMaxScaler::new()
    .with_range(-1.0, 1.0);

// Fit and transform
scaler.fit(&x_train)?;
let x_train_scaled = scaler.transform(&x_train)?;
let x_test_scaled = scaler.transform(&x_test)?;

// Access learned parameters
println!("Data min: {:?}", scaler.data_min());
println!("Data max: {:?}", scaler.data_max());

// Inverse transform
let x_recovered = scaler.inverse_transform(&x_train_scaled)?;
```

### Advantages

1. **Bounded output**: Guaranteed range [0, 1] or custom
2. **Preserves zero**: If data contains zeros, they remain zeros
3. **Interpretable**: "What percentage of the range?"
4. **No assumptions**: Works with any distribution

### Disadvantages

1. **Sensitive to outliers**: Single extreme value affects entire scaling
2. **Bounded by training data**: Test values outside [train_min, train_max] → outside [0, 1]
3. **Distorts distribution**: Outliers compress main data range

### When to Use

✅ **Use MinMaxScaler for**:
- Neural networks with sigmoid/tanh activation
- Bounded features needed (e.g., image pixels)
- No outliers present
- Features with known bounds
- When interpretability as "percentage" is useful

❌ **Avoid MinMaxScaler for**:
- Data with outliers (they compress everything else)
- Test data may have values outside training range
- Need to preserve distribution shape

## Outlier Handling Comparison

### Dataset with Outlier

```
Data: [1, 2, 3, 4, 5, 100]  ← 100 is an outlier
```

### StandardScaler (Less Affected)

```
μ = (1+2+3+4+5+100) / 6 ≈ 19.17
σ ≈ 37.85

Scaled:
  1   → (1-19.17)/37.85  ≈ -0.48
  2   → (2-19.17)/37.85  ≈ -0.45
  3   → (3-19.17)/37.85  ≈ -0.43
  4   → (4-19.17)/37.85  ≈ -0.40
  5   → (5-19.17)/37.85  ≈ -0.37
  100 → (100-19.17)/37.85 ≈ 2.14

Main data: [-0.48 to -0.37]  (range ≈ 0.11)
Outlier: 2.14
```

**Effect**: Outlier shifted but main data still usable, relatively compressed.

### MinMaxScaler (Heavily Affected)

```
min = 1, max = 100, range = 99

Scaled:
  1   → (1-1)/99   = 0.000
  2   → (2-1)/99   = 0.010
  3   → (3-1)/99   = 0.020
  4   → (4-1)/99   = 0.030
  5   → (5-1)/99   = 0.040
  100 → (100-1)/99 = 1.000

Main data: [0.000 to 0.040]  (compressed to 4% of range!)
Outlier: 1.000
```

**Effect**: Outlier uses 96% of range, main data compressed to tiny interval.

**Lesson**: Use StandardScaler or RobustScaler when outliers are present!

## When to Scale Features

### Algorithms That REQUIRE Scaling

These algorithms **fail or perform poorly** without scaling:

| Algorithm | Why Scaling Needed |
|-----------|-------------------|
| **K-Nearest Neighbors** | Distance calculation dominated by large-scale features |
| **K-Means Clustering** | Centroid calculation uses Euclidean distance |
| **Support Vector Machines** | Distance to hyperplane affected by feature scales |
| **Principal Component Analysis** | Variance calculation dominated by large-scale features |
| **Gradient Descent** | Elongated loss surface causes slow convergence |
| **Neural Networks** | Weights initialized for similar input scales |
| **Logistic Regression** | Gradient descent convergence issues |

### Algorithms That DON'T Need Scaling

These algorithms are **scale-invariant**:

| Algorithm | Why Scaling Not Needed |
|-----------|----------------------|
| **Decision Trees** | Splits based on thresholds, not distances |
| **Random Forests** | Ensemble of decision trees |
| **Gradient Boosting** | Based on decision trees |
| **Naive Bayes** | Works with probability distributions |

**Exception**: Even for tree-based models, scaling can help if using regularization or mixed with other algorithms.

## Critical Workflow Rules

### Rule 1: Fit on Training Data ONLY

```rust
// ❌ WRONG: Fitting on all data leaks information
scaler.fit(&x_all)?;
let x_train_scaled = scaler.transform(&x_train)?;
let x_test_scaled = scaler.transform(&x_test)?;

// ✅ CORRECT: Fit only on training data
scaler.fit(&x_train)?;  // Learn μ, σ from training only
let x_train_scaled = scaler.transform(&x_train)?;
let x_test_scaled = scaler.transform(&x_test)?;  // Apply same μ, σ
```

**Why?** Fitting on test data creates **data leakage**:
- Test set statistics influence scaling
- Model indirectly "sees" test data during training
- Overly optimistic performance estimates
- Fails in production (new data has different statistics)

### Rule 2: Same Scaler for Train and Test

```rust
// ❌ WRONG: Different scalers
let mut train_scaler = StandardScaler::new();
train_scaler.fit(&x_train)?;
let x_train_scaled = train_scaler.transform(&x_train)?;

let mut test_scaler = StandardScaler::new();
test_scaler.fit(&x_test)?;  // ← WRONG! Different statistics
let x_test_scaled = test_scaler.transform(&x_test)?;

// ✅ CORRECT: Same scaler
let mut scaler = StandardScaler::new();
scaler.fit(&x_train)?;
let x_train_scaled = scaler.transform(&x_train)?;
let x_test_scaled = scaler.transform(&x_test)?;  // Same statistics
```

### Rule 3: Scale Before Splitting? NO!

```rust
// ❌ WRONG: Scale before train/test split
scaler.fit(&x_all)?;
let x_scaled = scaler.transform(&x_all)?;
let (x_train, x_test, ...) = train_test_split(&x_scaled, ...)?;

// ✅ CORRECT: Split before scaling
let (x_train, x_test, ...) = train_test_split(&x, ...)?;
scaler.fit(&x_train)?;
let x_train_scaled = scaler.transform(&x_train)?;
let x_test_scaled = scaler.transform(&x_test)?;
```

### Rule 4: Save Scaler for Production

```rust
// Training phase
let mut scaler = StandardScaler::new();
scaler.fit(&x_train)?;

// Save scaler parameters
let scaler_params = ScalerParams {
    mean: scaler.mean().clone(),
    std: scaler.std().clone(),
};
save_to_disk(&scaler_params, "scaler.json")?;

// Production phase (months later)
let scaler_params = load_from_disk("scaler.json")?;
let mut scaler = StandardScaler::from_params(scaler_params);
let x_new_scaled = scaler.transform(&x_new)?;
```

## Feature-Specific Scaling Strategies

### Numerical Features

**Continuous variables** (age, salary, temperature):
- StandardScaler if approximately normal
- MinMaxScaler if bounded and no outliers
- RobustScaler if outliers present

### Binary Features (0/1)

**No scaling needed!**

```
Original: [0, 1, 0, 1, 1]  ← Already in [0, 1]

Don't scale: Breaks semantic meaning (presence/absence)
```

### Count Features

**Examples**: Number of purchases, page visits, words in document

**Strategy**: Consider log transformation first, then scale

```rust
// Apply log transform
let x_log: Vec<f32> = x.iter()
    .map(|&count| (count + 1.0).ln())  // +1 to handle zeros
    .collect();

// Then scale
scaler.fit(&x_log)?;
let x_scaled = scaler.transform(&x_log)?;
```

### Categorical Features (Encoded)

**One-hot encoded**: No scaling needed (already 0/1)
**Label encoded** (ordinal): Scale if using distance-based algorithms

## Impact on Model Performance

### Example: K-NN on Employee Data

```
Dataset:
  Feature 1: Salary [30k-120k]
  Feature 2: Age [25-40]
  Feature 3: Years of experience [1-15]

Task: Predict employee attrition

Without scaling:
  K-NN accuracy: 62%
  (Salary dominates distance calculation)

With StandardScaler:
  K-NN accuracy: 84%
  (All features contribute meaningfully)

Improvement: +22 percentage points! ✅
```

### Example: Neural Network Convergence

```
Network: 3-layer MLP
Dataset: Mixed-scale features

Without scaling:
  Epochs to converge: 500
  Training time: 45 seconds

With StandardScaler:
  Epochs to converge: 50
  Training time: 5 seconds

Speedup: 9x faster! ✅
```

## Decision Guide

### Flowchart: Which Scaler?

```
Start
  │
  ├─ Are there outliers?
  │    ├─ YES → RobustScaler
  │    └─ NO  → Continue
  │
  ├─ Need bounded range [0,1]?
  │    ├─ YES → MinMaxScaler
  │    └─ NO  → Continue
  │
  ├─ Is data approximately normal?
  │    ├─ YES → StandardScaler ✓ (default choice)
  │    └─ NO  → Continue
  │
  ├─ Is data sparse (many zeros)?
  │    ├─ YES → MaxAbsScaler
  │    └─ NO  → StandardScaler
```

### Quick Reference

| Your Situation | Recommended Scaler |
|----------------|-------------------|
| Default choice, unsure | **StandardScaler** |
| Neural networks | **StandardScaler** or **MinMaxScaler** |
| K-NN, K-Means, SVM | **StandardScaler** |
| Data has outliers | **RobustScaler** |
| Need [0,1] bounds | **MinMaxScaler** |
| Sparse data | **MaxAbsScaler** |
| Tree-based models | **No scaling** (optional) |

## Common Mistakes

### Mistake 1: Forgetting to Scale Test Data

```rust
// ❌ WRONG
scaler.fit(&x_train)?;
let x_train_scaled = scaler.transform(&x_train)?;
// ... train model on x_train_scaled ...
let predictions = model.predict(&x_test)?;  // ← Unscaled!
```

**Result**: Model sees different scale at test time, terrible performance.

### Mistake 2: Scaling Target Variable Unnecessarily

```rust
// ❌ Usually unnecessary for regression targets
scaler_y.fit(&y_train)?;
let y_train_scaled = scaler_y.transform(&y_train)?;
```

**When needed**: Only if target has extreme range (e.g., house prices in millions)

**Better solution**: Use regularization or log-transform target

### Mistake 3: Scaling Categorical Encoded Features

```rust
// One-hot encoded: [1, 0, 0] for category A
//                  [0, 1, 0] for category B

// ❌ WRONG: Scaling destroys categorical meaning
scaler.fit(&one_hot_encoded)?;
```

**Correct**: Don't scale one-hot encoded features!

## aprender Example: Complete Pipeline

```rust
use aprender::preprocessing::StandardScaler;
use aprender::classification::KNearestNeighbors;
use aprender::model_selection::train_test_split;
use aprender::prelude::*;

fn full_pipeline_example(x: &Matrix<f32>, y: &Vec<i32>) -> Result<f32> {
    // 1. Split data FIRST
    let (x_train, x_test, y_train, y_test) =
        train_test_split(x, y, 0.2, Some(42))?;

    // 2. Create and fit scaler on training data ONLY
    let mut scaler = StandardScaler::new();
    scaler.fit(&x_train)?;

    // 3. Transform both train and test using same scaler
    let x_train_scaled = scaler.transform(&x_train)?;
    let x_test_scaled = scaler.transform(&x_test)?;

    // 4. Train model on scaled data
    let mut model = KNearestNeighbors::new(5);
    model.fit(&x_train_scaled, &y_train)?;

    // 5. Evaluate on scaled test data
    let accuracy = model.score(&x_test_scaled, &y_test);

    println!("Learned scaling parameters:");
    println!("  Mean: {:?}", scaler.mean());
    println!("  Std:  {:?}", scaler.std());
    println!("\nTest accuracy: {:.4}", accuracy);

    Ok(accuracy)
}
```

## Further Reading

**Theory**:
- Standardization: Common practice in statistics since 1950s
- Min-Max Scaling: Standard normalization technique

**Practical**:
- sklearn documentation: Detailed scaler comparisons
- "Feature Engineering for Machine Learning" (Zheng & Casari)

## Related Chapters

- [Data Preprocessing with Scalers](../examples/data-preprocessing-scalers.md) - Hands-on examples
- [K-NN Iris Example](../examples/knn-iris.md) - Scaling impact on K-NN
- [Gradient Descent Theory](./gradient-descent.md) - Why scaling accelerates optimization

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| **Why scale?** | Distance-based algorithms and gradient descent need similar feature scales |
| **StandardScaler** | Default choice: centers at 0, scales by std dev |
| **MinMaxScaler** | When bounded [0,1] range needed, no outliers |
| **Fit on training** | CRITICAL: Only fit scaler on training data, apply to test |
| **Algorithms needing scaling** | K-NN, K-Means, SVM, Neural Networks, PCA |
| **Algorithms NOT needing scaling** | Decision Trees, Random Forests, Naive Bayes |
| **Performance impact** | Can improve accuracy by 20%+ and speed by 10-100x |

Feature scaling is often the **single most important preprocessing step** in machine learning pipelines. Proper scaling can mean the difference between a model that fails to converge and one that achieves state-of-the-art performance.
