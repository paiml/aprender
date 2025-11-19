# K-Nearest Neighbors (kNN)

K-Nearest Neighbors (kNN) is a simple yet powerful instance-based learning algorithm for classification and regression. Unlike parametric models that learn explicit parameters during training, kNN is a "lazy learner" that simply stores the training data and makes predictions by finding similar examples at inference time. This chapter covers the theory, implementation, and practical considerations for using kNN in aprender.

## What is K-Nearest Neighbors?

kNN is a non-parametric, instance-based learning algorithm that classifies new data points based on the majority class among their k nearest neighbors in the feature space.

**Key characteristics**:
- **Lazy learning**: No explicit training phase, just stores training data
- **Non-parametric**: Makes no assumptions about data distribution
- **Instance-based**: Predictions based on similarity to training examples
- **Multi-class**: Naturally handles any number of classes
- **Interpretable**: Predictions can be explained by examining nearest neighbors

## How kNN Works

### Algorithm Steps

For a new data point **x**:

1. **Compute distances** to all training examples
2. **Select k nearest** neighbors (smallest distances)
3. **Vote for class**: Majority class among k neighbors
4. **Return prediction**: Most frequent class (or weighted vote)

### Mathematical Formulation

Given:
- Training set: **X** = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
- New point: **x**
- Number of neighbors: **k**
- Distance metric: **d**(x, xᵢ)

**Prediction**:
```text
ŷ = argmax_c Σ_{i∈N_k(x)} w_i · 𝟙[y_i = c]

where:
  N_k(x) = k nearest neighbors of x
  w_i = weight of neighbor i
  𝟙[·] = indicator function (1 if true, 0 if false)
  c = class label
```

### Distance Metrics

kNN requires a distance metric to measure similarity between data points.

#### Euclidean Distance (L2 norm)

Most common metric, measures straight-line distance:

```text
d(x, y) = √(Σ(x_i - y_i)²)
```

**Properties**:
- Sensitive to feature scales → **standardization required**
- Works well for continuous features
- Intuitive geometric interpretation

#### Manhattan Distance (L1 norm)

Sum of absolute differences, measures "city block" distance:

```text
d(x, y) = Σ|x_i - y_i|
```

**Properties**:
- Less sensitive to outliers than Euclidean
- Works well for high-dimensional data
- Useful when features represent counts

#### Minkowski Distance (Generalized L_p norm)

Generalization of Euclidean and Manhattan:

```text
d(x, y) = (Σ|x_i - y_i|^p)^(1/p)
```

**Special cases**:
- p = 1: Manhattan distance
- p = 2: Euclidean distance
- p → ∞: Chebyshev distance (maximum coordinate difference)

**Choosing p**:
- Lower p (1-2): Emphasizes all features equally
- Higher p (>2): Emphasizes dimensions with largest differences

## Choosing k

The choice of k critically affects model performance:

### Small k (k=1 to 3)

**Advantages**:
- Captures fine-grained decision boundaries
- Low bias

**Disadvantages**:
- High variance (overfitting)
- Sensitive to noise and outliers
- Unstable predictions

### Large k (k=7 to 20+)

**Advantages**:
- Smooth decision boundaries
- Low variance
- Robust to noise

**Disadvantages**:
- High bias (underfitting)
- May blur class boundaries
- Computational cost increases

### Selecting k

**Methods**:
1. **Cross-validation**: Try k ∈ {1, 3, 5, 7, 9, ...} and select best validation accuracy
2. **Rule of thumb**: k ≈ √n (where n = training set size)
3. **Odd k**: Use odd numbers for binary classification to avoid ties
4. **Domain knowledge**: Small k for fine distinctions, large k for noisy data

**Typical range**: k ∈ [3, 10] works well for most problems.

## Weighted vs Uniform Voting

### Uniform Voting (Majority Vote)

All k neighbors contribute equally:

```text
ŷ = argmax_c |{i ∈ N_k(x) : y_i = c}|
```

**Use when**:
- Neighbors are roughly equidistant
- Simplicity preferred
- Small k

### Weighted Voting (Inverse Distance Weighting)

Closer neighbors have more influence:

```text
w_i = 1 / d(x, x_i)   (or 1 if d = 0)

ŷ = argmax_c Σ_{i∈N_k(x)} w_i · 𝟙[y_i = c]
```

**Advantages**:
- More intuitive: closer points matter more
- Reduces impact of distant outliers
- Better for large k

**Disadvantages**:
- More complex
- Can be dominated by very close points

**Recommendation**: Use weighted voting for k ≥ 5, uniform for k ≤ 3.

## Implementation in Aprender

### Basic Usage

```rust
use aprender::classification::{KNearestNeighbors, DistanceMetric};
use aprender::primitives::Matrix;

// Load data
let x_train = Matrix::from_vec(100, 4, train_data)?;
let y_train = vec![0, 1, 0, 1, ...]; // Class labels

// Create and train kNN
let mut knn = KNearestNeighbors::new(5);
knn.fit(&x_train, &y_train)?;

// Make predictions
let x_test = Matrix::from_vec(20, 4, test_data)?;
let predictions = knn.predict(&x_test)?;
```

### Builder Pattern

Configure kNN with fluent API:

```rust
let mut knn = KNearestNeighbors::new(5)
    .with_metric(DistanceMetric::Manhattan)
    .with_weights(true);  // Enable weighted voting

knn.fit(&x_train, &y_train)?;
let predictions = knn.predict(&x_test)?;
```

### Probabilistic Predictions

Get class probability estimates:

```rust
let probabilities = knn.predict_proba(&x_test)?;

// probabilities[i][c] = estimated probability of class c for sample i
for i in 0..x_test.n_rows() {
    println!("Sample {}: P(class 0) = {:.2}%", i, probabilities[i][0] * 100.0);
}
```

**Interpretation**:
- Uniform voting: probabilities = fraction of k neighbors in each class
- Weighted voting: probabilities = weighted fraction (normalized by total weight)

### Distance Metrics

```rust
use aprender::classification::DistanceMetric;

// Euclidean (default)
let mut knn_euclidean = KNearestNeighbors::new(5)
    .with_metric(DistanceMetric::Euclidean);

// Manhattan
let mut knn_manhattan = KNearestNeighbors::new(5)
    .with_metric(DistanceMetric::Manhattan);

// Minkowski with p=3
let mut knn_minkowski = KNearestNeighbors::new(5)
    .with_metric(DistanceMetric::Minkowski(3.0));
```

## Time and Space Complexity

### Training (fit)

| Operation | Time | Space |
|-----------|------|-------|
| Store training data | O(1) | O(n · p) |

where n = training samples, p = features

**Key insight**: kNN has **no training cost** (lazy learning).

### Prediction (predict)

| Operation | Time | Space |
|-----------|------|-------|
| Distance computation | O(m · n · p) | O(n) |
| Finding k nearest | O(m · n log k) | O(k) |
| Voting | O(m · k · c) | O(c) |
| **Total per sample** | **O(n · p + n log k)** | **O(n)** |
| **Total (m samples)** | **O(m · n · p)** | **O(m · n)** |

where:
- m = test samples
- n = training samples
- p = features
- k = neighbors
- c = classes

**Bottleneck**: Distance computation is O(n · p) per test sample.

### Scalability Challenges

**Large training sets** (n > 10,000):
- Prediction becomes very slow
- Every prediction requires n distance computations
- Solution: Use approximate nearest neighbors (ANN) algorithms

**High dimensions** (p > 100):
- "Curse of dimensionality": distances become meaningless
- All points become roughly equidistant
- Solution: Use dimensionality reduction (PCA) first

### Memory Usage

**Training**:
- X_train: 4n·p bytes (f32)
- y_train: 8n bytes (usize)
- **Total**: ~4(n·p + 2n) bytes

**Inference** (per sample):
- Distance array: 4n bytes
- Neighbor indices: 8k bytes
- **Total**: ~4n bytes per sample

**Example** (1000 samples, 10 features):
- Training storage: ~40 KB
- Inference (per sample): ~4 KB

## When to Use kNN

### Good Use Cases

✓ **Small to medium datasets** (n < 10,000)\
✓ **Low to medium dimensions** (p < 50)\
✓ **Non-linear decision boundaries** (captures local patterns)\
✓ **Multi-class problems** (naturally handles any number of classes)\
✓ **Interpretable predictions** (can show nearest neighbors as evidence)\
✓ **No training time available** (predictions can be made immediately)\
✓ **Online learning** (easy to add new training examples)

### When kNN Fails

✗ **Large datasets** (n > 100,000) → Prediction too slow\
✗ **High dimensions** (p > 100) → Curse of dimensionality\
✗ **Real-time requirements** → O(n) per prediction is prohibitive\
✗ **Unbalanced classes** → Majority class dominates voting\
✗ **Irrelevant features** → All features affect distance equally\
✗ **Memory constraints** → Must store entire training set

## Advantages and Disadvantages

### Advantages

1. **No training phase**: Instant model updates
2. **Non-parametric**: No assumptions about data distribution
3. **Naturally multi-class**: Handles 2+ classes without modification
4. **Adapts to local patterns**: Captures complex decision boundaries
5. **Interpretable**: Predictions explained by nearest neighbors
6. **Simple implementation**: Easy to understand and debug

### Disadvantages

1. **Slow predictions**: O(n) per test sample
2. **High memory**: Must store entire training set
3. **Curse of dimensionality**: Fails in high dimensions
4. **Feature scaling required**: Distances sensitive to scales
5. **Imbalanced classes**: Majority class bias
6. **Hyperparameter tuning**: k and distance metric selection

## Comparison with Other Classifiers

| Classifier | Training Time | Prediction Time | Memory | Interpretability |
|------------|---------------|-----------------|--------|------------------|
| **kNN** | O(1) | O(n · p) | High (O(n·p)) | High (neighbors) |
| Logistic Regression | O(n · p · iter) | O(p) | Low (O(p)) | High (coefficients) |
| Decision Tree | O(n · p · log n) | O(log n) | Medium (O(nodes)) | High (rules) |
| Random Forest | O(n · p · t · log n) | O(t · log n) | High (O(t·nodes)) | Medium (feature importance) |
| SVM | O(n² · p) to O(n³ · p) | O(SV · p) | Medium (O(SV·p)) | Low (kernel) |
| Neural Network | O(n · iter · layers) | O(layers) | Medium (O(params)) | Low (black box) |

**Legend**: n=samples, p=features, t=trees, SV=support vectors, iter=iterations

**kNN vs others**:
- **Fastest training** (no training at all)
- **Slowest prediction** (must compare to all training samples)
- **Highest memory** (stores entire dataset)
- **Good interpretability** (can show nearest neighbors)

## Practical Considerations

### 1. Feature Standardization

**Always standardize features before kNN**:

```rust
use aprender::preprocessing::StandardScaler;
use aprender::traits::Transformer;

let mut scaler = StandardScaler::new();
let x_train_scaled = scaler.fit_transform(&x_train)?;
let x_test_scaled = scaler.transform(&x_test)?;

let mut knn = KNearestNeighbors::new(5);
knn.fit(&x_train_scaled, &y_train)?;
let predictions = knn.predict(&x_test_scaled)?;
```

**Why?**
- Features with larger scales dominate distance
- Example: Age (0-100) vs Income ($0-$1M) → Income dominates
- Standardization ensures equal contribution

### 2. Handling Imbalanced Classes

**Problem**: Majority class dominates voting.

**Solutions**:
- Use weighted voting (gives more weight to closer neighbors)
- Undersample majority class
- Oversample minority class (SMOTE)
- Adjust class weights in voting

### 3. Feature Selection

**Problem**: Irrelevant features hurt distance computation.

**Solutions**:
- Remove low-variance features
- Use feature importance from tree-based models
- Apply PCA for dimensionality reduction
- Use distance metrics that weight features (Mahalanobis)

### 4. Hyperparameter Tuning

**k selection**:
```python
# Pseudocode (implement with cross-validation)
for k in [1, 3, 5, 7, 9, 11, 15, 20]:
    knn = KNN(k)
    score = cross_validate(knn, X, y)
    if score > best_score:
        best_k = k
```

**Distance metric selection**:
- Try Euclidean, Manhattan, Minkowski(p=3)
- Select based on validation accuracy

## Algorithm Details

### Distance Computation

Aprender implements optimized distance computation:

```rust
fn compute_distance(
    &self,
    x: &Matrix<f32>,
    i: usize,
    x_train: &Matrix<f32>,
    j: usize,
    n_features: usize,
) -> f32 {
    match self.metric {
        DistanceMetric::Euclidean => {
            let mut sum = 0.0;
            for k in 0..n_features {
                let diff = x.get(i, k) - x_train.get(j, k);
                sum += diff * diff;
            }
            sum.sqrt()
        }
        DistanceMetric::Manhattan => {
            let mut sum = 0.0;
            for k in 0..n_features {
                sum += (x.get(i, k) - x_train.get(j, k)).abs();
            }
            sum
        }
        DistanceMetric::Minkowski(p) => {
            let mut sum = 0.0;
            for k in 0..n_features {
                let diff = (x.get(i, k) - x_train.get(j, k)).abs();
                sum += diff.powf(p);
            }
            sum.powf(1.0 / p)
        }
    }
}
```

**Optimization opportunities**:
- SIMD vectorization for distance computation
- KD-trees or Ball-trees for faster neighbor search (O(log n))
- Approximate nearest neighbors (ANN) for very large datasets
- GPU acceleration for batch predictions

### Voting Strategies

**Uniform voting**:
```rust
fn majority_vote(&self, neighbors: &[(f32, usize)]) -> usize {
    let mut counts = HashMap::new();
    for (_dist, label) in neighbors {
        *counts.entry(*label).or_insert(0) += 1;
    }
    *counts.iter().max_by_key(|(_, &count)| count).unwrap().0
}
```

**Weighted voting**:
```rust
fn weighted_vote(&self, neighbors: &[(f32, usize)]) -> usize {
    let mut weights = HashMap::new();
    for (dist, label) in neighbors {
        let weight = if *dist < 1e-10 { 1.0 } else { 1.0 / dist };
        *weights.entry(*label).or_insert(0.0) += weight;
    }
    *weights.iter().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0
}
```

## Example: Iris Dataset

Complete example from `examples/knn_iris.rs`:

```rust
use aprender::classification::{KNearestNeighbors, DistanceMetric};
use aprender::primitives::Matrix;

// Load data
let (x_train, y_train, x_test, y_test) = load_iris_data()?;

// Compare different k values
for k in [1, 3, 5, 7, 9] {
    let mut knn = KNearestNeighbors::new(k);
    knn.fit(&x_train, &y_train)?;
    let predictions = knn.predict(&x_test)?;
    let accuracy = compute_accuracy(&predictions, &y_test);
    println!("k={}: Accuracy = {:.1}%", k, accuracy * 100.0);
}

// Best configuration: k=5 with weighted voting
let mut knn_best = KNearestNeighbors::new(5)
    .with_weights(true);
knn_best.fit(&x_train, &y_train)?;
let predictions = knn_best.predict(&x_test)?;
```

**Typical results**:
- k=1: 90% (overfitting risk)
- k=3: 90%
- k=5 (weighted): **90%** (best balance)
- k=7: 80% (underfitting starts)

## Further Reading

- **Foundations**: Cover, T. & Hart, P. "Nearest neighbor pattern classification" (1967)
- **Distance metrics**: Comprehensive survey of distance measures
- **Curse of dimensionality**: Beyer et al. "When is nearest neighbor meaningful?" (1999)
- **Approximate NN**: Locality-sensitive hashing (LSH), HNSW, FAISS
- **Weighted kNN**: Dudani, S.A. "The distance-weighted k-nearest-neighbor rule" (1976)

## API Reference

```rust
// Constructor
pub fn new(k: usize) -> Self

// Builder methods
pub fn with_metric(mut self, metric: DistanceMetric) -> Self
pub fn with_weights(mut self, weights: bool) -> Self

// Training
pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<(), &'static str>

// Prediction
pub fn predict(&self, x: &Matrix<f32>) -> Result<Vec<usize>, &'static str>
pub fn predict_proba(&self, x: &Matrix<f32>) -> Result<Vec<Vec<f32>>, &'static str>

// Distance metrics
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Minkowski(f32),  // p parameter
}
```

**See also**:
- `classification::KNearestNeighbors` - Implementation
- `classification::DistanceMetric` - Distance metrics
- `preprocessing::StandardScaler` - Always use before kNN
- `examples/knn_iris.rs` - Complete walkthrough
