# Naive Bayes

Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem with the "naive" assumption of feature independence. Despite this strong assumption, Naive Bayes classifiers are remarkably effective in practice, especially for text classification.

## Bayes' Theorem

The foundation of Naive Bayes is Bayes' theorem:

```text
P(y|X) = P(X|y) * P(y) / P(X)
```

Where:
- **P(y|X)**: Posterior probability (probability of class y given features X)
- **P(X|y)**: Likelihood (probability of features X given class y)
- **P(y)**: Prior probability (probability of class y)
- **P(X)**: Evidence (probability of features X)

## The Naive Assumption

Naive Bayes assumes **conditional independence** between features:

```text
P(X|y) = P(x₁|y) * P(x₂|y) * ... * P(xₚ|y)
```

This simplifies computation dramatically, reducing from exponential to linear complexity.

## Gaussian Naive Bayes

Assumes features follow a Gaussian (normal) distribution within each class.

### Training

For each class c and feature i:
1. Compute mean: μᵢ,c = mean(xᵢ where y=c)
2. Compute variance: σ²ᵢ,c = var(xᵢ where y=c)
3. Compute prior: P(y=c) = count(y=c) / n

### Prediction

For each class c:
```text
log P(y=c|X) = log P(y=c) + Σᵢ log P(xᵢ|y=c)

where P(xᵢ|y=c) ~ N(μᵢ,c, σ²ᵢ,c) (Gaussian PDF)
```

Return class with highest posterior probability.

## Implementation in Aprender

```rust
use aprender::classification::GaussianNB;
use aprender::primitives::Matrix;

// Create and train
let mut nb = GaussianNB::new();
nb.fit(&x_train, &y_train)?;

// Predict
let predictions = nb.predict(&x_test)?;

// Get probabilities
let probabilities = nb.predict_proba(&x_test)?;
```

### Variance Smoothing

Adds small constant to variances to prevent numerical instability:

```rust
let nb = GaussianNB::new().with_var_smoothing(1e-9);
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Training | O(n·p) | O(c·p) |
| Prediction | O(m·p·c) | O(m·c) |

Where: n=samples, p=features, c=classes, m=test samples

## Advantages

✓ **Extremely fast** training and prediction\
✓ **Probabilistic** predictions with confidence scores\
✓ **Works with small datasets**\
✓ **Handles high-dimensional** data well\
✓ **Naturally handles** imbalanced classes via priors

## Disadvantages

✗ **Independence assumption** rarely holds in practice\
✗ **Gaussian assumption** may not fit data\
✗ **Cannot capture** feature interactions\
✗ **Poor probability estimates** (despite good classification)

## When to Use

✓ Text classification (spam detection, sentiment analysis)\
✓ Small datasets (<1000 samples)\
✓ High-dimensional data (p > n)\
✓ Baseline classifier (fast to implement and test)\
✓ Real-time prediction requirements

## Example Results

On Iris dataset:
- **Training time**: <1ms
- **Test accuracy**: 100% (30 samples)
- **Outperforms kNN**: 100% vs 90%

See `examples/naive_bayes_iris.rs` for complete example.

## API Reference

```rust
// Constructor
pub fn new() -> Self

// Builder
pub fn with_var_smoothing(mut self, var_smoothing: f32) -> Self

// Training
pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<(), &'static str>

// Prediction
pub fn predict(&self, x: &Matrix<f32>) -> Result<Vec<usize>, &'static str>
pub fn predict_proba(&self, x: &Matrix<f32>) -> Result<Vec<Vec<f32>>, &'static str>
```
