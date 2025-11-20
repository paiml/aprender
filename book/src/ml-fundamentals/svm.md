# Support Vector Machines (SVM)

Support Vector Machines are powerful supervised learning models for classification and regression. SVMs find the optimal hyperplane that maximizes the margin between classes, making them particularly effective for binary classification.

## Core Concepts

### Maximum-Margin Classifier

SVM seeks the decision boundary (hyperplane) that maximizes the **margin** - the distance to the nearest training examples from either class. These nearest examples are called **support vectors**.

```text
           ╲ │ ╱
            ╲│╱     Class 1 (⊕)
    ─────────●───────  ← decision boundary
            ╱│╲
           ╱ │ ╲    Class 0 (⊖)
         margin
```

The optimal hyperplane is defined by:
```text
w·x + b = 0
```

Where:
- **w**: weight vector (normal to hyperplane)
- **x**: feature vector
- **b**: bias term

### Decision Function

For a sample **x**, the decision function is:
```text
f(x) = w·x + b
```

Prediction:
```text
y = { 1  if f(x) ≥ 0
    { 0  if f(x) < 0
```

The magnitude |f(x)| represents confidence - larger values indicate samples farther from the boundary.

## Linear SVM Optimization

### Primal Problem

SVM minimizes the objective:
```text
min  (1/2)||w||² + C Σᵢ ξᵢ
w,b,ξ

subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0
```

Where:
- **||w||²**: Maximizes margin (1/||w||)
- **C**: Regularization parameter
- **ξᵢ**: Slack variables (allow soft margins)

### Hinge Loss Formulation

Equivalently, minimize:
```text
min  λ||w||² + (1/n) Σᵢ max(0, 1 - yᵢ(w·xᵢ + b))
```

Where **λ = 1/(2nC)** controls regularization strength.

The **hinge loss** is:
```text
L(y, f(x)) = max(0, 1 - y·f(x))
```

This penalizes:
- Misclassified samples: y·f(x) < 0
- Correctly classified within margin: 0 ≤ y·f(x) < 1
- Correctly classified outside margin: y·f(x) ≥ 1 (zero loss)

## Training Algorithm: Subgradient Descent

Linear SVM can be trained efficiently using subgradient descent:

### Algorithm
```text
Initialize: w = 0, b = 0
For each epoch:
    For each sample (xᵢ, yᵢ):
        Compute margin: m = yᵢ(w·xᵢ + b)

        If m < 1 (within margin):
            w ← w - η(λw - yᵢxᵢ)
            b ← b + ηyᵢ
        Else (outside margin):
            w ← w - η(λw)

    Check convergence
```

### Learning Rate Decay
Use decreasing learning rate:
```text
η(t) = η₀ / (1 + t·α)
```

This ensures convergence to optimal solution.

## Regularization Parameter C

**C** controls the trade-off between margin size and training error:

### Small C (e.g., 0.01 - 0.1)
- **Large margin**: More regularization
- **Simpler model**: Ignores some training errors
- **Better generalization**: Less overfitting
- **Use when**: Noisy data, overlapping classes

### Large C (e.g., 10 - 100)
- **Small margin**: Less regularization
- **Complex model**: Fits training data closely
- **Risk of overfitting**: Sensitive to noise
- **Use when**: Clean data, well-separated classes

### Default C = 1.0
Balanced trade-off suitable for most problems.

## Comparison with Other Classifiers

| Aspect | SVM | Logistic Regression | Naive Bayes |
|--------|-----|---------------------|-------------|
| **Loss** | Hinge | Log-loss | Bayes' theorem |
| **Decision** | Margin-based | Probability | Probability |
| **Training** | O(n²p) - O(n³p) | O(n·p·iters) | O(n·p) |
| **Prediction** | O(p) | O(p) | O(p·c) |
| **Regularization** | C parameter | L1/L2 | Var smoothing |
| **Outliers** | Robust (soft margin) | Sensitive | Robust |

## Implementation in Aprender

```rust,ignore
use aprender::classification::LinearSVM;
use aprender::primitives::Matrix;

// Create and train
let mut svm = LinearSVM::new()
    .with_c(1.0)              // Regularization
    .with_learning_rate(0.1)  // Step size
    .with_max_iter(1000);     // Convergence

svm.fit(&x_train, &y_train)?;

// Predict
let predictions = svm.predict(&x_test)?;

// Get decision values
let decisions = svm.decision_function(&x_test)?;
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Training | O(n·p·iters) | O(p) |
| Prediction | O(m·p) | O(m) |

Where: n=train samples, p=features, m=test samples, iters=epochs

## Advantages

✓ **Maximum margin**: Optimal decision boundary\
✓ **Robust to outliers** with soft margins (C parameter)\
✓ **Convex optimization**: Guaranteed convergence\
✓ **Fast prediction**: O(p) per sample\
✓ **Effective in high dimensions**: p >> n\
✓ **Kernel trick**: Can handle non-linear boundaries

## Disadvantages

✗ **Binary classification** only (use One-vs-Rest for multi-class)\
✗ **Slower training** than Naive Bayes\
✗ **Hyperparameter tuning**: C requires validation\
✗ **No probabilistic output** (decision values only)\
✗ **Linear boundaries**: Need kernels for non-linear problems

## When to Use

✓ Binary classification with clear separation\
✓ High-dimensional data (text, images)\
✓ Need robust classifier (outliers present)\
✓ Want interpretable decision function\
✓ Have labeled data (<10K samples for linear)

## Extensions

### Kernel SVM
Map data to higher dimensions using kernel functions:
- **Linear**: K(x, x') = x·x'
- **RBF (Gaussian)**: K(x, x') = exp(-γ||x - x'||²)
- **Polynomial**: K(x, x') = (x·x' + c)ᵈ

### Multi-Class SVM
- **One-vs-Rest**: Train C binary classifiers
- **One-vs-One**: Train C(C-1)/2 pairwise classifiers

### Support Vector Regression (SVR)
Use ε-insensitive loss for regression tasks.

## Example Results

On binary Iris (Setosa vs Versicolor):
- **Training time**: <10ms (subgradient descent)
- **Test accuracy**: 100%
- **Comparison**: Matches Naive Bayes and kNN
- **Robustness**: Stable across C ∈ [0.1, 100]

See `examples/svm_iris.rs` for complete example.

## API Reference

```rust,ignore
// Constructor
pub fn new() -> Self

// Builder
pub fn with_c(mut self, c: f32) -> Self
pub fn with_learning_rate(mut self, learning_rate: f32) -> Self
pub fn with_max_iter(mut self, max_iter: usize) -> Self
pub fn with_tolerance(mut self, tol: f32) -> Self

// Training
pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<(), &'static str>

// Prediction
pub fn predict(&self, x: &Matrix<f32>) -> Result<Vec<usize>, &'static str>
pub fn decision_function(&self, x: &Matrix<f32>) -> Result<Vec<f32>, &'static str>
```

## Further Reading

- **Original Paper**: Vapnik, V. (1995). The Nature of Statistical Learning Theory
- **Tutorial**: Burges, C. (1998). A Tutorial on Support Vector Machines
- **SMO Algorithm**: Platt, J. (1998). Sequential Minimal Optimization
