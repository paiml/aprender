# Case Study: Linear SVM Iris

This case study demonstrates Linear Support Vector Machine (SVM) classification on the Iris dataset, achieving perfect 100% test accuracy for binary classification.

## Running the Example

```bash
cargo run --example svm_iris
```

## Results Summary

**Test Accuracy**: 100% (6/6 correct predictions on binary Setosa vs Versicolor)

### Comparison with Other Classifiers

| Classifier | Accuracy | Training Time | Prediction |
|------------|----------|---------------|------------|
| Linear SVM | **100.0%** | <10ms (iterative) | O(p) |
| Naive Bayes | **100.0%** | <1ms (instant) | O(p·c) |
| kNN (k=5) | **100.0%** | <1ms (lazy) | O(n·p) |

**Winner**: All three achieve perfect accuracy! Choice depends on:
- **SVM**: Need margin-based decisions, robust to outliers
- **Naive Bayes**: Need probabilistic predictions, instant training
- **kNN**: Need non-parametric approach, local patterns

## Decision Function Values

```text
Sample  True  Predicted  Decision  Margin
───────────────────────────────────────────
   0      0      0       -1.195    1.195
   1      0      0       -1.111    1.111
   2      0      0       -1.105    1.105
   3      1      1       0.463    0.463
   4      1      1       1.305    1.305
```

**Interpretation**:
- **Negative decision**: Predicted class 0 (Setosa)
- **Positive decision**: Predicted class 1 (Versicolor)
- **Margin**: Distance from decision boundary (confidence)
- **All samples** correctly classified with good margins

## Regularization Effect (C Parameter)

| C Value | Accuracy | Behavior |
|---------|----------|----------|
| 0.01 | 50.0% | Over-regularized (too simple) |
| 0.10 | 100.0% | Good regularization |
| 1.00 (default) | 100.0% | Balanced |
| 10.00 | 100.0% | Fits data closely |
| 100.00 | 100.0% | Minimal regularization |

**Insight**: C ∈ [0.1, 100] all achieve 100% accuracy, showing:
- **Robust**: Wide range of good C values
- **Well-separated**: Iris species have distinct features
- **Warning**: C=0.01 too restrictive (underfits)

## Per-Class Performance

| Species | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Setosa | 3/3 | 3 | 100.0% |
| Versicolor | 3/3 | 3 | 100.0% |

Both classes classified perfectly.

## Why SVM Excels Here

1. **Linearly separable**: Setosa and Versicolor well-separated in feature space
2. **Maximum margin**: SVM finds optimal decision boundary
3. **Robust**: Soft margin (C parameter) handles outliers
4. **Simple problem**: Binary classification easier than multi-class
5. **Clean data**: Iris dataset has low noise

## Implementation

```rust
use aprender::classification::LinearSVM;
use aprender::primitives::Matrix;

// Load binary data (Setosa vs Versicolor)
let (x_train, y_train, x_test, y_test) = load_binary_iris_data()?;

// Train Linear SVM
let mut svm = LinearSVM::new()
    .with_c(1.0)              // Regularization
    .with_max_iter(1000)      // Convergence
    .with_learning_rate(0.1); // Step size

svm.fit(&x_train, &y_train)?;

// Predict
let predictions = svm.predict(&x_test)?;
let decisions = svm.decision_function(&x_test)?;

// Evaluate
let accuracy = compute_accuracy(&predictions, &y_test);
println!("Accuracy: {:.1}%", accuracy * 100.0);
```

## Key Insights

### Advantages Demonstrated
✓ **100% accuracy** on test set\
✓ **Fast prediction** (O(p) per sample)\
✓ **Robust regularization** (wide C range works)\
✓ **Maximum margin** decision boundary\
✓ **Interpretable** decision function values

### When Linear SVM Wins
- Linearly separable classes
- Need margin-based decisions
- Want robust outlier handling
- High-dimensional data (p >> n)
- Binary classification problems

### When to Use Alternatives
- **Naive Bayes**: Need instant training, probabilistic output
- **kNN**: Non-linear boundaries, local patterns important
- **Logistic Regression**: Need calibrated probabilities
- **Kernel SVM**: Non-linear decision boundaries required

## Algorithm Details

### Training Process
1. **Initialize**: w = 0, b = 0
2. **Iterate**: Subgradient descent for 1000 epochs
3. **Update rule**:
   - If margin < 1: Update w and b (hinge loss)
   - Else: Only regularize w
4. **Converge**: When weight change < tolerance

### Optimization Objective
```text
min  λ||w||² + (1/n) Σᵢ max(0, 1 - yᵢ(w·xᵢ + b))
     ─────────   ──────────────────────────────
   regularization        hinge loss
```

### Hyperparameters
- **C = 1.0**: Regularization strength (balanced)
- **learning_rate = 0.1**: Step size for gradient descent
- **max_iter = 1000**: Maximum epochs (converges faster)
- **tol = 1e-4**: Convergence tolerance

## Performance Analysis

### Complexity
- **Training**: O(n·p·iters) = O(14 × 4 × 1000) ≈ 56K ops
- **Prediction**: O(m·p) = O(6 × 4) = 24 ops
- **Memory**: O(p) = O(4) for weight vector

### Training Time
- **Linear SVM**: <10ms (subgradient descent)
- **Naive Bayes**: <1ms (closed-form solution)
- **kNN**: <1ms (lazy learning, no training)

### Prediction Time
- **Linear SVM**: O(p) - Very fast, constant per sample
- **Naive Bayes**: O(p·c) - Fast, scales with classes
- **kNN**: O(n·p) - Slower, scales with training size

## Comparison: SVM vs Naive Bayes vs kNN

### Accuracy
All achieve 100% on this well-separated binary problem.

### Decision Mechanism
- **SVM**: Maximum margin hyperplane (w·x + b = 0)
- **Naive Bayes**: Bayes' theorem with Gaussian likelihoods
- **kNN**: Local majority vote from k neighbors

### Regularization
- **SVM**: C parameter (controls margin/complexity trade-off)
- **Naive Bayes**: Variance smoothing (prevents division by zero)
- **kNN**: k parameter (controls local region size)

### Output Type
- **SVM**: Decision values (signed distance from hyperplane)
- **Naive Bayes**: Probabilities (well-calibrated for independent features)
- **kNN**: Probabilities (vote proportions, less calibrated)

### Best Use Case
- **SVM**: High-dimensional, linearly separable, need margins
- **Naive Bayes**: Small data, need probabilities, instant training
- **kNN**: Non-linear, local patterns, non-parametric

## Related Examples

- [`examples/naive_bayes_iris.rs`](./naive-bayes-iris.md) - Gaussian Naive Bayes comparison
- [`examples/knn_iris.rs`](./knn-iris.md) - kNN comparison
- [`book/src/ml-fundamentals/svm.md`](../ml-fundamentals/svm.md) - SVM Theory

## Further Exploration

### Try Different C Values
```rust
for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] {
    let mut svm = LinearSVM::new().with_c(c);
    svm.fit(&x_train, &y_train)?;
    // Compare accuracy and margin sizes
}
```

### Visualize Decision Boundary
Plot the hyperplane w·x + b = 0 in 2D feature space (e.g., petal_length vs petal_width).

### Multi-Class Extension
Implement One-vs-Rest to handle all 3 Iris species:
```rust
// Train 3 binary classifiers:
// - Setosa vs (Versicolor, Virginica)
// - Versicolor vs (Setosa, Virginica)
// - Virginica vs (Setosa, Versicolor)
// Predict using argmax of decision functions
```

### Add Kernel Functions
Extend to non-linear boundaries with RBF kernel:
```text
K(x, x') = exp(-γ||x - x'||²)
```
