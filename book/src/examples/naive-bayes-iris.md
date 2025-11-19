# Case Study: Naive Bayes Iris

This case study demonstrates Gaussian Naive Bayes classification on the Iris dataset, achieving perfect 100% test accuracy and outperforming k-Nearest Neighbors.

## Running the Example

```bash
cargo run --example naive_bayes_iris
```

## Results Summary

**Test Accuracy**: 100% (10/10 correct predictions)

### Comparison with kNN

| Metric | Naive Bayes | kNN (k=5, weighted) |
|--------|-------------|---------------------|
| Accuracy | **100.0%** | 90.0% |
| Training Time | <1ms | <1ms (lazy) |
| Prediction Time | O(p) | O(n·p) per sample |
| Memory | O(c·p) | O(n·p) |

**Winner**: Naive Bayes (10% accuracy improvement, faster prediction)

## Probabilistic Predictions

```text
Sample  Predicted  Setosa  Versicolor  Virginica
──────────────────────────────────────────────────────
   0     Setosa       100.0%    0.0%       0.0%
   1     Setosa       100.0%    0.0%       0.0%
   2     Setosa       100.0%    0.0%       0.0%
   3     Versicolor   0.0%    100.0%       0.0%
   4     Versicolor   0.0%    100.0%       0.0%
```

**Perfect confidence** for all predictions - indicates well-separated classes.

## Per-Class Performance

| Species | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Setosa | 3/3 | 3 | 100.0% |
| Versicolor | 3/3 | 3 | 100.0% |
| Virginica | 4/4 | 4 | 100.0% |

All three species classified perfectly.

## Variance Smoothing Effect

| var_smoothing | Accuracy |
|---------------|----------|
| 1e-12 | 100.0% |
| 1e-9 (default) | 100.0% |
| 1e-6 | 100.0% |
| 1e-3 | 100.0% |

**Robust**: Accuracy stable across wide range of smoothing parameters.

## Why Naive Bayes Excels Here

1. **Well-separated classes**: Iris species have distinct feature distributions
2. **Gaussian features**: Flower measurements approximately normal
3. **Small dataset**: Only 20 training samples - NB handles small data well
4. **Feature independence**: Violation of independence assumption doesn't hurt
5. **Probabilistic**: Full confidence scores for interpretability

## Implementation

```rust
use aprender::classification::GaussianNB;
use aprender::primitives::Matrix;

// Load data
let (x_train, y_train, x_test, y_test) = load_iris_data()?;

// Train
let mut nb = GaussianNB::new();
nb.fit(&x_train, &y_train)?;

// Predict
let predictions = nb.predict(&x_test)?;
let probabilities = nb.predict_proba(&x_test)?;

// Evaluate
let accuracy = compute_accuracy(&predictions, &y_test);
println!("Accuracy: {:.1}%", accuracy * 100.0);
```

## Key Insights

### Advantages Demonstrated
✓ **Instant training** (<1ms for 20 samples)\
✓ **100% accuracy** on test set\
✓ **Perfect confidence** scores\
✓ **Outperforms kNN** by 10%\
✓ **Simple implementation** (~240 lines)

### When Naive Bayes Wins
- Small datasets (<1000 samples)
- Well-separated classes
- Features approximately Gaussian
- Need probabilistic predictions
- Real-time prediction requirements

### When to Use kNN Instead
- Non-linear decision boundaries
- Local patterns important
- Don't assume Gaussian distribution
- Have abundant training data

## Related Examples

- [`examples/knn_iris.rs`](./knn-iris.md) - kNN comparison
- [`book/src/ml-fundamentals/naive-bayes.md`](../ml-fundamentals/naive-bayes.md) - Theory
