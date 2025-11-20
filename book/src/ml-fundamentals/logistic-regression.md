# Logistic Regression Theory

<!-- DOC_STATUS_START -->
**Chapter Status**: ✅ 100% Working (All examples verified)

| Status | Count | Examples |
|--------|-------|----------|
| ✅ Working | 5+ | All verified by tests + SafeTensors |
| ⏳ In Progress | 0 | - |
| ⬜ Not Implemented | 0 | - |

*Last tested: 2025-11-19*
*Aprender version: 0.3.0*
*Test file: src/classification/mod.rs tests + SafeTensors tests*
<!-- DOC_STATUS_END -->

---

## Overview

Logistic regression is the foundation of binary classification. Despite its name, it's a **classification** algorithm that predicts probabilities using the logistic (sigmoid) function.

**Key Concepts**:
- **Sigmoid function**: Maps any value to [0, 1] probability
- **Binary classification**: Predict class 0 or 1
- **Gradient descent**: Iterative optimization (no closed-form)

**Why This Matters**:
Logistic regression powers countless applications: spam detection, medical diagnosis, credit scoring. It's interpretable, fast, and surprisingly effective.

---

## Mathematical Foundation

### The Sigmoid Function

The **sigmoid** (logistic) function squashes any real number to [0, 1]:

```text
σ(z) = 1 / (1 + e^(-z))
```

**Properties**:
- σ(0) = 0.5 (decision boundary)
- σ(+∞) → 1 (high confidence for class 1)
- σ(-∞) → 0 (high confidence for class 0)

### Logistic Regression Model

For input **x** and coefficients **β**:

```text
P(y=1|x) = σ(β·x + intercept)
         = 1 / (1 + e^(-(β·x + intercept)))
```

**Decision Rule**: Predict class 1 if P(y=1|x) ≥ 0.5, else class 0

### Training: Gradient Descent

Unlike linear regression, there's **no closed-form solution**. We use gradient descent to minimize the **binary cross-entropy loss**:

```text
Loss = -[y log(p) + (1-y) log(1-p)]
```

Where p = σ(β·x + intercept) is the predicted probability.

**Test Reference**: Implementation uses gradient descent in `src/classification/mod.rs`

---

## Implementation in Aprender

### Example 1: Binary Classification

```rust,ignore
use aprender::classification::LogisticRegression;
use aprender::primitives::{Matrix, Vector};

// Binary classification data (linearly separable)
let x = Matrix::from_vec(4, 2, vec![
    1.0, 1.0,  // Class 0
    1.0, 2.0,  // Class 0
    3.0, 3.0,  // Class 1
    3.0, 4.0,  // Class 1
]).unwrap();
let y = Vector::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

// Train with gradient descent
let mut model = LogisticRegression::new()
    .with_learning_rate(0.1)
    .with_max_iter(1000)
    .with_tol(1e-4);

model.fit(&x, &y).unwrap();

// Predict probabilities
let x_test = Matrix::from_vec(1, 2, vec![2.0, 2.5]).unwrap();
let proba = model.predict_proba(&x_test);
println!("P(class=1) = {:.3}", proba[0]); // e.g., 0.612
```

**Test Reference**: `src/classification/mod.rs::tests::test_logistic_regression_fit`

### Example 2: Model Serialization (SafeTensors)

Logistic regression models can be saved and loaded:

```rust,ignore
// Save model
model.save_safetensors("model.safetensors").unwrap();

// Load model (in production environment)
let loaded = LogisticRegression::load_safetensors("model.safetensors").unwrap();

// Predictions match exactly
let proba_original = model.predict_proba(&x_test);
let proba_loaded = loaded.predict_proba(&x_test);
assert_eq!(proba_original[0], proba_loaded[0]); // Exact match
```

**Why This Matters**: SafeTensors format is compatible with HuggingFace, PyTorch, TensorFlow, enabling cross-platform ML pipelines.

**Test Reference**: `src/classification/mod.rs::tests::test_save_load_safetensors_roundtrip`

**Case Study**: See [Case Study: Logistic Regression](../examples/logistic-regression.md) for complete SafeTensors implementation (281 lines)

---

## Verification Through Tests

Logistic regression has comprehensive test coverage:

**Core Functionality Tests**:
- Fitting on linearly separable data
- Probability predictions in [0, 1]
- Decision boundary at 0.5 threshold

**SafeTensors Tests** (5 tests):
- Unfitted model error handling
- Save/load roundtrip
- Corrupted file handling
- Missing file error
- **Probability preservation** (critical for classification)

All tests passing ensures production readiness.

---

## Practical Considerations

### When to Use Logistic Regression

- ✅ **Good for**:
  - Binary classification (2 classes)
  - Interpretable coefficients (feature importance)
  - Probability estimates needed
  - Linearly separable data

- ❌ **Not good for**:
  - Non-linear decision boundaries (use kernels or neural nets)
  - Multi-class classification (use softmax regression)
  - Imbalanced classes without adjustment

### Performance Characteristics

- **Time Complexity**: O(n·m·iter) where iter ≈ 100-1000
- **Space Complexity**: O(n·m)
- **Convergence**: Usually fast (< 1000 iterations)

### Common Pitfalls

1. **Unscaled Features**:
   - **Problem**: Features with different scales slow convergence
   - **Solution**: Use StandardScaler before training

2. **Non-convergence**:
   - **Problem**: Learning rate too high → oscillation
   - **Solution**: Reduce learning_rate or increase max_iter

3. **Assuming Linearity**:
   - **Problem**: Non-linear boundaries → poor accuracy
   - **Solution**: Add polynomial features or use kernel methods

---

## Comparison with Alternatives

| Approach | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **Logistic Regression** | - Interpretable<br>- Fast training<br>- Probabilities | - Linear boundaries only<br>- Gradient descent needed | Interpretable binary classification |
| **SVM** | - Non-linear kernels<br>- Max-margin | - No probabilities<br>- Slow on large data | Non-linear boundaries |
| **Decision Trees** | - Non-linear<br>- No feature scaling | - Overfitting<br>- Unstable | Quick baseline |

---

## Real-World Application

**Case Study Reference**: See [Case Study: Logistic Regression](../examples/logistic-regression.md) for:
- Complete SafeTensors implementation (281 lines)
- RED-GREEN-REFACTOR workflow
- 5 comprehensive tests
- Production deployment example (aprender → realizar)

**Key Insight**: SafeTensors enables cross-platform ML. Train in Rust, deploy anywhere (Python, C++, WASM).

---

## Further Reading

### Peer-Reviewed Paper

**Cox (1958)** - *The Regression Analysis of Binary Sequences*
- **Relevance**: Original paper introducing logistic regression
- **Link**: [JSTOR](https://www.jstor.org/stable/2983890) (publicly accessible)
- **Key Contribution**: Maximum likelihood estimation for binary outcomes
- **Applied in**: `src/classification/mod.rs`

### Related Chapters

- [Linear Regression Theory](./linear-regression.md) - Similar but for continuous targets
- [Classification Metrics Theory](./classification-metrics.md) - Evaluating logistic regression
- [Gradient Descent Theory](./gradient-descent.md) - Optimization algorithm used
- [Case Study: Logistic Regression](../examples/logistic-regression.md) - **REQUIRED READING**

---

## Summary

**What You Learned**:
- ✅ Sigmoid function: σ(z) = 1/(1 + e^(-z))
- ✅ Binary classification via probability thresholding
- ✅ Gradient descent training (no closed-form)
- ✅ SafeTensors serialization for production

**Verification Guarantee**: All logistic regression code is extensively tested (10+ tests) including SafeTensors roundtrip. See case study for complete implementation.

**Test Summary**:
- 5+ core tests (fitting, predictions, probabilities)
- 5 SafeTensors tests (serialization, errors)
- 100% passing rate

---

**Next Chapter**: [Decision Trees Theory](./decision-trees.md)

**Previous Chapter**: [Regularization Theory](./regularization.md)

**REQUIRED**: Read [Case Study: Logistic Regression](../examples/logistic-regression.md) for SafeTensors implementation
