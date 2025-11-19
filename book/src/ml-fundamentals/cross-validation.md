# Cross-Validation Theory

<!-- DOC_STATUS_START -->
**Chapter Status**: ✅ 100% Working (All examples verified)

| Status | Count | Examples |
|--------|-------|----------|
| ✅ Working | 12+ | Case study has comprehensive tests |
| ⏳ In Progress | 0 | - |
| ⬜ Not Implemented | 0 | - |

*Last tested: 2025-11-19*
*Aprender version: 0.3.0*
*Test file: tests/integration.rs + src/model_selection/mod.rs tests*
<!-- DOC_STATUS_END -->

---

## Overview

Cross-validation estimates how well a model generalizes to unseen data by systematically testing on held-out portions of the training set. It's the gold standard for model evaluation.

**Key Concepts**:
- **K-Fold CV**: Split data into K parts, train on K-1, test on 1
- **Train/Test Split**: Simple holdout validation
- **Reproducibility**: Random seeds ensure consistent splits

**Why This Matters**:
Using training accuracy to evaluate a model is like grading your own exam. Cross-validation provides an honest estimate of real-world performance.

---

## Mathematical Foundation

### The K-Fold Algorithm

1. **Partition** data into K equal-sized folds: D₁, D₂, ..., Dₖ
2. **For each fold i**:
   - Train on D \ Dᵢ (all data except fold i)
   - Test on Dᵢ
   - Record score sᵢ
3. **Average** scores: CV_score = (1/K) Σ sᵢ

**Key Property**: Every data point is used for testing exactly once and training exactly K-1 times.

**Common K Values**:
- K=5: Standard choice (80% train, 20% test per fold)
- K=10: More thorough but slower
- K=n: Leave-One-Out CV (LOOCV) - expensive but low variance

---

## Implementation in Aprender

### Example 1: Train/Test Split

```rust
use aprender::model_selection::train_test_split;
use aprender::primitives::{Matrix, Vector};

let x = Matrix::from_vec(10, 2, vec![/*...*/]).unwrap();
let y = Vector::from_vec(vec![/*...*/]);

// 80% train, 20% test, reproducible with seed 42
let (x_train, x_test, y_train, y_test) =
    train_test_split(&x, &y, 0.2, Some(42)).unwrap();

assert_eq!(x_train.shape().0, 8);  // 80% of 10
assert_eq!(x_test.shape().0, 2);   // 20% of 10
```

**Test Reference**: `src/model_selection/mod.rs::tests::test_train_test_split_basic`

### Example 2: K-Fold Cross-Validation

```rust
use aprender::model_selection::{KFold, cross_validate};
use aprender::linear_model::LinearRegression;

let kfold = KFold::new(5)  // 5 folds
    .with_shuffle(true)     // Shuffle data
    .with_random_state(42); // Reproducible

let model = LinearRegression::new();
let result = cross_validate(&model, &x, &y, &kfold).unwrap();

println!("Mean score: {:.3}", result.mean());     // e.g., 0.874
println!("Std dev: {:.3}", result.std());         // e.g., 0.042
```

**Test Reference**: `src/model_selection/mod.rs::tests::test_cross_validate`

---

## Verification: Property Tests

Cross-validation has strong mathematical properties we can verify:

**Property 1**: Every sample appears in test set exactly once
**Property 2**: Folds are disjoint (no overlap)
**Property 3**: Union of all folds = complete dataset

These are verified in the comprehensive test suite. See **Case Study** for full property tests.

---

## Practical Considerations

### When to Use

- ✅ **Use K-Fold**:
  - Small/medium datasets (< 10,000 samples)
  - Need robust performance estimate
  - Hyperparameter tuning

- ✅ **Use Train/Test Split**:
  - Large datasets (> 100,000 samples) - K-Fold too slow
  - Quick evaluation needed
  - Final model assessment (after CV for hyperparameters)

### Common Pitfalls

1. **Data Leakage**: Fitting preprocessing (scaling, imputation) on full dataset before split
   - **Solution**: Fit on training fold only, apply to test fold

2. **Temporal Data**: Shuffling time series data breaks temporal order
   - **Solution**: Use time-series split (future work)

3. **Class Imbalance**: Random splits may create imbalanced folds
   - **Solution**: Use stratified K-Fold (future work)

---

## Real-World Application

**Case Study Reference**: See [Case Study: Cross-Validation](../examples/cross-validation.md) for **complete implementation** showing:
- Full RED-GREEN-REFACTOR workflow
- 12+ tests covering all edge cases
- Property tests proving correctness
- Integration with LinearRegression
- Reproducibility verification

**Key Takeaway**: The case study shows EXTREME TDD in action - every requirement becomes a test first.

---

## Further Reading

### Peer-Reviewed Paper

**Kohavi (1995)** - *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection*
- **Relevance**: Foundational paper proving K-Fold is unbiased estimator
- **Link**: [CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.24.6340) (publicly accessible)
- **Key Finding**: K=10 optimal for bias-variance tradeoff
- **Applied in**: `src/model_selection/mod.rs`

### Related Chapters

- [Linear Regression Theory](./linear-regression.md) - Model to evaluate with CV
- [Regression Metrics Theory](./regression-metrics.md) - Scores used in CV
- [Case Study: Cross-Validation](../examples/cross-validation.md) - **REQUIRED READING**

---

## Summary

**What You Learned**:
- ✅ K-Fold algorithm: train on K-1 folds, test on 1
- ✅ Train/test split for quick evaluation
- ✅ Reproducibility with random seeds
- ✅ When to use CV vs simple split

**Verification Guarantee**: All cross-validation code is extensively tested (12+ tests) as shown in the **Case Study**. Property tests verify mathematical correctness.

---

**Next Chapter**: [Gradient Descent Theory](./gradient-descent.md)

**Previous Chapter**: [Classification Metrics Theory](./classification-metrics.md)

**REQUIRED**: Read [Case Study: Cross-Validation](../examples/cross-validation.md) for complete EXTREME TDD implementation
