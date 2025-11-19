# Logistic Regression

📝 **This chapter demonstrates binary classification using Logistic Regression.**

## Overview

Logistic Regression is a fundamental classification algorithm that uses the sigmoid function to model the probability of binary outcomes. This case study demonstrates the RED-GREEN-REFACTOR cycle for implementing a production-quality classifier.

## RED Phase: Writing Failing Tests

Following EXTREME TDD principles, we begin by writing comprehensive tests before implementation:

```rust
#[test]
fn test_logistic_regression_fit_simple() {
    let x = Matrix::from_vec(4, 2, vec![...]).unwrap();
    let y = vec![0, 0, 1, 1];

    let mut model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(1000);

    let result = model.fit(&x, &y);
    assert!(result.is_ok());
}
```

**Test categories implemented:**
- Unit tests (12 tests)
- Property tests (4 tests)
- Doc tests (1 test)

## GREEN Phase: Minimal Implementation

The implementation includes:
- **Sigmoid activation**: σ(z) = 1 / (1 + e^(-z))
- **Binary cross-entropy loss** (implicit in gradient)
- **Gradient descent optimization**
- **Builder pattern API**

## REFACTOR Phase: Code Quality

**Optimizations applied:**
- Used `.enumerate()` instead of manual indexing
- Applied clippy suggestion for range contains
- Added comprehensive error validation

## Key Learning Points

1. **Mathematical correctness**: Sigmoid function ensures probabilities in [0, 1]
2. **API design**: Builder pattern for flexible configuration
3. **Property testing**: Invariants verified across random inputs
4. **Error handling**: Input validation prevents runtime panics

## Test Results

- **Total tests**: 514 passing
- **Coverage**: 100% for classification module
- **Mutation testing**: Builder pattern mutants caught
- **Property tests**: All 4 invariants hold

## Example Output

```
Training Accuracy: 100.0%
Test predictions:
  Feature1=2.50, Feature2=2.00 -> Class 0 (0.043 probability)
  Feature1=7.50, Feature2=8.00 -> Class 1 (0.990 probability)
```

**See also:**
- [What is EXTREME TDD?](../methodology/what-is-extreme-tdd.md)
- [Property-Based Testing](../advanced-testing/property-based-testing.md)
- [Builder Pattern](../best-practices/builder-pattern.md)
