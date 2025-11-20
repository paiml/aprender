# Regression Metrics Theory

<!-- DOC_STATUS_START -->
**Chapter Status**: ✅ 100% Working (All metrics verified)

| Status | Count | Examples |
|--------|-------|----------|
| ✅ Working | 4 | All metrics tested in src/metrics/mod.rs |
| ⏳ In Progress | 0 | - |
| ⬜ Not Implemented | 0 | - |

*Last tested: 2025-11-19*
*Aprender version: 0.3.0*
*Test file: src/metrics/mod.rs tests*
<!-- DOC_STATUS_END -->

---

## Overview

Regression metrics measure how well a model predicts continuous values. Choosing the right metric is critical—it defines what "good" means for your model.

**Key Metrics**:
- **R² (R-squared)**: Proportion of variance explained (0-1, higher better)
- **MSE (Mean Squared Error)**: Average squared prediction error (0+, lower better)
- **RMSE (Root Mean Squared Error)**: MSE in original units (0+, lower better)
- **MAE (Mean Absolute Error)**: Average absolute error (0+, lower better)

**Why This Matters**:
"You can't improve what you don't measure." Metrics transform vague goals ("make better predictions") into concrete targets (R² > 0.8).

---

## Mathematical Foundation

### R² (Coefficient of Determination)

**Definition**:
```text
R² = 1 - (SS_res / SS_tot)

where:
SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
SS_tot = Σ(y_true - y_mean)²  (total sum of squares)
```

**Interpretation**:
- R² = 1.0: Perfect predictions (SS_res = 0)
- R² = 0.0: Model no better than predicting mean
- R² < 0.0: Model worse than mean (overfitting or bad fit)

**Key Insight**: R² measures variance explained. It answers: "What fraction of the target's variance does my model capture?"

### MSE (Mean Squared Error)

**Definition**:
```text
MSE = (1/n) Σ(y_true - y_pred)²
```

**Properties**:
- **Units**: Squared target units (e.g., dollars²)
- **Sensitivity**: Heavily penalizes large errors (quadratic)
- **Differentiable**: Good for gradient-based optimization

**When to Use**: When large errors are especially bad (e.g., financial predictions).

### RMSE (Root Mean Squared Error)

**Definition**:
```text
RMSE = √MSE = √[(1/n) Σ(y_true - y_pred)²]
```

**Advantage over MSE**: Same units as target (e.g., dollars, not dollars²)

**Interpretation**: "On average, predictions are off by X units"

### MAE (Mean Absolute Error)

**Definition**:
```text
MAE = (1/n) Σ|y_true - y_pred|
```

**Properties**:
- **Units**: Same as target
- **Robustness**: Less sensitive to outliers than MSE/RMSE
- **Interpretation**: Average prediction error magnitude

**When to Use**: When outliers shouldn't dominate the metric.

---

## Implementation in Aprender

### Example: All Metrics on Same Data

```rust,ignore
use aprender::metrics::{r_squared, mse, rmse, mae};
use aprender::primitives::Vector;

let y_true = Vector::from_vec(vec![3.0, -0.5, 2.0, 7.0]);
let y_pred = Vector::from_vec(vec![2.5, 0.0, 2.0, 8.0]);

// R² (higher is better, max = 1.0)
let r2 = r_squared(&y_true, &y_pred);
println!("R² = {:.3}", r2); // e.g., 0.948

// MSE (lower is better, min = 0.0)
let mse_val = mse(&y_true, &y_pred);
println!("MSE = {:.3}", mse_val); // e.g., 0.375

// RMSE (same units as target)
let rmse_val = rmse(&y_true, &y_pred);
println!("RMSE = {:.3}", rmse_val); // e.g., 0.612

// MAE (robust to outliers)
let mae_val = mae(&y_true, &y_pred);
println!("MAE = {:.3}", mae_val); // e.g., 0.500
```

**Test References**:
- `src/metrics/mod.rs::tests::test_r_squared`
- `src/metrics/mod.rs::tests::test_mse`
- `src/metrics/mod.rs::tests::test_rmse`
- `src/metrics/mod.rs::tests::test_mae`

---

## Choosing the Right Metric

### Decision Tree

```
Are large errors much worse than small errors?
├─ YES → Use MSE or RMSE (quadratic penalty)
└─ NO → Use MAE (linear penalty)

Do you need a unit-free measure of fit quality?
├─ YES → Use R² (0-1 scale)
└─ NO → Use RMSE or MAE (original units)

Are there outliers in your data?
├─ YES → Use MAE (robust) or Huber loss
└─ NO → Use RMSE (more sensitive)
```

### Comparison Table

| Metric | Range | Units | Outlier Sensitivity | Use Case |
|--------|-------|-------|---------------------|----------|
| **R²** | (-∞, 1] | Unitless | Medium | Overall fit quality |
| **MSE** | [0, ∞) | Squared | High | Optimization (differentiable) |
| **RMSE** | [0, ∞) | Original | High | Interpretable error magnitude |
| **MAE** | [0, ∞) | Original | Low | Robust to outliers |

---

## Practical Considerations

### R² Limitations

1. **Not Always 0-1**: R² can be negative if model is terrible
2. **Doesn't Catch Bias**: High R² doesn't mean unbiased predictions
3. **Sensitive to Range**: R² depends on target variance

**Example of R² Misleading**:
```
y_true = [10, 20, 30, 40, 50]
y_pred = [15, 25, 35, 45, 55]  # All predictions +5 (biased)

R² = 1.0 (perfect fit!)
But predictions are systematically wrong!
```

### MSE vs MAE Trade-off

**MSE Pros**:
- Differentiable everywhere (good for gradient descent)
- Heavily penalizes large errors
- Mathematically convenient (OLS minimizes MSE)

**MSE Cons**:
- Outliers dominate the metric
- Units are squared (hard to interpret)

**MAE Pros**:
- Robust to outliers
- Same units as target
- Intuitive interpretation

**MAE Cons**:
- Not differentiable at zero (complicates optimization)
- All errors weighted equally (may not reflect reality)

---

## Verification Through Tests

All metrics have comprehensive property tests:

**Property 1**: Perfect predictions → optimal metric value
- R² = 1.0
- MSE = RMSE = MAE = 0.0

**Property 2**: Constant predictions (mean) → baseline
- R² = 0.0

**Property 3**: Metrics are non-negative (except R²)
- MSE, RMSE, MAE ≥ 0.0

**Test Reference**: `src/metrics/mod.rs` has 10+ tests verifying these properties

---

## Real-World Application

**Example**: Evaluating Linear Regression

```rust,ignore
use aprender::linear_model::LinearRegression;
use aprender::metrics::{r_squared, rmse};
use aprender::traits::Estimator;

// Train model
let mut model = LinearRegression::new();
model.fit(&x_train, &y_train).unwrap();

// Evaluate on test set
let y_pred = model.predict(&x_test);
let r2 = r_squared(&y_test, &y_pred);
let error = rmse(&y_test, &y_pred);

println!("R² = {:.3}", r2);        // e.g., 0.874 (good fit)
println!("RMSE = {:.2}", error);   // e.g., 3.21 (avg error)

// Decision: R² > 0.8 and RMSE < 5.0 → Accept model
```

**Case Studies**:
- [Linear Regression](../examples/linear-regression.md) - Uses R² for evaluation
- [Cross-Validation](../examples/cross-validation.md) - Uses R² as CV score

---

## Further Reading

### Peer-Reviewed Papers

**Powers (2011)** - *Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation*
- **Relevance**: Comprehensive survey of evaluation metrics
- **Link**: [arXiv](https://arxiv.org/abs/2010.16061) (publicly accessible)
- **Key Insight**: No single metric is best—choose based on problem
- **Applied in**: `src/metrics/mod.rs`

### Related Chapters

- [Linear Regression Theory](./linear-regression.md) - OLS minimizes MSE
- [Cross-Validation Theory](./cross-validation.md) - Uses metrics for evaluation
- [Classification Metrics Theory](./classification-metrics.md) - For discrete targets

---

## Summary

**What You Learned**:
- ✅ R²: Variance explained (0-1, higher better)
- ✅ MSE: Average squared error (good for optimization)
- ✅ RMSE: MSE in original units (interpretable)
- ✅ MAE: Robust to outliers (linear penalty)
- ✅ Choose metric based on problem: outliers? units? optimization?

**Verification Guarantee**: All metrics extensively tested (10+ tests) in `src/metrics/mod.rs`. Property tests verify mathematical properties.

**Quick Reference**:
- **Overall fit**: R²
- **Optimization**: MSE
- **Interpretability**: RMSE or MAE
- **Robustness**: MAE

---

**Next Chapter**: [Classification Metrics Theory](./classification-metrics.md)

**Previous Chapter**: [Regularization Theory](./regularization.md)
