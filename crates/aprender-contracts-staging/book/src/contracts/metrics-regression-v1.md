# metrics-regression-v1

**Version:** 1.0.0

Regression metrics — error measurement for continuous predictions

## References

- Draper & Smith (1998) Applied Regression Analysis
- Hastie, Tibshirani & Friedman (2009) Elements of Statistical Learning

## Equations

### mae

$$
MAE = (1/n) \sum|yᵢ - ŷᵢ|
$$

**Domain:** $y, ŷ \in \mathbb{R}ⁿ, n \geq 1$

**Codomain:** $MAE \in [0, ∞)$

**Invariants:**

- $MAE \geq 0 (non-negativity from absolute value)$
- $MAE = 0 iff ŷᵢ = yᵢ for all i$
- $MAE \leq RMSE (Jensen's inequality)$

### mse

$$
MSE = (1/n) \sum(yᵢ - ŷᵢ)²
$$

**Domain:** $y, ŷ \in \mathbb{R}ⁿ, n \geq 1$

**Codomain:** $MSE \in [0, ∞)$

**Invariants:**

- $MSE \geq 0 (non-negativity from squared terms)$
- $MSE = 0 iff ŷᵢ = yᵢ for all i$
- $MSE(y, ŷ) = MSE(ŷ, y) (symmetry)$

### r_squared

$$
R² = 1 - \sum(yᵢ - ŷᵢ)² / \sum(yᵢ - ȳ)²
$$

**Domain:** $y, ŷ \in \mathbb{R}ⁿ, n \geq 1, Var(y) > 0$

**Codomain:** $R² \in (-∞, 1]$

**Invariants:**

- $R² \leq 1.0 (upper bound from Cauchy-Schwarz)$
- $R² = 1.0 iff ŷᵢ = yᵢ for all i (perfect fit)$
- $R² = 0.0 iff ŷᵢ = ȳ for all i (predict mean)$

### rmse

$$
RMSE = √MSE = √((1/n) \sum(yᵢ - ŷᵢ)²)
$$

**Domain:** $y, ŷ \in \mathbb{R}ⁿ, n \geq 1$

**Codomain:** $RMSE \in [0, ∞)$

**Invariants:**

- $RMSE \geq 0$
- $RMSE \geq MAE (Jensen's inequality)$
- $RMSE = 0 iff MSE = 0$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | R² upper bound | $R² \leq 1.0 for all y, ŷ with Var(y) > 0$ |
| 2 | bound | MSE non-negativity | $MSE \geq 0 for all y, ŷ$ |
| 3 | invariant | MAE-RMSE ordering (Jensen's inequality) | $MAE(y, ŷ) \leq RMSE(y, ŷ) for all y, ŷ$ |
| 4 | equivalence | Perfect prediction identity | $R² = 1 ∧ MSE = 0 ∧ MAE = 0 ∧ RMSE = 0 when ŷ = y$ |
| 5 | invariant | MSE symmetry | $MSE(y, ŷ) = MSE(ŷ, y)$ |
| 6 | bound | MAE non-negativity | $MAE \geq 0 for all y, ŷ$ |
| 7 | bound | RMSE non-negativity | $RMSE \geq 0 for all y, ŷ$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-RM-001 | R² upper bound | R² ≤ 1.0 for random y, ŷ ∈ [-1000, 1000]ⁿ | Division by zero or sign error in SS computation |
| FALSIFY-RM-002 | MSE non-negativity | MSE ≥ 0 for all inputs | Arithmetic overflow in squared difference |
| FALSIFY-RM-003 | MAE ≤ RMSE (Jensen's inequality) | MAE ≤ RMSE for random inputs | Implementation violates Jensen's inequality |
| FALSIFY-RM-004 | Perfect prediction identity | When ŷ = y: R²=1, MSE=0, MAE=0, RMSE=0 | Floating-point error accumulation |
| FALSIFY-RM-005 | MSE symmetry | MSE(y, ŷ) = MSE(ŷ, y) | Argument order dependence in implementation |
| FALSIFY-RM-006 | MAE non-negativity | MAE ≥ 0 for all y, ŷ ∈ [-1000, 1000]ⁿ | Sign error or underflow in absolute difference accumulation |
| FALSIFY-RM-007 | RMSE non-negativity | RMSE ≥ 0 for all y, ŷ ∈ [-1000, 1000]ⁿ | Negative value from sqrt or arithmetic overflow in squared sum |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-RM-001 | 1 | 4 | stub_float |
| KANI-RM-002 | 2 | 4 | stub_float |
| KANI-METRIC-003 | R² upper bound | 8 | stub_float |
| KANI-METRIC-004 | MSE non-negativity | 8 | exhaustive |
| KANI-METRIC-005 | MAE-RMSE ordering (Jensen's inequality) | 8 | exhaustive |
| KANI-METRIC-006 | Perfect prediction identity | 8 | exhaustive |
| KANI-METRIC-007 | MSE symmetry | 8 | exhaustive |
| KANI-METRIC-008 | MAE non-negativity | 8 | exhaustive |
| KANI-METRIC-009 | RMSE non-negativity | 8 | exhaustive |

## QA Gate

**Regression Metrics Contract** (F-RM-001)

Regression metric correctness quality gate

**Checks:** r_squared_upper_bound, mse_non_negativity, mae_rmse_ordering, perfect_prediction, mse_symmetry

**Pass criteria:** All 7 falsification tests pass

