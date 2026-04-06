# linear-models-v1

**Version:** 1.0.0

Linear models — OLS regression and logistic regression

## References

- Hastie, Tibshirani, Friedman (2009) Elements of Statistical Learning, §3-4
- Bishop (2006) Pattern Recognition and Machine Learning, §3-4

## Equations

### logistic_predict_proba

$$
P(y=1|x) = \sigma(x^T w + b) = 1/(1+\exp(-(x^T w + b)))
$$

**Domain:** $x \in \mathbb{R}^d, fitted w \in \mathbb{R}^d, b \in \mathbb{R}$

**Codomain:** $P \in (0, 1)$

**Invariants:**

- $Probability \in (0, 1) (sigmoid range)$
- $Monotone in x^T w (for fixed w)$
- $P(y=1) + P(y=0) = 1$

### ols_fit

$$
\beta = (X^T X)^{-1} X^T y
$$

**Domain:** $X \in \mathbb{R}^{n×d}, y \in \mathbb{R}ⁿ, n > d, rank(X) = d$

**Codomain:** $\beta \in \mathbb{R}^d, intercept \in \mathbb{R}$

**Invariants:**

- $Prediction: ŷ = X\beta + b$
- $Normal equations: X^T(y - X\beta) = 0$
- $R² \in (-∞, 1] on training data$

### ols_predict

$$
ŷ = X\beta + b
$$

**Domain:** $X \in \mathbb{R}^{m×d}, fitted \beta \in \mathbb{R}^d, b \in \mathbb{R}$

**Codomain:** $ŷ \in \mathbb{R}^m$

**Invariants:**

- $Prediction is linear: predict(\alpha x₁ + x₂) = \alpha·predict(x₁) + predict(x₂) - (\alpha-1)b$
- $Prediction is deterministic$

### r_squared_training

```
R² = 1 - SS_res/SS_tot
```

**Domain:** $y, ŷ \in \mathbb{R}ⁿ$

**Codomain:** $R² \in (-∞, 1]$

**Invariants:**

- $R² \leq 1 (upper bound)$
- $R² = 1 iff ŷ = y exactly$
- $OLS training R² \geq 0 (for model with intercept)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | OLS training R² non-negative | $R² \geq 0 for OLS with intercept on training data$ |
| 2 | invariant | Prediction deterministic | $predict(X) = predict(X) for all X$ |
| 3 | bound | Logistic probability bounded | $P(y=1\|x) \in (0, 1) for all finite x$ |
| 4 | invariant | Logistic probabilities sum to 1 | $P(y=0) + P(y=1) = 1$ |
| 5 | invariant | Perfect fit on collinear data | $y = X\beta_true + b ⟹ R² \approx 1 after fit$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-LM-001 | OLS training R² non-negative | R² ≥ 0 on training data for random linear regression | Intercept not included or normal equation numerics |
| FALSIFY-LM-002 | Prediction deterministic | predict(X) = predict(X) | Non-deterministic state |
| FALSIFY-LM-003 | Perfect fit on collinear data | R² ≈ 1 when y = 2*x + 3 + noise(0) | Solver numerical instability |
| FALSIFY-LM-004 | Logistic probability bounded | P(y=1\|x) ∈ (0, 1) for all finite x | Sigmoid overflow or underflow producing 0.0 or 1.0 |
| FALSIFY-LM-005 | Logistic probabilities sum to 1 | P(y=0) + P(y=1) = 1 for all predictions | Probability normalization error |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-LM-001 | LM-BND-001 | 8 | stub_float |
| KANI-LM-002 | LM-BND-002 | 8 | stub_float |
| KANI-LINEAR-003 | OLS training R² non-negative | 8 | exhaustive |
| KANI-LINEAR-004 | Prediction deterministic | 8 | exhaustive |
| KANI-LINEAR-005 | Logistic probability bounded | 8 | stub_float |
| KANI-LINEAR-006 | Logistic probabilities sum to 1 | 8 | stub_float |
| KANI-LINEAR-007 | Perfect fit on collinear data | 8 | exhaustive |

## QA Gate

**Linear Models Contract** (F-LM-001)

Linear model correctness quality gate

**Checks:** ols_r2_nonneg, prediction_deterministic, perfect_fit, logistic_prob_bounded, logistic_probs_sum_to_one

**Pass criteria:** All 5 falsification tests pass + Kani harnesses verify

