# gbm-v1

**Version:** 1.0.0

Gradient Boosting Machine -- sequential ensemble with gradient descent in function space

## References

- Friedman (2001) Greedy Function Approximation: A Gradient Boosting Machine
- Hastie, Tibshirani, Friedman (2009) ESL, Ch. 10

## Equations

### gradient_boost

$$
F_m(x) = F_{m-1}(x) + nu * h_m(x)
$$

**Domain:** $F_{m-1}: R^d -> R (current model), h_m: R^d -> R (weak learner), nu in (0, 1] (learning rate)$

**Codomain:** $F_m: R^d -> R$

**Invariants:**

- $Ensemble is additive: F_M = F_0 + nu * sum h_m$
- $nu > 0 ensures each tree contributes in gradient direction$
- $F_m is deterministic given training data and hyperparameters$

### negative_gradient

```
r_{im} = -(dL/dF)|_{F=F_{m-1}(x_i)}
```

**Domain:** $L: loss function, F_{m-1}: current ensemble, x_i in R^d$

**Codomain:** $r_{im} in R (pseudo-residuals)$

**Invariants:**

- `For squared loss: r_{im} = y_i - F_{m-1}(x_i)`
- `For log-loss: r_{im} = y_i - sigma(F_{m-1}(x_i))`
- $Pseudo-residuals are finite for bounded F and bounded data$

### predict

$$
y_hat = sigma(F_M(x)) thresholded at 0.5 for classification
$$

**Domain:** $x in R^d, fitted ensemble F_M$

**Codomain:** $y_hat in {0, 1}$

**Invariants:**

- $Predictions are binary {0, 1} for classification$
- $Predictions are deterministic$
- $Ensemble output F_M(x) is finite$

### training_loss

```
L_m = (1/n) * sum L(y_i, F_m(x_i))
```

**Domain:** $training data, current ensemble F_m$

**Codomain:** $L_m >= 0$

**Invariants:**

- $Training loss is non-negative$
- $L_m <= L_{m-1} (loss non-increasing with more boosting rounds)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Predictions binary | $predict(x) in {0, 1} for all x$ |
| 2 | invariant | Predictions deterministic | $predict(x) = predict(x) for same fitted model$ |
| 3 | bound | Ensemble output finite | $\|F_M(x)\| < infinity for bounded x$ |
| 4 | invariant | Training loss non-increasing | $L_m <= L_{m-1} for each boosting round m$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-GBM-001 | Binary prediction | All predictions in {0, 1} | Sigmoid threshold or label mapping error |
| FALSIFY-GBM-002 | Prediction deterministic | predict(X) = predict(X) | Non-deterministic tree splitting or random state leak |
| FALSIFY-GBM-003 | Ensemble output finite | F_M(x) is finite for all bounded test inputs | Numerical overflow in additive ensemble accumulation |
| FALSIFY-GBM-004 | Fit-predict consistency | Predictions only contain labels seen in training | Label encoding error in multi-class or binary mapping |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-GBM-001 | GBM-INV-001 | 8 | stub_float |
| KANI-GBM-002 | GBM-BND-001 | 8 | stub_float |
| KANI-GBM_V1-003 | Predictions binary | 8 | exhaustive |
| KANI-GBM_V1-004 | Predictions deterministic | 8 | exhaustive |
| KANI-GBM_V1-005 | Ensemble output finite | 8 | exhaustive |
| KANI-GBM_V1-006 | Training loss non-increasing | 8 | exhaustive |

## QA Gate

**GBM Contract** (F-GBM-001)

Gradient Boosting Machine correctness quality gate

**Checks:** binary_prediction, prediction_deterministic, ensemble_output_finite, fit_predict_consistency

**Pass criteria:** All 4 falsification tests pass + Kani harnesses verify

