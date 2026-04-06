# svm-v1

**Version:** 1.0.0

Support Vector Machine — linear binary classification with hinge loss

## References

- Cortes & Vapnik (1995) Support-Vector Networks
- Hastie, Tibshirani, Friedman (2009) ESL, §12

## Equations

### decision_function

$$
f(x) = w·x + b
$$

**Domain:** $x \in \mathbb{R}^d, fitted w \in \mathbb{R}^d, b \in \mathbb{R}$

**Codomain:** $f \in \mathbb{R}$

**Invariants:**

- $sign(f(x)) determines classification$
- $Deterministic for same input$

### hinge_loss

```
L = max(0, 1 - y_i(w·x_i + b))
```

**Domain:** $w \in \mathbb{R}^d, x_i \in \mathbb{R}^d, y_i \in {-1, +1}, b \in \mathbb{R}$

**Codomain:** $L \in [0, ∞)$

**Invariants:**

- $L \geq 0 (non-negative by construction)$
- `L = 0 when y_i(w·x_i + b) ≥ 1 (correct with margin)`

### margin

$$
margin = 2 / ||w||
$$

**Domain:** $w \in \mathbb{R}^d, ||w|| > 0$

**Codomain:** $margin > 0$

**Invariants:**

- $margin > 0 for fitted model$
- $Larger margin \to better generalization (SRM principle)$

### svm_predict

$$
ŷ = sign(w·x + b), mapped to {0, 1}
$$

**Domain:** $x \in \mathbb{R}^d, fitted model$

**Codomain:** $ŷ \in {0, 1}$

**Invariants:**

- $Prediction \in {0, 1} (binary only)$
- $Prediction is deterministic$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Hinge loss non-negative | $L \geq 0 for all inputs$ |
| 2 | invariant | Binary prediction | $predict(x) \in {0, 1} for all x$ |
| 3 | invariant | Prediction deterministic | $predict(x) = predict(x) for all x$ |
| 4 | invariant | Separable data perfect accuracy | $Linearly separable data ⟹ accuracy = 1.0 (given sufficient iterations)$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-SVM-001 | Binary prediction | All predictions ∈ {0, 1} | Prediction not mapped to {0, 1} |
| FALSIFY-SVM-002 | Prediction deterministic | predict(X) = predict(X) | Non-deterministic learning rate or state |
| FALSIFY-SVM-003 | Separable data accuracy | accuracy > 0.9 on well-separated 2D binary data | Convergence failure or wrong sign convention |
| FALSIFY-SVM-004 | Fit-predict consistency | Predictions only contain labels seen in training | Label mapping error |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-SVM-001 | SVM-BND-001 | 8 | stub_float |
| KANI-SVM-002 | SVM-INV-001 | 8 | stub_float |
| KANI-SVM_V1-003 | Hinge loss non-negative | 8 | exhaustive |
| KANI-SVM_V1-004 | Binary prediction | 8 | exhaustive |
| KANI-SVM_V1-005 | Prediction deterministic | 8 | exhaustive |
| KANI-SVM_V1-006 | Separable data perfect accuracy | 8 | exhaustive |

## QA Gate

**SVM Contract** (F-SVM-001)

SVM correctness quality gate

**Checks:** binary_prediction, prediction_deterministic, separable_accuracy, fit_predict_consistency

**Pass criteria:** All 4 falsification tests pass + Kani harnesses verify

