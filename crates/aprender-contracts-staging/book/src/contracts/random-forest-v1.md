# random-forest-v1

**Version:** 1.0.0

Random Forest -- bagged ensemble of decision trees with feature subsampling

## References

- Breiman (2001) Random Forests, Machine Learning 45(1)
- Hastie, Tibshirani, Friedman (2009) ESL, Ch. 15

## Equations

### bootstrap_sample

```
D_b = {(x_{i_j}, y_{i_j}) : j=1..n, i_j ~ Uniform(1,n)} (sample with replacement)
```

**Domain:** `D = {(x_i, y_i)}_{i=1}^n, n >= 1`

**Codomain:** $D_b with |D_b| = n (same size as original)$

**Invariants:**

- $|D_b| = n (bootstrap sample has same size as original)$
- $Each element of D_b drawn from D (no out-of-distribution samples)$
- $With fixed seed, bootstrap is deterministic$

### ensemble_size

$$
B = n_estimators (user-specified number of trees)
$$

**Domain:** $n_estimators >= 1$

**Codomain:** $forest contains exactly B trees$

**Invariants:**

- $Number of fitted trees equals n_estimators$
- $Each tree fitted on an independent bootstrap sample$

### majority_vote

$$
y_hat = argmax_c sum_{b=1}^{B} I(h_b(x) = c)
$$

**Domain:** $x in R^d, ensemble of B trees {h_1, ..., h_B}$

**Codomain:** $y_hat in {class labels}$

**Invariants:**

- $y_hat is one of the training labels$
- $Each tree contributes exactly one vote$
- $Ties broken deterministically$

### predict

```
y_hat_i = majority_vote(h_1(x_i), ..., h_B(x_i)) for classification
```

**Domain:** $X in R^{m x d}, fitted forest of B trees$

**Codomain:** $y_hat in {training labels}^m$

**Invariants:**

- $All predictions are training labels (closed over label set)$
- $Number of predictions equals number of input samples$
- $Deterministic with same seed$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Predictions in label range | $predict(x) in {labels seen in training} for all x$ |
| 2 | invariant | Deterministic with same seed | $predict(X, seed=s) = predict(X, seed=s) for all X$ |
| 3 | invariant | Ensemble size respected | $\|forest.trees\| = n_estimators$ |
| 4 | invariant | Prediction length matches input | $\|predict(X)\| = \|X\| (number of rows)$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-RF-001 | Predictions in label range | All predictions are labels seen during training | Tree leaf prediction not constrained to training labels |
| FALSIFY-RF-002 | Deterministic with seed | Same seed produces identical predictions | Random state not properly seeded or thread-dependent ordering |
| FALSIFY-RF-003 | Ensemble size | Forest contains exactly n_estimators trees | Tree construction loop off-by-one or early termination |
| FALSIFY-RF-004 | Prediction length | Number of predictions equals number of input samples | Prediction loop skipping or duplicating samples |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-RF-001 | RF-INV-001 | 8 | stub_float |
| KANI-RF-002 | RF-INV-002 | 8 | stub_float |
| KANI-RANDOM-003 | Predictions in label range | 8 | exhaustive |
| KANI-RANDOM-004 | Deterministic with same seed | 8 | exhaustive |
| KANI-RANDOM-005 | Ensemble size respected | 8 | exhaustive |
| KANI-RANDOM-006 | Prediction length matches input | 8 | exhaustive |

## QA Gate

**Random Forest Contract** (F-RF-001)

Random Forest correctness quality gate

**Checks:** predictions_in_label_range, deterministic_with_seed, ensemble_size, prediction_length

**Pass criteria:** All 4 falsification tests pass + Kani harnesses verify

