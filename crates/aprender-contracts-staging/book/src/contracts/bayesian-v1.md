# bayesian-v1

**Version:** 1.0.0

Bayesian inference -- conjugate prior updates and Bayesian Linear Regression

## References

- Gelman et al. (2013) Bayesian Data Analysis, 3rd ed.
- Murphy (2012) Machine Learning: A Probabilistic Perspective, Ch. 3,7

## Equations

### blr_predict

$$
y_hat = X * mu_post
$$

**Domain:** $X in R^{n x d}, mu_post in R^d (posterior mean of weights)$

**Codomain:** $y_hat in R^n$

**Invariants:**

- $Predictions are finite for bounded input$
- $Prediction length equals number of input samples$
- $Deterministic given same posterior and input$

### conjugate_update

$$
p(theta|data) proportional_to p(data|theta) * p(theta) = posterior proportional_to likelihood * prior
$$

**Domain:** $prior parameters (alpha, beta) > 0, observed data$

**Codomain:** $posterior parameters (alpha', beta') > 0$

**Invariants:**

- $Posterior is in the same family as the prior (conjugacy)$
- $Posterior parameters are deterministic given prior and data$

### posterior_predictive

```
p(y_new|X_new, data) = integral p(y_new|X_new, w) * p(w|data) dw
```

**Domain:** $X_new in R^{m x d}, fitted BLR model$

**Codomain:** $predictive distribution over R^m$

**Invariants:**

- $Predictive variance >= 0$
- $Predictive mean equals BLR point prediction$

### posterior_valid

$$
alpha' = alpha + n_successes, beta' = beta + n_failures (Beta-Binomial)
$$

**Domain:** $alpha, beta > 0, n_successes >= 0, n_failures >= 0$

**Codomain:** $alpha' > 0, beta' > 0$

**Invariants:**

- $alpha' > alpha (posterior concentration increases with successes)$
- $beta' > beta (posterior concentration increases with failures)$
- $alpha' > 0 and beta' > 0 always (positive parameters preserved)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Posterior parameters positive | $alpha' > 0 and beta' > 0 after any conjugate update$ |
| 2 | invariant | Predictions finite | `forall i: \|y_hat_i\| < infinity when \|\|X_i\|\| < infinity` |
| 3 | invariant | Prediction deterministic | $predict(X) = predict(X) for same posterior$ |
| 4 | invariant | Conjugacy preserved | $posterior family = prior family for conjugate models$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-BAYES-001 | Posterior positivity | alpha' > 0 and beta' > 0 after conjugate update | Conjugate update arithmetic error or underflow |
| FALSIFY-BAYES-002 | Prediction finiteness | All BLR predictions are finite | Posterior covariance inversion instability |
| FALSIFY-BAYES-003 | Prediction deterministic | predict(X) = predict(X) for same model | Random state leaking into point predictions |
| FALSIFY-BAYES-004 | Posterior concentration | More data leads to tighter posterior (variance decreases) | Posterior update not accumulating evidence correctly |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-BAYES-001 | BAYES-INV-001 | 8 | stub_float |
| KANI-BAYES-002 | BAYES-INV-002 | 8 | stub_float |
| KANI-BAYESI-003 | Posterior parameters positive | 8 | exhaustive |
| KANI-BAYESI-004 | Predictions finite | 8 | exhaustive |
| KANI-BAYESI-005 | Prediction deterministic | 8 | exhaustive |
| KANI-BAYESI-006 | Conjugacy preserved | 8 | exhaustive |

## QA Gate

**Bayesian Inference Contract** (F-BAYES-001)

Bayesian inference correctness quality gate

**Checks:** posterior_positivity, prediction_finiteness, prediction_deterministic, posterior_concentration

**Pass criteria:** All 4 falsification tests pass + Kani harnesses verify

