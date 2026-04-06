# dropout-v1

**Version:** 1.0.0

Dropout kernel — stochastic regularization via random masking

## References

- Srivastava et al. (2014) Dropout: A Simple Way to Prevent Neural Networks from Overfitting

## Equations

### dropout_eval

$$
y = x
$$

**Domain:** $x in R^n$

**Codomain:** $y in R^n$

**Invariants:**

- $y_i = x_i for all i (identity in eval mode)$
- $No randomness applied during evaluation$

### dropout_train

$$
y = mask * x / (1 - p), where mask_i ~ Bernoulli(1 - p)
$$

**Domain:** $x in R^n, p in [0, 1)$

**Codomain:** $y in R^n$

**Invariants:**

- $E[y_i] = x_i (unbiased expectation via inverted dropout)$
- $y_i = 0 when mask_i = 0 (dropped units are exactly zero)$
- $y_i = x_i / (1 - p) when mask_i = 1 (surviving units scaled)$
- $Output shape equals input shape$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Eval mode is identity | $dropout_eval(x) = x for all x$ |
| 2 | bound | Train mode is unbiased | $E[dropout_train(x, p)] = x for all x, p in [0, 1)$ |
| 3 | invariant | Output shape preserved | $shape(dropout(x)) = shape(x) for both train and eval modes$ |
| 4 | bound | Drop probability in valid range | $p in [0, 1) — p = 1 would cause division by zero$ |

## Kernel Phases

1. **generate_mask**: Sample Bernoulli(1 - p) mask of same shape as input — *mask_i in {0, 1}, P(mask_i = 1) = 1 - p*
2. **apply_mask**: Element-wise multiply input by mask — *y_i = 0 when mask_i = 0*
3. **scale**: Divide by (1 - p) for inverted dropout — *y_i = x_i / (1 - p) when mask_i = 1*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| dropout | avx2 | `dropout_avx2` |
| dropout | ptx | `dropout_ptx` |
| dropout | scalar | `dropout_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-DO-001 | Eval identity | dropout_eval(x) = x for all x | Eval mode incorrectly applies masking or scaling |
| FALSIFY-DO-002 | Unbiased expectation | mean(dropout_train(x, p)) converges to x over 10000 trials | Scale factor not 1/(1-p) or mask probability incorrect |
| FALSIFY-DO-003 | Shape preservation | shape(dropout(x)) = shape(x) for dims 1..128 | Buffer allocation does not match input dimensions |
| FALSIFY-DO-004 | Probability boundary | p = 0.0 yields identity, p near 1.0 drops almost all, p = 1.0 panics or errors | Missing guard for p = 1.0 causing division by zero |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-DO-001 | DO-INV-001 | 8 | stub_float |
| KANI-DO-002 | DO-BND-001 | 4 | stub_float |
| KANI-DROPOU-003 | Eval mode is identity | 8 | exhaustive |
| KANI-DROPOU-004 | Train mode is unbiased | 8 | exhaustive |
| KANI-DROPOU-005 | Output shape preserved | 8 | exhaustive |
| KANI-DROPOU-006 | Drop probability in valid range | 8 | exhaustive |

## QA Gate

**Dropout Contract** (F-DO-001)

Stochastic dropout regularization quality gate

**Checks:** eval_identity, unbiased_expectation, shape_preservation, probability_range

**Pass criteria:** All 4 falsification tests pass + Kani harnesses verify

