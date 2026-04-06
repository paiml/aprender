# layernorm-kernel-v1

**Version:** 1.0.0

LayerNorm kernel — layer normalization with affine transform

## References

- Ba et al. (2016) Layer Normalization
- Ioffe & Szegedy (2015) Batch Normalization

## Dependency Graph

```mermaid
graph LR
    encoder_forward_v1["encoder-forward-v1"] --> layernorm_kernel_v1["layernorm-kernel-v1"]
```

## Equations

### layernorm

$$
LN(x)_i = gamma_i * (x_i - mu) / \sqrt{sigma^2 + eps} + beta_i
$$

**Domain:** $x in R^d, gamma in R^d, beta in R^d, eps > 0$

**Codomain:** $LN(x) in R^d$

**Invariants:**

- $mean(LN(x)) = mean(beta) when gamma = 1 (centering)$
- $var(LN(x)) = 1 when gamma = 1, beta = 0 (standardization)$
- $LN is invariant to input shift: LN(x + c) = LN(x)$

### statistics

$$
mu = (1/d) * sum(x_i), sigma^2 = (1/d) * sum((x_i - mu)^2)
$$

**Domain:** $x in R^d, d >= 1$

**Codomain:** $mu in R, sigma^2 in R_>=0$

**Invariants:**

- $sigma^2 >= 0 (non-negative variance)$
- $sigma^2 = 0 iff x is constant$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Centering | $\|mean(LN(x)) - mean(beta)\| < eps when gamma = 1$ |
| 2 | invariant | Standardization | $\|var(LN(x)) - 1.0\| < eps when gamma = 1, beta = 0$ |
| 3 | bound | Denominator strictly positive | $\sqrt{sigma^2 + eps} > 0 when eps > 0$ |
| 4 | equivalence | SIMD matches scalar within ULP |  |
| 5 | idempotency | Idempotent under identity affine | $\|LN(LN(x)) - LN(x)\| < eps when gamma = 1, beta = 0$ |
| 6 | invariant | Shift invariance | $\|LN(x + c) - LN(x)\| < eps for any scalar c$ |

## Kernel Phases

1. **compute_mean**: Compute mu = mean(x) — *min(x) <= mu <= max(x)*
2. **compute_variance**: Compute sigma^2 = var(x) — *sigma^2 >= 0*
3. **normalize**: Compute (x - mu) / sqrt(sigma^2 + eps) — *denominator > 0*
4. **affine_transform**: Apply gamma * normalized + beta — *output dimension equals d*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| layernorm | avx2 | `layernorm_avx2` |
| layernorm | ptx | `layernorm_ptx` |
| layernorm | scalar | `layernorm_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-LN-001 | Centering | \|mean(LN(x)) - mean(beta)\| < 1e-5 with gamma=1 | Mean subtraction or affine transform incorrect |
| FALSIFY-LN-002 | Standardization | \|var(LN(x)) - 1.0\| < 1e-5 with gamma=1, beta=0 | Variance normalization incorrect |
| FALSIFY-LN-003 | Denominator safety | No NaN/Inf in output for any finite input when eps > 0 | Epsilon not added before sqrt, or overflow in variance |
| FALSIFY-LN-004 | SIMD equivalence | \|layernorm_avx2(x) - layernorm_scalar(x)\| < 8 ULP | SIMD reduction order differs from scalar |
| FALSIFY-LN-005 | Idempotency | \|LN(LN(x)) - LN(x)\| < 1e-5 with gamma=1, beta=0 | Second pass re-normalizes non-trivially |
| FALSIFY-LN-006 | Shift invariance | \|LN(x + c) - LN(x)\| < 1e-6 for random scalar c | Mean subtraction not canceling shift |
| FALSIFY-LN-007 | Boundary - constant input | LN([c,c,...,c]) = [beta_1,...,beta_d] when gamma=1 | Division by near-zero variance not handled |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-LN-001 | LN-INV-001 | 8 | stub_float |
| KANI-LN-002 | LN-INV-002 | 8 | stub_float |
| KANI-LN-003 | LN-BND-001 | 8 | stub_float |
| KANI-LAYERN-004 | Centering | 8 | exhaustive |
| KANI-LAYERN-005 | Standardization | 8 | exhaustive |
| KANI-LAYERN-006 | Denominator strictly positive | 8 | exhaustive |
| KANI-LAYERN-007 | SIMD matches scalar within ULP | 8 | exhaustive |
| KANI-LAYERN-008 | Idempotent under identity affine | 8 | exhaustive |
| KANI-LAYERN-009 | Shift invariance | 8 | exhaustive |

## QA Gate

**LayerNorm Contract** (F-LN-001)

Layer normalization with affine transform quality gate

**Checks:** centering, standardization, denominator_safety, simd_equivalence, idempotency

**Pass criteria:** All 7 falsification tests pass + Kani harnesses verify

