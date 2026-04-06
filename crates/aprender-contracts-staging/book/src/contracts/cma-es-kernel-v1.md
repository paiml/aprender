# cma-es-kernel-v1

**Version:** 1.0.0

CMA-ES kernel — covariance matrix adaptation evolution strategy

## References

- Hansen (2016) The CMA Evolution Strategy: A Tutorial
- Hansen & Ostermeier (2001) Completely Derandomized Self-Adaptation in Evolution Strategies

## Equations

### covariance_update

```
C_{t+1} = (1-c1-cmu)*C_t + c1*p_c*p_c^T + cmu*sum(w_i*(x_i-m)*(x_i-m)^T/sigma^2)
```

**Domain:** $C_t positive definite, c1 >= 0, cmu >= 0, c1+cmu <= 1$

**Codomain:** $C_{t+1} positive definite$

**Invariants:**

- $C remains symmetric positive definite$
- $Update is convex combination preserving positive definiteness$

### mean_update

$$
m_{t+1} = sum_{i=1}^{mu} w_i * x_{i:lambda}
$$

**Domain:** $x_{i:lambda} sorted by fitness, w_i > 0, sum(w_i) = 1$

**Codomain:** $m_{t+1} in R^d$

**Invariants:**

- $New mean is weighted average of best mu individuals$
- $Weights sum to 1 (convex combination)$

### sample

$$
x_i = m + sigma * N(0, C) for i = 1..lambda
$$

**Domain:** $m in R^d, sigma in R_>0, C in R^{d x d} positive definite, lambda >= 2$

**Codomain:** $x_i in R^d$

**Invariants:**

- $sigma > 0 (positive step size)$
- $C is symmetric positive definite$
- $Samples distributed as N(m, sigma^2 * C)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Step size positive | $sigma > 0 at every generation$ |
| 2 | invariant | Covariance positive definite | $eigenvalues(C) > 0 at every generation$ |
| 3 | invariant | Weights sum to 1 | $\|sum(w_i) - 1.0\| < eps for recombination weights$ |
| 4 | invariant | Covariance symmetry | $C = C^T at every generation$ |
| 5 | equivalence | SIMD matches scalar within ULP |  |

## Kernel Phases

1. **sample_population**: Generate lambda candidates from N(m, sigma^2*C) — *sigma > 0 and C positive definite*
2. **evaluate_sort**: Evaluate fitness and sort population — *Sorted by fitness value*
3. **update_mean**: Weighted recombination of mu best — *Weights sum to 1*
4. **update_paths**: Update evolution paths p_sigma and p_c — *Path lengths bounded*
5. **update_covariance**: Rank-mu and rank-one update of C — *C remains positive definite*
6. **update_step_size**: CSA step-size adaptation — *sigma > 0*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| cma_sample | avx2 | `cma_sample_avx2` |
| cma_sample | ptx | `cma_es_ptx` |
| cma_sample | scalar | `cma_sample_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-CMA-001 | Step size positivity | sigma > 0 after 1000 generations | Step-size adaptation drives sigma to zero or negative |
| FALSIFY-CMA-002 | Covariance positive definiteness | min(eigenvalues(C)) > 0 after 100 generations | Rank update introduces negative eigenvalues |
| FALSIFY-CMA-003 | Weight normalization | \|sum(w_i) - 1.0\| < 1e-10 | Weight computation formula incorrect |
| FALSIFY-CMA-004 | Covariance symmetry | \|C - C^T\| < 1e-10 at every generation | Asymmetric update in rank-one or rank-mu |
| FALSIFY-CMA-005 | SIMD equivalence | \|sample_avx2(m,sigma,C) - sample_scalar(m,sigma,C)\| < 8 ULP | SIMD Cholesky decomposition or matmul differs |
| FALSIFY-CMA-006 | Boundary - dimension 1 | CMA-ES reduces to (1+1)-ES behavior in 1D | Edge case in covariance matrix for scalar |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-CMA-001 | CMA-BND-001 | 4 | stub_float |
| KANI-CMA-002 | CMA-INV-001 | 8 | stub_float |
| KANI-CMA_ES-003 | Step size positive | 8 | exhaustive |
| KANI-CMA_ES-004 | Covariance positive definite | 8 | exhaustive |
| KANI-CMA_ES-005 | Weights sum to 1 | 8 | stub_float |
| KANI-CMA_ES-006 | Covariance symmetry | 8 | exhaustive |
| KANI-CMA_ES-007 | SIMD matches scalar within ULP | 8 | exhaustive |

## QA Gate

**CMA-ES Contract** (F-CMA-001)

Covariance matrix adaptation evolution strategy quality gate

**Checks:** step_size_positivity, covariance_validity, weight_normalization, simd_equivalence

**Pass criteria:** All 6 falsification tests pass + Kani harnesses verify

