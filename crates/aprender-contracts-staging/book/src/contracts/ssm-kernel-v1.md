# ssm-kernel-v1

**Version:** 1.0.0

SSM kernel — selective state space model (Mamba)

## References

- Gu & Dao (2023) Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Gu et al. (2021) Efficiently Modeling Long Sequences with Structured State Spaces

## Equations

### selective_gate

$$
Delta_t = softplus(Linear(x_t)), B_t = Linear(x_t), C_t = Linear(x_t)
$$

**Domain:** $x_t in R^d$

**Codomain:** $Delta_t in R_>0, B_t in R^n, C_t in R^n$

**Invariants:**

- $Delta_t > 0 (softplus ensures positivity)$
- $Input-dependent selectivity: different inputs get different dynamics$

### ssm_discretize

$$
A_bar = \exp(Delta * A), B_bar = (Delta * A)^{-1} * (\exp(Delta * A) - I) * Delta * B
$$

**Domain:** $A in R^{n x n}, B in R^{n x 1}, Delta in R_>0$

**Codomain:** $A_bar in R^{n x n}, B_bar in R^{n x 1}$

**Invariants:**

- $A_bar is stable when eigenvalues of A have negative real parts$
- $Discretization reduces to Euler method as Delta -> 0$

### ssm_scan

$$
h_t = A_bar * h_{t-1} + B_bar * x_t, y_t = C * h_t
$$

**Domain:** $x in R^L, h_0 = 0, A_bar in R^{n x n}, B_bar in R^{n x 1}, C in R^{1 x n}$

**Codomain:** $y in R^L$

**Invariants:**

- $Linear recurrence: output is linear in input for fixed parameters$
- `Causal: y_t depends only on x_1..x_t`

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Causality | `y_t depends only on x_1..x_t, not x_{t+1}..x_L` |
| 2 | bound | Softplus positivity | $Delta_t = softplus(z) > 0 for all z$ |
| 3 | invariant | Scan linearity | $SSM(alpha*x + beta*z) = alpha*SSM(x) + beta*SSM(z) for fixed params$ |
| 4 | equivalence | Parallel scan matches sequential scan | $\|parallel_scan(x) - sequential_scan(x)\| < eps$ |
| 5 | equivalence | SIMD matches scalar within ULP |  |

## Kernel Phases

1. **selective_projection**: Compute input-dependent Delta, B, C via linear projections — *Delta > 0 via softplus*
2. **discretize**: Convert continuous A, B to discrete A_bar, B_bar using Delta — *A_bar stable when A has negative eigenvalues*
3. **parallel_scan**: Associative parallel prefix scan over sequence — *Equivalent to sequential recurrence*
4. **output_projection**: Compute y = C * h for each timestep — *Output dimension matches input*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| ssm_scan | avx2 | `ssm_scan_avx2` |
| ssm_scan | ptx | `ssm_ptx` |
| ssm_scan | scalar | `ssm_scan_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-SSM-001 | Causality | Changing x_{t+1} does not affect y_t | Non-causal dependency in scan implementation |
| FALSIFY-SSM-002 | Softplus positivity | softplus(z) > 0 for z in [-1000, 1000] | Numerical underflow in softplus |
| FALSIFY-SSM-003 | Scan linearity | \|SSM(a*x+b*z) - (a*SSM(x)+b*SSM(z))\| < 1e-5 | Non-linear operation in scan path |
| FALSIFY-SSM-004 | Parallel-sequential equivalence | \|parallel_scan(x) - seq_scan(x)\| < 1e-5 | Associative scan reduction incorrect |
| FALSIFY-SSM-005 | SIMD equivalence | \|ssm_scan_avx2(x) - ssm_scan_scalar(x)\| < 8 ULP | SIMD vectorized scan differs |
| FALSIFY-SSM-006 | Boundary - zero input | SSM(0) = 0 when h_0 = 0 | Initial state or bias not handled |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-SSM-001 | SSM-INV-001 | 4 | stub_float |
| KANI-SSM-002 | SSM-BND-001 | 8 | stub_float |
| KANI-SSM_KE-003 | Causality | 8 | exhaustive |
| KANI-SSM_KE-004 | Softplus positivity | 8 | exhaustive |
| KANI-SSM_KE-005 | Scan linearity | 8 | exhaustive |
| KANI-SSM_KE-006 | Parallel scan matches sequential scan | 8 | exhaustive |
| KANI-SSM_KE-007 | SIMD matches scalar within ULP | 8 | exhaustive |

## QA Gate

**SSM Contract** (F-SSM-001)

Selective state space model (Mamba) quality gate

**Checks:** causality, softplus_positivity, scan_equivalence, simd_equivalence

**Pass criteria:** All 6 falsification tests pass + Kani harnesses verify

