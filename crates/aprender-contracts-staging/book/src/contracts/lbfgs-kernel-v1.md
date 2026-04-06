# lbfgs-kernel-v1

**Version:** 1.0.0

L-BFGS kernel — limited-memory BFGS quasi-Newton optimizer

## References

- Nocedal (1980) Updating Quasi-Newton Matrices with Limited Storage
- Liu & Nocedal (1989) On the Limited Memory BFGS Method for Large Scale Optimization

## Equations

### line_search

$$
alpha = argmin_a f(x_k + a * d_k) subject to Wolfe conditions
$$

**Domain:** $x_k in R^d, d_k descent direction, f: R^d -> R$

**Codomain:** $alpha in R_>0$

**Invariants:**

- $Sufficient decrease: f(x + alpha*d) <= f(x) + c1*alpha*g^T*d$
- $Curvature condition: |g(x+alpha*d)^T*d| <= c2*|g^T*d|$

### secant_condition

$$
H_{k+1} * y_k = s_k (secant equation)
$$

**Domain:** $s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k$

**Codomain:** $—$

**Invariants:**

- $y_k^T * s_k > 0 (curvature condition)$
- $Ensures positive definiteness of approximate Hessian$

### two_loop_recursion

$$
H_k * g_k via two-loop recursion using m stored (s, y) pairs
$$

**Domain:** $g_k in R^d, s_i = x_{i+1} - x_i, y_i = g_{i+1} - g_i, m >= 1$

**Codomain:** $direction in R^d$

**Invariants:**

- $Direction is descent direction: g_k^T * direction < 0$
- $Secant condition: y_i^T * s_i > 0 for all stored pairs$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Descent direction | $g_k^T * H_k * g_k > 0 (direction has negative dot with gradient)$ |
| 2 | invariant | Curvature condition | $y_k^T * s_k > 0 for all stored pairs$ |
| 3 | bound | History buffer bounded | $Number of stored (s, y) pairs <= m$ |
| 4 | monotonicity | Objective decrease | $f(x_{k+1}) < f(x_k) when Wolfe conditions satisfied$ |
| 5 | equivalence | SIMD matches scalar within ULP |  |

## Kernel Phases

1. **two_loop_backward**: Backward pass: compute alpha_i = rho_i * s_i^T * q — *rho_i = 1 / (y_i^T * s_i) well-defined when curvature holds*
2. **initial_scaling**: Scale by H_0 = (y_k^T*s_k)/(y_k^T*y_k) * I — *Scaling factor positive*
3. **two_loop_forward**: Forward pass: correct direction using stored betas — *Final direction is descent direction*
4. **line_search**: Backtracking or strong Wolfe line search — *Step size satisfies Wolfe conditions*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| lbfgs_direction | avx2 | `lbfgs_direction_avx2` |
| lbfgs_direction | ptx | `lbfgs_ptx` |
| lbfgs_direction | scalar | `lbfgs_direction_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-LB-001 | Descent direction | g^T * direction < 0 for non-zero gradient | Two-loop recursion sign error or scaling incorrect |
| FALSIFY-LB-002 | Curvature condition | y^T * s > 0 for all accepted (s, y) pairs | Pair accepted without curvature check |
| FALSIFY-LB-003 | History bound | len(history) <= m after any number of iterations | Old pairs not evicted from buffer |
| FALSIFY-LB-004 | Objective decrease | f(x_{k+1}) < f(x_k) on Rosenbrock function | Line search not satisfying Wolfe conditions |
| FALSIFY-LB-005 | SIMD equivalence | \|direction_avx2(g, history) - direction_scalar(g, history)\| < 8 ULP | SIMD dot product accumulation differs |
| FALSIFY-LB-006 | Boundary - first iteration | With empty history, direction = -g (steepest descent) | Initial scaling or empty-buffer case not handled |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-LB-001 | LB-INV-001 | 4 | stub_float |
| KANI-LB-002 | LB-BND-001 | 8 | exhaustive |
| KANI-LBFGS_-003 | Descent direction | 8 | exhaustive |
| KANI-LBFGS_-004 | Curvature condition | 8 | exhaustive |
| KANI-LBFGS_-005 | History buffer bounded | 8 | stub_float |
| KANI-LBFGS_-006 | Objective decrease | 8 | exhaustive |
| KANI-LBFGS_-007 | SIMD matches scalar within ULP | 8 | exhaustive |

## QA Gate

**L-BFGS Contract** (F-LB-001)

Limited-memory BFGS quasi-Newton optimizer quality gate

**Checks:** descent_direction, curvature_condition, history_bound, objective_decrease

**Pass criteria:** All 6 falsification tests pass + Kani harnesses verify

