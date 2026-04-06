# optimization-v1

**Version:** 1.0.0

Optimization -- Conjugate Gradient with Fletcher-Reeves and Wolfe line search

## References

- Nocedal & Wright (2006) Numerical Optimization, Ch. 5
- Fletcher & Reeves (1964) Function Minimization by Conjugate Gradients, Computer Journal

## Equations

### cg_minimize

$$
d_k = -g_k + beta_k * d_{k-1}, beta_k = ||g_k||^2 / ||g_{k-1}||^2 (Fletcher-Reeves)
$$

**Domain:** $f: R^n -> R differentiable, x_0 in R^n initial point$

**Codomain:** $x* in R^n (approximate minimizer)$

**Invariants:**

- `d_k is a descent direction: g_k^T d_k < 0 (when g_k != 0)`
- $beta_k >= 0 (Fletcher-Reeves always non-negative)$
- $Reduces to steepest descent when beta_k = 0$

### convergence

$$
||g_k|| -> 0 as k -> infinity (for smooth convex f)
$$

**Domain:** $f smooth convex, bounded below$

**Codomain:** $||g_k|| monotonically decreasing toward 0$

**Invariants:**

- $f(x_{k+1}) <= f(x_k) (monotone decrease with exact Wolfe)$
- $Iterates remain finite: ||x_k|| < infinity$
- $Gradient norm decreases on average$

### line_search

$$
alpha_k = argmin_{alpha > 0} f(x_k + alpha * d_k), subject to Wolfe conditions
$$

**Domain:** $x_k in R^n, d_k descent direction, c1 in (0, 0.5), c2 in (c1, 1)$

**Codomain:** $alpha_k > 0$

**Invariants:**

- `Sufficient decrease (Armijo): f(x_k + alpha*d_k) <= f(x_k) + c1*alpha*g_k^T*d_k`
- `Curvature condition: g(x_k + alpha*d_k)^T*d_k >= c2*g_k^T*d_k`
- $alpha_k > 0 (positive step size)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Monotone function decrease | $f(x_{k+1}) <= f(x_k) for all k (with Wolfe line search)$ |
| 2 | bound | Finite iterates | $\|\|x_k\|\| < infinity for all k$ |
| 3 | bound | Positive step size | $alpha_k > 0 for all k$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-OPT-001 | Monotone decrease | f(x_{k+1}) <= f(x_k) for each iteration | Line search not satisfying Armijo condition |
| FALSIFY-OPT-002 | Finite iterates | All iterates x_k are finite (no NaN, no Inf) | Step size too large or gradient explosion |
| FALSIFY-OPT-003 | Gradient norm convergence | \|\|g_final\|\| < \|\|g_initial\|\| after optimization on convex function | Conjugate direction computation error or beta overflow |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-OPT-001 | OPT-INV-001 | 8 | stub_float |
| KANI-OPT-002 | OPT-BND-001 | 8 | stub_float |
| KANI-OPTIMI-003 | Monotone function decrease | 8 | exhaustive |
| KANI-OPTIMI-004 | Finite iterates | 8 | exhaustive |
| KANI-OPTIMI-005 | Positive step size | 8 | exhaustive |

## QA Gate

**Optimization Contract** (F-OPT-001)

Conjugate Gradient optimization correctness quality gate

**Checks:** monotone_decrease, finite_iterates, gradient_convergence

**Pass criteria:** All 3 falsification tests pass + Kani harnesses verify

