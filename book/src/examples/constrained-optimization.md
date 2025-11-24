# Case Study: Constrained Optimization

This example demonstrates **Phase 3 constrained optimization methods** for handling various constraint types in optimization problems.

## Overview

Three complementary methods are presented:
- **Projected Gradient Descent (PGD)**: For projection constraints x ∈ C
- **Augmented Lagrangian**: For equality constraints h(x) = 0
- **Interior Point Method**: For inequality constraints g(x) ≤ 0

## Mathematical Background

### Projected Gradient Descent
**Problem**: minimize f(x) subject to x ∈ C (convex set)

**Algorithm**: x^{k+1} = P_C(x^k - α∇f(x^k))

where P_C is projection onto convex set C.

**Applications**: Portfolio optimization, signal processing, compressed sensing

### Augmented Lagrangian
**Problem**: minimize f(x) subject to h(x) = 0

**Augmented Lagrangian**: L_ρ(x, λ) = f(x) + λᵀh(x) + ½ρ‖h(x)‖²

**Updates**: λ^{k+1} = λ^k + ρh(x^{k+1})

**Applications**: Equality-constrained least squares, manifold optimization, PDEs

### Interior Point Method
**Problem**: minimize f(x) subject to g(x) ≤ 0

**Log-barrier**: B_μ(x) = f(x) - μ Σ log(-g_i(x))

As μ → 0, solution approaches constrained optimum.

**Applications**: Linear programming, quadratic programming, convex optimization

## Examples Covered

### 1. Non-Negative Quadratic with Projected GD
**Problem**: minimize ½‖x - target‖² subject to x ≥ 0

Simple but important problem appearing in:
- Portfolio optimization (long-only constraints)
- Non-negative matrix factorization
- Signal processing

### 2. Equality-Constrained Least Squares
**Problem**: minimize ½‖Ax - b‖² subject to Cx = d

Demonstrates Augmented Lagrangian with:
- x₀ + x₁ + x₂ = 1.0 (sum constraint)
- x₃ + x₄ = 0.5 (partial sum)
- x₅ - x₆ = 0.0 (equality relationship)

### 3. Linear Programming with Interior Point
**Problem**: maximize -2x₀ - 3x₁ subject to linear inequalities

Classic LP problem:
- x₀ + 2x₁ ≤ 8 (resource constraint 1)
- 3x₀ + 2x₁ ≤ 12 (resource constraint 2)
- x₀ ≥ 0, x₁ ≥ 0 (non-negativity)

### 4. Quadratic Programming with Interior Point
**Problem**: minimize ½xᵀQx + cᵀx subject to budget and non-negativity constraints

QP problems appear in:
- Model predictive control
- Portfolio optimization with risk constraints
- Support vector machines

### 5. Method Comparison - Box-Constrained Quadratic
**Problem**: minimize ½‖x - target‖² subject to 0 ≤ x ≤ 1

Compares all three methods on the same problem to demonstrate their relative strengths.

## Performance Comparison

| Method | Constraint Type | Iterations | Best For |
|--------|----------------|-----------|----------|
| Projected GD | Simple sets (box, simplex) | Medium | Fast projection available |
| Augmented Lagrangian | Equality | Low-Medium | Nonlinear equalities |
| Interior Point | Inequality | Low | LP/QP, strict feasibility |

## Key Insights

### When to Use Each Method

**Projected GD:**
- ✅ Simple convex constraints (box, ball, simplex)
- ✅ Fast projection operator available
- ✅ High-dimensional problems
- ❌ Complex constraint interactions

**Augmented Lagrangian:**
- ✅ Equality constraints
- ✅ Nonlinear constraints
- ✅ Can handle multiple constraint types
- ❌ Requires penalty parameter tuning

**Interior Point:**
- ✅ Inequality constraints g(x) ≤ 0
- ✅ LP and QP problems
- ✅ Guarantees feasibility throughout
- ❌ Requires strictly feasible starting point

## Constraint Handling Tips

1. **Check feasibility**: Ensure x₀ satisfies all constraints
2. **Active set identification**: Track which constraints are active (g(x) ≈ 0)
3. **Lagrange multipliers**: Provide sensitivity information
4. **Penalty parameters**: Start small (ρ ≈ 0.1-1.0), increase gradually
5. **Warm starts**: Use previous solutions when solving similar problems

## Convergence Analysis

Each method includes convergence metrics:
- **Status**: Converged, MaxIterations, Stalled
- **Constraint violation**: ‖h(x)‖ or max(g(x))
- **Gradient norm**: Measures first-order optimality
- **Objective value**: Final cost

## Running the Example

```bash
cargo run --example constrained_optimization
```

The example demonstrates all five constrained optimization scenarios with detailed analysis of:
- Constraint satisfaction
- Active constraints
- Convergence behavior
- Computational cost

## Implementation Notes

### Projected Gradient Descent
- Line search with backtracking
- Armijo condition after projection
- Simple projection operators (element-wise for box constraints)

### Augmented Lagrangian
- Penalty parameter starts at ρ = 0.1
- Multiplier update: λ += ρ * h(x)
- Inner optimization via L-BFGS

### Interior Point
- Log-barrier parameter μ decreases geometrically (μ *= 0.1)
- Newton direction with Hessian approximation
- Feasibility check on every iteration

## Code Location

See [`examples/constrained_optimization.rs`](../../examples/constrained_optimization.rs) for full implementation.

## Related Topics

- [ADMM Optimization](./admm-optimization.md)
- [Convex Optimization](./convex-optimization.md)
- [Regularized Regression](./regularized-regression.md)
- [Advanced Optimizers Theory](../ml-fundamentals/advanced-optimizers.md)
