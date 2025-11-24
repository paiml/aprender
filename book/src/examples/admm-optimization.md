# Case Study: ADMM Optimization

This example demonstrates the **Alternating Direction Method of Multipliers (ADMM)** for distributed and constrained optimization problems.

## Overview

ADMM is particularly powerful for:
- **Distributed ML**: Split data across workers
- **Federated learning**: Train models across devices
- **Constrained problems**: Equality constraints via consensus

## Mathematical Formulation

ADMM solves problems of the form:

```
minimize  f(x) + g(z)
subject to Ax + Bz = c
```

The algorithm alternates between three steps:
1. **x-update**: minimize f(x) + (ρ/2)‖Ax + Bz - c + u‖²
2. **z-update**: minimize g(z) + (ρ/2)‖Ax + Bz - c + u‖²
3. **u-update**: u ← u + (Ax + Bz - c)

**Consensus form** (x = z): A = I, B = -I, c = 0

## Examples Covered

### 1. Distributed Lasso Regression
**Problem**: minimize ½‖Dx - b‖² + λ‖x‖₁

Separates smooth (least squares) and non-smooth (L1) parts using consensus form, allowing each to be solved efficiently with closed-form solutions.

### 2. Consensus Optimization (Federated Learning)
**Problem**: Average solutions from N distributed workers

Each worker has local data and computes a local solution. ADMM enforces consensus: all workers converge to the same global solution.

### 3. Quadratic Programming with ADMM
**Problem**: minimize ½xᵀQx + cᵀx subject to x ≥ 0

Uses consensus form to separate the quadratic objective from constraints, with projection onto non-negativity constraints.

### 4. ADMM vs FISTA Comparison
Compares ADMM and FISTA on the same Lasso problem to demonstrate convergence behavior and computational tradeoffs.

## Key Insights

**When to use ADMM:**
- Distributed data across multiple workers
- Federated learning scenarios
- Complex constraints that benefit from splitting
- Problems with naturally separable structure

**Advantages:**
- Consensus form enables distribution
- Adaptive ρ adjustment improves convergence
- Handles non-smooth objectives elegantly
- Provably converges for convex problems

**Compared to FISTA:**
- ADMM: Better for distributed settings, complex constraints
- FISTA: Simpler for centralized, composite problems

## Running the Example

```bash
cargo run --example admm_optimization
```

The example demonstrates all four ADMM use cases with detailed convergence analysis and performance metrics.

## Reference

Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). "Distributed Optimization and Statistical Learning via ADMM". *Foundations and Trends in Machine Learning*, 3(1), 1-122.

## Code Location

See [`examples/admm_optimization.rs`](../../examples/admm_optimization.rs) for full implementation.

## Related Topics

- [Convex Optimization (FISTA)](./convex-optimization.md)
- [Constrained Optimization](./constrained-optimization.md)
- [Optimizer Demo](./optimizer-demo.md)
