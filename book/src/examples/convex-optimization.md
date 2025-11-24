# Case Study: Convex Optimization

This example demonstrates **Phase 2 convex optimization methods** designed for composite problems with non-smooth regularization.

## Overview

Two specialized algorithms are covered:
- **FISTA** (Fast Iterative Shrinkage-Thresholding Algorithm)
- **Coordinate Descent**

Both methods excel at solving **composite optimization**:
```
minimize f(x) + g(x)
```
where f is smooth (differentiable) and g is "simple" (has easy proximal operator).

## Mathematical Background

### FISTA
**Problem**: minimize f(x) + g(x)

**Key idea**: Proximal gradient method with Nesterov acceleration

**Achieves**: O(1/k²) convergence (faster than standard gradient descent's O(1/k))

**Proximal operator**: prox_g(v, α) = argmin_x {½‖x - v‖² + α·g(x)}

### Coordinate Descent
**Problem**: minimize f(x)

**Key idea**: Update one coordinate at a time

**Algorithm**: x^(k+1)_i = argmin_z f(x^(k)_1, ..., x^(k)_{i-1}, z, x^(k)_{i+1}, ..., x^(k)_n)

**Particularly effective when**:
- Coordinate updates have closed-form solutions
- Problem dimension is very high (n >> m)
- Hessian is expensive to compute

## Examples Covered

### 1. Lasso Regression with FISTA
**Problem**: minimize ½‖Ax - b‖² + λ‖x‖₁

The classic Lasso problem:
- **Smooth part**: f(x) = ½‖Ax - b‖² (least squares)
- **Non-smooth part**: g(x) = λ‖x‖₁ (L1 regularization for sparsity)
- **Proximal operator**: Soft-thresholding

Demonstrates sparse recovery with only 3 non-zero coefficients out of 20 features.

### 2. Non-Negative Least Squares with FISTA
**Problem**: minimize ½‖Ax - b‖² subject to x ≥ 0

Applications:
- Spectral unmixing
- Image processing
- Chemometrics

Uses projection onto non-negative orthant as proximal operator.

### 3. High-Dimensional Lasso with Coordinate Descent
**Problem**: minimize ½‖Ax - b‖² + λ‖x‖₁ (n >> m)

With 100 features and only 30 samples (n >> m), demonstrates:
- Coordinate Descent efficiency in high dimensions
- Closed-form soft-thresholding updates
- Sparse recovery (5 non-zero out of 100)

### 4. Box-Constrained Quadratic Programming
**Problem**: minimize ½xᵀQx - cᵀx subject to l ≤ x ≤ u

Coordinate Descent with projection:
- Each coordinate update is a simple 1D optimization
- Project onto box constraints [l, u]
- Track active constraints (variables at bounds)

### 5. FISTA vs Coordinate Descent Comparison
Side-by-side comparison on the same Lasso problem:
- Convergence behavior
- Computational cost
- Solution quality

## Proximal Operators

Key proximal operators used in examples:

### Soft-Thresholding (L1 norm)
```rust
prox::soft_threshold(v, λ) = {
    v_i - λ  if v_i > λ
    0        if |v_i| ≤ λ
    v_i + λ  if v_i < -λ
}
```

### Non-negative Projection
```rust
prox::nonnegative(v) = max(v, 0)
```

### Box Projection
```rust
prox::box(v, l, u) = clamp(v, l, u)
```

## Performance Comparison

| Method | Problem Type | Iterations | Memory | Best For |
|--------|-------------|-----------|---------|----------|
| FISTA | Composite f+g | Low (~50-200) | O(n) | General composite problems |
| Coordinate Descent | Separable updates | Medium (~100-500) | O(n) | High-dimensional (n >> m) |

## Key Insights

### When to Use FISTA
- ✅ General composite optimization (smooth + non-smooth)
- ✅ Fast O(1/k²) convergence with Nesterov acceleration
- ✅ Works well for medium-scale problems
- ✅ Proximal operator available in closed form
- ❌ Requires Lipschitz constant estimation (step size tuning)

### When to Use Coordinate Descent
- ✅ High-dimensional problems (n >> m)
- ✅ Coordinate updates have closed-form solutions
- ✅ Very simple implementation
- ✅ No global gradients needed
- ❌ Slower convergence rate than FISTA
- ❌ Performance depends on coordinate ordering

## Convergence Analysis

Both methods track:
- **Iterations**: Number of outer iterations
- **Objective value**: Final f(x) + g(x)
- **Sparsity**: Number of non-zero coefficients (for Lasso)
- **Constraint violation**: ‖max(0, -x)‖ for non-negativity
- **Elapsed time**: Total optimization time

## Running the Examples

```bash
cargo run --example convex_optimization
```

The examples demonstrate:
1. Lasso with FISTA (20 features, 50 samples)
2. Non-negative LS with FISTA (10 features, 30 samples)
3. High-dimensional Lasso with CD (100 features, 30 samples)
4. Box-constrained QP with CD (15 variables)
5. FISTA vs CD comparison (30 features, 50 samples)

## Practical Tips

### For FISTA
1. **Step size**: Start with α = 0.01, use line search or backtracking
2. **Tolerance**: Set to 1e-4 to 1e-6 depending on accuracy needs
3. **Restart**: Implement adaptive restart for non-strongly convex problems
4. **Acceleration**: Always use Nesterov momentum for faster convergence

### For Coordinate Descent
1. **Ordering**: Cyclic (1,2,...,n) is simplest, random can help
2. **Convergence**: Check ‖x^k - x^{k-1}‖ < tol for stopping
3. **Updates**: Precompute any expensive quantities (e.g., column norms)
4. **Warm starts**: Initialize with previous solution when solving sequence of problems

## Comparison Summary

**Solution Quality**: Both methods find nearly identical solutions (‖x_FISTA - x_CD‖ < 1e-5)

**Speed**:
- FISTA: Faster for moderate n (~30-100)
- Coordinate Descent: Faster for large n (>100)

**Memory**:
- FISTA: O(n) gradient storage
- Coordinate Descent: O(n) solution only

**Ease of Use**:
- FISTA: Requires step size tuning
- Coordinate Descent: Requires coordinate update implementation

## Code Location

See [`examples/convex_optimization.rs`](../../examples/convex_optimization.rs) for full implementation.

## Related Topics

- [ADMM Optimization](./admm-optimization.md)
- [Regularized Regression](./regularized-regression.md)
- [Constrained Optimization](./constrained-optimization.md)
- [Advanced Optimizers Theory](../ml-fundamentals/advanced-optimizers.md)
- [Gradient Descent Theory](../ml-fundamentals/gradient-descent.md)
