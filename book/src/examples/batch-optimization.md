# Case Study: Batch Optimization

This example demonstrates batch optimization algorithms for minimizing smooth, differentiable objective functions using gradient and Hessian information.

## Overview

Batch optimization algorithms process the entire dataset at once (as opposed to stochastic/mini-batch methods). This example covers three powerful second-order methods:

- **L-BFGS**: Limited-memory BFGS (quasi-Newton method)
- **Conjugate Gradient**: CG with multiple β formulas
- **Damped Newton**: Newton's method with finite differences

## Test Functions

The examples use classic optimization test functions:

### Rosenbrock Function
```
f(x,y) = (1-x)² + 100(y-x²)²
```
Global minimum at (1, 1). Features a narrow, curved valley making it challenging for optimizers.

### Sphere Function
```
f(x) = Σ x_i²
```
Convex quadratic with global minimum at origin. Easy test case - all optimizers should converge quickly.

### Booth Function
```
f(x,y) = (x + 2y - 7)² + (2x + y - 5)²
```
Global minimum at (1, 3) with f(1, 3) = 0.

## Examples Covered

### 1. Rosenbrock Function with Different Optimizers
Compares L-BFGS, Conjugate Gradient (Polak-Ribière and Fletcher-Reeves), and Damped Newton on the challenging Rosenbrock function.

### 2. Sphere Function (5D)
Tests all optimizers on a simple convex quadratic to verify correct implementation and fast convergence.

### 3. Booth Function
Demonstrates convergence on a moderately difficult quadratic problem.

### 4. Convergence Comparison
Runs optimizers from different initial points to analyze convergence behavior and robustness.

### 5. Optimizer Configuration
Shows how to configure:
- L-BFGS history size (m)
- CG periodic restart
- Damped Newton finite difference epsilon

## Key Insights

### L-BFGS
- **Memory**: Stores m recent gradients (typically m=10)
- **Convergence**: Superlinear for smooth convex functions
- **Use case**: General-purpose, large-scale optimization
- **Cost**: O(mn) per iteration

### Conjugate Gradient
- **Formulas**: Polak-Ribière, Fletcher-Reeves, Hestenes-Stiefel
- **Memory**: O(n) only (no history storage)
- **Convergence**: Linear for quadratics, can stall on non-quadratics
- **Use case**: When memory is limited, or Hessian is expensive
- **Tip**: Periodic restart (every n iterations) helps non-quadratic problems

### Damped Newton
- **Hessian**: Approximated via finite differences
- **Convergence**: Quadratic near minimum (fastest locally)
- **Use case**: High-accuracy solutions, few variables
- **Cost**: O(n²) Hessian approximation per iteration

## Convergence Comparison

| Method | Rosenbrock Iters | Sphere Iters | Memory |
|--------|-----------------|--------------|---------|
| L-BFGS | ~40-60 | ~10-15 | O(mn) |
| CG-PR  | ~80-120 | ~5-10 | O(n) |
| CG-FR  | ~100-150 | ~8-12 | O(n) |
| Damped Newton | ~20-30 | ~3-5 | O(n²) |

## Running the Example

```bash
cargo run --example batch_optimization
```

The example runs all test functions with all optimizers, displaying:
- Convergence status
- Iteration count
- Final solution
- Objective value
- Gradient norm
- Elapsed time

## Optimization Tips

1. **L-BFGS is the default choice** for most smooth optimization problems
2. **Use CG when memory is constrained** (large n)
3. **Use Damped Newton for high accuracy** on smaller problems
4. **Always try multiple starting points** to avoid local minima
5. **Monitor gradient norm** - should decrease to near-zero at optimum

## Code Location

See [`examples/batch_optimization.rs`](../../examples/batch_optimization.rs) for full implementation.

## Related Topics

- [Gradient Descent Theory](../ml-fundamentals/gradient-descent.md)
- [Advanced Optimizers Theory](../ml-fundamentals/advanced-optimizers.md)
- [Optimizer Demo](./optimizer-demo.md)
- [Convex Optimization](./convex-optimization.md)
