//! Shared test helpers for optimizer convergence tests
//!
//! These helpers provide common functions used across optimizer tests:
//! - Quadratic convergence (convex, optimal solution at origin)
//! - Rosenbrock function (non-convex, tests valley navigation)
//! - Ill-conditioned problems (tests numerical stability)
//! - High-dimensional problems (tests scalability)
//! - Numerical edge cases (very small/large gradients)

#[cfg(test)]
use crate::optim::Optimizer;
#[cfg(test)]
use crate::Tensor;

/// Test that optimizer converges on f(x) = x^2
#[cfg(test)]
pub fn test_quadratic_convergence<O: Optimizer>(
    mut optimizer: O,
    iterations: usize,
    threshold: f32,
) -> bool {
    let mut params = vec![Tensor::from_vec(vec![3.0, -2.0, 1.5, -2.5], true)];

    for _ in 0..iterations {
        // Compute gradient: grad(x^2) = 2x
        let grad = params[0].data().mapv(|x| 2.0 * x);
        params[0].set_grad(grad);

        optimizer.step(&mut params);
    }

    // All parameters should converge close to 0
    params[0].data().iter().all(|&val| val.abs() < threshold)
}

/// Test that optimizer decreases loss monotonically
#[cfg(test)]
pub fn test_loss_decreases<O: Optimizer>(mut optimizer: O, iterations: usize) -> bool {
    let mut params = vec![Tensor::from_vec(vec![10.0], true)];
    let mut prev_loss = f32::INFINITY;

    for _ in 0..iterations {
        // Compute loss and gradient for f(x) = x^2
        let x = params[0].data()[0];
        let loss = x * x;
        let grad = ndarray::arr1(&[2.0 * x]);

        // Loss should decrease (or stay same if converged)
        if loss > prev_loss + 1e-3 {
            return false; // Loss increased significantly
        }

        prev_loss = loss;
        params[0].set_grad(grad);
        optimizer.step(&mut params);
    }

    true
}

/// Test Rosenbrock function convergence (non-convex)
/// f(x,y) = (a-x)^2 + b(y-x^2)^2, minimum at (a, a^2)
#[cfg(test)]
#[allow(dead_code)]
pub fn test_rosenbrock_convergence<O: Optimizer>(
    mut optimizer: O,
    iterations: usize,
    threshold: f32,
) -> bool {
    // Start from [0, 0], optimal is [1, 1] for a=1, b=100
    let mut params = vec![Tensor::from_vec(vec![0.0, 0.0], true)];
    let a = 1.0f32;
    let b = 100.0f32;

    for _ in 0..iterations {
        let x = params[0].data()[0];
        let y = params[0].data()[1];

        // Gradient of Rosenbrock
        // df/dx = -2(a-x) - 4bx(y-x^2)
        // df/dy = 2b(y-x^2)
        let dx = -2.0 * (a - x) - 4.0 * b * x * (y - x * x);
        let dy = 2.0 * b * (y - x * x);

        let grad = ndarray::arr1(&[dx, dy]);
        params[0].set_grad(grad);
        optimizer.step(&mut params);
    }

    // Check if converged to [1, 1]
    let x = params[0].data()[0];
    let y = params[0].data()[1];
    (x - 1.0).abs() < threshold && (y - 1.0).abs() < threshold
}

/// Test ill-conditioned quadratic (high condition number)
/// f(x) = 0.5 * x^T * A * x where A has eigenvalues [1, 100]
#[cfg(test)]
pub fn test_ill_conditioned_convergence<O: Optimizer>(
    mut optimizer: O,
    iterations: usize,
    threshold: f32,
) -> bool {
    // 2D ill-conditioned problem: f(x,y) = 0.5*(x^2 + 100*y^2)
    let mut params = vec![Tensor::from_vec(vec![10.0, 10.0], true)];

    for _ in 0..iterations {
        let x = params[0].data()[0];
        let y = params[0].data()[1];

        // Gradient: [x, 100*y]
        let grad = ndarray::arr1(&[x, 100.0 * y]);
        params[0].set_grad(grad);
        optimizer.step(&mut params);
    }

    // Should converge to [0, 0]
    params[0].data().iter().all(|&val| val.abs() < threshold)
}

/// Test high-dimensional problem
#[cfg(test)]
pub fn test_high_dim_convergence<O: Optimizer>(
    mut optimizer: O,
    dim: usize,
    iterations: usize,
    threshold: f32,
) -> bool {
    let init: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let mut params = vec![Tensor::from_vec(init, true)];

    for _ in 0..iterations {
        // Gradient of f(x) = sum(x_i^2) is 2*x
        let grad = params[0].data().mapv(|x| 2.0 * x);
        params[0].set_grad(grad);
        optimizer.step(&mut params);
    }

    params[0].data().iter().all(|&val| val.abs() < threshold)
}

/// Test numerical stability with very small gradients
#[cfg(test)]
pub fn test_small_gradient_stability<O: Optimizer>(mut optimizer: O) -> bool {
    let mut params = vec![Tensor::from_vec(vec![1e-6, 1e-6], true)];

    for _ in 0..100 {
        let grad = params[0].data().mapv(|x| 2.0 * x);
        params[0].set_grad(grad);
        optimizer.step(&mut params);
    }

    // Should not produce NaN or Inf
    params[0].data().iter().all(|&val| val.is_finite())
}

/// Test numerical stability with large gradients
#[cfg(test)]
pub fn test_large_gradient_stability<O: Optimizer>(mut optimizer: O) -> bool {
    let mut params = vec![Tensor::from_vec(vec![1e4, 1e4], true)];

    for _ in 0..100 {
        let grad = params[0].data().mapv(|x| 2.0 * x);
        params[0].set_grad(grad);
        optimizer.step(&mut params);
    }

    // Should not produce NaN or Inf
    params[0].data().iter().all(|&val| val.is_finite())
}
