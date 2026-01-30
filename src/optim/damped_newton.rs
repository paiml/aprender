//! Damped Newton optimizer with finite-difference Hessian approximation.
//!
//! Newton's method uses second-order information (Hessian) to find the minimum
//! by solving H * d = -g. The damping factor and line search ensure global convergence.

use crate::primitives::{Matrix, Vector};

use super::line_search::{BacktrackingLineSearch, LineSearch};
use super::{ConvergenceStatus, OptimizationResult, Optimizer};

/// Damped Newton optimizer with finite-difference Hessian approximation.
///
/// Newton's method uses second-order information (Hessian) to find the minimum
/// by solving the linear system: H * d = -g, where H is the Hessian and g is
/// the gradient. The damping factor and line search ensure global convergence.
///
/// # Algorithm
///
/// 1. Compute gradient g = ∇f(x)
/// 2. Approximate Hessian H using finite differences
/// 3. Solve H * d = -g using Cholesky decomposition
/// 4. If Hessian not positive definite, fall back to steepest descent
/// 5. Line search along d to find step size α
/// 6. Update: x_{k+1} = `x_k` + α * `d_k`
///
/// # Parameters
///
/// - **`max_iter`**: Maximum number of iterations
/// - **tol**: Convergence tolerance (gradient norm)
/// - **epsilon**: Finite difference step size for Hessian approximation (default: 1e-5)
///
/// # Example
///
/// ```
/// use aprender::optim::{DampedNewton, Optimizer};
/// use aprender::primitives::Vector;
///
/// let mut optimizer = DampedNewton::new(100, 1e-5);
///
/// // Minimize quadratic function f(x,y) = x^2 + 2y^2
/// let f = |x: &Vector<f32>| x[0] * x[0] + 2.0 * x[1] * x[1];
/// let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 4.0 * x[1]]);
///
/// let x0 = Vector::from_slice(&[5.0, 3.0]);
/// let result = optimizer.minimize(f, grad, x0);
/// ```
#[derive(Debug, Clone)]
pub struct DampedNewton {
    /// Maximum number of iterations
    pub(crate) max_iter: usize,
    /// Convergence tolerance (gradient norm)
    pub(crate) tol: f32,
    /// Finite difference step size for Hessian approximation
    pub(crate) epsilon: f32,
    /// Line search strategy
    pub(crate) line_search: BacktrackingLineSearch,
}

impl DampedNewton {
    /// Creates a new Damped Newton optimizer.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum number of iterations (typical: 100-1000)
    /// * `tol` - Convergence tolerance for gradient norm (typical: 1e-5)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::DampedNewton;
    ///
    /// let optimizer = DampedNewton::new(100, 1e-5);
    /// ```
    #[must_use]
    pub fn new(max_iter: usize, tol: f32) -> Self {
        Self {
            max_iter,
            tol,
            epsilon: 1e-5, // Finite difference step size
            line_search: BacktrackingLineSearch::new(1e-4, 0.5, 50),
        }
    }

    /// Sets the finite difference epsilon for Hessian approximation.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Step size for finite differences (typical: 1e-5 to 1e-8)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::DampedNewton;
    ///
    /// let optimizer = DampedNewton::new(100, 1e-5).with_epsilon(1e-6);
    /// ```
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Approximates the Hessian matrix using finite differences.
    ///
    /// Uses central differences: H[i,j] ≈ (∂`²f/∂x_i∂x_j`)
    fn approximate_hessian<G>(&self, grad: &G, x: &Vector<f32>) -> Matrix<f32>
    where
        G: Fn(&Vector<f32>) -> Vector<f32>,
    {
        let n = x.len();
        let mut h_data = vec![0.0; n * n];

        let g0 = grad(x);

        // Compute Hessian using finite differences
        for i in 0..n {
            // Perturb x[i] by epsilon
            let mut x_plus = x.clone();
            x_plus[i] += self.epsilon;

            let g_plus = grad(&x_plus);

            // Approximate column i of Hessian: H[:,i] ≈ (g(x+ε*e_i) - g(x)) / ε
            for j in 0..n {
                h_data[j * n + i] = (g_plus[j] - g0[j]) / self.epsilon;
            }
        }

        // Symmetrize the Hessian (since it should be symmetric)
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = (h_data[i * n + j] + h_data[j * n + i]) / 2.0;
                h_data[i * n + j] = avg;
                h_data[j * n + i] = avg;
            }
        }

        Matrix::from_vec(n, n, h_data).expect("Matrix dimensions should be valid")
    }

    /// Computes the L2 norm of a vector.
    fn norm(v: &Vector<f32>) -> f32 {
        let mut sum = 0.0;
        for i in 0..v.len() {
            sum += v[i] * v[i];
        }
        sum.sqrt()
    }
}

impl Optimizer for DampedNewton {
    fn step(&mut self, _params: &mut Vector<f32>, _gradients: &Vector<f32>) {
        unimplemented!(
            "Damped Newton does not support stochastic updates (step). Use minimize() for batch optimization."
        )
    }

    fn minimize<F, G>(&mut self, objective: F, gradient: G, x0: Vector<f32>) -> OptimizationResult
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
    {
        let start_time = std::time::Instant::now();
        let n = x0.len();

        let mut x = x0;
        let mut fx = objective(&x);
        let mut grad = gradient(&x);
        let mut grad_norm = Self::norm(&grad);

        for iter in 0..self.max_iter {
            // Check convergence
            if grad_norm < self.tol {
                return OptimizationResult {
                    solution: x,
                    objective_value: fx,
                    iterations: iter,
                    status: ConvergenceStatus::Converged,
                    gradient_norm: grad_norm,
                    constraint_violation: 0.0,
                    elapsed_time: start_time.elapsed(),
                };
            }

            // Approximate Hessian
            let hessian = self.approximate_hessian(&gradient, &x);

            // Negate gradient for solving H * d = -g
            let mut neg_grad = Vector::zeros(n);
            for i in 0..n {
                neg_grad[i] = -grad[i];
            }

            // Solve H * d = -g using Cholesky decomposition
            let d = if let Ok(direction) = hessian.cholesky_solve(&neg_grad) {
                // Check if it's a descent direction
                let mut grad_dot_d = 0.0;
                for i in 0..n {
                    grad_dot_d += grad[i] * direction[i];
                }

                if grad_dot_d < 0.0 {
                    // Valid descent direction from Newton step
                    direction
                } else {
                    // Not a descent direction - fall back to steepest descent
                    let mut sd = Vector::zeros(n);
                    for i in 0..n {
                        sd[i] = -grad[i];
                    }
                    sd
                }
            } else {
                // Hessian not positive definite - fall back to steepest descent
                let mut sd = Vector::zeros(n);
                for i in 0..n {
                    sd[i] = -grad[i];
                }
                sd
            };

            // Line search
            let alpha = self.line_search.search(&objective, &gradient, &x, &d);

            // Check for stalled progress
            if alpha < 1e-12 {
                return OptimizationResult {
                    solution: x,
                    objective_value: fx,
                    iterations: iter,
                    status: ConvergenceStatus::Stalled,
                    gradient_norm: grad_norm,
                    constraint_violation: 0.0,
                    elapsed_time: start_time.elapsed(),
                };
            }

            // Update position: x_new = x + alpha * d
            let mut x_new = Vector::zeros(n);
            for i in 0..n {
                x_new[i] = x[i] + alpha * d[i];
            }

            // Compute new objective and gradient
            let fx_new = objective(&x_new);
            let grad_new = gradient(&x_new);

            // Check for numerical errors
            if fx_new.is_nan() || fx_new.is_infinite() {
                return OptimizationResult {
                    solution: x,
                    objective_value: fx,
                    iterations: iter,
                    status: ConvergenceStatus::NumericalError,
                    gradient_norm: grad_norm,
                    constraint_violation: 0.0,
                    elapsed_time: start_time.elapsed(),
                };
            }

            // Update for next iteration
            x = x_new;
            fx = fx_new;
            grad = grad_new;
            grad_norm = Self::norm(&grad);
        }

        // Max iterations reached
        OptimizationResult {
            solution: x,
            objective_value: fx,
            iterations: self.max_iter,
            status: ConvergenceStatus::MaxIterations,
            gradient_norm: grad_norm,
            constraint_violation: 0.0,
            elapsed_time: start_time.elapsed(),
        }
    }

    fn reset(&mut self) {
        // Damped Newton is stateless - nothing to reset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damped_newton_quadratic() {
        let mut optimizer = DampedNewton::new(100, 1e-5);

        // Simple quadratic: f(x,y) = x^2 + 2*y^2
        let f = |x: &Vector<f32>| x[0] * x[0] + 2.0 * x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 4.0 * x[1]]);

        let x0 = Vector::from_slice(&[5.0, 3.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!(result.solution[0].abs() < 1e-4);
        assert!(result.solution[1].abs() < 1e-4);
    }

    #[test]
    fn test_damped_newton_with_epsilon() {
        let optimizer = DampedNewton::new(100, 1e-5).with_epsilon(1e-6);
        assert!((optimizer.epsilon - 1e-6).abs() < 1e-10);
    }
}
