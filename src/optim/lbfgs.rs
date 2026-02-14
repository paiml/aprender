//! Limited-memory BFGS (L-BFGS) optimizer.
//!
//! L-BFGS is a quasi-Newton method for large-scale optimization that approximates
//! the inverse Hessian using a limited history of gradient information.

use crate::primitives::Vector;

use super::line_search::{LineSearch, WolfeLineSearch};
use super::{ConvergenceStatus, OptimizationResult, Optimizer};

/// Limited-memory BFGS (L-BFGS) optimizer.
///
/// L-BFGS is a quasi-Newton method that approximates the inverse Hessian using
/// a limited history of gradient information. It's efficient for large-scale
/// optimization problems where storing the full Hessian is infeasible.
///
/// # Algorithm
///
/// 1. Compute gradient `g_k` = ∇`f(x_k)`
/// 2. Compute search direction `d_k` using two-loop recursion (approximates H^(-1) * `g_k`)
/// 3. Find step size `α_k` via line search (Wolfe conditions)
/// 4. Update: x_{k+1} = `x_k` - `α_k` * `d_k`
/// 5. Store gradient and position differences for next iteration
///
/// # Parameters
///
/// - **`max_iter`**: Maximum number of iterations
/// - **tol**: Convergence tolerance (gradient norm)
/// - **m**: History size (typically 5-20, tradeoff between memory and convergence)
///
/// # Example
///
/// ```
/// use aprender::optim::{LBFGS, Optimizer};
/// use aprender::primitives::Vector;
///
/// let mut optimizer = LBFGS::new(100, 1e-5, 10);
///
/// // Define Rosenbrock function and its gradient
/// let f = |x: &Vector<f32>| {
///     let a = x[0];
///     let b = x[1];
///     (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
/// };
///
/// let grad = |x: &Vector<f32>| {
///     let a = x[0];
///     let b = x[1];
///     Vector::from_slice(&[
///         -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
///         200.0 * (b - a * a),
///     ])
/// };
///
/// let x0 = Vector::from_slice(&[0.0, 0.0]);
/// let result = optimizer.minimize(f, grad, x0);
///
/// // Should converge to (1, 1)
/// assert_eq!(result.status, aprender::optim::ConvergenceStatus::Converged);
/// ```
#[derive(Debug, Clone)]
pub struct LBFGS {
    /// Maximum number of iterations
    pub(crate) max_iter: usize,
    /// Convergence tolerance (gradient norm)
    pub(crate) tol: f32,
    /// History size (number of correction pairs to store)
    pub(crate) m: usize,
    /// Line search strategy
    line_search: WolfeLineSearch,
    /// Position differences: `s_k` = x_{k+1} - `x_k`
    pub(crate) s_history: Vec<Vector<f32>>,
    /// Gradient differences: `y_k` = g_{k+1} - `g_k`
    pub(crate) y_history: Vec<Vector<f32>>,
}

impl LBFGS {
    /// Creates a new L-BFGS optimizer.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum number of iterations (typical: 100-1000)
    /// * `tol` - Convergence tolerance for gradient norm (typical: 1e-5)
    /// * `m` - History size (typical: 5-20)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::LBFGS;
    ///
    /// let optimizer = LBFGS::new(100, 1e-5, 10);
    /// ```
    #[must_use]
    pub fn new(max_iter: usize, tol: f32, m: usize) -> Self {
        Self {
            max_iter,
            tol,
            m,
            line_search: WolfeLineSearch::new(1e-4, 0.9, 50),
            s_history: Vec::with_capacity(m),
            y_history: Vec::with_capacity(m),
        }
    }

    /// Two-loop recursion to compute search direction.
    ///
    /// Approximates H^(-1) * grad where H is the Hessian.
    /// Uses stored history of s (position diff) and y (gradient diff).
    fn compute_direction(&self, grad: &Vector<f32>) -> Vector<f32> {
        let n = grad.len();
        let k = self.s_history.len();

        if k == 0 {
            // No history: use steepest descent
            let mut d = Vector::zeros(n);
            for i in 0..n {
                d[i] = -grad[i];
            }
            return d;
        }

        // Initialize q = -grad
        let mut q = Vector::zeros(n);
        for i in 0..n {
            q[i] = -grad[i];
        }

        let mut alpha = vec![0.0; k];
        let mut rho = vec![0.0; k];

        // First loop: backward pass
        for i in (0..k).rev() {
            let s = &self.s_history[i];
            let y = &self.y_history[i];

            // rho_i = 1 / (y_i^T s_i)
            let mut y_dot_s = 0.0;
            for j in 0..n {
                y_dot_s += y[j] * s[j];
            }
            rho[i] = 1.0 / y_dot_s;

            // alpha_i = rho_i * s_i^T * q
            let mut s_dot_q = 0.0;
            for j in 0..n {
                s_dot_q += s[j] * q[j];
            }
            alpha[i] = rho[i] * s_dot_q;

            // q = q - alpha_i * y_i
            for j in 0..n {
                q[j] -= alpha[i] * y[j];
            }
        }

        // Scale by H_0 = (s^T y) / (y^T y) from most recent update
        let s_last = &self.s_history[k - 1];
        let y_last = &self.y_history[k - 1];

        let mut s_dot_y = 0.0;
        let mut y_dot_y = 0.0;
        for i in 0..n {
            s_dot_y += s_last[i] * y_last[i];
            y_dot_y += y_last[i] * y_last[i];
        }
        let gamma = s_dot_y / y_dot_y;

        // r = H_0 * q = gamma * q
        let mut r = Vector::zeros(n);
        for i in 0..n {
            r[i] = gamma * q[i];
        }

        // Second loop: forward pass
        for i in 0..k {
            let s = &self.s_history[i];
            let y = &self.y_history[i];

            // beta = rho_i * y_i^T * r
            let mut y_dot_r = 0.0;
            for j in 0..n {
                y_dot_r += y[j] * r[j];
            }
            let beta = rho[i] * y_dot_r;

            // r = r + s_i * (alpha_i - beta)
            for j in 0..n {
                r[j] += s[j] * (alpha[i] - beta);
            }
        }

        r
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

impl Optimizer for LBFGS {
    fn step(&mut self, _params: &mut Vector<f32>, _gradients: &Vector<f32>) {
        panic!(
            "L-BFGS does not support stochastic updates (step). Use minimize() for batch optimization."
        )
    }

    fn minimize<F, G>(&mut self, objective: F, gradient: G, x0: Vector<f32>) -> OptimizationResult
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
    {
        let start_time = std::time::Instant::now();
        let n = x0.len();

        // Clear history from previous runs
        self.s_history.clear();
        self.y_history.clear();

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

            // Compute search direction
            let d = self.compute_direction(&grad);

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

            // Compute s_k = x_new - x and y_k = grad_new - grad
            let mut s_k = Vector::zeros(n);
            let mut y_k = Vector::zeros(n);
            for i in 0..n {
                s_k[i] = x_new[i] - x[i];
                y_k[i] = grad_new[i] - grad[i];
            }

            // Check curvature condition: y^T s > 0
            let mut y_dot_s = 0.0;
            for i in 0..n {
                y_dot_s += y_k[i] * s_k[i];
            }

            if y_dot_s > 1e-10 {
                // Store in history
                if self.s_history.len() >= self.m {
                    self.s_history.remove(0);
                    self.y_history.remove(0);
                }
                self.s_history.push(s_k);
                self.y_history.push(y_k);
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
        self.s_history.clear();
        self.y_history.clear();
    }
}

#[cfg(test)]
#[path = "lbfgs_tests.rs"]
mod tests;
