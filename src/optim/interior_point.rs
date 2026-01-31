//! Interior Point (Barrier) method for inequality-constrained optimization.

use super::{ConvergenceStatus, OptimizationResult, Optimizer};
use crate::primitives::Vector;

/// Interior Point (Barrier) method for inequality-constrained optimization.
///
/// Solves problems with inequality constraints:
/// ```text
/// minimize f(x)
/// subject to: g(x) ≤ 0  (inequality constraints)
/// ```
///
/// # Algorithm
///
/// ```text
/// Barrier function: B_μ(x) = f(x) - μ Σ log(-g_i(x))
///
/// for k = 1, 2, ..., max_iter:
///     x_k = argmin B_μ(x)  (barrier subproblem)
///     μ_k+1 = β * μ_k      (decrease barrier parameter)
///     if ‖∇B_μ(x)‖ is small: converged
/// ```
///
/// # Key Features
///
/// - **Log-barrier**: Enforces g(x) < 0 via -μ log(-g_i(x))
/// - **Path-following**: Decreases μ → 0 to approach constrained optimum
/// - **Self-concordant**: Converges in O(√n log(1/ε)) iterations
/// - **Warm start**: Uses previous solution for next barrier value
///
/// # Applications
///
/// - **Linear programming**: Constraints Ax ≤ b
/// - **Quadratic programming**: QP with inequality constraints
/// - **Semidefinite programming**: Matrix constraints X ⪰ 0
/// - **Support Vector Machines**: Soft-margin constraints
/// - **Portfolio optimization**: Long-only constraints (x ≥ 0)
///
/// # Example
///
/// ```
/// use aprender::optim::InteriorPoint;
/// use aprender::primitives::Vector;
///
/// // Minimize: x₁² + x₂² subject to -x₁ ≤ 0, -x₂ ≤ 0 (i.e., x ≥ 0)
/// let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
///
/// let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
///
/// // Inequality constraints: g(x) = [-x₁, -x₂] ≤ 0
/// let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0], -x[1]]);
///
/// let inequality_jac = |_x: &Vector<f32>| {
///     vec![Vector::from_slice(&[-1.0, 0.0]), Vector::from_slice(&[0.0, -1.0])]
/// };
///
/// let mut ip = InteriorPoint::new(50, 1e-6, 1.0);
/// let x0 = Vector::from_slice(&[1.0, 1.0]); // Feasible start
/// let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);
///
/// // Solution should be [0, 0] (constrained minimum)
/// assert!(result.solution[0].abs() < 1e-3);
/// assert!(result.solution[1].abs() < 1e-3);
/// ```
///
/// # References
///
/// - Nesterov & Nemirovskii (1994). "Interior-Point Polynomial Algorithms in Convex Programming."
/// - Boyd & Vandenberghe (2004). "Convex Optimization." Chapter 11.
/// - Wright (1997). "Primal-Dual Interior-Point Methods."
#[derive(Debug, Clone)]
pub struct InteriorPoint {
    /// Maximum number of outer iterations (barrier parameter updates)
    max_iter: usize,
    /// Convergence tolerance
    tol: f32,
    /// Initial barrier parameter
    initial_mu: f32,
    /// Current barrier parameter
    mu: f32,
    /// Barrier decrease factor (0 < beta < 1, typically 0.1-0.5)
    beta: f32,
}

impl InteriorPoint {
    /// Creates a new Interior Point optimizer.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum number of outer iterations
    /// * `tol` - Convergence tolerance
    /// * `initial_mu` - Initial barrier parameter (typically 1.0-10.0)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::InteriorPoint;
    ///
    /// let optimizer = InteriorPoint::new(50, 1e-6, 1.0);
    /// ```
    #[must_use]
    pub fn new(max_iter: usize, tol: f32, initial_mu: f32) -> Self {
        Self {
            max_iter,
            tol,
            initial_mu,
            mu: initial_mu,
            beta: 0.2,
        }
    }

    /// Sets barrier decrease factor.
    ///
    /// # Arguments
    ///
    /// * `beta` - Barrier decrease factor (0 < beta < 1, typically 0.1-0.5)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::InteriorPoint;
    ///
    /// let optimizer = InteriorPoint::new(50, 1e-6, 1.0)
    ///     .with_beta(0.1); // Aggressive barrier decrease
    /// ```
    #[must_use]
    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Minimizes objective subject to inequality constraints.
    ///
    /// Solves: minimize f(x) subject to g(x) ≤ 0
    ///
    /// # Arguments
    ///
    /// * `objective` - Objective function f(x)
    /// * `gradient` - Gradient ∇f(x)
    /// * `inequality` - Inequality constraints g(x) ≤ 0 (returns vector)
    /// * `inequality_jac` - Jacobian of inequality constraints ∇g(x)
    /// * `x0` - Initial feasible point (must satisfy g(x0) < 0 strictly)
    ///
    /// # Returns
    ///
    /// Optimization result with constraint satisfaction metrics
    ///
    /// # Panics
    ///
    /// Panics if initial point is infeasible (g(x0) ≥ 0 for any constraint)
    #[allow(clippy::too_many_lines)]
    pub fn minimize<F, G, H, J>(
        &mut self,
        objective: F,
        gradient: G,
        inequality: H,
        inequality_jac: J,
        x0: Vector<f32>,
    ) -> OptimizationResult
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
        H: Fn(&Vector<f32>) -> Vector<f32>,
        J: Fn(&Vector<f32>) -> Vec<Vector<f32>>,
    {
        let start_time = std::time::Instant::now();

        // Check initial feasibility
        let g0 = inequality(&x0);
        for i in 0..g0.len() {
            assert!(
                g0[i] < 0.0,
                "Initial point is infeasible: g[{}] = {} ≥ 0. Interior point requires strictly feasible start.",
                i, g0[i]
            );
        }

        let mut x = x0;
        self.mu = self.initial_mu;
        let m = g0.len(); // Number of inequality constraints

        for outer_iter in 0..self.max_iter {
            // Solve barrier subproblem: min B_μ(x) = f(x) - μ Σ log(-g_i(x))

            let barrier_grad = |x_inner: &Vector<f32>| {
                let grad_f = gradient(x_inner);
                let g_val = inequality(x_inner);
                let jac_g = inequality_jac(x_inner);

                let n = x_inner.len();
                let mut barrier_g = Vector::zeros(n);

                // ∇f(x)
                for i in 0..n {
                    barrier_g[i] = grad_f[i];
                }

                // Subtract μ Σ (1/(-g_i)) * ∇g_i(x)
                for j in 0..m {
                    if g_val[j] >= 0.0 {
                        // Hit constraint boundary - project back
                        continue;
                    }
                    let coeff = -self.mu / g_val[j];
                    for i in 0..n {
                        barrier_g[i] += coeff * jac_g[j][i];
                    }
                }

                barrier_g
            };

            // Solve barrier subproblem using gradient descent
            let mut x_sub = x.clone();
            let alpha = 0.01; // Fixed step size
            for _sub_iter in 0..50 {
                let grad = barrier_grad(&x_sub);

                // Check if gradient is small (converged)
                let mut grad_norm_sq = 0.0;
                for i in 0..grad.len() {
                    grad_norm_sq += grad[i] * grad[i];
                }
                if grad_norm_sq < 1e-8 {
                    break;
                }

                // Gradient descent step
                for i in 0..x_sub.len() {
                    x_sub[i] -= alpha * grad[i];
                }

                // Check feasibility - if we violated constraints, step back
                let g_sub = inequality(&x_sub);
                let mut infeasible = false;
                for i in 0..m {
                    if g_sub[i] >= -1e-8 {
                        // Close to or past boundary
                        infeasible = true;
                        break;
                    }
                }
                if infeasible {
                    // Step back
                    for i in 0..x_sub.len() {
                        x_sub[i] += alpha * grad[i] * 0.5; // Half step back
                    }
                }
            }

            x = x_sub;

            // Check convergence via gradient of barrier function
            let grad_barrier = barrier_grad(&x);
            let mut grad_norm = 0.0;
            for i in 0..grad_barrier.len() {
                grad_norm += grad_barrier[i] * grad_barrier[i];
            }
            grad_norm = grad_norm.sqrt();

            // Also check constraint violation
            let g_val = inequality(&x);
            let mut max_violation = 0.0;
            for i in 0..m {
                if g_val[i] > max_violation {
                    max_violation = g_val[i];
                }
            }

            // Converged if gradient is small and μ is small
            if grad_norm < self.tol && self.mu < 1e-4 {
                let final_obj = objective(&x);
                return OptimizationResult {
                    solution: x,
                    objective_value: final_obj,
                    iterations: outer_iter + 1,
                    status: ConvergenceStatus::Converged,
                    gradient_norm: grad_norm,
                    constraint_violation: max_violation.max(0.0),
                    elapsed_time: start_time.elapsed(),
                };
            }

            // Decrease barrier parameter
            self.mu *= self.beta;
        }

        // Max iterations reached
        let final_obj = objective(&x);
        let g_val = inequality(&x);
        let mut max_violation = 0.0;
        for i in 0..g_val.len() {
            if g_val[i] > max_violation {
                max_violation = g_val[i];
            }
        }

        let grad_f = gradient(&x);
        let mut grad_norm = 0.0;
        for i in 0..grad_f.len() {
            grad_norm += grad_f[i] * grad_f[i];
        }
        grad_norm = grad_norm.sqrt();

        OptimizationResult {
            solution: x,
            objective_value: final_obj,
            iterations: self.max_iter,
            status: ConvergenceStatus::MaxIterations,
            gradient_norm: grad_norm,
            constraint_violation: max_violation.max(0.0),
            elapsed_time: start_time.elapsed(),
        }
    }
}

impl Optimizer for InteriorPoint {
    fn step(&mut self, _params: &mut Vector<f32>, _gradients: &Vector<f32>) {
        unimplemented!(
            "Interior Point does not support stochastic updates (step). Use minimize() for constrained optimization."
        )
    }

    fn reset(&mut self) {
        // Reset barrier parameter to initial value
        self.mu = self.initial_mu;
    }
}
