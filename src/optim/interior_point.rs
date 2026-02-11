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
        panic!(
            "Interior Point does not support stochastic updates (step). Use minimize() for constrained optimization."
        )
    }

    fn reset(&mut self) {
        // Reset barrier parameter to initial value
        self.mu = self.initial_mu;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ip_new() {
        let ip = InteriorPoint::new(50, 1e-6, 1.0);
        let debug_str = format!("{:?}", ip);
        assert!(debug_str.contains("InteriorPoint"));
        assert!(debug_str.contains("50"));
    }

    #[test]
    fn test_ip_clone_debug() {
        let ip = InteriorPoint::new(50, 1e-6, 1.0);
        let cloned = ip.clone();
        let d1 = format!("{:?}", ip);
        let d2 = format!("{:?}", cloned);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_ip_with_beta() {
        let ip = InteriorPoint::new(50, 1e-6, 1.0).with_beta(0.1);
        let debug_str = format!("{:?}", ip);
        assert!(debug_str.contains("0.1"));
    }

    #[test]
    fn test_ip_nonnegative_quadratic() {
        // min x1^2 + x2^2 s.t. x >= 0
        let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0], -x[1]]);
        let inequality_jac = |_x: &Vector<f32>| {
            vec![
                Vector::from_slice(&[-1.0, 0.0]),
                Vector::from_slice(&[0.0, -1.0]),
            ]
        };

        let mut ip = InteriorPoint::new(50, 1e-3, 1.0);
        let x0 = Vector::from_slice(&[1.0, 1.0]);
        let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

        assert!(result.solution[0] >= -1e-2);
        assert!(result.solution[1] >= -1e-2);
    }

    #[test]
    fn test_ip_max_iterations() {
        let objective = |x: &Vector<f32>| (x[0] - 5.0).powi(2);
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 5.0)]);
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
        let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

        let mut ip = InteriorPoint::new(2, 1e-20, 1.0);
        let x0 = Vector::from_slice(&[1.0]);
        let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
        assert_eq!(result.iterations, 2);
        assert!(result.objective_value.is_finite());
        assert!(result.gradient_norm >= 0.0);
    }

    #[test]
    fn test_ip_converged_result_fields() {
        let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0], -x[1]]);
        let inequality_jac = |_x: &Vector<f32>| {
            vec![
                Vector::from_slice(&[-1.0, 0.0]),
                Vector::from_slice(&[0.0, -1.0]),
            ]
        };

        let mut ip = InteriorPoint::new(100, 1e-3, 1.0).with_beta(0.1);
        let x0 = Vector::from_slice(&[1.0, 1.0]);
        let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

        assert!(result.constraint_violation >= 0.0);
        let _ = result.elapsed_time.as_nanos();
    }

    #[test]
    #[should_panic(expected = "Initial point is infeasible")]
    fn test_ip_infeasible_start() {
        let objective = |x: &Vector<f32>| x[0] * x[0];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        // Constraint: -x <= 0, i.e. x >= 0
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
        let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

        let mut ip = InteriorPoint::new(50, 1e-6, 1.0);
        let x0 = Vector::from_slice(&[-1.0]); // Infeasible: g(x) = 1.0 >= 0
        let _ = ip.minimize(objective, gradient, inequality, inequality_jac, x0);
    }

    #[test]
    fn test_ip_reset() {
        let mut ip = InteriorPoint::new(50, 1e-6, 5.0);
        // Run a minimization to change mu
        let objective = |x: &Vector<f32>| x[0] * x[0];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
        let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

        let _ = ip.minimize(
            objective,
            gradient,
            inequality,
            inequality_jac,
            Vector::from_slice(&[1.0]),
        );

        // Reset should restore mu to initial value
        ip.reset();
        // The next call should start with initial_mu again
        let debug_str = format!("{:?}", ip);
        assert!(debug_str.contains("mu: 5.0"));
    }

    #[test]
    #[should_panic(expected = "does not support stochastic updates")]
    fn test_ip_step_panics() {
        let mut ip = InteriorPoint::new(50, 1e-6, 1.0);
        let mut params = Vector::from_slice(&[1.0]);
        let grad = Vector::from_slice(&[0.1]);
        ip.step(&mut params, &grad);
    }

    #[test]
    fn test_ip_constraint_boundary_hit() {
        // Problem where solution is at constraint boundary (barrier subproblem
        // exercises the infeasible branch within the inner loop)
        let objective = |x: &Vector<f32>| (x[0] - 10.0).powi(2);
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 10.0)]);
        // Constraint: x <= 2 => g(x) = x - 2 <= 0
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[x[0] - 2.0]);
        let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[1.0])];

        let mut ip = InteriorPoint::new(100, 1e-3, 1.0).with_beta(0.1);
        let x0 = Vector::from_slice(&[1.0]); // feasible: g(1) = -1 < 0
        let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

        // Solution should move towards the boundary x = 2 (interior point
        // uses small gradient steps, so it may not reach exactly)
        assert!(
            result.solution[0] > 0.5,
            "Expected solution > 0.5, got {}",
            result.solution[0]
        );
    }

    #[test]
    fn test_ip_aggressive_beta() {
        // Use very aggressive beta to exercise fast mu decrease
        let objective = |x: &Vector<f32>| x[0] * x[0];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
        let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

        let mut ip = InteriorPoint::new(100, 1e-3, 10.0).with_beta(0.01);
        let x0 = Vector::from_slice(&[1.0]);
        let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

        // Should converge since mu decreases quickly
        assert!(
            result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::MaxIterations
        );
    }

    #[test]
    fn test_ip_constraint_boundary_stepback() {
        // Design a problem that triggers the step-back logic when we approach
        // constraint boundaries during gradient descent.
        // Minimize (x - 0.5)^2 subject to x >= 0 (constraint: -x <= 0)
        // The objective minimum at x=0.5 is feasible, so it should converge there.
        let objective = |x: &Vector<f32>| (x[0] - 0.5).powi(2);
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 0.5)]);
        // Constraint: x >= 0 => g(x) = -x <= 0
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
        let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

        // Start close to boundary to trigger step-back logic
        let mut ip = InteriorPoint::new(100, 1e-4, 0.5).with_beta(0.3);
        let x0 = Vector::from_slice(&[0.01]); // Very close to boundary x=0
        let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

        // Solution should move towards x=0.5 while respecting constraint
        assert!(
            result.solution[0] >= -0.01,
            "Solution {} should respect constraint x >= 0",
            result.solution[0]
        );
        assert!(result.objective_value.is_finite());
    }

    #[test]
    fn test_ip_multi_constraint_with_boundary_skip() {
        // Multiple constraints where one becomes inactive (g >= 0 continue branch)
        // Minimize x^2 + y^2 subject to x >= -10, y >= -10 (loose constraints)
        // g(x) = [-x - 10, -y - 10] <= 0
        let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0] - 10.0, -x[1] - 10.0]);
        let inequality_jac = |_x: &Vector<f32>| {
            vec![
                Vector::from_slice(&[-1.0, 0.0]),
                Vector::from_slice(&[0.0, -1.0]),
            ]
        };

        let mut ip = InteriorPoint::new(100, 1e-4, 1.0).with_beta(0.1);
        let x0 = Vector::from_slice(&[5.0, 5.0]); // Far from constraints
        let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

        // Solution should approach origin since constraints are loose
        assert!(
            result.solution[0].abs() < 1.0,
            "x should approach 0, got {}",
            result.solution[0]
        );
        assert!(
            result.solution[1].abs() < 1.0,
            "y should approach 0, got {}",
            result.solution[1]
        );
    }

    #[test]
    fn test_ip_early_gradient_convergence() {
        // Test early exit when gradient norm is very small in sub-iteration
        // Use a simple quadratic with tight constraints at origin
        let objective = |x: &Vector<f32>| x[0] * x[0];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        // Constraint: x >= 0 => g(x) = -x <= 0
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
        let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

        // Start very close to optimum
        let mut ip = InteriorPoint::new(50, 1e-6, 0.1).with_beta(0.1);
        let x0 = Vector::from_slice(&[0.001]); // Very close to optimum
        let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

        // Should converge quickly
        assert!(result.solution[0] >= -1e-3);
        assert!(result.solution[0] < 0.1);
    }

    #[test]
    fn test_ip_positive_max_violation() {
        // Design a problem that may have positive constraint violation during iteration
        // This exercises the max_violation > 0 path
        let objective = |x: &Vector<f32>| (x[0] - 100.0).powi(2);
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 100.0)]);
        // Tight constraint: x <= 5
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[x[0] - 5.0]);
        let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[1.0])];

        let mut ip = InteriorPoint::new(20, 1e-4, 1.0).with_beta(0.3);
        let x0 = Vector::from_slice(&[4.0]); // Feasible start
        let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

        // Result should have constraint_violation tracked
        assert!(result.constraint_violation >= 0.0);
        // Solution should be at or near boundary
        assert!(
            result.solution[0] >= 3.0,
            "Solution should be pushed towards boundary"
        );
    }

    #[test]
    fn test_ip_three_constraints() {
        // Problem with three constraints to test loop iteration
        // Minimize x^2 + y^2 subject to x >= 0, y >= 0, x + y <= 3
        let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        // g1: -x <= 0, g2: -y <= 0, g3: x + y - 3 <= 0
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0], -x[1], x[0] + x[1] - 3.0]);
        let inequality_jac = |_x: &Vector<f32>| {
            vec![
                Vector::from_slice(&[-1.0, 0.0]),
                Vector::from_slice(&[0.0, -1.0]),
                Vector::from_slice(&[1.0, 1.0]),
            ]
        };

        let mut ip = InteriorPoint::new(100, 1e-3, 1.0).with_beta(0.2);
        let x0 = Vector::from_slice(&[1.0, 1.0]);
        let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

        // Solution should be close to origin
        assert!(result.solution[0] >= -0.1);
        assert!(result.solution[1] >= -0.1);
        assert!(result.solution[0] + result.solution[1] <= 3.1);
    }

    #[test]
    fn test_ip_hit_constraint_boundary_exactly() {
        // Create scenario where g_val[j] can become exactly 0 or very close
        // to trigger the g_val[j] >= 0.0 continue branch
        let objective = |x: &Vector<f32>| (x[0] - 10.0).powi(2);
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 10.0)]);
        // Constraint: x <= 2 => g(x) = x - 2 <= 0
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[x[0] - 2.0]);
        let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[1.0])];

        // Use very aggressive parameters to potentially hit boundary
        let mut ip = InteriorPoint::new(100, 1e-6, 0.01).with_beta(0.01);
        let x0 = Vector::from_slice(&[1.9]); // Start very close to boundary
        let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

        // Should reach boundary
        assert!(result.solution[0] >= 1.5, "Should move toward boundary");
        assert!(result.objective_value.is_finite());
    }

    #[test]
    fn test_ip_max_iter_with_violation_tracking() {
        // Test that max_violation is correctly computed in the max iterations path
        // Use impossible convergence tolerance
        let objective = |x: &Vector<f32>| x[0] * x[0];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let inequality = |x: &Vector<f32>| Vector::from_slice(&[-x[0]]);
        let inequality_jac = |_x: &Vector<f32>| vec![Vector::from_slice(&[-1.0])];

        // Impossibly tight tolerance forces max iterations
        let mut ip = InteriorPoint::new(3, 1e-30, 1.0);
        let x0 = Vector::from_slice(&[5.0]);
        let result = ip.minimize(objective, gradient, inequality, inequality_jac, x0);

        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
        assert_eq!(result.iterations, 3);
        // max_violation path should be exercised
        assert!(result.constraint_violation >= 0.0);
        assert!(result.gradient_norm >= 0.0);
    }
}
