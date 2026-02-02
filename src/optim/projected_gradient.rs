//! Projected Gradient Descent for constrained optimization.

use super::{ConvergenceStatus, OptimizationResult, Optimizer};
use crate::primitives::Vector;

/// Projected Gradient Descent for constrained optimization.
///
/// Solves problems of the form:
/// ```text
/// minimize f(x)
/// subject to x ∈ C
/// ```
///
/// where C is a convex set with efficient projection operator.
///
/// # Algorithm
///
/// ```text
/// for k = 1, 2, ..., max_iter:
///     x_k+1 = P_C(x_k - α∇f(x_k))
/// ```
///
/// where `P_C` is the projection onto constraint set C.
///
/// # Key Applications
///
/// - **Non-negative least squares**: C = {x : x ≥ 0}
/// - **Box constraints**: C = {x : l ≤ x ≤ u}
/// - **L2 ball**: C = {x : ‖x‖₂ ≤ r}
/// - **Simplex**: C = {x : x ≥ 0, Σx = 1}
///
/// # Convergence
///
/// - For convex f: O(1/k) convergence rate
/// - For strongly convex f: Linear convergence
/// - Step size α can be constant or use line search
///
/// # Example
///
/// ```
/// use aprender::optim::{ProjectedGradientDescent, prox};
/// use aprender::primitives::Vector;
///
/// // Minimize: ½‖x - c‖² subject to x ≥ 0
/// let c = Vector::from_slice(&[1.0, -2.0, 3.0, -1.0]);
///
/// let objective = |x: &Vector<f32>| {
///     let mut obj = 0.0;
///     for i in 0..x.len() {
///         let diff = x[i] - c[i];
///         obj += 0.5 * diff * diff;
///     }
///     obj
/// };
///
/// let gradient = |x: &Vector<f32>| {
///     let mut grad = Vector::zeros(x.len());
///     for i in 0..x.len() {
///         grad[i] = x[i] - c[i];
///     }
///     grad
/// };
///
/// let project = |x: &Vector<f32>| prox::nonnegative(x);
///
/// let mut pgd = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
/// let x0 = Vector::zeros(4);
/// let result = pgd.minimize(objective, gradient, project, x0);
///
/// // Solution should be max(c, 0) = [1.0, 0.0, 3.0, 0.0]
/// assert!((result.solution[0] - 1.0).abs() < 1e-4);
/// assert!(result.solution[1].abs() < 1e-4);
/// assert!((result.solution[2] - 3.0).abs() < 1e-4);
/// assert!(result.solution[3].abs() < 1e-4);
/// ```
///
/// # References
///
/// - Bertsekas (1999). "Nonlinear Programming."
/// - Beck & Teboulle (2009). "Gradient-based algorithms with applications to signal recovery."
#[derive(Debug, Clone)]
pub struct ProjectedGradientDescent {
    /// Maximum number of iterations
    max_iter: usize,
    /// Step size (learning rate)
    step_size: f32,
    /// Convergence tolerance
    tol: f32,
    /// Use backtracking line search
    use_line_search: bool,
    /// Backtracking parameter (0 < beta < 1)
    beta: f32,
}

impl ProjectedGradientDescent {
    /// Creates a new Projected Gradient Descent optimizer.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum number of iterations
    /// * `step_size` - Initial step size (fixed if line search disabled)
    /// * `tol` - Convergence tolerance
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::ProjectedGradientDescent;
    ///
    /// let optimizer = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
    /// ```
    #[must_use]
    pub fn new(max_iter: usize, step_size: f32, tol: f32) -> Self {
        Self {
            max_iter,
            step_size,
            tol,
            use_line_search: false,
            beta: 0.5,
        }
    }

    /// Enables backtracking line search.
    ///
    /// # Arguments
    ///
    /// * `beta` - Backtracking parameter (0 < beta < 1, typically 0.5)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::ProjectedGradientDescent;
    ///
    /// let optimizer = ProjectedGradientDescent::new(1000, 1.0, 1e-6)
    ///     .with_line_search(0.5);
    /// ```
    #[must_use]
    pub fn with_line_search(mut self, beta: f32) -> Self {
        self.use_line_search = true;
        self.beta = beta;
        self
    }

    /// Minimizes objective function subject to projection constraint.
    ///
    /// # Arguments
    ///
    /// * `objective` - Objective function f(x)
    /// * `gradient` - Gradient function ∇f(x)
    /// * `project` - Projection operator `P_C(x)`
    /// * `x0` - Initial point
    ///
    /// # Returns
    ///
    /// Optimization result with converged solution
    pub fn minimize<F, G, P>(
        &mut self,
        objective: F,
        gradient: G,
        project: P,
        x0: Vector<f32>,
    ) -> OptimizationResult
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
        P: Fn(&Vector<f32>) -> Vector<f32>,
    {
        let start_time = std::time::Instant::now();

        let mut x = x0;
        let mut alpha = self.step_size;

        for iter in 0..self.max_iter {
            // Compute gradient
            let grad = gradient(&x);

            // Gradient step: y = x - α∇f(x)
            let mut y = Vector::zeros(x.len());
            for i in 0..x.len() {
                y[i] = x[i] - alpha * grad[i];
            }

            // Project onto constraint set
            let x_new = project(&y);

            // Line search if enabled
            if self.use_line_search {
                let f_x = objective(&x);
                let f_x_new = objective(&x_new);

                // Backtracking: reduce step size until sufficient decrease
                let mut ls_iter = 0;
                while f_x_new > f_x && ls_iter < 20 {
                    alpha *= self.beta;

                    for i in 0..x.len() {
                        y[i] = x[i] - alpha * grad[i];
                    }
                    let x_new_ls = project(&y);

                    if objective(&x_new_ls) <= f_x {
                        break;
                    }
                    ls_iter += 1;
                }
            }

            // Check convergence
            let mut diff_norm = 0.0;
            for i in 0..x.len() {
                let diff = x_new[i] - x[i];
                diff_norm += diff * diff;
            }
            diff_norm = diff_norm.sqrt();

            // Compute gradient norm at new point
            let grad_new = gradient(&x_new);
            let mut grad_norm = 0.0;
            for i in 0..grad_new.len() {
                grad_norm += grad_new[i] * grad_new[i];
            }
            grad_norm = grad_norm.sqrt();

            x = x_new;

            if diff_norm < self.tol {
                let final_obj = objective(&x);
                return OptimizationResult {
                    solution: x,
                    objective_value: final_obj,
                    iterations: iter + 1,
                    status: ConvergenceStatus::Converged,
                    gradient_norm: grad_norm,
                    constraint_violation: 0.0,
                    elapsed_time: start_time.elapsed(),
                };
            }
        }

        // Max iterations reached
        let final_obj = objective(&x);
        let grad_final = gradient(&x);
        let mut grad_norm = 0.0;
        for i in 0..grad_final.len() {
            grad_norm += grad_final[i] * grad_final[i];
        }
        grad_norm = grad_norm.sqrt();

        OptimizationResult {
            solution: x,
            objective_value: final_obj,
            iterations: self.max_iter,
            status: ConvergenceStatus::MaxIterations,
            gradient_norm: grad_norm,
            constraint_violation: 0.0,
            elapsed_time: start_time.elapsed(),
        }
    }
}

impl Optimizer for ProjectedGradientDescent {
    fn step(&mut self, _params: &mut Vector<f32>, _gradients: &Vector<f32>) {
        unimplemented!(
            "Projected Gradient Descent does not support stochastic updates (step). Use minimize() with projection operator."
        )
    }

    fn reset(&mut self) {
        // Reset step size to initial value
        // Note: step_size is not stored separately, so nothing to reset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::prox;

    #[test]
    fn test_pgd_new() {
        let pgd = ProjectedGradientDescent::new(500, 0.05, 1e-4);
        // Verify defaults via Debug output
        let debug_str = format!("{:?}", pgd);
        assert!(debug_str.contains("ProjectedGradientDescent"));
        assert!(debug_str.contains("500"));
    }

    #[test]
    fn test_pgd_clone() {
        let pgd = ProjectedGradientDescent::new(100, 0.1, 1e-6);
        let cloned = pgd.clone();
        let d1 = format!("{:?}", pgd);
        let d2 = format!("{:?}", cloned);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_pgd_with_line_search() {
        let pgd = ProjectedGradientDescent::new(100, 1.0, 1e-6).with_line_search(0.5);
        let debug_str = format!("{:?}", pgd);
        assert!(debug_str.contains("use_line_search: true"));
    }

    #[test]
    fn test_pgd_nonnegative_least_squares() {
        let c = Vector::from_slice(&[1.0, -2.0, 3.0, -1.0]);
        let objective = |x: &Vector<f32>| {
            let mut obj = 0.0;
            for i in 0..x.len() {
                let diff = x[i] - c[i];
                obj += 0.5 * diff * diff;
            }
            obj
        };
        let gradient = |x: &Vector<f32>| {
            let mut grad = Vector::zeros(x.len());
            for i in 0..x.len() {
                grad[i] = x[i] - c[i];
            }
            grad
        };
        let project = |x: &Vector<f32>| prox::nonnegative(x);

        let mut pgd = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
        let x0 = Vector::zeros(4);
        let result = pgd.minimize(objective, gradient, project, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!((result.solution[0] - 1.0).abs() < 1e-4);
        assert!(result.solution[1].abs() < 1e-4);
        assert!((result.solution[2] - 3.0).abs() < 1e-4);
        assert!(result.solution[3].abs() < 1e-4);
    }

    #[test]
    fn test_pgd_max_iterations() {
        // Tiny step size + tight tolerance = never converges in 3 iterations
        let objective = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        let project = |x: &Vector<f32>| x.clone();

        let mut pgd = ProjectedGradientDescent::new(3, 0.001, 1e-20);
        let x0 = Vector::from_slice(&[10.0, 10.0]);
        let result = pgd.minimize(objective, gradient, project, x0);

        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
        assert_eq!(result.iterations, 3);
        assert!(result.gradient_norm >= 0.0);
        assert!((result.constraint_violation - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pgd_line_search_with_decrease() {
        // Use line search on a problem where initial step is too large
        let objective = |x: &Vector<f32>| x[0] * x[0];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let project = |x: &Vector<f32>| x.clone();

        let mut pgd = ProjectedGradientDescent::new(1000, 10.0, 1e-6).with_line_search(0.5);
        let x0 = Vector::from_slice(&[5.0]);
        let result = pgd.minimize(objective, gradient, project, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!(result.solution[0].abs() < 1e-3);
    }

    #[test]
    fn test_pgd_line_search_backtracking_exhaustion() {
        // Objective that makes backtracking hit 20 iterations
        // The objective improves slowly so Armijo is hard to satisfy at the wrong step
        let objective = |x: &Vector<f32>| (x[0] - 3.0).powi(2);
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 3.0)]);
        let project = |x: &Vector<f32>| prox::nonnegative(x);

        // Use a very large initial step size to exercise the line search loop
        let mut pgd = ProjectedGradientDescent::new(500, 100.0, 1e-6).with_line_search(0.5);
        let x0 = Vector::from_slice(&[0.0]);
        let result = pgd.minimize(objective, gradient, project, x0);

        // Should still converge
        assert!((result.solution[0] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_pgd_box_constraint() {
        let lower = Vector::from_slice(&[0.0, 0.0]);
        let upper = Vector::from_slice(&[1.0, 1.0]);

        let objective = |x: &Vector<f32>| (x[0] - 5.0).powi(2) + (x[1] - 5.0).powi(2);
        let gradient =
            |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 5.0), 2.0 * (x[1] - 5.0)]);
        let project = move |x: &Vector<f32>| prox::project_box(x, &lower, &upper);

        let mut pgd = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[0.5, 0.5]);
        let result = pgd.minimize(objective, gradient, project, x0);

        // Optimum at boundary [1.0, 1.0]
        assert!((result.solution[0] - 1.0).abs() < 0.1);
        assert!((result.solution[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_pgd_l2_ball_constraint() {
        let objective = |x: &Vector<f32>| (x[0] - 10.0).powi(2) + (x[1] - 10.0).powi(2);
        let gradient =
            |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 10.0), 2.0 * (x[1] - 10.0)]);
        let project = |x: &Vector<f32>| prox::project_l2_ball(x, 1.0);

        let mut pgd = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[0.0, 0.0]);
        let result = pgd.minimize(objective, gradient, project, x0);

        // Solution should be on ball boundary
        let norm = (result.solution[0].powi(2) + result.solution[1].powi(2)).sqrt();
        assert!((norm - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_pgd_reset() {
        let mut pgd = ProjectedGradientDescent::new(100, 0.1, 1e-6);
        pgd.reset(); // Should not panic
    }

    #[test]
    #[should_panic(expected = "does not support stochastic updates")]
    fn test_pgd_step_panics() {
        let mut pgd = ProjectedGradientDescent::new(100, 0.1, 1e-6);
        let mut params = Vector::from_slice(&[1.0]);
        let grad = Vector::from_slice(&[0.1]);
        pgd.step(&mut params, &grad);
    }

    #[test]
    fn test_pgd_already_converged() {
        // Start at the projection of the unconstrained optimum
        let objective = |x: &Vector<f32>| x[0] * x[0];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let project = |x: &Vector<f32>| prox::nonnegative(x);

        let mut pgd = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[1e-8]);
        let result = pgd.minimize(objective, gradient, project, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!(result.iterations <= 5);
    }

    #[test]
    fn test_pgd_gradient_norm_tracked() {
        let objective = |x: &Vector<f32>| x[0] * x[0];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let project = |x: &Vector<f32>| x.clone();

        let mut pgd = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[5.0]);
        let result = pgd.minimize(objective, gradient, project, x0);

        assert!(result.gradient_norm >= 0.0);
    }

    #[test]
    fn test_pgd_objective_value_on_converge() {
        let objective = |x: &Vector<f32>| (x[0] - 2.0).powi(2);
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 2.0)]);
        let project = |x: &Vector<f32>| x.clone();

        let mut pgd = ProjectedGradientDescent::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[0.0]);
        let result = pgd.minimize(objective, gradient, project, x0);

        assert!(result.objective_value < 1e-4);
    }

    #[test]
    fn test_pgd_objective_value_on_max_iter() {
        let objective = |x: &Vector<f32>| x[0] * x[0];
        let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let project = |x: &Vector<f32>| x.clone();

        let mut pgd = ProjectedGradientDescent::new(2, 0.001, 1e-20);
        let x0 = Vector::from_slice(&[100.0]);
        let result = pgd.minimize(objective, gradient, project, x0);

        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
        assert!(result.objective_value.is_finite());
    }
}
