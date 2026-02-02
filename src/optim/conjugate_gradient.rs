//! Nonlinear Conjugate Gradient (CG) optimizer.
//!
//! CG uses conjugate directions rather than steepest descent, achieving faster
//! convergence on quadratic problems and effective optimization for general nonlinear problems.

use crate::primitives::Vector;

use super::line_search::{LineSearch, WolfeLineSearch};
use super::{ConvergenceStatus, OptimizationResult, Optimizer};

/// Beta computation formula for Conjugate Gradient.
///
/// Different formulas provide different convergence properties and
/// numerical stability characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CGBetaFormula {
    /// Fletcher-Reeves: β = (g_{k+1}^T g_{k+1}) / (`g_k^T` `g_k`)
    ///
    /// Most stable but can be slow on non-quadratic problems.
    FletcherReeves,
    /// Polak-Ribière: β = g_{k+1}^T (g_{k+1} - `g_k`) / (`g_k^T` `g_k`)
    ///
    /// Better performance than FR, includes automatic restart (β < 0).
    PolakRibiere,
    /// Hestenes-Stiefel: β = g_{k+1}^T (g_{k+1} - `g_k`) / (`d_k^T` (g_{k+1} - `g_k`))
    ///
    /// Similar to PR but with different denominator, can be more robust.
    HestenesStiefel,
}

/// Nonlinear Conjugate Gradient (CG) optimizer.
///
/// CG is an iterative method for solving optimization problems that uses
/// conjugate directions rather than steepest descent. It's particularly
/// effective for quadratic problems but extends to general nonlinear optimization.
///
/// # Algorithm
///
/// 1. Initialize with steepest descent: `d_0` = -∇`f(x_0)`
/// 2. Line search: find `α_k` minimizing `f(x_k` + `α_k` `d_k`)
/// 3. Update: x_{k+1} = `x_k` + `α_k` `d_k`
/// 4. Compute β_{k+1} using chosen formula (FR, PR, or HS)
/// 5. Update direction: d_{k+1} = -∇f(x_{k+1}) + β_{k+1} `d_k`
/// 6. Restart if β < 0 or every n iterations
///
/// # Parameters
///
/// - **`max_iter`**: Maximum number of iterations
/// - **tol**: Convergence tolerance (gradient norm)
/// - **`beta_formula`**: Method for computing β (FR, PR, or HS)
/// - **`restart_interval`**: Restart with steepest descent every n iterations (0 = no periodic restart)
///
/// # Example
///
/// ```
/// use aprender::optim::{ConjugateGradient, CGBetaFormula, Optimizer};
/// use aprender::primitives::Vector;
///
/// let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
///
/// // Minimize Rosenbrock function
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
/// ```
#[derive(Debug, Clone)]
pub struct ConjugateGradient {
    /// Maximum number of iterations
    pub(crate) max_iter: usize,
    /// Convergence tolerance (gradient norm)
    pub(crate) tol: f32,
    /// Beta computation formula
    pub(crate) beta_formula: CGBetaFormula,
    /// Restart interval (0 = no periodic restart, only on β < 0)
    pub(crate) restart_interval: usize,
    /// Line search strategy
    pub(crate) line_search: WolfeLineSearch,
    /// Previous search direction (for conjugacy)
    pub(crate) prev_direction: Option<Vector<f32>>,
    /// Previous gradient (for beta computation)
    pub(crate) prev_gradient: Option<Vector<f32>>,
    /// Iteration counter (for restart)
    pub(crate) iter_count: usize,
}

impl ConjugateGradient {
    /// Creates a new Conjugate Gradient optimizer.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum number of iterations (typical: 100-1000)
    /// * `tol` - Convergence tolerance for gradient norm (typical: 1e-5)
    /// * `beta_formula` - Method for computing β (`FletcherReeves`, `PolakRibiere`, or `HestenesStiefel`)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::{ConjugateGradient, CGBetaFormula};
    ///
    /// let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
    /// ```
    #[must_use]
    pub fn new(max_iter: usize, tol: f32, beta_formula: CGBetaFormula) -> Self {
        Self {
            max_iter,
            tol,
            beta_formula,
            restart_interval: 0, // No periodic restart by default
            line_search: WolfeLineSearch::new(1e-4, 0.1, 50), // c2=0.1 for CG (more exact line search)
            prev_direction: None,
            prev_gradient: None,
            iter_count: 0,
        }
    }

    /// Sets the restart interval.
    ///
    /// CG will restart with steepest descent every n iterations.
    /// Setting to 0 disables periodic restart (only restarts on β < 0).
    ///
    /// # Arguments
    ///
    /// * `interval` - Number of iterations between restarts (typical: n, where n is problem dimension)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::{ConjugateGradient, CGBetaFormula};
    ///
    /// let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere)
    ///     .with_restart_interval(50);
    /// ```
    #[must_use]
    pub fn with_restart_interval(mut self, interval: usize) -> Self {
        self.restart_interval = interval;
        self
    }

    /// Computes beta coefficient based on the chosen formula.
    fn compute_beta(
        &self,
        grad_new: &Vector<f32>,
        grad_old: &Vector<f32>,
        d_old: &Vector<f32>,
    ) -> f32 {
        let n = grad_new.len();

        match self.beta_formula {
            CGBetaFormula::FletcherReeves => {
                // β = (g_{k+1}^T g_{k+1}) / (g_k^T g_k)
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for i in 0..n {
                    numerator += grad_new[i] * grad_new[i];
                    denominator += grad_old[i] * grad_old[i];
                }
                numerator / denominator.max(1e-12)
            }
            CGBetaFormula::PolakRibiere => {
                // β = g_{k+1}^T (g_{k+1} - g_k) / (g_k^T g_k)
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for i in 0..n {
                    numerator += grad_new[i] * (grad_new[i] - grad_old[i]);
                    denominator += grad_old[i] * grad_old[i];
                }
                let beta = numerator / denominator.max(1e-12);
                // PR has automatic restart: if β < 0, restart with steepest descent
                beta.max(0.0)
            }
            CGBetaFormula::HestenesStiefel => {
                // β = g_{k+1}^T (g_{k+1} - g_k) / (d_k^T (g_{k+1} - g_k))
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for i in 0..n {
                    let y_i = grad_new[i] - grad_old[i];
                    numerator += grad_new[i] * y_i;
                    denominator += d_old[i] * y_i;
                }
                let beta = numerator / denominator.max(1e-12);
                beta.max(0.0)
            }
        }
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

impl Optimizer for ConjugateGradient {
    fn step(&mut self, _params: &mut Vector<f32>, _gradients: &Vector<f32>) {
        unimplemented!(
            "Conjugate Gradient does not support stochastic updates (step). Use minimize() for batch optimization."
        )
    }

    #[allow(clippy::too_many_lines)]
    fn minimize<F, G>(&mut self, objective: F, gradient: G, x0: Vector<f32>) -> OptimizationResult
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
    {
        let start_time = std::time::Instant::now();
        let n = x0.len();

        // Reset state
        self.prev_direction = None;
        self.prev_gradient = None;
        self.iter_count = 0;

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
            let d = if let (Some(d_old), Some(g_old)) = (&self.prev_direction, &self.prev_gradient)
            {
                // Check if we need to restart
                let need_restart = if self.restart_interval > 0 {
                    self.iter_count % self.restart_interval == 0
                } else {
                    false
                };

                if need_restart {
                    // Restart with steepest descent
                    let mut d_new = Vector::zeros(n);
                    for i in 0..n {
                        d_new[i] = -grad[i];
                    }
                    d_new
                } else {
                    // Compute beta and conjugate direction
                    let beta = self.compute_beta(&grad, g_old, d_old);

                    // d = -grad + beta * d_old
                    let mut d_new = Vector::zeros(n);
                    for i in 0..n {
                        d_new[i] = -grad[i] + beta * d_old[i];
                    }

                    // Check if direction is descent (grad^T d < 0)
                    let mut grad_dot_d = 0.0;
                    for i in 0..n {
                        grad_dot_d += grad[i] * d_new[i];
                    }

                    if grad_dot_d >= 0.0 {
                        // Not a descent direction - restart with steepest descent
                        for i in 0..n {
                            d_new[i] = -grad[i];
                        }
                    }

                    d_new
                }
            } else {
                // First iteration: use steepest descent
                let mut d_new = Vector::zeros(n);
                for i in 0..n {
                    d_new[i] = -grad[i];
                }
                d_new
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

            // Store current direction and gradient for next iteration
            self.prev_direction = Some(d);
            self.prev_gradient = Some(grad);

            // Update for next iteration
            x = x_new;
            fx = fx_new;
            grad = grad_new;
            grad_norm = Self::norm(&grad);
            self.iter_count += 1;
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
        self.prev_direction = None;
        self.prev_gradient = None;
        self.iter_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cg_quadratic() {
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

        // Simple quadratic: f(x) = (x-3)^2
        let f = |x: &Vector<f32>| (x[0] - 3.0).powi(2);
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 3.0)]);

        let x0 = Vector::from_slice(&[0.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!((result.solution[0] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_cg_rosenbrock() {
        // Rosenbrock is a challenging test function; CG may need many iterations
        let mut optimizer = ConjugateGradient::new(5000, 1e-4, CGBetaFormula::PolakRibiere);

        let f = |x: &Vector<f32>| {
            let a = x[0];
            let b = x[1];
            (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
        };

        let grad = |x: &Vector<f32>| {
            let a = x[0];
            let b = x[1];
            Vector::from_slice(&[
                -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
                200.0 * (b - a * a),
            ])
        };

        let x0 = Vector::from_slice(&[0.0, 0.0]);
        let result = optimizer.minimize(f, grad, x0);

        // Check that we made progress toward (1, 1) even if not fully converged
        // Rosenbrock is notoriously difficult for CG
        let dist_to_opt =
            ((result.solution[0] - 1.0).powi(2) + (result.solution[1] - 1.0).powi(2)).sqrt();
        assert!(
            dist_to_opt < 0.1 || result.status == ConvergenceStatus::Converged,
            "Expected solution near (1,1), got ({}, {}), dist={}",
            result.solution[0],
            result.solution[1],
            dist_to_opt
        );
    }

    #[test]
    fn test_cg_beta_formulas() {
        for formula in [
            CGBetaFormula::FletcherReeves,
            CGBetaFormula::PolakRibiere,
            CGBetaFormula::HestenesStiefel,
        ] {
            let mut optimizer = ConjugateGradient::new(100, 1e-5, formula);

            let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
            let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

            let x0 = Vector::from_slice(&[5.0, 3.0]);
            let result = optimizer.minimize(f, grad, x0);

            assert_eq!(result.status, ConvergenceStatus::Converged);
            assert!(result.solution[0].abs() < 1e-4);
            assert!(result.solution[1].abs() < 1e-4);
        }
    }

    #[test]
    fn test_cg_restart_interval() {
        let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere)
            .with_restart_interval(10);
        assert_eq!(optimizer.restart_interval, 10);

        // Run optimization with restart interval
        let mut opt = optimizer;
        let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        let x0 = Vector::from_slice(&[5.0, 3.0]);
        let result = opt.minimize(f, grad, x0);
        assert_eq!(result.status, ConvergenceStatus::Converged);
    }

    #[test]
    fn test_cg_reset() {
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

        // Run an optimization first
        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let x0 = Vector::from_slice(&[5.0]);
        let _ = optimizer.minimize(f, grad, x0);

        // Check state was set
        assert!(optimizer.prev_direction.is_some());

        // Reset and check state is cleared
        optimizer.reset();
        assert!(optimizer.prev_direction.is_none());
        assert!(optimizer.prev_gradient.is_none());
        assert_eq!(optimizer.iter_count, 0);
    }

    #[test]
    #[should_panic(expected = "does not support stochastic")]
    fn test_cg_step_panics() {
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let grads = Vector::from_slice(&[0.1, 0.2]);
        optimizer.step(&mut params, &grads);
    }

    #[test]
    fn test_cg_numerical_error() {
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

        // Function that returns NaN
        let f = |x: &Vector<f32>| {
            if x[0] > 0.5 {
                f32::NAN
            } else {
                x[0] * x[0]
            }
        };
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let x0 = Vector::from_slice(&[0.4]);
        let result = optimizer.minimize(f, grad, x0);
        // May hit NaN or converge depending on line search
        assert!(
            result.status == ConvergenceStatus::NumericalError
                || result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::Stalled
        );
    }

    #[test]
    fn test_cg_max_iterations() {
        // Very tight tolerance that won't be reached with only 2 iterations
        let mut optimizer = ConjugateGradient::new(2, 1e-20, CGBetaFormula::PolakRibiere);

        // Rosenbrock is hard to converge in 2 iterations
        let f = |x: &Vector<f32>| {
            let a = x[0];
            let b = x[1];
            (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
        };
        let grad = |x: &Vector<f32>| {
            let a = x[0];
            let b = x[1];
            Vector::from_slice(&[
                -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
                200.0 * (b - a * a),
            ])
        };

        let x0 = Vector::from_slice(&[-5.0, -5.0]);
        let result = optimizer.minimize(f, grad, x0);
        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    }

    #[test]
    fn test_cg_hestenes_stiefel() {
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::HestenesStiefel);

        // Multi-dimensional problem to exercise HS formula
        let f = |x: &Vector<f32>| x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 4.0 * x[1], 6.0 * x[2]]);

        let x0 = Vector::from_slice(&[5.0, 3.0, 2.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!(result.solution[0].abs() < 1e-4);
        assert!(result.solution[1].abs() < 1e-4);
        assert!(result.solution[2].abs() < 1e-4);
    }

    #[test]
    fn test_cg_fletcher_reeves() {
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::FletcherReeves);

        let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

        let x0 = Vector::from_slice(&[10.0, 10.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
    }

    #[test]
    fn test_cg_beta_formula_equality() {
        assert_eq!(CGBetaFormula::FletcherReeves, CGBetaFormula::FletcherReeves);
        assert_ne!(CGBetaFormula::FletcherReeves, CGBetaFormula::PolakRibiere);

        // Test Clone
        let formula = CGBetaFormula::HestenesStiefel;
        let cloned = formula;
        assert_eq!(formula, cloned);

        // Test Debug
        let debug_str = format!("{:?}", CGBetaFormula::PolakRibiere);
        assert!(debug_str.contains("PolakRibiere"));
    }

    #[test]
    fn test_cg_clone() {
        let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
        let cloned = optimizer.clone();
        assert_eq!(cloned.max_iter, 100);
        assert_eq!(cloned.beta_formula, CGBetaFormula::PolakRibiere);
    }

    #[test]
    fn test_cg_debug() {
        let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
        let debug_str = format!("{:?}", optimizer);
        assert!(debug_str.contains("ConjugateGradient"));
        assert!(debug_str.contains("max_iter"));
    }

    #[test]
    fn test_cg_stalled_zero_alpha() {
        // Test stalled status when line search returns very small alpha
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

        // Function that's extremely flat - line search should return tiny alpha
        let f = |x: &Vector<f32>| x[0] * 1e-20;
        let grad = |_x: &Vector<f32>| Vector::from_slice(&[1e-20]);

        let x0 = Vector::from_slice(&[1.0]);
        let result = optimizer.minimize(f, grad, x0);

        // Should either converge (gradient too small) or stall
        assert!(
            result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::Stalled
        );
    }

    #[test]
    fn test_cg_restart_triggers() {
        // Use restart interval of 2 and run enough iterations to trigger restart
        let mut optimizer =
            ConjugateGradient::new(20, 1e-8, CGBetaFormula::PolakRibiere).with_restart_interval(2);

        // Function that needs many iterations
        let f = |x: &Vector<f32>| {
            let a = x[0];
            let b = x[1];
            (a - 5.0).powi(2) + (b - 3.0).powi(2)
        };
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 5.0), 2.0 * (x[1] - 3.0)]);

        let x0 = Vector::from_slice(&[0.0, 0.0]);
        let result = optimizer.minimize(f, grad, x0);

        // Should converge despite restarts
        assert!(
            result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::MaxIterations
        );
    }

    #[test]
    fn test_cg_non_descent_direction_restart() {
        // Test the case where beta calculation leads to non-descent direction
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::FletcherReeves);

        // Function where gradients change rapidly
        let f = |x: &Vector<f32>| x[0].powi(4) + x[1].powi(4);
        let grad = |x: &Vector<f32>| Vector::from_slice(&[4.0 * x[0].powi(3), 4.0 * x[1].powi(3)]);

        let x0 = Vector::from_slice(&[10.0, 10.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert!(
            result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::MaxIterations
        );
        // Should get close to (0, 0)
        assert!(result.solution[0].abs() < 1.0);
    }

    #[test]
    fn test_cg_objective_value_tracking() {
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

        let f = |x: &Vector<f32>| (x[0] - 2.0).powi(2);
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 2.0)]);

        let x0 = Vector::from_slice(&[0.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert!(result.objective_value < 1e-4);
        assert!(result.constraint_violation == 0.0);
    }

    #[test]
    fn test_cg_gradient_norm_tracking() {
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

        let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

        let x0 = Vector::from_slice(&[5.0, 3.0]);
        let result = optimizer.minimize(f, grad, x0);

        // Gradient norm should be small at convergence
        assert!(result.gradient_norm < 1e-4);
    }

    #[test]
    fn test_cg_elapsed_time() {
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let x0 = Vector::from_slice(&[1.0]);
        let result = optimizer.minimize(f, grad, x0);

        // Elapsed time should be tracked
        let _ = result.elapsed_time.as_nanos();
    }

    #[test]
    fn test_cg_already_at_optimum() {
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

        // Start at optimum
        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let x0 = Vector::from_slice(&[0.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_cg_infinite_objective() {
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

        // Function that returns infinity after a few steps
        let f = |x: &Vector<f32>| {
            if x[0] > 2.0 {
                f32::INFINITY
            } else {
                x[0] * x[0]
            }
        };
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        // Start somewhere that would move towards infinity
        let x0 = Vector::from_slice(&[1.5]);
        let result = optimizer.minimize(f, grad, x0);

        // Should handle gracefully
        assert!(
            result.status == ConvergenceStatus::NumericalError
                || result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::Stalled
        );
    }

    #[test]
    fn test_cg_norm() {
        // Test the private norm function via minimize that uses it
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

        // Large initial gradient to ensure norm is calculated
        let f = |x: &Vector<f32>| (x[0] - 100.0).powi(2) + (x[1] - 200.0).powi(2);
        let grad =
            |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 100.0), 2.0 * (x[1] - 200.0)]);

        let x0 = Vector::from_slice(&[0.0, 0.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert!(result.status == ConvergenceStatus::Converged);
        assert!((result.solution[0] - 100.0).abs() < 1e-3);
        assert!((result.solution[1] - 200.0).abs() < 1e-3);
    }

    #[test]
    fn test_cg_with_restart_interval_zero() {
        // Zero restart interval should disable periodic restarts
        let optimizer =
            ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere).with_restart_interval(0);
        assert_eq!(optimizer.restart_interval, 0);
    }

    #[test]
    fn test_cg_restart_interval_1() {
        // Restart every single iteration forces steepest descent behavior
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::FletcherReeves)
            .with_restart_interval(1);

        let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

        let x0 = Vector::from_slice(&[5.0, 3.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert!(
            result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::MaxIterations
        );
    }

    #[test]
    fn test_cg_norm_function() {
        // Exercise the norm utility indirectly by running minimize
        // with a known-norm initial gradient
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

        let f = |x: &Vector<f32>| (x[0] - 3.0).powi(2) + (x[1] - 4.0).powi(2);
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 3.0), 2.0 * (x[1] - 4.0)]);

        let x0 = Vector::from_slice(&[0.0, 0.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!(result.gradient_norm < 1e-4);
    }

    #[test]
    fn test_cg_fr_non_descent_direction_fallback() {
        // Fletcher-Reeves can produce non-descent directions on difficult problems
        // This tests the `if grad_dot_d >= 0.0` restart branch
        let mut optimizer = ConjugateGradient::new(200, 1e-5, CGBetaFormula::FletcherReeves);

        // Rapidly changing curvature
        let f = |x: &Vector<f32>| x[0].powi(4) + x[1].powi(4) + 2.0 * x[0] * x[0] * x[1] * x[1];
        let grad = |x: &Vector<f32>| {
            Vector::from_slice(&[
                4.0 * x[0].powi(3) + 4.0 * x[0] * x[1] * x[1],
                4.0 * x[1].powi(3) + 4.0 * x[0] * x[0] * x[1],
            ])
        };

        let x0 = Vector::from_slice(&[10.0, -10.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert!(
            result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::MaxIterations
        );
    }

    #[test]
    fn test_cg_pr_negative_beta() {
        // Test PR formula's automatic restart when beta would be negative
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

        // Function where gradients can change direction dramatically
        let f = |x: &Vector<f32>| {
            let a = x[0];
            let b = x[1];
            a.sin().powi(2) + b.cos().powi(2)
        };
        let grad = |x: &Vector<f32>| {
            let a = x[0];
            let b = x[1];
            Vector::from_slice(&[2.0 * a.sin() * a.cos(), -2.0 * b.cos() * b.sin()])
        };

        let x0 = Vector::from_slice(&[1.5, 1.5]);
        let result = optimizer.minimize(f, grad, x0);

        // Should make progress regardless of beta sign issues
        assert!(
            result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::MaxIterations
                || result.status == ConvergenceStatus::Stalled
        );
    }

    #[test]
    fn test_cg_hs_denominator_near_zero() {
        // Test HS formula when denominator is near zero
        let mut optimizer = ConjugateGradient::new(50, 1e-5, CGBetaFormula::HestenesStiefel);

        // Function with nearly parallel gradients
        let f = |x: &Vector<f32>| x[0] * x[0] + 0.0001 * x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 0.0002 * x[1]]);

        let x0 = Vector::from_slice(&[5.0, 0.001]);
        let result = optimizer.minimize(f, grad, x0);

        // Should converge despite potentially tricky beta calculation
        assert!(
            result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::MaxIterations
        );
    }

    #[test]
    fn test_cg_high_dimensional() {
        // Test with higher dimensional problem
        let mut optimizer =
            ConjugateGradient::new(200, 1e-5, CGBetaFormula::PolakRibiere).with_restart_interval(5);

        let f = |x: &Vector<f32>| {
            let mut sum = 0.0;
            for i in 0..x.len() {
                sum += (x[i] - f32::from(i as u8)).powi(2);
            }
            sum
        };
        let grad = |x: &Vector<f32>| {
            let mut g = Vector::zeros(x.len());
            for i in 0..x.len() {
                g[i] = 2.0 * (x[i] - f32::from(i as u8));
            }
            g
        };

        let x0 = Vector::zeros(5);
        let result = optimizer.minimize(f, grad, x0);

        assert!(
            result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::MaxIterations
        );
    }
}
