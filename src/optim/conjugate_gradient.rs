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

    /// Returns the negated gradient as the steepest descent direction.
    fn steepest_descent(grad: &Vector<f32>, n: usize) -> Vector<f32> {
        let mut d = Vector::zeros(n);
        for i in 0..n {
            d[i] = -grad[i];
        }
        d
    }

    /// Computes the conjugate search direction for the current iteration.
    ///
    /// On the first iteration (no previous state), returns steepest descent.
    /// On subsequent iterations, computes the conjugate direction using β,
    /// with automatic restart when:
    /// - The periodic restart interval is hit
    /// - The computed direction is not a descent direction (grad^T d >= 0)
    fn compute_search_direction(&self, grad: &Vector<f32>, n: usize) -> Vector<f32> {
        let (d_old, g_old) = match (&self.prev_direction, &self.prev_gradient) {
            (Some(d), Some(g)) => (d, g),
            _ => return Self::steepest_descent(grad, n),
        };

        // Check if periodic restart is needed
        if self.restart_interval > 0 && self.iter_count % self.restart_interval == 0 {
            return Self::steepest_descent(grad, n);
        }

        // Compute beta and conjugate direction: d = -grad + beta * d_old
        let beta = self.compute_beta(grad, g_old, d_old);
        let mut d_new = Vector::zeros(n);
        for i in 0..n {
            d_new[i] = -grad[i] + beta * d_old[i];
        }

        // Verify descent direction (grad^T d < 0); restart if not
        let mut grad_dot_d = 0.0;
        for i in 0..n {
            grad_dot_d += grad[i] * d_new[i];
        }
        if grad_dot_d >= 0.0 {
            return Self::steepest_descent(grad, n);
        }

        d_new
    }

    /// Builds an `OptimizationResult` with the given parameters.
    ///
    /// `constraint_violation` is always 0.0 for unconstrained CG.
    fn make_result(
        solution: Vector<f32>,
        objective_value: f32,
        iterations: usize,
        status: ConvergenceStatus,
        gradient_norm: f32,
        elapsed_time: std::time::Duration,
    ) -> OptimizationResult {
        OptimizationResult {
            solution,
            objective_value,
            iterations,
            status,
            gradient_norm,
            constraint_violation: 0.0,
            elapsed_time,
        }
    }
}

impl Optimizer for ConjugateGradient {
    fn step(&mut self, _params: &mut Vector<f32>, _gradients: &Vector<f32>) {
        panic!(
            "Conjugate Gradient does not support stochastic updates (step). Use minimize() for batch optimization."
        )
    }

    // Contract: optimization-v1, equation = "cg_minimize"
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
                return Self::make_result(
                    x,
                    fx,
                    iter,
                    ConvergenceStatus::Converged,
                    grad_norm,
                    start_time.elapsed(),
                );
            }

            // Compute search direction (delegates restart/conjugacy logic)
            let d = self.compute_search_direction(&grad, n);

            // Line search
            let alpha = self.line_search.search(&objective, &gradient, &x, &d);

            // Check for stalled progress
            if alpha < 1e-12 {
                return Self::make_result(
                    x,
                    fx,
                    iter,
                    ConvergenceStatus::Stalled,
                    grad_norm,
                    start_time.elapsed(),
                );
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
                return Self::make_result(
                    x,
                    fx,
                    iter,
                    ConvergenceStatus::NumericalError,
                    grad_norm,
                    start_time.elapsed(),
                );
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
        Self::make_result(
            x,
            fx,
            self.max_iter,
            ConvergenceStatus::MaxIterations,
            grad_norm,
            start_time.elapsed(),
        )
    }

    fn reset(&mut self) {
        self.prev_direction = None;
        self.prev_gradient = None;
        self.iter_count = 0;
    }
}

#[cfg(test)]
#[path = "conjugate_gradient_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_cg_contract.rs"]
mod tests_cg_contract;
