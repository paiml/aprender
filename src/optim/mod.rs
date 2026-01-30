//! Optimization algorithms for gradient-based learning.
//!
//! This module provides both stochastic (mini-batch) and batch (deterministic) optimizers
//! following the unified [`Optimizer`] trait architecture.
//!
//! # Available Optimizers
//!
//! ## Stochastic (Mini-Batch) Optimizers
//! - [`SGD`] - Stochastic Gradient Descent with optional momentum
//! - [`Adam`] - Adaptive Moment Estimation (adaptive learning rates)
//!
//! ## Batch (Deterministic) Optimizers
//! - [`LBFGS`] - Limited-memory BFGS (memory-efficient quasi-Newton)
//! - [`ConjugateGradient`] - Conjugate Gradient with three beta formulas
//! - [`DampedNewton`] - Newton's method with automatic damping for stability
//!
//! ## Convex Optimization (Phase 2)
//! - [`FISTA`] - Fast Iterative Shrinkage-Thresholding (proximal gradient)
//! - [`CoordinateDescent`] - Coordinate-wise optimization for high dimensions
//! - [`ADMM`] - Alternating Direction Method of Multipliers (distributed ML)
//!
//! ## Constrained Optimization (Phase 3)
//! - [`ProjectedGradientDescent`] - Projection onto convex sets
//! - [`AugmentedLagrangian`] - Equality-constrained optimization
//! - [`InteriorPoint`] - Inequality-constrained optimization (log-barrier)
//!
//! ## Line Search Strategies
//! - [`BacktrackingLineSearch`] - Simple Armijo condition (sufficient decrease)
//! - [`WolfeLineSearch`] - Armijo + curvature conditions (for quasi-Newton methods)
//!
//! ## Utility Functions
//! - [`safe_cholesky_solve`] - Cholesky solver with automatic regularization
//!
//! # Stochastic Optimization (Mini-Batch)
//!
//! Stochastic optimizers update parameters incrementally using mini-batch gradients.
//! Use the [`Optimizer::step`] method for parameter updates:
//!
//! ```
//! use aprender::optim::SGD;
//! use aprender::primitives::Vector;
//!
//! // Create optimizer with learning rate 0.01
//! let mut optimizer = SGD::new(0.01);
//!
//! // Initialize parameters and gradients
//! let mut params = Vector::from_slice(&[1.0, 2.0, 3.0]);
//! let gradients = Vector::from_slice(&[0.1, 0.2, 0.3]);
//!
//! // Update parameters
//! optimizer.step(&mut params, &gradients);
//!
//! // Parameters are updated: params = params - lr * gradients
//! assert!((params[0] - 0.999).abs() < 1e-6);
//! ```
//!
//! # Batch Optimization (Full Dataset)
//!
//! Batch optimizers minimize objective functions using full dataset access.
//! They use the `minimize` method which returns detailed convergence information:
//!
//! ```
//! use aprender::optim::{LBFGS, ConvergenceStatus, Optimizer};
//! use aprender::primitives::Vector;
//!
//! // Create L-BFGS optimizer: 100 max iterations, 1e-5 tolerance, 10 memory size
//! let mut optimizer = LBFGS::new(100, 1e-5, 10);
//!
//! // Define objective and gradient functions
//! let objective = |x: &Vector<f32>| (x[0] - 5.0).powi(2) + (x[1] - 3.0).powi(2);
//! let gradient = |x: &Vector<f32>| {
//!     Vector::from_slice(&[2.0 * (x[0] - 5.0), 2.0 * (x[1] - 3.0)])
//! };
//!
//! let x0 = Vector::from_slice(&[0.0, 0.0]);
//! let result = optimizer.minimize(objective, gradient, x0);
//!
//! assert_eq!(result.status, ConvergenceStatus::Converged);
//! assert!((result.solution[0] - 5.0).abs() < 1e-4);
//! assert!((result.solution[1] - 3.0).abs() < 1e-4);
//! ```
//!
//! # See Also
//!
//! - [`examples/batch_optimization.rs`](https://github.com/paiml/aprender/tree/main/examples/batch_optimization.rs) - Comprehensive examples
//! - Specification: `docs/specifications/comprehensive-optimization-spec.md`

use serde::{Deserialize, Serialize};

use crate::primitives::{Matrix, Vector};

// Submodules (PMAT-085: file health improvement)
mod lbfgs;
mod line_search;
mod stochastic;

// Re-exports from submodules
pub use lbfgs::LBFGS;
pub use line_search::{BacktrackingLineSearch, LineSearch, WolfeLineSearch};
pub use stochastic::{Adam, SGD};

/// Result of an optimization procedure.
///
/// Contains the final solution, convergence information, and diagnostic metrics.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Final solution (optimized parameters)
    pub solution: Vector<f32>,
    /// Final objective function value
    pub objective_value: f32,
    /// Number of iterations performed
    pub iterations: usize,
    /// Convergence status
    pub status: ConvergenceStatus,
    /// Final gradient norm (‖∇f(x)‖)
    pub gradient_norm: f32,
    /// Constraint violation (0.0 for unconstrained problems)
    pub constraint_violation: f32,
    /// Total elapsed time
    pub elapsed_time: std::time::Duration,
}

impl OptimizationResult {
    /// Creates a converged result.
    #[must_use]
    pub fn converged(solution: Vector<f32>, iterations: usize) -> Self {
        Self {
            solution,
            objective_value: 0.0,
            iterations,
            status: ConvergenceStatus::Converged,
            gradient_norm: 0.0,
            constraint_violation: 0.0,
            elapsed_time: std::time::Duration::ZERO,
        }
    }

    /// Creates a max-iterations result.
    #[must_use]
    pub fn max_iterations(solution: Vector<f32>) -> Self {
        Self {
            solution,
            objective_value: 0.0,
            iterations: 0,
            status: ConvergenceStatus::MaxIterations,
            gradient_norm: 0.0,
            constraint_violation: 0.0,
            elapsed_time: std::time::Duration::ZERO,
        }
    }
}

/// Convergence status of an optimization procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    /// Converged (gradient norm < tolerance)
    Converged,
    /// Reached maximum iteration limit
    MaxIterations,
    /// Progress stalled (step size too small)
    Stalled,
    /// Numerical error (NaN, Inf, etc.)
    NumericalError,
    /// Optimization still running
    Running,
    /// User-requested termination
    UserTerminated,
}

/// Safely solves a linear system Ax = b using Cholesky decomposition with automatic regularization.
///
/// When the matrix is not positive definite or near-singular, this function automatically
/// adds regularization (λI) to make the system solvable. This is essential for numerical
/// stability in second-order optimization methods like Newton's method.
///
/// # Algorithm
///
/// 1. Try standard Cholesky solve with A
/// 2. If it fails, add regularization: (A + λI)
/// 3. Increase λ progressively (×10) until solve succeeds or max attempts reached
///
/// # Arguments
///
/// * `A` - Symmetric matrix (should be positive definite, but may not be)
/// * `b` - Right-hand side vector
/// * `initial_lambda` - Initial regularization parameter (typical: 1e-8)
/// * `max_attempts` - Maximum regularization attempts (typical: 10)
///
/// # Returns
///
/// * `Ok(x)` - Solution vector if successful
/// * `Err(msg)` - Error message if regularization failed
///
/// # Example
///
/// ```
/// use aprender::optim::safe_cholesky_solve;
/// use aprender::primitives::{Matrix, Vector};
///
/// // Slightly ill-conditioned matrix
/// let A = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1e-8]).expect("valid dimensions");
/// let b = Vector::from_slice(&[1.0, 1.0]);
///
/// let x = safe_cholesky_solve(&A, &b, 1e-8, 10).expect("should solve with regularization");
/// assert_eq!(x.len(), 2);
/// ```
///
/// # Use Cases
///
/// - **Damped Newton**: Regularize Hessian when not positive definite
/// - **Levenberg-Marquardt**: Add damping parameter for stability
/// - **Trust region**: Solve trust region subproblem with ill-conditioned Hessian
///
/// # References
///
/// - Nocedal & Wright (2006), *Numerical Optimization*, Chapter 3
pub fn safe_cholesky_solve(
    a: &Matrix<f32>,
    b: &Vector<f32>,
    initial_lambda: f32,
    max_attempts: usize,
) -> Result<Vector<f32>, &'static str> {
    // First try without regularization
    if let Ok(x) = a.cholesky_solve(b) {
        return Ok(x);
    }

    // Matrix not positive definite, try with regularization
    solve_with_regularization(a, b, initial_lambda, max_attempts)
}

/// Solve linear system with Tikhonov regularization (A + λI)x = b.
fn solve_with_regularization(
    a: &Matrix<f32>,
    b: &Vector<f32>,
    initial_lambda: f32,
    max_attempts: usize,
) -> Result<Vector<f32>, &'static str> {
    let n = a.n_rows();
    let mut lambda = initial_lambda;

    for _attempt in 0..max_attempts {
        let a_reg = create_regularized_matrix(a, n, lambda);

        if let Ok(x) = a_reg.cholesky_solve(b) {
            return Ok(x);
        }

        lambda *= 10.0;
        if lambda > 1e6 {
            return Err(
                "Cholesky solve failed: matrix too ill-conditioned even with regularization",
            );
        }
    }

    Err("Cholesky solve failed after maximum regularization attempts")
}

/// Create regularized matrix `A_reg` = A + λI.
fn create_regularized_matrix(a: &Matrix<f32>, n: usize, lambda: f32) -> Matrix<f32> {
    let mut a_reg_data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let diagonal_term = if i == j { lambda } else { 0.0 };
            a_reg_data[i * n + j] = a.get(i, j) + diagonal_term;
        }
    }
    Matrix::from_vec(n, n, a_reg_data).expect("Matrix dimensions should be valid")
}

// LBFGS moved to lbfgs.rs (PMAT-085)

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
    max_iter: usize,
    /// Convergence tolerance (gradient norm)
    tol: f32,
    /// Beta computation formula
    beta_formula: CGBetaFormula,
    /// Restart interval (0 = no periodic restart, only on β < 0)
    restart_interval: usize,
    /// Line search strategy
    line_search: WolfeLineSearch,
    /// Previous search direction (for conjugacy)
    prev_direction: Option<Vector<f32>>,
    /// Previous gradient (for beta computation)
    prev_gradient: Option<Vector<f32>>,
    /// Iteration counter (for restart)
    iter_count: usize,
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
    max_iter: usize,
    /// Convergence tolerance (gradient norm)
    tol: f32,
    /// Finite difference step size for Hessian approximation
    epsilon: f32,
    /// Line search strategy
    line_search: BacktrackingLineSearch,
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

// ==================== Proximal Operators ====================

/// Proximal operators for non-smooth regularization.
///
/// A proximal operator for function g is defined as:
/// ```text
/// prox_g(v) = argmin_x { g(x) + ½‖x - v‖² }
/// ```
///
/// These are essential building blocks for proximal gradient methods like FISTA.
pub mod prox {
    use crate::primitives::Vector;

    /// Soft-thresholding operator for L1 regularization.
    ///
    /// Computes the proximal operator of the L1 norm: prox_{λ‖·‖₁}(v).
    ///
    /// # Formula
    ///
    /// ```text
    /// prox_{λ‖·‖₁}(v) = sign(v) ⊙ max(|v| - λ, 0)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `v` - Input vector
    /// * `lambda` - Regularization parameter (λ ≥ 0)
    ///
    /// # Returns
    ///
    /// Soft-thresholded vector
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::prox::soft_threshold;
    /// use aprender::primitives::Vector;
    ///
    /// let v = Vector::from_slice(&[2.0, -1.5, 0.5]);
    /// let result = soft_threshold(&v, 1.0);
    ///
    /// assert!((result[0] - 1.0).abs() < 1e-6);  // 2.0 - 1.0 = 1.0
    /// assert!((result[1] + 0.5).abs() < 1e-6);  // -1.5 + 1.0 = -0.5
    /// assert!(result[2].abs() < 1e-6);          // 0.5 - 1.0 = 0 (thresholded)
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Lasso regression**: Sparse linear models with L1 penalty
    /// - **Compressed sensing**: Sparse signal recovery
    /// - **Feature selection**: Automatic variable selection via sparsity
    #[must_use]
    pub fn soft_threshold(v: &Vector<f32>, lambda: f32) -> Vector<f32> {
        let mut result = Vector::zeros(v.len());
        for i in 0..v.len() {
            let val = v[i];
            result[i] = if val > lambda {
                val - lambda
            } else if val < -lambda {
                val + lambda
            } else {
                0.0
            };
        }
        result
    }

    /// Projects onto the non-negative orthant: x ≥ 0.
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector
    ///
    /// # Returns
    ///
    /// Vector with all negative components set to zero
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::prox::nonnegative;
    /// use aprender::primitives::Vector;
    ///
    /// let x = Vector::from_slice(&[1.0, -2.0, 3.0, -0.5]);
    /// let result = nonnegative(&x);
    ///
    /// assert_eq!(result[0], 1.0);
    /// assert_eq!(result[1], 0.0);
    /// assert_eq!(result[2], 3.0);
    /// assert_eq!(result[3], 0.0);
    /// ```
    #[must_use]
    pub fn nonnegative(x: &Vector<f32>) -> Vector<f32> {
        let mut result = Vector::zeros(x.len());
        for i in 0..x.len() {
            result[i] = x[i].max(0.0);
        }
        result
    }

    /// Projects onto an L2 ball: ‖x‖₂ ≤ radius.
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector
    /// * `radius` - Ball radius (r > 0)
    ///
    /// # Returns
    ///
    /// Projected vector satisfying ‖result‖₂ ≤ radius
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::prox::project_l2_ball;
    /// use aprender::primitives::Vector;
    ///
    /// let x = Vector::from_slice(&[3.0, 4.0]); // norm = 5.0
    /// let result = project_l2_ball(&x, 2.0);
    ///
    /// // Should be scaled to norm = 2.0
    /// let norm = (result[0] * result[0] + result[1] * result[1]).sqrt();
    /// assert!((norm - 2.0).abs() < 1e-5);
    /// ```
    #[must_use]
    pub fn project_l2_ball(x: &Vector<f32>, radius: f32) -> Vector<f32> {
        let mut norm_sq = 0.0;
        for i in 0..x.len() {
            norm_sq += x[i] * x[i];
        }
        let norm = norm_sq.sqrt();

        if norm <= radius {
            x.clone()
        } else {
            let scale = radius / norm;
            let mut result = Vector::zeros(x.len());
            for i in 0..x.len() {
                result[i] = scale * x[i];
            }
            result
        }
    }

    /// Projects onto box constraints: lower ≤ x ≤ upper.
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector
    /// * `lower` - Lower bounds
    /// * `upper` - Upper bounds
    ///
    /// # Returns
    ///
    /// Vector with components clipped to [lower, upper]
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::prox::project_box;
    /// use aprender::primitives::Vector;
    ///
    /// let x = Vector::from_slice(&[-1.0, 0.5, 2.0]);
    /// let lower = Vector::from_slice(&[0.0, 0.0, 0.0]);
    /// let upper = Vector::from_slice(&[1.0, 1.0, 1.0]);
    ///
    /// let result = project_box(&x, &lower, &upper);
    ///
    /// assert_eq!(result[0], 0.0);  // Clipped to lower
    /// assert_eq!(result[1], 0.5);  // Within bounds
    /// assert_eq!(result[2], 1.0);  // Clipped to upper
    /// ```
    #[must_use]
    pub fn project_box(x: &Vector<f32>, lower: &Vector<f32>, upper: &Vector<f32>) -> Vector<f32> {
        let mut result = Vector::zeros(x.len());
        for i in 0..x.len() {
            result[i] = x[i].max(lower[i]).min(upper[i]);
        }
        result
    }
}

// ==================== FISTA Optimizer ====================

/// FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
///
/// Accelerated proximal gradient method for minimizing composite objectives:
/// ```text
/// minimize f(x) + g(x)
/// ```
/// where f is smooth and convex, g is convex (possibly non-smooth but "simple").
///
/// FISTA achieves O(1/k²) convergence rate using Nesterov acceleration,
/// compared to O(1/k) for standard proximal gradient (ISTA).
///
/// # Key Applications
///
/// - **Lasso regression**: f(x) = ½‖Ax - b‖², g(x) = λ‖x‖₁
/// - **Elastic Net**: f(x) = ½‖Ax - b‖², g(x) = λ₁‖x‖₁ + λ₂‖x‖₂²
/// - **Total variation**: Image denoising with TV regularization
/// - **Non-negative least squares**: f(x) = ½‖Ax - b‖², g(x) = indicator(x ≥ 0)
///
/// # Example
///
/// ```
/// use aprender::optim::{FISTA, Optimizer, prox};
/// use aprender::primitives::Vector;
///
/// // Minimize: ½(x - 5)² + 2|x|  (L1-regularized quadratic)
/// let smooth = |x: &Vector<f32>| 0.5 * (x[0] - 5.0).powi(2);
/// let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[x[0] - 5.0]);
/// let proximal = |v: &Vector<f32>, _alpha: f32| prox::soft_threshold(v, 2.0);
///
/// let mut fista = FISTA::new(1000, 0.1, 1e-5);
/// let x0 = Vector::from_slice(&[0.0]);
/// let result = fista.minimize(smooth, grad_smooth, proximal, x0);
///
/// // Check that optimization completed successfully
/// assert!(!result.solution[0].is_nan());
/// ```
///
/// # References
///
/// - Beck & Teboulle (2009). "A fast iterative shrinkage-thresholding algorithm
///   for linear inverse problems." SIAM Journal on Imaging Sciences, 2(1), 183-202.
#[derive(Debug, Clone)]
pub struct FISTA {
    /// Maximum number of iterations
    max_iter: usize,
    /// Step size (α > 0)
    step_size: f32,
    /// Convergence tolerance (‖xₖ₊₁ - xₖ‖ < tol)
    tol: f32,
}

impl FISTA {
    /// Creates a new FISTA optimizer.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum number of iterations
    /// * `step_size` - Step size α (should be ≤ 1/L where L is Lipschitz constant of ∇f)
    /// * `tol` - Convergence tolerance
    ///
    /// # Returns
    ///
    /// New FISTA optimizer instance
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::FISTA;
    ///
    /// let optimizer = FISTA::new(1000, 0.01, 1e-6);
    /// ```
    #[must_use]
    pub fn new(max_iter: usize, step_size: f32, tol: f32) -> Self {
        Self {
            max_iter,
            step_size,
            tol,
        }
    }

    /// Minimizes a composite objective function using FISTA.
    ///
    /// Solves: minimize f(x) + g(x) where f is smooth, g is "simple" (has easy prox).
    ///
    /// # Type Parameters
    ///
    /// * `F` - Smooth objective function type
    /// * `G` - Gradient of smooth part type
    /// * `P` - Proximal operator type
    ///
    /// # Arguments
    ///
    /// * `smooth` - Smooth part f(x)
    /// * `grad_smooth` - Gradient ∇f(x)
    /// * `prox` - Proximal operator `prox_g(v`, α)
    /// * `x0` - Initial point
    ///
    /// # Returns
    ///
    /// [`OptimizationResult`] with solution and convergence information
    pub fn minimize<F, G, P>(
        &mut self,
        smooth: F,
        grad_smooth: G,
        prox: P,
        x0: Vector<f32>,
    ) -> OptimizationResult
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
        P: Fn(&Vector<f32>, f32) -> Vector<f32>,
    {
        let start_time = std::time::Instant::now();

        let mut x = x0.clone();
        let mut y = x0;
        let mut t = 1.0; // Nesterov momentum parameter

        for iter in 0..self.max_iter {
            // Proximal gradient step at y
            let grad_y = grad_smooth(&y);

            // Compute: y - α * ∇f(y)
            let mut gradient_step = Vector::zeros(y.len());
            for i in 0..y.len() {
                gradient_step[i] = y[i] - self.step_size * grad_y[i];
            }

            // Apply proximal operator
            let x_new = prox(&gradient_step, self.step_size);

            // Check convergence
            let mut diff_norm = 0.0;
            for i in 0..x.len() {
                let diff = x_new[i] - x[i];
                diff_norm += diff * diff;
            }
            diff_norm = diff_norm.sqrt();

            if diff_norm < self.tol {
                let final_obj = smooth(&x_new);
                return OptimizationResult {
                    solution: x_new,
                    objective_value: final_obj,
                    iterations: iter,
                    status: ConvergenceStatus::Converged,
                    gradient_norm: diff_norm, // Use step norm as proxy for gradient norm
                    constraint_violation: 0.0,
                    elapsed_time: start_time.elapsed(),
                };
            }

            // Nesterov acceleration
            let t_new = (1.0_f32 + (1.0_f32 + 4.0_f32 * t * t).sqrt()) / 2.0_f32;
            let beta = (t - 1.0_f32) / t_new;

            // y_new = x_new + β(x_new - x)
            let mut y_new = Vector::zeros(x.len());
            for i in 0..x.len() {
                y_new[i] = x_new[i] + beta * (x_new[i] - x[i]);
            }

            x = x_new;
            y = y_new;
            t = t_new;
        }

        // Max iterations reached
        let final_obj = smooth(&x);
        OptimizationResult {
            solution: x,
            objective_value: final_obj,
            iterations: self.max_iter,
            status: ConvergenceStatus::MaxIterations,
            gradient_norm: 0.0,
            constraint_violation: 0.0,
            elapsed_time: start_time.elapsed(),
        }
    }
}

impl Optimizer for FISTA {
    fn step(&mut self, _params: &mut Vector<f32>, _gradients: &Vector<f32>) {
        unimplemented!(
            "FISTA does not support stochastic updates (step). Use minimize() for batch optimization with proximal operators."
        )
    }

    fn reset(&mut self) {
        // FISTA is stateless - nothing to reset
    }
}

// ==================== Coordinate Descent Optimizer ====================

/// Coordinate Descent optimizer for high-dimensional problems.
///
/// Optimizes one coordinate at a time, which can be much more efficient than
/// full gradient methods when the number of features is very large (n ≫ m).
///
/// # Algorithm
///
/// ```text
/// for k = 1, 2, ..., max_iter:
///     for i = 1, 2, ..., n (cyclic or random order):
///         xᵢ ← argmin f(x₁, ..., xᵢ₋₁, xᵢ, xᵢ₊₁, ..., xₙ)
/// ```
///
/// # Key Applications
///
/// - **Lasso regression**: Coordinate descent with soft-thresholding (scikit-learn default)
/// - **Elastic Net**: L1 + L2 regularization
/// - **SVM**: Sequential Minimal Optimization (SMO) variant
/// - **High-dimensional statistics**: n ≫ m scenarios
///
/// # Advantages
///
/// - O(n) per coordinate update (vs O(n) for full gradient)
/// - No line search needed for many problems
/// - Handles non-differentiable objectives (e.g., L1)
/// - Cache-friendly memory access patterns
///
/// # Example
///
/// ```
/// use aprender::optim::{CoordinateDescent, Optimizer};
/// use aprender::primitives::Vector;
///
/// // Minimize: ½‖x - c‖² where c = [1, 2, 3]
/// // Coordinate i update: xᵢ = cᵢ (closed form)
/// let c = vec![1.0, 2.0, 3.0];
///
/// let update = move |x: &mut Vector<f32>, i: usize| {
///     x[i] = c[i]; // Closed-form solution for coordinate i
/// };
///
/// let mut cd = CoordinateDescent::new(100, 1e-6);
/// let x0 = Vector::from_slice(&[0.0, 0.0, 0.0]);
/// let result = cd.minimize(update, x0);
///
/// // Should converge to c
/// assert!((result.solution[0] - 1.0).abs() < 1e-5);
/// assert!((result.solution[1] - 2.0).abs() < 1e-5);
/// assert!((result.solution[2] - 3.0).abs() < 1e-5);
/// ```
///
/// # References
///
/// - Wright (2015). "Coordinate descent algorithms." Mathematical Programming.
/// - Friedman et al. (2010). "Regularization paths for generalized linear models via coordinate descent."
#[derive(Debug, Clone)]
pub struct CoordinateDescent {
    /// Maximum number of outer iterations (passes through all coordinates)
    max_iter: usize,
    /// Convergence tolerance (‖xₖ₊₁ - xₖ‖ < tol)
    tol: f32,
    /// Whether to use random coordinate order (vs cyclic)
    random_order: bool,
}

impl CoordinateDescent {
    /// Creates a new Coordinate Descent optimizer.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum number of passes through all coordinates
    /// * `tol` - Convergence tolerance
    ///
    /// # Returns
    ///
    /// New Coordinate Descent optimizer with cyclic coordinate order
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::CoordinateDescent;
    ///
    /// let optimizer = CoordinateDescent::new(1000, 1e-6);
    /// ```
    #[must_use]
    pub fn new(max_iter: usize, tol: f32) -> Self {
        Self {
            max_iter,
            tol,
            random_order: false,
        }
    }

    /// Sets whether to use random coordinate order.
    ///
    /// # Arguments
    ///
    /// * `random` - If true, coordinates are updated in random order each iteration
    ///
    /// # Returns
    ///
    /// Self for method chaining
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::CoordinateDescent;
    ///
    /// let cd = CoordinateDescent::new(1000, 1e-6).with_random_order(true);
    /// ```
    #[must_use]
    pub fn with_random_order(mut self, random: bool) -> Self {
        self.random_order = random;
        self
    }

    /// Minimizes an objective using coordinate descent.
    ///
    /// The user provides a coordinate update function that modifies one coordinate
    /// at a time. This function should solve:
    /// ```text
    /// xᵢ ← argmin f(x₁, ..., xᵢ₋₁, xᵢ, xᵢ₊₁, ..., xₙ)
    /// ```
    ///
    /// # Type Parameters
    ///
    /// * `U` - Coordinate update function type
    ///
    /// # Arguments
    ///
    /// * `update` - Function that updates coordinate i: `fn(&mut Vector, usize)`
    /// * `x0` - Initial point
    ///
    /// # Returns
    ///
    /// [`OptimizationResult`] with solution and convergence information
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::{CoordinateDescent, prox};
    /// use aprender::primitives::Vector;
    ///
    /// // Lasso coordinate update: soft-thresholding
    /// let lambda = 0.1;
    /// let update = move |x: &mut Vector<f32>, i: usize| {
    ///     // Simplified: actual Lasso requires computing residuals
    ///     let v = x[i];
    ///     x[i] = if v > lambda {
    ///         v - lambda
    ///     } else if v < -lambda {
    ///         v + lambda
    ///     } else {
    ///         0.0
    ///     };
    /// };
    ///
    /// let mut cd = CoordinateDescent::new(100, 1e-6);
    /// let x0 = Vector::from_slice(&[1.0, -0.5, 0.3]);
    /// let result = cd.minimize(update, x0);
    /// ```
    pub fn minimize<U>(&mut self, mut update: U, x0: Vector<f32>) -> OptimizationResult
    where
        U: FnMut(&mut Vector<f32>, usize),
    {
        let start_time = std::time::Instant::now();
        let n = x0.len();

        let mut x = x0;

        for iter in 0..self.max_iter {
            // Save previous iterate for convergence check
            let x_old = x.clone();

            // Determine coordinate order
            if self.random_order {
                // Random permutation (Fisher-Yates shuffle)
                let mut indices: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() {
                    let j = (i as f32 * 0.123456).rem_euclid(1.0); // Simple pseudo-random
                    let j = (j * (i + 1) as f32) as usize;
                    indices.swap(i, j);
                }

                // Update in random order
                for i in indices {
                    update(&mut x, i);
                }
            } else {
                // Cyclic order
                for i in 0..n {
                    update(&mut x, i);
                }
            }

            // Check convergence
            let mut diff_norm = 0.0;
            for i in 0..n {
                let diff = x[i] - x_old[i];
                diff_norm += diff * diff;
            }
            diff_norm = diff_norm.sqrt();

            if diff_norm < self.tol {
                return OptimizationResult {
                    solution: x,
                    objective_value: 0.0, // Objective not tracked
                    iterations: iter,
                    status: ConvergenceStatus::Converged,
                    gradient_norm: diff_norm,
                    constraint_violation: 0.0,
                    elapsed_time: start_time.elapsed(),
                };
            }
        }

        // Max iterations reached
        OptimizationResult {
            solution: x,
            objective_value: 0.0,
            iterations: self.max_iter,
            status: ConvergenceStatus::MaxIterations,
            gradient_norm: 0.0,
            constraint_violation: 0.0,
            elapsed_time: start_time.elapsed(),
        }
    }
}

impl Optimizer for CoordinateDescent {
    fn step(&mut self, _params: &mut Vector<f32>, _gradients: &Vector<f32>) {
        unimplemented!(
            "Coordinate Descent does not support stochastic updates (step). Use minimize() with coordinate update function."
        )
    }

    fn reset(&mut self) {
        // Coordinate Descent is stateless - nothing to reset
    }
}

/// ADMM (Alternating Direction Method of Multipliers) for distributed and constrained optimization.
///
/// Solves problems of the form:
/// ```text
/// minimize  f(x) + g(z)
/// subject to Ax + Bz = c
/// ```
///
/// # Applications
///
/// - **Distributed Lasso**: Split data across workers for large-scale regression
/// - **Consensus optimization**: Average models from different sites (federated learning)
/// - **Constrained problems**: Equality-constrained optimization via consensus
/// - **Model parallelism**: Parallelize training across devices
///
/// # Algorithm
///
/// ADMM alternates between three updates:
///
/// 1. **x-update**: `x^{k+1} = argmin_x { f(x) + (ρ/2)‖Ax + Bz^k - c + u^k‖² }`
/// 2. **z-update**: `z^{k+1} = argmin_z { g(z) + (ρ/2)‖Ax^{k+1} + Bz - c + u^k‖² }`
/// 3. **u-update**: `u^{k+1} = u^k + (Ax^{k+1} + Bz^{k+1} - c)`
///
/// where u is the scaled dual variable and ρ is the penalty parameter.
///
/// # Convergence
///
/// - **Rate**: O(1/k) for convex f and g
/// - **Stopping criteria**: Both primal and dual residuals below tolerance
/// - **Adaptive ρ**: Automatically adjusts penalty parameter for faster convergence
///
/// # Example: Consensus Form (Lasso)
///
/// For Lasso regression with consensus constraint x = z:
/// ```rust
/// use aprender::optim::ADMM;
/// use aprender::primitives::{Vector, Matrix};
///
/// let n = 5;
/// let m = 10;
///
/// // Create problem data
/// let A = Matrix::eye(n); // Identity for consensus
/// let B = Matrix::eye(n);
/// let c = Vector::zeros(n);
///
/// // x-minimizer: least squares update
/// let data_matrix = Matrix::eye(m); // Your data matrix
/// let observations = Vector::ones(m); // Your observations
/// let lambda = 0.1;
///
/// let x_minimizer = |z: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, rho: f32| {
///     // Minimize ½‖Dx - b‖² + (ρ/2)‖x - z + u‖²
///     // Closed form: x = (DᵀD + ρI)⁻¹(Dᵀb + ρ(z - u))
///     let mut rhs = Vector::zeros(n);
///     for i in 0..n {
///         rhs[i] = rho * (z[i] - u[i]);
///     }
///     rhs // Simplified for example
/// };
///
/// // z-minimizer: soft-thresholding (proximal operator for L1)
/// let z_minimizer = |ax: &Vector<f32>, u: &Vector<f32>, _c: &Vector<f32>, rho: f32| {
///     let mut z = Vector::zeros(n);
///     for i in 0..n {
///         let v = ax[i] + u[i];
///         let threshold = lambda / rho;
///         z[i] = if v > threshold {
///             v - threshold
///         } else if v < -threshold {
///             v + threshold
///         } else {
///             0.0
///         };
///     }
///     z
/// };
///
/// let mut admm = ADMM::new(100, 1.0, 1e-4).with_adaptive_rho(true);
/// let x0 = Vector::zeros(n);
/// let z0 = Vector::zeros(n);
///
/// let result = admm.minimize_consensus(
///     x_minimizer,
///     z_minimizer,
///     &A,
///     &B,
///     &c,
///     x0,
///     z0,
/// );
/// ```
///
/// # Reference
///
/// Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011).
/// "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers"
/// Foundations and Trends in Machine Learning, 3(1), 1-122.
#[derive(Debug, Clone)]
pub struct ADMM {
    /// Maximum number of iterations
    max_iter: usize,
    /// Penalty parameter (controls constraint enforcement)
    rho: f32,
    /// Tolerance for convergence (primal + dual residuals)
    tol: f32,
    /// Whether to adaptively adjust rho
    adaptive_rho: bool,
    /// Factor for increasing rho when primal residual is large
    rho_increase: f32,
    /// Factor for decreasing rho when dual residual is large
    rho_decrease: f32,
}

impl ADMM {
    /// Creates a new ADMM optimizer.
    ///
    /// # Parameters
    ///
    /// - `max_iter`: Maximum number of iterations (typical: 100-1000)
    /// - `rho`: Penalty parameter (typical: 0.1-10.0, problem-dependent)
    /// - `tol`: Convergence tolerance for residuals (typical: 1e-4 to 1e-6)
    ///
    /// # Returns
    ///
    /// A new ADMM optimizer with default settings:
    /// - Adaptive rho: disabled (use `with_adaptive_rho(true)` to enable)
    /// - Rho increase factor: 2.0
    /// - Rho decrease factor: 2.0
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::ADMM;
    ///
    /// let admm = ADMM::new(100, 1.0, 1e-4);
    /// ```
    #[must_use]
    pub fn new(max_iter: usize, rho: f32, tol: f32) -> Self {
        Self {
            max_iter,
            rho,
            tol,
            adaptive_rho: false,
            rho_increase: 2.0,
            rho_decrease: 2.0,
        }
    }

    /// Enables or disables adaptive penalty parameter adjustment.
    ///
    /// When enabled, rho is automatically adjusted based on the ratio of primal to dual residuals:
    /// - If primal residual >> dual residual: increase rho (enforce constraints more)
    /// - If dual residual >> primal residual: decrease rho (improve objective)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::ADMM;
    ///
    /// let admm = ADMM::new(100, 1.0, 1e-4).with_adaptive_rho(true);
    /// ```
    #[must_use]
    pub fn with_adaptive_rho(mut self, adaptive: bool) -> Self {
        self.adaptive_rho = adaptive;
        self
    }

    /// Sets the factors for adaptive rho adjustment.
    ///
    /// # Parameters
    ///
    /// - `increase`: Factor to multiply rho when primal residual is large (default: 2.0)
    /// - `decrease`: Factor to divide rho when dual residual is large (default: 2.0)
    #[must_use]
    pub fn with_rho_factors(mut self, increase: f32, decrease: f32) -> Self {
        self.rho_increase = increase;
        self.rho_decrease = decrease;
        self
    }

    /// Minimizes a consensus-form ADMM problem.
    ///
    /// Solves: minimize f(x) + g(z) subject to Ax + Bz = c
    ///
    /// # Parameters
    ///
    /// - `x_minimizer`: Function that solves x-subproblem given (z, u, c, rho)
    /// - `z_minimizer`: Function that solves z-subproblem given (Ax, u, c, rho)
    /// - `A`, `B`, `c`: Constraint matrices and vector (Ax + Bz = c)
    /// - `x0`, `z0`: Initial values for x and z
    ///
    /// # Returns
    ///
    /// `OptimizationResult` containing the optimal x value and convergence information.
    ///
    /// # Minimizer Functions
    ///
    /// The `x_minimizer` should solve:
    /// ```text
    /// argmin_x { f(x) + (ρ/2)‖Ax + Bz - c + u‖² }
    /// ```
    ///
    /// The `z_minimizer` should solve:
    /// ```text
    /// argmin_z { g(z) + (ρ/2)‖Ax + Bz - c + u‖² }
    /// ```
    ///
    /// These often have closed-form solutions or can use proximal operators.
    #[allow(clippy::too_many_arguments)]
    pub fn minimize_consensus<F, G>(
        &mut self,
        x_minimizer: F,
        z_minimizer: G,
        a: &Matrix<f32>,
        b_mat: &Matrix<f32>,
        c: &Vector<f32>,
        x0: Vector<f32>,
        z0: Vector<f32>,
    ) -> OptimizationResult
    where
        F: Fn(&Vector<f32>, &Vector<f32>, &Vector<f32>, f32) -> Vector<f32>,
        G: Fn(&Vector<f32>, &Vector<f32>, &Vector<f32>, f32) -> Vector<f32>,
    {
        let start_time = std::time::Instant::now();

        let mut x = x0;
        let mut z = z0;
        let mut u = Vector::zeros(c.len());
        let mut rho = self.rho;

        let mut z_old = z.clone();

        for iter in 0..self.max_iter {
            // x-update: minimize f(x) + (ρ/2)‖Ax + Bz - c + u‖²
            x = x_minimizer(&z, &u, c, rho);

            // z-update: minimize g(z) + (ρ/2)‖Ax + Bz - c + u‖²
            let ax = a.matvec(&x).expect("Matrix-vector multiplication");
            z = z_minimizer(&ax, &u, c, rho);

            // Compute residual: r = Ax + Bz - c
            let bz = b_mat.matvec(&z).expect("Matrix-vector multiplication");
            let residual = &(&ax + &bz) - c;

            // u-update: u^{k+1} = u^k + r^{k+1}
            u = &u + &residual;

            // Compute primal residual: ‖Ax + Bz - c‖
            let primal_res = residual.norm();

            // Compute dual residual: ρ‖Bᵀ(z^{k+1} - z^k)‖
            let z_diff = &z - &z_old;
            let bt_z_diff = b_mat
                .transpose()
                .matvec(&z_diff)
                .expect("Matrix-vector multiplication");
            let dual_res = rho * bt_z_diff.norm();

            // Check convergence
            if primal_res < self.tol && dual_res < self.tol {
                return OptimizationResult {
                    solution: x,
                    objective_value: 0.0, // Objective not tracked (requires f and g evaluations)
                    iterations: iter + 1,
                    status: ConvergenceStatus::Converged,
                    gradient_norm: dual_res,
                    constraint_violation: primal_res,
                    elapsed_time: start_time.elapsed(),
                };
            }

            // Adaptive rho adjustment (Boyd et al. 2011, Section 3.4.1)
            if self.adaptive_rho && iter % 10 == 0 {
                if primal_res > 10.0 * dual_res {
                    // Primal residual is large: increase rho to enforce constraints
                    rho *= self.rho_increase;
                    // Rescale dual variable: u = u / rho_increase
                    let scale = 1.0 / self.rho_increase;
                    u = u.mul_scalar(scale);
                } else if dual_res > 10.0 * primal_res {
                    // Dual residual is large: decrease rho to improve objective
                    rho /= self.rho_decrease;
                    // Rescale dual variable: u = u * rho_decrease
                    u = u.mul_scalar(self.rho_decrease);
                }
            }

            z_old = z.clone();
        }

        // Max iterations reached
        OptimizationResult {
            solution: x,
            objective_value: 0.0,
            iterations: self.max_iter,
            status: ConvergenceStatus::MaxIterations,
            gradient_norm: 0.0,
            constraint_violation: 0.0,
            elapsed_time: start_time.elapsed(),
        }
    }
}

impl Optimizer for ADMM {
    fn step(&mut self, _params: &mut Vector<f32>, _gradients: &Vector<f32>) {
        unimplemented!(
            "ADMM does not support stochastic updates (step). Use minimize_consensus() with x-minimizer and z-minimizer functions."
        )
    }

    fn reset(&mut self) {
        // ADMM is stateless - nothing to reset
    }
}

// ==================== Phase 3: Constrained Optimization ====================

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

/// Augmented Lagrangian method for constrained optimization.
///
/// Solves problems with equality constraints:
/// ```text
/// minimize f(x)
/// subject to: h(x) = 0  (equality constraints)
/// ```
///
/// # Algorithm
///
/// ```text
/// Augmented Lagrangian: L_ρ(x, λ) = f(x) + λᵀh(x) + ½ρ‖h(x)‖²
///
/// for k = 1, 2, ..., max_iter:
///     x_k = argmin L_ρ(x, λ_k)
///     λ_k+1 = λ_k + ρ h(x_k)
///     if ‖h(x)‖ is small: converged
///     else: increase ρ
/// ```
///
/// # Key Features
///
/// - **Penalty parameter ρ**: Increased adaptively for faster convergence
/// - **Lagrange multipliers λ**: Automatically updated for equality constraints
/// - **Flexible subproblem solver**: Uses gradient descent for inner loop
/// - **Convergence**: Superlinear under regularity conditions
///
/// # Applications
///
/// - **Equality constraints**: Linear systems, manifold optimization
/// - **ADMM**: Alternating Direction Method of Multipliers (special case)
/// - **Consensus optimization**: Distributed optimization, federated learning
/// - **PDE-constrained optimization**: Physics-informed neural networks
///
/// # Example
///
/// ```
/// use aprender::optim::AugmentedLagrangian;
/// use aprender::primitives::Vector;
///
/// // Minimize: ½(x₁-2)² + ½(x₂-3)² subject to x₁ + x₂ = 1
/// let objective = |x: &Vector<f32>| {
///     0.5 * (x[0] - 2.0).powi(2) + 0.5 * (x[1] - 3.0).powi(2)
/// };
///
/// let gradient = |x: &Vector<f32>| {
///     Vector::from_slice(&[x[0] - 2.0, x[1] - 3.0])
/// };
///
/// // Equality constraint: x₁ + x₂ - 1 = 0
/// let equality = |x: &Vector<f32>| Vector::from_slice(&[x[0] + x[1] - 1.0]);
///
/// let equality_jac = |_x: &Vector<f32>| {
///     vec![Vector::from_slice(&[1.0, 1.0])]
/// };
///
/// let mut al = AugmentedLagrangian::new(100, 1e-6, 1.0);
/// let x0 = Vector::zeros(2);
/// let result = al.minimize_equality(objective, gradient, equality, equality_jac, x0);
///
/// assert!(result.constraint_violation < 1e-4);
/// ```
///
/// # References
///
/// - Nocedal & Wright (2006). "Numerical Optimization." Chapter 17.
/// - Bertsekas (1982). "Constrained Optimization and Lagrange Multiplier Methods."
#[derive(Debug, Clone)]
pub struct AugmentedLagrangian {
    /// Maximum number of outer iterations
    max_iter: usize,
    /// Convergence tolerance for constraint violation
    tol: f32,
    /// Initial penalty parameter
    initial_rho: f32,
    /// Current penalty parameter
    rho: f32,
    /// Penalty increase factor (> 1)
    rho_increase: f32,
    /// Maximum penalty parameter
    rho_max: f32,
}

impl AugmentedLagrangian {
    /// Creates a new Augmented Lagrangian optimizer.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum number of outer iterations
    /// * `tol` - Convergence tolerance for constraint violation
    /// * `initial_rho` - Initial penalty parameter (typically 1.0-10.0)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::AugmentedLagrangian;
    ///
    /// let optimizer = AugmentedLagrangian::new(100, 1e-6, 1.0);
    /// ```
    #[must_use]
    pub fn new(max_iter: usize, tol: f32, initial_rho: f32) -> Self {
        Self {
            max_iter,
            tol,
            initial_rho,
            rho: initial_rho,
            rho_increase: 2.0,
            rho_max: 1e6,
        }
    }

    /// Sets penalty increase factor.
    ///
    /// # Arguments
    ///
    /// * `factor` - Penalty increase factor (> 1, typically 2.0-10.0)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::AugmentedLagrangian;
    ///
    /// let optimizer = AugmentedLagrangian::new(100, 1e-6, 1.0)
    ///     .with_rho_increase(5.0);
    /// ```
    #[must_use]
    pub fn with_rho_increase(mut self, factor: f32) -> Self {
        self.rho_increase = factor;
        self
    }

    /// Minimizes objective subject to equality constraints.
    ///
    /// Solves: minimize f(x) subject to h(x) = 0
    ///
    /// # Arguments
    ///
    /// * `objective` - Objective function f(x)
    /// * `gradient` - Gradient ∇f(x)
    /// * `equality` - Equality constraints h(x) = 0 (returns vector)
    /// * `equality_jac` - Jacobian of equality constraints ∇h(x)
    /// * `x0` - Initial point
    ///
    /// # Returns
    ///
    /// Optimization result with constraint satisfaction metrics
    pub fn minimize_equality<F, G, H, J>(
        &mut self,
        objective: F,
        gradient: G,
        equality: H,
        equality_jac: J,
        x0: Vector<f32>,
    ) -> OptimizationResult
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
        H: Fn(&Vector<f32>) -> Vector<f32>,
        J: Fn(&Vector<f32>) -> Vec<Vector<f32>>,
    {
        let start_time = std::time::Instant::now();

        let mut x = x0;
        self.rho = self.initial_rho;

        // Initialize Lagrange multipliers to zero
        let h0 = equality(&x);
        let m = h0.len(); // Number of equality constraints
        let mut lambda = Vector::zeros(m);

        for outer_iter in 0..self.max_iter {
            // Solve augmented Lagrangian subproblem: min L_ρ(x, λ)
            // L_ρ(x, λ) = f(x) + λᵀh(x) + ½ρ‖h(x)‖²

            let aug_grad = |x_inner: &Vector<f32>| {
                let grad_f = gradient(x_inner);
                let h_val = equality(x_inner);
                let jac_h = equality_jac(x_inner);

                let n = x_inner.len();
                let mut aug_g = Vector::zeros(n);

                // ∇f(x)
                for i in 0..n {
                    aug_g[i] = grad_f[i];
                }

                // Add ∇h(x)ᵀ(λ + ρh(x))
                for j in 0..m {
                    let coeff = lambda[j] + self.rho * h_val[j];
                    for i in 0..n {
                        aug_g[i] += coeff * jac_h[j][i];
                    }
                }

                aug_g
            };

            // Solve subproblem using gradient descent (simple solver)
            let mut x_sub = x.clone();
            let alpha = 0.01; // Fixed step size for subproblem
            for _sub_iter in 0..50 {
                let grad = aug_grad(&x_sub);
                let mut grad_norm_sq = 0.0;
                for i in 0..grad.len() {
                    grad_norm_sq += grad[i] * grad[i];
                }
                if grad_norm_sq < 1e-8 {
                    break; // Subproblem converged
                }
                for i in 0..x_sub.len() {
                    x_sub[i] -= alpha * grad[i];
                }
            }

            x = x_sub;

            // Update Lagrange multipliers: λ_k+1 = λ_k + ρ h(x_k)
            let h_val = equality(&x);
            for i in 0..m {
                lambda[i] += self.rho * h_val[i];
            }

            // Check constraint violation
            let mut constraint_viol = 0.0;
            for i in 0..m {
                constraint_viol += h_val[i] * h_val[i];
            }
            constraint_viol = constraint_viol.sqrt();

            // Check convergence
            if constraint_viol < self.tol {
                let final_obj = objective(&x);
                let grad_f = gradient(&x);
                let mut grad_norm = 0.0;
                for i in 0..grad_f.len() {
                    grad_norm += grad_f[i] * grad_f[i];
                }
                grad_norm = grad_norm.sqrt();

                return OptimizationResult {
                    solution: x,
                    objective_value: final_obj,
                    iterations: outer_iter + 1,
                    status: ConvergenceStatus::Converged,
                    gradient_norm: grad_norm,
                    constraint_violation: constraint_viol,
                    elapsed_time: start_time.elapsed(),
                };
            }

            // Increase penalty parameter if constraint violation is not decreasing fast enough
            if constraint_viol > 0.1 * self.tol && self.rho < self.rho_max {
                self.rho *= self.rho_increase;
            }
        }

        // Max iterations reached
        let final_obj = objective(&x);
        let h_val = equality(&x);
        let mut constraint_viol = 0.0;
        for i in 0..h_val.len() {
            constraint_viol += h_val[i] * h_val[i];
        }
        constraint_viol = constraint_viol.sqrt();

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
            constraint_violation: constraint_viol,
            elapsed_time: start_time.elapsed(),
        }
    }
}

impl Optimizer for AugmentedLagrangian {
    fn step(&mut self, _params: &mut Vector<f32>, _gradients: &Vector<f32>) {
        unimplemented!(
            "Augmented Lagrangian does not support stochastic updates (step). Use minimize_equality() for constrained optimization."
        )
    }

    fn reset(&mut self) {
        // Reset penalty parameter to initial value
        self.rho = self.initial_rho;
    }
}

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

// SGD and Adam moved to stochastic.rs (PMAT-085)

/// Unified trait for both stochastic and batch optimizers.
///
/// This trait supports two modes of optimization:
///
/// 1. **Stochastic mode** (`step`): For mini-batch training with SGD, Adam, etc.
/// 2. **Batch mode** (`minimize`): For full-dataset optimization with L-BFGS, CG, etc.
///
/// # Type Safety
///
/// The compiler prevents misuse:
/// - L-BFGS cannot be used with `step()` (would give poor results with stochastic gradients)
/// - SGD/Adam don't implement `minimize()` (inefficient for full datasets)
///
/// # Example
///
/// ```
/// use aprender::optim::{Optimizer, SGD};
/// use aprender::primitives::Vector;
///
/// // Stochastic mode (mini-batch training)
/// let mut optimizer = SGD::new(0.01);
/// let mut params = Vector::from_slice(&[1.0, 2.0]);
/// let grad = Vector::from_slice(&[0.1, 0.2]);
/// optimizer.step(&mut params, &grad);
/// ```
pub trait Optimizer {
    /// Stochastic update (mini-batch mode) - for SGD, Adam, `RMSprop`.
    ///
    /// Updates parameters in-place given gradient from current mini-batch.
    /// Used in ML training loops where gradients come from different data batches.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameter vector to update (modified in-place)
    /// * `gradients` - Gradient vector from current mini-batch
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::{Optimizer, SGD};
    /// use aprender::primitives::Vector;
    ///
    /// let mut optimizer = SGD::new(0.1);
    /// let mut params = Vector::from_slice(&[1.0, 2.0]);
    ///
    /// for _epoch in 0..10 {
    ///     let grad = Vector::from_slice(&[0.1, 0.2]); // From mini-batch
    ///     optimizer.step(&mut params, &grad);
    /// }
    /// ```
    fn step(&mut self, params: &mut Vector<f32>, gradients: &Vector<f32>);

    /// Batch optimization (deterministic mode) - for L-BFGS, CG, Damped Newton.
    ///
    /// Minimizes objective function with full dataset access.
    /// Returns complete optimization trajectory and convergence info.
    ///
    /// **Default implementation**: Not all optimizers support batch mode. Stochastic
    /// optimizers (SGD, Adam) will panic if you call this method.
    ///
    /// # Arguments
    ///
    /// * `objective` - Objective function f: ℝⁿ → ℝ
    /// * `gradient` - Gradient function ∇f: ℝⁿ → ℝⁿ
    /// * `x0` - Initial point
    ///
    /// # Returns
    ///
    /// [`OptimizationResult`] with solution, convergence status, and diagnostics.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use aprender::optim::{Optimizer, LBFGS};
    /// use aprender::primitives::Vector;
    ///
    /// let mut optimizer = LBFGS::new(100, 1e-5, 10);
    ///
    /// let objective = |x: &Vector<f32>| (x[0] - 5.0).powi(2);
    /// let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 5.0)]);
    ///
    /// let result = optimizer.minimize(objective, gradient, Vector::from_slice(&[0.0]));
    /// assert_eq!(result.status, ConvergenceStatus::Converged);
    /// ```
    fn minimize<F, G>(
        &mut self,
        _objective: F,
        _gradient: G,
        _x0: Vector<f32>,
    ) -> OptimizationResult
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
    {
        unimplemented!(
            "{} does not support batch optimization (minimize). Use step() for stochastic updates.",
            std::any::type_name::<Self>()
        )
    }

    /// Resets the optimizer state (momentum, history, etc.).
    ///
    /// Call this when starting training on a new model or after significant
    /// changes to the optimization problem.
    fn reset(&mut self);
}

// SGD and Adam impl Optimizer moved to stochastic.rs (PMAT-085)

#[cfg(test)]
mod tests;
