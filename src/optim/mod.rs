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
mod admm;
mod augmented_lagrangian;
mod conjugate_gradient;
mod coordinate_descent;
mod damped_newton;
mod fista;
mod interior_point;
mod lbfgs;
mod line_search;
mod projected_gradient;
mod stochastic;

// Re-exports from submodules
pub use admm::ADMM;
pub use augmented_lagrangian::AugmentedLagrangian;
pub use conjugate_gradient::{CGBetaFormula, ConjugateGradient};
pub use coordinate_descent::CoordinateDescent;
pub use damped_newton::DampedNewton;
pub use fista::FISTA;
pub use interior_point::InteriorPoint;
pub use lbfgs::LBFGS;
pub use line_search::{BacktrackingLineSearch, LineSearch, WolfeLineSearch};
pub use projected_gradient::ProjectedGradientDescent;
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
        panic!(
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

#[cfg(test)]
mod tests;
