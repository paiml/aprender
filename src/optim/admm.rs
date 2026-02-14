//! ADMM (Alternating Direction Method of Multipliers) for distributed and constrained optimization.

use super::{ConvergenceStatus, OptimizationResult, Optimizer};
use crate::primitives::{Matrix, Vector};

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
    // ==================== Getters (for testing) ====================

    /// Returns the maximum number of iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the penalty parameter.
    #[must_use]
    pub fn rho(&self) -> f32 {
        self.rho
    }

    /// Returns the convergence tolerance.
    #[must_use]
    pub fn tol(&self) -> f32 {
        self.tol
    }

    /// Returns whether adaptive rho is enabled.
    #[must_use]
    pub fn adaptive_rho(&self) -> bool {
        self.adaptive_rho
    }

    /// Returns the rho increase factor.
    #[must_use]
    pub fn rho_increase(&self) -> f32 {
        self.rho_increase
    }

    /// Returns the rho decrease factor.
    #[must_use]
    pub fn rho_decrease(&self) -> f32 {
        self.rho_decrease
    }
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
        panic!(
            "ADMM does not support stochastic updates (step). Use minimize_consensus() with x-minimizer and z-minimizer functions."
        )
    }

    fn reset(&mut self) {
        // ADMM is stateless - nothing to reset
    }
}

#[cfg(test)]
#[path = "admm_tests.rs"]
mod tests;
