//! Optimization algorithms for gradient-based learning.
//!
//! This module provides both stochastic (mini-batch) and batch (deterministic) optimizers:
//!
//! - **Stochastic optimizers**: [`SGD`], [`Adam`] - for training with mini-batches
//! - **Batch optimizers**: L-BFGS, Conjugate Gradient, Damped Newton (coming in v0.8.0)
//!
//! # Stochastic Optimization (Mini-Batch)
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
//! Batch optimizers will be available in v0.8.0. They support the `minimize` method
//! for optimizing over the full dataset:
//!
//! ```ignore
//! use aprender::optim::LBFGS;
//! use aprender::primitives::Vector;
//!
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
//! ```

use serde::{Deserialize, Serialize};

use crate::primitives::Vector;

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

/// Line search strategy for determining step size in batch optimization.
///
/// Line search methods find an appropriate step size α along a search direction d
/// by solving the 1D optimization problem:
///
/// ```text
/// minimize f(x + α*d) over α > 0
/// ```
///
/// Different strategies enforce different conditions on the step size.
pub trait LineSearch {
    /// Finds a suitable step size along the search direction.
    ///
    /// # Arguments
    ///
    /// * `f` - Objective function f: ℝⁿ → ℝ
    /// * `grad` - Gradient function ∇f: ℝⁿ → ℝⁿ
    /// * `x` - Current point
    /// * `d` - Search direction (typically descent direction, ∇f(x)·d < 0)
    ///
    /// # Returns
    ///
    /// Step size α > 0 satisfying the line search conditions
    fn search<F, G>(
        &self,
        f: &F,
        grad: &G,
        x: &Vector<f32>,
        d: &Vector<f32>,
    ) -> f32
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>;
}

/// Backtracking line search with Armijo condition.
///
/// Starts with step size α = 1 and repeatedly shrinks it by factor ρ until
/// the Armijo condition is satisfied:
///
/// ```text
/// f(x + α*d) ≤ f(x) + c₁*α*∇f(x)ᵀd
/// ```
///
/// This ensures sufficient decrease in the objective function.
///
/// # Parameters
///
/// - **c1**: Armijo constant (typical: 1e-4), controls acceptable decrease
/// - **rho**: Backtracking factor (typical: 0.5), shrinkage rate for α
/// - **max_iter**: Maximum backtracking iterations (safety limit)
///
/// # Example
///
/// ```
/// use aprender::optim::BacktrackingLineSearch;
/// use aprender::primitives::Vector;
///
/// let line_search = BacktrackingLineSearch::new(1e-4, 0.5, 50);
///
/// // Define a simple quadratic function
/// let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
/// let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
///
/// let x = Vector::from_slice(&[1.0, 1.0]);
/// let d = Vector::from_slice(&[-2.0, -2.0]); // Descent direction
///
/// let alpha = line_search.search(&f, &grad, &x, &d);
/// assert!(alpha > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct BacktrackingLineSearch {
    /// Armijo constant (c₁ ∈ (0, 1), typical: 1e-4)
    c1: f32,
    /// Backtracking factor (ρ ∈ (0, 1), typical: 0.5)
    rho: f32,
    /// Maximum backtracking iterations
    max_iter: usize,
}

impl BacktrackingLineSearch {
    /// Creates a new backtracking line search.
    ///
    /// # Arguments
    ///
    /// * `c1` - Armijo constant (typical: 1e-4)
    /// * `rho` - Backtracking factor (typical: 0.5)
    /// * `max_iter` - Maximum iterations (typical: 50)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::BacktrackingLineSearch;
    ///
    /// let line_search = BacktrackingLineSearch::new(1e-4, 0.5, 50);
    /// ```
    #[must_use]
    pub fn new(c1: f32, rho: f32, max_iter: usize) -> Self {
        Self { c1, rho, max_iter }
    }

    /// Creates a backtracking line search with default parameters.
    ///
    /// Defaults: c1=1e-4, rho=0.5, max_iter=50
    #[must_use]
    pub fn default() -> Self {
        Self::new(1e-4, 0.5, 50)
    }
}

impl LineSearch for BacktrackingLineSearch {
    fn search<F, G>(
        &self,
        f: &F,
        grad: &G,
        x: &Vector<f32>,
        d: &Vector<f32>,
    ) -> f32
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
    {
        let mut alpha = 1.0;
        let fx = f(x);
        let grad_x = grad(x);

        // Compute directional derivative: ∇f(x)ᵀd
        let mut dir_deriv = 0.0;
        for i in 0..x.len() {
            dir_deriv += grad_x[i] * d[i];
        }

        // Backtracking loop
        for _ in 0..self.max_iter {
            // Compute x_new = x + alpha * d
            let mut x_new = Vector::zeros(x.len());
            for i in 0..x.len() {
                x_new[i] = x[i] + alpha * d[i];
            }

            let fx_new = f(&x_new);

            // Check Armijo condition: f(x + α*d) ≤ f(x) + c₁*α*∇f(x)ᵀd
            if fx_new <= fx + self.c1 * alpha * dir_deriv {
                return alpha;
            }

            // Shrink step size
            alpha *= self.rho;
        }

        // Return the last alpha if max iterations reached
        alpha
    }
}

/// Wolfe line search with Armijo and curvature conditions.
///
/// Enforces both the Armijo condition (sufficient decrease) and the curvature
/// condition (sufficient curvature):
///
/// ```text
/// Armijo:    f(x + α*d) ≤ f(x) + c₁*α*∇f(x)ᵀd
/// Curvature: |∇f(x + α*d)ᵀd| ≤ c₂*|∇f(x)ᵀd|
/// ```
///
/// The curvature condition ensures the step size is not too small by requiring
/// that the gradient has decreased sufficiently along the search direction.
///
/// # Parameters
///
/// - **c1**: Armijo constant (typical: 1e-4), c₁ ∈ (0, c₂)
/// - **c2**: Curvature constant (typical: 0.9), c₂ ∈ (c₁, 1)
/// - **max_iter**: Maximum line search iterations
///
/// # Example
///
/// ```
/// use aprender::optim::WolfeLineSearch;
/// use aprender::primitives::Vector;
///
/// let line_search = WolfeLineSearch::new(1e-4, 0.9, 50);
///
/// let f = |x: &Vector<f32>| x[0] * x[0];
/// let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
///
/// let x = Vector::from_slice(&[1.0]);
/// let d = Vector::from_slice(&[-2.0]);
///
/// let alpha = line_search.search(&f, &grad, &x, &d);
/// assert!(alpha > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct WolfeLineSearch {
    /// Armijo constant (c₁ ∈ (0, c₂), typical: 1e-4)
    c1: f32,
    /// Curvature constant (c₂ ∈ (c₁, 1), typical: 0.9)
    c2: f32,
    /// Maximum line search iterations
    max_iter: usize,
}

impl WolfeLineSearch {
    /// Creates a new Wolfe line search.
    ///
    /// # Arguments
    ///
    /// * `c1` - Armijo constant (typical: 1e-4)
    /// * `c2` - Curvature constant (typical: 0.9)
    /// * `max_iter` - Maximum iterations (typical: 50)
    ///
    /// # Panics
    ///
    /// Panics if c1 >= c2 or values are outside (0, 1).
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::WolfeLineSearch;
    ///
    /// let line_search = WolfeLineSearch::new(1e-4, 0.9, 50);
    /// ```
    #[must_use]
    pub fn new(c1: f32, c2: f32, max_iter: usize) -> Self {
        assert!(
            c1 < c2 && c1 > 0.0 && c2 < 1.0,
            "Wolfe conditions require 0 < c1 < c2 < 1"
        );
        Self { c1, c2, max_iter }
    }

    /// Creates a Wolfe line search with default parameters.
    ///
    /// Defaults: c1=1e-4, c2=0.9, max_iter=50
    #[must_use]
    pub fn default() -> Self {
        Self::new(1e-4, 0.9, 50)
    }
}

impl LineSearch for WolfeLineSearch {
    fn search<F, G>(
        &self,
        f: &F,
        grad: &G,
        x: &Vector<f32>,
        d: &Vector<f32>,
    ) -> f32
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
    {
        let fx = f(x);
        let grad_x = grad(x);

        // Compute directional derivative: ∇f(x)ᵀd
        let mut dir_deriv = 0.0;
        for i in 0..x.len() {
            dir_deriv += grad_x[i] * d[i];
        }

        // Start with alpha = 1.0
        let mut alpha = 1.0;
        let mut alpha_lo = 0.0;
        let mut alpha_hi = f32::INFINITY;

        for _ in 0..self.max_iter {
            // Compute x_new = x + alpha * d
            let mut x_new = Vector::zeros(x.len());
            for i in 0..x.len() {
                x_new[i] = x[i] + alpha * d[i];
            }

            let fx_new = f(&x_new);
            let grad_new = grad(&x_new);

            // Compute new directional derivative
            let mut dir_deriv_new = 0.0;
            for i in 0..x.len() {
                dir_deriv_new += grad_new[i] * d[i];
            }

            // Check Armijo condition
            if fx_new > fx + self.c1 * alpha * dir_deriv {
                // Armijo fails - alpha too large
                alpha_hi = alpha;
                alpha = (alpha_lo + alpha_hi) / 2.0;
                continue;
            }

            // Check curvature condition: |∇f(x + α*d)ᵀd| ≤ c₂*|∇f(x)ᵀd|
            if dir_deriv_new.abs() <= self.c2 * dir_deriv.abs() {
                // Both conditions satisfied
                return alpha;
            }

            // Curvature condition fails
            if dir_deriv_new > 0.0 {
                // Gradient sign changed - reduce alpha
                alpha_hi = alpha;
            } else {
                // Gradient still negative - increase alpha
                alpha_lo = alpha;
            }

            // Update alpha
            if alpha_hi.is_finite() {
                alpha = (alpha_lo + alpha_hi) / 2.0;
            } else {
                alpha *= 2.0;
            }
        }

        // Return the last alpha if max iterations reached
        alpha
    }
}

/// Limited-memory BFGS (L-BFGS) optimizer.
///
/// L-BFGS is a quasi-Newton method that approximates the inverse Hessian using
/// a limited history of gradient information. It's efficient for large-scale
/// optimization problems where storing the full Hessian is infeasible.
///
/// # Algorithm
///
/// 1. Compute gradient g_k = ∇f(x_k)
/// 2. Compute search direction d_k using two-loop recursion (approximates H^(-1) * g_k)
/// 3. Find step size α_k via line search (Wolfe conditions)
/// 4. Update: x_{k+1} = x_k - α_k * d_k
/// 5. Store gradient and position differences for next iteration
///
/// # Parameters
///
/// - **max_iter**: Maximum number of iterations
/// - **tol**: Convergence tolerance (gradient norm)
/// - **m**: History size (typically 5-20, tradeoff between memory and convergence)
///
/// # Example
///
/// ```
/// use aprender::optim::LBFGS;
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
    max_iter: usize,
    /// Convergence tolerance (gradient norm)
    tol: f32,
    /// History size (number of correction pairs to store)
    m: usize,
    /// Line search strategy
    line_search: WolfeLineSearch,
    /// Position differences: s_k = x_{k+1} - x_k
    s_history: Vec<Vector<f32>>,
    /// Gradient differences: y_k = g_{k+1} - g_k
    y_history: Vec<Vector<f32>>,
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
        unimplemented!(
            "L-BFGS does not support stochastic updates (step). Use minimize() for batch optimization."
        )
    }

    fn minimize<F, G>(
        &mut self,
        objective: F,
        gradient: G,
        x0: Vector<f32>,
    ) -> OptimizationResult
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

/// Beta computation formula for Conjugate Gradient.
///
/// Different formulas provide different convergence properties and
/// numerical stability characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CGBetaFormula {
    /// Fletcher-Reeves: β = (g_{k+1}^T g_{k+1}) / (g_k^T g_k)
    ///
    /// Most stable but can be slow on non-quadratic problems.
    FletcherReeves,
    /// Polak-Ribière: β = g_{k+1}^T (g_{k+1} - g_k) / (g_k^T g_k)
    ///
    /// Better performance than FR, includes automatic restart (β < 0).
    PolakRibiere,
    /// Hestenes-Stiefel: β = g_{k+1}^T (g_{k+1} - g_k) / (d_k^T (g_{k+1} - g_k))
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
/// 1. Initialize with steepest descent: d_0 = -∇f(x_0)
/// 2. Line search: find α_k minimizing f(x_k + α_k d_k)
/// 3. Update: x_{k+1} = x_k + α_k d_k
/// 4. Compute β_{k+1} using chosen formula (FR, PR, or HS)
/// 5. Update direction: d_{k+1} = -∇f(x_{k+1}) + β_{k+1} d_k
/// 6. Restart if β < 0 or every n iterations
///
/// # Parameters
///
/// - **max_iter**: Maximum number of iterations
/// - **tol**: Convergence tolerance (gradient norm)
/// - **beta_formula**: Method for computing β (FR, PR, or HS)
/// - **restart_interval**: Restart with steepest descent every n iterations (0 = no periodic restart)
///
/// # Example
///
/// ```
/// use aprender::optim::{ConjugateGradient, CGBetaFormula};
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
    /// * `beta_formula` - Method for computing β (FletcherReeves, PolakRibiere, or HestenesStiefel)
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
    fn compute_beta(&self, grad_new: &Vector<f32>, grad_old: &Vector<f32>, d_old: &Vector<f32>) -> f32 {
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

    fn minimize<F, G>(
        &mut self,
        objective: F,
        gradient: G,
        x0: Vector<f32>,
    ) -> OptimizationResult
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
            let d = if let (Some(d_old), Some(g_old)) = (&self.prev_direction, &self.prev_gradient) {
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

/// Stochastic Gradient Descent optimizer.
///
/// SGD updates parameters using the gradient of the loss function:
///
/// ```text
/// θ = θ - η * ∇L(θ)
/// ```
///
/// With momentum:
///
/// ```text
/// v = γ * v + η * ∇L(θ)
/// θ = θ - v
/// ```
///
/// where:
/// - θ is the parameter vector
/// - η is the learning rate
/// - γ is the momentum coefficient
/// - v is the velocity vector
/// - ∇L(θ) is the gradient of the loss
///
/// # Example
///
/// ```
/// use aprender::optim::SGD;
/// use aprender::primitives::Vector;
///
/// // Create SGD with momentum
/// let mut optimizer = SGD::new(0.1).with_momentum(0.9);
///
/// let mut params = Vector::from_slice(&[0.0, 0.0]);
/// let gradients = Vector::from_slice(&[1.0, 2.0]);
///
/// // First step
/// optimizer.step(&mut params, &gradients);
///
/// // With momentum, velocity builds up over iterations
/// optimizer.step(&mut params, &gradients);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGD {
    /// Learning rate (step size)
    learning_rate: f32,
    /// Momentum coefficient (0.0 = no momentum)
    momentum: f32,
    /// Velocity vectors for momentum
    velocity: Option<Vec<f32>>,
}

impl SGD {
    /// Creates a new SGD optimizer with the given learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Step size for parameter updates
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::SGD;
    ///
    /// let optimizer = SGD::new(0.01);
    /// assert!((optimizer.learning_rate() - 0.01).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            velocity: None,
        }
    }

    /// Sets the momentum coefficient.
    ///
    /// Momentum helps accelerate SGD in the relevant direction and dampens
    /// oscillations. Typical values are 0.9 or 0.99.
    ///
    /// # Arguments
    ///
    /// * `momentum` - Momentum coefficient between 0.0 and 1.0
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::SGD;
    ///
    /// let optimizer = SGD::new(0.01).with_momentum(0.9);
    /// assert!((optimizer.momentum() - 0.9).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Returns the learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Returns the momentum coefficient.
    #[must_use]
    pub fn momentum(&self) -> f32 {
        self.momentum
    }

    /// Updates parameters using gradients.
    ///
    /// If momentum is enabled, maintains velocity vectors for each parameter.
    ///
    /// # Arguments
    ///
    /// * `params` - Mutable reference to parameter vector
    /// * `gradients` - Gradient vector (same length as params)
    ///
    /// # Panics
    ///
    /// Panics if params and gradients have different lengths.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::SGD;
    /// use aprender::primitives::Vector;
    ///
    /// let mut optimizer = SGD::new(0.1);
    /// let mut params = Vector::from_slice(&[1.0, 2.0]);
    /// let gradients = Vector::from_slice(&[0.5, 1.0]);
    ///
    /// optimizer.step(&mut params, &gradients);
    ///
    /// // params = [1.0 - 0.1*0.5, 2.0 - 0.1*1.0] = [0.95, 1.9]
    /// assert!((params[0] - 0.95).abs() < 1e-6);
    /// assert!((params[1] - 1.9).abs() < 1e-6);
    /// ```
    pub fn step(&mut self, params: &mut Vector<f32>, gradients: &Vector<f32>) {
        assert_eq!(
            params.len(),
            gradients.len(),
            "Parameters and gradients must have same length"
        );

        let n = params.len();

        if self.momentum > 0.0 {
            // Initialize velocity if needed
            if self.velocity.is_none()
                || self
                    .velocity
                    .as_ref()
                    .expect("Velocity must be initialized")
                    .len()
                    != n
            {
                self.velocity = Some(vec![0.0; n]);
            }

            let velocity = self
                .velocity
                .as_mut()
                .expect("Velocity was just initialized");

            for i in 0..n {
                // v = γ * v + η * gradient
                velocity[i] = self.momentum * velocity[i] + self.learning_rate * gradients[i];
                // θ = θ - v
                params[i] -= velocity[i];
            }
        } else {
            // Standard SGD: θ = θ - η * gradient
            for i in 0..n {
                params[i] -= self.learning_rate * gradients[i];
            }
        }
    }

    /// Resets the optimizer state (velocity vectors).
    ///
    /// Call this when starting training on a new model or after significant
    /// changes to the optimization problem.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::SGD;
    /// use aprender::primitives::Vector;
    ///
    /// let mut optimizer = SGD::new(0.1).with_momentum(0.9);
    /// let mut params = Vector::from_slice(&[1.0]);
    /// let gradients = Vector::from_slice(&[1.0]);
    ///
    /// optimizer.step(&mut params, &gradients);
    /// optimizer.reset();
    ///
    /// // Velocity is now reset to zero
    /// ```
    pub fn reset(&mut self) {
        self.velocity = None;
    }

    /// Returns whether momentum is enabled.
    #[must_use]
    pub fn has_momentum(&self) -> bool {
        self.momentum > 0.0
    }
}

/// Adam (Adaptive Moment Estimation) optimizer.
///
/// Adam combines the benefits of AdaGrad and RMSprop by computing adaptive learning
/// rates for each parameter using estimates of first and second moments of gradients.
///
/// Update rules:
///
/// ```text
/// m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
/// v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
/// m̂_t = m_t / (1 - β₁^t)
/// v̂_t = v_t / (1 - β₂^t)
/// θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
/// ```
///
/// where:
/// - m_t is the first moment (mean) estimate
/// - v_t is the second moment (variance) estimate
/// - β₁, β₂ are exponential decay rates (typically 0.9, 0.999)
/// - α is the learning rate (step size)
/// - ε is a small constant for numerical stability (typically 1e-8)
///
/// # Example
///
/// ```
/// use aprender::optim::Adam;
/// use aprender::primitives::Vector;
///
/// // Create Adam optimizer with default hyperparameters
/// let mut optimizer = Adam::new(0.001);
///
/// let mut params = Vector::from_slice(&[1.0, 2.0]);
/// let gradients = Vector::from_slice(&[0.1, 0.2]);
///
/// // Update parameters with adaptive learning rates
/// optimizer.step(&mut params, &gradients);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam {
    /// Learning rate (step size)
    learning_rate: f32,
    /// Exponential decay rate for first moment estimates (default: 0.9)
    beta1: f32,
    /// Exponential decay rate for second moment estimates (default: 0.999)
    beta2: f32,
    /// Small constant for numerical stability (default: 1e-8)
    epsilon: f32,
    /// First moment estimates (mean)
    m: Option<Vec<f32>>,
    /// Second moment estimates (uncentered variance)
    v: Option<Vec<f32>>,
    /// Number of steps taken (for bias correction)
    t: usize,
}

impl Adam {
    /// Creates a new Adam optimizer with the given learning rate and default hyperparameters.
    ///
    /// Default values:
    /// - beta1 = 0.9
    /// - beta2 = 0.999
    /// - epsilon = 1e-8
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Step size (typical values: 0.001, 0.0001)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::Adam;
    ///
    /// let optimizer = Adam::new(0.001);
    /// assert!((optimizer.learning_rate() - 0.001).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: None,
            v: None,
            t: 0,
        }
    }

    /// Sets the beta1 parameter (exponential decay rate for first moment).
    ///
    /// # Arguments
    ///
    /// * `beta1` - Value between 0.0 and 1.0 (typical: 0.9)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::Adam;
    ///
    /// let optimizer = Adam::new(0.001).with_beta1(0.95);
    /// assert!((optimizer.beta1() - 0.95).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Sets the beta2 parameter (exponential decay rate for second moment).
    ///
    /// # Arguments
    ///
    /// * `beta2` - Value between 0.0 and 1.0 (typical: 0.999)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::Adam;
    ///
    /// let optimizer = Adam::new(0.001).with_beta2(0.9999);
    /// assert!((optimizer.beta2() - 0.9999).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Sets the epsilon parameter (numerical stability constant).
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Small positive value (typical: 1e-8)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::Adam;
    ///
    /// let optimizer = Adam::new(0.001).with_epsilon(1e-7);
    /// assert!((optimizer.epsilon() - 1e-7).abs() < 1e-15);
    /// ```
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Returns the learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Returns the beta1 parameter.
    #[must_use]
    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Returns the beta2 parameter.
    #[must_use]
    pub fn beta2(&self) -> f32 {
        self.beta2
    }

    /// Returns the epsilon parameter.
    #[must_use]
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Returns the number of steps taken.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.t
    }

    /// Updates parameters using gradients with adaptive learning rates.
    ///
    /// # Arguments
    ///
    /// * `params` - Mutable reference to parameter vector
    /// * `gradients` - Gradient vector (same length as params)
    ///
    /// # Panics
    ///
    /// Panics if params and gradients have different lengths.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::Adam;
    /// use aprender::primitives::Vector;
    ///
    /// let mut optimizer = Adam::new(0.001);
    /// let mut params = Vector::from_slice(&[1.0, 2.0]);
    /// let gradients = Vector::from_slice(&[0.1, 0.2]);
    ///
    /// optimizer.step(&mut params, &gradients);
    /// ```
    pub fn step(&mut self, params: &mut Vector<f32>, gradients: &Vector<f32>) {
        assert_eq!(
            params.len(),
            gradients.len(),
            "Parameters and gradients must have same length"
        );

        let n = params.len();

        // Initialize moment estimates if needed
        if self.m.is_none()
            || self
                .m
                .as_ref()
                .expect("First moment estimate must be initialized")
                .len()
                != n
        {
            self.m = Some(vec![0.0; n]);
            self.v = Some(vec![0.0; n]);
            self.t = 0;
        }

        self.t += 1;
        let t = self.t as f32;

        let m = self.m.as_mut().expect("First moment was just initialized");
        let v = self.v.as_mut().expect("Second moment was just initialized");

        // Compute bias-corrected learning rate
        let lr_t =
            self.learning_rate * (1.0 - self.beta2.powf(t)).sqrt() / (1.0 - self.beta1.powf(t));

        for i in 0..n {
            let g = gradients[i];

            // Update biased first moment estimate
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;

            // Update biased second raw moment estimate
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;

            // Update parameters
            params[i] -= lr_t * m[i] / (v[i].sqrt() + self.epsilon);
        }
    }

    /// Resets the optimizer state (moment estimates and step counter).
    ///
    /// Call this when starting training on a new model or after significant
    /// changes to the optimization problem.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::Adam;
    /// use aprender::primitives::Vector;
    ///
    /// let mut optimizer = Adam::new(0.001);
    /// let mut params = Vector::from_slice(&[1.0]);
    /// let gradients = Vector::from_slice(&[1.0]);
    ///
    /// optimizer.step(&mut params, &gradients);
    /// assert_eq!(optimizer.steps(), 1);
    ///
    /// optimizer.reset();
    /// assert_eq!(optimizer.steps(), 0);
    /// ```
    pub fn reset(&mut self) {
        self.m = None;
        self.v = None;
        self.t = 0;
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut Vector<f32>, gradients: &Vector<f32>) {
        self.step(params, gradients);
    }

    fn reset(&mut self) {
        self.reset();
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
/// let mut adam = SGD::new(0.01);
/// let mut params = Vector::from_slice(&[1.0, 2.0]);
/// let grad = Vector::from_slice(&[0.1, 0.2]);
/// adam.step(&mut params, &grad);
/// ```
pub trait Optimizer {
    /// Stochastic update (mini-batch mode) - for SGD, Adam, RMSprop.
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

impl Optimizer for SGD {
    fn step(&mut self, params: &mut Vector<f32>, gradients: &Vector<f32>) {
        self.step(params, gradients);
    }

    fn reset(&mut self) {
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_new() {
        let optimizer = SGD::new(0.01);
        assert!((optimizer.learning_rate() - 0.01).abs() < 1e-6);
        assert!((optimizer.momentum() - 0.0).abs() < 1e-6);
        assert!(!optimizer.has_momentum());
    }

    #[test]
    fn test_sgd_with_momentum() {
        let optimizer = SGD::new(0.01).with_momentum(0.9);
        assert!((optimizer.learning_rate() - 0.01).abs() < 1e-6);
        assert!((optimizer.momentum() - 0.9).abs() < 1e-6);
        assert!(optimizer.has_momentum());
    }

    #[test]
    fn test_sgd_step_basic() {
        let mut optimizer = SGD::new(0.1);
        let mut params = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let gradients = Vector::from_slice(&[1.0, 2.0, 3.0]);

        optimizer.step(&mut params, &gradients);

        // params = params - lr * gradients
        assert!((params[0] - 0.9).abs() < 1e-6);
        assert!((params[1] - 1.8).abs() < 1e-6);
        assert!((params[2] - 2.7).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_step_with_momentum() {
        let mut optimizer = SGD::new(0.1).with_momentum(0.9);
        let mut params = Vector::from_slice(&[1.0, 1.0]);
        let gradients = Vector::from_slice(&[1.0, 1.0]);

        // First step: v = 0.9*0 + 0.1*1 = 0.1, params = 1.0 - 0.1 = 0.9
        optimizer.step(&mut params, &gradients);
        assert!((params[0] - 0.9).abs() < 1e-6);

        // Second step: v = 0.9*0.1 + 0.1*1 = 0.19, params = 0.9 - 0.19 = 0.71
        optimizer.step(&mut params, &gradients);
        assert!((params[0] - 0.71).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_momentum_accumulation() {
        let mut optimizer = SGD::new(0.1).with_momentum(0.9);
        let mut params = Vector::from_slice(&[0.0]);
        let gradients = Vector::from_slice(&[1.0]);

        // Velocity should accumulate over iterations
        let mut prev_step = 0.0;
        for _ in 0..10 {
            let before = params[0];
            optimizer.step(&mut params, &gradients);
            let step = before - params[0];
            // Each step should be larger (velocity builds up)
            assert!(step >= prev_step);
            prev_step = step;
        }
    }

    #[test]
    fn test_sgd_reset() {
        let mut optimizer = SGD::new(0.1).with_momentum(0.9);
        let mut params = Vector::from_slice(&[1.0]);
        let gradients = Vector::from_slice(&[1.0]);

        optimizer.step(&mut params, &gradients);
        optimizer.reset();

        // After reset, velocity should be zero again
        let mut params2 = Vector::from_slice(&[1.0]);
        optimizer.step(&mut params2, &gradients);
        assert!((params2[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_zero_gradient() {
        let mut optimizer = SGD::new(0.1);
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let gradients = Vector::from_slice(&[0.0, 0.0]);

        optimizer.step(&mut params, &gradients);

        // No change with zero gradients
        assert!((params[0] - 1.0).abs() < 1e-6);
        assert!((params[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_negative_gradients() {
        let mut optimizer = SGD::new(0.1);
        let mut params = Vector::from_slice(&[1.0]);
        let gradients = Vector::from_slice(&[-1.0]);

        optimizer.step(&mut params, &gradients);

        // params = 1.0 - 0.1 * (-1.0) = 1.1
        assert!((params[0] - 1.1).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_sgd_mismatched_lengths() {
        let mut optimizer = SGD::new(0.1);
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let gradients = Vector::from_slice(&[1.0]);

        optimizer.step(&mut params, &gradients);
    }

    #[test]
    fn test_sgd_large_learning_rate() {
        let mut optimizer = SGD::new(10.0);
        let mut params = Vector::from_slice(&[1.0]);
        let gradients = Vector::from_slice(&[0.1]);

        optimizer.step(&mut params, &gradients);

        // params = 1.0 - 10.0 * 0.1 = 0.0
        assert!((params[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_small_learning_rate() {
        let mut optimizer = SGD::new(0.001);
        let mut params = Vector::from_slice(&[1.0]);
        let gradients = Vector::from_slice(&[1.0]);

        optimizer.step(&mut params, &gradients);

        // params = 1.0 - 0.001 * 1.0 = 0.999
        assert!((params[0] - 0.999).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_clone() {
        let optimizer = SGD::new(0.01).with_momentum(0.9);
        let cloned = optimizer.clone();

        assert!((cloned.learning_rate() - optimizer.learning_rate()).abs() < 1e-6);
        assert!((cloned.momentum() - optimizer.momentum()).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_multiple_steps() {
        let mut optimizer = SGD::new(0.1);
        let mut params = Vector::from_slice(&[10.0]);
        let gradients = Vector::from_slice(&[1.0]);

        for _ in 0..10 {
            optimizer.step(&mut params, &gradients);
        }

        // params = 10.0 - 10 * 0.1 * 1.0 = 9.0
        assert!((params[0] - 9.0).abs() < 1e-4);
    }

    #[test]
    fn test_sgd_velocity_reinitialization() {
        let mut optimizer = SGD::new(0.1).with_momentum(0.9);

        // First with 2 params
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let gradients = Vector::from_slice(&[1.0, 1.0]);
        optimizer.step(&mut params, &gradients);

        // Now with 3 params - velocity should reinitialize
        let mut params3 = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let gradients3 = Vector::from_slice(&[1.0, 1.0, 1.0]);
        optimizer.step(&mut params3, &gradients3);

        // Should work without error, velocity reinitialized
        assert!((params3[0] - 0.9).abs() < 1e-6);
    }

    // ==================== Adam Tests ====================

    #[test]
    fn test_adam_new() {
        let optimizer = Adam::new(0.001);
        assert!((optimizer.learning_rate() - 0.001).abs() < 1e-9);
        assert!((optimizer.beta1() - 0.9).abs() < 1e-9);
        assert!((optimizer.beta2() - 0.999).abs() < 1e-9);
        assert!((optimizer.epsilon() - 1e-8).abs() < 1e-15);
        assert_eq!(optimizer.steps(), 0);
    }

    #[test]
    fn test_adam_with_beta1() {
        let optimizer = Adam::new(0.001).with_beta1(0.95);
        assert!((optimizer.beta1() - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_adam_with_beta2() {
        let optimizer = Adam::new(0.001).with_beta2(0.9999);
        assert!((optimizer.beta2() - 0.9999).abs() < 1e-9);
    }

    #[test]
    fn test_adam_with_epsilon() {
        let optimizer = Adam::new(0.001).with_epsilon(1e-7);
        assert!((optimizer.epsilon() - 1e-7).abs() < 1e-15);
    }

    #[test]
    fn test_adam_step_basic() {
        let mut optimizer = Adam::new(0.001);
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let gradients = Vector::from_slice(&[0.1, 0.2]);

        optimizer.step(&mut params, &gradients);

        // Adam should update parameters (exact values depend on bias correction)
        assert!(params[0] < 1.0); // Should decrease
        assert!(params[1] < 2.0); // Should decrease
        assert_eq!(optimizer.steps(), 1);
    }

    #[test]
    fn test_adam_multiple_steps() {
        let mut optimizer = Adam::new(0.001);
        let mut params = Vector::from_slice(&[1.0]);
        let gradients = Vector::from_slice(&[1.0]);

        let initial = params[0];
        for _ in 0..5 {
            optimizer.step(&mut params, &gradients);
        }

        // Parameters should decrease over multiple steps
        assert!(params[0] < initial);
        assert_eq!(optimizer.steps(), 5);
    }

    #[test]
    fn test_adam_bias_correction() {
        let mut optimizer = Adam::new(0.01);
        let mut params = Vector::from_slice(&[10.0]);
        let gradients = Vector::from_slice(&[1.0]);

        // First step should have larger effective learning rate due to bias correction
        optimizer.step(&mut params, &gradients);
        let first_step_size = 10.0 - params[0];

        // Reset and try second step
        let mut optimizer2 = Adam::new(0.01);
        let mut params2 = Vector::from_slice(&[10.0]);
        optimizer2.step(&mut params2, &gradients);
        optimizer2.step(&mut params2, &gradients);
        let second_step_size = params[0] - params2[0];

        // First step should have larger update due to bias correction
        assert!(first_step_size > second_step_size * 0.5);
    }

    #[test]
    fn test_adam_reset() {
        let mut optimizer = Adam::new(0.001);
        let mut params = Vector::from_slice(&[1.0]);
        let gradients = Vector::from_slice(&[1.0]);

        optimizer.step(&mut params, &gradients);
        assert_eq!(optimizer.steps(), 1);

        optimizer.reset();
        assert_eq!(optimizer.steps(), 0);
    }

    #[test]
    fn test_adam_zero_gradient() {
        let mut optimizer = Adam::new(0.001);
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let gradients = Vector::from_slice(&[0.0, 0.0]);

        optimizer.step(&mut params, &gradients);

        // With zero gradients, params should not change significantly
        assert!((params[0] - 1.0).abs() < 0.01);
        assert!((params[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_adam_negative_gradients() {
        let mut optimizer = Adam::new(0.001);
        let mut params = Vector::from_slice(&[1.0]);
        let gradients = Vector::from_slice(&[-1.0]);

        optimizer.step(&mut params, &gradients);

        // With negative gradient, params should increase
        assert!(params[0] > 1.0);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_adam_mismatched_lengths() {
        let mut optimizer = Adam::new(0.001);
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let gradients = Vector::from_slice(&[1.0]);

        optimizer.step(&mut params, &gradients);
    }

    #[test]
    fn test_adam_clone() {
        let optimizer = Adam::new(0.001).with_beta1(0.95).with_beta2(0.9999);
        let cloned = optimizer.clone();

        assert!((cloned.learning_rate() - optimizer.learning_rate()).abs() < 1e-9);
        assert!((cloned.beta1() - optimizer.beta1()).abs() < 1e-9);
        assert!((cloned.beta2() - optimizer.beta2()).abs() < 1e-9);
        assert!((cloned.epsilon() - optimizer.epsilon()).abs() < 1e-15);
    }

    #[test]
    fn test_adam_adaptive_learning() {
        // Test that Adam adapts to gradient magnitudes
        let mut optimizer = Adam::new(0.01);
        let mut params = Vector::from_slice(&[1.0, 1.0]);

        // Large gradient on first param, small on second
        let gradients1 = Vector::from_slice(&[10.0, 0.1]);
        optimizer.step(&mut params, &gradients1);

        let step1_0 = 1.0 - params[0];
        let step1_1 = 1.0 - params[1];

        // Continue with same gradients
        optimizer.step(&mut params, &gradients1);

        // Adam should adapt - second param should take relatively larger steps
        // because it has more consistent small gradients
        assert!(step1_0 > 0.0);
        assert!(step1_1 > 0.0);
    }

    #[test]
    fn test_adam_vs_sgd_behavior() {
        // Test that Adam and SGD behave differently (not necessarily one better)
        let mut adam = Adam::new(0.001);
        let mut sgd = SGD::new(0.1);

        let mut params_adam = Vector::from_slice(&[5.0]);
        let mut params_sgd = Vector::from_slice(&[5.0]);

        // Gradient pointing towards 0
        for _ in 0..10 {
            let gradients = Vector::from_slice(&[1.0]);
            adam.step(&mut params_adam, &gradients);
            sgd.step(&mut params_sgd, &gradients);
        }

        // Both should decrease but behave differently
        assert!(params_adam[0] < 5.0);
        assert!(params_sgd[0] < 5.0);
        // They should produce different results due to different mechanisms
        assert!((params_adam[0] - params_sgd[0]).abs() > 0.01);
    }

    #[test]
    fn test_adam_moment_initialization() {
        let mut optimizer = Adam::new(0.001);

        // First with 2 params
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let gradients = Vector::from_slice(&[0.1, 0.2]);
        optimizer.step(&mut params, &gradients);

        // Now with 3 params - moments should reinitialize
        let mut params3 = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let gradients3 = Vector::from_slice(&[0.1, 0.2, 0.3]);
        optimizer.step(&mut params3, &gradients3);

        // Should work without error
        assert!(params3[0] < 1.0);
        assert!(params3[1] < 2.0);
        assert!(params3[2] < 3.0);
    }

    #[test]
    fn test_adam_numerical_stability() {
        let mut optimizer = Adam::new(0.001);
        let mut params = Vector::from_slice(&[1.0]);

        // Very large gradients should be handled stably
        let gradients = Vector::from_slice(&[1000.0]);
        optimizer.step(&mut params, &gradients);

        // Should not produce NaN or extreme values
        assert!(!params[0].is_nan());
        assert!(params[0].is_finite());
    }

    // ==================== Line Search Tests ====================

    #[test]
    fn test_backtracking_line_search_new() {
        let ls = BacktrackingLineSearch::new(1e-4, 0.5, 50);
        assert!((ls.c1 - 1e-4).abs() < 1e-10);
        assert!((ls.rho - 0.5).abs() < 1e-10);
        assert_eq!(ls.max_iter, 50);
    }

    #[test]
    fn test_backtracking_line_search_default() {
        let ls = BacktrackingLineSearch::default();
        assert!((ls.c1 - 1e-4).abs() < 1e-10);
        assert!((ls.rho - 0.5).abs() < 1e-10);
        assert_eq!(ls.max_iter, 50);
    }

    #[test]
    fn test_backtracking_line_search_quadratic() {
        // Test on simple quadratic: f(x) = x^2
        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let ls = BacktrackingLineSearch::default();
        let x = Vector::from_slice(&[2.0]);
        let d = Vector::from_slice(&[-4.0]); // Gradient direction at x=2

        let alpha = ls.search(&f, &grad, &x, &d);

        // Should find positive step size
        assert!(alpha > 0.0);
        assert!(alpha <= 1.0);

        // Verify Armijo condition is satisfied
        let x_new_data = x[0] + alpha * d[0];
        let x_new = Vector::from_slice(&[x_new_data]);
        let fx = f(&x);
        let fx_new = f(&x_new);
        let grad_x = grad(&x);
        let dir_deriv = grad_x[0] * d[0];

        assert!(fx_new <= fx + ls.c1 * alpha * dir_deriv);
    }

    #[test]
    fn test_backtracking_line_search_rosenbrock() {
        // Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        let f = |v: &Vector<f32>| {
            let x = v[0];
            let y = v[1];
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        };

        let grad = |v: &Vector<f32>| {
            let x = v[0];
            let y = v[1];
            let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
            let dy = 200.0 * (y - x * x);
            Vector::from_slice(&[dx, dy])
        };

        let ls = BacktrackingLineSearch::default();
        let x = Vector::from_slice(&[0.0, 0.0]);
        let g = grad(&x);
        let d = Vector::from_slice(&[-g[0], -g[1]]); // Descent direction

        let alpha = ls.search(&f, &grad, &x, &d);

        assert!(alpha > 0.0);
    }

    #[test]
    fn test_backtracking_line_search_multidimensional() {
        // f(x) = ||x||^2
        let f = |x: &Vector<f32>| {
            let mut sum = 0.0;
            for i in 0..x.len() {
                sum += x[i] * x[i];
            }
            sum
        };

        let grad = |x: &Vector<f32>| {
            let mut g = Vector::zeros(x.len());
            for i in 0..x.len() {
                g[i] = 2.0 * x[i];
            }
            g
        };

        let ls = BacktrackingLineSearch::default();
        let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let g = grad(&x);
        let d = Vector::from_slice(&[-g[0], -g[1], -g[2]]);

        let alpha = ls.search(&f, &grad, &x, &d);

        assert!(alpha > 0.0);
        assert!(alpha <= 1.0);
    }

    #[test]
    fn test_wolfe_line_search_new() {
        let ls = WolfeLineSearch::new(1e-4, 0.9, 50);
        assert!((ls.c1 - 1e-4).abs() < 1e-10);
        assert!((ls.c2 - 0.9).abs() < 1e-10);
        assert_eq!(ls.max_iter, 50);
    }

    #[test]
    #[should_panic(expected = "Wolfe conditions require 0 < c1 < c2 < 1")]
    fn test_wolfe_line_search_invalid_c1_c2() {
        // c1 >= c2 should panic
        WolfeLineSearch::new(0.9, 0.5, 50);
    }

    #[test]
    #[should_panic(expected = "Wolfe conditions require 0 < c1 < c2 < 1")]
    fn test_wolfe_line_search_c1_negative() {
        WolfeLineSearch::new(-0.1, 0.9, 50);
    }

    #[test]
    #[should_panic(expected = "Wolfe conditions require 0 < c1 < c2 < 1")]
    fn test_wolfe_line_search_c2_too_large() {
        WolfeLineSearch::new(0.1, 1.5, 50);
    }

    #[test]
    fn test_wolfe_line_search_default() {
        let ls = WolfeLineSearch::default();
        assert!((ls.c1 - 1e-4).abs() < 1e-10);
        assert!((ls.c2 - 0.9).abs() < 1e-10);
        assert_eq!(ls.max_iter, 50);
    }

    #[test]
    fn test_wolfe_line_search_quadratic() {
        // Test on simple quadratic: f(x) = x^2
        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let ls = WolfeLineSearch::default();
        let x = Vector::from_slice(&[2.0]);
        let d = Vector::from_slice(&[-4.0]); // Gradient direction at x=2

        let alpha = ls.search(&f, &grad, &x, &d);

        // Should find positive step size
        assert!(alpha > 0.0);

        // Verify both Wolfe conditions
        let x_new_data = x[0] + alpha * d[0];
        let x_new = Vector::from_slice(&[x_new_data]);
        let fx = f(&x);
        let fx_new = f(&x_new);
        let grad_x = grad(&x);
        let grad_new = grad(&x_new);
        let dir_deriv = grad_x[0] * d[0];
        let dir_deriv_new = grad_new[0] * d[0];

        // Armijo condition
        assert!(fx_new <= fx + ls.c1 * alpha * dir_deriv + 1e-6);

        // Curvature condition
        assert!(dir_deriv_new.abs() <= ls.c2 * dir_deriv.abs() + 1e-6);
    }

    #[test]
    fn test_wolfe_line_search_multidimensional() {
        // f(x) = ||x||^2
        let f = |x: &Vector<f32>| {
            let mut sum = 0.0;
            for i in 0..x.len() {
                sum += x[i] * x[i];
            }
            sum
        };

        let grad = |x: &Vector<f32>| {
            let mut g = Vector::zeros(x.len());
            for i in 0..x.len() {
                g[i] = 2.0 * x[i];
            }
            g
        };

        let ls = WolfeLineSearch::default();
        let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let g = grad(&x);
        let d = Vector::from_slice(&[-g[0], -g[1], -g[2]]);

        let alpha = ls.search(&f, &grad, &x, &d);

        assert!(alpha > 0.0);
    }

    #[test]
    fn test_backtracking_vs_wolfe() {
        // Compare backtracking and Wolfe on same problem
        let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

        let bt = BacktrackingLineSearch::default();
        let wolfe = WolfeLineSearch::default();

        let x = Vector::from_slice(&[1.0, 1.0]);
        let g = grad(&x);
        let d = Vector::from_slice(&[-g[0], -g[1]]);

        let alpha_bt = bt.search(&f, &grad, &x, &d);
        let alpha_wolfe = wolfe.search(&f, &grad, &x, &d);

        // Both should find valid step sizes
        assert!(alpha_bt > 0.0);
        assert!(alpha_wolfe > 0.0);

        // Wolfe often finds larger steps due to curvature condition
        // but not always, so just verify both are reasonable
        assert!(alpha_bt <= 1.0);
    }

    // ==================== OptimizationResult Tests ====================

    #[test]
    fn test_optimization_result_converged() {
        let solution = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = OptimizationResult::converged(solution.clone(), 42);

        assert_eq!(result.solution.len(), 3);
        assert_eq!(result.iterations, 42);
        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!((result.objective_value - 0.0).abs() < 1e-10);
        assert!((result.gradient_norm - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_optimization_result_max_iterations() {
        let solution = Vector::from_slice(&[5.0]);
        let result = OptimizationResult::max_iterations(solution);

        assert_eq!(result.iterations, 0);
        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
    }

    #[test]
    fn test_convergence_status_equality() {
        assert_eq!(ConvergenceStatus::Converged, ConvergenceStatus::Converged);
        assert_ne!(ConvergenceStatus::Converged, ConvergenceStatus::MaxIterations);
        assert_ne!(ConvergenceStatus::Stalled, ConvergenceStatus::NumericalError);
    }

    // ==================== L-BFGS Tests ====================

    #[test]
    fn test_lbfgs_new() {
        let optimizer = LBFGS::new(100, 1e-5, 10);
        assert_eq!(optimizer.max_iter, 100);
        assert!((optimizer.tol - 1e-5).abs() < 1e-10);
        assert_eq!(optimizer.m, 10);
        assert_eq!(optimizer.s_history.len(), 0);
        assert_eq!(optimizer.y_history.len(), 0);
    }

    #[test]
    fn test_lbfgs_simple_quadratic() {
        // Minimize f(x) = x^2, optimal at x = 0
        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let mut optimizer = LBFGS::new(100, 1e-5, 5);
        let x0 = Vector::from_slice(&[5.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!(result.solution[0].abs() < 1e-3);
        assert!(result.iterations < 100);
        assert!(result.gradient_norm < 1e-5);
    }

    #[test]
    fn test_lbfgs_multidimensional_quadratic() {
        // Minimize f(x) = ||x - c||^2 where c = [1, 2, 3]
        let c = vec![1.0, 2.0, 3.0];
        let f = |x: &Vector<f32>| {
            let mut sum = 0.0;
            for i in 0..x.len() {
                sum += (x[i] - c[i]).powi(2);
            }
            sum
        };

        let grad = |x: &Vector<f32>| {
            let mut g = Vector::zeros(x.len());
            for i in 0..x.len() {
                g[i] = 2.0 * (x[i] - c[i]);
            }
            g
        };

        let mut optimizer = LBFGS::new(100, 1e-5, 10);
        let x0 = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        for i in 0..3 {
            assert!((result.solution[i] - c[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn test_lbfgs_rosenbrock() {
        // Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        // Global minimum at (1, 1)
        let f = |v: &Vector<f32>| {
            let x = v[0];
            let y = v[1];
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        };

        let grad = |v: &Vector<f32>| {
            let x = v[0];
            let y = v[1];
            let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
            let dy = 200.0 * (y - x * x);
            Vector::from_slice(&[dx, dy])
        };

        let mut optimizer = LBFGS::new(200, 1e-4, 10);
        let x0 = Vector::from_slice(&[0.0, 0.0]);
        let result = optimizer.minimize(f, grad, x0);

        // Should converge close to (1, 1)
        assert!(
            result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::MaxIterations
        );
        assert!((result.solution[0] - 1.0).abs() < 0.1);
        assert!((result.solution[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_lbfgs_sphere() {
        // Sphere function: f(x) = sum(x_i^2)
        let f = |x: &Vector<f32>| {
            let mut sum = 0.0;
            for i in 0..x.len() {
                sum += x[i] * x[i];
            }
            sum
        };

        let grad = |x: &Vector<f32>| {
            let mut g = Vector::zeros(x.len());
            for i in 0..x.len() {
                g[i] = 2.0 * x[i];
            }
            g
        };

        let mut optimizer = LBFGS::new(100, 1e-5, 10);
        let x0 = Vector::from_slice(&[5.0, -3.0, 2.0, -1.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        for i in 0..4 {
            assert!(result.solution[i].abs() < 1e-3);
        }
    }

    #[test]
    fn test_lbfgs_different_history_sizes() {
        let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        let x0 = Vector::from_slice(&[3.0, 4.0]);

        // Small history
        let mut opt_small = LBFGS::new(100, 1e-5, 3);
        let result_small = opt_small.minimize(&f, &grad, x0.clone());
        assert_eq!(result_small.status, ConvergenceStatus::Converged);

        // Large history
        let mut opt_large = LBFGS::new(100, 1e-5, 20);
        let result_large = opt_large.minimize(&f, &grad, x0);
        assert_eq!(result_large.status, ConvergenceStatus::Converged);

        // Both should converge to same solution
        assert!((result_small.solution[0] - result_large.solution[0]).abs() < 1e-3);
        assert!((result_small.solution[1] - result_large.solution[1]).abs() < 1e-3);
    }

    #[test]
    fn test_lbfgs_reset() {
        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let mut optimizer = LBFGS::new(100, 1e-5, 10);
        let x0 = Vector::from_slice(&[5.0]);

        // First run
        optimizer.minimize(&f, &grad, x0.clone());
        assert!(optimizer.s_history.len() > 0);

        // Reset
        optimizer.reset();
        assert_eq!(optimizer.s_history.len(), 0);
        assert_eq!(optimizer.y_history.len(), 0);

        // Second run should work
        let result = optimizer.minimize(&f, &grad, x0);
        assert_eq!(result.status, ConvergenceStatus::Converged);
    }

    #[test]
    fn test_lbfgs_max_iterations() {
        // Use Rosenbrock with very few iterations to force max_iter
        let f = |v: &Vector<f32>| {
            let x = v[0];
            let y = v[1];
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        };

        let grad = |v: &Vector<f32>| {
            let x = v[0];
            let y = v[1];
            let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
            let dy = 200.0 * (y - x * x);
            Vector::from_slice(&[dx, dy])
        };

        let mut optimizer = LBFGS::new(3, 1e-10, 5);
        let x0 = Vector::from_slice(&[0.0, 0.0]);
        let result = optimizer.minimize(f, grad, x0);

        // With only 3 iterations, should hit max_iter on Rosenbrock
        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
        assert_eq!(result.iterations, 3);
    }

    #[test]
    #[should_panic(expected = "does not support stochastic")]
    fn test_lbfgs_step_panics() {
        let mut optimizer = LBFGS::new(100, 1e-5, 10);
        let mut params = Vector::from_slice(&[1.0]);
        let grad = Vector::from_slice(&[0.1]);

        // Should panic - L-BFGS doesn't support step()
        optimizer.step(&mut params, &grad);
    }

    #[test]
    fn test_lbfgs_numerical_error_detection() {
        // Function that produces NaN
        let f = |x: &Vector<f32>| {
            if x[0] < -100.0 {
                f32::NAN
            } else {
                x[0] * x[0]
            }
        };
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let mut optimizer = LBFGS::new(100, 1e-5, 5);
        let x0 = Vector::from_slice(&[0.0]);
        let result = optimizer.minimize(f, grad, x0);

        // Should detect numerical error or converge normally
        assert!(
            result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::NumericalError
                || result.status == ConvergenceStatus::MaxIterations
        );
    }

    #[test]
    fn test_lbfgs_computes_elapsed_time() {
        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let mut optimizer = LBFGS::new(100, 1e-5, 5);
        let x0 = Vector::from_slice(&[5.0]);
        let result = optimizer.minimize(f, grad, x0);

        // Should have non-zero elapsed time
        assert!(result.elapsed_time.as_nanos() > 0);
    }

    #[test]
    fn test_lbfgs_gradient_norm_tracking() {
        let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

        let mut optimizer = LBFGS::new(100, 1e-5, 10);
        let x0 = Vector::from_slice(&[3.0, 4.0]);
        let result = optimizer.minimize(f, grad, x0);

        // Gradient norm at convergence should be small
        if result.status == ConvergenceStatus::Converged {
            assert!(result.gradient_norm < 1e-5);
        }
    }

    // ==================== Conjugate Gradient Tests ====================

    #[test]
    fn test_cg_new_fletcher_reeves() {
        let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::FletcherReeves);
        assert_eq!(optimizer.max_iter, 100);
        assert!((optimizer.tol - 1e-5).abs() < 1e-10);
        assert_eq!(optimizer.beta_formula, CGBetaFormula::FletcherReeves);
        assert_eq!(optimizer.restart_interval, 0);
    }

    #[test]
    fn test_cg_new_polak_ribiere() {
        let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
        assert_eq!(optimizer.beta_formula, CGBetaFormula::PolakRibiere);
    }

    #[test]
    fn test_cg_new_hestenes_stiefel() {
        let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::HestenesStiefel);
        assert_eq!(optimizer.beta_formula, CGBetaFormula::HestenesStiefel);
    }

    #[test]
    fn test_cg_with_restart_interval() {
        let optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere)
            .with_restart_interval(50);
        assert_eq!(optimizer.restart_interval, 50);
    }

    #[test]
    fn test_cg_simple_quadratic_fr() {
        // Minimize f(x) = x^2, optimal at x = 0
        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::FletcherReeves);
        let x0 = Vector::from_slice(&[5.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!(result.solution[0].abs() < 1e-3);
    }

    #[test]
    fn test_cg_simple_quadratic_pr() {
        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
        let x0 = Vector::from_slice(&[5.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!(result.solution[0].abs() < 1e-3);
    }

    #[test]
    fn test_cg_simple_quadratic_hs() {
        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::HestenesStiefel);
        let x0 = Vector::from_slice(&[5.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!(result.solution[0].abs() < 1e-3);
    }

    #[test]
    fn test_cg_multidimensional_quadratic() {
        // Minimize f(x) = ||x - c||^2 where c = [1, 2, 3]
        let c = vec![1.0, 2.0, 3.0];
        let f = |x: &Vector<f32>| {
            let mut sum = 0.0;
            for i in 0..x.len() {
                sum += (x[i] - c[i]).powi(2);
            }
            sum
        };

        let grad = |x: &Vector<f32>| {
            let mut g = Vector::zeros(x.len());
            for i in 0..x.len() {
                g[i] = 2.0 * (x[i] - c[i]);
            }
            g
        };

        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
        let x0 = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        for i in 0..3 {
            assert!((result.solution[i] - c[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn test_cg_rosenbrock() {
        // Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        let f = |v: &Vector<f32>| {
            let x = v[0];
            let y = v[1];
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        };

        let grad = |v: &Vector<f32>| {
            let x = v[0];
            let y = v[1];
            let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
            let dy = 200.0 * (y - x * x);
            Vector::from_slice(&[dx, dy])
        };

        let mut optimizer = ConjugateGradient::new(500, 1e-4, CGBetaFormula::PolakRibiere);
        let x0 = Vector::from_slice(&[0.0, 0.0]);
        let result = optimizer.minimize(f, grad, x0);

        // Should converge close to (1, 1)
        assert!(
            result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::MaxIterations
        );
        assert!((result.solution[0] - 1.0).abs() < 0.1);
        assert!((result.solution[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_cg_sphere() {
        // Sphere function: f(x) = sum(x_i^2)
        let f = |x: &Vector<f32>| {
            let mut sum = 0.0;
            for i in 0..x.len() {
                sum += x[i] * x[i];
            }
            sum
        };

        let grad = |x: &Vector<f32>| {
            let mut g = Vector::zeros(x.len());
            for i in 0..x.len() {
                g[i] = 2.0 * x[i];
            }
            g
        };

        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
        let x0 = Vector::from_slice(&[5.0, -3.0, 2.0, -1.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        for i in 0..4 {
            assert!(result.solution[i].abs() < 1e-3);
        }
    }

    #[test]
    fn test_cg_compare_beta_formulas() {
        // Compare different beta formulas on same problem
        let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        let x0 = Vector::from_slice(&[3.0, 4.0]);

        let mut opt_fr = ConjugateGradient::new(100, 1e-5, CGBetaFormula::FletcherReeves);
        let result_fr = opt_fr.minimize(&f, &grad, x0.clone());

        let mut opt_pr = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
        let result_pr = opt_pr.minimize(&f, &grad, x0.clone());

        let mut opt_hs = ConjugateGradient::new(100, 1e-5, CGBetaFormula::HestenesStiefel);
        let result_hs = opt_hs.minimize(&f, &grad, x0);

        // All should converge to same solution
        assert_eq!(result_fr.status, ConvergenceStatus::Converged);
        assert_eq!(result_pr.status, ConvergenceStatus::Converged);
        assert_eq!(result_hs.status, ConvergenceStatus::Converged);

        assert!(result_fr.solution[0].abs() < 1e-3);
        assert!(result_pr.solution[0].abs() < 1e-3);
        assert!(result_hs.solution[0].abs() < 1e-3);
    }

    #[test]
    fn test_cg_with_periodic_restart() {
        let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere)
            .with_restart_interval(5);
        let x0 = Vector::from_slice(&[5.0, 5.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!(result.solution[0].abs() < 1e-3);
        assert!(result.solution[1].abs() < 1e-3);
    }

    #[test]
    fn test_cg_reset() {
        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
        let x0 = Vector::from_slice(&[5.0]);

        // First run
        optimizer.minimize(&f, &grad, x0.clone());
        assert!(optimizer.prev_direction.is_some());

        // Reset
        optimizer.reset();
        assert!(optimizer.prev_direction.is_none());
        assert!(optimizer.prev_gradient.is_none());
        assert_eq!(optimizer.iter_count, 0);

        // Second run should work
        let result = optimizer.minimize(&f, &grad, x0);
        assert_eq!(result.status, ConvergenceStatus::Converged);
    }

    #[test]
    fn test_cg_max_iterations() {
        // Use Rosenbrock with very few iterations
        let f = |v: &Vector<f32>| {
            let x = v[0];
            let y = v[1];
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        };

        let grad = |v: &Vector<f32>| {
            let x = v[0];
            let y = v[1];
            let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
            let dy = 200.0 * (y - x * x);
            Vector::from_slice(&[dx, dy])
        };

        let mut optimizer = ConjugateGradient::new(3, 1e-10, CGBetaFormula::PolakRibiere);
        let x0 = Vector::from_slice(&[0.0, 0.0]);
        let result = optimizer.minimize(f, grad, x0);

        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
        assert_eq!(result.iterations, 3);
    }

    #[test]
    #[should_panic(expected = "does not support stochastic")]
    fn test_cg_step_panics() {
        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
        let mut params = Vector::from_slice(&[1.0]);
        let grad = Vector::from_slice(&[0.1]);

        // Should panic - CG doesn't support step()
        optimizer.step(&mut params, &grad);
    }

    #[test]
    fn test_cg_numerical_error_detection() {
        // Function that produces NaN
        let f = |x: &Vector<f32>| {
            if x[0] < -100.0 {
                f32::NAN
            } else {
                x[0] * x[0]
            }
        };
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
        let x0 = Vector::from_slice(&[0.0]);
        let result = optimizer.minimize(f, grad, x0);

        // Should detect numerical error or converge normally
        assert!(
            result.status == ConvergenceStatus::Converged
                || result.status == ConvergenceStatus::NumericalError
                || result.status == ConvergenceStatus::MaxIterations
        );
    }

    #[test]
    fn test_cg_gradient_norm_tracking() {
        let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

        let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
        let x0 = Vector::from_slice(&[3.0, 4.0]);
        let result = optimizer.minimize(f, grad, x0);

        // Gradient norm at convergence should be small
        if result.status == ConvergenceStatus::Converged {
            assert!(result.gradient_norm < 1e-5);
        }
    }

    #[test]
    fn test_cg_vs_lbfgs_quadratic() {
        // Compare CG and L-BFGS on a quadratic problem
        let f = |x: &Vector<f32>| x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 4.0 * x[1], 6.0 * x[2]]);
        let x0 = Vector::from_slice(&[5.0, 3.0, 2.0]);

        let mut cg = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);
        let result_cg = cg.minimize(&f, &grad, x0.clone());

        let mut lbfgs = LBFGS::new(100, 1e-5, 10);
        let result_lbfgs = lbfgs.minimize(&f, &grad, x0);

        // Both should converge to same solution
        assert_eq!(result_cg.status, ConvergenceStatus::Converged);
        assert_eq!(result_lbfgs.status, ConvergenceStatus::Converged);

        for i in 0..3 {
            assert!((result_cg.solution[i] - result_lbfgs.solution[i]).abs() < 1e-3);
        }
    }
}
