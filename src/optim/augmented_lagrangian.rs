//! Augmented Lagrangian method for constrained optimization.

use super::{ConvergenceStatus, OptimizationResult, Optimizer};
use crate::primitives::Vector;

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
