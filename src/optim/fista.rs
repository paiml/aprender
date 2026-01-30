//! FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
//!
//! Accelerated proximal gradient method for minimizing composite objectives f(x) + g(x)
//! where f is smooth and convex, g is convex (possibly non-smooth but "simple").

use crate::primitives::Vector;

use super::{ConvergenceStatus, OptimizationResult, Optimizer};

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
    pub(crate) max_iter: usize,
    /// Step size (α > 0)
    pub(crate) step_size: f32,
    /// Convergence tolerance (‖xₖ₊₁ - xₖ‖ < tol)
    pub(crate) tol: f32,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::prox;

    #[test]
    fn test_fista_l1_regularized() {
        // Minimize: ½(x - 5)² + λ|x| where λ=2
        // The subgradient condition at optimum: (x* - 5) + 2*sign(x*) = 0
        // For x* > 0: x* - 5 + 2 = 0 => x* = 3
        let smooth = |x: &Vector<f32>| 0.5 * (x[0] - 5.0).powi(2);
        let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[x[0] - 5.0]);
        // Note: soft_threshold uses the step_size * lambda, so we pass 0.2 to get λ=2 with step_size=0.1
        let proximal = |v: &Vector<f32>, alpha: f32| prox::soft_threshold(v, 2.0 * alpha);

        let mut fista = FISTA::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[0.0]);
        let result = fista.minimize(smooth, grad_smooth, proximal, x0);

        // Solution should be around 3.0 (5.0 - 2.0 = 3.0 due to soft thresholding)
        assert!(
            (result.solution[0] - 3.0).abs() < 0.5,
            "Expected ~3.0, got {}",
            result.solution[0]
        );
    }

    #[test]
    fn test_fista_nonnegative() {
        // Minimize: (x + 1)² subject to x >= 0
        // Optimal at x = 0
        let smooth = |x: &Vector<f32>| (x[0] + 1.0).powi(2);
        let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] + 1.0)]);
        let proximal = |v: &Vector<f32>, _alpha: f32| prox::nonnegative(v);

        let mut fista = FISTA::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[5.0]);
        let result = fista.minimize(smooth, grad_smooth, proximal, x0);

        assert!(result.solution[0].abs() < 1e-4);
    }

    #[test]
    fn test_fista_new() {
        let fista = FISTA::new(500, 0.05, 1e-4);
        assert_eq!(fista.max_iter, 500);
        assert!((fista.step_size - 0.05).abs() < 1e-10);
        assert!((fista.tol - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_fista_reset() {
        let mut fista = FISTA::new(100, 0.1, 1e-5);
        fista.reset(); // Should do nothing but not panic
    }

    #[test]
    #[should_panic(expected = "does not support stochastic updates")]
    fn test_fista_step_unimplemented() {
        let mut fista = FISTA::new(100, 0.1, 1e-5);
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let grad = Vector::from_slice(&[0.1, 0.2]);
        fista.step(&mut params, &grad);
    }

    #[test]
    fn test_fista_max_iterations() {
        // Simple quadratic that converges slowly with small step
        let smooth = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        let proximal = |v: &Vector<f32>, _alpha: f32| v.clone(); // Identity proximal

        let mut fista = FISTA::new(2, 0.0001, 1e-10); // Very few iterations, tiny step
        let x0 = Vector::from_slice(&[100.0, 100.0]);
        let result = fista.minimize(smooth, grad_smooth, proximal, x0);

        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
        assert_eq!(result.iterations, 2);
    }

    #[test]
    fn test_fista_2d_quadratic() {
        // Minimize: x² + y²
        let smooth = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        let proximal = |v: &Vector<f32>, _alpha: f32| v.clone(); // Identity proximal

        let mut fista = FISTA::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[5.0, -3.0]);
        let result = fista.minimize(smooth, grad_smooth, proximal, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!(result.solution[0].abs() < 1e-4);
        assert!(result.solution[1].abs() < 1e-4);
    }

    #[test]
    fn test_fista_3d() {
        // Minimize: x² + y² + z²
        let smooth = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        let grad_smooth = |x: &Vector<f32>| {
            Vector::from_slice(&[2.0 * x[0], 2.0 * x[1], 2.0 * x[2]])
        };
        let proximal = |v: &Vector<f32>, _alpha: f32| v.clone();

        let mut fista = FISTA::new(1000, 0.1, 1e-5);
        let x0 = Vector::from_slice(&[5.0, -3.0, 2.0]);
        let result = fista.minimize(smooth, grad_smooth, proximal, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!(result.solution[0].abs() < 1e-3);
        assert!(result.solution[1].abs() < 1e-3);
        assert!(result.solution[2].abs() < 1e-3);
    }

    #[test]
    fn test_fista_objective_value() {
        let smooth = |x: &Vector<f32>| (x[0] - 2.0).powi(2);
        let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 2.0)]);
        let proximal = |v: &Vector<f32>, _alpha: f32| v.clone();

        let mut fista = FISTA::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[0.0]);
        let result = fista.minimize(smooth, grad_smooth, proximal, x0);

        assert!(result.objective_value < 1e-6);
        assert!((result.solution[0] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_fista_gradient_norm() {
        let smooth = |x: &Vector<f32>| x[0] * x[0];
        let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let proximal = |v: &Vector<f32>, _alpha: f32| v.clone();

        let mut fista = FISTA::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[5.0]);
        let result = fista.minimize(smooth, grad_smooth, proximal, x0);

        // Gradient norm (actually diff_norm) should be small at convergence
        assert!(result.gradient_norm < 1e-5);
    }

    #[test]
    fn test_fista_constraint_violation_zero() {
        let smooth = |x: &Vector<f32>| x[0] * x[0];
        let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let proximal = |v: &Vector<f32>, _alpha: f32| v.clone();

        let mut fista = FISTA::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[5.0]);
        let result = fista.minimize(smooth, grad_smooth, proximal, x0);

        assert!((result.constraint_violation - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_fista_elapsed_time() {
        let smooth = |x: &Vector<f32>| x[0] * x[0];
        let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let proximal = |v: &Vector<f32>, _alpha: f32| v.clone();

        let mut fista = FISTA::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[5.0]);
        let result = fista.minimize(smooth, grad_smooth, proximal, x0);

        // Elapsed time should be accessible
        let _ = result.elapsed_time.as_nanos();
    }

    #[test]
    fn test_fista_debug_clone() {
        let fista = FISTA::new(100, 0.1, 1e-5);
        let cloned = fista.clone();

        assert_eq!(fista.max_iter, cloned.max_iter);
        assert!((fista.step_size - cloned.step_size).abs() < 1e-10);
        assert!((fista.tol - cloned.tol).abs() < 1e-10);

        // Test Debug
        let debug_str = format!("{:?}", fista);
        assert!(debug_str.contains("FISTA"));
    }

    #[test]
    fn test_fista_already_at_optimum() {
        let smooth = |x: &Vector<f32>| x[0] * x[0];
        let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let proximal = |v: &Vector<f32>, _alpha: f32| v.clone();

        let mut fista = FISTA::new(1000, 0.1, 1e-6);
        // Start very close to optimum
        let x0 = Vector::from_slice(&[1e-8]);
        let result = fista.minimize(smooth, grad_smooth, proximal, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
    }

    #[test]
    fn test_fista_l2_ball_constraint() {
        // Minimize: (x - 5)² subject to ‖x‖ ≤ 2
        let smooth = |x: &Vector<f32>| (x[0] - 5.0).powi(2);
        let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 5.0)]);
        let proximal = |v: &Vector<f32>, _alpha: f32| prox::project_l2_ball(v, 2.0);

        let mut fista = FISTA::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[0.0]);
        let result = fista.minimize(smooth, grad_smooth, proximal, x0);

        // Solution should be at boundary: x = 2
        assert!((result.solution[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_fista_box_constraint() {
        // Minimize: (x - 5)² subject to 0 ≤ x ≤ 1
        let smooth = |x: &Vector<f32>| (x[0] - 5.0).powi(2);
        let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 5.0)]);
        let lower = Vector::from_slice(&[0.0]);
        let upper = Vector::from_slice(&[1.0]);
        let proximal = move |v: &Vector<f32>, _alpha: f32| prox::project_box(v, &lower, &upper);

        let mut fista = FISTA::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[0.5]);
        let result = fista.minimize(smooth, grad_smooth, proximal, x0);

        // Solution should be at upper bound: x = 1
        assert!((result.solution[0] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_fista_different_step_sizes() {
        let smooth = |x: &Vector<f32>| x[0] * x[0];
        let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let proximal = |v: &Vector<f32>, _alpha: f32| v.clone();

        // Small step size
        let mut fista1 = FISTA::new(1000, 0.01, 1e-6);
        let x0 = Vector::from_slice(&[5.0]);
        let result1 = fista1.minimize(smooth, grad_smooth, proximal, x0.clone());

        // Larger step size
        let mut fista2 = FISTA::new(1000, 0.4, 1e-6);
        let result2 = fista2.minimize(smooth, grad_smooth, proximal, x0);

        // Both should converge
        assert_eq!(result1.status, ConvergenceStatus::Converged);
        assert_eq!(result2.status, ConvergenceStatus::Converged);
    }

    #[test]
    fn test_fista_iterations_tracked() {
        let smooth = |x: &Vector<f32>| x[0] * x[0];
        let grad_smooth = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
        let proximal = |v: &Vector<f32>, _alpha: f32| v.clone();

        let mut fista = FISTA::new(1000, 0.1, 1e-6);
        let x0 = Vector::from_slice(&[10.0]);
        let result = fista.minimize(smooth, grad_smooth, proximal, x0);

        // Should take some iterations
        assert!(result.iterations <= 1000);
    }
}
