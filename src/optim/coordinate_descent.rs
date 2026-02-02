//! Coordinate Descent optimizer for high-dimensional problems.

use super::{ConvergenceStatus, OptimizationResult, Optimizer};
use crate::primitives::Vector;

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
    // ==================== Getters (for testing) ====================

    /// Returns the maximum number of iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the convergence tolerance.
    #[must_use]
    pub fn tol(&self) -> f32 {
        self.tol
    }

    /// Returns whether random coordinate order is enabled.
    #[must_use]
    pub fn random_order(&self) -> bool {
        self.random_order
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cd_new() {
        let cd = CoordinateDescent::new(500, 1e-4);
        assert_eq!(cd.max_iter(), 500);
        assert!((cd.tol() - 1e-4).abs() < 1e-10);
        assert!(!cd.random_order());
    }

    #[test]
    fn test_cd_clone_debug() {
        let cd = CoordinateDescent::new(100, 1e-6);
        let cloned = cd.clone();
        assert_eq!(cd.max_iter(), cloned.max_iter());
        let debug_str = format!("{:?}", cd);
        assert!(debug_str.contains("CoordinateDescent"));
    }

    #[test]
    fn test_cd_with_random_order() {
        let cd = CoordinateDescent::new(100, 1e-6).with_random_order(true);
        assert!(cd.random_order());
    }

    #[test]
    fn test_cd_with_random_order_false() {
        let cd = CoordinateDescent::new(100, 1e-6).with_random_order(false);
        assert!(!cd.random_order());
    }

    #[test]
    fn test_cd_cyclic_convergence() {
        let c = vec![1.0, 2.0, 3.0];
        let update = move |x: &mut Vector<f32>, i: usize| {
            x[i] = c[i];
        };

        let mut cd = CoordinateDescent::new(100, 1e-6);
        let x0 = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = cd.minimize(update, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!((result.solution[0] - 1.0).abs() < 1e-5);
        assert!((result.solution[1] - 2.0).abs() < 1e-5);
        assert!((result.solution[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_cd_random_order_convergence() {
        let c = vec![4.0, 5.0, 6.0];
        let update = move |x: &mut Vector<f32>, i: usize| {
            x[i] = c[i];
        };

        let mut cd = CoordinateDescent::new(100, 1e-6).with_random_order(true);
        let x0 = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = cd.minimize(update, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!((result.solution[0] - 4.0).abs() < 1e-5);
        assert!((result.solution[1] - 5.0).abs() < 1e-5);
        assert!((result.solution[2] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_cd_max_iterations() {
        // Update function that never converges (keeps oscillating)
        let update = |x: &mut Vector<f32>, i: usize| {
            x[i] += 1.0; // Always changes, never settles
        };

        let mut cd = CoordinateDescent::new(3, 1e-10);
        let x0 = Vector::from_slice(&[0.0, 0.0]);
        let result = cd.minimize(update, x0);

        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
        assert_eq!(result.iterations, 3);
        assert!((result.gradient_norm - 0.0).abs() < 1e-10);
        assert!((result.objective_value - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cd_already_converged() {
        // Update function that makes no changes (already at optimum)
        let update = |_x: &mut Vector<f32>, _i: usize| {
            // No change - already at minimum
        };

        let mut cd = CoordinateDescent::new(100, 1e-6);
        let x0 = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = cd.minimize(update, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_cd_soft_threshold_update() {
        let lambda = 0.1;
        let update = move |x: &mut Vector<f32>, i: usize| {
            let v = x[i];
            x[i] = if v > lambda {
                v - lambda
            } else if v < -lambda {
                v + lambda
            } else {
                0.0
            };
        };

        let mut cd = CoordinateDescent::new(100, 1e-6);
        let x0 = Vector::from_slice(&[1.0, -0.5, 0.05]);
        let result = cd.minimize(update, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        // All should converge to 0 after enough soft-thresholding
        assert!(result.solution[2].abs() < 1e-5);
    }

    #[test]
    fn test_cd_reset() {
        let mut cd = CoordinateDescent::new(100, 1e-6);
        cd.reset(); // Stateless, should not panic
    }

    #[test]
    #[should_panic(expected = "does not support stochastic updates")]
    fn test_cd_step_panics() {
        let mut cd = CoordinateDescent::new(100, 1e-6);
        let mut params = Vector::from_slice(&[1.0]);
        let grad = Vector::from_slice(&[0.1]);
        cd.step(&mut params, &grad);
    }

    #[test]
    fn test_cd_1d() {
        let target = 7.0;
        let update = move |x: &mut Vector<f32>, _i: usize| {
            x[0] = target;
        };

        let mut cd = CoordinateDescent::new(100, 1e-6);
        let x0 = Vector::from_slice(&[0.0]);
        let result = cd.minimize(update, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!((result.solution[0] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_cd_convergence_fields() {
        let update = |_x: &mut Vector<f32>, _i: usize| {};

        let mut cd = CoordinateDescent::new(100, 1e-6);
        let x0 = Vector::from_slice(&[1.0]);
        let result = cd.minimize(update, x0);

        assert!((result.constraint_violation - 0.0).abs() < 1e-10);
        let _ = result.elapsed_time.as_nanos();
    }

    #[test]
    fn test_cd_random_order_with_single_coordinate() {
        // Edge case: single coordinate with random order
        let update = |x: &mut Vector<f32>, _i: usize| {
            x[0] = 42.0;
        };

        let mut cd = CoordinateDescent::new(100, 1e-6).with_random_order(true);
        let x0 = Vector::from_slice(&[0.0]);
        let result = cd.minimize(update, x0);

        assert_eq!(result.status, ConvergenceStatus::Converged);
        assert!((result.solution[0] - 42.0).abs() < 1e-5);
    }

    #[test]
    fn test_cd_getters() {
        let cd = CoordinateDescent::new(200, 1e-3).with_random_order(true);
        assert_eq!(cd.max_iter(), 200);
        assert!((cd.tol() - 1e-3).abs() < 1e-10);
        assert!(cd.random_order());
    }
}
