use super::OptimizationResult;
use crate::primitives::Vector;

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
