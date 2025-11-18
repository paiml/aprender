//! Optimization algorithms for gradient-based learning.
//!
//! # Usage
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

use serde::{Deserialize, Serialize};

use crate::primitives::Vector;

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
            if self.velocity.is_none() || self.velocity.as_ref().unwrap().len() != n {
                self.velocity = Some(vec![0.0; n]);
            }

            let velocity = self.velocity.as_mut().unwrap();

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

/// Trait for optimizers that update parameters based on gradients.
pub trait Optimizer {
    /// Updates parameters using the provided gradients.
    fn step(&mut self, params: &mut Vector<f32>, gradients: &Vector<f32>);

    /// Resets the optimizer state.
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
    fn test_optimizer_trait() {
        let mut optimizer: Box<dyn Optimizer> = Box::new(SGD::new(0.1));
        let mut params = Vector::from_slice(&[1.0]);
        let gradients = Vector::from_slice(&[1.0]);

        optimizer.step(&mut params, &gradients);
        assert!((params[0] - 0.9).abs() < 1e-6);

        optimizer.reset();
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
}
