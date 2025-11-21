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
    fn test_adam_optimizer_trait() {
        let mut optimizer: Box<dyn Optimizer> = Box::new(Adam::new(0.001));
        let mut params = Vector::from_slice(&[1.0]);
        let gradients = Vector::from_slice(&[1.0]);

        optimizer.step(&mut params, &gradients);
        assert!(params[0] < 1.0);

        optimizer.reset();
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
}
