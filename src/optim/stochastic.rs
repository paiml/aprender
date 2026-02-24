//! Stochastic optimizers for mini-batch gradient descent.
//!
//! These optimizers update parameters incrementally using gradients from mini-batches,
//! making them suitable for large-scale machine learning training.
//!
//! # Available Optimizers
//!
//! - [`SGD`] - Stochastic Gradient Descent with optional momentum
//! - [`Adam`] - Adaptive Moment Estimation (adaptive learning rates)

use serde::{Deserialize, Serialize};

use crate::primitives::Vector;

use super::Optimizer;

/// Stochastic Gradient Descent (SGD) optimizer with optional momentum.
///
/// SGD is the foundation of deep learning optimization. With momentum, it
/// accumulates velocity to accelerate through flat regions and dampen oscillations.
///
/// # Update Rule
///
/// Without momentum: `θ = θ - η * ∇f(θ)`
///
/// With momentum:
/// ```text
/// v = γ * v + η * ∇f(θ)
/// θ = θ - v
/// ```
///
/// # Parameters
///
/// - **learning_rate** (η): Step size for parameter updates
/// - **momentum** (γ): Velocity decay rate (0.0 = no momentum, typical: 0.9)
///
/// # Example
///
/// ```
/// use aprender::optim::SGD;
/// use aprender::primitives::Vector;
///
/// // SGD without momentum
/// let mut optimizer = SGD::new(0.01);
///
/// // SGD with momentum
/// let mut optimizer_momentum = SGD::new(0.01).with_momentum(0.9);
///
/// let mut params = Vector::from_slice(&[1.0, 2.0, 3.0]);
/// let gradients = Vector::from_slice(&[0.1, 0.2, 0.3]);
///
/// // Update parameters
/// optimizer.step(&mut params, &gradients);
///
/// // Parameters are updated: params = params - lr * gradients
/// assert!((params[0] - 0.999).abs() < 1e-6);
/// ```
///
/// # Momentum Behavior
///
/// ```
/// use aprender::optim::SGD;
/// use aprender::primitives::Vector;
///
/// let mut optimizer = SGD::new(0.1).with_momentum(0.9);
/// let mut params = Vector::from_slice(&[0.0]);
/// let gradients = Vector::from_slice(&[1.0]);
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

impl Optimizer for SGD {
    fn step(&mut self, params: &mut Vector<f32>, gradients: &Vector<f32>) {
        self.step(params, gradients);
    }

    fn reset(&mut self) {
        self.reset();
    }
}

/// Adam (Adaptive Moment Estimation) optimizer.
///
/// Adam combines the benefits of `AdaGrad` and `RMSprop` by computing adaptive learning
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
/// - `m_t` is the first moment (mean) estimate
/// - `v_t` is the second moment (variance) estimate
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

#[path = "parameter.rs"]
mod parameter;

#[cfg(test)]
#[path = "sgd_tests.rs"]
mod sgd_tests;

#[cfg(test)]
#[path = "tests_sgd_contract.rs"]
mod tests_sgd_contract;
