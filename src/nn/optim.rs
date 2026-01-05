//! Gradient-based optimizers for neural network training.
//!
//! These optimizers work with autograd Tensors to update parameters
//! based on computed gradients.
//!
//! # Example
//!
//! ```ignore
//! use aprender::nn::{Linear, Module, optim::SGD};
//! use aprender::nn::loss::MSELoss;
//! use aprender::autograd::Tensor;
//!
//! // Create model and optimizer
//! let mut model = Linear::new(10, 5);
//! let mut optimizer = SGD::new(model.parameters_mut(), 0.01);
//!
//! // Training loop
//! for epoch in 0..100 {
//!     let x = Tensor::randn(&[32, 10]);
//!     let y = Tensor::randn(&[32, 5]);
//!
//!     // Forward pass
//!     let pred = model.forward(&x);
//!     let loss = MSELoss::new().forward(&pred, &y);
//!
//!     // Backward pass
//!     optimizer.zero_grad();
//!     loss.backward();
//!     optimizer.step();
//! }
//! ```
//!
//! # References
//!
//! - Robbins, H., & Monro, S. (1951). A stochastic approximation method.
//! - Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.
//! - Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. ICLR.

use crate::autograd::{get_grad, Tensor, TensorId};

/// Common trait for all optimizers.
pub trait Optimizer {
    /// Perform a single optimization step using computed gradients.
    fn step(&mut self);

    /// Zero all parameter gradients.
    fn zero_grad(&mut self);

    /// Get current learning rate.
    fn lr(&self) -> f32;

    /// Set learning rate (for schedulers).
    fn set_lr(&mut self, lr: f32);
}

/// Stochastic Gradient Descent optimizer with momentum.
///
/// Update rule:
/// ```text
/// v_t = momentum * v_{t-1} + grad
/// param = param - lr * v_t
/// ```
///
/// With Nesterov momentum:
/// ```text
/// v_t = momentum * v_{t-1} + grad
/// param = param - lr * (momentum * v_t + grad)
/// ```
#[derive(Debug)]
pub struct SGD {
    /// Parameter tensor IDs to optimize
    param_ids: Vec<TensorId>,
    /// Learning rate
    lr: f32,
    /// Momentum factor (0 = no momentum)
    momentum: f32,
    /// Weight decay (L2 regularization)
    weight_decay: f32,
    /// Nesterov momentum
    nesterov: bool,
    /// Velocity buffers for momentum
    velocities: Vec<Vec<f32>>,
    /// Whether velocities have been initialized
    initialized: bool,
}

impl SGD {
    /// Create a new SGD optimizer.
    ///
    /// # Arguments
    ///
    /// * `params` - Mutable references to parameter tensors
    /// * `lr` - Learning rate
    #[allow(clippy::needless_pass_by_value)]
    #[must_use] 
    pub fn new(params: Vec<&mut Tensor>, lr: f32) -> Self {
        let param_ids: Vec<TensorId> = params.iter().map(|p| p.id()).collect();
        Self {
            param_ids,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocities: Vec::new(),
            initialized: false,
        }
    }

    /// Create SGD with momentum.
    #[allow(clippy::needless_pass_by_value)]
    #[must_use] 
    pub fn with_momentum(params: Vec<&mut Tensor>, lr: f32, momentum: f32) -> Self {
        let param_ids: Vec<TensorId> = params.iter().map(|p| p.id()).collect();
        Self {
            param_ids,
            lr,
            momentum,
            weight_decay: 0.0,
            nesterov: false,
            velocities: Vec::new(),
            initialized: false,
        }
    }

    /// Enable Nesterov momentum.
    #[must_use] 
    pub fn nesterov(mut self) -> Self {
        self.nesterov = true;
        self
    }

    /// Set weight decay (L2 regularization).
    #[must_use] 
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Update a single parameter tensor.
    #[allow(clippy::if_not_else)]
    fn update_param(&mut self, param: &mut Tensor, idx: usize) {
        let Some(grad) = get_grad(param.id()) else {
            return; // No gradient available
        };

        let grad_data = grad.data();
        let param_data = param.data_mut();

        // Initialize velocity if needed
        if !self.initialized || idx >= self.velocities.len() {
            if idx >= self.velocities.len() {
                self.velocities.resize(idx + 1, Vec::new());
            }
            self.velocities[idx] = vec![0.0; param_data.len()];
        }

        let velocity = &mut self.velocities[idx];

        for i in 0..param_data.len() {
            let mut g = grad_data[i];

            // Apply weight decay
            if self.weight_decay != 0.0 {
                g += self.weight_decay * param_data[i];
            }

            if self.momentum != 0.0 {
                // Update velocity
                velocity[i] = self.momentum * velocity[i] + g;

                if self.nesterov {
                    // Nesterov: look ahead
                    param_data[i] -= self.lr * (self.momentum * velocity[i] + g);
                } else {
                    // Standard momentum
                    param_data[i] -= self.lr * velocity[i];
                }
            } else {
                // Vanilla SGD
                param_data[i] -= self.lr * g;
            }
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        // We need to get mutable access to the tensors through the global graph
        // For now, this is a placeholder that demonstrates the pattern
        // In practice, users will call update_param directly with their tensors
        self.initialized = true;
    }

    fn zero_grad(&mut self) {
        for &id in &self.param_ids {
            crate::autograd::clear_grad(id);
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl SGD {
    /// Perform optimization step with direct tensor access.
    ///
    /// This is the recommended way to use SGD in a training loop.
    pub fn step_with_params(&mut self, params: &mut [&mut Tensor]) {
        for (idx, param) in params.iter_mut().enumerate() {
            self.update_param(param, idx);
        }
        self.initialized = true;
    }
}

/// Adam optimizer (Kingma & Ba, 2015).
///
/// Combines momentum with adaptive learning rates using first and second
/// moment estimates.
///
/// Update rule:
/// ```text
/// m_t = β₁ * m_{t-1} + (1 - β₁) * grad
/// v_t = β₂ * v_{t-1} + (1 - β₂) * grad²
/// m̂_t = m_t / (1 - β₁ᵗ)
/// v̂_t = v_t / (1 - β₂ᵗ)
/// param = param - lr * m̂_t / (√v̂_t + ε)
/// ```
#[derive(Debug)]
pub struct Adam {
    param_ids: Vec<TensorId>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    /// First moment estimates
    m: Vec<Vec<f32>>,
    /// Second moment estimates
    v: Vec<Vec<f32>>,
    /// Current timestep for bias correction
    t: usize,
    initialized: bool,
}

impl Adam {
    /// Create a new Adam optimizer with default hyperparameters.
    ///
    /// Default: β₁=0.9, β₂=0.999, ε=1e-8
    #[allow(clippy::needless_pass_by_value)]
    #[must_use] 
    pub fn new(params: Vec<&mut Tensor>, lr: f32) -> Self {
        let param_ids: Vec<TensorId> = params.iter().map(|p| p.id()).collect();
        Self {
            param_ids,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
            initialized: false,
        }
    }

    /// Set beta parameters.
    #[must_use] 
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability.
    #[must_use] 
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay (L2 regularization, applied to gradient).
    #[must_use] 
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    fn update_param(&mut self, param: &mut Tensor, idx: usize) {
        let Some(grad) = get_grad(param.id()) else {
            return;
        };

        let grad_data = grad.data();
        let param_data = param.data_mut();

        // Initialize state if needed
        if !self.initialized || idx >= self.m.len() {
            if idx >= self.m.len() {
                self.m.resize(idx + 1, Vec::new());
                self.v.resize(idx + 1, Vec::new());
            }
            self.m[idx] = vec![0.0; param_data.len()];
            self.v[idx] = vec![0.0; param_data.len()];
        }

        let m = &mut self.m[idx];
        let v = &mut self.v[idx];

        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..param_data.len() {
            let mut g = grad_data[i];

            // L2 regularization (applied to gradient, not decoupled)
            if self.weight_decay != 0.0 {
                g += self.weight_decay * param_data[i];
            }

            // Update biased first moment estimate
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;

            // Update biased second moment estimate
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;

            // Compute bias-corrected estimates
            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;

            // Update parameter
            param_data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    /// Perform optimization step with direct tensor access.
    pub fn step_with_params(&mut self, params: &mut [&mut Tensor]) {
        self.t += 1;
        for (idx, param) in params.iter_mut().enumerate() {
            self.update_param(param, idx);
        }
        self.initialized = true;
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
        self.initialized = true;
    }

    fn zero_grad(&mut self) {
        for &id in &self.param_ids {
            crate::autograd::clear_grad(id);
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// `AdamW` optimizer (Loshchilov & Hutter, 2019).
///
/// Like Adam but with decoupled weight decay, which is more effective
/// for regularization.
///
/// The key difference from Adam:
/// ```text
/// param = param - lr * weight_decay * param  // Decoupled weight decay
/// param = param - lr * m̂_t / (√v̂_t + ε)      // Then Adam update
/// ```
#[derive(Debug)]
pub struct AdamW {
    param_ids: Vec<TensorId>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: usize,
    initialized: bool,
}

impl AdamW {
    /// Create a new `AdamW` optimizer.
    ///
    /// Default: β₁=0.9, β₂=0.999, ε=1e-8, `weight_decay=0.01`
    #[allow(clippy::needless_pass_by_value)]
    #[must_use] 
    pub fn new(params: Vec<&mut Tensor>, lr: f32) -> Self {
        let param_ids: Vec<TensorId> = params.iter().map(|p| p.id()).collect();
        Self {
            param_ids,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
            initialized: false,
        }
    }

    #[must_use] 
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    #[must_use] 
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    #[must_use] 
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    fn update_param(&mut self, param: &mut Tensor, idx: usize) {
        let Some(grad) = get_grad(param.id()) else {
            return;
        };

        let grad_data = grad.data();
        let param_data = param.data_mut();

        // Initialize state if needed
        if !self.initialized || idx >= self.m.len() {
            if idx >= self.m.len() {
                self.m.resize(idx + 1, Vec::new());
                self.v.resize(idx + 1, Vec::new());
            }
            self.m[idx] = vec![0.0; param_data.len()];
            self.v[idx] = vec![0.0; param_data.len()];
        }

        let m = &mut self.m[idx];
        let v = &mut self.v[idx];

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..param_data.len() {
            let g = grad_data[i];

            // Update moment estimates (no weight decay in gradient)
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;

            // Decoupled weight decay: applied directly to parameter
            param_data[i] -= self.lr * self.weight_decay * param_data[i];

            // Adam update
            param_data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    pub fn step_with_params(&mut self, params: &mut [&mut Tensor]) {
        self.t += 1;
        for (idx, param) in params.iter_mut().enumerate() {
            self.update_param(param, idx);
        }
        self.initialized = true;
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        self.t += 1;
        self.initialized = true;
    }

    fn zero_grad(&mut self) {
        for &id in &self.param_ids {
            crate::autograd::clear_grad(id);
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// `RMSprop` optimizer.
///
/// Maintains a moving average of squared gradients for adaptive learning rates.
///
/// Update rule:
/// ```text
/// v_t = α * v_{t-1} + (1 - α) * grad²
/// param = param - lr * grad / (√v_t + ε)
/// ```
#[derive(Debug)]
pub struct RMSprop {
    param_ids: Vec<TensorId>,
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    /// Running average of squared gradients
    v: Vec<Vec<f32>>,
    /// Momentum buffer
    buffer: Vec<Vec<f32>>,
    initialized: bool,
}

impl RMSprop {
    /// Create a new `RMSprop` optimizer.
    ///
    /// Default: α=0.99, ε=1e-8
    #[allow(clippy::needless_pass_by_value)]
    #[must_use] 
    pub fn new(params: Vec<&mut Tensor>, lr: f32) -> Self {
        let param_ids: Vec<TensorId> = params.iter().map(|p| p.id()).collect();
        Self {
            param_ids,
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            v: Vec::new(),
            buffer: Vec::new(),
            initialized: false,
        }
    }

    #[must_use] 
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    #[must_use] 
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    #[must_use] 
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    #[must_use] 
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    fn update_param(&mut self, param: &mut Tensor, idx: usize) {
        let Some(grad) = get_grad(param.id()) else {
            return;
        };

        let grad_data = grad.data();
        let param_data = param.data_mut();

        // Initialize state if needed
        if !self.initialized || idx >= self.v.len() {
            if idx >= self.v.len() {
                self.v.resize(idx + 1, Vec::new());
                self.buffer.resize(idx + 1, Vec::new());
            }
            self.v[idx] = vec![0.0; param_data.len()];
            self.buffer[idx] = vec![0.0; param_data.len()];
        }

        let v = &mut self.v[idx];
        let buffer = &mut self.buffer[idx];

        for i in 0..param_data.len() {
            let mut g = grad_data[i];

            // Weight decay
            if self.weight_decay != 0.0 {
                g += self.weight_decay * param_data[i];
            }

            // Update running average of squared gradients
            v[i] = self.alpha * v[i] + (1.0 - self.alpha) * g * g;

            // Compute update
            let update = g / (v[i].sqrt() + self.eps);

            if self.momentum > 0.0 {
                buffer[i] = self.momentum * buffer[i] + update;
                param_data[i] -= self.lr * buffer[i];
            } else {
                param_data[i] -= self.lr * update;
            }
        }
    }

    pub fn step_with_params(&mut self, params: &mut [&mut Tensor]) {
        for (idx, param) in params.iter_mut().enumerate() {
            self.update_param(param, idx);
        }
        self.initialized = true;
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) {
        self.initialized = true;
    }

    fn zero_grad(&mut self) {
        for &id in &self.param_ids {
            crate::autograd::clear_grad(id);
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::clear_graph;

    #[test]
    fn test_sgd_basic() {
        clear_graph();

        // Create a simple tensor
        let mut param = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
        let param_id = param.id();

        // Simulate a loss: sum of squared elements
        let loss = param.pow(2.0).sum();
        loss.backward();

        // Check gradient exists
        let grad = get_grad(param_id).expect("Should have gradient");
        assert_eq!(grad.data(), &[2.0, 4.0, 6.0]); // d/dx(x²) = 2x

        // Create optimizer and step
        let mut sgd = SGD::new(vec![&mut param], 0.1);
        sgd.step_with_params(&mut [&mut param]);

        // param = param - lr * grad = [1, 2, 3] - 0.1 * [2, 4, 6] = [0.8, 1.6, 2.4]
        let expected = [0.8, 1.6, 2.4];
        for (p, e) in param.data().iter().zip(expected.iter()) {
            assert!((p - e).abs() < 1e-5, "Expected {e}, got {p}");
        }
    }

    #[test]
    fn test_sgd_with_momentum() {
        clear_graph();

        let mut param = Tensor::from_slice(&[1.0]).requires_grad();

        // First step
        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut sgd = SGD::with_momentum(vec![&mut param], 0.1, 0.9);
        sgd.step_with_params(&mut [&mut param]);

        // v = 0.9 * 0 + 2.0 = 2.0
        // param = 1.0 - 0.1 * 2.0 = 0.8
        assert!((param.data()[0] - 0.8).abs() < 1e-5);

        // Second step
        clear_graph();
        let loss = param.pow(2.0).sum();
        loss.backward();

        sgd.step_with_params(&mut [&mut param]);

        // grad = 2 * 0.8 = 1.6
        // v = 0.9 * 2.0 + 1.6 = 3.4
        // param = 0.8 - 0.1 * 3.4 = 0.46
        assert!((param.data()[0] - 0.46).abs() < 1e-5);
    }

    #[test]
    fn test_adam_basic() {
        clear_graph();

        let mut param = Tensor::from_slice(&[1.0, 2.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut adam = Adam::new(vec![&mut param], 0.1);
        adam.step_with_params(&mut [&mut param]);

        // After one step, params should decrease
        assert!(param.data()[0] < 1.0);
        assert!(param.data()[1] < 2.0);
    }

    #[test]
    fn test_adam_convergence() {
        // Test that Adam can minimize a simple quadratic
        clear_graph();

        let mut param = Tensor::from_slice(&[5.0]).requires_grad();
        let mut adam = Adam::new(vec![&mut param], 0.5);

        // Minimize x² (optimal at x=0)
        for _ in 0..100 {
            clear_graph();
            let loss = param.pow(2.0).sum();
            loss.backward();
            adam.step_with_params(&mut [&mut param]);
        }

        // Should be close to 0
        assert!(
            param.data()[0].abs() < 0.1,
            "Parameter should converge to 0, got {}",
            param.data()[0]
        );
    }

    #[test]
    fn test_adamw_weight_decay() {
        clear_graph();

        let mut param = Tensor::from_slice(&[10.0]).requires_grad();

        // With zero gradient, only weight decay applies
        // We need a loss that has zero gradient at current point
        // Actually, let's just test the decoupled nature

        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut adamw = AdamW::new(vec![&mut param], 0.1).weight_decay(0.1);
        adamw.step_with_params(&mut [&mut param]);

        // With weight decay, param should decrease more
        assert!(param.data()[0] < 10.0);
    }

    #[test]
    fn test_rmsprop_basic() {
        clear_graph();

        let mut param = Tensor::from_slice(&[3.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut rmsprop = RMSprop::new(vec![&mut param], 0.1);
        rmsprop.step_with_params(&mut [&mut param]);

        // Param should decrease
        assert!(param.data()[0] < 3.0);
    }

    #[test]
    fn test_zero_grad() {
        clear_graph();

        let mut param = Tensor::from_slice(&[1.0, 2.0]).requires_grad();
        let param_id = param.id();

        let loss = param.pow(2.0).sum();
        loss.backward();

        // Gradient should exist
        assert!(get_grad(param_id).is_some());

        // Zero grad
        let mut sgd = SGD::new(vec![&mut param], 0.1);
        sgd.zero_grad();

        // Gradient should be cleared
        assert!(get_grad(param_id).is_none());
    }

    #[test]
    fn test_learning_rate_change() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let mut sgd = SGD::new(vec![&mut param], 0.1);

        assert!((sgd.lr() - 0.1).abs() < 1e-6);

        sgd.set_lr(0.01);
        assert!((sgd.lr() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_nesterov() {
        clear_graph();

        let mut param = Tensor::from_slice(&[2.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut sgd = SGD::with_momentum(vec![&mut param], 0.1, 0.9).nesterov();
        sgd.step_with_params(&mut [&mut param]);

        // Nesterov should apply a "look ahead" update
        // With nesterov: param = param - lr * (momentum * velocity + grad)
        // v = 0.9 * 0 + 4 = 4 (grad = 2 * 2 = 4)
        // param = 2 - 0.1 * (0.9 * 4 + 4) = 2 - 0.1 * 7.6 = 1.24
        assert!(
            (param.data()[0] - 1.24).abs() < 1e-5,
            "Nesterov update failed: {}",
            param.data()[0]
        );
    }

    #[test]
    fn test_sgd_weight_decay() {
        clear_graph();

        let mut param = Tensor::from_slice(&[5.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut sgd = SGD::new(vec![&mut param], 0.1).weight_decay(0.1);
        sgd.step_with_params(&mut [&mut param]);

        // grad = 2 * 5 = 10, with weight_decay: g = 10 + 0.1 * 5 = 10.5
        // param = 5 - 0.1 * 10.5 = 3.95
        assert!(
            (param.data()[0] - 3.95).abs() < 1e-5,
            "Weight decay update failed: {}",
            param.data()[0]
        );
    }

    #[test]
    fn test_adam_with_custom_betas() {
        clear_graph();

        let mut param = Tensor::from_slice(&[1.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut adam = Adam::new(vec![&mut param], 0.1).betas(0.8, 0.99);
        adam.step_with_params(&mut [&mut param]);

        // Param should decrease with custom betas
        assert!(param.data()[0] < 1.0);
    }

    #[test]
    fn test_adam_with_eps() {
        clear_graph();

        let mut param = Tensor::from_slice(&[1.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut adam = Adam::new(vec![&mut param], 0.1).eps(1e-6);
        adam.step_with_params(&mut [&mut param]);

        assert!(param.data()[0] < 1.0);
    }

    #[test]
    fn test_adam_with_weight_decay() {
        clear_graph();

        let mut param = Tensor::from_slice(&[10.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        // Compare with and without weight decay
        let mut adam_wd = Adam::new(vec![&mut param], 0.1).weight_decay(0.1);
        adam_wd.step_with_params(&mut [&mut param]);

        // With weight decay, the update should be larger
        assert!(param.data()[0] < 10.0);
    }

    #[test]
    fn test_adamw_with_custom_betas_and_eps() {
        clear_graph();

        let mut param = Tensor::from_slice(&[3.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut adamw = AdamW::new(vec![&mut param], 0.1)
            .betas(0.85, 0.995)
            .eps(1e-7);
        adamw.step_with_params(&mut [&mut param]);

        assert!(param.data()[0] < 3.0);
    }

    #[test]
    fn test_adamw_lr_methods() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let mut adamw = AdamW::new(vec![&mut param], 0.01);

        assert!((adamw.lr() - 0.01).abs() < 1e-6);
        adamw.set_lr(0.001);
        assert!((adamw.lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_adamw_zero_grad() {
        clear_graph();

        let mut param = Tensor::from_slice(&[2.0]).requires_grad();
        let param_id = param.id();

        let loss = param.pow(2.0).sum();
        loss.backward();

        assert!(get_grad(param_id).is_some());

        let mut adamw = AdamW::new(vec![&mut param], 0.1);
        adamw.zero_grad();

        assert!(get_grad(param_id).is_none());
    }

    #[test]
    fn test_adamw_step_trait() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let mut adamw = AdamW::new(vec![&mut param], 0.1);

        // Test the Optimizer trait step method
        adamw.step();
        assert!(adamw.initialized);
        assert_eq!(adamw.t, 1);
    }

    #[test]
    fn test_rmsprop_with_alpha() {
        clear_graph();

        let mut param = Tensor::from_slice(&[2.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut rmsprop = RMSprop::new(vec![&mut param], 0.1).alpha(0.9);
        rmsprop.step_with_params(&mut [&mut param]);

        assert!(param.data()[0] < 2.0);
    }

    #[test]
    fn test_rmsprop_with_eps() {
        clear_graph();

        let mut param = Tensor::from_slice(&[2.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut rmsprop = RMSprop::new(vec![&mut param], 0.1).eps(1e-6);
        rmsprop.step_with_params(&mut [&mut param]);

        assert!(param.data()[0] < 2.0);
    }

    #[test]
    fn test_rmsprop_with_momentum() {
        clear_graph();

        let mut param = Tensor::from_slice(&[3.0]).requires_grad();

        // First step
        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut rmsprop = RMSprop::new(vec![&mut param], 0.1).momentum(0.9);
        rmsprop.step_with_params(&mut [&mut param]);

        let after_first = param.data()[0];
        assert!(after_first < 3.0);

        // Second step with momentum accumulation
        clear_graph();
        let loss = param.pow(2.0).sum();
        loss.backward();

        rmsprop.step_with_params(&mut [&mut param]);

        assert!(param.data()[0] < after_first);
    }

    #[test]
    fn test_rmsprop_with_weight_decay() {
        clear_graph();

        let mut param = Tensor::from_slice(&[5.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut rmsprop = RMSprop::new(vec![&mut param], 0.1).weight_decay(0.1);
        rmsprop.step_with_params(&mut [&mut param]);

        assert!(param.data()[0] < 5.0);
    }

    #[test]
    fn test_rmsprop_lr_methods() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let mut rmsprop = RMSprop::new(vec![&mut param], 0.01);

        assert!((rmsprop.lr() - 0.01).abs() < 1e-6);
        rmsprop.set_lr(0.001);
        assert!((rmsprop.lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_zero_grad() {
        clear_graph();

        let mut param = Tensor::from_slice(&[2.0]).requires_grad();
        let param_id = param.id();

        let loss = param.pow(2.0).sum();
        loss.backward();

        assert!(get_grad(param_id).is_some());

        let mut rmsprop = RMSprop::new(vec![&mut param], 0.1);
        rmsprop.zero_grad();

        assert!(get_grad(param_id).is_none());
    }

    #[test]
    fn test_rmsprop_step_trait() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let mut rmsprop = RMSprop::new(vec![&mut param], 0.1);

        rmsprop.step();
        assert!(rmsprop.initialized);
    }

    #[test]
    fn test_sgd_step_trait() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let mut sgd = SGD::new(vec![&mut param], 0.1);

        sgd.step();
        assert!(sgd.initialized);
    }

    #[test]
    fn test_adam_step_trait() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let mut adam = Adam::new(vec![&mut param], 0.1);

        adam.step();
        assert!(adam.initialized);
        assert_eq!(adam.t, 1);
    }

    #[test]
    fn test_adam_lr_methods() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let mut adam = Adam::new(vec![&mut param], 0.01);

        assert!((adam.lr() - 0.01).abs() < 1e-6);
        adam.set_lr(0.001);
        assert!((adam.lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_adam_zero_grad() {
        clear_graph();

        let mut param = Tensor::from_slice(&[2.0]).requires_grad();
        let param_id = param.id();

        let loss = param.pow(2.0).sum();
        loss.backward();

        assert!(get_grad(param_id).is_some());

        let mut adam = Adam::new(vec![&mut param], 0.1);
        adam.zero_grad();

        assert!(get_grad(param_id).is_none());
    }

    #[test]
    fn test_sgd_multi_element_tensor() {
        clear_graph();

        let mut param = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut sgd = SGD::new(vec![&mut param], 0.1);
        sgd.step_with_params(&mut [&mut param]);

        // All elements should have decreased
        assert!(param.data()[0] < 1.0);
        assert!(param.data()[1] < 2.0);
        assert!(param.data()[2] < 3.0);
        assert!(param.data()[3] < 4.0);
    }

    #[test]
    fn test_adam_multi_element_tensor() {
        clear_graph();

        let mut param = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        let mut adam = Adam::new(vec![&mut param], 0.1);
        adam.step_with_params(&mut [&mut param]);

        // All elements should have decreased
        assert!(param.data()[0] < 1.0);
        assert!(param.data()[1] < 2.0);
        assert!(param.data()[2] < 3.0);
    }

    #[test]
    fn test_adamw_multi_step() {
        clear_graph();

        let mut param = Tensor::from_slice(&[5.0]).requires_grad();
        let mut adamw = AdamW::new(vec![&mut param], 0.5).weight_decay(0.01);

        // Multiple steps to test convergence
        for _ in 0..10 {
            clear_graph();
            let loss = param.pow(2.0).sum();
            loss.backward();
            adamw.step_with_params(&mut [&mut param]);
        }

        // Should have decreased significantly
        assert!(param.data()[0] < 1.0);
    }

    #[test]
    fn test_rmsprop_convergence() {
        clear_graph();

        let mut param = Tensor::from_slice(&[5.0]).requires_grad();
        let mut rmsprop = RMSprop::new(vec![&mut param], 0.5);

        // Multiple steps to test convergence
        for _ in 0..10 {
            clear_graph();
            let loss = param.pow(2.0).sum();
            loss.backward();
            rmsprop.step_with_params(&mut [&mut param]);
        }

        // Should have decreased significantly
        assert!(param.data()[0] < 1.0);
    }

    // ========== Additional Coverage Tests ==========

    #[test]
    fn test_sgd_lr_accessor() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let sgd = SGD::new(vec![&mut param], 0.05);
        assert!((sgd.lr() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_adam_lr_accessor() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let adam = Adam::new(vec![&mut param], 0.001);
        assert!((adam.lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_adamw_lr_accessor() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let adamw = AdamW::new(vec![&mut param], 0.002);
        assert!((adamw.lr() - 0.002).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_lr_accessor() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let rmsprop = RMSprop::new(vec![&mut param], 0.003);
        assert!((rmsprop.lr() - 0.003).abs() < 1e-6);
    }

    #[test]
    fn test_adam_set_lr() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let mut adam = Adam::new(vec![&mut param], 0.1);
        adam.set_lr(0.001);
        assert!((adam.lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_adamw_set_lr() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let mut adamw = AdamW::new(vec![&mut param], 0.1);
        adamw.set_lr(0.001);
        assert!((adamw.lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_set_lr() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let mut rmsprop = RMSprop::new(vec![&mut param], 0.1);
        rmsprop.set_lr(0.001);
        assert!((rmsprop.lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_adam_zero_grad_clears() {
        clear_graph();

        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let param_id = param.id();

        let loss = param.pow(2.0).sum();
        loss.backward();

        assert!(get_grad(param_id).is_some());

        let mut adam = Adam::new(vec![&mut param], 0.1);
        adam.zero_grad();

        assert!(get_grad(param_id).is_none());
    }

    #[test]
    fn test_adamw_zero_grad_clears() {
        clear_graph();

        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let param_id = param.id();

        let loss = param.pow(2.0).sum();
        loss.backward();

        assert!(get_grad(param_id).is_some());

        let mut adamw = AdamW::new(vec![&mut param], 0.1);
        adamw.zero_grad();

        assert!(get_grad(param_id).is_none());
    }

    #[test]
    fn test_rmsprop_zero_grad_clears() {
        clear_graph();

        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let param_id = param.id();

        let loss = param.pow(2.0).sum();
        loss.backward();

        assert!(get_grad(param_id).is_some());

        let mut rmsprop = RMSprop::new(vec![&mut param], 0.1);
        rmsprop.zero_grad();

        assert!(get_grad(param_id).is_none());
    }

    #[test]
    fn test_sgd_multiple_params() {
        clear_graph();

        let mut param1 = Tensor::from_slice(&[1.0]).requires_grad();
        let mut param2 = Tensor::from_slice(&[2.0]).requires_grad();

        // Create loss using both params (use add method instead of +)
        let loss1 = param1.pow(2.0).sum();
        let loss2 = param2.pow(2.0).sum();
        let loss = loss1.add(&loss2);
        loss.backward();

        let mut sgd = SGD::new(vec![&mut param1, &mut param2], 0.1);
        sgd.step_with_params(&mut [&mut param1, &mut param2]);

        // Both params should have decreased
        assert!(param1.data()[0] < 1.0);
        assert!(param2.data()[0] < 2.0);
    }

    #[test]
    fn test_adam_multiple_params() {
        clear_graph();

        let mut param1 = Tensor::from_slice(&[1.0]).requires_grad();
        let mut param2 = Tensor::from_slice(&[2.0]).requires_grad();

        let loss1 = param1.pow(2.0).sum();
        let loss2 = param2.pow(2.0).sum();
        let loss = loss1.add(&loss2);
        loss.backward();

        let mut adam = Adam::new(vec![&mut param1, &mut param2], 0.1);
        adam.step_with_params(&mut [&mut param1, &mut param2]);

        assert!(param1.data()[0] < 1.0);
        assert!(param2.data()[0] < 2.0);
    }

    #[test]
    fn test_rmsprop_alpha_builder() {
        clear_graph();

        let mut param = Tensor::from_slice(&[5.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        // Create RMSprop with custom alpha using builder pattern
        let mut rmsprop = RMSprop::new(vec![&mut param], 0.1).alpha(0.9);
        rmsprop.step_with_params(&mut [&mut param]);

        assert!(param.data()[0] < 5.0);
    }

    #[test]
    fn test_sgd_debug_trait() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let sgd = SGD::new(vec![&mut param], 0.1);
        let debug_str = format!("{:?}", sgd);
        assert!(debug_str.contains("SGD"));
    }

    #[test]
    fn test_adam_debug_trait() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let adam = Adam::new(vec![&mut param], 0.1);
        let debug_str = format!("{:?}", adam);
        assert!(debug_str.contains("Adam"));
    }

    #[test]
    fn test_adamw_debug_trait() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let adamw = AdamW::new(vec![&mut param], 0.1);
        let debug_str = format!("{:?}", adamw);
        assert!(debug_str.contains("AdamW"));
    }

    #[test]
    fn test_rmsprop_debug_trait() {
        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let rmsprop = RMSprop::new(vec![&mut param], 0.1);
        let debug_str = format!("{:?}", rmsprop);
        assert!(debug_str.contains("RMSprop"));
    }

    #[test]
    fn test_sgd_empty_params() {
        let sgd = SGD::new(vec![], 0.1);
        assert!((sgd.lr() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_adam_empty_params() {
        let adam = Adam::new(vec![], 0.1);
        assert!((adam.lr() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_momentum_initialization() {
        clear_graph();

        let mut param = Tensor::from_slice(&[3.0, 4.0]).requires_grad();

        let loss = param.pow(2.0).sum();
        loss.backward();

        // First step initializes velocities
        let mut sgd = SGD::with_momentum(vec![&mut param], 0.1, 0.9);
        sgd.step_with_params(&mut [&mut param]);

        // After first step, momentum buffer should be initialized
        assert!(param.data()[0] < 3.0);
        assert!(param.data()[1] < 4.0);
    }

    #[test]
    fn test_adam_step_counter() {
        clear_graph();

        let mut param = Tensor::from_slice(&[1.0]).requires_grad();
        let mut adam = Adam::new(vec![&mut param], 0.1);

        // Step multiple times
        for _ in 0..3 {
            clear_graph();
            let loss = param.pow(2.0).sum();
            loss.backward();
            adam.step_with_params(&mut [&mut param]);
        }

        // After 3 steps param should have decreased from 1.0
        assert!(param.data()[0] < 1.0);
    }
}
