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
    pub(crate) initialized: bool,
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
        // SGD step marks the optimizer as initialized.
        // Parameter updates happen via update_param() with individual tensors.
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
    pub(crate) t: usize,
    pub(crate) initialized: bool,
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
    pub(crate) param_ids: Vec<TensorId>,
    pub(crate) lr: f32,
    pub(crate) beta1: f32,
    pub(crate) beta2: f32,
    pub(crate) eps: f32,
    pub(crate) weight_decay: f32,
    pub(crate) m: Vec<Vec<f32>>,
    pub(crate) v: Vec<Vec<f32>>,
    pub(crate) t: usize,
    pub(crate) initialized: bool,
}

#[path = "mod_part_02.rs"]
mod mod_part_02;
pub use mod_part_02::*;
