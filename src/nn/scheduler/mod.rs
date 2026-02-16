//! Learning rate schedulers for training neural networks.
//!
//! Schedulers adjust the learning rate during training to improve convergence
//! and final model quality.
//!
//! # Example
//!
//! ```ignore
//! use aprender::nn::optim::{Adam, Optimizer};
//! use aprender::nn::scheduler::{StepLR, LRScheduler};
//!
//! let mut optimizer = Adam::new(params, 0.1);
//! let mut scheduler = StepLR::new(10, 0.1);  // Decay by 0.1 every 10 epochs
//!
//! for epoch in 0..100 {
//!     // Training loop...
//!
//!     // Update learning rate at end of epoch
//!     scheduler.step(&mut optimizer);
//! }
//! ```
//!
//! # References
//!
//! - Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent
//!   with warm restarts. ICLR.
//! - Goyal, P., et al. (2017). Accurate, large minibatch SGD: Training
//!   `ImageNet` in 1 hour. arXiv.

use super::optim::Optimizer;

/// Common trait for learning rate schedulers.
pub trait LRScheduler {
    /// Update the optimizer's learning rate.
    fn step<O: Optimizer>(&mut self, optimizer: &mut O);

    /// Get the current learning rate.
    fn get_lr(&self) -> f32;

    /// Get the current epoch/step count.
    fn last_epoch(&self) -> usize;
}

/// Step decay scheduler.
///
/// Decays learning rate by `gamma` every `step_size` epochs.
///
/// ```text
/// lr = initial_lr * gamma^(epoch // step_size)
/// ```
#[derive(Debug, Clone)]
pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
    current_lr: f32,
    last_epoch: usize,
}

impl StepLR {
    /// Create a new `StepLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `step_size` - Number of epochs between LR decays
    /// * `gamma` - Multiplicative factor of LR decay (e.g., 0.1)
    #[must_use]
    pub fn new(step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr: 0.0, // Will be set on first step
            step_size,
            gamma,
            current_lr: 0.0,
            last_epoch: 0,
        }
    }

    /// Create with initial learning rate already known.
    #[must_use]
    pub fn with_lr(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_lr: initial_lr,
            last_epoch: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn step<O: Optimizer>(&mut self, optimizer: &mut O) {
        // Initialize on first step
        if self.last_epoch == 0 && self.initial_lr == 0.0 {
            self.initial_lr = optimizer.lr();
            self.current_lr = self.initial_lr;
        }

        self.last_epoch += 1;

        // Decay at step boundaries
        if self.last_epoch % self.step_size == 0 {
            self.current_lr *= self.gamma;
            optimizer.set_lr(self.current_lr);
        }
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// Exponential decay scheduler.
///
/// Decays learning rate by `gamma` every epoch.
///
/// ```text
/// lr = initial_lr * gamma^epoch
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    initial_lr: f32,
    gamma: f32,
    current_lr: f32,
    last_epoch: usize,
}

impl ExponentialLR {
    /// Create a new `ExponentialLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Multiplicative factor (e.g., 0.99)
    #[must_use]
    pub fn new(gamma: f32) -> Self {
        Self {
            initial_lr: 0.0,
            gamma,
            current_lr: 0.0,
            last_epoch: 0,
        }
    }

    #[must_use]
    pub fn with_lr(initial_lr: f32, gamma: f32) -> Self {
        Self {
            initial_lr,
            gamma,
            current_lr: initial_lr,
            last_epoch: 0,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step<O: Optimizer>(&mut self, optimizer: &mut O) {
        if self.last_epoch == 0 && self.initial_lr == 0.0 {
            self.initial_lr = optimizer.lr();
            self.current_lr = self.initial_lr;
        }

        self.last_epoch += 1;
        self.current_lr *= self.gamma;
        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// Cosine annealing scheduler (Loshchilov & Hutter, 2017).
///
/// Anneals learning rate following a cosine curve from initial to minimum.
///
/// ```text
/// lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * epoch / T_max))
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    initial_lr: f32,
    min_lr: f32,
    t_max: usize,
    current_lr: f32,
    last_epoch: usize,
}

impl CosineAnnealingLR {
    /// Create a new `CosineAnnealingLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `t_max` - Maximum number of epochs
    /// * `min_lr` - Minimum learning rate (default: 0)
    #[must_use]
    pub fn new(t_max: usize) -> Self {
        Self {
            initial_lr: 0.0,
            min_lr: 0.0,
            t_max,
            current_lr: 0.0,
            last_epoch: 0,
        }
    }

    #[must_use]
    pub fn with_min_lr(t_max: usize, min_lr: f32) -> Self {
        Self {
            initial_lr: 0.0,
            min_lr,
            t_max,
            current_lr: 0.0,
            last_epoch: 0,
        }
    }

    #[must_use]
    pub fn with_lr(initial_lr: f32, t_max: usize, min_lr: f32) -> Self {
        Self {
            initial_lr,
            min_lr,
            t_max,
            current_lr: initial_lr,
            last_epoch: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step<O: Optimizer>(&mut self, optimizer: &mut O) {
        if self.last_epoch == 0 && self.initial_lr == 0.0 {
            self.initial_lr = optimizer.lr();
            self.current_lr = self.initial_lr;
        }

        self.last_epoch += 1;

        // Cosine annealing formula
        let progress = self.last_epoch as f32 / self.t_max as f32;
        let cosine = (std::f32::consts::PI * progress).cos();
        self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + cosine);

        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// Linear warmup scheduler.
///
/// Linearly increases learning rate from 0 to `initial_lr` over `warmup_steps`.
///
/// ```text
/// if epoch < warmup_steps:
///     lr = initial_lr * epoch / warmup_steps
/// else:
///     lr = initial_lr
/// ```
#[derive(Debug, Clone)]
pub struct LinearWarmup {
    initial_lr: f32,
    warmup_steps: usize,
    current_lr: f32,
    last_epoch: usize,
}

impl LinearWarmup {
    /// Create a new `LinearWarmup` scheduler.
    ///
    /// # Arguments
    ///
    /// * `warmup_steps` - Number of warmup epochs
    #[must_use]
    pub fn new(warmup_steps: usize) -> Self {
        Self {
            initial_lr: 0.0,
            warmup_steps,
            current_lr: 0.0,
            last_epoch: 0,
        }
    }

    #[must_use]
    pub fn with_lr(initial_lr: f32, warmup_steps: usize) -> Self {
        Self {
            initial_lr,
            warmup_steps,
            current_lr: 0.0,
            last_epoch: 0,
        }
    }
}

impl LRScheduler for LinearWarmup {
    fn step<O: Optimizer>(&mut self, optimizer: &mut O) {
        if self.last_epoch == 0 && self.initial_lr == 0.0 {
            self.initial_lr = optimizer.lr();
        }

        self.last_epoch += 1;

        if self.last_epoch <= self.warmup_steps {
            // Linear warmup
            self.current_lr = self.initial_lr * (self.last_epoch as f32 / self.warmup_steps as f32);
        } else {
            self.current_lr = self.initial_lr;
        }

        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// Warmup + Cosine decay scheduler.
///
/// Combines linear warmup with cosine annealing, commonly used in modern
/// transformer training.
///
/// ```text
/// if epoch < warmup_steps:
///     lr = initial_lr * epoch / warmup_steps
/// else:
///     lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * (epoch - warmup) / (total - warmup)))
/// ```
#[derive(Debug, Clone)]
pub struct WarmupCosineScheduler {
    initial_lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_lr: f32,
    last_epoch: usize,
}

impl WarmupCosineScheduler {
    /// Create a new `WarmupCosineScheduler`.
    ///
    /// # Arguments
    ///
    /// * `warmup_steps` - Number of warmup epochs
    /// * `total_steps` - Total number of training epochs
    #[must_use]
    pub fn new(warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            initial_lr: 0.0,
            min_lr: 0.0,
            warmup_steps,
            total_steps,
            current_lr: 0.0,
            last_epoch: 0,
        }
    }

    #[must_use]
    pub fn with_min_lr(warmup_steps: usize, total_steps: usize, min_lr: f32) -> Self {
        Self {
            initial_lr: 0.0,
            min_lr,
            warmup_steps,
            total_steps,
            current_lr: 0.0,
            last_epoch: 0,
        }
    }
}

impl LRScheduler for WarmupCosineScheduler {
    fn step<O: Optimizer>(&mut self, optimizer: &mut O) {
        if self.last_epoch == 0 && self.initial_lr == 0.0 {
            self.initial_lr = optimizer.lr();
        }

        self.last_epoch += 1;

        if self.last_epoch <= self.warmup_steps {
            // Linear warmup
            self.current_lr = self.initial_lr * (self.last_epoch as f32 / self.warmup_steps as f32);
        } else {
            // Cosine decay
            let decay_steps = self.total_steps - self.warmup_steps;
            let decay_epoch = self.last_epoch - self.warmup_steps;
            let progress = decay_epoch as f32 / decay_steps as f32;
            let cosine = (std::f32::consts::PI * progress).cos();
            self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + cosine);
        }

        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn last_epoch(&self) -> usize {
        self.last_epoch
    }
}

/// Reduce LR on plateau scheduler.
///
/// Reduces learning rate when a metric has stopped improving.
#[derive(Debug, Clone)]
pub struct ReduceLROnPlateau {
    factor: f32,
    patience: usize,
    min_lr: f32,
    threshold: f32,
    current_lr: f32,
    best_metric: f32,
    num_bad_epochs: usize,
    last_epoch: usize,
    mode: PlateauMode,
}

/// Mode for plateau detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateauMode {
    /// Lower metric is better (e.g., loss)
    Min,
    /// Higher metric is better (e.g., accuracy)
    Max,
}

mod mod_part_02;
