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

impl ReduceLROnPlateau {
    /// Create a new `ReduceLROnPlateau` scheduler.
    ///
    /// # Arguments
    ///
    /// * `mode` - Whether to minimize or maximize the metric
    /// * `factor` - Factor to reduce LR by (e.g., 0.1)
    /// * `patience` - Number of epochs with no improvement before reducing
    #[must_use]
    pub fn new(mode: PlateauMode, factor: f32, patience: usize) -> Self {
        let best_metric = match mode {
            PlateauMode::Min => f32::INFINITY,
            PlateauMode::Max => f32::NEG_INFINITY,
        };

        Self {
            factor,
            patience,
            min_lr: 1e-8,
            threshold: 1e-4,
            current_lr: 0.0,
            best_metric,
            num_bad_epochs: 0,
            last_epoch: 0,
            mode,
        }
    }

    /// Set minimum learning rate.
    #[must_use]
    pub fn min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Set threshold for measuring improvement.
    #[must_use]
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Update scheduler with current metric value.
    pub fn step_with_metric<O: Optimizer>(&mut self, optimizer: &mut O, metric: f32) {
        if self.last_epoch == 0 && self.current_lr == 0.0 {
            self.current_lr = optimizer.lr();
        }

        self.last_epoch += 1;

        // Check if metric improved
        let is_better = match self.mode {
            PlateauMode::Min => metric < self.best_metric - self.threshold,
            PlateauMode::Max => metric > self.best_metric + self.threshold,
        };

        if is_better {
            self.best_metric = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        // Reduce LR if patience exceeded
        if self.num_bad_epochs >= self.patience {
            let new_lr = (self.current_lr * self.factor).max(self.min_lr);
            if new_lr < self.current_lr {
                self.current_lr = new_lr;
                optimizer.set_lr(self.current_lr);
                self.num_bad_epochs = 0;
            }
        }
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn step<O: Optimizer>(&mut self, _optimizer: &mut O) {
        // This scheduler needs a metric, use step_with_metric instead
        self.last_epoch += 1;
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn last_epoch(&self) -> usize {
        self.last_epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock optimizer for testing
    struct MockOptimizer {
        lr: f32,
    }

    impl MockOptimizer {
        fn new(lr: f32) -> Self {
            Self { lr }
        }
    }

    impl Optimizer for MockOptimizer {
        fn step(&mut self) {}
        fn zero_grad(&mut self) {}
        fn lr(&self) -> f32 {
            self.lr
        }
        fn set_lr(&mut self, lr: f32) {
            self.lr = lr;
        }
    }

    #[test]
    fn test_step_lr() {
        let mut optimizer = MockOptimizer::new(0.1);
        let mut scheduler = StepLR::new(3, 0.1);

        // First 3 epochs: lr = 0.1
        scheduler.step(&mut optimizer);
        assert!((optimizer.lr() - 0.1).abs() < 1e-6);
        scheduler.step(&mut optimizer);
        assert!((optimizer.lr() - 0.1).abs() < 1e-6);
        scheduler.step(&mut optimizer);
        // After step 3: lr = 0.1 * 0.1 = 0.01
        assert!((optimizer.lr() - 0.01).abs() < 1e-6);

        // Next 3 epochs
        scheduler.step(&mut optimizer);
        scheduler.step(&mut optimizer);
        scheduler.step(&mut optimizer);
        // After step 6: lr = 0.01 * 0.1 = 0.001
        assert!((optimizer.lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_lr() {
        let mut optimizer = MockOptimizer::new(0.1);
        let mut scheduler = ExponentialLR::new(0.9);

        scheduler.step(&mut optimizer);
        assert!((optimizer.lr() - 0.09).abs() < 1e-6);

        scheduler.step(&mut optimizer);
        assert!((optimizer.lr() - 0.081).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing() {
        let mut optimizer = MockOptimizer::new(0.1);
        let mut scheduler = CosineAnnealingLR::new(10);

        // At epoch 0 (before step): lr = 0.1
        scheduler.step(&mut optimizer);
        // At epoch 1: should be close to initial (cosine starts at 1)
        assert!(optimizer.lr() < 0.1);
        assert!(optimizer.lr() > 0.09);

        // At epoch 5 (halfway): should be around 0.05
        for _ in 0..4 {
            scheduler.step(&mut optimizer);
        }
        assert!((optimizer.lr() - 0.05).abs() < 0.01);

        // At epoch 10: should be close to 0
        for _ in 0..5 {
            scheduler.step(&mut optimizer);
        }
        assert!(optimizer.lr() < 0.01);
    }

    #[test]
    fn test_linear_warmup() {
        let mut optimizer = MockOptimizer::new(0.1);
        let mut scheduler = LinearWarmup::new(5);

        // During warmup
        scheduler.step(&mut optimizer);
        assert!((optimizer.lr() - 0.02).abs() < 1e-6); // 0.1 * 1/5

        scheduler.step(&mut optimizer);
        assert!((optimizer.lr() - 0.04).abs() < 1e-6); // 0.1 * 2/5

        // After warmup
        for _ in 0..3 {
            scheduler.step(&mut optimizer);
        }
        assert!((optimizer.lr() - 0.1).abs() < 1e-6);

        scheduler.step(&mut optimizer);
        assert!((optimizer.lr() - 0.1).abs() < 1e-6); // Stays at initial
    }

    #[test]
    fn test_warmup_cosine() {
        let mut optimizer = MockOptimizer::new(0.1);
        let mut scheduler = WarmupCosineScheduler::new(5, 20);

        // Warmup phase
        scheduler.step(&mut optimizer);
        assert!((optimizer.lr() - 0.02).abs() < 1e-6);

        // Complete warmup
        for _ in 0..4 {
            scheduler.step(&mut optimizer);
        }
        assert!((optimizer.lr() - 0.1).abs() < 1e-6);

        // Decay phase starts
        scheduler.step(&mut optimizer);
        assert!(optimizer.lr() < 0.1);
    }

    #[test]
    fn test_reduce_on_plateau() {
        let mut optimizer = MockOptimizer::new(0.1);
        let mut scheduler = ReduceLROnPlateau::new(PlateauMode::Min, 0.1, 3);

        // Improving
        scheduler.step_with_metric(&mut optimizer, 1.0);
        assert!((optimizer.lr() - 0.1).abs() < 1e-6);

        scheduler.step_with_metric(&mut optimizer, 0.9);
        assert!((optimizer.lr() - 0.1).abs() < 1e-6);

        // Plateau (no improvement for 3 epochs)
        scheduler.step_with_metric(&mut optimizer, 0.9);
        scheduler.step_with_metric(&mut optimizer, 0.9);
        scheduler.step_with_metric(&mut optimizer, 0.9);

        // LR should be reduced
        assert!((optimizer.lr() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_on_plateau_max_mode() {
        let mut optimizer = MockOptimizer::new(0.1);
        let mut scheduler = ReduceLROnPlateau::new(PlateauMode::Max, 0.5, 2);

        // Improving
        scheduler.step_with_metric(&mut optimizer, 0.5);
        scheduler.step_with_metric(&mut optimizer, 0.6);
        assert!((optimizer.lr() - 0.1).abs() < 1e-6);

        // Plateau
        scheduler.step_with_metric(&mut optimizer, 0.6);
        scheduler.step_with_metric(&mut optimizer, 0.6);

        // LR should be reduced
        assert!((optimizer.lr() - 0.05).abs() < 1e-6);
    }

    // Additional tests for coverage

    #[test]
    fn test_step_lr_with_lr() {
        let mut optimizer = MockOptimizer::new(0.1);
        let mut scheduler = StepLR::with_lr(0.2, 2, 0.5);

        assert_eq!(scheduler.get_lr(), 0.2);
        assert_eq!(scheduler.last_epoch(), 0);

        scheduler.step(&mut optimizer);
        assert_eq!(scheduler.last_epoch(), 1);
        scheduler.step(&mut optimizer);
        // After 2 steps: 0.2 * 0.5 = 0.1
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_lr_with_lr() {
        let mut optimizer = MockOptimizer::new(0.1);
        let mut scheduler = ExponentialLR::with_lr(0.5, 0.8);

        assert_eq!(scheduler.get_lr(), 0.5);
        assert_eq!(scheduler.last_epoch(), 0);

        scheduler.step(&mut optimizer);
        assert!((scheduler.get_lr() - 0.4).abs() < 1e-6);
        assert_eq!(scheduler.last_epoch(), 1);
    }

    #[test]
    fn test_cosine_annealing_with_min_lr() {
        let mut optimizer = MockOptimizer::new(0.1);
        let mut scheduler = CosineAnnealingLR::with_min_lr(10, 0.01);

        scheduler.step(&mut optimizer);
        assert!(scheduler.get_lr() > 0.01);
        assert!(scheduler.get_lr() < 0.1);
    }

    #[test]
    fn test_cosine_annealing_with_lr() {
        let mut optimizer = MockOptimizer::new(0.05);
        let mut scheduler = CosineAnnealingLR::with_lr(0.2, 10, 0.02);

        assert_eq!(scheduler.get_lr(), 0.2);
        scheduler.step(&mut optimizer);
        // Should use initial_lr of 0.2, not optimizer's 0.05
        assert!(scheduler.get_lr() < 0.2);
        assert!(scheduler.get_lr() > 0.02);
    }

    #[test]
    fn test_linear_warmup_with_lr() {
        let mut optimizer = MockOptimizer::new(0.05);
        let mut scheduler = LinearWarmup::with_lr(0.2, 4);

        assert_eq!(scheduler.get_lr(), 0.0); // before any step
        scheduler.step(&mut optimizer);
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-6); // 0.2 * 1/4
        assert_eq!(scheduler.last_epoch(), 1);
    }

    #[test]
    fn test_warmup_cosine_with_min_lr() {
        let mut optimizer = MockOptimizer::new(0.1);
        let mut scheduler = WarmupCosineScheduler::with_min_lr(5, 20, 0.001);

        // Complete warmup
        for _ in 0..5 {
            scheduler.step(&mut optimizer);
        }
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);

        // Start decay
        scheduler.step(&mut optimizer);
        assert!(scheduler.get_lr() < 0.1);
        assert!(scheduler.get_lr() > 0.001);
        assert_eq!(scheduler.last_epoch(), 6);
    }

    #[test]
    fn test_reduce_on_plateau_min_lr_builder() {
        let scheduler = ReduceLROnPlateau::new(PlateauMode::Min, 0.1, 3).min_lr(0.0001);
        assert!((scheduler.min_lr - 0.0001).abs() < 1e-8);
    }

    #[test]
    fn test_reduce_on_plateau_threshold_builder() {
        let scheduler = ReduceLROnPlateau::new(PlateauMode::Min, 0.1, 3).threshold(0.001);
        assert!((scheduler.threshold - 0.001).abs() < 1e-8);
    }

    #[test]
    fn test_reduce_on_plateau_step_without_metric() {
        let mut optimizer = MockOptimizer::new(0.1);
        let mut scheduler = ReduceLROnPlateau::new(PlateauMode::Min, 0.1, 2);

        // Call step without metric (should just increment epoch)
        scheduler.step(&mut optimizer);
        assert_eq!(scheduler.last_epoch(), 1);
        scheduler.step(&mut optimizer);
        assert_eq!(scheduler.last_epoch(), 2);
    }

    #[test]
    fn test_reduce_on_plateau_min_lr_clamp() {
        let mut optimizer = MockOptimizer::new(0.001);
        let mut scheduler = ReduceLROnPlateau::new(PlateauMode::Min, 0.1, 1).min_lr(0.0005);

        // First metric establishes baseline
        scheduler.step_with_metric(&mut optimizer, 1.0);
        // No improvement triggers reduction
        scheduler.step_with_metric(&mut optimizer, 1.0);
        // LR should be clamped at min_lr
        assert!(scheduler.get_lr() >= 0.0005);
    }

    #[test]
    fn test_step_lr_getters() {
        let scheduler = StepLR::with_lr(0.1, 5, 0.9);
        assert_eq!(scheduler.get_lr(), 0.1);
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_exponential_lr_getters() {
        let scheduler = ExponentialLR::with_lr(0.1, 0.9);
        assert_eq!(scheduler.get_lr(), 0.1);
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_cosine_annealing_getters() {
        let scheduler = CosineAnnealingLR::with_lr(0.1, 10, 0.01);
        assert_eq!(scheduler.get_lr(), 0.1);
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_linear_warmup_getters() {
        let scheduler = LinearWarmup::with_lr(0.1, 5);
        assert_eq!(scheduler.get_lr(), 0.0);
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_warmup_cosine_getters() {
        let scheduler = WarmupCosineScheduler::with_min_lr(5, 20, 0.01);
        assert_eq!(scheduler.get_lr(), 0.0);
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_reduce_on_plateau_getters() {
        let scheduler = ReduceLROnPlateau::new(PlateauMode::Max, 0.5, 3);
        assert_eq!(scheduler.get_lr(), 0.0);
        assert_eq!(scheduler.last_epoch(), 0);
    }

    #[test]
    fn test_plateau_mode_eq() {
        assert_eq!(PlateauMode::Min, PlateauMode::Min);
        assert_eq!(PlateauMode::Max, PlateauMode::Max);
        assert_ne!(PlateauMode::Min, PlateauMode::Max);
    }

    #[test]
    fn test_scheduler_clone() {
        let scheduler = StepLR::with_lr(0.1, 5, 0.9);
        let cloned = scheduler.clone();
        assert_eq!(scheduler.get_lr(), cloned.get_lr());
    }

    #[test]
    fn test_scheduler_debug() {
        let scheduler = StepLR::with_lr(0.1, 5, 0.9);
        let debug = format!("{scheduler:?}");
        assert!(debug.contains("StepLR"));
    }
}
