//! Linear warmup learning rate scheduler

use super::LRScheduler;
use crate::optim::Optimizer;

/// Linear Warmup Learning Rate Scheduler
///
/// Linearly increases learning rate from 0 to target over warmup_steps.
/// After warmup, maintains target learning rate.
///
/// Formula: lr_t = lr_target * min(1, t / warmup_steps)
pub struct LinearWarmupLR {
    lr_target: f32,
    warmup_steps: usize,
    current_step: usize,
}

impl LinearWarmupLR {
    /// Create a new linear warmup scheduler
    ///
    /// # Arguments
    /// * `lr_target` - Target learning rate after warmup
    /// * `warmup_steps` - Number of steps for warmup
    pub fn new(lr_target: f32, warmup_steps: usize) -> Self {
        Self { lr_target, warmup_steps, current_step: 0 }
    }

    /// Apply the current learning rate to an optimizer
    pub fn apply<O: Optimizer>(&self, optimizer: &mut O) {
        optimizer.set_lr(self.get_lr());
    }
}

impl LRScheduler for LinearWarmupLR {
    fn get_lr(&self) -> f32 {
        if self.warmup_steps == 0 {
            return self.lr_target;
        }

        let progress = (self.current_step as f32 / self.warmup_steps as f32).min(1.0);
        self.lr_target * progress
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}
