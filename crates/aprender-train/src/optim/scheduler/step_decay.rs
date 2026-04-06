//! Step decay learning rate scheduler

use super::LRScheduler;
use crate::optim::Optimizer;

/// Step Decay Learning Rate Scheduler
///
/// Multiplies learning rate by gamma every step_size epochs.
///
/// Formula: lr_t = lr_initial * gamma^(floor(epoch / step_size))
pub struct StepDecayLR {
    lr_initial: f32,
    gamma: f32,
    step_size: usize,
    current_epoch: usize,
}

impl StepDecayLR {
    /// Create a new step decay scheduler
    ///
    /// # Arguments
    /// * `lr_initial` - Initial learning rate
    /// * `step_size` - Decay LR every step_size epochs
    /// * `gamma` - Multiplicative factor (e.g., 0.1 for 10x reduction)
    pub fn new(lr_initial: f32, step_size: usize, gamma: f32) -> Self {
        Self { lr_initial, gamma, step_size, current_epoch: 0 }
    }

    /// Apply the current learning rate to an optimizer
    pub fn apply<O: Optimizer>(&self, optimizer: &mut O) {
        optimizer.set_lr(self.get_lr());
    }
}

impl LRScheduler for StepDecayLR {
    fn get_lr(&self) -> f32 {
        if self.step_size == 0 {
            return self.lr_initial;
        }
        let num_decays = self.current_epoch / self.step_size;
        self.lr_initial * self.gamma.powi(num_decays as i32)
    }

    fn step(&mut self) {
        self.current_epoch += 1;
    }
}
