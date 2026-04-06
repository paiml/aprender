//! Cosine annealing learning rate scheduler

use super::LRScheduler;
use crate::optim::Optimizer;
use std::f32::consts::PI;

/// Cosine Annealing Learning Rate Scheduler
///
/// Decreases the learning rate following a cosine curve from lr_max to lr_min.
///
/// Formula: lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
///
/// Where:
/// - t is the current step
/// - T is the total number of steps
/// - lr_max is the initial learning rate
/// - lr_min is the minimum learning rate (default 0)
pub struct CosineAnnealingLR {
    lr_max: f32,
    lr_min: f32,
    t_max: usize,
    current_step: usize,
}

impl CosineAnnealingLR {
    /// Create a new cosine annealing scheduler
    ///
    /// # Arguments
    /// * `lr_max` - Initial (maximum) learning rate
    /// * `t_max` - Total number of steps for the schedule
    /// * `lr_min` - Minimum learning rate (default 0)
    pub fn new(lr_max: f32, t_max: usize, lr_min: f32) -> Self {
        Self { lr_max, lr_min, t_max, current_step: 0 }
    }

    /// Create scheduler with lr_min = 0
    pub fn default_min(lr_max: f32, t_max: usize) -> Self {
        Self::new(lr_max, t_max, 0.0)
    }

    /// Apply the current learning rate to an optimizer
    pub fn apply<O: Optimizer>(&self, optimizer: &mut O) {
        optimizer.set_lr(self.get_lr());
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self) -> f32 {
        if self.current_step >= self.t_max {
            return self.lr_min;
        }

        let progress = self.current_step as f32 / self.t_max as f32;
        let cosine_decay = 0.5 * (1.0 + (PI * progress).cos());
        self.lr_min + (self.lr_max - self.lr_min) * cosine_decay
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}
