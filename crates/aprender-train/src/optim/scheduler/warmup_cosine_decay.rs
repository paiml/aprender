//! Warmup + cosine decay learning rate scheduler

use super::LRScheduler;
use crate::optim::Optimizer;
use std::f32::consts::PI;

/// Warmup + Cosine Decay Learning Rate Scheduler
///
/// Combines linear warmup with cosine annealing decay.
/// - Phase 1 (warmup): Linear increase from 0 to lr_max
/// - Phase 2 (decay): Cosine decay from lr_max to lr_min
pub struct WarmupCosineDecayLR {
    lr_max: f32,
    lr_min: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl WarmupCosineDecayLR {
    /// Create a new warmup + cosine decay scheduler
    ///
    /// # Arguments
    /// * `lr_max` - Maximum learning rate (after warmup)
    /// * `lr_min` - Minimum learning rate (at end)
    /// * `warmup_steps` - Number of warmup steps
    /// * `total_steps` - Total training steps (including warmup)
    pub fn new(lr_max: f32, lr_min: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self { lr_max, lr_min, warmup_steps, total_steps, current_step: 0 }
    }

    /// Apply the current learning rate to an optimizer
    pub fn apply<O: Optimizer>(&self, optimizer: &mut O) {
        optimizer.set_lr(self.get_lr());
    }
}

impl LRScheduler for WarmupCosineDecayLR {
    fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // Warmup phase: linear increase
            if self.warmup_steps == 0 {
                return self.lr_max;
            }
            let progress = self.current_step as f32 / self.warmup_steps as f32;
            return self.lr_max * progress;
        }

        // Cosine decay phase
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
        if decay_steps == 0 {
            return self.lr_min;
        }

        let decay_step = self.current_step - self.warmup_steps;
        if decay_step >= decay_steps {
            return self.lr_min;
        }

        let progress = decay_step as f32 / decay_steps as f32;
        let cosine_decay = 0.5 * (1.0 + (PI * progress).cos());
        self.lr_min + (self.lr_max - self.lr_min) * cosine_decay
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}
