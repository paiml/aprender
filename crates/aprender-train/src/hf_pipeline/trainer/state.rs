//! Training state tracking.

use std::time::{Duration, Instant};

/// Training state tracking
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current epoch (0-indexed)
    pub epoch: usize,
    /// Current global step
    pub global_step: usize,
    /// Steps completed in current epoch
    pub epoch_step: usize,
    /// Best validation loss seen
    pub best_val_loss: f32,
    /// Training start time
    pub start_time: Instant,
    /// Loss history (step, loss)
    pub loss_history: Vec<(usize, f32)>,
    /// Validation loss history (step, loss)
    pub val_loss_history: Vec<(usize, f32)>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingState {
    /// Create new training state
    #[must_use]
    pub fn new() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            epoch_step: 0,
            best_val_loss: f32::INFINITY,
            start_time: Instant::now(),
            loss_history: Vec::new(),
            val_loss_history: Vec::new(),
        }
    }

    /// Record training loss
    pub fn record_loss(&mut self, loss: f32) {
        self.loss_history.push((self.global_step, loss));
    }

    /// Record validation loss
    pub fn record_val_loss(&mut self, loss: f32) -> bool {
        self.val_loss_history.push((self.global_step, loss));
        if loss < self.best_val_loss {
            self.best_val_loss = loss;
            true // New best
        } else {
            false
        }
    }

    /// Advance one step
    pub fn step(&mut self) {
        self.global_step += 1;
        self.epoch_step += 1;
    }

    /// Start new epoch
    pub fn new_epoch(&mut self) {
        self.epoch += 1;
        self.epoch_step = 0;
    }

    /// Get elapsed time
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get average loss over last N steps
    #[must_use]
    pub fn avg_loss(&self, n: usize) -> Option<f32> {
        if self.loss_history.is_empty() {
            return None;
        }
        let start = self.loss_history.len().saturating_sub(n);
        let sum: f32 = self.loss_history[start..].iter().map(|(_, l)| l).sum();
        Some(sum / (self.loss_history.len() - start) as f32)
    }

    /// Get steps per second
    #[must_use]
    pub fn steps_per_second(&self) -> f32 {
        let elapsed = self.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            self.global_step as f32 / elapsed
        } else {
            0.0
        }
    }

    /// Get estimated time remaining
    #[must_use]
    pub fn eta(&self, total_steps: usize) -> Duration {
        let sps = self.steps_per_second();
        if sps > 0.0 {
            let remaining = total_steps.saturating_sub(self.global_step);
            Duration::from_secs_f32(remaining as f32 / sps)
        } else {
            Duration::ZERO
        }
    }
}
