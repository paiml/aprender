#[allow(clippy::wildcard_imports)]
use super::*;

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
#[path = "tests.rs"]
mod tests;
