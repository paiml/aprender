//! Training configuration and metrics

use std::path::PathBuf;

/// Training configuration
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Maximum gradient norm for clipping (None = no clipping)
    pub max_grad_norm: Option<f32>,

    /// Print training progress every N steps
    pub log_interval: usize,

    /// Save checkpoint every N epochs
    pub save_interval: Option<usize>,

    /// Directory to save checkpoints
    pub checkpoint_dir: Option<PathBuf>,

    /// Use mixed precision training
    pub mixed_precision: bool,

    /// Gradient accumulation steps (1 = no accumulation)
    ///
    /// Simulates larger batch sizes by accumulating gradients over
    /// multiple mini-batches before performing an optimizer step.
    /// Effective batch size = batch_size * gradient_accumulation_steps
    pub gradient_accumulation_steps: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            max_grad_norm: Some(1.0),
            log_interval: 10,
            save_interval: None,
            checkpoint_dir: None,
            mixed_precision: false,
            gradient_accumulation_steps: 1,
        }
    }
}

impl TrainConfig {
    /// Create a new training configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set gradient clipping norm
    pub fn with_grad_clip(mut self, max_norm: f32) -> Self {
        self.max_grad_norm = Some(max_norm);
        self
    }

    /// Disable gradient clipping
    pub fn without_grad_clip(mut self) -> Self {
        self.max_grad_norm = None;
        self
    }

    /// Set logging interval
    pub fn with_log_interval(mut self, interval: usize) -> Self {
        self.log_interval = interval;
        self
    }

    /// Set checkpoint saving
    pub fn with_checkpoints(mut self, interval: usize, dir: PathBuf) -> Self {
        self.save_interval = Some(interval);
        self.checkpoint_dir = Some(dir);
        self
    }

    /// Set gradient accumulation steps
    ///
    /// Simulates larger batch sizes by accumulating gradients over
    /// multiple mini-batches before performing an optimizer step.
    /// Effective batch size = batch_size * gradient_accumulation_steps
    pub fn with_gradient_accumulation(mut self, steps: usize) -> Self {
        self.gradient_accumulation_steps = steps.max(1);
        self
    }
}

/// Tracks training metrics across epochs
#[derive(Clone, Debug)]
pub struct MetricsTracker {
    /// Training loss history (one per epoch)
    pub losses: Vec<f32>,

    /// Validation loss history (one per epoch, if validation is used)
    pub val_losses: Vec<f32>,

    /// Learning rates (one per epoch)
    pub learning_rates: Vec<f32>,

    /// Training step count
    pub steps: usize,

    /// Current epoch
    pub epoch: usize,
}

impl MetricsTracker {
    /// Create a new metrics tracker
    pub fn new() -> Self {
        Self {
            losses: Vec::new(),
            val_losses: Vec::new(),
            learning_rates: Vec::new(),
            steps: 0,
            epoch: 0,
        }
    }

    /// Record an epoch's training metrics
    pub fn record_epoch(&mut self, loss: f32, lr: f32) {
        self.losses.push(loss);
        self.learning_rates.push(lr);
        self.epoch += 1;
    }

    /// Record validation loss for the current epoch
    pub fn record_val_loss(&mut self, val_loss: f32) {
        self.val_losses.push(val_loss);
    }

    /// Get best (minimum) validation loss
    pub fn best_val_loss(&self) -> Option<f32> {
        self.val_losses.iter().copied().min_by(f32::total_cmp)
    }

    /// Check if validation loss is improving
    pub fn is_val_improving(&self, patience: usize) -> bool {
        if self.val_losses.len() < patience {
            return true;
        }

        let recent = self.val_losses[self.val_losses.len() - patience..].to_vec();
        let mut sorted = recent.clone();
        sorted.sort_by(f32::total_cmp);

        // Check if val losses are generally decreasing
        recent != sorted
    }

    /// Increment step counter
    pub fn increment_step(&mut self) {
        self.steps += 1;
    }

    /// Get average loss over last N epochs
    pub fn avg_loss(&self, n: usize) -> f32 {
        if self.losses.is_empty() {
            return 0.0;
        }

        let start = self.losses.len().saturating_sub(n);
        let window = &self.losses[start..];
        window.iter().sum::<f32>() / window.len() as f32
    }

    /// Get best (minimum) loss
    pub fn best_loss(&self) -> Option<f32> {
        self.losses.iter().copied().min_by(f32::total_cmp)
    }

    /// Check if training is improving (loss decreasing)
    pub fn is_improving(&self, patience: usize) -> bool {
        if self.losses.len() < patience {
            return true;
        }

        let recent = self.losses[self.losses.len() - patience..].to_vec();
        let mut sorted = recent.clone();
        sorted.sort_by(f32::total_cmp);

        // Check if losses are generally decreasing
        recent != sorted
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_config_default() {
        let config = TrainConfig::default();
        assert_eq!(config.max_grad_norm, Some(1.0));
        assert_eq!(config.log_interval, 10);
        assert!(config.save_interval.is_none());
        assert_eq!(config.gradient_accumulation_steps, 1);
    }

    #[test]
    fn test_train_config_builder() {
        let config =
            TrainConfig::new().with_grad_clip(0.5).with_log_interval(20).without_grad_clip();

        assert_eq!(config.max_grad_norm, None);
        assert_eq!(config.log_interval, 20);
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new();

        tracker.record_epoch(1.0, 0.001);
        tracker.record_epoch(0.8, 0.001);
        tracker.record_epoch(0.6, 0.001);

        assert_eq!(tracker.epoch, 3);
        assert_eq!(tracker.losses.len(), 3);
        assert_eq!(tracker.best_loss(), Some(0.6));
    }

    #[test]
    fn test_metrics_avg_loss() {
        let mut tracker = MetricsTracker::new();

        tracker.record_epoch(1.0, 0.001);
        tracker.record_epoch(0.8, 0.001);
        tracker.record_epoch(0.6, 0.001);

        let avg = tracker.avg_loss(2);
        assert!((avg - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_metrics_is_improving() {
        let mut tracker = MetricsTracker::new();

        // Decreasing losses = improving
        tracker.record_epoch(1.0, 0.001);
        tracker.record_epoch(0.8, 0.001);
        tracker.record_epoch(0.6, 0.001);

        assert!(tracker.is_improving(2));
    }

    #[test]
    fn test_gradient_accumulation_builder() {
        let config = TrainConfig::new().with_gradient_accumulation(4);
        assert_eq!(config.gradient_accumulation_steps, 4);
    }

    #[test]
    fn test_gradient_accumulation_min_value() {
        // Should clamp to minimum of 1
        let config = TrainConfig::new().with_gradient_accumulation(0);
        assert_eq!(config.gradient_accumulation_steps, 1);
    }

    #[test]
    fn test_validation_loss_tracking() {
        let mut tracker = MetricsTracker::new();

        tracker.record_epoch(1.0, 0.001);
        tracker.record_val_loss(0.9);
        tracker.record_epoch(0.8, 0.001);
        tracker.record_val_loss(0.7);
        tracker.record_epoch(0.6, 0.001);
        tracker.record_val_loss(0.5);

        assert_eq!(tracker.val_losses.len(), 3);
        assert_eq!(tracker.best_val_loss(), Some(0.5));
    }

    #[test]
    fn test_validation_is_improving() {
        let mut tracker = MetricsTracker::new();

        // Decreasing val losses = improving
        tracker.record_val_loss(0.9);
        tracker.record_val_loss(0.7);
        tracker.record_val_loss(0.5);

        assert!(tracker.is_val_improving(2));
    }

    #[test]
    fn test_validation_not_improving() {
        let mut tracker = MetricsTracker::new();

        // Increasing val losses = not improving
        tracker.record_val_loss(0.5);
        tracker.record_val_loss(0.6);
        tracker.record_val_loss(0.7);

        assert!(!tracker.is_val_improving(2));
    }

    #[test]
    fn test_with_checkpoints() {
        let config = TrainConfig::new().with_checkpoints(5, PathBuf::from("/tmp/checkpoints"));
        assert_eq!(config.save_interval, Some(5));
        assert_eq!(config.checkpoint_dir, Some(PathBuf::from("/tmp/checkpoints")));
    }

    #[test]
    fn test_increment_step() {
        let mut tracker = MetricsTracker::new();
        assert_eq!(tracker.steps, 0);
        tracker.increment_step();
        assert_eq!(tracker.steps, 1);
        tracker.increment_step();
        assert_eq!(tracker.steps, 2);
    }

    #[test]
    fn test_metrics_tracker_default() {
        let tracker = MetricsTracker::default();
        assert!(tracker.losses.is_empty());
        assert!(tracker.val_losses.is_empty());
        assert_eq!(tracker.steps, 0);
        assert_eq!(tracker.epoch, 0);
    }

    #[test]
    fn test_avg_loss_empty() {
        let tracker = MetricsTracker::new();
        assert_eq!(tracker.avg_loss(5), 0.0);
    }

    #[test]
    fn test_best_loss_empty() {
        let tracker = MetricsTracker::new();
        assert!(tracker.best_loss().is_none());
    }

    #[test]
    fn test_best_val_loss_empty() {
        let tracker = MetricsTracker::new();
        assert!(tracker.best_val_loss().is_none());
    }

    #[test]
    fn test_is_improving_insufficient_data() {
        let mut tracker = MetricsTracker::new();
        tracker.record_epoch(1.0, 0.001);
        // With patience=3 and only 1 data point, should return true
        assert!(tracker.is_improving(3));
    }

    #[test]
    fn test_is_val_improving_insufficient_data() {
        let mut tracker = MetricsTracker::new();
        tracker.record_val_loss(0.5);
        // With patience=3 and only 1 data point, should return true
        assert!(tracker.is_val_improving(3));
    }

    #[test]
    fn test_train_config_clone() {
        let config = TrainConfig::new().with_grad_clip(0.5);
        let cloned = config.clone();
        assert_eq!(config.max_grad_norm, cloned.max_grad_norm);
    }

    #[test]
    fn test_metrics_tracker_clone() {
        let mut tracker = MetricsTracker::new();
        tracker.record_epoch(1.0, 0.001);
        let cloned = tracker.clone();
        assert_eq!(tracker.losses, cloned.losses);
    }
}
