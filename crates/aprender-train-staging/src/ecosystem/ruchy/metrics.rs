//! Session metrics for training history tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Training metrics from a session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMetrics {
    /// Loss values over time
    pub loss_history: Vec<f64>,
    /// Accuracy values over time (optional)
    pub accuracy_history: Vec<f64>,
    /// Learning rate schedule
    pub lr_history: Vec<f64>,
    /// Gradient norms (for debugging)
    pub grad_norm_history: Vec<f64>,
    /// Custom metrics
    pub custom: HashMap<String, Vec<f64>>,
}

impl SessionMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a loss value.
    pub fn add_loss(&mut self, loss: f64) {
        self.loss_history.push(loss);
    }

    /// Add an accuracy value.
    pub fn add_accuracy(&mut self, accuracy: f64) {
        self.accuracy_history.push(accuracy);
    }

    /// Add a learning rate value.
    pub fn add_lr(&mut self, lr: f64) {
        self.lr_history.push(lr);
    }

    /// Add a gradient norm value.
    pub fn add_grad_norm(&mut self, norm: f64) {
        self.grad_norm_history.push(norm);
    }

    /// Add a custom metric value.
    pub fn add_custom(&mut self, name: impl Into<String>, value: f64) {
        self.custom.entry(name.into()).or_default().push(value);
    }

    /// Get final loss (last value).
    pub fn final_loss(&self) -> Option<f64> {
        self.loss_history.last().copied()
    }

    /// Get final accuracy (last value).
    pub fn final_accuracy(&self) -> Option<f64> {
        self.accuracy_history.last().copied()
    }

    /// Get best loss (minimum).
    pub fn best_loss(&self) -> Option<f64> {
        self.loss_history.iter().copied().reduce(f64::min)
    }

    /// Get best accuracy (maximum).
    pub fn best_accuracy(&self) -> Option<f64> {
        self.accuracy_history.iter().copied().reduce(f64::max)
    }

    /// Get total training steps.
    pub fn total_steps(&self) -> usize {
        self.loss_history.len()
    }

    /// Check if metrics are empty.
    pub fn is_empty(&self) -> bool {
        self.loss_history.is_empty() && self.accuracy_history.is_empty() && self.custom.is_empty()
    }
}
