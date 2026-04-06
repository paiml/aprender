//! Online learning with SGD updates (MLT-12)
//!
//! Momentum SGD with replay buffer for catastrophic forgetting prevention.

use super::super::pretrained;

/// Online learning state for SGD updates (MLT-12)
#[derive(Debug, Clone, Default)]
pub struct OnlineLearner {
    /// Current weights
    pub(super) weights: Vec<f32>,
    /// Learning rate
    learning_rate: f32,
    /// Momentum term
    momentum: f32,
    /// Velocity for momentum SGD
    velocity: Vec<f32>,
    /// Number of updates
    num_updates: usize,
    /// Exponential moving average of loss
    ema_loss: f32,
    /// Replay buffer for catastrophic forgetting prevention
    replay_buffer: Vec<(Vec<f32>, f32)>,
    /// Max replay buffer size
    replay_buffer_size: usize,
}

impl OnlineLearner {
    /// Create new online learner with pretrained weights
    pub fn new() -> Self {
        let weights = pretrained::THROUGHPUT_WEIGHTS.to_vec();
        let velocity = vec![0.0; weights.len()];
        Self {
            weights,
            learning_rate: 0.001,
            momentum: 0.9,
            velocity,
            num_updates: 0,
            ema_loss: 0.0,
            replay_buffer: Vec::new(),
            replay_buffer_size: 100,
        }
    }

    /// Create learner with custom learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Observe a new sample and update weights (SGD step)
    pub fn observe(&mut self, features: &[f32], actual_throughput: f32) {
        if features.len() + 1 != self.weights.len() {
            return; // Dimension mismatch
        }

        // Forward pass: predict
        let predicted = self.predict(features);
        let error = predicted - actual_throughput;

        // Update EMA loss
        let alpha = 0.1;
        self.ema_loss = alpha * error.abs() + (1.0 - alpha) * self.ema_loss;

        // Backward pass: compute gradients
        // For linear model: dL/dw_i = 2 * error * x_i
        let mut gradients = vec![0.0; self.weights.len()];
        gradients[0] = 2.0 * error; // bias gradient
        for (i, &x) in features.iter().enumerate() {
            gradients[i + 1] = 2.0 * error * x;
        }

        // Momentum SGD update
        for i in 0..self.weights.len() {
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * gradients[i];
            self.weights[i] += self.velocity[i];
        }

        // Add to replay buffer
        if self.replay_buffer.len() >= self.replay_buffer_size {
            // Remove oldest
            self.replay_buffer.remove(0);
        }
        self.replay_buffer.push((features.to_vec(), actual_throughput));

        self.num_updates += 1;

        // Periodic replay to prevent catastrophic forgetting
        if self.num_updates % 10 == 0 && !self.replay_buffer.is_empty() {
            self.replay_step();
        }
    }

    /// Replay a random sample from buffer
    fn replay_step(&mut self) {
        if self.replay_buffer.is_empty() {
            return;
        }

        // Simple: replay oldest sample
        let (features, target) = self.replay_buffer[0].clone();

        let predicted = self.predict(&features);
        let error = predicted - target;

        // Smaller learning rate for replay
        let replay_lr = self.learning_rate * 0.1;
        self.weights[0] -= replay_lr * 2.0 * error;
        for (i, &x) in features.iter().enumerate() {
            self.weights[i + 1] -= replay_lr * 2.0 * error * x;
        }
    }

    /// Predict throughput
    pub fn predict(&self, features: &[f32]) -> f32 {
        if features.is_empty() {
            return self.weights[0]; // bias-only prediction
        }
        contract_pre_predict!(features);
        let mut result = self.weights[0]; // bias
        for (i, &x) in features.iter().enumerate() {
            if i + 1 < self.weights.len() {
                result += self.weights[i + 1] * x;
            }
        }
        result.max(0.0) // Throughput must be non-negative
    }

    /// Get current weights
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get number of updates
    pub fn num_updates(&self) -> usize {
        self.num_updates
    }

    /// Get current EMA loss
    pub fn ema_loss(&self) -> f32 {
        self.ema_loss
    }

    /// Check if model is converging (loss decreasing)
    pub fn is_converging(&self) -> bool {
        self.ema_loss < 0.15 // 15% MAPE threshold
    }
}
