//! Gating networks for expert routing

use serde::{Deserialize, Serialize};

/// Trait for gating networks that route inputs to experts
pub trait GatingNetwork: Send + Sync {
    /// Compute expert weights for input
    fn forward(&self, x: &[f32]) -> Vec<f32>;

    /// Number of input features
    fn n_features(&self) -> usize;

    /// Number of experts
    fn n_experts(&self) -> usize;
}

/// Softmax gating with learnable weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxGating {
    n_features: usize,
    n_experts: usize,
    temperature: f32,
    weights: Vec<f32>,
}

impl SoftmaxGating {
    /// Create new softmax gating
    #[must_use]
    pub fn new(n_features: usize, n_experts: usize) -> Self {
        let scale = (2.0 / (n_features + n_experts) as f32).sqrt();
        let weights: Vec<f32> = (0..n_features * n_experts)
            .map(|i| {
                let row = i / n_experts;
                let col = i % n_experts;
                scale * ((row + col) as f32 * 0.1 - 0.5)
            })
            .collect();

        Self {
            n_features,
            n_experts,
            temperature: 1.0,
            weights,
        }
    }

    /// Set temperature
    #[must_use]
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Get temperature
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let scaled: Vec<f32> = logits.iter().map(|&x| x / self.temperature).collect();
        let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|&x| x / sum).collect()
    }
}

impl GatingNetwork for SoftmaxGating {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.n_experts];
        for (j, logit) in logits.iter_mut().enumerate() {
            for (i, &xi) in x.iter().take(self.n_features).enumerate() {
                *logit += xi * self.weights[i * self.n_experts + j];
            }
        }
        self.softmax(&logits)
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn n_experts(&self) -> usize {
        self.n_experts
    }
}
