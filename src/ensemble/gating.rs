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

    /// ONE PATH: Scales then delegates to `nn::functional::softmax_1d` (UCBD §4).
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let scaled: Vec<f32> = logits.iter().map(|&x| x / self.temperature).collect();
        crate::nn::functional::softmax_1d(&scaled)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_gating_new() {
        let gating = SoftmaxGating::new(10, 4);
        assert_eq!(gating.n_features(), 10);
        assert_eq!(gating.n_experts(), 4);
        assert_eq!(gating.temperature(), 1.0);
    }

    #[test]
    fn test_softmax_gating_with_temperature() {
        let gating = SoftmaxGating::new(5, 3).with_temperature(0.5);
        assert!((gating.temperature() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_gating_forward() {
        let gating = SoftmaxGating::new(3, 2);
        let input = vec![1.0, 0.5, 0.2];
        let weights = gating.forward(&input);

        // Should return weights for each expert
        assert_eq!(weights.len(), 2);

        // Weights should sum to ~1.0 (softmax property)
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // All weights should be positive
        for w in &weights {
            assert!(*w >= 0.0);
        }
    }

    #[test]
    fn test_softmax_gating_forward_longer_input() {
        let gating = SoftmaxGating::new(3, 2);
        // Input longer than n_features - should only use first n_features
        let input = vec![1.0, 0.5, 0.2, 0.8, 0.9];
        let weights = gating.forward(&input);
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_softmax_gating_temperature_effect() {
        let gating_high_temp = SoftmaxGating::new(3, 4).with_temperature(10.0);
        let gating_low_temp = SoftmaxGating::new(3, 4).with_temperature(0.1);

        let input = vec![1.0, 2.0, 3.0];
        let weights_high = gating_high_temp.forward(&input);
        let weights_low = gating_low_temp.forward(&input);

        // High temperature should give more uniform distribution
        let high_max = weights_high.iter().cloned().fold(0.0f32, f32::max);
        let low_max = weights_low.iter().cloned().fold(0.0f32, f32::max);

        // Low temperature should have a more peaked distribution
        assert!(low_max > high_max);
    }

    #[test]
    fn test_softmax_gating_clone() {
        let gating = SoftmaxGating::new(5, 3).with_temperature(2.0);
        let cloned = gating.clone();
        assert_eq!(cloned.n_features(), gating.n_features());
        assert_eq!(cloned.n_experts(), gating.n_experts());
        assert!((cloned.temperature() - gating.temperature()).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_gating_debug() {
        let gating = SoftmaxGating::new(3, 2);
        let debug_str = format!("{:?}", gating);
        assert!(debug_str.contains("SoftmaxGating"));
    }

    #[test]
    fn test_softmax_gating_weights_initialized() {
        let gating = SoftmaxGating::new(4, 3);
        assert_eq!(gating.weights.len(), 4 * 3); // n_features * n_experts
    }
}
