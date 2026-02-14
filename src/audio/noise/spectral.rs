//! Spectral MLP for magnitude spectrum prediction

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::{NoiseError, NoiseResult};

/// Small MLP for predicting noise magnitude spectra
/// Architecture: \[config_dim\] -> \[hidden_dim\] -> \[hidden_dim\] -> \[n_freqs\]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralMLP {
    /// Layer 1 weights: [config_dim × hidden_dim]
    weights_1: Vec<f32>,
    /// Layer 1 bias: [hidden_dim]
    bias_1: Vec<f32>,

    /// Layer 2 weights: [hidden_dim × hidden_dim]
    weights_2: Vec<f32>,
    /// Layer 2 bias: [hidden_dim]
    bias_2: Vec<f32>,

    /// Layer 3 weights: [hidden_dim × n_freqs]
    weights_3: Vec<f32>,
    /// Layer 3 bias: [n_freqs]
    bias_3: Vec<f32>,

    /// Input dimension (config encoding size)
    config_dim: usize,
    /// Hidden layer dimension
    hidden_dim: usize,
    /// Output dimension (number of frequency bins)
    n_freqs: usize,
}

impl SpectralMLP {
    /// Create a new MLP with the specified dimensions and random weights
    #[must_use]
    pub fn random_init(config_dim: usize, hidden_dim: usize, n_freqs: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);

        // Xavier/Glorot initialization
        let scale_1 = (2.0 / (config_dim + hidden_dim) as f32).sqrt();
        let scale_2 = (2.0 / (hidden_dim + hidden_dim) as f32).sqrt();
        let scale_3 = (2.0 / (hidden_dim + n_freqs) as f32).sqrt();

        let weights_1: Vec<f32> = (0..config_dim * hidden_dim)
            .map(|_| rng.gen_range(-scale_1..scale_1))
            .collect();
        let bias_1: Vec<f32> = vec![0.0; hidden_dim];

        let weights_2: Vec<f32> = (0..hidden_dim * hidden_dim)
            .map(|_| rng.gen_range(-scale_2..scale_2))
            .collect();
        let bias_2: Vec<f32> = vec![0.0; hidden_dim];

        let weights_3: Vec<f32> = (0..hidden_dim * n_freqs)
            .map(|_| rng.gen_range(-scale_3..scale_3))
            .collect();
        let bias_3: Vec<f32> = vec![0.0; n_freqs];

        Self {
            weights_1,
            bias_1,
            weights_2,
            bias_2,
            weights_3,
            bias_3,
            config_dim,
            hidden_dim,
            n_freqs,
        }
    }

    /// Create from pre-trained weights
    pub fn from_weights(
        weights_1: Vec<f32>,
        bias_1: Vec<f32>,
        weights_2: Vec<f32>,
        bias_2: Vec<f32>,
        weights_3: Vec<f32>,
        bias_3: Vec<f32>,
        config_dim: usize,
        hidden_dim: usize,
        n_freqs: usize,
    ) -> NoiseResult<Self> {
        // Validate dimensions
        if weights_1.len() != config_dim * hidden_dim {
            return Err(NoiseError::ModelError(format!(
                "weights_1 size mismatch: expected {}, got {}",
                config_dim * hidden_dim,
                weights_1.len()
            )));
        }
        if bias_1.len() != hidden_dim {
            return Err(NoiseError::ModelError(format!(
                "bias_1 size mismatch: expected {}, got {}",
                hidden_dim,
                bias_1.len()
            )));
        }
        if weights_2.len() != hidden_dim * hidden_dim {
            return Err(NoiseError::ModelError(format!(
                "weights_2 size mismatch: expected {}, got {}",
                hidden_dim * hidden_dim,
                weights_2.len()
            )));
        }
        if bias_2.len() != hidden_dim {
            return Err(NoiseError::ModelError(format!(
                "bias_2 size mismatch: expected {}, got {}",
                hidden_dim,
                bias_2.len()
            )));
        }
        if weights_3.len() != hidden_dim * n_freqs {
            return Err(NoiseError::ModelError(format!(
                "weights_3 size mismatch: expected {}, got {}",
                hidden_dim * n_freqs,
                weights_3.len()
            )));
        }
        if bias_3.len() != n_freqs {
            return Err(NoiseError::ModelError(format!(
                "bias_3 size mismatch: expected {}, got {}",
                n_freqs,
                bias_3.len()
            )));
        }

        Ok(Self {
            weights_1,
            bias_1,
            weights_2,
            bias_2,
            weights_3,
            bias_3,
            config_dim,
            hidden_dim,
            n_freqs,
        })
    }

    /// Forward pass: config -> magnitude spectrum
    /// Uses ReLU for hidden layers, softplus for output (ensures positive magnitudes)
    #[must_use]
    pub fn forward(&self, config: &[f32]) -> Vec<f32> {
        assert_eq!(
            config.len(),
            self.config_dim,
            "Config dimension mismatch: expected {}, got {}",
            self.config_dim,
            config.len()
        );

        // Layer 1: ReLU(W1 @ x + b1)
        let mut h1 = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut sum = self.bias_1[i];
            for j in 0..self.config_dim {
                sum += self.weights_1[j * self.hidden_dim + i] * config[j];
            }
            h1[i] = relu(sum);
        }

        // Layer 2: ReLU(W2 @ h1 + b2)
        let mut h2 = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut sum = self.bias_2[i];
            for j in 0..self.hidden_dim {
                sum += self.weights_2[j * self.hidden_dim + i] * h1[j];
            }
            h2[i] = relu(sum);
        }

        // Layer 3: Softplus(W3 @ h2 + b3) - ensures positive output
        let mut output = vec![0.0; self.n_freqs];
        for i in 0..self.n_freqs {
            let mut sum = self.bias_3[i];
            for j in 0..self.hidden_dim {
                sum += self.weights_3[j * self.n_freqs + i] * h2[j];
            }
            output[i] = softplus(sum);
        }

        output
    }

    /// Get config dimension
    #[must_use]
    pub fn config_dim(&self) -> usize {
        self.config_dim
    }

    /// Get hidden dimension
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get output dimension (number of frequency bins)
    #[must_use]
    pub fn n_freqs(&self) -> usize {
        self.n_freqs
    }

    /// Get total number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        self.weights_1.len()
            + self.bias_1.len()
            + self.weights_2.len()
            + self.bias_2.len()
            + self.weights_3.len()
            + self.bias_3.len()
    }

    /// Save model to .apr format
    pub fn save_apr<P: AsRef<Path>>(&self, path: P) -> NoiseResult<()> {
        let json = serde_json::to_string(self)
            .map_err(|e| NoiseError::ModelError(format!("Failed to serialize model: {}", e)))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load model from .apr format
    pub fn load_apr<P: AsRef<Path>>(path: P) -> NoiseResult<Self> {
        let json = std::fs::read_to_string(path)?;
        let model: Self = serde_json::from_str(&json)
            .map_err(|e| NoiseError::ModelError(format!("Failed to deserialize model: {}", e)))?;
        Ok(model)
    }

    /// Get mutable reference to weights for training
    pub fn weights_mut(
        &mut self,
    ) -> (
        &mut Vec<f32>,
        &mut Vec<f32>,
        &mut Vec<f32>,
        &mut Vec<f32>,
        &mut Vec<f32>,
        &mut Vec<f32>,
    ) {
        (
            &mut self.weights_1,
            &mut self.bias_1,
            &mut self.weights_2,
            &mut self.bias_2,
            &mut self.weights_3,
            &mut self.bias_3,
        )
    }

    /// Get reference to all weights (for serialization)
    #[must_use]
    pub fn weights(&self) -> (&[f32], &[f32], &[f32], &[f32], &[f32], &[f32]) {
        (
            &self.weights_1,
            &self.bias_1,
            &self.weights_2,
            &self.bias_2,
            &self.weights_3,
            &self.bias_3,
        )
    }
}

/// ReLU activation function
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Softplus activation function: log(1 + exp(x))
/// Ensures positive output, smoother than ReLU
#[inline]
fn softplus(x: f32) -> f32 {
    // Numerical stability: for large x, softplus(x) ≈ x
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
#[path = "spectral_tests.rs"]
mod tests;
