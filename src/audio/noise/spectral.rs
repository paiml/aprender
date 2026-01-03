//! Spectral MLP for magnitude spectrum prediction

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::{NoiseError, NoiseResult};

/// Small MLP for predicting noise magnitude spectra
/// Architecture: [config_dim] -> [hidden_dim] -> [hidden_dim] -> [n_freqs]
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
mod tests {
    use super::*;

    // ========== NG4: Forward pass is deterministic ==========

    #[test]
    fn test_ng4_forward_deterministic() {
        let model = SpectralMLP::random_init(8, 64, 513, 42);
        let config = vec![0.5, 0.3, 0.0, 0.1, 0.0, 1.0, 0.5, 0.9];

        let output1 = model.forward(&config);
        let output2 = model.forward(&config);

        assert_eq!(output1.len(), output2.len());
        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "Forward pass not deterministic: {} != {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_ng4_forward_deterministic_multiple_calls() {
        let model = SpectralMLP::random_init(8, 64, 513, 42);

        for i in 0..10 {
            let config = vec![i as f32 / 10.0; 8];
            let output1 = model.forward(&config);
            let output2 = model.forward(&config);

            for (a, b) in output1.iter().zip(output2.iter()) {
                assert!((a - b).abs() < f32::EPSILON);
            }
        }
    }

    #[test]
    fn test_ng4_same_seed_same_model() {
        let model1 = SpectralMLP::random_init(8, 64, 513, 42);
        let model2 = SpectralMLP::random_init(8, 64, 513, 42);

        let config = vec![0.5; 8];
        let output1 = model1.forward(&config);
        let output2 = model2.forward(&config);

        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_ng4_different_seed_different_output() {
        let model1 = SpectralMLP::random_init(8, 64, 513, 42);
        let model2 = SpectralMLP::random_init(8, 64, 513, 43);

        let config = vec![0.5; 8];
        let output1 = model1.forward(&config);
        let output2 = model2.forward(&config);

        // At least some values should differ
        let mut all_same = true;
        for (a, b) in output1.iter().zip(output2.iter()) {
            if (a - b).abs() > f32::EPSILON {
                all_same = false;
                break;
            }
        }
        assert!(
            !all_same,
            "Different seeds should produce different outputs"
        );
    }

    // ========== NG5: Output dimensions match n_freqs ==========

    #[test]
    fn test_ng5_output_dimensions_513() {
        let model = SpectralMLP::random_init(8, 64, 513, 42);
        let config = vec![0.5; 8];
        let output = model.forward(&config);
        assert_eq!(output.len(), 513, "Output should have 513 frequency bins");
    }

    #[test]
    fn test_ng5_output_dimensions_256() {
        let model = SpectralMLP::random_init(8, 32, 256, 42);
        let config = vec![0.5; 8];
        let output = model.forward(&config);
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_ng5_output_dimensions_1024() {
        let model = SpectralMLP::random_init(8, 64, 1024, 42);
        let config = vec![0.5; 8];
        let output = model.forward(&config);
        assert_eq!(output.len(), 1024);
    }

    #[test]
    fn test_ng5_config_dim_accessor() {
        let model = SpectralMLP::random_init(8, 64, 513, 42);
        assert_eq!(model.config_dim(), 8);
    }

    #[test]
    fn test_ng5_hidden_dim_accessor() {
        let model = SpectralMLP::random_init(8, 64, 513, 42);
        assert_eq!(model.hidden_dim(), 64);
    }

    #[test]
    fn test_ng5_n_freqs_accessor() {
        let model = SpectralMLP::random_init(8, 64, 513, 42);
        assert_eq!(model.n_freqs(), 513);
    }

    // ========== NG6: All outputs are non-negative (magnitudes) ==========

    #[test]
    fn test_ng6_outputs_non_negative() {
        let model = SpectralMLP::random_init(8, 64, 513, 42);

        // Test with various inputs
        for i in 0..100 {
            let config: Vec<f32> = (0..8)
                .map(|j| ((i + j) as f32 / 100.0) * 2.0 - 1.0)
                .collect();

            let output = model.forward(&config);

            for (idx, &val) in output.iter().enumerate() {
                assert!(
                    val >= 0.0,
                    "Output[{}] = {} is negative (input {})",
                    idx,
                    val,
                    i
                );
            }
        }
    }

    #[test]
    fn test_ng6_outputs_non_negative_negative_inputs() {
        let model = SpectralMLP::random_init(8, 64, 513, 42);

        // All negative inputs
        let config = vec![-1.0, -0.5, -0.8, -0.3, -1.0, -0.2, -0.7, -0.9];
        let output = model.forward(&config);

        for &val in &output {
            assert!(
                val >= 0.0,
                "Output should be non-negative even with negative inputs"
            );
        }
    }

    #[test]
    fn test_ng6_outputs_non_negative_zero_inputs() {
        let model = SpectralMLP::random_init(8, 64, 513, 42);

        let config = vec![0.0; 8];
        let output = model.forward(&config);

        for &val in &output {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_ng6_softplus_properties() {
        // Softplus should be positive for any input
        assert!(softplus(-100.0) >= 0.0);
        assert!(softplus(0.0) > 0.0);
        assert!(softplus(100.0) > 0.0);

        // Softplus(0) ≈ ln(2)
        assert!((softplus(0.0) - 2.0_f32.ln()).abs() < 0.001);
    }

    // ========== NG7: APR round-trip preserves weights exactly ==========

    #[test]
    fn test_ng7_apr_roundtrip() {
        let original = SpectralMLP::random_init(8, 64, 513, 42);

        // Save to temp file
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_ng7_spectral_mlp.apr");

        original.save_apr(&path).expect("Failed to save");
        let loaded = SpectralMLP::load_apr(&path).expect("Failed to load");

        // Clean up
        std::fs::remove_file(&path).ok();

        // Compare dimensions
        assert_eq!(original.config_dim, loaded.config_dim);
        assert_eq!(original.hidden_dim, loaded.hidden_dim);
        assert_eq!(original.n_freqs, loaded.n_freqs);

        // Compare weights exactly
        assert_eq!(original.weights_1, loaded.weights_1);
        assert_eq!(original.bias_1, loaded.bias_1);
        assert_eq!(original.weights_2, loaded.weights_2);
        assert_eq!(original.bias_2, loaded.bias_2);
        assert_eq!(original.weights_3, loaded.weights_3);
        assert_eq!(original.bias_3, loaded.bias_3);
    }

    #[test]
    fn test_ng7_apr_roundtrip_output_identical() {
        let original = SpectralMLP::random_init(8, 64, 513, 42);

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_ng7_output_check.apr");

        original.save_apr(&path).expect("Failed to save");
        let loaded = SpectralMLP::load_apr(&path).expect("Failed to load");

        std::fs::remove_file(&path).ok();

        // Test that outputs are identical
        let config = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let output_original = original.forward(&config);
        let output_loaded = loaded.forward(&config);

        for (a, b) in output_original.iter().zip(output_loaded.iter()) {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "Outputs differ after roundtrip"
            );
        }
    }

    #[test]
    fn test_ng7_load_nonexistent_file() {
        let result = SpectralMLP::load_apr("/nonexistent/path/model.apr");
        assert!(result.is_err());
    }

    // ========== Additional spectral tests ==========

    #[test]
    fn test_from_weights_valid() {
        let config_dim = 4;
        let hidden_dim = 8;
        let n_freqs = 16;

        let result = SpectralMLP::from_weights(
            vec![0.0; config_dim * hidden_dim],
            vec![0.0; hidden_dim],
            vec![0.0; hidden_dim * hidden_dim],
            vec![0.0; hidden_dim],
            vec![0.0; hidden_dim * n_freqs],
            vec![0.0; n_freqs],
            config_dim,
            hidden_dim,
            n_freqs,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_from_weights_invalid_w1() {
        let result = SpectralMLP::from_weights(
            vec![0.0; 10], // Wrong size
            vec![0.0; 8],
            vec![0.0; 64],
            vec![0.0; 8],
            vec![0.0; 128],
            vec![0.0; 16],
            4,
            8,
            16,
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("weights_1"));
    }

    #[test]
    fn test_from_weights_invalid_b1() {
        let result = SpectralMLP::from_weights(
            vec![0.0; 32],
            vec![0.0; 5], // Wrong size
            vec![0.0; 64],
            vec![0.0; 8],
            vec![0.0; 128],
            vec![0.0; 16],
            4,
            8,
            16,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_num_parameters() {
        let model = SpectralMLP::random_init(8, 64, 513, 42);
        let expected = 8 * 64 + 64 + 64 * 64 + 64 + 64 * 513 + 513;
        assert_eq!(model.num_parameters(), expected);
    }

    #[test]
    fn test_relu() {
        assert!((relu(5.0) - 5.0).abs() < f32::EPSILON);
        assert!((relu(-5.0) - 0.0).abs() < f32::EPSILON);
        assert!((relu(0.0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_softplus_numerical_stability() {
        // Very large positive
        assert!((softplus(100.0) - 100.0).abs() < 0.01);
        // Very large negative
        assert!(softplus(-100.0) < 0.01);
        // Normal range
        assert!(softplus(1.0) > 1.0 && softplus(1.0) < 2.0);
    }

    #[test]
    fn test_weights_accessors() {
        let mut model = SpectralMLP::random_init(8, 64, 513, 42);

        // Test immutable access
        let (w1, b1, w2, b2, w3, b3) = model.weights();
        assert_eq!(w1.len(), 8 * 64);
        assert_eq!(b1.len(), 64);
        assert_eq!(w2.len(), 64 * 64);
        assert_eq!(b2.len(), 64);
        assert_eq!(w3.len(), 64 * 513);
        assert_eq!(b3.len(), 513);

        // Test mutable access
        let (w1_mut, _, _, _, _, _) = model.weights_mut();
        w1_mut[0] = 999.0;
        assert!((model.weights_1[0] - 999.0).abs() < f32::EPSILON);
    }

    #[test]
    #[should_panic(expected = "Config dimension mismatch")]
    fn test_forward_wrong_config_dim() {
        let model = SpectralMLP::random_init(8, 64, 513, 42);
        let wrong_config = vec![0.5; 4]; // Wrong dimension
        model.forward(&wrong_config);
    }
}
