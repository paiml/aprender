//! Training infrastructure for noise generator models

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::config::{NoiseConfig, NoiseType};
use super::spectral::SpectralMLP;

/// Training result containing metrics and history
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final loss value
    pub final_loss: f32,
    /// Number of epochs trained
    pub epochs: usize,
    /// Loss history per epoch
    pub loss_history: Vec<f32>,
}

/// Trainer for spectral MLP noise models
pub struct NoiseTrainer {
    model: SpectralMLP,
    learning_rate: f32,
    rng: SmallRng,
}

impl std::fmt::Debug for NoiseTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoiseTrainer")
            .field("model", &self.model)
            .field("learning_rate", &self.learning_rate)
            .finish_non_exhaustive()
    }
}

impl NoiseTrainer {
    /// Create a new trainer with the given model
    #[must_use]
    pub fn new(model: SpectralMLP) -> Self {
        Self {
            model,
            learning_rate: 0.01,
            rng: SmallRng::seed_from_u64(42),
        }
    }

    /// Set learning rate
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set random seed for reproducibility
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = SmallRng::seed_from_u64(seed);
    }

    /// Generate target spectrum using analytical DSP formula
    /// For colored noise: `magnitude[f] ∝ f^(slope/6)`
    /// - Brown noise (slope=-6): magnitude ∝ 1/f (low frequencies emphasized)
    /// - Blue noise (slope=+3): magnitude ∝ sqrt(f) (high frequencies emphasized)
    #[must_use]
    pub fn generate_target_spectrum(noise_type: NoiseType, n_freqs: usize) -> Vec<f32> {
        let slope = noise_type.spectral_slope();
        // Convert dB/octave slope to frequency power exponent
        // For magnitude: exponent = slope / 6 (since dB/octave = 3*log2 and mag^2 = power)
        let exponent = slope / 6.0;

        let mut spectrum = Vec::with_capacity(n_freqs);

        for i in 0..n_freqs {
            let freq = (i + 1) as f32; // Avoid division by zero at DC
            let magnitude = if exponent.abs() < 0.001 {
                // White noise: flat spectrum
                1.0
            } else {
                // Colored noise: power law
                // Positive slope (blue/violet) → exponent > 0 → higher freqs louder
                // Negative slope (pink/brown) → exponent < 0 → lower freqs louder
                freq.powf(exponent)
            };
            spectrum.push(magnitude);
        }

        // Normalize to reasonable range
        let max_mag = spectrum.iter().cloned().fold(0.0f32, f32::max);
        if max_mag > 0.0 {
            for m in &mut spectrum {
                *m /= max_mag;
            }
        }

        spectrum
    }

    /// Encode a noise config into model input
    fn encode_config(&self, config: &NoiseConfig) -> Vec<f32> {
        config.encode(0.0) // Use time=0 for training
    }

    /// Single training step with manual backprop
    /// Returns the loss for this batch
    pub fn train_step(&mut self, configs: &[NoiseConfig], targets: &[Vec<f32>]) -> f32 {
        assert_eq!(configs.len(), targets.len());
        if configs.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let n_freqs = self.model.n_freqs();
        let hidden_dim = self.model.hidden_dim();
        let config_dim = self.model.config_dim();

        // Initialize gradient accumulators
        let mut grad_w3 = vec![0.0; hidden_dim * n_freqs];
        let mut grad_b3 = vec![0.0; n_freqs];
        let mut grad_w2 = vec![0.0; hidden_dim * hidden_dim];
        let mut grad_b2 = vec![0.0; hidden_dim];
        let mut grad_w1 = vec![0.0; config_dim * hidden_dim];
        let mut grad_b1 = vec![0.0; hidden_dim];

        for (config, target) in configs.iter().zip(targets.iter()) {
            let input = self.encode_config(config);
            let (h1, h2, output) = self.forward_with_cache(&input);

            total_loss += spectral_loss(&output, target);

            let d_out = compute_output_gradient(&output, target, n_freqs);
            let d_z3 = apply_softplus_derivative(&output, &d_out);

            let (_w1, _b1, w2, _b2, w3, _b3) = self.model.weights();

            accumulate_layer_grads(&d_z3, &h2, &mut grad_w3, &mut grad_b3, n_freqs, hidden_dim);

            let d_h2 = backprop_delta(&d_z3, w3, n_freqs, hidden_dim);
            let d_z2 = apply_relu_derivative(&d_h2, &h2);

            accumulate_layer_grads(
                &d_z2,
                &h1,
                &mut grad_w2,
                &mut grad_b2,
                hidden_dim,
                hidden_dim,
            );

            let d_h1 = backprop_delta(&d_z2, w2, hidden_dim, hidden_dim);
            let d_z1 = apply_relu_derivative(&d_h1, &h1);

            accumulate_layer_grads(
                &d_z1,
                &input,
                &mut grad_w1,
                &mut grad_b1,
                hidden_dim,
                config_dim,
            );
        }

        // Average and apply gradients
        let batch_size = configs.len() as f32;
        for grads in [
            &mut grad_w1,
            &mut grad_b1,
            &mut grad_w2,
            &mut grad_b2,
            &mut grad_w3,
            &mut grad_b3,
        ] {
            for g in grads.iter_mut() {
                *g /= batch_size;
            }
        }
        self.apply_gradients(&grad_w1, &grad_b1, &grad_w2, &grad_b2, &grad_w3, &grad_b3);

        total_loss / batch_size
    }

    /// Forward pass with cached activations for backprop
    fn forward_with_cache(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (w1, b1, w2, b2, w3, b3) = self.model.weights();
        let hidden_dim = self.model.hidden_dim();
        let n_freqs = self.model.n_freqs();
        let config_dim = self.model.config_dim();

        // Layer 1
        let mut h1 = vec![0.0; hidden_dim];
        for i in 0..hidden_dim {
            let mut sum = b1[i];
            for j in 0..config_dim {
                sum += w1[j * hidden_dim + i] * input[j];
            }
            h1[i] = sum.max(0.0); // ReLU
        }

        // Layer 2
        let mut h2 = vec![0.0; hidden_dim];
        for i in 0..hidden_dim {
            let mut sum = b2[i];
            for j in 0..hidden_dim {
                sum += w2[j * hidden_dim + i] * h1[j];
            }
            h2[i] = sum.max(0.0); // ReLU
        }

        // Layer 3
        let mut output = vec![0.0; n_freqs];
        for i in 0..n_freqs {
            let mut sum = b3[i];
            for j in 0..hidden_dim {
                sum += w3[j * n_freqs + i] * h2[j];
            }
            // Softplus
            output[i] = if sum > 20.0 {
                sum
            } else if sum < -20.0 {
                0.0
            } else {
                (1.0 + sum.exp()).ln()
            };
        }

        (h1, h2, output)
    }

    /// Full training loop
    pub fn train(&mut self, epochs: usize) -> TrainingResult {
        let n_freqs = self.model.n_freqs();
        let mut loss_history = Vec::with_capacity(epochs);

        // Generate training data
        let noise_types = [
            NoiseType::White,
            NoiseType::Pink,
            NoiseType::Brown,
            NoiseType::Blue,
            NoiseType::Violet,
        ];

        let mut configs: Vec<NoiseConfig> =
            noise_types.iter().map(|&nt| NoiseConfig::new(nt)).collect();

        // Add random custom slopes
        for _ in 0..10 {
            let slope = self.rng.gen_range(-12.0..12.0);
            configs.push(NoiseConfig::new(NoiseType::Custom(slope)));
        }

        let targets: Vec<Vec<f32>> = configs
            .iter()
            .map(|c| Self::generate_target_spectrum(c.noise_type, n_freqs))
            .collect();

        for _epoch in 0..epochs {
            let loss = self.train_step(&configs, &targets);
            loss_history.push(loss);
        }

        let final_loss = loss_history.last().cloned().unwrap_or(0.0);

        TrainingResult {
            final_loss,
            epochs,
            loss_history,
        }
    }

    /// Get the trained model
    #[must_use]
    pub fn into_model(self) -> SpectralMLP {
        self.model
    }

    /// Get reference to current model
    #[must_use]
    pub fn model(&self) -> &SpectralMLP {
        &self.model
    }
}

/// Compute gradient of spectral loss w.r.t. output
fn compute_output_gradient(output: &[f32], target: &[f32], n_freqs: usize) -> Vec<f32> {
    (0..n_freqs)
        .map(|i| {
            let weight = 1.0 / (1.0 + i as f32 * 0.01);
            let p_log = (output[i] + 1e-10).ln();
            let t_log = (target[i] + 1e-10).ln();
            2.0 * weight * (p_log - t_log) / (output[i] + 1e-10) / n_freqs as f32
        })
        .collect()
}

/// Apply softplus derivative: d_softplus/dx = sigmoid(x)
fn apply_softplus_derivative(output: &[f32], d_out: &[f32]) -> Vec<f32> {
    output
        .iter()
        .zip(d_out.iter())
        .map(|(&o, &d)| {
            if o > 20.0 {
                d
            } else {
                d * (1.0 - (-o).exp().min(1.0))
            }
        })
        .collect()
}

/// Apply ReLU derivative: pass gradient where activation > 0
fn apply_relu_derivative(d_h: &[f32], h: &[f32]) -> Vec<f32> {
    d_h.iter()
        .zip(h.iter())
        .map(|(&d, &h_val)| if h_val > 0.0 { d } else { 0.0 })
        .collect()
}

/// Accumulate weight and bias gradients for a single layer
fn accumulate_layer_grads(
    d_z: &[f32],
    activation: &[f32],
    grad_w: &mut [f32],
    grad_b: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    for i in 0..out_dim {
        grad_b[i] += d_z[i];
        for j in 0..in_dim {
            grad_w[j * out_dim + i] += d_z[i] * activation[j];
        }
    }
}

/// Backpropagate delta through a weight matrix
fn backprop_delta(d_z: &[f32], weights: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut d_h = vec![0.0; in_dim];
    for j in 0..in_dim {
        for i in 0..out_dim {
            d_h[j] += d_z[i] * weights[j * out_dim + i];
        }
    }
    d_h
}

/// Spectral loss function with perceptual weighting
/// Compares predicted and target spectra in log domain
#[must_use]
pub fn spectral_loss(predicted: &[f32], target: &[f32]) -> f32 {
    assert_eq!(predicted.len(), target.len());
    if predicted.is_empty() {
        return 0.0;
    }

    let n = predicted.len() as f32;

    predicted
        .iter()
        .zip(target.iter())
        .enumerate()
        .map(|(i, (p, t))| {
            let weight = 1.0 / (1.0 + i as f32 * 0.01); // Low-freq emphasis
            let p_log = (*p + 1e-10).ln();
            let t_log = (*t + 1e-10).ln();
            weight * (p_log - t_log).powi(2)
        })
        .sum::<f32>()
        / n
}

#[cfg(test)]
#[path = "train_tests.rs"]
mod tests;
