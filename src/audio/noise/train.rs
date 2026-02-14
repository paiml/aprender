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

        // Accumulate gradients
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

            // Forward pass with intermediate activations
            let (h1, h2, output) = self.forward_with_cache(&input);

            // Compute loss
            let loss = spectral_loss(&output, target);
            total_loss += loss;

            // Backward pass
            // dL/d_output (gradient of spectral_loss)
            let mut d_out = Vec::with_capacity(n_freqs);
            for i in 0..n_freqs {
                let weight = 1.0 / (1.0 + i as f32 * 0.01);
                let p_log = (output[i] + 1e-10).ln();
                let t_log = (target[i] + 1e-10).ln();
                // d/dp of weight * (ln(p) - ln(t))^2 = 2 * weight * (ln(p) - ln(t)) / p
                let grad = 2.0 * weight * (p_log - t_log) / (output[i] + 1e-10) / n_freqs as f32;
                d_out.push(grad);
            }

            // Through softplus: d_softplus/dx = sigmoid(x)
            // For output[i] = softplus(z[i]), we need z[i]
            // Since softplus(z) = ln(1 + exp(z)), we have z = ln(exp(output) - 1)
            // But numerically: d_softplus = 1 - exp(-output) for output > 0
            let d_z3: Vec<f32> = output
                .iter()
                .zip(d_out.iter())
                .map(|(&o, &d)| {
                    if o > 20.0 {
                        d // Derivative is 1 for large values
                    } else {
                        d * (1.0 - (-o).exp().min(1.0))
                    }
                })
                .collect();

            // Gradient for W3 and b3
            let (_w1, _b1, w2, _b2, w3, _b3) = self.model.weights();

            for i in 0..n_freqs {
                grad_b3[i] += d_z3[i];
                for j in 0..hidden_dim {
                    grad_w3[j * n_freqs + i] += d_z3[i] * h2[j];
                }
            }

            // Backprop through layer 2
            let mut d_h2 = vec![0.0; hidden_dim];
            for j in 0..hidden_dim {
                for i in 0..n_freqs {
                    d_h2[j] += d_z3[i] * w3[j * n_freqs + i];
                }
            }

            // Through ReLU
            let d_z2: Vec<f32> = d_h2
                .iter()
                .zip(h2.iter())
                .map(|(&d, &h)| if h > 0.0 { d } else { 0.0 })
                .collect();

            // Gradient for W2 and b2
            for i in 0..hidden_dim {
                grad_b2[i] += d_z2[i];
                for j in 0..hidden_dim {
                    grad_w2[j * hidden_dim + i] += d_z2[i] * h1[j];
                }
            }

            // Backprop through layer 1
            let mut d_h1 = vec![0.0; hidden_dim];
            for j in 0..hidden_dim {
                for i in 0..hidden_dim {
                    d_h1[j] += d_z2[i] * w2[j * hidden_dim + i];
                }
            }

            // Through ReLU
            let d_z1: Vec<f32> = d_h1
                .iter()
                .zip(h1.iter())
                .map(|(&d, &h)| if h > 0.0 { d } else { 0.0 })
                .collect();

            // Gradient for W1 and b1
            for i in 0..hidden_dim {
                grad_b1[i] += d_z1[i];
                for j in 0..config_dim {
                    grad_w1[j * hidden_dim + i] += d_z1[i] * input[j];
                }
            }
        }

        // Average gradients
        let batch_size = configs.len() as f32;
        for g in &mut grad_w3 {
            *g /= batch_size;
        }
        for g in &mut grad_b3 {
            *g /= batch_size;
        }
        for g in &mut grad_w2 {
            *g /= batch_size;
        }
        for g in &mut grad_b2 {
            *g /= batch_size;
        }
        for g in &mut grad_w1 {
            *g /= batch_size;
        }
        for g in &mut grad_b1 {
            *g /= batch_size;
        }

        // Apply gradients
        let (w1_mut, b1_mut, w2_mut, b2_mut, w3_mut, b3_mut) = self.model.weights_mut();

        for (w, g) in w1_mut.iter_mut().zip(grad_w1.iter()) {
            *w -= self.learning_rate * g;
        }
        for (b, g) in b1_mut.iter_mut().zip(grad_b1.iter()) {
            *b -= self.learning_rate * g;
        }
        for (w, g) in w2_mut.iter_mut().zip(grad_w2.iter()) {
            *w -= self.learning_rate * g;
        }
        for (b, g) in b2_mut.iter_mut().zip(grad_b2.iter()) {
            *b -= self.learning_rate * g;
        }
        for (w, g) in w3_mut.iter_mut().zip(grad_w3.iter()) {
            *w -= self.learning_rate * g;
        }
        for (b, g) in b3_mut.iter_mut().zip(grad_b3.iter()) {
            *b -= self.learning_rate * g;
        }

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
