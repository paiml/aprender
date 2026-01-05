//! Variational Autoencoder (VAE) module.
//!
//! Implements the VAE architecture from Kingma & Welling (2014).
//!
//! # Architecture
//!
//! ```text
//! Input x → Encoder → (μ, log σ²) → z = μ + σ * ε → Decoder → x̂
//!                                    ↑
//!                              ε ~ N(0, I)
//! ```
//!
//! # Loss Function
//!
//! ```text
//! L = E[log p(x|z)] - KL(q(z|x) || p(z))
//!   ≈ Reconstruction Loss + KL Divergence
//! ```
//!
//! # Example
//!
//! ```ignore
//! use aprender::nn::vae::{VAE, VAEOutput};
//!
//! // Create VAE with 784 input, 256 hidden, 20 latent dimensions
//! let vae = VAE::new(784, vec![256], 20);
//!
//! // Forward pass
//! let x = Tensor::randn(&[32, 784]);  // batch of 32
//! let output = vae.forward_vae(&x);
//!
//! // Compute loss
//! let loss = vae.loss(&output, &x);
//! ```
//!
//! # References
//!
//! - Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.
//!   ICLR.

use crate::autograd::Tensor;
use crate::nn::{Linear, Module, ReLU};

/// Output from VAE forward pass.
#[derive(Debug)]
pub struct VAEOutput {
    /// Reconstructed output
    pub reconstruction: Tensor,
    /// Mean of latent distribution
    pub mu: Tensor,
    /// Log variance of latent distribution
    pub log_var: Tensor,
    /// Sampled latent vector
    pub z: Tensor,
}

/// Variational Autoencoder (VAE).
///
/// Learns a latent representation by encoding inputs to a
/// distribution in latent space and decoding samples from it.
///
/// # Example
///
/// ```ignore
/// let vae = VAE::new(784, vec![512, 256], 32);
/// let output = vae.forward_vae(&input);
/// let loss = vae.loss(&output, &input);
/// ```
pub struct VAE {
    // Encoder layers
    encoder_layers: Vec<Linear>,
    encoder_activation: ReLU,
    // Latent space projections
    fc_mu: Linear,
    fc_log_var: Linear,
    // Decoder layers
    decoder_layers: Vec<Linear>,
    decoder_activation: ReLU,
    output_layer: Linear,

    input_dim: usize,
    latent_dim: usize,
    hidden_dims: Vec<usize>,
    training: bool,
    beta: f32, // KL weight for β-VAE
}

impl VAE {
    /// Create a new VAE.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Dimension of input data
    /// * `hidden_dims` - Dimensions of hidden layers
    /// * `latent_dim` - Dimension of latent space
    ///
    /// # Example
    ///
    /// ```ignore
    /// let vae = VAE::new(784, vec![512, 256], 32);
    /// ```
    #[must_use]
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, latent_dim: usize) -> Self {
        // Build encoder
        let mut encoder_layers = Vec::new();
        let mut prev_dim = input_dim;

        for &hidden_dim in &hidden_dims {
            encoder_layers.push(Linear::new(prev_dim, hidden_dim));
            prev_dim = hidden_dim;
        }

        let last_hidden = *hidden_dims.last().unwrap_or(&input_dim);

        // Latent space projections
        let fc_mu = Linear::new(last_hidden, latent_dim);
        let fc_log_var = Linear::new(last_hidden, latent_dim);

        // Build decoder (reverse of encoder)
        let mut decoder_layers = Vec::new();
        prev_dim = latent_dim;

        for &hidden_dim in hidden_dims.iter().rev() {
            decoder_layers.push(Linear::new(prev_dim, hidden_dim));
            prev_dim = hidden_dim;
        }

        let output_layer = Linear::new(prev_dim, input_dim);

        Self {
            encoder_layers,
            encoder_activation: ReLU::new(),
            fc_mu,
            fc_log_var,
            decoder_layers,
            decoder_activation: ReLU::new(),
            output_layer,
            input_dim,
            latent_dim,
            hidden_dims,
            training: true,
            beta: 1.0,
        }
    }

    /// Set β for β-VAE (controls KL weight).
    ///
    /// β > 1 encourages disentangled representations.
    #[must_use]
    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Encode input to latent distribution parameters.
    ///
    /// # Returns
    ///
    /// Tuple of (mean, `log_variance`)
    #[must_use]
    pub fn encode(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mut h = x.clone();

        // Pass through encoder layers
        for layer in &self.encoder_layers {
            h = layer.forward(&h);
            h = self.encoder_activation.forward(&h);
        }

        // Project to latent parameters
        let mu = self.fc_mu.forward(&h);
        let log_var = self.fc_log_var.forward(&h);

        (mu, log_var)
    }

    /// Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I).
    ///
    /// Enables backpropagation through sampling.
    #[must_use]
    pub fn reparameterize(&self, mu: &Tensor, log_var: &Tensor) -> Tensor {
        if !self.training {
            // At test time, just use the mean
            return mu.clone();
        }

        // Sample ε ~ N(0, I)
        let epsilon = sample_standard_normal(mu.shape());

        // σ = exp(0.5 * log_var)
        let std = exp_half(log_var);

        // z = μ + σ * ε
        add_mul(mu, &std, &epsilon)
    }

    /// Decode latent vector to reconstruction.
    #[must_use]
    pub fn decode(&self, z: &Tensor) -> Tensor {
        let mut h = z.clone();

        // Pass through decoder layers
        for layer in &self.decoder_layers {
            h = layer.forward(&h);
            h = self.decoder_activation.forward(&h);
        }

        // Output layer (sigmoid for binary data, linear for continuous)
        self.output_layer.forward(&h)
    }

    /// Full forward pass through VAE.
    #[must_use]
    pub fn forward_vae(&self, x: &Tensor) -> VAEOutput {
        // Encode
        let (mu, log_var) = self.encode(x);

        // Sample latent
        let z = self.reparameterize(&mu, &log_var);

        // Decode
        let reconstruction = self.decode(&z);

        VAEOutput {
            reconstruction,
            mu,
            log_var,
            z,
        }
    }

    /// Compute VAE loss = Reconstruction + β * KL.
    ///
    /// # Arguments
    ///
    /// * `output` - VAE forward pass output
    /// * `target` - Original input
    ///
    /// # Returns
    ///
    /// Tuple of (`total_loss`, `reconstruction_loss`, `kl_loss`)
    #[must_use]
    pub fn loss(&self, output: &VAEOutput, target: &Tensor) -> (f32, f32, f32) {
        // Reconstruction loss (MSE)
        let recon_loss = mse_loss(&output.reconstruction, target);

        // KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        let kl_loss = kl_divergence_loss(&output.mu, &output.log_var);

        let total_loss = recon_loss + self.beta * kl_loss;

        (total_loss, recon_loss, kl_loss)
    }

    /// Sample from the latent space.
    #[must_use]
    pub fn sample(&self, num_samples: usize) -> Tensor {
        let z = sample_standard_normal(&[num_samples, self.latent_dim]);
        self.decode(&z)
    }

    /// Interpolate between two points in latent space.
    #[must_use]
    pub fn interpolate(&self, x1: &Tensor, x2: &Tensor, steps: usize) -> Vec<Tensor> {
        let (mu1, _) = self.encode(x1);
        let (mu2, _) = self.encode(x2);

        let mut results = Vec::with_capacity(steps);

        for i in 0..steps {
            let alpha = i as f32 / (steps - 1) as f32;
            let z = lerp(&mu1, &mu2, alpha);
            results.push(self.decode(&z));
        }

        results
    }

    /// Get latent dimension.
    #[must_use]
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Get input dimension.
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get β value.
    #[must_use]
    pub fn beta(&self) -> f32 {
        self.beta
    }
}

impl Module for VAE {
    fn forward(&self, input: &Tensor) -> Tensor {
        let output = self.forward_vae(input);
        output.reconstruction
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();

        for layer in &self.encoder_layers {
            params.extend(layer.parameters());
        }
        params.extend(self.fc_mu.parameters());
        params.extend(self.fc_log_var.parameters());

        for layer in &self.decoder_layers {
            params.extend(layer.parameters());
        }
        params.extend(self.output_layer.parameters());

        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();

        for layer in &mut self.encoder_layers {
            params.extend(layer.parameters_mut());
        }
        params.extend(self.fc_mu.parameters_mut());
        params.extend(self.fc_log_var.parameters_mut());

        for layer in &mut self.decoder_layers {
            params.extend(layer.parameters_mut());
        }
        params.extend(self.output_layer.parameters_mut());

        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for VAE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VAE")
            .field("input_dim", &self.input_dim)
            .field("hidden_dims", &self.hidden_dims)
            .field("latent_dim", &self.latent_dim)
            .field("beta", &self.beta)
            .finish_non_exhaustive()
    }
}

/// Conditional VAE (CVAE) for class-conditioned generation.
///
/// Concatenates class label to encoder input and uses separate decoder
/// that accepts latent + class.
pub struct ConditionalVAE {
    // Encoder layers (input + class -> hidden)
    encoder_layers: Vec<Linear>,
    encoder_activation: ReLU,
    fc_mu: Linear,
    fc_log_var: Linear,

    // Decoder layers (latent + class -> hidden -> output)
    decoder_layers: Vec<Linear>,
    decoder_activation: ReLU,
    output_layer: Linear,

    input_dim: usize,
    latent_dim: usize,
    num_classes: usize,
    hidden_dims: Vec<usize>,
    training: bool,
}

impl ConditionalVAE {
    /// Create a new CVAE.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Dimension of input data
    /// * `num_classes` - Number of classes for conditioning
    /// * `hidden_dims` - Dimensions of hidden layers
    /// * `latent_dim` - Dimension of latent space
    #[must_use]
    pub fn new(
        input_dim: usize,
        num_classes: usize,
        hidden_dims: Vec<usize>,
        latent_dim: usize,
    ) -> Self {
        // Build encoder (input + class -> hidden)
        let mut encoder_layers = Vec::new();
        let mut prev_dim = input_dim + num_classes;

        for &hidden_dim in &hidden_dims {
            encoder_layers.push(Linear::new(prev_dim, hidden_dim));
            prev_dim = hidden_dim;
        }

        let last_hidden = *hidden_dims.last().unwrap_or(&(input_dim + num_classes));
        let fc_mu = Linear::new(last_hidden, latent_dim);
        let fc_log_var = Linear::new(last_hidden, latent_dim);

        // Build decoder (latent + class -> hidden -> output)
        let mut decoder_layers = Vec::new();
        prev_dim = latent_dim + num_classes;

        for &hidden_dim in hidden_dims.iter().rev() {
            decoder_layers.push(Linear::new(prev_dim, hidden_dim));
            prev_dim = hidden_dim;
        }

        let output_layer = Linear::new(prev_dim, input_dim);

        Self {
            encoder_layers,
            encoder_activation: ReLU::new(),
            fc_mu,
            fc_log_var,
            decoder_layers,
            decoder_activation: ReLU::new(),
            output_layer,
            input_dim,
            latent_dim,
            num_classes,
            hidden_dims,
            training: true,
        }
    }

    /// Encode with class label.
    #[must_use]
    pub fn encode(&self, x: &Tensor, class_label: usize) -> (Tensor, Tensor) {
        let x_cond = concat_one_hot(x, class_label, self.num_classes);

        let mut h = x_cond;
        for layer in &self.encoder_layers {
            h = layer.forward(&h);
            h = self.encoder_activation.forward(&h);
        }

        let mu = self.fc_mu.forward(&h);
        let log_var = self.fc_log_var.forward(&h);

        (mu, log_var)
    }

    /// Decode with class label.
    #[must_use]
    pub fn decode(&self, z: &Tensor, class_label: usize) -> Tensor {
        let z_cond = concat_one_hot(z, class_label, self.num_classes);

        let mut h = z_cond;
        for layer in &self.decoder_layers {
            h = layer.forward(&h);
            h = self.decoder_activation.forward(&h);
        }

        self.output_layer.forward(&h)
    }

    /// Reparameterization trick.
    fn reparameterize(&self, mu: &Tensor, log_var: &Tensor) -> Tensor {
        if !self.training {
            return mu.clone();
        }
        let epsilon = sample_standard_normal(mu.shape());
        let std = exp_half(log_var);
        add_mul(mu, &std, &epsilon)
    }

    /// Full forward pass.
    #[must_use]
    pub fn forward_cvae(&self, x: &Tensor, class_label: usize) -> VAEOutput {
        let (mu, log_var) = self.encode(x, class_label);
        let z = self.reparameterize(&mu, &log_var);
        let reconstruction = self.decode(&z, class_label);

        VAEOutput {
            reconstruction,
            mu,
            log_var,
            z,
        }
    }

    /// Sample with class label.
    #[must_use]
    pub fn sample(&self, num_samples: usize, class_label: usize) -> Tensor {
        let z = sample_standard_normal(&[num_samples, self.latent_dim]);
        self.decode(&z, class_label)
    }

    /// Get number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Get latent dimension.
    #[must_use]
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Get input dimension.
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
}

impl std::fmt::Debug for ConditionalVAE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConditionalVAE")
            .field("input_dim", &self.input_dim)
            .field("latent_dim", &self.latent_dim)
            .field("num_classes", &self.num_classes)
            .field("hidden_dims", &self.hidden_dims)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Sample from standard normal distribution N(0, I).
fn sample_standard_normal(shape: &[usize]) -> Tensor {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let size: usize = shape.iter().product();
    let data: Vec<f32> = (0..size)
        .map(|_| {
            // Box-Muller transform for normal distribution
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();

    Tensor::new(&data, shape)
}

/// Compute exp(0.5 * x) for standard deviation from log variance.
fn exp_half(log_var: &Tensor) -> Tensor {
    let data: Vec<f32> = log_var.data().iter().map(|&x| (0.5 * x).exp()).collect();
    Tensor::new(&data, log_var.shape())
}

/// Compute a + b * c element-wise.
fn add_mul(a: &Tensor, b: &Tensor, c: &Tensor) -> Tensor {
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .zip(c.data().iter())
        .map(|((&ai, &bi), &ci)| ai + bi * ci)
        .collect();
    Tensor::new(&data, a.shape())
}

/// MSE loss between reconstruction and target.
fn mse_loss(pred: &Tensor, target: &Tensor) -> f32 {
    let sum_sq: f32 = pred
        .data()
        .iter()
        .zip(target.data().iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum();
    sum_sq / pred.data().len() as f32
}

/// KL divergence loss for VAE: -0.5 * sum(1 + `log_var` - mu^2 - `exp(log_var)`).
fn kl_divergence_loss(mu: &Tensor, log_var: &Tensor) -> f32 {
    let kl: f32 = mu
        .data()
        .iter()
        .zip(log_var.data().iter())
        .map(|(&m, &lv)| -0.5 * (1.0 + lv - m * m - lv.exp()))
        .sum();
    kl / mu.shape()[0] as f32 // Average over batch
}

/// Linear interpolation between two tensors.
fn lerp(a: &Tensor, b: &Tensor, alpha: f32) -> Tensor {
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&ai, &bi)| (1.0 - alpha) * ai + alpha * bi)
        .collect();
    Tensor::new(&data, a.shape())
}

/// Concatenate one-hot encoding of class label to tensor.
fn concat_one_hot(x: &Tensor, class_label: usize, num_classes: usize) -> Tensor {
    let batch_size = x.shape()[0];
    let input_dim = x.shape()[1];

    let mut data = vec![0.0; batch_size * (input_dim + num_classes)];

    for b in 0..batch_size {
        // Copy original data
        for i in 0..input_dim {
            data[b * (input_dim + num_classes) + i] = x.data()[b * input_dim + i];
        }
        // Add one-hot encoding
        data[b * (input_dim + num_classes) + input_dim + class_label] = 1.0;
    }

    Tensor::new(&data, &[batch_size, input_dim + num_classes])
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // VAE Tests
    // ========================================================================

    #[test]
    fn test_vae_creation() {
        let vae = VAE::new(784, vec![256, 128], 20);

        assert_eq!(vae.input_dim(), 784);
        assert_eq!(vae.latent_dim(), 20);
        assert_eq!(vae.beta(), 1.0);
    }

    #[test]
    fn test_vae_with_beta() {
        let vae = VAE::new(784, vec![256], 20).with_beta(4.0);
        assert_eq!(vae.beta(), 4.0);
    }

    #[test]
    fn test_vae_encode() {
        let vae = VAE::new(100, vec![64], 10);

        let x = Tensor::ones(&[4, 100]);
        let (mu, log_var) = vae.encode(&x);

        assert_eq!(mu.shape(), &[4, 10]);
        assert_eq!(log_var.shape(), &[4, 10]);
    }

    #[test]
    fn test_vae_decode() {
        let vae = VAE::new(100, vec![64], 10);

        let z = Tensor::ones(&[4, 10]);
        let reconstruction = vae.decode(&z);

        assert_eq!(reconstruction.shape(), &[4, 100]);
    }

    #[test]
    fn test_vae_forward() {
        let vae = VAE::new(100, vec![64], 10);

        let x = Tensor::ones(&[4, 100]);
        let output = vae.forward_vae(&x);

        assert_eq!(output.reconstruction.shape(), &[4, 100]);
        assert_eq!(output.mu.shape(), &[4, 10]);
        assert_eq!(output.log_var.shape(), &[4, 10]);
        assert_eq!(output.z.shape(), &[4, 10]);
    }

    #[test]
    fn test_vae_module_forward() {
        let vae = VAE::new(100, vec![64], 10);

        let x = Tensor::ones(&[4, 100]);
        let output = vae.forward(&x);

        assert_eq!(output.shape(), &[4, 100]);
    }

    #[test]
    fn test_vae_loss() {
        let vae = VAE::new(100, vec![64], 10);

        let x = Tensor::ones(&[4, 100]);
        let output = vae.forward_vae(&x);
        let (total, recon, kl) = vae.loss(&output, &x);

        // Loss should be non-negative
        assert!(recon >= 0.0);
        // KL can be any real value but typically positive
        assert!(total.is_finite());
        assert!(kl.is_finite());
    }

    #[test]
    fn test_vae_sample() {
        let vae = VAE::new(100, vec![64], 10);

        let samples = vae.sample(8);

        assert_eq!(samples.shape(), &[8, 100]);
    }

    #[test]
    fn test_vae_interpolate() {
        let vae = VAE::new(100, vec![64], 10);

        let x1 = Tensor::ones(&[1, 100]);
        let x2 = Tensor::zeros(&[1, 100]);

        let interpolations = vae.interpolate(&x1, &x2, 5);

        assert_eq!(interpolations.len(), 5);
        for interp in &interpolations {
            assert_eq!(interp.shape(), &[1, 100]);
        }
    }

    #[test]
    fn test_vae_train_eval() {
        let mut vae = VAE::new(100, vec![64], 10);

        assert!(vae.training());

        vae.eval();
        assert!(!vae.training());

        vae.train();
        assert!(vae.training());
    }

    #[test]
    fn test_vae_parameters() {
        let vae = VAE::new(100, vec![64], 10);
        let params = vae.parameters();

        // encoder_layer (64) + fc_mu + fc_log_var + decoder_layer (64) + output
        // Each linear has weight + bias = 2 params
        // 1 encoder + 2 latent + 1 decoder + 1 output = 5 layers * 2 = 10 params
        assert_eq!(params.len(), 10);
    }

    #[test]
    fn test_vae_reparameterize_training() {
        let vae = VAE::new(100, vec![64], 10);

        let mu = Tensor::zeros(&[4, 10]);
        let log_var = Tensor::zeros(&[4, 10]);

        let z = vae.reparameterize(&mu, &log_var);

        assert_eq!(z.shape(), &[4, 10]);
        // z should have some variance due to sampling
    }

    #[test]
    fn test_vae_reparameterize_eval() {
        let mut vae = VAE::new(100, vec![64], 10);
        vae.eval();

        let mu = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let log_var = Tensor::zeros(&[2, 2]);

        let z = vae.reparameterize(&mu, &log_var);

        // In eval mode, z should equal mu
        assert_eq!(z.data(), mu.data());
    }

    // ========================================================================
    // Conditional VAE Tests
    // ========================================================================

    #[test]
    fn test_cvae_creation() {
        let cvae = ConditionalVAE::new(100, 10, vec![64], 20);
        assert_eq!(cvae.num_classes(), 10);
    }

    #[test]
    fn test_cvae_encode() {
        let cvae = ConditionalVAE::new(100, 10, vec![64], 20);

        let x = Tensor::ones(&[4, 100]);
        let (mu, log_var) = cvae.encode(&x, 5);

        assert_eq!(mu.shape(), &[4, 20]);
        assert_eq!(log_var.shape(), &[4, 20]);
    }

    #[test]
    fn test_cvae_sample() {
        let cvae = ConditionalVAE::new(100, 10, vec![64], 20);

        let samples = cvae.sample(8, 3);

        // Output should be the original input dimension (not including class)
        assert_eq!(samples.shape(), &[8, 100]);
    }

    #[test]
    fn test_cvae_forward() {
        let cvae = ConditionalVAE::new(100, 10, vec![64], 20);

        let x = Tensor::ones(&[4, 100]);
        let output = cvae.forward_cvae(&x, 5);

        assert_eq!(output.reconstruction.shape(), &[4, 100]);
        assert_eq!(output.mu.shape(), &[4, 20]);
        assert_eq!(output.log_var.shape(), &[4, 20]);
        assert_eq!(output.z.shape(), &[4, 20]);
    }

    #[test]
    fn test_cvae_getters() {
        let cvae = ConditionalVAE::new(100, 10, vec![64], 20);

        assert_eq!(cvae.input_dim(), 100);
        assert_eq!(cvae.latent_dim(), 20);
        assert_eq!(cvae.num_classes(), 10);
    }

    // ========================================================================
    // Helper Function Tests
    // ========================================================================

    #[test]
    fn test_sample_standard_normal() {
        let samples = sample_standard_normal(&[1000]);

        // Mean should be close to 0
        let mean: f32 = samples.data().iter().sum::<f32>() / 1000.0;
        assert!(mean.abs() < 0.2);

        // Variance should be close to 1
        let variance: f32 = samples
            .data()
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / 1000.0;
        assert!((variance - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_exp_half() {
        let log_var = Tensor::new(&[0.0, 2.0, -2.0], &[3]);
        let std = exp_half(&log_var);

        // exp(0.5 * 0) = 1, exp(0.5 * 2) = e, exp(0.5 * -2) = 1/e
        assert!((std.data()[0] - 1.0).abs() < 1e-6);
        assert!((std.data()[1] - std::f32::consts::E).abs() < 1e-5);
        assert!((std.data()[2] - 1.0 / std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn test_add_mul() {
        let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::new(&[2.0, 2.0, 2.0], &[3]);
        let c = Tensor::new(&[1.0, 1.0, 1.0], &[3]);

        let result = add_mul(&a, &b, &c);

        // 1 + 2*1 = 3, 2 + 2*1 = 4, 3 + 2*1 = 5
        assert_eq!(result.data(), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_mse_loss() {
        let pred = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let target = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);

        let loss = mse_loss(&pred, &target);
        assert_eq!(loss, 0.0);

        let pred2 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[4]);
        let loss2 = mse_loss(&pred2, &target);
        assert_eq!(loss2, 1.0); // Each diff is 1, squared is 1, mean is 1
    }

    #[test]
    fn test_kl_divergence_loss() {
        // When mu=0 and log_var=0 (var=1), KL should be 0
        let mu = Tensor::zeros(&[2, 3]);
        let log_var = Tensor::zeros(&[2, 3]);

        let kl = kl_divergence_loss(&mu, &log_var);
        assert!(kl.abs() < 1e-6);
    }

    #[test]
    fn test_lerp() {
        let a = Tensor::new(&[0.0, 0.0], &[2]);
        let b = Tensor::new(&[10.0, 10.0], &[2]);

        let mid = lerp(&a, &b, 0.5);
        assert_eq!(mid.data(), &[5.0, 5.0]);

        let quarter = lerp(&a, &b, 0.25);
        assert_eq!(quarter.data(), &[2.5, 2.5]);
    }

    #[test]
    fn test_concat_one_hot() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = concat_one_hot(&x, 1, 3);

        assert_eq!(result.shape(), &[2, 5]); // 2 + 3 = 5

        // First sample: [1, 2, 0, 1, 0]
        assert_eq!(result.data()[0], 1.0);
        assert_eq!(result.data()[1], 2.0);
        assert_eq!(result.data()[2], 0.0);
        assert_eq!(result.data()[3], 1.0);
        assert_eq!(result.data()[4], 0.0);

        // Second sample: [3, 4, 0, 1, 0]
        assert_eq!(result.data()[5], 3.0);
        assert_eq!(result.data()[6], 4.0);
        assert_eq!(result.data()[7], 0.0);
        assert_eq!(result.data()[8], 1.0);
        assert_eq!(result.data()[9], 0.0);
    }

    #[test]
    fn test_vae_no_hidden_layers() {
        // Edge case: direct connection from input to latent
        let vae = VAE::new(100, vec![], 10);

        let x = Tensor::ones(&[4, 100]);
        let output = vae.forward_vae(&x);

        assert_eq!(output.reconstruction.shape(), &[4, 100]);
    }
}
