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

include!("vae_part_02.rs");
include!("vae_part_03.rs");
