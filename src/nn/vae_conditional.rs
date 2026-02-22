#[allow(clippy::wildcard_imports)]
use super::*;

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
pub(super) fn sample_standard_normal(shape: &[usize]) -> Tensor {
    use rand::Rng;
    let mut rng = rand::rng();

    let size: usize = shape.iter().product();
    let data: Vec<f32> = (0..size)
        .map(|_| {
            // Box-Muller transform for normal distribution
            let u1: f32 = rng.random::<f32>().max(1e-10);
            let u2: f32 = rng.random();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();

    Tensor::new(&data, shape)
}

/// Compute exp(0.5 * x) for standard deviation from log variance.
pub(super) fn exp_half(log_var: &Tensor) -> Tensor {
    let data: Vec<f32> = log_var.data().iter().map(|&x| (0.5 * x).exp()).collect();
    Tensor::new(&data, log_var.shape())
}

/// Compute a + b * c element-wise.
pub(super) fn add_mul(a: &Tensor, b: &Tensor, c: &Tensor) -> Tensor {
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
pub(super) fn mse_loss(pred: &Tensor, target: &Tensor) -> f32 {
    let sum_sq: f32 = pred
        .data()
        .iter()
        .zip(target.data().iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum();
    sum_sq / pred.data().len() as f32
}

/// KL divergence loss for VAE: -0.5 * sum(1 + `log_var` - mu^2 - `exp(log_var)`).
pub(super) fn kl_divergence_loss(mu: &Tensor, log_var: &Tensor) -> f32 {
    let kl: f32 = mu
        .data()
        .iter()
        .zip(log_var.data().iter())
        .map(|(&m, &lv)| -0.5 * (1.0 + lv - m * m - lv.exp()))
        .sum();
    kl / mu.shape()[0] as f32 // Average over batch
}

/// Linear interpolation between two tensors.
pub(super) fn lerp(a: &Tensor, b: &Tensor, alpha: f32) -> Tensor {
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&ai, &bi)| (1.0 - alpha) * ai + alpha * bi)
        .collect();
    Tensor::new(&data, a.shape())
}

/// Concatenate one-hot encoding of class label to tensor.
pub(super) fn concat_one_hot(x: &Tensor, class_label: usize, num_classes: usize) -> Tensor {
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
