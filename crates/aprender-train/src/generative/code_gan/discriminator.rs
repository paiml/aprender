//! Discriminator network for Code GAN.

use rand::Rng;

use super::config::DiscriminatorConfig;

/// Type alias for discriminator weights structure
type DiscriminatorWeights = (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>);

/// Discriminator network: classifies code as real or fake
#[derive(Debug)]
pub struct Discriminator {
    /// Configuration
    pub config: DiscriminatorConfig,
    /// Token embeddings
    embeddings: Vec<Vec<f32>>,
    /// Weights for each layer
    weights: Vec<Vec<Vec<f32>>>,
    /// Biases for each layer
    biases: Vec<Vec<f32>>,
}

impl Discriminator {
    /// Create a new discriminator with random initialization
    pub fn new(config: DiscriminatorConfig) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::from_os_rng();
        let (embeddings, weights, biases) = Self::init_weights(&config, &mut rng);
        Self { config, embeddings, weights, biases }
    }

    /// Create a new discriminator with a seed for reproducibility
    pub fn with_seed(config: DiscriminatorConfig, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let (embeddings, weights, biases) = Self::init_weights(&config, &mut rng);
        Self { config, embeddings, weights, biases }
    }

    fn init_weights<R: Rng>(config: &DiscriminatorConfig, rng: &mut R) -> DiscriminatorWeights {
        // Helper function for Box-Muller normal sampling
        let sample_normal = |rng: &mut R, std: f64| -> f32 {
            let u1: f64 = rng.random::<f64>().max(1e-10);
            let u2: f64 = rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            (z * std) as f32
        };

        // Initialize embeddings
        let embed_std = (1.0 / config.embed_dim as f64).sqrt();
        let embeddings: Vec<Vec<f32>> = (0..config.vocab_size)
            .map(|_| (0..config.embed_dim).map(|_| sample_normal(rng, embed_std)).collect())
            .collect();

        // Initialize dense layers
        let input_dim = config.embed_dim * config.max_seq_len;
        let mut dims = vec![input_dim];
        dims.extend(&config.hidden_dims);
        dims.push(1); // Output: single logit for real/fake

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..dims.len() - 1 {
            let in_dim = dims[i];
            let out_dim = dims[i + 1];

            let std = (2.0 / (in_dim + out_dim) as f64).sqrt();

            let w: Vec<Vec<f32>> = (0..out_dim)
                .map(|_| (0..in_dim).map(|_| sample_normal(rng, std)).collect())
                .collect();
            let b: Vec<f32> = vec![0.0; out_dim];

            weights.push(w);
            biases.push(b);
        }

        (embeddings, weights, biases)
    }

    /// Discriminate: returns probability that input is real (valid code)
    pub fn discriminate(&self, tokens: &[u32]) -> f32 {
        // Pad or truncate to max_seq_len
        let mut padded = tokens.to_vec();
        padded.resize(self.config.max_seq_len, 0);

        // Embed tokens
        let mut x = Vec::with_capacity(self.config.max_seq_len * self.config.embed_dim);
        for &token in &padded {
            let token_idx = (token as usize).min(self.config.vocab_size - 1);
            x.extend(&self.embeddings[token_idx]);
        }

        // Forward pass through dense layers
        for (i, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            x = Self::linear_forward(&x, w, b);
            // Leaky ReLU for all but last layer
            if i < self.weights.len() - 1 {
                x = x.iter().map(|&v| if v > 0.0 { v } else { 0.01 * v }).collect();
            }
        }

        // Sigmoid on output
        sigmoid(x[0])
    }

    fn linear_forward(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
        let output_dim = weights.len();
        let mut output = Vec::with_capacity(output_dim);

        for (i, w_row) in weights.iter().enumerate() {
            let dot: f32 = w_row.iter().zip(input).map(|(a, b)| a * b).sum();
            output.push(dot + bias[i]);
        }

        output
    }

    /// Get number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let embed_params = self.embeddings.len() * self.config.embed_dim;
        let weight_params: usize = self.weights.iter().map(|w| w.len() * w[0].len()).sum();
        let bias_params: usize = self.biases.iter().map(Vec::len).sum();
        embed_params + weight_params + bias_params
    }
}

/// Sigmoid activation function
pub fn sigmoid(x: f32) -> f32 {
    contract_pre_sigmoid!();
    let result = 1.0 / (1.0 + (-x).exp());
    contract_post_silu!(&[result]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_discriminator_creation() {
        let config = DiscriminatorConfig {
            vocab_size: 100,
            max_seq_len: 10,
            embed_dim: 16,
            hidden_dims: vec![32, 16],
            dropout: 0.1,
            spectral_norm: true,
        };
        let disc = Discriminator::with_seed(config, 42);
        assert!(disc.num_parameters() > 0);
    }

    #[test]
    fn test_discriminator_output_range() {
        let config = DiscriminatorConfig {
            vocab_size: 50,
            max_seq_len: 8,
            embed_dim: 8,
            hidden_dims: vec![16],
            dropout: 0.0,
            spectral_norm: false,
        };
        let disc = Discriminator::with_seed(config, 42);

        let tokens = vec![1, 2, 3, 4, 5];
        let prob = disc.discriminate(&tokens);

        // Output should be in [0, 1] due to sigmoid
        assert!((0.0..=1.0).contains(&prob));
    }

    #[test]
    fn test_discriminator_deterministic() {
        let config = DiscriminatorConfig {
            vocab_size: 50,
            max_seq_len: 8,
            embed_dim: 8,
            hidden_dims: vec![16],
            dropout: 0.0,
            spectral_norm: false,
        };
        let disc = Discriminator::with_seed(config, 42);

        let tokens = vec![1, 2, 3, 4, 5];
        let prob1 = disc.discriminate(&tokens);
        let prob2 = disc.discriminate(&tokens);

        assert!((prob1 - prob2).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_function() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    proptest! {
        #[test]
        fn test_discriminator_output_bounds(tokens in prop::collection::vec(0u32..50, 1..10)) {
            let config = DiscriminatorConfig {
                vocab_size: 50,
                max_seq_len: 10,
                embed_dim: 8,
                hidden_dims: vec![16],
                dropout: 0.0,
                spectral_norm: false,
            };
            let disc = Discriminator::with_seed(config, 42);

            let prob = disc.discriminate(&tokens);
            prop_assert!((0.0..=1.0).contains(&prob));
        }
    }
}
