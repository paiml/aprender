//! Generator network for Code GAN.

use rand::Rng;

use super::config::GeneratorConfig;
use super::latent::LatentCode;

/// Generator network: maps latent vectors to AST token sequences
#[derive(Debug)]
pub struct Generator {
    /// Configuration
    pub config: GeneratorConfig,
    /// Weights for each layer (simplified representation)
    weights: Vec<Vec<Vec<f32>>>,
    /// Biases for each layer
    biases: Vec<Vec<f32>>,
}

impl Generator {
    /// Create a new generator with random initialization
    pub fn new(config: GeneratorConfig) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::from_os_rng();
        let (weights, biases) = Self::init_weights(&config, &mut rng);
        Self { config, weights, biases }
    }

    /// Create a new generator with a seed for reproducibility
    pub fn with_seed(config: GeneratorConfig, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let (weights, biases) = Self::init_weights(&config, &mut rng);
        Self { config, weights, biases }
    }

    fn init_weights<R: Rng>(
        config: &GeneratorConfig,
        rng: &mut R,
    ) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
        let mut dims = vec![config.latent_dim];
        dims.extend(&config.hidden_dims);
        dims.push(config.vocab_size * config.max_seq_len);

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..dims.len() - 1 {
            let input_dim = dims[i];
            let output_dim = dims[i + 1];

            // Xavier initialization using Box-Muller transform
            let std = (2.0 / (input_dim + output_dim) as f64).sqrt();

            let w: Vec<Vec<f32>> = (0..output_dim)
                .map(|_| {
                    (0..input_dim)
                        .map(|_| {
                            let u1: f64 = rng.random::<f64>().max(1e-10);
                            let u2: f64 = rng.random::<f64>();
                            let z =
                                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                            (z * std) as f32
                        })
                        .collect()
                })
                .collect();
            let b: Vec<f32> = vec![0.0; output_dim];

            weights.push(w);
            biases.push(b);
        }

        (weights, biases)
    }

    /// Generate AST tokens from a latent code
    pub fn generate(&self, latent: &LatentCode) -> Vec<u32> {
        assert_eq!(latent.dim(), self.config.latent_dim);

        // Forward pass through network
        let mut x = latent.vector.clone();

        for (w, b) in self.weights.iter().zip(&self.biases) {
            x = Self::linear_forward(&x, w, b);
            // ReLU activation (except last layer)
            if w != self.weights.last().expect("non-empty weights") {
                x = x.iter().map(|&v| v.max(0.0)).collect();
            }
        }

        // Reshape to (max_seq_len, vocab_size) and take argmax for each position
        let vocab_size = self.config.vocab_size;
        let max_seq_len = self.config.max_seq_len;

        let mut tokens = Vec::with_capacity(max_seq_len);
        for pos in 0..max_seq_len {
            let start = pos * vocab_size;
            let end = start + vocab_size;
            if end <= x.len() {
                let logits = &x[start..end];
                let max_idx = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(i, _)| i as u32);
                tokens.push(max_idx);
            }
        }

        tokens
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
        let weight_params: usize = self.weights.iter().map(|w| w.len() * w[0].len()).sum();
        let bias_params: usize = self.biases.iter().map(Vec::len).sum();
        weight_params + bias_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_generator_creation() {
        let config = GeneratorConfig {
            latent_dim: 32,
            hidden_dims: vec![64, 64],
            vocab_size: 100,
            max_seq_len: 10,
            dropout: 0.1,
            batch_norm: true,
        };
        let gen = Generator::with_seed(config, 42);
        assert!(gen.num_parameters() > 0);
    }

    #[test]
    fn test_generator_generate() {
        let config = GeneratorConfig {
            latent_dim: 16,
            hidden_dims: vec![32],
            vocab_size: 50,
            max_seq_len: 8,
            dropout: 0.0,
            batch_norm: false,
        };
        let gen = Generator::with_seed(config.clone(), 42);

        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let z = LatentCode::sample(&mut rng, config.latent_dim);

        let tokens = gen.generate(&z);
        assert_eq!(tokens.len(), config.max_seq_len);
        assert!(tokens.iter().all(|&t| t < config.vocab_size as u32));
    }

    #[test]
    fn test_generator_deterministic() {
        let config = GeneratorConfig {
            latent_dim: 16,
            hidden_dims: vec![32],
            vocab_size: 50,
            max_seq_len: 8,
            dropout: 0.0,
            batch_norm: false,
        };

        let gen = Generator::with_seed(config.clone(), 42);
        let z = LatentCode::new(vec![0.5; config.latent_dim]);

        let tokens1 = gen.generate(&z);
        let tokens2 = gen.generate(&z);

        assert_eq!(tokens1, tokens2);
    }

    proptest! {
        #[test]
        fn test_generator_output_valid_tokens(seed in 0u64..10000) {
            let config = GeneratorConfig {
                latent_dim: 16,
                hidden_dims: vec![32],
                vocab_size: 50,
                max_seq_len: 8,
                dropout: 0.0,
                batch_norm: false,
            };
            let gen = Generator::with_seed(config.clone(), seed);

            use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let z = LatentCode::sample(&mut rng, config.latent_dim);

            let tokens = gen.generate(&z);
            prop_assert!(tokens.iter().all(|&t| t < 50));
        }
    }
}
