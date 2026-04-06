//! Configuration types for Code GAN components.

use serde::{Deserialize, Serialize};

/// Configuration for the Generator network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Dimension of the latent space
    pub latent_dim: usize,
    /// Hidden layer sizes
    pub hidden_dims: Vec<usize>,
    /// Output vocabulary size (number of AST token types)
    pub vocab_size: usize,
    /// Maximum sequence length to generate
    pub max_seq_len: usize,
    /// Dropout rate during training
    pub dropout: f32,
    /// Use batch normalization
    pub batch_norm: bool,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            latent_dim: 128,
            hidden_dims: vec![256, 512, 256],
            vocab_size: 1000,
            max_seq_len: 256,
            dropout: 0.1,
            batch_norm: true,
        }
    }
}

/// Configuration for the Discriminator network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscriminatorConfig {
    /// Input vocabulary size (number of AST token types)
    pub vocab_size: usize,
    /// Maximum sequence length to process
    pub max_seq_len: usize,
    /// Embedding dimension for tokens
    pub embed_dim: usize,
    /// Hidden layer sizes
    pub hidden_dims: Vec<usize>,
    /// Dropout rate during training
    pub dropout: f32,
    /// Use spectral normalization
    pub spectral_norm: bool,
}

impl Default for DiscriminatorConfig {
    fn default() -> Self {
        Self {
            vocab_size: 1000,
            max_seq_len: 256,
            embed_dim: 64,
            hidden_dims: vec![256, 128, 64],
            dropout: 0.2,
            spectral_norm: true,
        }
    }
}

/// Configuration for the complete Code GAN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeGanConfig {
    /// Generator configuration
    pub generator: GeneratorConfig,
    /// Discriminator configuration
    pub discriminator: DiscriminatorConfig,
    /// Learning rate for generator
    pub gen_lr: f32,
    /// Learning rate for discriminator
    pub disc_lr: f32,
    /// Number of discriminator updates per generator update
    pub n_critic: usize,
    /// Gradient penalty coefficient (for WGAN-GP)
    pub gradient_penalty: f32,
    /// Label smoothing for real samples
    pub label_smoothing: f32,
    /// Batch size for training
    pub batch_size: usize,
}

impl Default for CodeGanConfig {
    fn default() -> Self {
        Self {
            generator: GeneratorConfig::default(),
            discriminator: DiscriminatorConfig::default(),
            gen_lr: 0.0002,
            disc_lr: 0.0002,
            n_critic: 5,
            gradient_penalty: 10.0,
            label_smoothing: 0.1,
            batch_size: 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_config_default() {
        let config = GeneratorConfig::default();
        assert_eq!(config.latent_dim, 128);
        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.max_seq_len, 256);
    }

    #[test]
    fn test_discriminator_config_default() {
        let config = DiscriminatorConfig::default();
        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.max_seq_len, 256);
    }

    #[test]
    fn test_code_gan_config_default() {
        let config = CodeGanConfig::default();
        assert_eq!(config.n_critic, 5);
        assert!(config.gen_lr > 0.0);
        assert!(config.disc_lr > 0.0);
    }
}
