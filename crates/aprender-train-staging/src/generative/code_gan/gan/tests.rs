//! Tests for Code GAN.

use super::*;
use crate::generative::code_gan::config::{CodeGanConfig, DiscriminatorConfig, GeneratorConfig};
use crate::generative::code_gan::latent::LatentCode;
use proptest::prelude::*;

/// Create a small test config to avoid slow network initialization
fn small_test_config() -> CodeGanConfig {
    CodeGanConfig {
        generator: GeneratorConfig {
            latent_dim: 16,
            hidden_dims: vec![32],
            vocab_size: 50,
            max_seq_len: 8,
            dropout: 0.0,
            batch_norm: false,
        },
        discriminator: DiscriminatorConfig {
            vocab_size: 50,
            max_seq_len: 8,
            embed_dim: 8,
            hidden_dims: vec![16],
            dropout: 0.0,
            spectral_norm: false,
        },
        ..Default::default()
    }
}

#[test]
fn test_interpolation_endpoints() {
    let config = CodeGanConfig {
        generator: GeneratorConfig {
            latent_dim: 16,
            hidden_dims: vec![32],
            vocab_size: 50,
            max_seq_len: 8,
            ..Default::default()
        },
        ..Default::default()
    };
    let gan = CodeGan::with_seed(config, 42);

    let z1 = LatentCode::new(vec![0.0; 16]);
    let z2 = LatentCode::new(vec![1.0; 16]);

    let samples = gan.interpolate(&z1, &z2, 4);
    assert_eq!(samples.len(), 5); // 0, 0.25, 0.5, 0.75, 1.0

    // First should match z1 generation
    let direct_z1 = gan.generator.generate(&z1);
    assert_eq!(samples[0], direct_z1);

    // Last should match z2 generation
    let direct_z2 = gan.generator.generate(&z2);
    assert_eq!(samples[4], direct_z2);
}

#[test]
fn test_code_gan_creation() {
    let config = small_test_config();
    let gan = CodeGan::new(config);
    assert!(gan.num_parameters() > 0);
    assert_eq!(gan.stats.steps, 0);
}

#[test]
fn test_code_gan_sample_latent() {
    let config = small_test_config();
    let mut gan = CodeGan::with_seed(config.clone(), 42);

    let latents = gan.sample_latent(10);
    assert_eq!(latents.len(), 10);
    assert!(latents.iter().all(|z| z.dim() == config.generator.latent_dim));
}

#[test]
fn test_code_gan_generate() {
    let config = CodeGanConfig {
        generator: GeneratorConfig {
            latent_dim: 16,
            hidden_dims: vec![32],
            vocab_size: 50,
            max_seq_len: 8,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut gan = CodeGan::with_seed(config, 42);

    let latents = gan.sample_latent(5);
    let samples = gan.generate(&latents);

    assert_eq!(samples.len(), 5);
    assert!(samples.iter().all(|s| s.len() == 8));
}

#[test]
fn test_code_gan_discriminator_loss() {
    let config = CodeGanConfig {
        generator: GeneratorConfig {
            latent_dim: 16,
            hidden_dims: vec![32],
            vocab_size: 50,
            max_seq_len: 8,
            ..Default::default()
        },
        discriminator: DiscriminatorConfig {
            vocab_size: 50,
            max_seq_len: 8,
            embed_dim: 8,
            hidden_dims: vec![16],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut gan = CodeGan::with_seed(config, 42);

    let real_samples: Vec<Vec<u32>> =
        (0..5).map(|i| (0..8).map(|j| ((i + j) % 50) as u32).collect()).collect();

    let latents = gan.sample_latent(5);
    let fake_samples = gan.generate(&latents);

    let loss = gan.discriminator_loss(&real_samples, &fake_samples);
    assert!(loss >= 0.0);
    assert!(!loss.is_nan());
}

#[test]
fn test_code_gan_generator_loss() {
    let config = CodeGanConfig {
        generator: GeneratorConfig {
            latent_dim: 16,
            hidden_dims: vec![32],
            vocab_size: 50,
            max_seq_len: 8,
            ..Default::default()
        },
        discriminator: DiscriminatorConfig {
            vocab_size: 50,
            max_seq_len: 8,
            embed_dim: 8,
            hidden_dims: vec![16],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut gan = CodeGan::with_seed(config, 42);

    let latents = gan.sample_latent(5);
    let fake_samples = gan.generate(&latents);

    let loss = gan.generator_loss(&fake_samples);
    assert!(loss >= 0.0);
    assert!(!loss.is_nan());
}

#[test]
fn test_record_step() {
    let config = small_test_config();
    let mut gan = CodeGan::new(config);

    let result = TrainingResult {
        gen_loss: 0.5,
        disc_loss: 0.3,
        disc_real_acc: 0.8,
        disc_fake_acc: 0.7,
        gradient_penalty: 0.1,
    };

    gan.record_step(&result);
    assert_eq!(gan.stats.steps, 1);
    assert_eq!(gan.stats.gen_losses.len(), 1);
    assert_eq!(gan.stats.disc_losses.len(), 1);
}

#[test]
fn test_code_gan_stats_default() {
    let stats = CodeGanStats::default();
    assert_eq!(stats.steps, 0);
    assert!(stats.gen_losses.is_empty());
    assert!(stats.disc_losses.is_empty());
    assert_eq!(stats.mode_collapse_score, 0.0);
}

#[test]
fn test_avg_loss_empty() {
    let config = small_test_config();
    let gan = CodeGan::new(config);
    assert_eq!(gan.avg_gen_loss(), 0.0);
    assert_eq!(gan.avg_disc_loss(), 0.0);
}

#[test]
fn test_avg_loss_with_history() {
    let config = small_test_config();
    let mut gan = CodeGan::new(config);

    for i in 0..10 {
        let result = TrainingResult {
            gen_loss: i as f32,
            disc_loss: i as f32 * 2.0,
            disc_real_acc: 0.8,
            disc_fake_acc: 0.7,
            gradient_penalty: 0.1,
        };
        gan.record_step(&result);
    }

    // Average of 0,1,2,...,9 = 4.5
    assert!((gan.avg_gen_loss() - 4.5).abs() < 1e-6);
    assert!((gan.avg_disc_loss() - 9.0).abs() < 1e-6);
}

#[test]
fn test_generate_one() {
    let config = CodeGanConfig {
        generator: GeneratorConfig {
            latent_dim: 16,
            hidden_dims: vec![32],
            vocab_size: 50,
            max_seq_len: 8,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut gan = CodeGan::with_seed(config, 42);

    let tokens = gan.generate_one();
    assert_eq!(tokens.len(), 8);
}

#[test]
fn test_history_size_limit() {
    let config = small_test_config();
    let mut gan = CodeGan::new(config);

    // Add more than 100 steps
    for i in 0..150 {
        let result = TrainingResult {
            gen_loss: i as f32,
            disc_loss: i as f32,
            disc_real_acc: 0.8,
            disc_fake_acc: 0.7,
            gradient_penalty: 0.1,
        };
        gan.record_step(&result);
    }

    // Should be capped at 100
    assert_eq!(gan.stats.gen_losses.len(), 100);
    assert_eq!(gan.stats.disc_losses.len(), 100);
}

proptest! {
    #[test]
    fn test_loss_non_negative(
        real_vals in prop::collection::vec(prop::collection::vec(0u32..50, 8..9), 1..5),
        fake_vals in prop::collection::vec(prop::collection::vec(0u32..50, 8..9), 1..5),
    ) {
        let config = CodeGanConfig {
            generator: GeneratorConfig {
                latent_dim: 16,
                hidden_dims: vec![32],
                vocab_size: 50,
                max_seq_len: 8,
                ..Default::default()
            },
            discriminator: DiscriminatorConfig {
                vocab_size: 50,
                max_seq_len: 8,
                embed_dim: 8,
                hidden_dims: vec![16],
                ..Default::default()
            },
            ..Default::default()
        };
        let gan = CodeGan::with_seed(config, 42);

        let disc_loss = gan.discriminator_loss(&real_vals, &fake_vals);
        let gen_loss = gan.generator_loss(&fake_vals);

        prop_assert!(disc_loss >= 0.0);
        prop_assert!(gen_loss >= 0.0);
    }

    #[test]
    fn test_mode_collapse_detection(num_samples in 10usize..50) {
        let config = small_test_config();
        let mut gan = CodeGan::with_seed(config, 42);

        let score = gan.detect_mode_collapse(num_samples);
        prop_assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_interpolation_length(steps in 1usize..20) {
        let config = small_test_config();
        let gan = CodeGan::with_seed(config, 42);

        let z1 = LatentCode::new(vec![0.0; 16]);
        let z2 = LatentCode::new(vec![1.0; 16]);

        let samples = gan.interpolate(&z1, &z2, steps);
        prop_assert_eq!(samples.len(), steps + 1);
    }
}
