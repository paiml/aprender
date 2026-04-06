//! Code GAN main struct and training logic.

use crate::generative::code_gan::config::CodeGanConfig;
use crate::generative::code_gan::discriminator::Discriminator;
use crate::generative::code_gan::generator::Generator;
use crate::generative::code_gan::latent::LatentCode;

use super::stats::CodeGanStats;
use super::training_result::TrainingResult;

/// Complete Code GAN for generating Rust AST
pub struct CodeGan {
    /// Configuration
    pub config: CodeGanConfig,
    /// Generator network
    pub generator: Generator,
    /// Discriminator network
    pub discriminator: Discriminator,
    /// Training statistics
    pub stats: CodeGanStats,
    /// Random number generator
    rng: rand::rngs::StdRng,
}

impl CodeGan {
    /// Create a new Code GAN
    pub fn new(config: CodeGanConfig) -> Self {
        use rand::SeedableRng;
        let generator = Generator::new(config.generator.clone());
        let discriminator = Discriminator::new(config.discriminator.clone());
        Self {
            config,
            generator,
            discriminator,
            stats: CodeGanStats::default(),
            rng: rand::rngs::StdRng::from_os_rng(),
        }
    }

    /// Create a new Code GAN with a seed for reproducibility
    pub fn with_seed(config: CodeGanConfig, seed: u64) -> Self {
        use rand::SeedableRng;
        let generator = Generator::with_seed(config.generator.clone(), seed);
        let discriminator = Discriminator::with_seed(config.discriminator.clone(), seed + 1);
        Self {
            config,
            generator,
            discriminator,
            stats: CodeGanStats::default(),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Sample latent codes for generation
    pub fn sample_latent(&mut self, batch_size: usize) -> Vec<LatentCode> {
        (0..batch_size)
            .map(|_| LatentCode::sample(&mut self.rng, self.config.generator.latent_dim))
            .collect()
    }

    /// Generate code from latent codes
    pub fn generate(&self, latent_codes: &[LatentCode]) -> Vec<Vec<u32>> {
        latent_codes.iter().map(|z| self.generator.generate(z)).collect()
    }

    /// Generate a single code sample
    pub fn generate_one(&mut self) -> Vec<u32> {
        let z = LatentCode::sample(&mut self.rng, self.config.generator.latent_dim);
        self.generator.generate(&z)
    }

    /// Discriminate a batch of code samples
    pub fn discriminate(&self, samples: &[Vec<u32>]) -> Vec<f32> {
        samples.iter().map(|tokens| self.discriminator.discriminate(tokens)).collect()
    }

    /// Compute discriminator loss (binary cross-entropy)
    pub fn discriminator_loss(&self, real_samples: &[Vec<u32>], fake_samples: &[Vec<u32>]) -> f32 {
        let real_probs = self.discriminate(real_samples);
        let fake_probs = self.discriminate(fake_samples);

        // BCE loss: -[y*log(p) + (1-y)*log(1-p)]
        // For real: y=1, for fake: y=0
        let smoothed_real = 1.0 - self.config.label_smoothing;

        let real_loss: f32 =
            real_probs.iter().map(|&p| -smoothed_real * p.max(1e-7).ln()).sum::<f32>()
                / real_probs.len().max(1) as f32;

        let fake_loss: f32 = fake_probs.iter().map(|&p| -(1.0 - p).max(1e-7).ln()).sum::<f32>()
            / fake_probs.len().max(1) as f32;

        real_loss + fake_loss
    }

    /// Compute generator loss (try to fool discriminator)
    pub fn generator_loss(&self, fake_samples: &[Vec<u32>]) -> f32 {
        let fake_probs = self.discriminate(fake_samples);

        // Generator wants discriminator to output 1 (real) for fakes
        let loss: f32 = fake_probs.iter().map(|&p| -p.max(1e-7).ln()).sum::<f32>()
            / fake_probs.len().max(1) as f32;

        loss
    }

    /// Detect mode collapse by measuring diversity of generated samples
    pub fn detect_mode_collapse(&mut self, num_samples: usize) -> f32 {
        use std::collections::HashSet;

        let latent_codes = self.sample_latent(num_samples);
        let samples = self.generate(&latent_codes);

        // Count unique token sequences
        let unique_seqs: HashSet<Vec<u32>> = samples.into_iter().collect();
        let diversity = unique_seqs.len() as f32 / num_samples as f32;

        // Also check token diversity
        let all_tokens: HashSet<u32> =
            unique_seqs.iter().flat_map(|seq| seq.iter().copied()).collect();

        self.stats.unique_tokens = all_tokens.len();

        // Mode collapse score: 1 - diversity
        let mode_collapse_score = 1.0 - diversity;
        self.stats.mode_collapse_score = mode_collapse_score;

        mode_collapse_score
    }

    /// Interpolate between two latent codes and generate intermediate samples
    pub fn interpolate(&self, z1: &LatentCode, z2: &LatentCode, steps: usize) -> Vec<Vec<u32>> {
        (0..=steps)
            .map(|i| {
                let t = i as f32 / steps as f32;
                let z = z1.slerp(z2, t);
                self.generator.generate(&z)
            })
            .collect()
    }

    /// Get total number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        self.generator.num_parameters() + self.discriminator.num_parameters()
    }

    /// Record training step
    pub fn record_step(&mut self, result: &TrainingResult) {
        self.stats.steps += 1;

        if self.stats.gen_losses.len() >= 100 {
            self.stats.gen_losses.pop_front();
        }
        self.stats.gen_losses.push_back(result.gen_loss);

        if self.stats.disc_losses.len() >= 100 {
            self.stats.disc_losses.pop_front();
        }
        self.stats.disc_losses.push_back(result.disc_loss);
    }

    /// Get average generator loss over recent history
    #[must_use]
    pub fn avg_gen_loss(&self) -> f32 {
        if self.stats.gen_losses.is_empty() {
            return 0.0;
        }
        self.stats.gen_losses.iter().sum::<f32>() / self.stats.gen_losses.len() as f32
    }

    /// Get average discriminator loss over recent history
    #[must_use]
    pub fn avg_disc_loss(&self) -> f32 {
        if self.stats.disc_losses.is_empty() {
            return 0.0;
        }
        self.stats.disc_losses.iter().sum::<f32>() / self.stats.disc_losses.len() as f32
    }
}
