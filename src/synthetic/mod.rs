//! Synthetic Data Generation for `AutoML`.
//!
//! This module provides automatic synthetic data generation capabilities
//! to improve model performance in low-resource domains. Generated data
//! is validated, quality-scored, and integrated into the `AutoML` optimization loop.
//!
//! # Quick Start
//!
//! ```
//! use aprender::synthetic::{SyntheticConfig, GenerationStrategy};
//!
//! // Configure synthetic data generation
//! let config = SyntheticConfig::default()
//!     .with_augmentation_ratio(0.5)
//!     .with_quality_threshold(0.7);
//!
//! assert_eq!(config.augmentation_ratio, 0.5);
//! assert_eq!(config.quality_threshold, 0.7);
//! ```
//!
//! # Design Principles
//!
//! - **Quality-First**: All generated samples validated before inclusion
//! - **Diversity-Aware**: Monitors for mode collapse and distribution shift
//! - **`AutoML` Integration**: Augmentation parameters jointly optimized with model hyperparameters
//!
//! # References
//!
//! - Cubuk et al. (2019). `AutoAugment`: Learning Augmentation Strategies. CVPR.
//! - Wei & Zou (2019). EDA: Easy Data Augmentation. EMNLP.
//! - Ratner et al. (2017). Snorkel: Weak Supervision. VLDB.

pub mod andon;
pub mod cache;
pub mod code_eda;
pub mod code_features;
mod config;
mod diversity;
pub mod eda;
pub mod mixup;
mod params;
mod quality;
pub mod shell;
mod strategy;
pub mod template;
mod validator;
pub mod weak_supervision;

pub use andon::{AndonConfig, AndonEvent, AndonHandler, AndonSeverity, DefaultAndon, TestAndon};
pub use config::SyntheticConfig;
pub use diversity::{DiversityMonitor, DiversityScore};
pub use params::SyntheticParam;
pub use quality::QualityDegradationDetector;
pub use strategy::GenerationStrategy;
pub use validator::{SyntheticValidator, ValidationResult};

use crate::error::Result;

/// Trait for synthetic data generators.
///
/// Implement this trait for domain-specific data generation (e.g., shell commands,
/// code translation pairs). The generator produces synthetic samples from seed data,
/// with quality and diversity scoring for filtering.
///
/// # Type Parameters
///
/// - `Input`: The type of seed samples used for generation
/// - `Output`: The type of generated synthetic samples
///
/// # Example
///
/// ```
/// use aprender::synthetic::{SyntheticGenerator, SyntheticConfig};
/// use aprender::error::Result;
///
/// struct TextGenerator;
///
/// impl SyntheticGenerator for TextGenerator {
///     type Input = String;
///     type Output = String;
///
///     fn generate(&self, seeds: &[Self::Input], config: &SyntheticConfig)
///         -> Result<Vec<Self::Output>>
///     {
///         let target = (seeds.len() as f32 * config.augmentation_ratio) as usize;
///         let synthetic: Vec<_> = seeds.iter()
///             .take(target)
///             .map(|s| format!("{s} [synthetic]"))
///             .collect();
///         Ok(synthetic)
///     }
///
///     fn quality_score(&self, _generated: &Self::Output, _seed: &Self::Input) -> f32 {
///         0.85
///     }
///
///     fn diversity_score(&self, batch: &[Self::Output]) -> f32 {
///         if batch.is_empty() { 0.0 } else { 1.0 / batch.len() as f32 }
///     }
/// }
/// ```
pub trait SyntheticGenerator {
    /// Type of seed samples used for generation.
    type Input;
    /// Type of generated synthetic samples.
    type Output;

    /// Generate synthetic examples from seed data.
    ///
    /// # Arguments
    ///
    /// * `seeds` - Original samples to use as generation seeds
    /// * `config` - Configuration controlling generation behavior
    ///
    /// # Returns
    ///
    /// Vector of generated synthetic samples, filtered by quality threshold.
    fn generate(
        &self,
        seeds: &[Self::Input],
        config: &SyntheticConfig,
    ) -> Result<Vec<Self::Output>>;

    /// Estimate quality of a generated sample relative to its seed.
    ///
    /// Returns a score in [0.0, 1.0] where higher is better quality.
    /// Samples below `config.quality_threshold` are rejected.
    fn quality_score(&self, generated: &Self::Output, seed: &Self::Input) -> f32;

    /// Measure diversity of a batch of generated samples.
    ///
    /// Returns a score in [0.0, 1.0] where higher indicates more diverse samples.
    /// Low diversity suggests mode collapse in generation.
    fn diversity_score(&self, batch: &[Self::Output]) -> f32;
}

/// Callback trait for monitoring synthetic data generation.
///
/// Implement this to receive notifications during generation for logging,
/// metrics collection, or early termination.
pub trait SyntheticCallback: Send + Sync {
    /// Called after each batch of synthetic samples is generated.
    fn on_batch_generated(&mut self, count: usize, config: &SyntheticConfig);

    /// Called when quality falls below threshold.
    fn on_quality_below_threshold(&mut self, actual: f32, threshold: f32);

    /// Called when diversity metrics indicate potential collapse.
    fn on_diversity_collapse(&mut self, score: &DiversityScore);
}

/// Check Andon conditions and trigger events if thresholds exceeded.
///
/// Returns an error with Andon halt if critical conditions detected.
///
/// # Arguments
///
/// * `accepted` - Number of accepted samples
/// * `total` - Total samples generated
/// * `diversity` - Current diversity score
/// * `config` - Synthetic configuration with Andon settings
/// * `andon` - Optional Andon handler for event notification
///
/// # Returns
///
/// Ok(()) if generation should continue, Err if Andon halt triggered.
pub fn check_andon<A: AndonHandler>(
    accepted: usize,
    total: usize,
    diversity: f32,
    config: &SyntheticConfig,
    andon: Option<&A>,
) -> Result<()> {
    if !config.andon.enabled || total == 0 {
        return Ok(());
    }

    let rejection_rate = 1.0 - (accepted as f32 / total as f32);
    check_rejection_rate(rejection_rate, config, andon)?;
    check_diversity(diversity, config, andon);

    Ok(())
}

/// Check rejection rate and trigger Andon event if threshold exceeded.
fn check_rejection_rate<A: AndonHandler>(
    rejection_rate: f32,
    config: &SyntheticConfig,
    andon: Option<&A>,
) -> Result<()> {
    if !config.andon.exceeds_rejection_threshold(rejection_rate) {
        return Ok(());
    }

    let event = AndonEvent::HighRejectionRate {
        rate: rejection_rate,
        threshold: config.andon.rejection_threshold,
    };

    if let Some(handler) = andon {
        handler.on_event(&event);
        if handler.should_halt(&event) {
            return Err(crate::error::AprenderError::Other(format!(
                "ANDON HALT: Rejection rate {:.1}% exceeds threshold {:.1}%",
                rejection_rate * 100.0,
                config.andon.rejection_threshold * 100.0
            )));
        }
    }
    Ok(())
}

/// Check diversity and trigger Andon event if collapsed.
fn check_diversity<A: AndonHandler>(diversity: f32, config: &SyntheticConfig, andon: Option<&A>) {
    if !config.andon.has_diversity_collapse(diversity) {
        return;
    }

    let event = AndonEvent::DiversityCollapse {
        score: diversity,
        minimum: config.andon.diversity_minimum,
    };

    if let Some(handler) = andon {
        handler.on_event(&event);
        // Diversity collapse is warning, not halt
    }
}

/// Generate synthetic data in batches to manage memory.
///
/// # Arguments
///
/// * `generator` - The synthetic data generator to use
/// * `seeds` - Original samples to use as generation seeds
/// * `config` - Configuration controlling generation behavior
/// * `batch_size` - Number of seeds to process per batch
///
/// # Example
///
/// ```
/// use aprender::synthetic::{generate_batched, SyntheticGenerator, SyntheticConfig};
/// use aprender::error::Result;
///
/// struct SimpleGenerator;
///
/// impl SyntheticGenerator for SimpleGenerator {
///     type Input = i32;
///     type Output = i32;
///
///     fn generate(&self, seeds: &[i32], config: &SyntheticConfig) -> Result<Vec<i32>> {
///         Ok(seeds.iter().map(|x| x * 2).collect())
///     }
///
///     fn quality_score(&self, _: &i32, _: &i32) -> f32 { 1.0 }
///     fn diversity_score(&self, _: &[i32]) -> f32 { 1.0 }
/// }
///
/// let gen = SimpleGenerator;
/// let seeds = vec![1, 2, 3, 4, 5];
/// let config = SyntheticConfig::default();
/// let result = generate_batched(&gen, &seeds, &config, 2).expect("generation should succeed");
/// assert_eq!(result, vec![2, 4, 6, 8, 10]);
/// ```
pub fn generate_batched<G>(
    generator: &G,
    seeds: &[G::Input],
    config: &SyntheticConfig,
    batch_size: usize,
) -> Result<Vec<G::Output>>
where
    G: SyntheticGenerator,
{
    let mut all_synthetic = Vec::new();

    for chunk in seeds.chunks(batch_size.max(1)) {
        let batch = generator.generate(chunk, config)?;
        all_synthetic.extend(batch);
    }

    Ok(all_synthetic)
}

/// Streaming iterator for memory-constrained synthetic generation.
///
/// Generates synthetic data on-demand rather than all at once,
/// reducing peak memory usage for large datasets.
#[derive(Debug)]
pub struct SyntheticStream<'a, G: SyntheticGenerator + std::fmt::Debug> {
    generator: &'a G,
    seeds: &'a [G::Input],
    config: &'a SyntheticConfig,
    current_idx: usize,
    batch_size: usize,
}

impl<'a, G: SyntheticGenerator + std::fmt::Debug> SyntheticStream<'a, G> {
    /// Create a new streaming generator.
    ///
    /// # Arguments
    ///
    /// * `generator` - The synthetic data generator to use
    /// * `seeds` - Original samples to use as generation seeds
    /// * `config` - Configuration controlling generation behavior
    /// * `batch_size` - Number of seeds to process per iteration
    #[must_use]
    pub fn new(
        generator: &'a G,
        seeds: &'a [G::Input],
        config: &'a SyntheticConfig,
        batch_size: usize,
    ) -> Self {
        Self {
            generator,
            seeds,
            config,
            current_idx: 0,
            batch_size: batch_size.max(1),
        }
    }

    /// Check if there are more batches to generate.
    #[must_use]
    pub fn has_next(&self) -> bool {
        self.current_idx < self.seeds.len()
    }

    /// Get the number of seeds remaining to process.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.seeds.len().saturating_sub(self.current_idx)
    }
}

impl<G: SyntheticGenerator + std::fmt::Debug> Iterator for SyntheticStream<'_, G> {
    type Item = Result<Vec<G::Output>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.seeds.len() {
            return None;
        }
        let end = (self.current_idx + self.batch_size).min(self.seeds.len());
        let chunk = &self.seeds[self.current_idx..end];
        self.current_idx = end;
        Some(self.generator.generate(chunk, self.config))
    }
}

#[cfg(test)]
#[path = "synthetic_tests.rs"]
mod tests;
